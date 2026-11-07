"""电流累积单元的独立逐周期模型，接口边界与同名硬件模块一致。

分片单元和累加器通过显式输入、请求和响应驱动本模型。每拍先读取旧状态，
再提交状态更新，避免软件调用顺序让新写入、定稿和标签释放提前一拍可见。
本模块不包含外围任务派发、连续空闲确认和遍历result array以完成LIF计算。
"""

import os
from dataclasses import dataclass, field

# 临时插桩：验证 PE 推进到新 ifmap 行时，是否还有更早行的 express 包未写回。
_ROW_BARRIER = {'advance': 0, 'advance_with_queue': 0, 'max_queue': 0}
# 临时插桩：排空分解，issue 完成到末次提交、末次提交到条目全部释放。
_DRAIN_PARTS = {'issue_to_last_submit': [], 'last_submit_to_empty': [],
                'empty_to_idle': [], 'total': []}


def signed_wrap(value, width):
    """按硬件有符号位宽截断，保留溢出的模加语义。"""
    return ((int(value) + (1 << (width - 1))) % (1 << width)) - (1 << (width - 1))


@dataclass(frozen=True)
class Packet:
    """分片单元输出包；向量元素按硬件低位到高位排列。"""

    row: int
    column: int
    mode: int
    vector: tuple

    @property
    def packed_vector(self):
        return sum(int(value) << (2 * index) for index, value in enumerate(self.vector))

    @property
    def run_length(self):
        return 1 if self.mode == 1 else sum((self.vector[index] & 1) << (2 - index) for index in range(3))


@dataclass
class PartialSumEntry:
    """按输出行和半行关联分配的部分和存储单元。"""

    row: int
    half: int
    current_tile: bool = True
    final: bool = False
    modified: set = field(default_factory=set)
    retired: set = field(default_factory=set)
    data: dict = field(default_factory=dict)


class CurrentAccumulatingUnit:
    """建模双通道发射、计算、部分和池和行屏障。"""

    def __init__(self, weights, fully_connected=False, pool_banks=8, conv_shape=(12, 12)):
        if len(weights) != 16 or any(len(group) != 3 for group in weights):
            raise ValueError('电流累积单元需要十六组三行三列权重')
        self.weights = weights
        self.fully_connected = fully_connected
        self.conv_shape = tuple(conv_shape)
        assert len(self.conv_shape) == 2 and all(0 < size <= 12 for size in self.conv_shape)
        if pool_banks not in (8, 9):
            raise ValueError('psum pool 支持八个或九个 bank')
        self.entries = [None] * pool_banks
        self.express_queue = []
        self.front = 0
        self.c1 = 0
        self.c2 = 0
        self._pe_cursor_row = -1
        self._flush_tick = None
        self._last_submit_tick = None
        self._idle_tick = None
        self._empty_tick = None
        self._last_row = -1
        self.draining = False
        self.cycles = 0

    @property
    def idle(self):
        # 完成判据只看“还有没有未写入 psum 的数据”：部分和存储单元全部释放且 express 队列排空。
        # bundle 一经 accumulator 的深度 2 表项接收即视为提交完成，本单元不再保留在途状态。
        # 因此池排空当拍就可以交接下一个 tile，不必等 accumulator 的写流水排空；
        # 整层收尾的写流水尾拍由 `all_drained`（调度完成且写流水空）负责，不在本模块内计费。
        return (not any(entry is not None for entry in self.entries)
                and not self.express_queue)

    def _key(self, row, column):
        if self.fully_connected:
            row, column = column, row
        return row, int(column >= 6)

    def _valid_output(self, row, column):
        """边界 tile 只为有效输出点分配和写入 psum pool。"""
        if self.fully_connected:
            return 0 <= row < 12
        return 0 <= row < self.conv_shape[0] and 0 <= column < self.conv_shape[1]

    def _find(self, key):
        return next((index for index, entry in enumerate(self.entries)
                     if entry is not None and entry.current_tile
                     and (entry.row, entry.half) == key), None)

    def close_tile(self):
        """计算流水排空后封闭当前 tile，不等待其部分和退休。"""
        assert not self.express_queue, 'Express 尚有未完成的计算'
        for entry in self.entries:
            if entry is not None and entry.current_tile:
                entry.current_tile = False
                entry.final = True

    def begin_tile(self, weights=None):
        """保留旧 bank，仅初始化下一 tile 的计算控制和权重。"""
        assert not self.express_queue
        assert not any(entry is not None and entry.current_tile for entry in self.entries)
        if weights is not None:
            self.weights = weights
        self.front = self.c1 = self.c2 = 0
        self._pe_cursor_row = self._last_row = -1
        self.draining = False
        self._flush_tick = self._last_submit_tick = self._idle_tick = self._empty_tick = None

    def _delta(self, packet, convolution_row, column):
        """直接计算当前点的电流增量，不读取硬件仿真结果。"""
        if packet.mode == 0:
            return [signed_wrap(sum(self.weights[unit][convolution_row][tap] * packet.vector[tap]
                                    for tap in range(3)), 12) for unit in range(16)]
        low = max(0, min(2, packet.column - column))
        high = max(0, min(2, packet.column + packet.run_length - 1 - column))
        value = packet.vector[0] if packet.mode == 1 else 1
        return [signed_wrap(sum(self.weights[unit][convolution_row][tap]
                                for tap in range(low, high + 1)) * value, 12) for unit in range(16)]

    def step(self, normal, express, mode0_front, express_front, front_column=None,
             request_ready=(True, True), flush=False, mode0_absent=False,
             express_absent=False):
        """推进一个上升沿，返回该沿实际发生的输入握手和输出提交。

        两路各自给出仍可能产生 package 的最小 ifmap 行，行屏障取两路较小值，
        因此 PE 与 express 可以停在不同 ifmap 行，不再需要 hold 互锁。
        """
        self.cycles += 1
        active = not self.draining
        if not flush and self._flush_tick is not None:
            self._flush_tick = self._last_submit_tick = self._idle_tick = None
            self._empty_tick = None
        head = self.express_queue[0] if self.express_queue else None
        free = [index for index, entry in enumerate(self.entries) if entry is None]
        normal_row = (int(normal.row) if normal is not None
                      else (int(mode0_front) if mode0_front is not None else 0))
        normal_column = int(normal.column) if normal is not None else 0
        normal_keys = [self._key(normal_row - lane, normal_column)
                       for lane in range(3) if self._valid_output(normal_row-lane, normal_column)]
        missing_normal = set(key for key in normal_keys if self._find(key) is None)
        # 两路各自的最小未完成行：只要某一行仍可能产生该路 package，就必须计入。
        # express 还要计入后端队列队头：队列里的包仍会写它自己的 r-2 行，即使它的 FIFO 已经排空。
        if self.fully_connected:
            rmin_mode0 = rmin_express = 12
        else:
            rmin_mode0 = int(mode0_front) if mode0_front is not None else int(self.front)
            rmin_express = int(express_front) if express_front is not None else int(self.front)
            if normal is not None:
                rmin_mode0 = min(rmin_mode0, int(normal.row))
            if express is not None:
                rmin_express = min(rmin_express, int(express.row))
            if head is not None:
                rmin_express = min(rmin_express, int(head['packet'].row))
        # psum pool 的每个 (row, half) 身份占一个表项。两路最小未完成行相差 s 行时，
        # 同时在世的输出行跨度为 s+3 行、最多 2×（s+3）个身份，因此表项数直接决定
        # 解耦深度：8 个表项只允许 s≤1；再大就会出现无法分配表项、且落后一路无法推进的死锁。
        skew_limit = int(os.environ.get('EVA_MAX_ROW_SKEW',
                                        str(max(0, len(self.entries) // 2 - 3))))
        normal_ready = active and len(missing_normal) <= len(free)
        express_ready = (active and not flush and not self.fully_connected
                         and len(self.express_queue) < 2)
        row_dependency_blocked = False
        if not self.fully_connected and mode0_front is not None and express_front is not None:
            row_dependency_blocked = bool(
                (normal is not None and active and normal_row > rmin_express + skew_limit)
                or (express is not None and express_ready
                    and int(express.row) > rmin_mode0 + skew_limit))
            normal_ready = normal_ready and normal_row <= rmin_express + skew_limit
            express_ready = (express_ready and express is not None
                             and int(express.row) <= rmin_mode0 + skew_limit)
        normal_fire = normal is not None and normal_ready
        # psum 分配不足：本拍确有常规包要分配，但空闲表项不够覆盖全部新身份。
        normal_alloc_stall = bool(normal is not None and active
                                  and len(missing_normal) > len(free))
        # 本拍要写进池的增量在退休判定之后才落地，所以按列上界必须用**本拍写入之前**的推进位置；
        # 否则刚发出那一列会被用写前值退休，随后的 lane2 增量永久丢失。
        # c1 只描述 PE 当前 ifmap 行的 mode0 列推进位置；只有 PE 换成新行时才归零。
        # express 换行不再影响 PE 的列游标。
        if normal is not None and int(normal.row) != self._pe_cursor_row:
            self._pe_cursor_row = int(normal.row)
            self.c1 = 0
        c1_before_write = self.c1
        if normal_fire:
            self.c1 = max(self.c1, int(normal.column) + 1)
            if int(normal.row) > self._last_row:
                _ROW_BARRIER['advance'] += 1
                _ROW_BARRIER['advance_with_queue'] += int(bool(self.express_queue))
                _ROW_BARRIER['max_queue'] = max(_ROW_BARRIER['max_queue'], len(self.express_queue))
                self._last_row = int(normal.row)
        express_fire = express is not None and express_ready

        # 常规计算先占用分配名额；同拍 express 可以复用相同的行、半行身份。
        allocation = {(entry.row, entry.half): index for index, entry in enumerate(self.entries)
                      if entry is not None and entry.current_tile}
        writes = []
        if normal_fire:
            for lane in range(3):
                row = normal.row - lane
                if not self._valid_output(row, normal.column):
                    continue
                key = self._key(row, normal.column)
                if key not in allocation:
                    allocation[key] = free.pop(0)
                column = row if self.fully_connected else normal.column
                writes.append((allocation[key], key, column, self._delta(normal, lane, normal.column)))

        # 队头每拍最多处理四列；常规计算占用一列带宽，且写入不能跨半行。
        express_columns = []
        attempted = {}
        express_writes = []
        express_can_write = True
        express_alloc_stall = False
        if head:
            packet = head['packet']
            for column in range(head['cursor'], min(head['end'], head['cursor'] + 4 - int(normal_fire))):
                if column // 6 != head['cursor'] // 6:
                    break
                if normal_fire and column == normal.column and abs(packet.row - normal.row) <= 2:
                    break
                express_columns.append(column)
            wanted = [self._key(packet.row - lane, express_columns[0]) for lane in range(3)
                      if express_columns and self._valid_output(packet.row-lane, express_columns[0])]
            new_keys = list(dict.fromkeys(key for key in wanted if key not in allocation))
            express_can_write = len(new_keys) <= len(free)
            express_alloc_stall = bool(express_columns and not express_can_write)
            trial = dict(allocation)
            for key, index in zip(new_keys, free):
                trial[key] = index
            for lane in range(3):
                row = packet.row - lane
                if not self._valid_output(row, head['cursor']):
                    continue
                for column in express_columns:
                    if not self._valid_output(row, column):
                        continue
                    key = self._key(row, column)
                    if key in trial:
                        index = trial[key]
                        attempted.setdefault(index, set()).add(column)
                        if express_can_write:
                            express_writes.append((index, key, column, self._delta(packet, lane, column)))
            writes.extend(express_writes)

        # 定稿线取**行屏障前沿**（self.front），不是本拍呈现的包行号。
        # current_row 只用于诊断，不参与定稿判定。
        current_row = int(self.front)
        if self.fully_connected:
            margin = 1
            barrier_valid = (flush and not head) or ((active or flush) and self.front >= margin)
            barrier_last = 12 if (flush and not head) else max(0, self.front - margin)
            row_full = row_part = barrier_last if barrier_valid else -1
            col_limit = 12
        else:
            if flush and not self.express_queue:
                row_full = row_part = 11
                col_limit = 12
            else:
                row_full = (current_row - 3) if current_row >= 3 else -1
                # 对照实验：lookahead 只有窗口外整行可退，没有 r-2 的按列退这一档。
                row_part = (row_full if os.environ.get('EVA_RETIRE_LOOKAHEAD')
                            else ((current_row - 2) if current_row >= 2 else -1))
                # 只有仍可能写 current_row-2 的那一路才施加列上界：
                # PE 仍在这一行时用 c1（该行没有 mode0 包则放开到整行）；
                # express 仍在这一行时保持 0（该行完全没有 express 包才放开到整行）。
                if rmin_mode0 <= current_row:
                    c1_now = (12 if mode0_absent else
                              c1_before_write if self._pe_cursor_row == current_row else 0)
                else:
                    c1_now = 12
                if rmin_express <= current_row:
                    c2_now = 12 if express_absent else 0
                else:
                    c2_now = 12
                col_limit = min(c1_now, c2_now)
                if flush and head:
                    # 队头包只覆盖 head.row, head.row-1, head.row-2，更早的行可以整行退。
                    row_full = max(row_full, int(head['packet'].row) - 3)
        # 提交只读取旧状态；同拍 express 试图写入的点必须从候选中排除。
        candidates = []
        for index, entry in enumerate(self.entries):
            if entry is None or (entry.current_tile and entry.row > row_part and not entry.final):
                continue
            free_cols = entry.modified - entry.retired - attempted.get(index, set())
            if entry.current_tile and entry.row > row_full:
                free_cols = {column for column in free_cols if column < col_limit}
            columns = sorted(free_cols)
            if columns:
                priority = 0 if not entry.current_tile else (1 if entry.row <= row_full else 2)
                candidates.append((entry.row, index, columns[:3], priority))
        candidates.sort(key=lambda candidate: (candidate[3], candidate[0], candidate[1]))
        selected = []
        for candidate in candidates:
            if not selected or candidate[0] != selected[0][0]:
                selected.append(candidate)
            if len(selected) == 2:
                break
        submissions = []
        for port, (row, index, columns, priority) in enumerate(selected):
            if request_ready[port]:
                entry = self.entries[index]
                submissions.append({'port': port, 'bank': row, 'pool_bank': index,
                                    'old_tile': not entry.current_tile, 'columns': tuple(columns),
                                    'values': tuple(tuple(entry.data[column]) for column in columns)})

        # 提交握手当拍即退休对应点：accumulator 的深度 2 表项接收 bundle 就算完成，
        # 池里不再保留任何在途状态；空闲存储下一拍才参与分配。
        touched = {write[0] for write in writes}
        for index, entry in enumerate(self.entries):
            if entry is None:
                continue
            old_final = entry.final
            for submission in submissions:
                if submission['pool_bank'] == index:
                    entry.retired.update(submission['columns'])
            if entry.current_tile and entry.row <= row_full:
                entry.final = True
            if old_final and index not in touched and entry.modified <= entry.retired:
                self.entries[index] = None
        for index, key, column, delta in writes:
            if self.entries[index] is None:
                self.entries[index] = PartialSumEntry(*key)
            entry = self.entries[index]
            if not entry.current_tile or (entry.row, entry.half) != key:
                raise AssertionError('部分和分配身份冲突')
            old = entry.data.get(column, [0] * 16)
            entry.data[column] = [signed_wrap(left + right, 13) for left, right in zip(old, delta)]
            entry.modified.add(column)

        # 新入队请求本拍不执行；已满队列即使同拍出队，也不能提前接收新包。
        if head and (head['cursor'] >= head['end'] or (express_columns and express_can_write)):
            head['cursor'] += len(express_columns)
            self.c2 = max(self.c2, head['cursor'])
            if head['cursor'] >= head['end']:
                self.express_queue.pop(0)
        if express_fire:
            new_start = max(0, express.column - 2)
            # 取消同体行互锁后，队头可能是更早行尚在推进的包，本拍新收的包可以属于下一行，
            # 其列号从 0 重新开始；每个队项各自推进自己的列游标，因此不再要求全局列号单调。
            self.express_queue.append({'packet': express, 'cursor': new_start,
                                       'end': min(12, express.column + express.run_length)})
        if flush and self._flush_tick is None:
            self._flush_tick = self.cycles
        if submissions:
            self._last_submit_tick = self.cycles
        if (self._flush_tick is not None and self._empty_tick is None
                and all(entry is None for entry in self.entries)):
            self._empty_tick = self.cycles
        if self._flush_tick is not None and self._idle_tick is None and self.idle:
            self._idle_tick = self.cycles
            base = (self._last_submit_tick if self._last_submit_tick is not None
                    else self._flush_tick)
            empty = self._empty_tick if self._empty_tick is not None else self.cycles
            _DRAIN_PARTS['issue_to_last_submit'].append(base - self._flush_tick)
            _DRAIN_PARTS['last_submit_to_empty'].append(empty - base)
            _DRAIN_PARTS['empty_to_idle'].append(self.cycles - empty)
            _DRAIN_PARTS['total'].append(self.cycles - self._flush_tick)
            if not self.fully_connected:
                if self.c1 < 12:
                    _DRAIN_PARTS.setdefault('c1_short', []).append(self.c1)
                if self.c2 < 12:
                    _DRAIN_PARTS.setdefault('c2_short', []).append(self.c2)
        if not active:
            self.front = 0
            self.c1 = 0
            self.c2 = 0
            self._pe_cursor_row = -1
        elif self.fully_connected:
            # 全连接模式的 front 是列游标，只在有包呈现时推进。
            if front_column is not None:
                self.front = max(self.front, int(front_column))
        else:
            # 两路一起决定前沿：任一路仍可能产生更早行的包，前沿就不能越过它。
            self.front = max(self.front, min(rmin_mode0, rmin_express))
        if flush:
            self.draining = True
        elif self.draining and self.idle:
            self.draining = False
        return {'normal_fire': normal_fire, 'express_fire': express_fire, 'submissions': submissions,
                'retire_candidate_ports': len(selected),
                'retire_blocked_ports': sum(not request_ready[port] for port in range(len(selected))),
                'row_dependency_blocked': row_dependency_blocked,
                'compute_active': bool(normal_fire or express_writes),
                'express_compute_active': bool(express_writes),
                'submission_ports': [submission['port'] for submission in submissions],
                'normal_ready': normal_ready, 'express_ready': express_ready,
                'normal_alloc_stall': normal_alloc_stall,
                'express_alloc_stall': express_alloc_stall,
                'front': int(self.front),
                'mode0_front': mode0_front, 'express_front': express_front,
                'pool_occupancy': sum(entry is not None for entry in self.entries),
                'express_occupancy': len(self.express_queue),
                'idle': self.idle}
