import os
import collections
from collections import deque

import torch


def _make_packet(r, c, mode, bits):
    """Create the complete package consumed by Core."""
    return {
        'bitstream': torch.tensor(bits, dtype=torch.int8),
        'r': torch.tensor([int(r)], dtype=torch.int8),
        'c': torch.tensor([int(c)], dtype=torch.int8),
        'mode': torch.tensor([int(mode)], dtype=torch.int8),
    }


# 编码结束到核看到 finished 之间的寄存拍数，与 RTL 的编码结束登记一致。
FINISH_LATENCY_CYCLES = 2


class split:
    """Encode one assigned input row into enabled-mode packages in real time."""

    def __init__(self, kernel_size=3, enabled_modes=(0, 1, 2)):
        self.k = kernel_size
        self.enabled_modes = frozenset(int(mode) for mode in enabled_modes)
        if 0 not in self.enabled_modes or not self.enabled_modes.issubset({0, 1, 2}):
            raise ValueError("enabled_modes must contain mode 0 and only use modes 0, 1, and 2")

    def init_stream(self, if_map, mode0_fifo, express_fifo, fifo_depth):
        """Initialize one real-time Split without scanning the feature map.

        mode0 包与 mode1/mode2 包进入两个独立的有界 FIFO，两个发射口因此可以
        各自推进，不再要求 PE 与 express 停在同一 ifmap 行。
        """
        self.if_map = if_map
        self.assigned_row = None
        self.fifo_mode0 = mode0_fifo
        self.fifo_express = express_fifo
        self.fifo_depth = max(1, int(fifo_depth))
        self.row_work = None
        self.row_r = None
        self.row_done = True
        self.row_gap = os.environ.get("EVA_ROW_GAP", "0") == "1"
        self.finished = False
        self.row_last_idx = -1
        self.row_last_mode0 = -1
        self.row_last_mode = -1
        self.mode0_known_bits = []
        self.mode0_pending_cols = deque()
        self.fifo_write_used = False
        self.conv_mode_counts = collections.Counter({0: 0, 1: 0, 2: 0})
        # 本行是否完全没有 mode0 包，行结束时才定稿。
        self.row_mode0_absent = {}
        self.row_mode0_count = 0
        self.row_express_absent = {}
        self.row_express_count = 0

    def _start_assigned_row(self):
        """Load the row selected by SplitUnit's priority encoder."""
        self.row_r = self.assigned_row
        self.assigned_row = None
        self.row_work = self.if_map[self.row_r].view(-1).clone()
        self.row_done = False
        self.row_last_idx = -1
        self.row_last_mode0 = -1
        self.row_last_mode = -1
        self.mode0_known_bits = [None] * self.row_work.numel()
        self.mode0_pending_cols.clear()
        self.row_mode0_count = 0
        self.row_express_count = 0

    def _emit_ready_mode0_cols(self):
        """最多写入一个 mode0 包；本拍写端口也与 express 包共享。"""
        if self.fifo_write_used or not self.mode0_pending_cols or len(self.fifo_mode0) >= self.fifo_depth:
            return
        c = self.mode0_pending_cols[0]
        # A mode0 window is usable only after all k input positions are
        # known. Positions handled by mode1/mode2 are represented as zero
        # in overlapping mode0 windows; their work is carried by their own
        # packages in this same FIFO.
        if any(self.mode0_known_bits[c + offset] is None for offset in range(self.k)):
            return
        self.mode0_pending_cols.popleft()
        bits = [self.mode0_known_bits[c + offset] for offset in range(self.k)]
        self.fifo_mode0.append(_make_packet(self.row_r, c, 0, bits))
        self.row_mode0_count += 1
        self.fifo_write_used = True

    def _finish_row_if_possible(self):
        """Mark the row complete only after delayed mode0 packages are emitted."""
        if self.mode0_pending_cols:
            return
        self.row_mode0_absent[self.row_r] = (self.row_mode0_count == 0)
        self.row_express_absent[self.row_r] = (self.row_express_count == 0)
        self.row_work = None
        self.row_done = True

    def tick_stream(self, allow_next_row=False):
        """Advance this Split by one cycle of real-time row processing."""
        self.fifo_write_used = False
        if self.finished:
            return

        # 本行编码结束即可派发下一行：包自带行号，两个 FIFO 允许跨行积压，
        # 快的流不再被慢的流卡在编码层。
        if self.row_done:
            if allow_next_row and not self.mode0_pending_cols:
                self._start_assigned_row()
                # RTL shift/split.v：`start && in_ready` 只把这条 W 宽输入行
                # **锁存**进来；编码器从**下一拍**才开始找未处理非零脉冲。
                # 所以换行的那一拍不产包 —— 这就是 split_unit_spec 里那句
                # 「剩余行间空拍仅在下一行首包尚未进入发射窗口时出现」。
                if self.row_gap:
                    return
            else:
                return

        # First drain packages delayed by FIFO back pressure. Only then can
        # the scanner inspect another nonzero pulse from this row.
        self._emit_ready_mode0_cols()
        # 共享反压：任一 FIFO 满就停止本 Split 的扫描，下一拍再继续。
        if len(self.fifo_mode0) >= self.fifo_depth or len(self.fifo_express) >= self.fifo_depth:
            return

        nz_mask = self.row_work != 0
        if not nz_mask.any():
            # Fill the tail before checking which delayed mode0 windows are
            # complete.
            for idx in range(self.row_last_idx + 1, self.row_work.numel()):
                self.mode0_known_bits[idx] = 0
            self._emit_ready_mode0_cols()
            self._finish_row_if_possible()
            return

        curr_idx = nz_mask.nonzero(as_tuple=True)[0][0].item()    # priority encoder
        zero_count = curr_idx - self.row_last_idx - 1  
        width = self.row_work.numel()
        max_c = width - self.k

        run_len = 0
        while curr_idx + run_len < width and self.row_work[curr_idx + run_len] == 1:
            run_len += 1

        if 2 in self.enabled_modes and run_len >= 4:
            process_len = min(run_len, 7)
            # The gap is filled first, then the mode2 run is masked. This is
            # required before testing newly completed mode0 windows.
            for idx in range(self.row_last_idx + 1, curr_idx):
                self.mode0_known_bits[idx] = 0
            for idx in range(curr_idx, curr_idx + process_len):
                self.mode0_known_bits[idx] = 0
            self._emit_ready_mode0_cols()

            # mode1/mode2 共用 express FIFO，mode0 独占自己的 FIFO；
            # 只有 express FIFO 满才阻塞本分支。
            if self.fifo_write_used or len(self.fifo_express) >= self.fifo_depth:
                return
            self.row_express_count += 1
            self.fifo_express.append(
                _make_packet(
                    self.row_r,
                    curr_idx,
                    2,
                    [(process_len >> 2) & 1, (process_len >> 1) & 1, process_len & 1],
                )
            )
            self.fifo_write_used = True
            self.conv_mode_counts[2] += 1
            self.row_work[curr_idx : curr_idx + process_len] = 0
            self.row_last_idx = curr_idx
            self.row_last_mode = 2
            return

        is_mode1 = (
            zero_count >= 2
            and curr_idx < width - 2
            and self.row_work[curr_idx + 1] == 0
            and self.row_work[curr_idx + 2] == 0
        )
        if 1 in self.enabled_modes and is_mode1:
            for idx in range(self.row_last_idx + 1, curr_idx):
                self.mode0_known_bits[idx] = 0
            self.mode0_known_bits[curr_idx] = 0
            self._emit_ready_mode0_cols()

            # mode1 与 mode2 共用 express FIFO，不再与 mode0 抢同一个队尾。
            if self.fifo_write_used or len(self.fifo_express) >= self.fifo_depth:
                return
            self.row_express_count += 1
            self.fifo_express.append(
                _make_packet(
                    self.row_r,
                    curr_idx,
                    1,
                    [int(self.row_work[curr_idx].item())],
                )
            )
            self.fifo_write_used = True
            self.conv_mode_counts[1] += 1
            self.row_last_mode = 1
        else:
            for idx in range(self.row_last_idx + 1, curr_idx):
                self.mode0_known_bits[idx] = 0
            self.mode0_known_bits[curr_idx] = int(self.row_work[curr_idx].item())

            distance = curr_idx - self.row_last_mode0
            end_k = min(curr_idx, max_c)
            start_k = curr_idx - self.k + 1 if distance >= self.k else curr_idx - distance + 1
            for c in range(start_k, end_k + 1):
                if 0 <= c <= max_c:
                    self.mode0_pending_cols.append(c)
                    # Count mode0 when shift generates it, before FIFO issue.
                    self.conv_mode_counts[0] += 1
            self._emit_ready_mode0_cols()
            self.row_last_mode0 = curr_idx
            self.row_last_mode = 0

        self.row_work[curr_idx] = 0
        self.row_last_idx = curr_idx

class SplitUnit:
    """Cycle-level streaming split front-end for the cluster simulator.

    Conv mode dynamically assigns the next unclaimed row to whichever Split
    becomes idle first. 每路拥有独立 FIFO：mode0 包进 mode0 FIFO，mode1/mode2
    包进 express FIFO；每一路各自维护期望行，保证自身仍按行号升序发射，
    但 PE 与 express 可以停在不同 ifmap 行。
    """

    def __init__(
        self,
        kernel_size=3,
        w=32,
        fifo_depth=4,
        sram_vec_capacity=10,
        fetch_rows=4,
        decode_lanes=None,
        rob_depth=None,
        enabled_modes=(0, 1, 2),
        num_split=2,
    ):
        self.k = kernel_size
        self.w = w
        self.enabled_modes = frozenset(int(mode) for mode in enabled_modes)
        if 0 not in self.enabled_modes or not self.enabled_modes.issubset({0, 1, 2}):
            raise ValueError("enabled_modes must contain mode 0 and only use modes 0, 1, and 2")
        # Keep state inside the two named Split objects.  This is deliberately
        # written as split0/split1 rather than a generic lane abstraction so
        # the row ownership is explicit in the simulator trace.
        self.split0 = split(kernel_size=kernel_size, enabled_modes=self.enabled_modes)
        self.split1 = split(kernel_size=kernel_size, enabled_modes=self.enabled_modes)
        # Existing simulator setup code updates split.enabled_modes directly;
        # retain this alias for compatibility with that code.
        self.splits = [
            split(kernel_size=kernel_size, enabled_modes=self.enabled_modes)
            for _ in range(num_split)
        ]
        self.num_split = num_split

        self.fifo_depth = max(1, int(fifo_depth))
        # mode0 与 mode1/mode2 各自独立排队；两个发射口分别从自己的 FIFO 取包，
        # 因此 PE 与 express 可以停在不同 ifmap 行。
        self.mode0_fifos = [deque(), deque()]
        self.express_fifos = [deque(), deque()]
        # 每路单独维护期望行：该行在该路 FIFO 中排空且编码完成后才前进，
        # 保证每一路自身仍按行号升序发射。row_owner 记录该行属于哪个 Split。
        self.expected_mode0_row = 0
        self.expected_express_row = 0
        self.next_unassigned_row = 0
        self.row_owner = {}
        self.skipped_zero_rows = set()
        self.encoded_rows = set()
        self.is_finished = True
        # 编码结束到核看到 finished 之间的寄存拍数（见 _update_finished_state）。
        self.finished_latency = FINISH_LATENCY_CYCLES
        # SplitUnit owns the histogram because modes are classified when their
        # packages enter the per-Split FIFO, before Core issues them.
        self.conv_mode_counts = collections.Counter({0: 0, 1: 0, 2: 0})

    @staticmethod
    def _packet_row(packet):
        return int(packet['r'].item())

    def _stream_fifo(self, kind, owner):
        return (self.mode0_fifos if kind == 'mode0' else self.express_fifos)[owner]

    def _head_of_expected_row(self, kind, row):
        """Return the FIFO head of the row's owner when it is still that row.

        线性（全连接）模式没有行队列，直接取 split0 的 mode0 FIFO。
        """
        if self.mode != 'conv':
            if self.mode0_fifos[0]:
                return self.mode0_fifos[0][0]
            # 空队列直接呈现本拍可生成的全连接package；读取接口不推进状态。
            if not self.startup_cycles_remaining and self.linear_entry_ptr < len(self.linear_entries):
                return self.linear_entries[self.linear_entry_ptr]
            return None
        owner = self.row_owner.get(row)
        if owner is None:
            return None
        fifo = self._stream_fifo(kind, owner)
        if not fifo or self._packet_row(fifo[0]) != int(row):
            return None
        return fifo[0]

    def _pop_expected_row(self, kind, row):
        if self.mode != 'conv':
            if self.mode0_fifos[0]:
                return self.mode0_fifos[0].popleft()
            package = self._head_of_expected_row(kind, row)
            if package is not None:
                assert not self.linear_bypass_consumed
                self.linear_entry_ptr += 1
                self.linear_bypass_consumed = True
            return package
        packet = self._head_of_expected_row(kind, row)
        if packet is None:
            return None
        self._stream_fifo(kind, self.row_owner[row]).popleft()
        return packet

    def peek_mode0_packet(self):
        return self._head_of_expected_row('mode0', self.expected_mode0_row)

    def supply_wait_diagnosis(self):
        """只读拍初状态；解释当前mode0接口无包，不消费package或推进编码。"""
        row = self.expected_mode0_row
        mode0 = [package for queue in self.mode0_fifos for package in queue]
        owner = self.row_owner.get(row) if self.mode == 'conv' else None
        encoder = (self.split0, self.split1)[owner] if owner is not None else None
        flags = dict(later_row_buffered=bool(self.mode == 'conv' and
                         any(self._packet_row(package) > row for package in mode0)),
                     mode0_fifo_nonempty=bool(mode0),
                     express_fifo_nonempty=any(self.express_fifos),
                     owner_mode0_fifo_full=bool(encoder is not None and
                         len(encoder.fifo_mode0) >= self.fifo_depth),
                     owner_express_fifo_full=bool(encoder is not None and
                         len(encoder.fifo_express) >= self.fifo_depth),
                     expected_row_encoded=row in self.encoded_rows)
        if self.startup_cycles_remaining:
            return 'startup', flags
        input_done = (self.linear_entry_ptr >= len(self.linear_entries) if self.mode == 'linear'
                      else self.split0.finished and self.split1.finished)
        if input_done and not mode0 and not any(self.express_fifos):
            return 'finish_confirmation', flags
        if self.mode == 'linear':
            return 'linear_supply', flags
        if row >= self.if_map.shape[0]:
            return 'mode0_stream_complete', flags
        if flags['owner_mode0_fifo_full'] or flags['owner_express_fifo_full']:
            return 'fifo_backpressure', flags
        if flags['later_row_buffered'] or row in self.encoded_rows:
            return 'row_order_wait', flags
        if encoder is None:
            return 'row_dispatch', flags
        if encoder.row_done or encoder.row_r != row:
            return 'row_handoff', flags
        if encoder.mode0_pending_cols:
            column = encoder.mode0_pending_cols[0]
            if any(encoder.mode0_known_bits[column+offset] is None for offset in range(self.k)):
                return 'window_assembly', flags
        if encoder.row_work is not None:
            return 'encoder_supply', flags
        return 'unclassified', flags

    def pop_mode0_packet(self):
        return self._pop_expected_row('mode0', self.expected_mode0_row)

    def peek_express_packet(self):
        return self._head_of_expected_row('express', self.expected_express_row)

    def pop_express_packet(self):
        return self._pop_expected_row('express', self.expected_express_row)

    def init_stream(self, if_map, mode='conv'):
        if not isinstance(if_map, torch.Tensor):
            if_map = torch.tensor(if_map, dtype=torch.int8)

        self.if_map = if_map
        self.mode = mode
        # 任务派发和输入接收由调度器推进，此处只描述编码入口等待。
        # 全连接首行地址已随整块输入写入完成预取，编码输出直接连接发射通路。
        # 全连接保留一拍编码入口等待，卷积仍沿用当前两拍编码入口模型。
        self.startup_cycles_remaining = 1 if mode == 'linear' else 2
        for fifo in self.mode0_fifos + self.express_fifos:
            fifo.clear()
        self.expected_mode0_row = 0
        self.expected_express_row = 0
        self.next_unassigned_row = 0
        self.row_owner.clear()
        self.skipped_zero_rows.clear()
        self.encoded_rows.clear()
        # Reset per feature map. The simulator later scales this representative
        # histogram by Cout groups when printing mode ratios.
        self.conv_mode_counts = collections.Counter({0: 0, 1: 0, 2: 0})

        self.linear_entries = []
        self.linear_entry_ptr = 0
        self.linear_bypass_consumed = False
        if mode == 'conv':
            # 初始化只建立两套 Split 状态，不预先分配输入行。真正的行队列
            # 调度从第一次 tick 开始，与 simulator 中 Cin 的派发时机一致。
            self.split0.enabled_modes = frozenset(self.enabled_modes)
            self.split1.enabled_modes = frozenset(self.enabled_modes)
            self.split0.init_stream(if_map, mode0_fifo=self.mode0_fifos[0],
                                    express_fifo=self.express_fifos[0],
                                    fifo_depth=self.fifo_depth)
            self.split1.init_stream(if_map, mode0_fifo=self.mode0_fifos[1],
                                    express_fifo=self.express_fifos[1],
                                    fifo_depth=self.fifo_depth)
        elif mode == 'linear':
            # 对于线性层，ifmap只要一整行非全零，就生成一个 packet
            # 现在的实现是在initial时就全生成，不过即使tick时才生成，也能保证每拍都有package
            # 毕竟快速找到非零行，还是容易的。
            self.split0.finished = True
            self.split1.finished = True
            base_r = 2
            for tb in range(if_map.shape[0]):
                tb_vec = if_map[tb].view(-1)
                if not torch.any(tb_vec != 0):
                    continue
                bits = [int(v) for v in tb_vec[:3].tolist()]
                bits.extend([0] * (3 - len(bits)))
                self.linear_entries.append(
                    _make_packet(base_r + (tb // self.w) * 3, tb % self.w, 0, bits[:3])
                )
        else:
            raise ValueError(f"Unsupported split mode: {mode}")

        self.is_finished = False
        self._update_finished_state()

    def mode0_absent(self, row):
        """该 ifmap 行是否没有任何 mode0 包；行未结束时返回 False（保守）。"""
        return (getattr(self.split0, 'row_mode0_absent', {}).get(row, False)
                or getattr(self.split1, 'row_mode0_absent', {}).get(row, False))

    def express_absent(self, row):
        """该 ifmap 行是否没有任何 mode1/2 包；行未结束时返回 False（保守）。"""
        return (getattr(self.split0, 'row_express_absent', {}).get(row, False)
                or getattr(self.split1, 'row_express_absent', {}).get(row, False))

    def tick(self):
        if self.is_finished:
            return

        # 行缓冲读地址发出到数据回到编码器之间的一拍（见 init_stream）。
        if self.startup_cycles_remaining:
            self.startup_cycles_remaining -= 1
            return

        if self.mode == 'conv':
            # Core has consumed the current package before this method. Save
            # drained row completions before assigning new work to that Split.
            self._record_completed_rows()
            self._advance_expected_rows()

            start0 = self._assign_next_row_if_idle(self.split0, 0)
            start1 = self._assign_next_row_if_idle(self.split1, 1)
            self.split0.tick_stream(allow_next_row=start0)
            self.split1.tick_stream(allow_next_row=start1)

            self._record_completed_rows()
            self._advance_expected_rows()
            self._sync_mode_counts()
        elif self.mode == 'linear':
            if self.linear_bypass_consumed:
                # 本拍旁路已经完成一次生产和握手，不能再向队列生产第二个package。
                self.linear_bypass_consumed = False
            elif len(self.mode0_fifos[0]) < self.fifo_depth:
                if self.linear_entry_ptr < len(self.linear_entries):
                    self.mode0_fifos[0].append(self.linear_entries[self.linear_entry_ptr])
                    self.linear_entry_ptr += 1
        self._update_finished_state()

    def _record_completed_rows(self):
        """登记编码结束的行；包可能仍积压在 FIFO 中，是否排空由各流单独判断。"""
        for current in (self.split0, self.split1):
            if current.row_r is not None and current.row_done:
                self.encoded_rows.add(int(current.row_r))

    def _row_drained(self, kind, row):
        """该行在该路是否已经既编码结束、又没有任何残留包。"""
        if row in self.skipped_zero_rows:
            return True
        if row not in self.encoded_rows:
            return False
        owner = self.row_owner.get(row)
        if owner is None:
            return False
        return all(self._packet_row(packet) != row
                   for packet in self._stream_fifo(kind, owner))

    def _advance_expected_rows(self):
        """两路各自越过已经排空的行，互不等待。"""
        while self._row_drained('mode0', self.expected_mode0_row):
            self.expected_mode0_row += 1
        while self._row_drained('express', self.expected_express_row):
            self.expected_express_row += 1

    def mode0_front_row(self):
        """仍可能产生 mode0 包的最小行；行屏障据此判断哪些低行已定稿。"""
        return 12 if self.mode != 'conv' else self.expected_mode0_row

    def express_front_row(self):
        """仍可能产生 mode1/mode2 包的最小行。"""
        return 12 if self.mode != 'conv' else self.expected_express_row

    def _assign_next_row_if_idle(self, current, split_id):
        """从全局行队列向空闲 Split 派发下一条非零行。"""
        if current.finished or not current.row_done:
            return False

        # 与 Cin 队列派发相同：始终检查队首。全零行没有计算任务，直接
        # 标记完成并继续检查下一行，不占用 split0/split1 的处理周期。
        while self.next_unassigned_row < self.if_map.shape[0]:
            row = self.next_unassigned_row
            self.next_unassigned_row += 1
            if not torch.any(self.if_map[row] != 0):
                self.skipped_zero_rows.add(row)
                continue

            self.row_owner[row] = split_id
            # tick_stream() 在本拍装载并扫描该行。
            current.assigned_row = row
            return True

        current.finished = True
        return False

    def _sync_mode_counts(self):
        """Aggregate mode counters maintained independently by split0/split1."""
        self.conv_mode_counts = collections.Counter({0: 0, 1: 0, 2: 0})
        self.conv_mode_counts.update(self.split0.conv_mode_counts)
        self.conv_mode_counts.update(self.split1.conv_mode_counts)

    def _update_finished_state(self):
        if self.mode == 'conv':
            input_done = self.split0.finished and self.split1.finished
        elif self.mode == 'linear':
            input_done = self.linear_entry_ptr >= len(self.linear_entries)
        else:
            input_done = True

        encoded = (
            input_done
            and not self.mode0_fifos[0]
            and not self.mode0_fifos[1]
            and not self.express_fifos[0]
            and not self.express_fifos[1]
        )
        # 与 RTL 对齐：编码结束到核看到 finished 之间有两拍寄存（编码结束登记与核侧判定），
        # 这两拍正是 RTL 进入排空状态前的固定延迟。
        if not encoded:
            self.finished_latency = 0
        elif not self.is_finished:
            self.finished_latency += 1
        self.is_finished = encoded and self.finished_latency >= FINISH_LATENCY_CYCLES
