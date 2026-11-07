"""同一空间tiles内保持 psum pool，重叠下一 tile 的计算与旧 bank 的退休。"""
from collections import Counter

from .current_accumulating_unit import CurrentAccumulatingUnit, Packet
from .shift import SplitUnit


def packet(value):
    if value is None:
        return None
    return Packet(int(value['r']), int(value['c']), int(value['mode']),
                  tuple(int(item) for item in value['bitstream']))


class StreamingCore:
    """计算完成脉冲释放任务位置，psum pool 排空状态独立维护。"""

    def __init__(self, fully_connected=False, pool_banks=8, fifo_depth=8, weights=None,
                 conv_shape=(12, 12)):
        self.fully_connected = fully_connected
        self.weights = weights if weights is not None else [[[1] * 3 for _ in range(3)] for _ in range(16)]
        self.unit = CurrentAccumulatingUnit(self.weights, fully_connected, pool_banks, conv_shape)
        self.fifo_depth = fifo_depth
        self.split = None
        self.pending = None
        self.pending_split = None
        self.channel = None
        self.state = 'idle'
        self.done = False
        self.statistics = Counter()
        self._previous_express = False

    @property
    def ready(self):
        # 保留一次预取容量，当前 splitunit 发射完毕后才接收后继任务。
        return self.pending is None and (self.split is None or self.split.is_finished)

    @property
    def is_finished(self):
        return self.split is None and self.pending is None and self.unit.idle

    @property
    def pending_tile(self):
        return self.pending

    @property
    def has_old_banks(self):
        return any(entry is not None and not entry.current_tile for entry in self.unit.entries)

    def step(self, start, load, tile, channel, request_ready):
        """只使用拍初完成条件交接，旧部分和在交接后继续参加退休仲裁。"""
        ready_before = self.ready
        self.done = False
        if self.split is not None and self.split.is_finished and not self.unit.express_queue:
            self.unit.close_tile()
            self.split = None
            self.done = True
        if self.split is None and self.pending_split is not None:
            self.split, self.pending_split = self.pending_split, None
            self.channel, self.pending = self.pending, None
            self.unit.begin_tile(self.weights)
        split = self.split
        normal = packet(split.peek_mode0_packet()) if split is not None else None
        express = packet(split.peek_express_packet()) if split is not None else None
        normal_front = split.mode0_front_row() if split is not None else None
        express_front = split.express_front_row() if split is not None else None
        finished = split is None or split.is_finished
        supply_diagnosis = split.supply_wait_diagnosis() if split is not None and normal is None else None
        # 统计只观察实际送入计算单元的拍初接口，不推进队列或修改握手条件。
        express_busy_before = bool(self.unit.express_queue)
        result = self.unit.step(
            normal, express, normal_front, express_front,
            int(normal.column) if normal is not None else None, request_ready,
            flush=finished,
            mode0_absent=bool(split is not None and normal_front is not None
                             and split.mode0_absent(normal_front)),
            express_absent=bool(split is not None and express_front is not None
                               and split.express_absent(express_front)))
        if result['normal_fire']:
            split.pop_mode0_packet()
        if result['express_fire']:
            split.pop_express_packet()
        result['express_mode'] = express.mode if result['express_fire'] else None
        result['split_present'] = split is not None
        result['express_busy_before'] = express_busy_before
        result['split_supply_stall'] = bool(split is not None and normal is None and (
            self.fully_connected or (not finished and not express_busy_before
                                     and not result['express_fire'])))
        result['split_wait_diagnosis'] = supply_diagnosis
        result['tile_handoff'] = bool(start or load or self.pending is not None
                                      or self.pending_split is not None or self.done)
        self.statistics['normal'] += int(result['normal_fire'])
        self.statistics['express'] += int(result['express_fire'])
        self.statistics['express_consecutive_cycles'] += int(result['express_fire'] and self._previous_express)
        self.statistics['express_issue_cycles'] += int(result['express_fire'])
        self.statistics['issue_cycles'] += int(result['normal_fire'] or result['express_fire'])
        self._previous_express = bool(result['express_fire'])
        # 新接收的输入下一拍才进入计算，当前周期只推进已经存在的 splitunit。
        if split is not None:
            split.tick()
        if self.pending_split is not None:
            self.pending_split.tick()
        if start:
            assert ready_before and self.pending is None
            self.pending = channel
        if load:
            assert self.pending is not None and self.pending_split is None
            self.pending_split = SplitUnit(w=12, fifo_depth=self.fifo_depth)
            self.pending_split.init_stream(tile, 'linear' if self.fully_connected else 'conv')
        self.state = ('compute' if self.split is not None else
                      ('drain' if not self.unit.idle else 'idle'))
        return result
