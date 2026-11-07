"""遍历result array以完成LIF计算的地址及两拍尾流水周期契约；有效性只由输出分块形状决定。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadBeat:
    group: int
    slot: int
    banks: tuple


class LIFScan:
    """safe_axis 是独占上界：卷积为 bank，全连接为 slot。

    调用方必须证明该上界以下没有未来贡献、部分和、请求束或写回流水。
    全连接优先读取低编号 slot，以便前三行输出点在后续输出行组计算期间完成泄漏积分发放计算。
    """

    def __init__(self, bank_h, bank_w, *, conv_shape=None, fc_m=None,
                 fc_pe_rows=3, points=3):
        if points <= 0:
            raise ValueError('LIF points must be positive')
        self.fc = fc_m is not None
        self.beats = []
        if self.fc:
            if not 1 <= fc_pe_rows <= 3 or not 0 <= fc_m <= bank_w * (bank_h // 3):
                raise ValueError('FC shape exceeds result array')
            for m_group in range((fc_m + bank_w - 1) // bank_w):
                banks = min(bank_w, fc_m - m_group * bank_w)
                for pe_row in range(fc_pe_rows):
                    for bank in range(0, banks, points):
                        self.beats.append(ReadBeat(bank // points, 3*m_group + pe_row,
                                                   tuple(range(bank, min(bank + points, banks)))))
        else:
            oh, ow = conv_shape or (bank_h, bank_w)
            if not 0 <= oh <= bank_h or not 0 <= ow <= bank_w:
                raise ValueError('CONV shape exceeds result array')
            for bank in range(0, oh, points):
                for slot in range(ow):
                    self.beats.append(ReadBeat(bank // points, slot,
                                               tuple(range(bank, min(bank + points, oh)))))
        self.index = 0
        self.clock = 0
        self.inflight = []
        self.done = not self.beats
        self.read_trace = []
        self.write_trace = []
        self.safe_trace = []

    def tick(self, safe_axis):
        """一次时钟沿；读请求在 n 采样，写回在 n+2 采样。"""
        self.safe_trace.append(int(safe_axis))
        read = None
        writes = [beat for cycle, beat in self.inflight if cycle == self.clock]
        self.inflight = [(cycle, beat) for cycle, beat in self.inflight if cycle > self.clock]
        self.write_trace.extend((self.clock, beat) for beat in writes)
        if not self.done and self.index < len(self.beats):
            beat = self.beats[self.index]
            required = beat.slot + 1 if self.fc else max(beat.banks) + 1
            if required <= safe_axis:
                read = beat
                self.index += 1
                self.read_trace.append((self.clock, beat))
                self.inflight.append((self.clock + 2, beat))
        if self.index == len(self.beats) and not self.inflight:
            self.done = True
        self.clock += 1
        return read, writes

    def replay_cycles(self, **shape):
        """复用后端安全时间线，估计最后输出通道组的可见泄漏积分发放计算周期差。"""
        scan = LIFScan(**shape)
        for safe in self.safe_trace:
            scan.tick(safe)
            if scan.done:
                return scan.clock
        while not scan.done:
            scan.tick(max(shape['bank_h'], shape['bank_w']))
        return scan.clock


def lif_value(vmem, threshold, leak, reset=0, width=16):
    """与现有 lif.v 一致：有符号阈值判断优先，减去泄漏值并截取规定宽度后将负值归零。"""
    mask = (1 << width) - 1

    def signed(value):
        value &= mask
        return value - (1 << width) if value & (1 << (width-1)) else value

    if signed(vmem) >= signed(threshold):
        return reset & mask, 1
    value = signed((vmem - leak) & mask)
    return max(0, value), 0
