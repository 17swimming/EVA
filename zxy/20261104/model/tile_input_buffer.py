"""有界输入窗口：ping-pong 两组，各组按 tile 奇偶分为两个物理 buffer。"""


class TileInputBuffer:
    """限制可派发的数据窗口；片外访存周期仍由整网络原有模型计量。"""

    def __init__(self, source, capacity=128):
        self.source = source
        self.capacity = capacity
        self.groups = [None, None]
        self.loaded = set()
        self.next_window = 0
        self.window_count = (len(source) + capacity - 1) // capacity
        self.resident_peak = 0
        self.trace = []
        self.refill()

    def refill(self):
        """释放已复制到 core 的窗口，按原有输入就绪假设准备后继窗口。"""
        for index, group in enumerate(self.groups):
            if group is not None and group['remaining'] == 0:
                self.trace.append(('release', group['window'], index))
                self.groups[index] = None
        while self.next_window < self.window_count:
            index = self.next_window % 2
            if self.groups[index] is not None:
                break
            begin = self.next_window * self.capacity
            end = min(begin + self.capacity, len(self.source))
            self.groups[index] = dict(window=self.next_window, begin=begin, end=end,
                even=self.source[begin:end:2], odd=self.source[begin+1:end:2], remaining=end-begin)
            self.trace.append(('prepare', self.next_window, index))
            self.next_window += 1
        resident = sum(group['end']-group['begin'] for group in self.groups if group is not None)
        self.resident_peak = max(self.resident_peak, resident)
        assert resident <= 2*self.capacity

    def available(self, channel):
        group = self.groups[(channel // self.capacity) % 2]
        return group is not None and group['begin'] <= channel < group['end']

    def read(self, channel):
        assert self.available(channel), '输入尚未进入可读取的 ping-pong buffer'
        group = self.groups[(channel // self.capacity) % 2]
        offset = channel - group['begin']
        return group['odd' if offset % 2 else 'even'][offset // 2]

    def release_input(self, channel):
        """整块输入已经复制到 core，此后 psum 退休不再占用输入窗口。"""
        assert channel not in self.loaded
        assert self.available(channel)
        self.loaded.add(channel)
        self.groups[(channel // self.capacity) % 2]['remaining'] -= 1
