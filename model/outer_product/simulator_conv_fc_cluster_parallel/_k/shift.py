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


class split:
    """Encode one assigned input row into mode0 packages in real time."""

    def __init__(self, kernel_size=3, enabled_modes=(0, 1, 2)):
        self.k = kernel_size
        self.enabled_modes = frozenset(int(mode) for mode in enabled_modes)
        if 0 not in self.enabled_modes or not self.enabled_modes.issubset({0, 1, 2}):
            raise ValueError("enabled_modes must contain mode 0 and only use modes 0, 1, and 2")

    def init_stream(self, if_map, fifo, fifo_depth):
        """Initialize one real-time Split without scanning the feature map."""
        self.if_map = if_map
        self.assigned_row = None
        self.fifo = fifo
        self.fifo_depth = max(1, int(fifo_depth))
        self.row_work = None
        self.row_r = None
        self.row_done = True
        self.finished = False
        self.row_last_idx = -1
        self.row_last_mode0 = -1
        self.row_last_mode = -1
        self.mode0_known_bits = []
        self.mode0_pending_cols = deque()
        self.conv_mode_counts = collections.Counter({0: 0, 1: 0, 2: 0})

    def _start_assigned_row(self):
        """Load the row selected by SplitUnit's global row dispatcher."""
        self.row_r = self.assigned_row
        self.assigned_row = None
        self.row_work = self.if_map[self.row_r].view(-1).clone()
        self.row_done = False
        self.row_last_idx = -1
        self.row_last_mode0 = -1
        self.row_last_mode = -1
        self.mode0_known_bits = [None] * self.row_work.numel()
        self.mode0_pending_cols.clear()

    def _emit_ready_mode0_cols(self):
        """Write complete mode0 packages into this Split's bounded FIFO."""
        while self.mode0_pending_cols and len(self.fifo) < self.fifo_depth:
            c = self.mode0_pending_cols[0]
            # A mode0 window is usable only after all k input positions are
            # known. Dropped mode1/mode2 positions are represented as zero.
            if any(self.mode0_known_bits[c + offset] is None for offset in range(self.k)):
                return
            self.mode0_pending_cols.popleft()
            bits = [self.mode0_known_bits[c + offset] for offset in range(self.k)]
            self.fifo.append(_make_packet(self.row_r, c, 0, bits))

    def _finish_row_if_possible(self):
        """Mark the row complete only after delayed mode0 packages are emitted."""
        if self.mode0_pending_cols:
            return
        self.row_work = None
        self.row_done = True

    def tick_stream(self, allow_next_row=False):
        """Advance this Split by one cycle of real-time row processing."""
        if self.finished:
            return

        # 只有 SplitUnit 派发了新行，并且上一行 FIFO 已排空，才能开始下一行。
        if self.row_done:
            if allow_next_row and not self.fifo and not self.mode0_pending_cols:
                self._start_assigned_row()
            else:
                return

        # First drain packages delayed by FIFO back pressure. Only then can
        # the scanner inspect another nonzero pulse from this row.
        self._emit_ready_mode0_cols()
        # A pending window may still be waiting for future input bits, so it
        # must not block scanning. Only a full FIFO applies back pressure.
        if len(self.fifo) >= self.fifo_depth:
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
            # Mode2 is counted but never inserted into an issue FIFO.
            self.conv_mode_counts[2] += 1
            # The gap is filled first, then the mode2 run is masked. This is
            # required before testing newly completed mode0 windows.
            for idx in range(self.row_last_idx + 1, curr_idx):
                self.mode0_known_bits[idx] = 0
            for idx in range(curr_idx, curr_idx + process_len):
                self.mode0_known_bits[idx] = 0
            self._emit_ready_mode0_cols()
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
            # Mode1 is counted but never inserted into an issue FIFO.
            self.conv_mode_counts[1] += 1
            for idx in range(self.row_last_idx + 1, curr_idx):
                self.mode0_known_bits[idx] = 0
            self.mode0_known_bits[curr_idx] = 0
            self._emit_ready_mode0_cols()
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
    becomes idle first. Each Split owns one bounded FIFO, while expected_row
    still enforces strictly increasing row order at Core issue time.
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
        self.issue_fifo0 = deque()    # split0 的 FIFO，存放完整 package
        self.issue_fifo1 = deque()    # split1 的 FIFO，存放完整 package
        # The next row allowed to reach Core. row_owner records which FIFO
        # must be selected because dynamic assignment is not tied to parity.
        self.expected_row = 0
        self.next_unassigned_row = 0
        self.row_owner = {}
        self.skipped_zero_rows = set()
        self.completed_rows = set()
        self.is_finished = True
        # SplitUnit-owned mode histogram for conv. This is intentionally not
        # gathered at PE issue time, because cluster_parallel drops detected
        # mode1/mode2 packets before they reach the FIFO.
        self.conv_mode_counts = collections.Counter({0: 0, 1: 0, 2: 0})

    @property
    def issue_fifo(self):
        """Return the FIFO for the input row currently being issued.

        Core still reads one complete package through this existing interface;
        selecting the deque here avoids introducing a third FIFO.
        """
        owner = self.row_owner.get(self.expected_row, 0)
        return self.issue_fifo0 if owner == 0 else self.issue_fifo1

    def init_stream(self, if_map, mode='conv'):
        if not isinstance(if_map, torch.Tensor):
            if_map = torch.tensor(if_map, dtype=torch.int8)

        self.if_map = if_map
        self.mode = mode
        self.issue_fifo0.clear()
        self.issue_fifo1.clear()
        self.expected_row = 0
        self.next_unassigned_row = 0
        self.row_owner.clear()
        self.skipped_zero_rows.clear()
        self.completed_rows.clear()
        # Reset per feature map. The simulator later scales this representative
        # histogram by Cout groups when printing mode ratios.
        self.conv_mode_counts = collections.Counter({0: 0, 1: 0, 2: 0})

        self.linear_entries = []
        self.linear_entry_ptr = 0
        if mode == 'conv':
            # 初始化只建立两套 Split 状态，不预先分配输入行。真正的行队列
            # 调度从第一次 tick 开始，与 simulator 中 Cin 的派发时机一致。
            self.split0.enabled_modes = frozenset(self.enabled_modes)
            self.split1.enabled_modes = frozenset(self.enabled_modes)
            self.split0.init_stream(if_map, fifo=self.issue_fifo0, fifo_depth=self.fifo_depth)
            self.split1.init_stream(if_map, fifo=self.issue_fifo1, fifo_depth=self.fifo_depth)
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

    def tick(self):
        if self.is_finished:
            return

        if self.mode == 'conv':
            # Core has consumed the current package before this method. Save
            # drained row completions before assigning new work to that Split.
            self._record_completed_rows()
            self._advance_expected_row()

            start0 = self._assign_next_row_if_idle(self.split0, 0)
            start1 = self._assign_next_row_if_idle(self.split1, 1)
            self.split0.tick_stream(allow_next_row=start0)
            self.split1.tick_stream(allow_next_row=start1)

            self._record_completed_rows()
            self._advance_expected_row()
            self._sync_mode_counts()
        elif self.mode == 'linear':
            if len(self.issue_fifo) < self.fifo_depth:
                if self.linear_entry_ptr < len(self.linear_entries):
                    self.issue_fifo.append(self.linear_entries[self.linear_entry_ptr])
                    self.linear_entry_ptr += 1
        self._update_finished_state()

    def _record_completed_rows(self):
        """Keep completion state after an idle Split starts another row."""
        for current in (self.split0, self.split1):
            if current.row_r is None:
                continue
            if current.row_done and not current.fifo and not current.mode0_pending_cols:
                self.completed_rows.add(current.row_r)

    def _assign_next_row_if_idle(self, current, split_id):
        """从全局行队列向空闲 Split 派发下一条非零行。"""
        if (
            current.finished
            or not current.row_done
            or current.fifo
            or current.mode0_pending_cols
        ):
            return False

        # 与 Cin 队列派发相同：始终检查队首。全零行没有计算任务，直接
        # 标记完成并继续检查下一行，不占用 split0/split1 的处理周期。
        while self.next_unassigned_row < self.if_map.shape[0]:
            row = self.next_unassigned_row
            self.next_unassigned_row += 1
            if not torch.any(self.if_map[row] != 0):
                self.completed_rows.add(row)
                self.skipped_zero_rows.add(row)
                continue

            self.row_owner[row] = split_id
            # tick_stream() 在本拍装载并扫描该行。
            current.assigned_row = row
            return True

        current.finished = True
        return False

    def _advance_expected_row(self):
        """Advance only across completed, fully drained rows."""
        while self.expected_row in self.completed_rows:
            self.expected_row += 1

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

        self.is_finished = (
            input_done
            and not self.issue_fifo0
            and not self.issue_fifo1
        )
