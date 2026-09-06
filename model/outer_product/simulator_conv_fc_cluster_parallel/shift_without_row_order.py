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
    """Reference row encoder used by tests and compatibility paths."""

    def __init__(self, kernel_size=3, enabled_modes=(0, 1, 2)):
        self.k = kernel_size
        self.enabled_modes = frozenset(int(mode) for mode in enabled_modes)
        if 0 not in self.enabled_modes or not self.enabled_modes.issubset({0, 1, 2}):
            raise ValueError("enabled_modes must contain mode 0 and only use modes 0, 1, and 2")

    def process(self, if_line, r: int):
        if not isinstance(if_line, torch.Tensor):
            if_line = torch.tensor(if_line, dtype=torch.int8)

        work_line = if_line.view(-1).clone()
        width = len(work_line)
        max_c = width - self.k
        bitstream = []
        c_array = []
        mode_array = []

        last_idx = -1
        last_mode0 = -1
        last_mode = -1
        while True:
            nz_mask = work_line != 0
            if not nz_mask.any():
                break

            curr_idx = nz_mask.nonzero(as_tuple=True)[0][0].item()
            zero_count = curr_idx - last_idx - 1

            run_len = 0
            while curr_idx + run_len < width and work_line[curr_idx + run_len] == 1:
                run_len += 1

            if 2 in self.enabled_modes and run_len >= 4:
                if last_mode == 0:
                    bitstream.extend([0] * (curr_idx - last_idx - 1))
                process_len = min(run_len, 7)
                c_array.append(curr_idx)
                mode_array.append(2)
                bitstream.extend([(process_len >> 2) & 1, (process_len >> 1) & 1, process_len & 1])
                work_line[curr_idx : curr_idx + process_len] = 0
                last_idx = curr_idx
                last_mode = 2
                continue

            is_mode1 = (
                zero_count >= 2
                and curr_idx < width - 2
                and work_line[curr_idx + 1] == 0
                and work_line[curr_idx + 2] == 0
            )
            if 1 in self.enabled_modes and is_mode1:
                if last_mode == 0:
                    bitstream.extend([0, 0])
                c_array.append(curr_idx)
                mode_array.append(1)
                bitstream.append(work_line[curr_idx].item())
                last_mode = 1
            else:
                bitstream.extend([0, 0] if zero_count >= 2 else [0] * zero_count)
                bitstream.append(work_line[curr_idx].item())

                distance = curr_idx - last_mode0
                end_k = min(curr_idx, max_c)
                start_k = curr_idx - self.k + 1 if distance >= self.k else curr_idx - distance + 1
                for c in range(start_k, end_k + 1):
                    if 0 <= c <= max_c:
                        c_array.append(c)
                        mode_array.append(0)
                last_mode0 = curr_idx
                last_mode = 0

            work_line[curr_idx] = 0
            last_idx = curr_idx

        if last_mode == 0:
            tail_zeros = width - 1 - last_idx
            if tail_zeros > 0:
                bitstream.extend([0, 0] if tail_zeros >= 2 else [0] * tail_zeros)

        return (
            torch.tensor(bitstream, dtype=torch.int8),
            torch.tensor([r] * len(c_array), dtype=torch.int16),
            torch.tensor(c_array, dtype=torch.int16),
            torch.tensor(mode_array, dtype=torch.int8),
        )

    def init_stream(self, if_map, first_row, row_step, fifo, fifo_depth):
        """Initialize one real-time Split without scanning the feature map."""
        self.if_map = if_map
        self.next_row = int(first_row)
        self.row_step = int(row_step)
        self.fifo = fifo
        self.fifo_depth = max(1, int(fifo_depth))
        self.row_work = None
        self.row_r = None
        self.current_row_data = None
        self.row_done = False
        self.finished = False
        self.row_last_idx = -1
        self.row_last_mode0 = -1
        self.row_last_mode = -1
        self.mode0_known_bits = []
        self.mode0_pending_cols = deque()
        self.conv_mode_counts = collections.Counter({0: 0, 1: 0, 2: 0})

    def _start_next_row(self):
        """Load one assigned row; its contents are scanned later by tick."""
        if self.next_row >= self.if_map.shape[0]:
            self.row_work = None
            self.row_r = None
            self.current_row_data = None
            self.row_done = True
            self.finished = True
            return

        self.row_r = self.next_row
        self.next_row += self.row_step
        self.row_work = self.if_map[self.row_r].view(-1).clone()
        self.current_row_data = self.if_map[self.row_r]
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
        self.current_row_data = None
        self.row_done = True

    def tick_stream(self, allow_next_row=False):
        """Advance this Split by one cycle of real-time row processing."""
        if self.finished:
            return

        # A completed row can be replaced immediately.  Both Split objects
        # share one FIFO, so packages from different rows may be interleaved.
        if self.row_work is None and not self.row_done:
            # The first tick loads the first assigned row. Loading is kept
            # separate from scanning so init_stream never walks the ifmap.
            self._start_next_row()
        elif self.row_done:
            if allow_next_row and not self.mode0_pending_cols:
                self._start_next_row()
            else:
                return
        if self.finished:
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

        curr_idx = nz_mask.nonzero(as_tuple=True)[0][0].item()
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

    def remaining_work(self):
        """Return a lightweight debug estimate of work left in this Split."""
        if self.finished:
            return len(self.fifo)
        pending = len(self.mode0_pending_cols)
        current = int(torch.count_nonzero(self.row_work).item()) if self.row_work is not None else 0
        rows_left = max(0, (self.if_map.shape[0] - self.next_row + self.row_step - 1) // self.row_step)
        return len(self.fifo) + pending + current + rows_left + int(self.row_work is None)


class SplitUnit:
    """Cycle-level streaming split front-end for the cluster simulator.

    Conv mode uses num_split real-time Split objects. Split i handles rows
    i, i + num_split, i + 2 * num_split, ... and all objects share one FIFO.
    Linear mode keeps the previous single-stream behavior.
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
        self.num_split = max(1, int(num_split))
        self.splits = [
            split(kernel_size=kernel_size, enabled_modes=self.enabled_modes)
            for _ in range(self.num_split)
        ]
        # Keep the old names as compatibility aliases for the first two
        # Split objects.
        self.split0 = self.splits[0]
        self.split1 = self.splits[1] if self.num_split > 1 else self.splits[0]
        self.split = self.split0
        self.hazard_num = 0

        self.fifo_depth = max(1, int(fifo_depth))
        # split0 and split1 write to the same bounded FIFO.  The aliases keep
        # older diagnostics that inspect issue_fifo0/issue_fifo1 working.
        self.issue_fifo = deque()
        self.issue_fifo0 = self.issue_fifo
        self.issue_fifo1 = self.issue_fifo
        self.sram_vec_capacity = sram_vec_capacity
        self.fetch_rows = fetch_rows
        self.decode_lanes = max(1, int(fetch_rows if decode_lanes is None else decode_lanes))
        self.rob_depth = max(
            1,
            int((fetch_rows + self.decode_lanes + self.fifo_depth) if rob_depth is None else rob_depth),
        )

        self.if_reg = deque(maxlen=fetch_rows)
        self.decode_slots = []
        self.completed_rows = {}
        self.processing_cycles_left = 0
        self.current_row_data = None
        self.is_finished = True
        # SplitUnit-owned mode histogram for conv. This is intentionally not
        # gathered at PE issue time, because cluster_parallel drops detected
        # mode1/mode2 packets before they reach the FIFO.
        self.conv_mode_counts = collections.Counter({0: 0, 1: 0, 2: 0})

    def init_stream(self, if_map, mode='conv'):
        if not isinstance(if_map, torch.Tensor):
            if_map = torch.tensor(if_map, dtype=torch.int8)

        self.if_map = if_map
        self.mode = mode
        self.current_r = 0
        self.issue_fifo.clear()
        self.if_reg.clear()
        self.decode_slots.clear()
        self.completed_rows.clear()
        self.current_row_data = None
        self.processing_cycles_left = 0
        # Reset per feature map. The simulator later scales this representative
        # histogram by Cout groups when printing mode ratios.
        self.conv_mode_counts = collections.Counter({0: 0, 1: 0, 2: 0})

        self.row_nonzero_bitmask = 0
        self.row_nonzero_flags = []
        self.next_row_idx = 0

        self.linear_entries = []
        self.linear_entry_ptr = 0
        if mode == 'conv':
            # Initialization only installs the feature-map reference. Every
            # Split starts scanning its assigned rows in tick().
            for split_idx, current_split in enumerate(self.splits):
                current_split.enabled_modes = frozenset(self.enabled_modes)
                current_split.init_stream(
                    if_map,
                    first_row=split_idx,
                    row_step=self.num_split,
                    fifo=self.issue_fifo,
                    fifo_depth=self.fifo_depth,
                )
        elif mode == 'linear':
            for current_split in self.splits:
                current_split.finished = True
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
            # Both Split objects scan every cycle and may load their next row
            # independently.  A full shared FIFO applies back pressure to
            # whichever Split is trying to emit at that moment.
            for current_split in self.splits:
                current_split.tick_stream(allow_next_row=True)
            self._sync_mode_counts()
        elif self.mode == 'linear':
            if len(self.issue_fifo) < self.fifo_depth:
                if self.linear_entry_ptr < len(self.linear_entries):
                    self.issue_fifo.append(self.linear_entries[self.linear_entry_ptr])
                    self.linear_entry_ptr += 1
                    self.current_r = min(self.if_map.shape[0], self.linear_entry_ptr)
                else:
                    self.current_r = self.if_map.shape[0]

        self.current_row_data = next(
            (
                current_split.current_row_data
                for current_split in self.splits
                if current_split.current_row_data is not None
            ),
            None,
        )
        self._update_processing_debug_state()
        self._update_finished_state()

    def _sync_mode_counts(self):
        """Aggregate mode counters maintained independently by each Split."""
        self.conv_mode_counts = collections.Counter({0: 0, 1: 0, 2: 0})
        for current_split in self.splits:
            self.conv_mode_counts.update(current_split.conv_mode_counts)

    def _update_processing_debug_state(self):
        if self.mode == 'conv':
            self.processing_cycles_left = (
                sum(current_split.remaining_work() for current_split in self.splits)
                + len(self.issue_fifo)
            )
        elif self.mode == 'linear':
            self.processing_cycles_left = len(self.issue_fifo) + max(
                0, len(self.linear_entries) - self.linear_entry_ptr
            )
        else:
            self.processing_cycles_left = len(self.issue_fifo)

    def _update_finished_state(self):
        if self.mode == 'conv':
            input_done = all(current_split.finished for current_split in self.splits)
        elif self.mode == 'linear':
            input_done = self.linear_entry_ptr >= len(self.linear_entries)
        else:
            input_done = True

        self.is_finished = (
            input_done
            and not self.issue_fifo
        )
