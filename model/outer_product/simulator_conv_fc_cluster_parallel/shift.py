import collections
from collections import deque

import torch


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


class SplitUnit:
    """Cycle-level streaming split front-end for the cluster simulator.

    cluster_parallel preprocesses the whole conv feature map in init_stream().
    The split scan still classifies rows as mode0/mode1/mode2 for statistics,
    but only mode0 packets are inserted into the issue FIFO. Linear mode keeps
    the previous behavior: skip all-zero vectors while building issue entries,
    then emit one entry per tick.
    """

    def __init__(
        self,
        kernel_size=3,
        w=32,
        fifo_depth=2,
        sram_vec_capacity=10,
        fetch_rows=4,
        decode_lanes=None,
        rob_depth=None,
        enabled_modes=(0, 1, 2),
    ):
        self.k = kernel_size
        self.w = w
        self.enabled_modes = frozenset(int(mode) for mode in enabled_modes)
        if 0 not in self.enabled_modes or not self.enabled_modes.issubset({0, 1, 2}):
            raise ValueError("enabled_modes must contain mode 0 and only use modes 0, 1, and 2")
        self.split = split(kernel_size=kernel_size, enabled_modes=self.enabled_modes)
        self.hazard_num = 0

        self.row_fifo = deque()
        self.fifo_depth = max(1, int(fifo_depth))
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
        self.pending_packet = None
        self.ready_packets = deque()
        self.row_fifo.clear()
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
        self.row_work = None
        self.row_r = None
        self.row_last_idx = -1
        self.row_last_mode0 = -1
        self.row_last_mode = -1
        self.mode0_known_bits = []
        self.mode0_pending_cols = deque()

        self.linear_entries = []
        self.linear_entry_ptr = 0
        if mode == 'conv':
            self._build_all_conv_mode0_packets(if_map)
        elif mode == 'linear':
            base_r = 2
            for tb in range(if_map.shape[0]):
                tb_vec = if_map[tb].view(-1)
                if not torch.any(tb_vec != 0):
                    continue
                bits = [int(v) for v in tb_vec[:3].tolist()]
                bits.extend([0] * (3 - len(bits)))
                self.linear_entries.append(
                    self._packet(base_r + (tb // self.w) * 3, tb % self.w, 0, bits[:3])
                )
        else:
            raise ValueError(f"Unsupported split mode: {mode}")

        self.is_finished = False
        self._update_finished_state()

    def _build_all_conv_mode0_packets(self, if_map):
        # Conv is fully scanned up front in cluster_parallel. After this pass,
        # tick() only moves prebuilt packets through the FIFO timing model.
        self.current_r = if_map.shape[0]
        self.next_row_idx = if_map.shape[0]
        for row_idx in range(if_map.shape[0]):
            row = if_map[row_idx].view(-1)
            is_nonzero = bool(torch.any(row != 0).item())
            self.row_nonzero_flags.append(is_nonzero)
            if not is_nonzero:
                continue
            self.row_nonzero_bitmask |= 1 << row_idx
            self._encode_conv_row_mode0_only(row, row_idx)

    def _encode_conv_row_mode0_only(self, row, row_idx):
        # The row is scanned with the same mode-detection rules as the normal
        # splitter. The difference is policy: mode1/mode2 detections contribute
        # only to conv_mode_counts, while the issued packet stream remains mode0.
        work_line = row.clone()
        width = work_line.numel()
        max_c = width - self.k
        if max_c < 0:
            return

        known_bits = [None] * width
        pending_cols = deque()
        last_idx = -1
        last_mode0 = -1

        def emit_ready_mode0_cols():
            # A mode0 window can be emitted once all k input bits are known.
            # Windows overlapping dropped mode1/mode2 regions see those bits as
            # zero, so no mode1/mode2 packet enters ready_packets.
            while pending_cols:
                c = pending_cols[0]
                if any(known_bits[c + offset] is None for offset in range(self.k)):
                    return
                pending_cols.popleft()
                bits = [known_bits[c + offset] for offset in range(self.k)]
                self.ready_packets.append(self._packet(row_idx, c, 0, bits))

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
                process_len = min(run_len, 7)
                # Count the detected mode2 run for ratio reporting only.
                # The run itself is zeroed in the mode0 reconstruction below.
                self.conv_mode_counts[2] += 1
                for idx in range(last_idx + 1, curr_idx):
                    known_bits[idx] = 0
                for idx in range(curr_idx, curr_idx + process_len):
                    known_bits[idx] = 0
                emit_ready_mode0_cols()
                work_line[curr_idx : curr_idx + process_len] = 0
                last_idx = curr_idx
                continue

            is_mode1 = (
                zero_count >= 2
                and curr_idx < width - 2
                and work_line[curr_idx + 1] == 0
                and work_line[curr_idx + 2] == 0
            )
            if 1 in self.enabled_modes and is_mode1:
                # Count the isolated point for ratio reporting only. It is
                # masked to zero before any mode0 windows are emitted.
                self.conv_mode_counts[1] += 1
                for idx in range(last_idx + 1, curr_idx):
                    known_bits[idx] = 0
                known_bits[curr_idx] = 0
                emit_ready_mode0_cols()
            else:
                for idx in range(last_idx + 1, curr_idx):
                    known_bits[idx] = 0
                known_bits[curr_idx] = int(work_line[curr_idx].item())

                distance = curr_idx - last_mode0
                end_k = min(curr_idx, max_c)
                if distance >= self.k:
                    start_k = curr_idx - self.k + 1
                else:
                    start_k = curr_idx - distance + 1
                for c in range(start_k, end_k + 1):
                    if 0 <= c <= max_c:
                        pending_cols.append(c)
                        # This is the true number of mode0 split instructions
                        # found by shift.py, not a PE issue-stage counter.
                        self.conv_mode_counts[0] += 1
                emit_ready_mode0_cols()
                last_mode0 = curr_idx

            work_line[curr_idx] = 0
            last_idx = curr_idx

        for idx in range(last_idx + 1, width):
            known_bits[idx] = 0
        emit_ready_mode0_cols()

    def tick(self):
        if self.is_finished:
            return

        if self.pending_packet is not None:
            if len(self.row_fifo) < self.fifo_depth:
                self.row_fifo.append(self.pending_packet)
                self.pending_packet = None
            else:
                self.hazard_num += 1

        # One issue-FIFO write path. Produced packets become visible next tick.
        if self.pending_packet is None and len(self.row_fifo) < self.fifo_depth:
            if self.ready_packets:
                self.pending_packet = self.ready_packets.popleft()
            elif self.mode == 'conv':
                self._tick_conv_encoder()
                if self.ready_packets:
                    self.pending_packet = self.ready_packets.popleft()
            elif self.mode == 'linear':
                if self.linear_entry_ptr < len(self.linear_entries):
                    self.pending_packet = self.linear_entries[self.linear_entry_ptr]
                    self.linear_entry_ptr += 1
                    self.current_r = min(self.if_map.shape[0], self.linear_entry_ptr)
                else:
                    self.current_r = self.if_map.shape[0]

        self._update_processing_debug_state()
        self._update_finished_state()

    def _tick_conv_encoder(self):
        # Fallback incremental encoder kept for compatibility. The current
        # cluster_parallel conv path prebuilds packets in init_stream(), but if
        # this path is used, it follows the same rule: count mode1/mode2 here,
        # issue only reconstructed mode0 packets.
        if self.row_work is None:
            while self.next_row_idx < len(self.row_nonzero_flags):
                row_idx = self.next_row_idx
                self.next_row_idx += 1
                if not self.row_nonzero_flags[row_idx]:
                    continue

                self.row_work = self.if_map[row_idx].view(-1).clone()
                self.row_r = row_idx
                self.row_last_idx = -1
                self.row_last_mode0 = -1
                self.row_last_mode = -1
                self.mode0_known_bits = [None] * self.if_map.shape[1]
                self.mode0_pending_cols.clear()
                self.current_row_data = self.if_map[row_idx]
                self.current_r = row_idx + 1
                break
            if self.row_work is None:
                self.current_r = self.if_map.shape[0]
                return

        nz_mask = self.row_work != 0
        if not nz_mask.any():
            for idx in range(self.row_last_idx + 1, self.row_work.numel()):
                self.mode0_known_bits[idx] = 0
            self._emit_ready_mode0_cols()
            self.row_work = None
            self.row_r = None
            self.current_row_data = None
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
            # Statistics only; no mode2 packet is appended to ready_packets.
            self.conv_mode_counts[2] += 1
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
            # Statistics only; no mode1 packet is appended to ready_packets.
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
            if distance >= self.k:
                start_k = curr_idx - self.k + 1
            else:
                start_k = curr_idx - distance + 1
            for c in range(start_k, end_k + 1):
                if 0 <= c <= max_c:
                    self.mode0_pending_cols.append(c)
                    # Count mode0 at split time, before FIFO/PE issue.
                    self.conv_mode_counts[0] += 1
            self._emit_ready_mode0_cols()
            self.row_last_mode0 = curr_idx
            self.row_last_mode = 0

        self.row_work[curr_idx] = 0
        self.row_last_idx = curr_idx

    def _emit_ready_mode0_cols(self):
        while self.mode0_pending_cols:
            c = self.mode0_pending_cols[0]
            if any(self.mode0_known_bits[c + offset] is None for offset in range(self.k)):
                return
            self.mode0_pending_cols.popleft()
            bits = [self.mode0_known_bits[c + offset] for offset in range(self.k)]
            self.ready_packets.append(self._packet(self.row_r, c, 0, bits))

    @staticmethod
    def _packet(r, c, mode, bits):
        return {
            'bitstream': torch.tensor(bits, dtype=torch.int8),
            'r': torch.tensor([int(r)], dtype=torch.int16),
            'c': torch.tensor([int(c)], dtype=torch.int16),
            'mode': torch.tensor([int(mode)], dtype=torch.int8),
        }

    def _update_processing_debug_state(self):
        queued = len(self.ready_packets) + int(self.pending_packet is not None)
        if self.mode == 'conv':
            row_nz = int(torch.count_nonzero(self.row_work).item()) if self.row_work is not None else 0
            later_rows = sum(1 for idx in range(self.next_row_idx, len(self.row_nonzero_flags)) if self.row_nonzero_flags[idx])
            self.processing_cycles_left = queued + row_nz + later_rows + len(self.mode0_pending_cols)
        elif self.mode == 'linear':
            self.processing_cycles_left = queued + max(0, len(self.linear_entries) - self.linear_entry_ptr)
        else:
            self.processing_cycles_left = queued

    def _update_finished_state(self):
        if self.mode == 'conv':
            input_done = (
                self.row_work is None
                and self.next_row_idx >= len(self.row_nonzero_flags)
                and not self.mode0_pending_cols
            )
        elif self.mode == 'linear':
            input_done = self.linear_entry_ptr >= len(self.linear_entries)
        else:
            input_done = True

        self.is_finished = (
            input_done
            and not self.ready_packets
            and self.pending_packet is None
            and len(self.row_fifo) == 0
        )
