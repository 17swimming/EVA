import collections

import torch

from model.outer_product.simulator_conv_fc_cluster_parallel.PE import PE
from model.outer_product.simulator_conv_fc_cluster_parallel.SharedPsumPool import SharedPsumPool
from model.outer_product.simulator_conv_fc_cluster_parallel.shift import SplitUnit
from model.utils import ceil_a_by_b


class Core:
    def __init__(
        self,
        kernel_size=3,
        num_pus=16,
        psum_w=32,
        bank_h=32,
        split_fifo_depth=16,
        fetch_rows=4,
        retire_column=3,
        psum_pool_rows=6,
        enabled_modes=(0, 1, 2),
    ):
        self.kernel_size = kernel_size
        self.num_pus = num_pus
        self.psum_w = psum_w
        self.bank_h = bank_h
        self.retire_column = retire_column
        self.psum_pool_rows = int(psum_pool_rows)
        self.enabled_modes = tuple(sorted({int(mode) for mode in enabled_modes}))
        if 0 not in self.enabled_modes or any(mode not in (0, 1, 2) for mode in self.enabled_modes):
            raise ValueError("enabled_modes must contain mode 0 and only use modes 0, 1, and 2")
        self.split_unit = SplitUnit(
            kernel_size=kernel_size,
            w=psum_w,
            fifo_depth=split_fifo_depth,
            fetch_rows=fetch_rows,
            enabled_modes=self.enabled_modes,
        )

        self.pus = [
            {'W0': PE('W0', psum_w), 'W1': PE('W1', psum_w), 'W2': PE('W2', psum_w)}
            for _ in range(num_pus)
        ]

        self.core_id = -1
        self.pool = None
        self.is_finished = True
        self.total_tail_draining_cycles = 0
        self.reset_performance_counters()

    def reset_performance_counters(self):
        self.psum_buffer_alloc_stalls = 0
        self.retire_backpressure_stalls = 0
        self.stalls = 0
        self.frontend_stalls = 0
        self.pe_cycles = torch.zeros(3, dtype=torch.int64)
        self.issued_packets = 0
        # Accumulates SplitUnit-owned mode statistics across all cin feature
        # maps assigned to this core in the current representative run.
        self.conv_mode_split_counts = collections.Counter({0: 0, 1: 0, 2: 0})
        self.retire_trace_cycle = 0
        self.psum_alloc_stall_reasons = collections.Counter()
        self.psum_alloc_stall_occupancy_totals = collections.Counter()
        self.psum_alloc_stall_active_rows_hist = collections.Counter()
        self.retire_column_cycle_stats = collections.defaultdict(
            lambda: {
                'span_cols': 0,
                'marked_cols': 0,
                'events': 0,
            }
        )
        self.retire_column_total_stats = {
            'span_cols': 0,
            'marked_cols': 0,
            'events': 0,
        }

    def has_pending_issue(self):
        return (
            getattr(self, 'pending_linear_retire', None) is not None
            or self.current_row_insts is not None
            or len(self.split_unit.row_fifo) > 0
        )

    def configure_conv_weights(self, cin, kernel):
        k_h = kernel.shape[2]

        for i in range(min(self.num_pus, kernel.shape[0])):
            pu_kernel = kernel[i, cin, :, :]
            flat_kernel = pu_kernel.reshape(-1).to(torch.float32)

            if k_h == 3:
                weights_w0 = flat_kernel[:3]
                weights_w1 = flat_kernel[3:6]
                weights_w2 = flat_kernel[6:9]
            elif k_h == 1:
                weights_w0 = flat_kernel[:3]
                weights_w1 = torch.zeros(3, dtype=torch.float32)
                weights_w2 = torch.zeros(3, dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported kernel height: {k_h}")

            self.pus[i]['W0'].set_weights(weights_w0)
            self.pus[i]['W1'].set_weights(weights_w1)
            self.pus[i]['W2'].set_weights(weights_w2)

    def configure_fc_weights(self, weights):
        pu_count = min(self.num_pus, ceil_a_by_b(weights.shape[1], 3))

        for i in range(pu_count):
            pu_kernel = weights[:, i * 3 : (i + 1) * 3].to(torch.float32)
            if pu_kernel.shape[1] < 3:
                pad_cols = 3 - pu_kernel.shape[1]
                pu_kernel = torch.nn.functional.pad(pu_kernel, (0, pad_cols))

            self.pus[i]['W0'].set_weights(pu_kernel[:, 2])
            self.pus[i]['W1'].set_weights(pu_kernel[:, 1])
            self.pus[i]['W2'].set_weights(pu_kernel[:, 0])

    def init_for_new_tile(self, accumulator, core_id, if_map, H):
        self.core_id = core_id
        self.pool = SharedPsumPool(
            core_id=core_id,
            accumulator=accumulator,
            w=self.psum_w,
            num_buffers=self.psum_pool_rows,
            retire_column=self.retire_column,
        )
        self.H = H
        self.is_finished = False

        self.split_unit.init_stream(if_map)
        self.conv_mode_split_counts.update(self.split_unit.conv_mode_counts)
        for mode in (0, 1, 2):
            self.conv_mode_split_counts.setdefault(mode, 0)
        self.current_row_insts = None
        self.pending_linear_retire = None
        self.inst_ptr = 0
        self.bitstream_ptr = 0

    def tick_compute(self):
        if self.is_finished:
            return

        if not self._ensure_current_insts():
            return

        r = int(self.current_row_insts['r'][self.inst_ptr])
        c = int(self.current_row_insts['c'][self.inst_ptr])
        mode = int(self.current_row_insts['mode'][self.inst_ptr])
        mask = self._read_mode_mask(mode)

        active_rs = self._active_rows_for_conv(r)
        self.pool.retire_old(active_rs)

        buffers_to_use = {}
        for tr in active_rs:
            buf = self.pool.get_or_allocate(tr)
            if buf is None:
                self._record_psum_alloc_stall(active_rs)
                return
            buffers_to_use[tr] = buf

        pu = self.pus[0]
        if r in buffers_to_use:
            pu['W0'].process_v2(mask, c, r, buffers_to_use[r]['data'], mode)
            self._mark_modified_columns(buffers_to_use[r], c, mode, mask)
            self.pe_cycles[0] += 1

        if getattr(self, 'kernel_h', 3) == 3:
            if r - 1 in buffers_to_use:
                pu['W1'].process_v2(mask, c, r - 1, buffers_to_use[r - 1]['data'], mode)
                self._mark_modified_columns(buffers_to_use[r - 1], c, mode, mask)
                self.pe_cycles[1] += 1
            if r - 2 in buffers_to_use:
                pu['W2'].process_v2(mask, c, r - 2, buffers_to_use[r - 2]['data'], mode)
                self._mark_modified_columns(buffers_to_use[r - 2], c, mode, mask)
                self._record_pe2_retire_columns(buffers_to_use[r - 2], c, mode, mask)
                self.pe_cycles[2] += 1

        self._advance_conv_issue_ptr(mode)

    def init_for_linear_tile(self, accumulator, core_id, if_map, valid_pu):
        self.core_id = core_id
        self.valid_pu = valid_pu
        self.is_finished = False

        self.pool = SharedPsumPool(
            core_id=core_id,
            accumulator=accumulator,
            w=self.psum_w,
            num_buffers=self.psum_pool_rows,
            retire_column=self.retire_column,
        )
        for buf in self.pool.buffers:
            buf['data'] = torch.zeros(self.psum_w, dtype=torch.float32)

        self.split_unit.init_stream(if_map, mode='linear')
        self.current_row_insts = None
        self.pending_linear_retire = None
        self.inst_ptr = 0
        self.bitstream_ptr = 0

        self.H = 999999

    def tick_compute_linear(self):
        if self.is_finished:
            return

        if self.pending_linear_retire is not None:
            if self._try_commit_linear_retire():
                self._advance_linear_issue_ptr()
            else:
                self._record_retire_backpressure_stall()
            return

        if not self._ensure_current_insts():
            return

        r = int(self.current_row_insts['r'][self.inst_ptr])
        c = int(self.current_row_insts['c'][self.inst_ptr])
        mode = int(self.current_row_insts['mode'][self.inst_ptr])
        mask = self._read_bitstream(3)

        target_r0 = r
        target_r1 = r - 1
        target_r2 = r - 2
        active_rs = [target_r0, target_r1, target_r2]

        self.pool.retire_old(active_rs)

        buffers_to_use = {}
        for tr in active_rs:
            buf = self.pool.get_or_allocate(tr)
            if buf is None:
                self._record_psum_alloc_stall(active_rs)
                return
            buffers_to_use[tr] = buf

        if self.valid_pu > 0:
            pu = self.pus[0]
            pu['W0'].process_v2(mask, c, target_r0, buffers_to_use[target_r0]['data'], mode)
            pu['W1'].process_v2(mask, c, target_r1, buffers_to_use[target_r1]['data'], mode)
            pu['W2'].process_v2(mask, c, target_r2, buffers_to_use[target_r2]['data'], mode)
            self._mark_modified_columns(buffers_to_use[target_r0], c, mode, mask)
            self._mark_modified_columns(buffers_to_use[target_r1], c, mode, mask)
            self._mark_modified_columns(buffers_to_use[target_r2], c, mode, mask)
            self.pe_cycles += 1
            self.pending_linear_retire = {
                'col': c,
                'row_buffers': [
                    (target_r2, buffers_to_use[target_r2]),
                    (target_r1, buffers_to_use[target_r1]),
                    (target_r0, buffers_to_use[target_r0]),
                ],
            }
            if not self._try_commit_linear_retire():
                self._record_retire_backpressure_stall()
                return

        self._advance_linear_issue_ptr()

    def _ensure_current_insts(self):
        if self.current_row_insts is not None:
            return True

        if len(self.split_unit.row_fifo) > 0:
            self.current_row_insts = self.split_unit.row_fifo.popleft()
            self.inst_ptr = 0
            self.bitstream_ptr = 0
            self.issued_packets += 1
            return True

        if self.split_unit.is_finished:
            self.pool.retire_old([])
            if not self.pool.has_pending():
                self.is_finished = True
        else:
            self.frontend_stalls += 1
        return False

    def _active_rows_for_conv(self, r):
        active_rs = []
        if r < self.H:
            active_rs.append(r)

        if getattr(self, 'kernel_h', 3) == 3:
            if 0 <= r - 1 < self.H:
                active_rs.append(r - 1)
            if 0 <= r - 2 < self.H:
                active_rs.append(r - 2)

        return active_rs

    def _record_psum_alloc_stall(self, active_rs=None):
        self.psum_buffer_alloc_stalls += 1
        self.stalls += 1
        if active_rs is not None:
            self.psum_alloc_stall_active_rows_hist[len(active_rs)] += 1
        if self.pool is None:
            self.psum_alloc_stall_reasons["no_pool"] += 1
            return

        summary = self.pool.occupancy_summary()
        for key, value in summary.items():
            self.psum_alloc_stall_occupancy_totals[key] += int(value)

        if (
            summary.get("pending_fc_bundles", 0)
            or summary.get("active_fc_pending", 0)
            or summary.get("pending_wb_fc_pending", 0)
        ):
            reason = "fc_pending_retire"
        elif summary.get("pending_wb", 0):
            reason = "row_pending_wb"
        elif summary.get("active_stream_pending", 0):
            reason = "stream_pending_retire"
        elif summary.get("modified_buffers", 0):
            reason = "modified_active_buffers"
        elif summary.get("active", 0):
            reason = "all_buffers_active"
        else:
            reason = "unknown"
        self.psum_alloc_stall_reasons[reason] += 1

    def _record_retire_backpressure_stall(self):
        self.retire_backpressure_stalls += 1

    def _read_mode_mask(self, mode):
        if mode == 0:
            return self._read_bitstream(3)
        if mode == 1:
            value = self._read_bitstream(1)
            return torch.tensor([value[0].item(), 0, 0], dtype=value.dtype)
        if mode == 2:
            return self._read_bitstream(3)
        return torch.zeros(3, dtype=torch.int8)

    def _read_bitstream(self, width):
        bitstream = self.current_row_insts['bitstream']
        mask = torch.zeros(width, dtype=bitstream.dtype)
        available = max(0, min(width, len(bitstream) - self.bitstream_ptr))
        if available > 0:
            mask[:available] = bitstream[self.bitstream_ptr : self.bitstream_ptr + available]
        return mask

    def _advance_conv_issue_ptr(self, mode):
        if mode == 0:
            if self._next_inst_is_mode0():
                self.bitstream_ptr += 1
            else:
                self.bitstream_ptr += 3
        elif mode == 1:
            self.bitstream_ptr += 1
        elif mode == 2:
            self.bitstream_ptr += 3

        self.inst_ptr += 1
        self._drop_finished_insts()

    def _advance_linear_issue_ptr(self):
        self.inst_ptr += 1
        self.bitstream_ptr += 3
        self._drop_finished_insts()

    def _try_commit_linear_retire(self):
        pending = self.pending_linear_retire
        if pending is None:
            return True

        success = self.pool.try_retire_fc_bundle(
            pending['row_buffers'],
            pending['col'],
        )
        if success:
            self.pending_linear_retire = None
        return success

    def _next_inst_is_mode0(self):
        next_ptr = self.inst_ptr + 1
        return (
            next_ptr < len(self.current_row_insts['mode'])
            and int(self.current_row_insts['mode'][next_ptr]) == 0
        )

    def _drop_finished_insts(self):
        if self.inst_ptr >= len(self.current_row_insts['r']):
            self.current_row_insts = None

    def _decode_run_len(self, mode, mask):
        if mode == 2 and mask.numel() >= 3:
            return int(mask[0].item()) * 4 + int(mask[1].item()) * 2 + int(mask[2].item())
        if mode == 1:
            return 1
        return 0

    def _modified_col_range(self, c, mode, mask):
        if mode == 0:
            start = c
            end = c + 1
        elif mode == 1:
            start = c - 2
            end = c + 1
        elif mode == 2:
            run_len = self._decode_run_len(mode, mask)
            if run_len <= 0:
                return None
            start = c - 2
            end = c + run_len
        else:
            return None

        start = max(0, min(self.psum_w, int(start)))
        end = max(0, min(self.psum_w, int(end)))
        if end <= start:
            return None
        return start, end

    def _mark_modified_columns(self, buf, c, mode, mask):
        col_range = self._modified_col_range(c, mode, mask)
        if col_range is None:
            return
        start, end = col_range
        buf['modified'][start:end] = True

    def _retire_end_exclusive(self, c, mode, mask):
        if self.psum_w <= 0:
            return None

        if mode in (0, 1):
            end = c + 1
        elif mode == 2:
            run_len = self._decode_run_len(mode, mask)
            if run_len <= 0:
                return None
            end = c + run_len
        else:
            return None

        return max(0, min(self.psum_w, int(end)))

    def _record_pe2_retire_columns(self, buf, c, mode, mask):
        end_idx = self._retire_end_exclusive(c, mode, mask)
        if end_idx is None:
            return

        start_idx = max(0, min(self.psum_w, int(buf.get('retire_limit', 0))))
        if end_idx <= start_idx:
            return

        span_cols = end_idx - start_idx
        marked_cols = int(buf['modified'][start_idx:end_idx].sum().item())
        cycle = int(getattr(self, 'retire_trace_cycle', 0))
        stats = self.retire_column_cycle_stats[cycle]
        stats['span_cols'] += span_cols
        stats['marked_cols'] += marked_cols
        stats['events'] += 1

        self.retire_column_total_stats['span_cols'] += span_cols
        self.retire_column_total_stats['marked_cols'] += marked_cols
        self.retire_column_total_stats['events'] += 1

        self.pool.request_stream_retire(buf, end_idx)
