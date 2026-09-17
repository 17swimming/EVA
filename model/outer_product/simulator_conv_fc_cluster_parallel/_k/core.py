import collections

import torch

from model.outer_product.simulator_conv_fc_cluster_parallel._k.PE import PE
from model.outer_product.simulator_conv_fc_cluster_parallel._k.SharedPsumPool import SharedPsumPool
from model.outer_product.simulator_conv_fc_cluster_parallel._k.shift import SplitUnit
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
        num_split=2,
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
            num_split=num_split,
        )

        self.pus = [
            {'W0': PE('W0', psum_w), 'W1': PE('W1', psum_w), 'W2': PE('W2', psum_w)}
            for _ in range(num_pus)
        ]

        self.core_id = -1
        self.pool = None
        # The issue FIFO stores complete packages.  Keep only the package
        # currently being processed so it can be retried after a stall.
        self.current_packet = None
        self.is_finished = True
        self.psum_buffer_alloc_stalls = 0
        self.frontend_stalls = 0
        self.pe_cycles = torch.zeros(3, dtype=torch.int64)
        self.compute_issue_cycles = 0
        # Express issue is an independent ready/valid channel. Until an
        # Express Unit is connected, ready stays high so mode1/2 packages are
        # accepted without consuming the normal mode0 issue slot.
        self.express_ready = True
        self.express_valid = False
        self.express_packet = None
        self.express_fire = False
        self.express_issue_cycles = 0
        # Accumulates SplitUnit-owned mode statistics across all cin feature
        # maps assigned to this core in the current representative run.
        self.conv_mode_split_counts = collections.Counter({0: 0, 1: 0, 2: 0})
        self.conv_mode_counts_pending = False

    def reset_performance_counters(self):
        self.frontend_stalls = 0
        self.psum_buffer_alloc_stalls = 0
        self.pe_cycles = torch.zeros(3, dtype=torch.int64)
        self.compute_issue_cycles = 0
        self.express_valid = False
        self.express_packet = None
        self.express_fire = False
        self.express_issue_cycles = 0
        self.conv_mode_split_counts = collections.Counter({0: 0, 1: 0, 2: 0})
        self.conv_mode_counts_pending = False

    def has_pending_issue(self):
        return (
            getattr(self, 'pending_linear_retire', None) is not None
            or self.current_packet is not None
            or len(self.split_unit.issue_fifo) > 0
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
        # 实时 Split 在后续 tick 中检测模式，必须等当前 Cin tile 完成后再汇总。
        self.conv_mode_counts_pending = True
        self.current_packet = None
        self.pending_linear_retire = None
        self.express_valid = False
        self.express_packet = None
        self.express_fire = False

    def tick_compute(self):
        if self.is_finished:
            return

        # Normal and Express issue are independent. Both may fire in this
        # cycle from the same four-entry Split issue window.
        self._issue_express_packet()

        if not self._ensure_current_packet():
            return

        packet = self.current_packet
        r = int(packet['r'].item())
        c = int(packet['c'].item())
        mode = int(packet['mode'].item())
        mask = self._read_packet_mask(packet, mode)

        active_rs = self._active_rows_for_conv(r)
        self.pool.retire_old(active_rs)

        buffers_to_use = {}
        for tr in active_rs:
            buf = self.pool.get_or_allocate(tr)
            if buf is None:
                self._record_psum_alloc_stall()
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
                self._request_pe2_stream_retire(buffers_to_use[r - 2], c, mode, mask)
                self.pe_cycles[2] += 1

        # 按 Core 计数：多个 PE 同拍执行只算一次，Psum 分配失败不计入。
        if buffers_to_use:
            self.compute_issue_cycles += 1
        self.current_packet = None

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
        self.current_packet = None
        self.pending_linear_retire = None
        self.conv_mode_counts_pending = False
        self.express_valid = False
        self.express_packet = None
        self.express_fire = False

        self.H = 999999

    def tick_compute_linear(self):
        if self.is_finished:
            return

        if self.pending_linear_retire is not None:
            if self._try_commit_linear_retire():
                self.current_packet = None
            return

        if not self._ensure_current_packet():
            return

        packet = self.current_packet
        r = int(packet['r'].item())
        c = int(packet['c'].item())
        mode = int(packet['mode'].item())
        mask = self._read_packet_mask(packet, mode)

        # 不同pe对应不同cout，每个cout对应一行psum————这里我把psum看成3x12
        target_r0 = r
        target_r1 = r - 1
        target_r2 = r - 2
        active_rs = [target_r0, target_r1, target_r2]

        self.pool.retire_old(active_rs)

        buffers_to_use = {}
        for tr in active_rs:
            buf = self.pool.get_or_allocate(tr)
            if buf is None:
                self._record_psum_alloc_stall()
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
            self.compute_issue_cycles += 1
            self.pending_linear_retire = {
                'col': c,
                'row_buffers': [
                    (target_r2, buffers_to_use[target_r2]),
                    (target_r1, buffers_to_use[target_r1]),
                    (target_r0, buffers_to_use[target_r0]),
                ],
            }
            if not self._try_commit_linear_retire():
                return

        self.current_packet = None

    def _ensure_current_packet(self):
        """Keep a stalled package or select the oldest mode0 package.

        The first branch and the issue-window branch are mutually exclusive. A
        package that has already been fetched remains in ``current_packet``
        while the backend waits for a psum buffer or retire capacity.  Only
        after that package completes do we select another mode0 package.
        """
        if self.current_packet is not None:
            # Retry the same package after a resource stall.  It must not be
            # removed from the FIFO again or decoded as a second package.
            return True

        packet = self.split_unit.pop_mode0_packet()
        if packet is not None:
            # Mode1/2 packages remain available to the independent Express
            # issue channel and never occupy the normal PE issue slot.
            self.current_packet = packet
            return True

        if self.split_unit.is_finished:
            # No packet remains in the FIFO and SplitUnit is done producing
            # input.  The Core can finish only after all pending psum rows
            # have been handed to the accumulator.
            self.pool.retire_old([])
            if not self.pool.has_pending():
                if self.conv_mode_counts_pending:
                    self.conv_mode_split_counts.update(self.split_unit.conv_mode_counts)
                    self.conv_mode_counts_pending = False
                self.is_finished = True
        else:
            # SplitUnit still has input to produce, but no packet is
            # available this cycle.  The compute stage is therefore stalled
            # by an empty issue FIFO.
            self.frontend_stalls += 1
        return False

    def _issue_express_packet(self):
        """Drive one mode1/2 package on the independent Express interface."""
        self.express_packet = self.split_unit.peek_express_packet()
        self.express_valid = self.express_packet is not None
        self.express_fire = self.express_valid and bool(self.express_ready)
        if not self.express_fire:
            return

        # Remove the exact oldest Express candidate observed above. No Psum
        # work is modeled here; the future Express Unit owns that behavior.
        self.express_packet = self.split_unit.pop_express_packet()
        self.express_issue_cycles += 1

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

    def _record_psum_alloc_stall(self):
        self.psum_buffer_alloc_stalls += 1

    @staticmethod
    def _read_packet_mask(packet, mode):
        """Return one package mask padded to the PE's three-bit interface."""
        bitstream = packet.get('bitstream')
        if bitstream is None:
            return torch.zeros(3, dtype=torch.int8)

        mask = torch.zeros(3, dtype=bitstream.dtype)
        width = 1 if mode == 1 else 3
        available = min(width, int(bitstream.numel()))
        if available > 0:
            mask[:available] = bitstream[:available]
        return mask

    def _try_commit_linear_retire(self):
        pending = self.pending_linear_retire
        if pending is None:
            return True

        success = self.pool.try_retire_fc_bundle(
            pending['row_buffers'],
            pending['col'],
            on_result=self._finish_linear_retire,
        )
        return success

    def _finish_linear_retire(self, success):
        if success:
            self.pending_linear_retire = None
            self.current_packet = None

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

    def _request_pe2_stream_retire(self, buf, c, mode, mask):
        end_idx = self._retire_end_exclusive(c, mode, mask)
        if end_idx is None:
            return
        if end_idx <= int(buf.get('retire_limit', 0)):
            return

        self.pool.request_stream_retire(buf, end_idx)
