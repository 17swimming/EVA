import torch


class PE:
    def __init__(self, name='W0', w=32):
        self.name = name
        self.w = w
        self.weights = torch.zeros(3, dtype=torch.float32)

    def set_weights(self, weights):
        if len(weights) == 3:
            self.weights = weights.to(torch.float32)

    def process_v2(self, mask, c, target_r, psum_buffer_ref, mode):
        """Accumulate one decoded instruction into the target psum buffer."""
        del target_r

        out_w = psum_buffer_ref.numel()
        if c < 0 or c >= out_w:
            return

        weights = self.weights.to(psum_buffer_ref.dtype)
        mask = mask.to(psum_buffer_ref.dtype)

        if mode == 0:
            padded_mask = torch.zeros(3, dtype=psum_buffer_ref.dtype)
            take = min(3, mask.numel())
            if take > 0:
                padded_mask[:take] = mask[:take]
            psum_buffer_ref[c] += torch.sum(padded_mask * weights)

        elif mode == 1:
            value = mask[0] if mask.numel() > 0 else psum_buffer_ref.new_tensor(0)
            self._accumulate_run(psum_buffer_ref, c, 1, value, weights)

        elif mode == 2:
            if mask.numel() < 3:
                return
            run_len = int(mask[0].item()) * 4 + int(mask[1].item()) * 2 + int(mask[2].item())
            if run_len > 0:
                self._accumulate_run(psum_buffer_ref, c, run_len, psum_buffer_ref.new_tensor(1), weights)

    @staticmethod
    def _accumulate_run(psum_buffer_ref, start_c, run_len, value, weights):
        """Add one run of equal input values to all affected output columns."""
        out_w = psum_buffer_ref.numel()
        out_start = max(0, start_c - 2)
        out_end = min(out_w, start_c + run_len)

        for out_c in range(out_start, out_end):
            acc = psum_buffer_ref.new_tensor(0)
            in_start = max(start_c, out_c)
            in_end = min(start_c + run_len, out_c + 3)
            for in_c in range(in_start, in_end):
                acc = acc + weights[in_c - out_c] * value
            psum_buffer_ref[out_c] += acc
