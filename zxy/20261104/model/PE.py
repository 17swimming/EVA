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
        """Accumulate one mode0 instruction into the target psum buffer.

        mode0 is one packet, one output column, one psum read-modify-write, so
        this is the whole PE datapath: 3 multipliers plus an adder tree.
        mode1/mode2 cover a *range* of columns and are handled by ExpressUnit
        through the other psum write port (see express_unit.py).
        """
        del target_r

        if mode != 0:
            return

        out_w = psum_buffer_ref.numel()
        if c < 0 or c >= out_w:
            return

        weights = self.weights.to(psum_buffer_ref.dtype)
        mask = mask.to(psum_buffer_ref.dtype)

        padded_mask = torch.zeros(3, dtype=psum_buffer_ref.dtype)
        take = min(3, mask.numel())
        if take > 0:
            padded_mask[:take] = mask[:take]
        psum_buffer_ref[c] += torch.sum(padded_mask * weights)
