import collections

import torch

# The six contiguous tap ranges of a 3-tap kernel, in table order.
_LUT_KEYS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
_LUT_INDEX = {key: index for index, key in enumerate(_LUT_KEYS)}


def build_tap_lut(weights):
    """Partial sums of every contiguous tap range: 6 entries, 3 adders.

    Output column ``out_c`` of a run starting at ``c`` with length ``L`` gets
    ``sum(w[lo..hi]) * value`` where ``lo = max(0, c - out_c)`` and
    ``hi = min(2, c + L - 1 - out_c)``.  ``[lo, hi]`` is always a contiguous
    sub-range of ``[0, 2]``, so the whole datapath is one table lookup plus one
    psum add per column.

    PRECONDITION -- activations are 2-bit, so this factoring is NOT generally
    valid.  ``sum(w[lo..hi]) * value`` equals the true ``sum(w[i] * a[c+i])``
    only when every activation the range covers holds the *same* value; pulling
    the multiply out of the sum is a binary-activation trick.  It is sound here
    only because ``shift.py`` builds runs with ``== 1`` (see ``_decode``), so a
    multi-tap range is always all-ones.  mode1 covers one input column, so it
    only ever indexes lo == hi and its 2-bit scalar multiply is exact.

    Six entries and not seven: a run covers contiguous input columns, so the
    taps hitting any one output column are adjacent and every key is a
    contiguous ``[lo, hi]``.  Of the 7 non-empty subsets of {w0,w1,w2} only
    ``w0 + w2`` is non-contiguous, and it cannot arise.

    Reachability splits by mode (pinned by ``test_lut_entries_reachable_per_mode``):
    mode1 (L=1) touches only the single taps (0,0), (1,1), (2,2); mode2 (L>=4)
    touches 5 entries -- everything except (1,1), which would need lo==1 and
    hi==1 at once and so forces L==1.  The union is all 6, so no entry can be
    dropped: (1,1) is dead weight for mode2 but mode1 needs it.

    Taps are summed in ascending order so the entries are bit-identical to the
    ``for in_c in range(in_start, in_end)`` accumulation the PE used to do.
    """
    w0 = weights[0]
    w1 = weights[1]
    w2 = weights[2]
    s01 = w0 + w1
    lut = [None] * len(_LUT_KEYS)
    lut[_LUT_INDEX[(0, 0)]] = w0
    lut[_LUT_INDEX[(0, 1)]] = s01
    lut[_LUT_INDEX[(0, 2)]] = s01 + w2
    lut[_LUT_INDEX[(1, 1)]] = w1
    lut[_LUT_INDEX[(1, 2)]] = w1 + w2
    lut[_LUT_INDEX[(2, 2)]] = w2
    return lut


class ExpressUnit:
    """Sequential Express writer sharing column write ports with the PE.

    PE has priority and consumes one port when it writes. With PE writes present
    Express gets ``COLS_PER_CYCLE - 1`` columns, otherwise all ``COLS_PER_CYCLE``.
    A conflict on any lane stops all lanes at that column. Completed packets never
    authorize retirement: Core's input-row barrier does that.
    """

    # PE 与 Express 共享的 psum 列写端口总数（2026-09-20 由 3 增至 4）。
    # Express 的每拍预算由它推出，不要再在位端口数。
    COLS_PER_CYCLE = 4

    def __init__(self, pool, w, queue_depth=2):
        self.pool = pool
        self.w = int(w)
        self.queue_depth = max(1, int(queue_depth))
        self.queue = collections.deque()
        self.reset_counters()

    def reset_counters(self):
        self.express_unit_busy_cycles = 0
        self.express_write_cols = 0
        self.lane_write_cols = [0, 0, 0]
        self.express_bank_alloc_stalls = 0
        self.express_packets_done = 0
        self.express_pe_merged_writes = 0  # compatibility: always zero
        self.express_pe_conflict_stalls = 0
        self.express_mode_counts = collections.Counter({1: 0, 2: 0})

    def reset(self, pool=None):
        """Drop in-flight state, e.g. when the core rebuilds its psum pool."""
        if pool is not None:
            self.pool = pool
        self.queue.clear()
        if self.pool is not None:
            self.pool.set_express_floors({})

    # ------------------------------------------------------------------ issue

    def can_accept(self):
        return len(self.queue) < self.queue_depth

    def accept(self, mode, c, mask, lanes):
        """Take one mode1/mode2 packet.

        ``lanes`` is a sequence of ``(lane_index, target_r, weights)``, one per
        output row the packet feeds (lane 0 = ``r``, 1 = ``r-1``, 2 = ``r-2``),
        matching the W0/W1/W2 weight split.  Returns False when the packet
        covers no column, in which case there is nothing to write back.
        """
        run_len, value = self._decode(mode, mask)
        c = int(c)
        if run_len <= 0 or not lanes:
            return False

        start = max(0, c - 2)
        end = min(self.w, c + run_len)
        if end <= start:
            return False

        if not self.can_accept():
            return False

        self.queue.append({
            "mode": int(mode),
            "c": c,
            "run_len": int(run_len),
            "value": value,
            "start": start,
            "end": end,
            "cur": start,
            "lanes": [
                (int(lane_index), int(target_r), build_tap_lut(weights.to(torch.float32)))
                for lane_index, target_r, weights in lanes
            ],
        })
        self.express_mode_counts[int(mode)] += 1
        self._publish_floors()
        return True

    @staticmethod
    def _decode(mode, mask):
        """Return ``(run_len, value)``; ``value=None`` means an implicit 1."""
        mode = int(mode)
        if mode == 1:
            if mask.numel() < 1:
                return 0, None
            return 1, mask[0].to(torch.float32)
        if mode == 2:
            if mask.numel() < 3:
                return 0, None
            run_len = (
                int(mask[0].item()) * 4
                + int(mask[1].item()) * 2
                + int(mask[2].item())
            )
            # All 3 mask bits encode L (MSB first), so a mode2 packet carries no
            # activation values at all -- value is implicitly 1 and the LUT alone
            # is the datapath.  That is exact only while ``shift.py`` forms runs
            # with ``row_work[i] == 1``, which makes every run all-ones.  If the
            # encoder is ever widened to run over 2-bit values, this returns the
            # wrong product and no side channel can recover it: mode2 would need
            # a value field, or per-column values, or to stay 1-only by
            # construction.  See ``build_tap_lut``.
            return run_len, None
        return 0, None

    # ---------------------------------------------------------------- writeback

    def tick(self, pe_writes=()):
        """Write back up to ``COLS_PER_CYCLE`` adjacent columns of the head packet."""
        if not self.queue:
            self.pool.set_express_floors({})
            return 0

        self.express_unit_busy_cycles += 1
        pe_targets = {(int(row), int(col)) for row, col in pe_writes}

        packet = self.queue[0]
        touched = collections.OrderedDict()
        written = 0
        # PE 优先：本拍有 PE 写入就占掉一个端口。
        budget = (self.COLS_PER_CYCLE - 1) if pe_targets else self.COLS_PER_CYCLE
        while written < budget and packet["cur"] < packet["end"]:
            cur = int(packet["cur"])
            if any((target_r, cur) in pe_targets for _, target_r, _ in packet["lanes"]):
                self.express_pe_conflict_stalls += 1
                break
            slots = self._resolve_lane_slots(packet, cur)
            if slots is None:
                # Atomic lane allocation failed: hold at ``cur``.
                self.express_bank_alloc_stalls += 1
                break

            index = self._lut_index(packet, cur)
            for lane_index, target_r, lut, slot in slots:
                contribution = lut[index]
                if packet["value"] is not None:
                    contribution = contribution * packet["value"]
                slot["data"][cur] += contribution
                slot["modified"][cur] = True
                self.express_write_cols += 1
                if 0 <= lane_index < len(self.lane_write_cols):
                    self.lane_write_cols[lane_index] += 1
                touched.setdefault(int(slot["id"]), (slot, []))[1].append(cur)

            packet["cur"] = cur + 1
            written += 1

        for slot, cols in touched.values():
            self.pool.note_write(slot, cols, "B")

        if packet["cur"] >= packet["end"]:
            self.queue.popleft()
            self.express_packets_done += 1

        self._publish_floors()
        return written

    def _resolve_lane_slots(self, packet, col):
        points = [(target_r, col) for _, target_r, _ in packet['lanes']]
        if not self.pool.can_allocate_points(points):
            # 归类失败理由（仅统计）：与 PE 侧共用同一套键，便于汇总
            reason = self.pool.classify_alloc_failure(points)
            if reason:
                self.pool.stall_reasons[reason] += 1
            return None
        slots = []
        for lane_index, target_r, lut in packet["lanes"]:
            slot = self.pool.get_or_allocate(target_r, col)
            if slot is None:
                return None
            slots.append((lane_index, target_r, lut, slot))
        return slots

    @staticmethod
    def _lut_index(packet, out_c):
        lo = max(0, packet["c"] - out_c)
        hi = min(2, packet["c"] + packet["run_len"] - 1 - out_c)
        return _LUT_INDEX[(lo, hi)]

    # ---------------------------------------------------------------- queries

    def busy(self):
        return bool(self.queue)

    def inflight_rows(self):
        rows = set()
        for packet in self.queue:
            for _, target_r, _ in packet["lanes"]:
                rows.add(target_r)
        return rows

    def _publish_floors(self):
        """Tell the pool which column range this unit still owes, per row.

        This is a defensive guard for explicit prefix-retire callers. Core
        normally authorizes retirement only after the entire input row drains.
        """
        floors = {}
        for packet in self.queue:
            span = (int(packet["cur"]), int(packet["end"]))
            for _, target_r, _ in packet["lanes"]:
                prev = floors.get(target_r)
                if prev is None:
                    floors[target_r] = span
                else:
                    floors[target_r] = (min(prev[0], span[0]), max(prev[1], span[1]))
        self.pool.set_express_floors(floors)
