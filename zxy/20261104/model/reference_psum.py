import collections
import os

import torch


class SharedPsumPool:
    """Psum storage shared by PE and Express. Two allocation granularities.

    ``alloc_mode="point"`` (default, historical): ``num_buffers`` retains the old
    half-row *capacity* unit -- six at w=16 gives 48 independently allocated
    points. Each slot owns exactly one column.

    ``alloc_mode="halfrow"``: one slot owns a whole contiguous half of one output
    row. The two halves cover ``[0, ceil(w/2))`` and ``[ceil(w/2), w)``, so
    ``num_buffers`` slots -- six, i.e. three whole rows -- are the entire pool.
    Touching any column of a half occupies the whole half, including columns that
    are never written. A half is released only once every column it actually got
    written for has been accepted by the accumulator; partial retirement keeps
    the half resident.

    Retire granularity is the same in both modes: one bundle per output row per
    cycle, up to ``retire_column`` points. Under half-row allocation a bundle's
    columns may come from both halves of the row. ``data`` stays a full-width
    tensor in both modes -- it is only an indexing adapter for the PE datapath,
    so column addressing is unchanged. CONV retires one row with several columns.
    FC retires one column across at most three rows from the same PE row group,
    matching _k's column bundle and independent point acknowledgements.
    between retries and fresh submissions. Storage port implementation/area is
    not inferred from these functional bandwidth limits.
    """

    def __init__(self, core_id, accumulator, w=32, num_buffers=8, retire_column=3,
                 alloc_mode="halfrow", retire_packets_per_cycle=2, fc_mode=False,
                 retire_iface="scatter"):
        self.core_id = int(core_id)
        self.accumulator = accumulator
        self.w = int(w)
        self.num_buffers = int(num_buffers)
        self.retire_column = int(retire_column)
        # RTL psum_retire.v：一个 bundle 只装**一个半行槽**（同一行的两半留到下一拍）。
        self.half_bundle = os.environ.get("EVA_HALF_BUNDLE", "0") == "1"
        self.alloc_mode = str(alloc_mode)
        if self.alloc_mode not in ("point", "halfrow"):
            raise ValueError(f"unknown psum allocation mode: {alloc_mode!r}")
        # Retire read-port count: how many bundles this core may hand the
        # accumulator per cycle. None = unlimited (historical). One bundle per
        # output row for CONV, or one column/three-row group for FC.
        # 0 或负数 = 不限（旧行为），便于与历史基线对照；正数 = 每拍读端口数。
        _packets = (None if retire_packets_per_cycle is None
                    else int(retire_packets_per_cycle))
        self.retire_packets_per_cycle = (
            None if _packets is None or _packets <= 0 else _packets)
        # FC shares allocation with CONV, but retires three rows at one column.
        self.fc_mode = bool(fc_mode)
        # Retire interface shape: "scatter" sends the selected columns as a
        # sparse point list; "positional" sends the whole row plus a per-column
        # write enable, which is what lets the accumulator drop its per-point
        # write crossbar. Timing is identical by construction -- see
        # Accumulator.request_row_bundle.
        self.retire_iface = str(retire_iface)
        if self.retire_iface not in ("scatter", "positional"):
            raise ValueError(f"unknown retire interface: {retire_iface!r}")
        if min(self.w, self.num_buffers, self.retire_column) <= 0:
            raise ValueError("psum width, capacity and retire bandwidth must be positive")
        self.bank_w = (self.w + 1) // 2  # legacy capacity unit, not allocation
        # Fixed split: half 0 = [0, ceil(w/2)), half 1 = [ceil(w/2), w). The two
        # halves differ in length by one when w is odd.
        self.half_w = self.bank_w
        self.half_len = (self.half_w, self.w - self.half_w)
        # Half-row allocation halves the slot count: 6 whole halves, not 36 points.
        self.num_units = (self.num_buffers if self.alloc_mode == "halfrow"
                          else self.num_buffers * self.bank_w)
        # Capacity keeps its point unit in both modes, so Core.init_for_new_tile's
        # feasibility check and the published per-core capacity stay comparable.
        self.point_capacity = self.num_buffers * self.bank_w
        # Keep existing retired-point counters for Core's cross-tile accounting.
        self.stats = {
            "stream_retire_cols": 0,
            "row_retire_cols": 0,
            "retire_blocked": 0,
            "conv_stream_requests": 0,
            "conv_stream_accepts": 0,
            "conv_stream_rejects": 0,
            "conv_row_requests": 0,
            "conv_row_accepts": 0,
            "conv_row_rejects": 0,
            "bank_early_releases": 0,
            "bank_allocations": 0,
            "max_slots_in_use": 0,
        }
        self.request_col_hist = collections.Counter()
        self.request_row_hist = collections.Counter()
        self.blocked_reason_counts = collections.Counter()
        self.current_cycle = 0

        self.stats.update(point_allocations=0, point_releases=0)
        self.buffers = [self._new_buffer(i) for i in range(self.num_units)]
        self.row_retire_limits = {}
        self.row_express_floors = {}
        self.write_trace_hook = None
        self._retry_after = 0
        self._packets_this_cycle = 0

        # ---------------------------------------------------------------
        # 停顿理由统计。本 pool 归属于唯一一个 Core，因此这些计数天然是
        # per-core 的；模拟器层再按 core 汇总。
        # 注意池会在每个 tile 重建（Core._new_pool），Core 换池时会把旧池的
        # 计数折进自己的累计器，避免跨 tile 丢失。
        # ---------------------------------------------------------------
        self.stall_reasons = collections.Counter()
        self.retire_submitted_points = 0   # 提交给 accumulator 的 retire 点数（含被拒）
        self.retire_rejected_points = 0    # 其中被仲裁拒绝的点数
        self.retire_busy_cycles = 0        # 历史字段名；实际逐提交包计数，双包同拍加 2
        self.backoff_wait_cycles = 0       # 因 _retry_after 错开而未提交的拍数

    def classify_alloc_failure(self, points):
        """返回 can_allocate_points 失败的首要理由键，仅用于统计，不改行为。

        与 can_allocate_points 的判定保持一致：
          - 某点已有驻留 buffer 但状态不是 ACTIVE → 'alloc_point_in_flight'
            （该点正在等写回或等 FC 确认）
          - 需要新分配的点数超过剩余槽位 → 'alloc_no_free_slot'
        返回 None 表示本次调用其实能成功（调用方不应记录停顿）。
        """
        unique = set((int(r), int(c)) for r, c in points)
        if self.alloc_mode == 'halfrow':
            needed = set()
            for unit in self._units_needed(unique):
                resident = self._resident_unit(*unit)
                if resident is None:
                    needed.add(unit)
                elif resident['state'] != 'ACTIVE':
                    return 'alloc_point_in_flight'
            return 'alloc_no_free_slot' if len(needed) > self.free_slot_count() else None
        missing = 0
        for row, col in unique:
            resident = self._resident_point(row, col)
            if resident is None:
                missing += 1
            elif resident['state'] != 'ACTIVE':
                return 'alloc_point_in_flight'
        return 'alloc_no_free_slot' if missing > self.free_slot_count() else None

    def set_cycle(self, cycle):
        cycle = int(cycle)
        if cycle != self.current_cycle:
            self._packets_this_cycle = 0
        self.current_cycle = cycle

    def _packet_budget_exhausted(self):
        """True once this core has spent its retire read ports for the cycle."""
        return (self.retire_packets_per_cycle is not None
                and self._packets_this_cycle >= self.retire_packets_per_cycle)

    def _new_buffer(self, index):
        return dict(id=index, state="FREE", target_r=-1, col=-1, final_col=-1,
                    col_base=0, col_end=0, data=torch.zeros(self.w),
                    modified=torch.zeros(self.w, dtype=torch.bool),
                    retired=torch.zeros(self.w, dtype=torch.bool),
                    retire_inflight=False,
                    # 逐列在途掩码。槽级 `retire_inflight` 会让一个槽在回调
                    # 回来之前整槽不能出包；加了 retire 流水之后回调要等到下
                    # 一拍，单槽吞吐会直接减半——池只有 8 个 unit，这个约束会
                    # 先于「2 包/拍」读口预算触顶。逐列记就只挡住在途的那几列。
                    inflight=torch.zeros(self.w, dtype=torch.bool))

    def _clear_buffer(self, buf):
        buf['data'].zero_()
        buf['modified'].zero_()
        buf['retired'].zero_()
        buf['inflight'].zero_()
        buf.update(state='FREE', target_r=-1, col=-1, final_col=-1,
                   col_base=0, col_end=0, retire_inflight=False)
        self.stats['point_releases'] += 1

    def _set_inflight(self, buf, col, value):
        """Mark one column in flight; the slot-level flag is always their OR.

        The slot-level flag used to lock a whole slot for a callback round-trip.
        Once retire is pipelined that round-trip is two cycles, so a slot could
        present a packet only every other cycle -- and with eight units in the
        pool that ceiling binds well before the two-packets-per-cycle read-port
        budget does. Per-column marking holds back only the columns in flight.
        """
        buf['inflight'][int(col)] = bool(value)
        buf['retire_inflight'] = bool(buf['inflight'].any())

    # ------------------------------------------------------- half-row helpers

    def _half_of(self, col):
        """Which positional half a column falls in. Fixed split, no wrap."""
        return 0 if int(col) < self.half_w else 1

    def _half_range(self, half):
        """[lo, hi) column span of a half. Lengths differ by one for odd w."""
        lo = 0 if int(half) == 0 else self.half_w
        return lo, lo + self.half_len[int(half)]

    def _units_needed(self, points):
        """Collapse a point set to the distinct (row, half) units it touches."""
        return {(int(r), self._half_of(c)) for r, c in points}

    def _resident_unit(self, row, half):
        """The non-FREE slot owning (row, half), or None."""
        lo, _ = self._half_range(half)
        row = int(row)
        return next((b for b in self.buffers if b['state'] != 'FREE'
                     and b['target_r'] == row and b['col_base'] == lo), None)

    def _resident_point(self, row, col):
        """The non-FREE slot owning exactly (row, col), or None."""
        col = int(col)
        row = int(row)
        return next((b for b in self.buffers if b['state'] != 'FREE'
                     and b['target_r'] == row
                     and b['col_base'] <= col < b['col_end']), None)

    def _unit_drained(self, buf):
        """True once every column this half was written for has been accepted."""
        return not bool((buf['modified'] & ~buf['retired']).any())

    def resolve(self, target_r, col):
        if self.alloc_mode == 'halfrow':
            buf = self._resident_unit(target_r, self._half_of(col))
            return buf if buf is not None and buf['state'] == 'ACTIVE' else None
        return next((b for b in self.buffers if b['state'] == 'ACTIVE'
                     and b['target_r'] == int(target_r) and b['col'] == int(col)), None)

    def free_slot_count(self):
        return sum(b['state'] == 'FREE' for b in self.buffers)

    def can_allocate_points(self, points):
        unique = set((int(r), int(c)) for r, c in points)
        for _, col in unique:
            if not 0 <= col < self.w:
                raise ValueError(f"psum column outside tile: {col}")
        if self.alloc_mode == 'halfrow':
            needed = set()
            for unit in self._units_needed(unique):
                resident = self._resident_unit(*unit)
                if resident is None:
                    needed.add(unit)
                elif resident['state'] != 'ACTIVE':
                    return False
            return len(needed) <= self.free_slot_count()
        missing = 0
        for row, col in unique:
            resident = self._resident_point(row, col)
            if resident is None:
                missing += 1
            elif resident['state'] != 'ACTIVE':
                return False
        return missing <= self.free_slot_count()

    def get_or_allocate(self, target_r, col):
        target_r, col = int(target_r), int(col)
        if not self.can_allocate_points([(target_r, col)]):
            return None
        buf = self.resolve(target_r, col)
        if buf is not None:
            return buf
        buf = next(b for b in self.buffers if b['state'] == 'FREE')
        if self.alloc_mode == 'halfrow':
            # Touching any column of a half occupies the whole half.
            lo, hi = self._half_range(self._half_of(col))
            buf.update(state='ACTIVE', target_r=target_r, col=lo,
                       col_base=lo, col_end=hi)
        else:
            buf.update(state='ACTIVE', target_r=target_r, col=col,
                       col_base=col, col_end=col + 1)
        self.stats['point_allocations'] += 1
        self.stats['max_slots_in_use'] = max(
            self.stats['max_slots_in_use'], self.num_units - self.free_slot_count())
        return buf

    def retire_old(self, active_target_rs):
        """Caller certifies rows outside this set will receive no more writes."""
        active = set(active_target_rs)
        for buf in self.buffers:
            if buf['state'] == 'ACTIVE' and buf['target_r'] not in active:
                buf['state'] = 'PENDING_WB'
                buf['final_col'] = self.w - 1

    def complete_rows_through(self, last_row):
        for buf in self.buffers:
            if buf['state'] == 'ACTIVE' and buf['target_r'] <= last_row:
                buf['state'] = 'PENDING_WB'
                buf['final_col'] = self.w - 1

    def complete_columns_through(self, target_rs, last_col):
        """Certify columns ``<= last_col`` of the given rows as final.

        Used by the fully-connected path, whose packet sweep advances the column
        one step at a time: once column ``c`` has been written, columns ``< c``
        of the same rows can receive no further writes. Finer than the row
        barrier Conv uses, so a slot can start retiring -- and hence be released
        -- long before its whole row is done.
        """
        last_col = int(last_col)
        if last_col < 0:
            return
        rows = {int(r) for r in target_rs}
        for buf in self.buffers:
            if buf['state'] != 'FREE' and buf['target_r'] in rows:
                buf['final_col'] = max(buf['final_col'], last_col)

    def request_stream_retire(self, target_r, end_exclusive):
        """Compatibility API: caller must prove this prefix is final.

        Core no longer uses packet end columns as this proof.
        """
        self.row_retire_limits[int(target_r)] = max(
            self.row_retire_limits.get(int(target_r), 0), int(end_exclusive))

    def set_express_floors(self, floors):
        self.row_express_floors = {
            int(r): tuple(span) if isinstance(span, (tuple, list)) else (int(span), self.w)
            for r, span in floors.items()}

    def _express_floor_clear(self, buf):
        """True when no in-flight Express packet still owes this buffer's span.

        Interval overlap: a half-row slot owns ``[col_base, col_end)`` while an
        Express packet owes ``span``; in point mode the slot is one column wide
        and this degenerates to the original containment test.
        """
        span = self.row_express_floors.get(buf['target_r'])
        if span is None:
            return True
        return not (span[0] < buf['col_end'] and buf['col_base'] < span[1])

    def _has_stream_pending(self, buf):
        # `request_stream_retire` has no caller -- Core authorises retirement
        # only through the row barrier. Half-row allocation cannot key on a
        # single column anyway, so this legacy prefix-retire path is inert.
        if self.alloc_mode == 'halfrow':
            return False
        return (buf['state'] == 'ACTIVE'
                and buf['col'] < self.row_retire_limits.get(buf['target_r'], 0)
                and self._express_floor_clear(buf))

    def try_flush_to_accumulator(self):
        if self.current_cycle < self._retry_after:
            # 有活可干、却被上一次被拒后的错开退避挡住 → 记一拍退避等待。
            if self.has_pending():
                self.backoff_wait_cycles += 1
                self.stall_reasons['retire_backoff_wait'] += 1
            return
        if self.fc_mode:
            self._flush_fc_columns()
            return
        if self.alloc_mode == 'halfrow':
            self._flush_halfrow_retire()
            return
        rows = collections.defaultdict(list)
        for buf in self.buffers:
            if buf['state'] == 'FREE' or buf['retire_inflight']:
                continue
            # 本列的定稿由「行定稿」或「列级定稿」之一证明
            if not (buf['col'] <= buf['final_col'] or self._has_stream_pending(buf)):
                continue
            if self._unit_drained(buf):
                self._clear_buffer(buf)      # 没有待退的点，且本列已定稿
                continue
            if self._express_floor_clear(buf):
                rows[buf['target_r']].append(buf)
            else:
                # 该点仍被 Express 欠账占住，本轮无法提交。
                self.stall_reasons['express_floor_block'] += 1
        for row, candidates in sorted(rows.items()):
            if self._packet_budget_exhausted():
                break
            candidates.sort(key=lambda b: b['col'])
            dirty = []
            for buf in candidates:
                if not buf['modified'][buf['col']]:
                    self._clear_buffer(buf)
                else:
                    dirty.append(buf)
            row_final = any(b['state'] == 'PENDING_WB' for b in dirty)
            if not self._row_ready_to_emit(len(dirty), row_final):
                continue
            selected = dirty[:self.retire_column]
            if not selected:
                continue
            cols = [b['col'] for b in selected]
            kind = 'conv_row' if selected[0]['state'] == 'PENDING_WB' else 'conv_stream'
            for buf in selected:
                self._set_inflight(buf, buf['col'], True)
            # 本拍确实发起了一次 retire 提交
            self.retire_busy_cycles += 1
            self.retire_submitted_points += len(selected)
            self._packets_this_cycle += 1
            self._submit_retire(
                row, [(b['col'], b) for b in selected], cols, kind,
                lambda accepted, subset, selected=selected, row=row, cols=cols, kind=kind:
                    self._finish_conv_retire(selected, row, cols, kind, accepted))

    def _flush_fc_columns(self):
        """FC: same column, up to three rows within PE group [3g, 3g+2].

        Only final, modified and unaccepted points are eligible. Rejected points
        remain in their existing Psum slots; no unbounded retry queue is added.
        FC always uses the sparse point interface, even if CONV uses positional.
        Both allocation granularities are supported.
        """
        columns = collections.defaultdict(list)
        for buf in self.buffers:
            if buf['state'] == 'FREE':
                continue
            final_end = min(buf['col_end'], buf['final_col'] + 1)
            if self._unit_drained(buf) and final_end >= buf['col_end']:
                self._clear_buffer(buf)
                continue
            if not self._express_floor_clear(buf):
                self.stall_reasons['express_floor_block'] += 1
                continue
            for col in range(buf['col_base'], final_end):
                if (buf['modified'][col] and not buf['retired'][col]
                        and not buf['inflight'][col]):
                    columns[(buf['target_r'] // 3, col)].append(buf)
        for (_, col), candidates in sorted(columns.items()):
            if self._packet_budget_exhausted():
                break
            selected = sorted(candidates, key=lambda b: b['target_r'])[:min(3, self.retire_column)]
            rows = [buf['target_r'] for buf in selected]
            for buf in selected:
                self._set_inflight(buf, col, True)
            self._packets_this_cycle += 1
            self.retire_busy_cycles += 1
            self.retire_submitted_points += len(selected)
            self.accumulator.request_bundle_partial(
                self.core_id, rows, [col],
                [buf['data'][col].clone() for buf in selected], kind='fc_column',
                on_result=lambda mask, selected=selected, rows=rows, col=col:
                    self._finish_fc_columns(selected, rows, col, mask))

    def _finish_fc_columns(self, selected, rows, col, accepted_mask):
        if len(accepted_mask) != len(selected):
            raise AssertionError('FC response must contain one grant per point')
        self._record_request('fc_column', rows, [col], all(accepted_mask))
        accepted_count = sum(bool(ok) for ok in accepted_mask)
        rejected_count = len(selected) - accepted_count
        self.stats['row_retire_cols'] += accepted_count
        for buf, accepted in zip(selected, accepted_mask):
            self._set_inflight(buf, col, False)
            if accepted:
                buf['retired'][col] = True
        if rejected_count:
            self.stats['retire_blocked'] += 1
            self.retire_rejected_points += rejected_count
            self.stall_reasons['retire_rejected'] += 1
            self._backoff()
        for buf in selected:
            if (buf['state'] != 'FREE' and self._unit_drained(buf)
                    and buf['final_col'] >= buf['col_end'] - 1):
                self._clear_buffer(buf)

    def _finish_conv_retire(self, selected, row, cols, kind, accepted):
        self._record_request(kind, [row], cols, accepted)
        for buf in selected:
            self._set_inflight(buf, buf['col'], False)
        if accepted:
            self.stats['row_retire_cols' if kind == 'conv_row' else 'stream_retire_cols'] += len(selected)
            for buf in selected:
                self._clear_buffer(buf)
        else:
            self.stats['retire_blocked'] += 1
            # accumulator 反压：仲裁无树可用，整包被拒
            self.retire_rejected_points += len(selected)
            self.stall_reasons['retire_rejected'] += 1
            self._backoff()

    def _row_ready_to_emit(self, count, row_final):
        """CONV row batching; a final row may emit a short trailing bundle."""
        return row_final or count >= self.retire_column

    def _row_payload(self, row, pairs):
        """Assemble a positional bundle: the whole row's values plus write enable.

        `pairs` is the selected (col, buffer) list. The payload is always full
        width -- unwritten columns ride along as 0 with their enable low -- which
        is exactly what the interface trades bus width for: the column only ever
        means "which lane of this row", so the accumulator can use fixed wiring.

        A row may be split across two half-row slots and either half may be
        absent; missing columns simply stay disabled.
        """
        values = [None] * self.w
        mask = [False] * self.w
        for col, buf in pairs:
            col = int(col)
            values[col] = buf['data'][col].clone()
            mask[col] = True
        for col in range(self.w):
            if values[col] is None:
                values[col] = torch.zeros(())
        return values, mask

    def _submit_retire(self, row, pairs, cols, kind, on_result):
        """Hand one row's selected columns to the accumulator.

        ``on_result(accepted, subset)`` is called once with the full packet and
        ``subset is pairs``. The scatter form sends the selected columns as a
        sparse point list; the positional form sends the whole row plus a
        per-column write enable (see Accumulator.request_row_bundle).
        """
        if self.retire_iface == 'positional':
            values, mask = self._row_payload(row, pairs)
            self.accumulator.request_row_bundle(
                self.core_id, row, values, mask, kind=kind,
                on_result=lambda accepted: on_result(accepted, pairs))
        else:
            self.accumulator.request_bundle(
                self.core_id, [row], cols,
                [buf['data'][col].clone() for col, buf in pairs],
                kind=kind,
                on_result=lambda accepted: on_result(accepted, pairs))

    def _flush_halfrow_retire(self):
        """Half-row allocation: retire points, but release whole halves.

        Candidates are (column, slot) pairs rather than slots, because one slot
        now holds a whole half and may be only partly retired. One bundle per
        output row per cycle, still capped at ``retire_column`` columns; a
        bundle's columns may come from both halves of the same row, which is why
        the point path's per-slot bookkeeping cannot be reused verbatim.

        A half whose written columns have all landed is released here, including
        halves that never got written at all -- those never reach the retire port.
        """
        rows = collections.defaultdict(list)
        for buf in self.buffers:
            if buf['state'] == 'FREE':
                continue
            # 本半行里已定稿的列区间：Conv 由行屏障把 final_col 推到 w-1，
            # FC 则由列级定稿逐步推进，所以半行可以边写边退。
            final_end = min(buf['col_end'], buf['final_col'] + 1)
            if self._unit_drained(buf) and final_end >= buf['col_end']:
                self._clear_buffer(buf)
                continue
            if final_end <= buf['col_base']:
                continue                      # 还没有任何定稿列
            if not self._express_floor_clear(buf):
                self.stall_reasons['express_floor_block'] += 1
                continue
            for col in range(buf['col_base'], final_end):
                # 只挡在途的那几列，不整槽挡——否则加了 retire 流水之后单槽
                # 每两拍才能出一个包。
                if buf['inflight'][col]:
                    continue
                if buf['modified'][col] and not buf['retired'][col]:
                    rows[buf['target_r']].append((col, buf))
        for row, candidates in sorted(rows.items()):
            if self._packet_budget_exhausted():
                break
            candidates.sort(key=lambda item: item[0])
            if self.half_bundle:
                # 一个 bundle 只装一个半行槽；同一行的两半留到下一拍
                # （本循环本来就一行一拍，所以只需限制「不许跨半」）。
                halves = collections.defaultdict(list)
                for col, buf in candidates:
                    halves[self._half_of(col)].append((col, buf))
                picked = halves[max(halves, key=lambda h: (len(halves[h]), -h))]
            else:
                picked = candidates
            row_final = any(b['final_col'] >= b['col_end'] - 1 for _, b in picked)
            if not self._row_ready_to_emit(len(picked), row_final):
                continue
            selected = picked[:self.retire_column]
            cols = [col for col, _ in selected]
            for col, buf in selected:
                self._set_inflight(buf, col, True)
            self.retire_busy_cycles += 1
            self.retire_submitted_points += len(selected)
            self._packets_this_cycle += 1
            kind = 'conv_row'
            self._submit_retire(
                row, selected, cols, kind,
                lambda accepted, subset, row=row, kind=kind:
                    self._finish_halfrow_retire(subset, row, kind, accepted))

    def _finish_halfrow_retire(self, selected, row, kind, accepted):
        """One half-row retire write landed or bounced.

        ``selected`` covers the emitted packet's columns; on rejection every
        column stays in its slot and is retried later.
        """
        cols = [col for col, _ in selected]
        self._record_request(kind, [row], cols, accepted)
        for col, buf in selected:
            self._set_inflight(buf, col, False)
            if accepted:
                buf['retired'][col] = True
        if accepted:
            self.stats['row_retire_cols'] += len(selected)
        else:
            self.stats['retire_blocked'] += 1
            # accumulator 反压：仲裁无树可用，整包被拒；列保留原值下拍重试
            self.retire_rejected_points += len(selected)
            self.stall_reasons['retire_rejected'] += 1
            self._backoff()
        # Release only after every written column of the half has landed AND the
        # whole span is final. Guarded on state because several selected columns
        # can share one slot.
        for _, buf in selected:
            if (buf['state'] != 'FREE' and self._unit_drained(buf)
                    and buf['final_col'] >= buf['col_end'] - 1):
                self._clear_buffer(buf)

    def _backoff(self):
        # _k rejects every writer of an ungranted conflicting address. Identical
        # lockstep retries would livelock at zero/limited tree capacity. Stagger
        # retries by core ID without changing accumulator grant semantics.
        self._retry_after = max(self._retry_after, self.current_cycle + 1 + self.core_id)

    def note_write(self, buf, cols, port):
        cols = [cols] if isinstance(cols, int) else list(cols)
        inside = all(buf['col_base'] <= int(c) < buf['col_end'] for c in cols)
        if buf['state'] != 'ACTIVE' or not inside:
            raise AssertionError("write must address the allocated slot")
        if self.write_trace_hook is not None:
            self.write_trace_hook(self.current_cycle, buf['id'], port, cols)

    def occupancy_summary(self):
        summary = dict(free=0, active=0, pending_wb=0, active_stream_pending=0,
                       modified_buffers=0)
        for b in self.buffers:
            if b['state'] == 'FREE': summary['free'] += 1
            elif b['state'] == 'ACTIVE': summary['active'] += 1
            elif b['state'] == 'PENDING_WB': summary['pending_wb'] += 1
            summary['modified_buffers'] += int(bool(b['modified'].any()))
            summary['active_stream_pending'] += int(self._has_stream_pending(b))
        return summary

    def pending_summary(self):
        return dict(self.occupancy_summary(), has_pending=int(self.has_pending()))

    def _has_retirable(self, buf):
        """True when some final column of this slot is still un-retired."""
        if buf['state'] == 'FREE':
            return False
        end = min(buf['col_end'], buf['final_col'] + 1)
        for col in range(buf['col_base'], end):
            if buf['modified'][col] and not buf['retired'][col]:
                return True
        return False

    def has_pending(self):
        return any(
            b['state'] == 'PENDING_WB' or self._has_stream_pending(b)
            or self._has_retirable(b)
            for b in self.buffers)

    def _record_request(self, kind, rows, cols, success):
        if not rows or not cols:
            return

        rows = [int(row) for row in rows]
        cols = [int(col) for col in cols]
        self.request_col_hist[len(cols)] += 1
        self.request_row_hist[len(set(rows))] += 1

        request_key = f"{kind}_requests"
        accept_key = f"{kind}_accepts"
        reject_key = f"{kind}_rejects"
        self.stats[request_key] = self.stats.get(request_key, 0) + 1
        if success:
            self.stats[accept_key] = self.stats.get(accept_key, 0) + 1
        else:
            self.stats[reject_key] = self.stats.get(reject_key, 0) + 1
            self.blocked_reason_counts["row_overlap_or_busy"] += 1
