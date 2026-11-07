import collections


class Accumulator:
    """Register-array accumulator model.

    The cluster accumulator is modeled as a full ``bank_h * bank_w`` register
    array: any distinct (row, col) point can be written in the same cycle.
    Requests are collected for the whole cycle before arbitration. A point
    with multiple offered writes needs a tree, including its first writer.
    Without a tree, every request to that point waits for another cycle.

    ``num_sub_banks`` is accepted only for backwards-compatible construction;
    it is not used by the cluster register-array arbitration model.
    """

    def __init__(
        self,
        num_sub_banks=None,
        retire_column=3,
        bank_h=None,
        bank_w=None,
        enable_reduce_tree=True,
        num_pus=1,
        mp_size=16,
        drain_words_per_cycle=3,
        num_reduce_trees=None,
        record_tree_trace=False,
        retire_pipeline_stages=0,
    ):
        self.num_sub_banks = None if num_sub_banks is None else int(num_sub_banks)
        self.bank_h = None if bank_h is None else int(bank_h)
        self.bank_w = None if bank_w is None else int(bank_w)
        self.retire_column = int(retire_column)
        self.enable_reduce_tree = bool(enable_reduce_tree)
        self.num_reduce_trees = None if num_reduce_trees is None else int(num_reduce_trees)
        if self.num_reduce_trees is not None and self.num_reduce_trees < 0:
            raise ValueError("num_reduce_trees must be nonnegative or None")
        self.record_tree_trace = bool(record_tree_trace)
        # 提交到仲裁之间的拍数。0 = 当拍提交当拍仲裁（今天的行为）。
        self.retire_pipeline_stages = int(retire_pipeline_stages)
        if self.retire_pipeline_stages not in (0, 1):
            raise ValueError("retire_pipeline_stages must be 0 or 1")
        # 流水暂存区：提交拍只进这里，下一拍的 begin_cycle 才搬进
        # `_pending_requests` 并登记树请求。暂存区非空 = 有一个包在途。
        self._staged_requests = []
        self.tree_points = set()
        self.tree_request_counts = collections.Counter()
        self.tree_demand_hist = collections.Counter()
        self.tree_used_hist = collections.Counter()
        self.tree_trace = []
        self._cycle_open = False
        self._tree_rejected_points = 0
        self._pending_requests = []
        self._arbitrated = False
        self.num_pus = int(num_pus)
        self.mp_size = int(mp_size)
        self.drain_words_per_cycle = int(drain_words_per_cycle)
        if min(self.num_pus, self.mp_size, self.drain_words_per_cycle) <= 0:
            raise ValueError("Accumulator widths and drain throughput must be positive")
        self.word_width_bits = self.num_pus * self.mp_size
        self.lif_units = self.drain_words_per_cycle * self.num_pus
        self.drain_words_remaining = 0
        self.drain_scan_words_total = 0
        self.drain_valid_words_remaining = 0
        self.drain_valid_words_per_beat = None
        self.drain_scan_banks = 0
        self.drain_scan_slots = 0

        self.accessed_rows = set()
        self.write_lookup = {}
        # A core can present more than one retire bundle in a cycle.  The DSE
        # conflict metric counts that core only once for the cycle.
        self._conflicted_cores_this_cycle = set()
        self.per_core_conflict_cycles = collections.Counter()
        self.writes = []
        self.bundles = []

        self.stats = {
            "tree_sampled_cycles": 0,
            "tree_demand_peak": 0,
            "tree_used_peak": 0,
            "tree_exhausted_cycles": 0,
            "tree_rejected_points": 0,
            "drain_requests": 0,
            "drain_cycles": 0,
            "drained_words": 0,
            "drained_bits": 0,
            "processed_writes": 0,
            "processed_bundles": 0,
            "stalled_requests": 0,
            "port_conflict_stalls": 0,
            "row_conflict_stalls": 0,
            "conflict_cycles": 0,
            "conflict_core_cycles": 0,
            "column_writes": 0,
            "reduced_requests": 0,
            "reduced_writes": 0,
            "fc_reduced_writes": 0,
            "conv_reduced_writes": 0,
            "point_conflict_requests": 0,
        }
        for category in ("all", "conv", "fc", "unknown"):
            for overlap in range(4):
                self.stats[f"{category}_request_overlap_{overlap}_requests"] = 0
                self.stats[f"{category}_conflict_overlap_{overlap}_requests"] = 0
        # Positional-interface bus accounting. Stays zero under the scatter
        # interface, so the two can be compared directly.
        self.stats.update(retire_row_bundles=0, retire_bus_bits=0,
                          retire_payload_points=0)
        self.stats.update(pipeline_staged_cycles=0, pipeline_staged_peak=0)

    def request_bundle(self, core_id, rows, cols, data, kind="unknown", on_result=None):
        """Submit an atomic bundle; callback receives bool after arbitrate()."""
        self._submit(core_id, rows, cols, data, kind, False, on_result)

    def request_row_bundle(self, core_id, row, values, mask, kind="unknown",
                           on_result=None):
        """Submit a *positional* bundle: one whole output row plus a write enable.

        The address is the row alone. That is the whole point of the interface:
        with the column only ever meaning "which lane of this row", the array can
        be written with a row select and fixed per-column lanes instead of a
        crossbar that must route every write port to every cell.

        ``values`` is the full row (never sparse -- unwritten columns travel as 0
        and still occupy the bus, which is what buys the fixed wiring), and
        ``mask`` says which columns actually drive a write. The mask is resolved
        *here*, at the interface boundary: everything below -- conflict
        detection, reduce-tree accounting, accumulation -- sees the same
        ``(row, col, value)`` points it always did, so the positional interface
        is bit-identical to the scatter one by construction rather than by luck.
        """
        if self._arbitrated:
            raise RuntimeError("Submit all cycle requests before calling arbitrate()")
        row = int(row)
        expanded = [(row, col, value)
                    for col, (value, enabled) in enumerate(zip(values, mask))
                    if enabled]
        # Bus cost of the interface itself: the full row travels every time.
        self.stats["retire_row_bundles"] += 1
        self.stats["retire_bus_bits"] += len(values) * self.word_width_bits
        self.stats["retire_payload_points"] += len(expanded)
        self._enqueue(core_id, expanded, kind, False, on_result)

    def request_bundle_partial(self, core_id, rows, cols, data, kind="fc", on_result=None):
        """Submit a partial bundle; callback receives its per-point mask."""
        self._submit(core_id, rows, cols, data, kind, True, on_result)

    def request_write(self, core_id, target_r, col, data, on_result=None):
        self.request_bundle(core_id, [target_r], [col], [data], on_result=on_result)

    def _submit(self, core_id, rows, cols, data, kind, partial, on_result):
        if self._arbitrated:
            raise RuntimeError("Submit all cycle requests before calling arbitrate()")
        expanded = self._prepare_expanded(rows, cols, data)
        self._enqueue(core_id, expanded, kind, partial, on_result)

    def _enqueue(self, core_id, expanded, kind, partial, on_result):
        """Place one expanded request on the path to arbitration.

        With ``retire_pipeline_stages == 0`` this is byte-for-byte what the
        three submit entry points always did: register the tree demand here and
        queue for this cycle's ``arbitrate()``. With one stage, the request
        parks in ``_staged_requests`` instead and is promoted -- and only then
        registered for the tree -- at the next ``begin_cycle()``.

        Registering the tree demand at *promotion* rather than at submit is not
        a stylistic choice: ``tick()`` clears ``tree_request_counts`` at the end
        of every cycle, so a count recorded at submit time would be gone before
        the arbitration that is supposed to consume it.
        """
        request = (int(core_id), expanded, kind, partial, on_result)
        if self.retire_pipeline_stages:
            self._staged_requests.append(request)
            return
        # FC results are not accumulated here -- each contribution travels to the
        # LIF on its own, so FC addresses never contest a reduce tree. Not
        # offering them also keeps them out of `conflicts`, which is what makes
        # an FC address grantable without a tree.
        if self._kind_category(kind) != "fc":
            self._record_tree_requests(expanded)
        self._pending_requests.append(request)

    def _promote_staged(self):
        """Advance the pipeline register: staged requests become this cycle's.

        Called from ``begin_cycle()`` only. That is what defines the stage
        boundary -- the simulator calls ``begin_cycle()`` exactly once per
        cycle, before the pools flush, so a packet submitted in cycle T is
        arbitrated in T+1. Test paths that drive ``tick()`` directly must call
        ``begin_cycle()`` themselves, otherwise their requests stay staged and
        the ``_staged_requests`` assertion in ``_clear_accumulator_state``
        fails loudly rather than silently dropping a packet.
        """
        if not self._staged_requests:
            return
        staged, self._staged_requests = self._staged_requests, []
        for core_id, expanded, kind, partial, on_result in staged:
            if self._kind_category(kind) != "fc":
                self._record_tree_requests(expanded)
            self._pending_requests.append(
                (core_id, expanded, kind, partial, on_result))

    def arbitrate(self):
        """Grant trees from the complete offered set, then commit all results.

        Fixed core/address priority makes grants independent of Python core
        traversal order. Atomic bundles reserve all their required trees or
        none; partial bundles may reserve individual addresses. An address
        classified as conflicting never becomes a bypass after a rejection.
        """
        if self._arbitrated:
            raise RuntimeError("Accumulator arbitration already ran this cycle")
        self._arbitrated = True
        requests, self._pending_requests = self._pending_requests, []
        conflicts = {point for point, count in self.tree_request_counts.items() if count > 1}
        grants = set()
        capacity = self.num_reduce_trees if self.enable_reduce_tree else 0
        ready = self.is_empty()
        # Grants are assigned in priority order but committed in submission
        # order, so sort indices rather than the request tuples: the two passes
        # stay decoupled and the tuple shape does not have to carry an id.
        order = sorted(range(len(requests)), key=lambda i: (
            requests[i][0],
            tuple(sorted((r, c) for r, c, _ in requests[i][1])),
            requests[i][3]))
        tree_ok = [True] * len(requests)
        if ready:
            for i in order:
                expanded, partial = requests[i][1], requests[i][3]
                required = {(r, c) for r, c, _ in expanded} & conflicts
                ok = True
                if partial:
                    for point in sorted(required):
                        if capacity is None or len(grants) < capacity:
                            grants.add(point)
                else:
                    ok = capacity is None or len(grants | required) <= capacity
                tree_ok[i] = ok
                if not ok:
                    continue
                if not partial:
                    grants.update(required)

        results = []
        used = set()
        for i, (core_id, expanded, kind, partial, callback) in enumerate(requests):
            self._record_request_overlap(kind, self._row_overlap(expanded))
            mask = [ready and tree_ok[i]
                    and ((r, c) not in conflicts or (r, c) in grants)
                    for r, c, _ in expanded]
            blocked = [item for item, accepted in zip(expanded, mask) if not accepted]
            if blocked and not partial:
                mask = [False] * len(expanded)
            accepted = [item for item, allowed in zip(expanded, mask) if allowed]
            if accepted:
                used.update((r, c) for r, c, _ in accepted if (r, c) in grants)
                self._accept_or_reduce_expanded_bundle(core_id, accepted, kind)
            if blocked:
                self._tree_rejected_points += len(blocked)
                self._record_conflict(core_id, kind, blocked)
            results.append((callback, mask if partial else not blocked))
        self.tree_points = used
        # Apply feedback only after every acceptance decision has been made.
        for callback, result in results:
            if callback is not None:
                callback(result)

    def begin_cycle(self):
        """Open a cycle: advance the retire pipeline, then sample the cycle.

        This is the stage boundary. The simulator calls it once per cycle before
        the pools flush, which is what makes a packet submitted in cycle T get
        arbitrated in T+1 when ``retire_pipeline_stages == 1``.
        """
        self._promote_staged()
        self._cycle_open = True

    def tick(self):
        # 本拍入口先统一处理上一拍收集的全部请求。request_bundle() 只负责
        # 异步提交，所有 Core 的请求都到齐后才在这里完成仲裁和回调。
        if not self._arbitrated:
            self.arbitrate()
        self.end_cycle()
        self.accessed_rows.clear()
        self.write_lookup.clear()
        self._conflicted_cores_this_cycle.clear()
        self.tree_points.clear()
        self.tree_request_counts.clear()
        self._tree_rejected_points = 0
        self._arbitrated = False
        self._cycle_open = True
        if self.drain_words_remaining:
            beat = (self.drain_scan_words_total - self.drain_words_remaining) // self.drain_words_per_cycle
            if self.drain_valid_words_per_beat is None:
                bank_group = beat // self.drain_scan_slots
                valid_words = min(self.drain_words_per_cycle,
                                  self.drain_scan_banks - bank_group * self.drain_words_per_cycle)
            else:
                valid_words = self.drain_valid_words_per_beat[beat]
            words = min(self.drain_words_per_cycle, self.drain_words_remaining)
            self.drain_words_remaining -= words
            self.drain_valid_words_remaining -= valid_words
            self.stats["drain_cycles"] += 1
            self.stats["drained_words"] += valid_words
            self.stats["drained_bits"] += valid_words * self.word_width_bits

    def end_cycle(self):
        """Sample all offered writes, including rejected points, exactly once.

        Demand is the number of addresses requested at least twice this cycle.
        It is not capped by tree capacity. Under a limit it includes retries;
        the unlimited baseline histogram is the undisturbed sizing reference.
        Call after the final cycle too, so neither it nor zero-demand cycles
        are lost. Optional trace entries are (cycle, needed, used, rejected).
        """
        if self._pending_requests:
            raise RuntimeError("Unresolved accumulator requests at end of cycle")
        if not self._cycle_open:
            return
        needed = sum(count > 1 for count in self.tree_request_counts.values())
        used = len(self.tree_points)
        cycle = self.stats['tree_sampled_cycles']
        self.tree_demand_hist[needed] += 1
        self.tree_used_hist[used] += 1
        self.stats['tree_sampled_cycles'] += 1
        self.stats['tree_demand_peak'] = max(self.stats['tree_demand_peak'], needed)
        self.stats['tree_used_peak'] = max(self.stats['tree_used_peak'], used)
        self.stats['tree_exhausted_cycles'] += int(self._tree_rejected_points > 0)
        self.stats['tree_rejected_points'] += self._tree_rejected_points
        self.stats['pipeline_staged_cycles'] += int(bool(self._staged_requests))
        self.stats['pipeline_staged_peak'] = max(
            self.stats['pipeline_staged_peak'], len(self._staged_requests))
        if self.record_tree_trace:
            self.tree_trace.append((cycle, needed, used, self._tree_rejected_points))
        self._cycle_open = False

    def _record_tree_requests(self, expanded):
        self._cycle_open = True
        self.tree_request_counts.update((int(r), int(c)) for r, c, _ in expanded)

    def request_drain(self, *, fc_m=None, fc_pe_rows=None):
        """Signal completed accumulation; each tick scans one bank group and slot.

        A word is one (row, col) point containing num_pus membrane potentials.
        Without FC shape configuration, scan the full bank/slot array.
        With FC M and active PE rows, skip unoccupied slot/bank groups while
        retaining the same three-bank-per-cycle read granularity as RTL.
        The caller must first finish all K tasks and retire their Psum pools.
        """
        if not self.is_empty():
            raise RuntimeError("Accumulator drain is already in progress")
        if self.bank_h is None or self.bank_w is None or min(self.bank_h, self.bank_w) <= 0:
            raise ValueError("Accumulator drain requires positive bank_h and bank_w")
        self.drain_scan_banks, self.drain_scan_slots = (
            (self.bank_w, self.bank_h) if getattr(self, 'mode', None) == 'fc'
            else (self.bank_h, self.bank_w))
        self.drain_valid_words_per_beat = None
        if fc_m is not None:
            # FC 的有效范围由 M 与 PE 行数决定，不能用数值是否为零
            # 判断：零膜电位仍属于本 tile，必须经过 LIF 并保持 RTL 拍数。
            if fc_pe_rows is None or not 1 <= fc_pe_rows <= 3:
                raise ValueError('FC drain requires one to three active PE rows')
            full_groups, tail_m = divmod(int(fc_m), self.bank_w)
            if not 1 <= fc_m <= self.bank_w * (self.bank_h // 3):
                raise ValueError('FC M tile exceeds the physical result array')
            tail_bank_groups = ((tail_m + self.drain_words_per_cycle - 1)
                                // self.drain_words_per_cycle)
            bank_groups = ((self.bank_w + self.drain_words_per_cycle - 1)
                           // self.drain_words_per_cycle) if full_groups else tail_bank_groups
            beats = []
            # 对齐 RTL 的 bank 组→M 组→PE 行扫描顺序；每个 beat
            # 仍占一次三 bank 读口，尾组仅减少有效 lane 而非读口宽度。
            for bank_group in range(bank_groups):
                for m_group in range(full_groups + int(bank_group < tail_bank_groups)):
                    active_banks = self.bank_w if m_group < full_groups else tail_m
                    valid_banks = min(self.drain_words_per_cycle,
                                      active_banks - bank_group * self.drain_words_per_cycle)
                    beats.extend([valid_banks] * fc_pe_rows)
            self.drain_valid_words_per_beat = beats
            self.drain_scan_words_total = len(beats) * self.drain_words_per_cycle
            self.drain_valid_words_remaining = sum(beats)
        else:
            self.drain_scan_words_total = (
                ((self.drain_scan_banks + self.drain_words_per_cycle - 1)
                 // self.drain_words_per_cycle)
                * self.drain_scan_slots * self.drain_words_per_cycle)
            self.drain_valid_words_remaining = self.bank_h * self.bank_w
        self.drain_words_remaining = self.drain_scan_words_total
        self.stats["drain_requests"] += 1

    def is_empty(self):
        # No pending output operation. Accepted Psum contributions themselves
        # do not start a drain; only the scheduler's completion signal does.
        return self.drain_words_remaining == 0

    def _prepare_expanded(self, rows, cols, data):
        rows = [int(row) for row in rows]
        cols = [int(col) for col in cols]
        return self._expand_bundle(rows, cols, data)

    def _point_conflicts(self, expanded):
        return [
            item for item in expanded
            if (int(item[0]), int(item[1])) in self.write_lookup
        ]

    def _accept_or_reduce_expanded_bundle(self, core_id, expanded, kind):
        bundle_id = len(self.bundles)
        write_records = []
        reduced_count = 0
        # FC is a pass-through: without accumulation a repeated address simply
        # takes the latest contribution, and the Cin sum is left to the LIF.
        pass_through = self._kind_category(kind) == "fc"

        for target_r, col, value in expanded:
            point = (int(target_r), int(col))
            existing = self.write_lookup.get(point)
            if existing is not None:
                if pass_through:
                    existing["data"] = value
                    continue
                self.tree_points.add(point)
                existing["data"] = self._merge_value(existing["data"], value)
                reduced_count += 1
                continue

            record = {
                "core_id": int(core_id),
                "target_r": int(target_r),
                "col": int(col),
                "data": value,
                "bundle_id": bundle_id,
            }
            self.writes.append(record)
            write_records.append(record)
            self.write_lookup[point] = record
            self.accessed_rows.add(int(target_r))

        self.bundles.append(
            {
                "core_id": int(core_id),
                "rows": sorted({int(target_r) for target_r, _, _ in expanded}),
                "cols": [int(col) for _, col, _ in expanded],
                "writes": write_records,
            }
        )
        self.stats["processed_bundles"] += 1
        self.stats["processed_writes"] += len(write_records)
        self.stats["column_writes"] += len(write_records)

        if reduced_count:
            category = self._kind_category(kind)
            self.stats["reduced_requests"] += 1
            self.stats["reduced_writes"] += reduced_count
            self.stats["point_conflict_requests"] += 1
            if category == "fc":
                self.stats["fc_reduced_writes"] += reduced_count
            elif category == "conv":
                self.stats["conv_reduced_writes"] += reduced_count

    def _accept_expanded_bundle(self, core_id, expanded):
        bundle_id = len(self.bundles)
        write_records = []

        for target_r, col, value in expanded:
            point = (int(target_r), int(col))
            record = {
                "core_id": int(core_id),
                "target_r": int(target_r),
                "col": int(col),
                "data": value,
                "bundle_id": bundle_id,
            }
            self.writes.append(record)
            write_records.append(record)
            self.write_lookup[point] = record
            self.accessed_rows.add(int(target_r))

        self.bundles.append(
            {
                "core_id": int(core_id),
                "rows": sorted({int(target_r) for target_r, _, _ in expanded}),
                "cols": [int(col) for _, col, _ in expanded],
                "writes": write_records,
            }
        )
        self.stats["processed_bundles"] += 1
        self.stats["processed_writes"] += len(write_records)
        self.stats["column_writes"] += len(write_records)

    def _record_conflict(self, core_id, kind, conflicting):
        overlap = self._row_overlap(conflicting)
        overlap = max(0, min(3, int(overlap)))
        category = self._kind_category(kind)
        self.stats["stalled_requests"] += 1
        self.stats["port_conflict_stalls"] += 1
        self.stats["conflict_cycles"] += 1
        core_id = int(core_id)
        if core_id not in self._conflicted_cores_this_cycle:
            self._conflicted_cores_this_cycle.add(core_id)
            self.stats["conflict_core_cycles"] += 1
            self.per_core_conflict_cycles[core_id] += 1
        self.stats["point_conflict_requests"] += 1
        self.stats[f"all_conflict_overlap_{overlap}_requests"] += 1
        self.stats[f"{category}_conflict_overlap_{overlap}_requests"] += 1

    def _row_overlap(self, expanded):
        row_set = {int(target_r) for target_r, _, _ in expanded}
        return len(row_set & self.accessed_rows)

    def _expand_bundle(self, rows, cols, data):
        if data is None:
            data_items = [None]
        elif isinstance(data, (list, tuple)):
            data_items = list(data)
        else:
            data_items = [data]

        if len(rows) == 1:
            data_items = self._fit_data_items(data_items, len(cols))
            return [(rows[0], col, data_items[idx]) for idx, col in enumerate(cols)]

        if len(cols) == 1:
            data_items = self._fit_data_items(data_items, len(rows))
            return [(row, cols[0], data_items[idx]) for idx, row in enumerate(rows)]

        if len(rows) != len(cols):
            raise ValueError(
                f"Invalid accumulator bundle shape: rows={len(rows)}, cols={len(cols)}"
            )

        data_items = self._fit_data_items(data_items, len(rows))
        return [
            (rows[idx], cols[idx], data_items[idx])
            for idx in range(len(rows))
        ]

    @staticmethod
    def _fit_data_items(data_items, target_len):
        if target_len <= 0:
            return []
        if len(data_items) == target_len:
            return data_items
        if len(data_items) == 1:
            return data_items * target_len
        raise ValueError(
            f"Invalid accumulator bundle data length: {len(data_items)} for {target_len} writes"
        )

    def _record_request_overlap(self, kind, overlap):
        overlap = max(0, min(3, int(overlap)))
        category = self._kind_category(kind)
        self.stats[f"all_request_overlap_{overlap}_requests"] += 1
        self.stats[f"{category}_request_overlap_{overlap}_requests"] += 1

    @staticmethod
    def _merge_value(left, right):
        if left is None:
            return right
        if right is None:
            return left
        return left + right

    @staticmethod
    def _kind_category(kind):
        kind = str(kind)
        if kind.startswith("conv"):
            return "conv"
        if kind.startswith("fc"):
            return "fc"
        return "unknown"
