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
        drain_words_per_cycle=4,
        num_reduce_trees=None,
        record_tree_trace=False,
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

    def request_bundle(self, core_id, rows, cols, data, kind="unknown", on_result=None):
        """Submit an atomic bundle; callback receives bool after arbitrate()."""
        self._submit(core_id, rows, cols, data, kind, False, on_result)

    def request_bundle_partial(self, core_id, rows, cols, data, kind="fc", on_result=None):
        """Submit a partial bundle; callback receives its per-point mask."""
        self._submit(core_id, rows, cols, data, kind, True, on_result)

    def request_write(self, core_id, target_r, col, data, on_result=None):
        self.request_bundle(core_id, [target_r], [col], [data], on_result=on_result)

    def _submit(self, core_id, rows, cols, data, kind, partial, on_result):
        if self._arbitrated:
            raise RuntimeError("Submit all cycle requests before calling arbitrate()")
        expanded = self._prepare_expanded(rows, cols, data)
        self._record_tree_requests(expanded)
        self._pending_requests.append((int(core_id), expanded, kind, partial, on_result))

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
        priority = sorted(requests, key=lambda req: (
            req[0], tuple(sorted((r, c) for r, c, _ in req[1])), req[3]))
        if ready:
            for _, expanded, _, partial, _ in priority:
                required = {(r, c) for r, c, _ in expanded} & conflicts
                if partial:
                    for point in sorted(required):
                        if capacity is None or len(grants) < capacity:
                            grants.add(point)
                elif capacity is None or len(grants | required) <= capacity:
                    grants.update(required)

        results = []
        used = set()
        for core_id, expanded, kind, partial, callback in requests:
            self._record_request_overlap(kind, self._row_overlap(expanded))
            mask = [ready and ((r, c) not in conflicts or (r, c) in grants)
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
            words = min(self.drain_words_per_cycle, self.drain_words_remaining)
            self.drain_words_remaining -= words
            self.stats["drain_cycles"] += 1
            self.stats["drained_words"] += words
            self.stats["drained_bits"] += words * self.word_width_bits

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
        if self.record_tree_trace:
            self.tree_trace.append((cycle, needed, used, self._tree_rejected_points))
        self._cycle_open = False

    def _record_tree_requests(self, expanded):
        self._cycle_open = True
        self.tree_request_counts.update((int(r), int(c)) for r, c, _ in expanded)

    def request_drain(self):
        """Signal completed accumulation; the next tick drains up to four words.

        A word is one (row, col) point containing num_pus membrane potentials.
        Scan the full physical array, including zero and unused entries.
        The caller must first finish all K tasks and retire their Psum pools.
        """
        if not self.is_empty():
            raise RuntimeError("Accumulator drain is already in progress")
        if self.bank_h is None or self.bank_w is None or min(self.bank_h, self.bank_w) <= 0:
            raise ValueError("Accumulator drain requires positive bank_h and bank_w")
        self.drain_words_remaining = self.bank_h * self.bank_w
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

        for target_r, col, value in expanded:
            point = (int(target_r), int(col))
            existing = self.write_lookup.get(point)
            if existing is not None:
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
        if kind == "fc":
            return "fc"
        return "unknown"
