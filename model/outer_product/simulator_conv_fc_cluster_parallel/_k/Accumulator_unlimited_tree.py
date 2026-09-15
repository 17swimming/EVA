import collections


class Accumulator:
    """Register-array accumulator model.

    The cluster accumulator is modeled as a full ``bank_h * bank_w`` register
    array: any distinct (row, col) point can be written in the same cycle.
    With ``enable_reduce_tree=True``, multiple writes to the same point are
    merged before the logical accumulator write. With it disabled, the later
    same-point request must retry.

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
    ):
        self.num_sub_banks = None if num_sub_banks is None else int(num_sub_banks)
        self.bank_h = None if bank_h is None else int(bank_h)
        self.bank_w = None if bank_w is None else int(bank_w)
        self.retire_column = int(retire_column)
        self.enable_reduce_tree = bool(enable_reduce_tree)
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

    def request_bundle(self, core_id, rows, cols, data, kind="unknown"):
        if not self.is_empty():
            return False
        expanded = self._prepare_expanded(rows, cols, data)
        if not expanded:
            return True
        self._record_request_overlap(kind, self._row_overlap(expanded))
        if self.enable_reduce_tree:
            self._accept_or_reduce_expanded_bundle(core_id, expanded, kind)
            return True

        conflicts = self._point_conflicts(expanded)
        if conflicts:
            self._record_conflict(core_id, kind, conflicts)
            return False

        self._accept_expanded_bundle(core_id, expanded)
        return True

    def request_bundle_partial(self, core_id, rows, cols, data, kind="fc"):
        expanded = self._prepare_expanded(rows, cols, data)
        if not self.is_empty():
            return [False] * len(expanded)
        if not expanded:
            return []
        self._record_request_overlap(kind, self._row_overlap(expanded))
        if self.enable_reduce_tree:
            self._accept_or_reduce_expanded_bundle(core_id, expanded, kind)
            return [True] * len(expanded)

        accepted = []
        accepted_mask = []
        conflicting = []

        for item in expanded:
            point = (int(item[0]), int(item[1]))
            if point in self.write_lookup:
                conflicting.append(item)
                accepted_mask.append(False)
            else:
                accepted.append(item)
                accepted_mask.append(True)

        if accepted:
            self._accept_expanded_bundle(core_id, accepted)
        if conflicting:
            self._record_conflict(core_id, kind, conflicting)

        return accepted_mask

    def request_write(self, core_id, target_r, col, data):
        return self.request_bundle(core_id, [target_r], [col], [data])

    def tick(self):
        self.accessed_rows.clear()
        self.write_lookup.clear()
        self._conflicted_cores_this_cycle.clear()
        if self.drain_words_remaining:
            words = min(self.drain_words_per_cycle, self.drain_words_remaining)
            self.drain_words_remaining -= words
            self.stats["drain_cycles"] += 1
            self.stats["drained_words"] += words
            self.stats["drained_bits"] += words * self.word_width_bits

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
