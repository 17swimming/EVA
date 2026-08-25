import collections

import torch

from model.outer_product.simulator_conv_fc_cluster_parallel.Accumulator import Accumulator


class SharedPsumPool:
    def __init__(
        self,
        core_id,
        accumulator: Accumulator,
        w=32,
        num_buffers=6,
        retire_column=3,
    ):
        self.core_id = int(core_id)
        self.accumulator = accumulator
        self.num_buffers = int(num_buffers)
        self.w = int(w)
        self.retire_column = int(retire_column)
        self.stats = {
            "stream_retire_cols": 0,
            "row_retire_cols": 0,
            "fc_retire_cols": 0,
            "fc_retire_bundles": 0,
            "retire_blocked": 0,
            "conv_stream_requests": 0,
            "conv_stream_accepts": 0,
            "conv_stream_rejects": 0,
            "conv_row_requests": 0,
            "conv_row_accepts": 0,
            "conv_row_rejects": 0,
            "fc_requests": 0,
            "fc_accepts": 0,
            "fc_rejects": 0,
            "fc_pending_retries": 0,
            "fc_pending_accepts": 0,
            "pending_wait_cycles": 0,
            "pending_wait_events": 0,
            "pending_wait_max": 0,
            "pending_retry_attempts": 0,
            "pending_wait_rows": 0,
            "fc_partial_requests": 0,
            "fc_accepted_rows": 0,
            "fc_rejected_rows": 0,
        }
        self.request_col_hist = collections.Counter()
        self.request_row_hist = collections.Counter()
        self.fc_bundle_row_hist = collections.Counter()
        self.fc_pending_wait_hist = collections.Counter()
        self.blocked_reason_counts = collections.Counter()
        self.current_cycle = 0

        self.buffers = [self._new_buffer(i) for i in range(self.num_buffers)]
        self.pending_fc_bundles = collections.deque()

    def set_cycle(self, cycle):
        self.current_cycle = int(cycle)

    def _new_buffer(self, buf_id):
        return {
            "id": buf_id,
            "state": "FREE",
            "target_r": -1,
            "data": torch.zeros(self.w, dtype=torch.float32),
            "modified": torch.zeros(self.w, dtype=torch.bool),
            "retire_cursor": 0,
            "retire_limit": 0,
            "pending_fc_counts": collections.Counter(),
        }

    def _clear_buffer(self, buf):
        buf["state"] = "FREE"
        buf["target_r"] = -1
        buf["data"].zero_()
        buf["modified"].zero_()
        buf["retire_cursor"] = 0
        buf["retire_limit"] = 0
        buf["pending_fc_counts"].clear()

    def try_flush_to_accumulator(self):
        self._try_flush_pending_fc_bundle()

        for buf in self.buffers:
            if buf["state"] == "FREE":
                continue

            cols, is_stream = self._collect_retire_cols(buf)
            if not cols:
                if buf["state"] == "PENDING_WB" and not self._has_pending_fc(buf):
                    self._clear_buffer(buf)
                continue

            kind = "conv_stream" if is_stream else "conv_row"
            success = self.accumulator.request_bundle(
                core_id=self.core_id,
                rows=[buf["target_r"]],
                cols=cols,
                data=[buf["data"][col].clone() for col in cols],
                kind=kind,
            )
            self._record_request(kind, [buf["target_r"]], cols, success)
            if not success:
                self.stats["retire_blocked"] += 1
                continue

            for col in cols:
                buf["data"][col] = 0
                buf["modified"][col] = False

            if is_stream:
                self.stats["stream_retire_cols"] += len(cols)
                self._advance_stream_cursor(buf)
            else:
                self.stats["row_retire_cols"] += len(cols)

    def request_stream_retire(self, buf, end_exclusive):
        end_exclusive = max(0, min(self.w, int(end_exclusive)))
        if end_exclusive > buf["retire_limit"]:
            buf["retire_limit"] = end_exclusive
        self._advance_stream_cursor(buf)

    def retire_old(self, active_target_rs):
        active_set = set(int(row) for row in active_target_rs)
        for buf in self.buffers:
            if buf["state"] == "ACTIVE" and buf["target_r"] not in active_set:
                buf["state"] = "PENDING_WB"
                buf["retire_limit"] = self.w

    def try_retire_fc_bundle(self, row_buffers, col):
        bundle = self._make_fc_bundle(row_buffers, col)
        if bundle is None:
            return True

        accepted_mask = self.accumulator.request_bundle_partial(
            core_id=self.core_id,
            rows=bundle["rows"],
            cols=bundle["cols"],
            data=bundle["data"],
        )
        return self._handle_fc_partial_result(bundle, accepted_mask, is_pending=False)

    def _make_fc_bundle(self, row_buffers, col):
        col = int(col)
        if col < 0 or col >= self.w:
            return None

        ordered = [
            (int(target_r), buf)
            for target_r, buf in row_buffers
        ]
        if not ordered:
            return None

        return {
            "rows": [target_r for target_r, _ in ordered],
            "cols": [col] * len(ordered),
            "data": [buf["data"][col].clone() for _, buf in ordered],
            "buffers": [buf for _, buf in ordered],
            "col": col,
            "enqueue_cycle": None,
            "retry_count": 0,
        }

    def _enqueue_fc_bundle(self, bundle):
        col = bundle["col"]
        for buf in bundle["buffers"]:
            buf["pending_fc_counts"][col] += 1
        self._clear_fc_bundle_columns(bundle)
        bundle["enqueue_cycle"] = int(self.current_cycle)
        self.pending_fc_bundles.append(bundle)

    def _try_flush_pending_fc_bundle(self):
        if not self.pending_fc_bundles:
            return True

        bundle = self.pending_fc_bundles[0]
        bundle["retry_count"] += 1
        self.stats["fc_pending_retries"] += 1
        accepted_mask = self.accumulator.request_bundle_partial(
            core_id=self.core_id,
            rows=bundle["rows"],
            cols=bundle["cols"],
            data=bundle["data"],
        )
        return self._handle_fc_partial_result(bundle, accepted_mask, is_pending=True)

    def _handle_fc_partial_result(self, bundle, accepted_mask, is_pending):
        accepted_indices = [
            index for index, accepted in enumerate(accepted_mask)
            if accepted
        ]
        rejected_indices = [
            index for index, accepted in enumerate(accepted_mask)
            if not accepted
        ]

        self._record_request(
            "fc",
            bundle["rows"],
            bundle["cols"],
            not rejected_indices,
            accepted_count=len(accepted_indices),
            rejected_count=len(rejected_indices),
        )

        if accepted_indices:
            accepted_bundle = self._slice_fc_bundle(bundle, accepted_indices)
            if is_pending:
                self._drop_fc_pending_counts(accepted_bundle)
                self.stats["fc_pending_accepts"] += 1
                self._record_pending_wait(accepted_bundle)
            else:
                self._clear_fc_bundle_columns(accepted_bundle)
            self._record_fc_bundle_accepted(accepted_bundle)

        if not rejected_indices:
            if is_pending:
                self.pending_fc_bundles.popleft()
            return True

        self.stats["retire_blocked"] += 1
        rejected_bundle = self._slice_fc_bundle(bundle, rejected_indices)
        if is_pending:
            self.pending_fc_bundles[0] = rejected_bundle
        else:
            self._enqueue_fc_bundle(rejected_bundle)
        return False

    def _slice_fc_bundle(self, bundle, indices):
        return {
            "rows": [bundle["rows"][index] for index in indices],
            "cols": [bundle["cols"][index] for index in indices],
            "data": [bundle["data"][index] for index in indices],
            "buffers": [bundle["buffers"][index] for index in indices],
            "col": bundle["col"],
            "enqueue_cycle": bundle.get("enqueue_cycle"),
            "retry_count": bundle.get("retry_count", 0),
        }

    @staticmethod
    def _clear_fc_bundle_columns(bundle):
        col = bundle["col"]
        for buf in bundle["buffers"]:
            buf["data"][col] = 0
            buf["modified"][col] = False

    @staticmethod
    def _drop_fc_pending_counts(bundle):
        col = bundle["col"]
        for buf in bundle["buffers"]:
            counts = buf["pending_fc_counts"]
            counts[col] -= 1
            if counts[col] <= 0:
                del counts[col]

    def _record_fc_bundle_accepted(self, bundle):
        self.stats["fc_retire_cols"] += len(bundle["rows"])
        self.stats["fc_retire_bundles"] += 1

    def get_or_allocate(self, target_r):
        target_r = int(target_r)
        for buf in self.buffers:
            if buf["state"] == "ACTIVE" and buf["target_r"] == target_r:
                return buf

        for buf in self.buffers:
            if buf["state"] == "FREE":
                buf["state"] = "ACTIVE"
                buf["target_r"] = target_r
                buf["data"].zero_()
                buf["modified"].zero_()
                buf["retire_cursor"] = 0
                buf["retire_limit"] = 0
                return buf

        return None

    def occupancy_summary(self):
        summary = {
            "free": 0,
            "active": 0,
            "pending_wb": 0,
            "active_stream_pending": 0,
            "active_fc_pending": 0,
            "pending_wb_fc_pending": 0,
            "modified_buffers": 0,
            "pending_fc_bundles": len(self.pending_fc_bundles),
        }
        for buf in self.buffers:
            state = buf["state"]
            if state == "FREE":
                summary["free"] += 1
            elif state == "ACTIVE":
                summary["active"] += 1
            elif state == "PENDING_WB":
                summary["pending_wb"] += 1

            if bool(buf["modified"].any().item()):
                summary["modified_buffers"] += 1
            if state == "ACTIVE" and self._has_stream_pending(buf):
                summary["active_stream_pending"] += 1
            if state == "ACTIVE" and self._has_pending_fc(buf):
                summary["active_fc_pending"] += 1
            if state == "PENDING_WB" and self._has_pending_fc(buf):
                summary["pending_wb_fc_pending"] += 1
        return summary

    def pending_summary(self):
        summary = self.occupancy_summary()
        summary["has_pending"] = int(self.has_pending())
        return summary

    def has_pending(self):
        if self.pending_fc_bundles:
            return True

        for buf in self.buffers:
            if buf["state"] == "PENDING_WB":
                return True
            if self._has_pending_fc(buf):
                return True
            if buf["state"] == "ACTIVE" and self._has_stream_pending(buf):
                return True
        return False

    @staticmethod
    def _has_pending_fc(buf):
        return bool(buf["pending_fc_counts"])

    def _has_stream_pending(self, buf):
        limit = min(self.w, int(buf["retire_limit"]))
        cursor = min(limit, int(buf["retire_cursor"]))
        if cursor >= limit:
            return False
        return bool(buf["modified"][cursor:limit].any().item())

    def _collect_retire_cols(self, buf):
        limit = min(self.w, int(buf["retire_limit"]))
        cursor = min(limit, int(buf["retire_cursor"]))
        cols = []
        while cursor < limit:
            if bool(buf["modified"][cursor].item()):
                cols.append(cursor)
                if len(cols) >= self.retire_column:
                    break
            cursor += 1

        if cols:
            buf["retire_cursor"] = cols[0]
            return cols, True

        buf["retire_cursor"] = limit

        if buf["state"] != "PENDING_WB":
            return [], False

        cols = torch.nonzero(buf["modified"], as_tuple=True)[0]
        if cols.numel() == 0:
            return [], False
        return [
            int(col.item())
            for col in cols[: self.retire_column]
        ], False

    def _advance_stream_cursor(self, buf):
        limit = min(self.w, int(buf["retire_limit"]))
        cursor = min(limit, int(buf["retire_cursor"]))
        while cursor < limit and not bool(buf["modified"][cursor].item()):
            cursor += 1
        buf["retire_cursor"] = cursor

    def _record_request(
        self,
        kind,
        rows,
        cols,
        success,
        accepted_count=None,
        rejected_count=None,
    ):
        if not rows or not cols:
            return

        rows = [int(row) for row in rows]
        cols = [int(col) for col in cols]
        self.request_col_hist[len(cols)] += 1
        self.request_row_hist[len(set(rows))] += 1
        if kind == "fc":
            self.fc_bundle_row_hist[len(set(rows))] += 1
            if accepted_count is not None:
                self.stats["fc_accepted_rows"] += int(accepted_count)
            if rejected_count is not None:
                self.stats["fc_rejected_rows"] += int(rejected_count)
            if accepted_count and rejected_count:
                self.stats["fc_partial_requests"] += 1

        request_key = f"{kind}_requests"
        accept_key = f"{kind}_accepts"
        reject_key = f"{kind}_rejects"
        self.stats[request_key] = self.stats.get(request_key, 0) + 1
        if success:
            self.stats[accept_key] = self.stats.get(accept_key, 0) + 1
        else:
            self.stats[reject_key] = self.stats.get(reject_key, 0) + 1
            self.blocked_reason_counts["row_overlap_or_busy"] += 1

    def _record_pending_wait(self, bundle):
        enqueue_cycle = bundle.get("enqueue_cycle")
        if enqueue_cycle is None:
            return
        wait = max(0, int(self.current_cycle) - int(enqueue_cycle))
        self.stats["pending_wait_cycles"] += wait
        self.stats["pending_wait_events"] += 1
        self.stats["pending_wait_rows"] += len(bundle["rows"])
        self.stats["pending_wait_max"] = max(self.stats["pending_wait_max"], wait)
        self.stats["pending_retry_attempts"] += int(bundle.get("retry_count", 0))
        self.fc_pending_wait_hist[wait] += 1
