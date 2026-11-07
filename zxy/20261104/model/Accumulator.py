"""有界退休队列、可配置候选数的 bank 仲裁和本地 result_array 读出、累加及写回。

握手确认请求入队。新入队请求不能在同拍发射；已发射请求束下一拍写入，
排空判断包含请求队列和写回寄存器。按 bank 组和 slot 遍历result array以完成LIF计算。
"""
import collections
from dataclasses import dataclass

from .reference_accumulator import Accumulator as ReferenceAccumulator
from .lif_scan import LIFScan, lif_value


@dataclass(frozen=True)
class Bundle:
    core: int
    bank: int
    points: tuple
    born: int
    serial: int
    epoch: int = 0


class Accumulator(ReferenceAccumulator):
    buffered_retire = True

    def __init__(self, *args, num_cores=6, buffer_depth=2, lookahead=True,
                 input_width=12, record_bank_trace=False, bank_lookahead_entries=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.retire_pipeline_stages:
            raise ValueError('This experiment uses scatter bundles and its own commit stage')
        if not self.bank_h or not self.bank_w:
            raise ValueError('Physical result-array dimensions are required')
        if not isinstance(buffer_depth, int) or buffer_depth < 1:
            raise ValueError('bundle buffer 深度必须为正整数')
        if bank_lookahead_entries is None:
            bank_lookahead_entries = min(2, buffer_depth) if lookahead else 1
        if not isinstance(bank_lookahead_entries, int) or not 1 <= bank_lookahead_entries <= buffer_depth:
            raise ValueError('仲裁候选数必须位于一和 bundle buffer 深度之间')
        self.enable_reduce_tree = False
        self.num_reduce_trees = 0
        self.num_cores = int(num_cores)
        self.depth = int(buffer_depth)
        self.lookahead = bool(lookahead)
        self.candidate_count = bank_lookahead_entries if self.lookahead else 1
        self.input_width = int(input_width)
        self.record_bank_trace = bool(record_bank_trace)
        self.bank_trace = []
        self.queues = [[None] * self.depth for _ in range(self.num_cores)]
        self.pipeline = []
        self._free_at_start = [[] for _ in self.queues]
        self._clock = 0
        self._serial = 0
        self.mode = None
        self.result_array = None
        self.rr = [0] * max(self.bank_h, self.bank_w)
        self.bank_stats = collections.Counter()
        self.occupancy_hist = collections.Counter()
        self.wait_hist = collections.Counter()
        self.bank_services = collections.Counter()
        self.core_full_cycles = collections.Counter()
        self._opened = False
        self.lif_contexts = collections.deque()
        self.completed_lif = []
        self.submit_epoch = -1
        self.core_frontiers = [0] * self.num_cores
        self.core_epochs = [None] * self.num_cores
        self._accept_epoch = None
        self._lif_during_compute = False

    def begin_cycle(self):
        super().begin_cycle()
        self._opened = True
        self._lif_during_compute = False
        # 采样发射计数，以继续统计下一轮与当前轮计算重叠的周期。
        # 集成把拍初安全查询的 cores 换成了只读快照，因此这里用真实 core 对象取样。
        self._issue_before = {
            core: getattr(core, 'compute_issue_cycles', 0) + getattr(core, 'express_issue_cycles', 0)
            for core in (self.lif_contexts[0].get('core_objects', ()) if self.lif_contexts else ())
        }
        self._accept_epoch = self.lif_contexts[0]['epoch'] if self.lif_contexts else None
        self._free_at_start = [[i for i, item in enumerate(queue) if item is None]
                               for queue in self.queues]
        for core, queue in enumerate(self.queues):
            occupancy = sum(item is not None for item in queue)
            self.occupancy_hist[occupancy] += 1
            if occupancy == self.depth:
                self.core_full_cycles[core] += 1
        self.bank_stats['sampled_cycles'] += 1

    def _enqueue(self, core_id, expanded, kind, partial, on_result):
        if not self._opened:
            raise RuntimeError('Call begin_cycle before submitting retirement requests')
        core_id = int(core_id)
        if not 0 <= core_id < self.num_cores:
            raise ValueError('Core index outside cluster')
        mode = 'fc' if self._kind_category(kind) == 'fc' else 'conv'
        if self.mode is None:
            self.mode = mode
            banks, slots = ((self.bank_w, self.bank_h) if mode == 'fc'
                            else (self.bank_h, self.bank_w))
            self.result_array = [[[0] * self.num_pus for _ in range(slots)]
                                 for _ in range(banks)]
        elif mode != self.mode:
            raise ValueError('Drain the output context before changing CONV/FC mode')
        if not 1 <= len(expanded) <= 3:
            raise ValueError('Each bundle must contain one to three valid points')
        points = []
        bank_set = set()
        for row, col, value in expanded:
            if not (0 <= row < self.bank_h and 0 <= col < self.bank_w):
                raise ValueError('Result address outside physical array')
            bank, slot = (col, row) if mode == 'fc' else (row, col)
            bank_set.add(bank)
            points.append((slot, self._lanes(value)))
        if len(bank_set) != 1 or len({slot for slot, _ in points}) != len(points):
            raise ValueError('A bundle requires one bank and distinct slots')
        # Payload is captured here, independently of the source Psum lifetime.
        bundle = Bundle(core_id, bank_set.pop(), tuple(points), self._clock, self._serial,
                        (self.core_epochs[core_id] if self.core_epochs[core_id] is not None
                         else max(0, self.submit_epoch)))
        self._serial += 1
        self._pending_requests.append((bundle, partial, on_result, kind))

    def _lanes(self, value):
        if hasattr(value, 'detach'):
            value = value.detach().reshape(-1).tolist()
        elif not isinstance(value, (list, tuple)):
            value = [value]
        if len(value) not in (1, self.num_pus):
            raise ValueError('Payload must be the representative PU or all PU lanes')
        lanes = tuple(int(item) for item in value)
        if any(int(item) != item or not 0 <= int(item) < (1 << self.input_width)
               for item in value):
            raise ValueError('Payload must fit the unsigned Core increment width')
        # zxy_k computes only PU0 numerically. Do not fabricate other PU values.
        return lanes

    def _commit(self, bundles):
        modulus_mask = (1 << self.mp_size) - 1
        for bundle in bundles:
            for slot, lanes in bundle.points:
                target = self.result_array[bundle.bank][slot]
                for lane, delta in enumerate(lanes):
                    target[lane] = (target[lane] + delta) & modulus_mask
            self.stats['processed_bundles'] += 1
            self.stats['processed_writes'] += len(bundle.points)
            self.stats['column_writes'] += len(bundle.points)
            self.bank_stats['committed_bundles'] += 1
            self.bank_stats['committed_points'] += len(bundle.points)
            if self.record_bank_trace:
                self.bank_trace.append((self._clock, 'commit', bundle.core,
                                        bundle.bank, bundle.serial))

    def arbitrate(self):
        if self._arbitrated:
            raise RuntimeError('Duplicate arbitration')
        self._arbitrated = True
        self._advance_lif()
        self._commit(self.pipeline)
        self.pipeline = []
        ordered = [sorted(((i, b) for i, b in enumerate(q) if b is not None),
                          key=lambda pair: pair[1].serial) for q in self.queues]
        issued_cores, used_banks = set(), set()
        if self._opened:
            for queue in ordered:
                if len(queue) == 2:
                    key = 'two_entries_same_bank' if queue[0][1].bank == queue[1][1].bank else 'two_entries_different_bank'
                    self.bank_stats[key] += 1
        for round_index in range(self.candidate_count):
            candidates = collections.defaultdict(list)
            for core, queue in enumerate(ordered):
                if core in issued_cores or len(queue) <= round_index:
                    continue
                index, bundle = queue[round_index]
                if bundle.bank not in used_banks:
                    candidates[bundle.bank].append((core, index, bundle))
            for bank, choices in sorted(candidates.items()):
                core, index, bundle = min(choices, key=lambda item: (item[0] - self.rr[bank]) % self.num_cores)
                self.queues[core][index] = None
                issued_cores.add(core)
                used_banks.add(bank)
                self.rr[bank] = (core + 1) % self.num_cores
                self.pipeline.append(bundle)
                self.bank_services[bank] += 1
                wait = self._clock - bundle.born
                self.wait_hist[wait] += 1
                self.bank_stats['wait_sum'] += wait
                self.bank_stats['wait_max'] = max(self.bank_stats['wait_max'], wait)
                self.bank_stats['issued_bundles'] += 1
                self.bank_stats['issued_points'] += len(bundle.points)
                self.bank_stats['bypass_bundles'] += int(round_index > 0)
                self.bank_stats[f'candidate_{round_index+1}_issued_bundles'] += 1
                if self.record_bank_trace:
                    self.bank_trace.append((self._clock, 'issue', core, bank, bundle.serial))
        if self._opened:
            self.bank_stats['head_bank_blocked_core_cycles'] += sum(
                bool(q) and c not in issued_cores for c, q in enumerate(ordered))
            self.bank_stats['issued_bundle_peak'] = max(
                self.bank_stats['issued_bundle_peak'], len(issued_cores))
        requests, self._pending_requests = self._pending_requests, []
        callbacks = []
        offered_per_core = [0] * self.num_cores
        accepted_axes = collections.defaultdict(list)
        for bundle, partial, callback, kind in requests:
            core = bundle.core
            port = offered_per_core[core]
            if port >= 2:
                raise AssertionError('At most two offered bundles per Core per cycle are allowed')
            offered_per_core[core] += 1
            self.bank_stats['offered_bundles'] += 1
            # 与 RTL ready[0]/ready[1] 一致：只看周期开始时的空表项。
            # 即使本拍发射腾出表项，也不能旁路给新包；只剩一空位时
            # 首包可接收，第二包必须在源 Psum 中保留并于下拍重试。
            accepted = (len(self._free_at_start[core]) > port
                        and not self.drain_words_remaining
                        and (self._accept_epoch is None or bundle.epoch == self._accept_epoch))
            if accepted:
                index = self._free_at_start[core][port]
                assert self.queues[core][index] is None
                self.queues[core][index] = bundle
                self.bank_stats['enqueued_bundles'] += 1
                self.bank_stats['enqueued_points'] += len(bundle.points)
                accepted_axes[core].append(min(s for s, _ in bundle.points)//3*3
                                           if self.mode == 'fc' else bundle.bank)
            else:
                self.bank_stats['enqueue_rejected_bundles'] += 1
                self._record_conflict(core, kind, [(bundle.bank, s, v) for s, v in bundle.points])
            callbacks.append((callback, [accepted] * len(bundle.points) if partial else accepted))
        # One acknowledgement at enqueue. There is no later response to a pool
        # which may have been released and reused for the next K/Cin task.
        for callback, response in callbacks:
            if callback is not None:
                callback(response)
        for core, axes in accepted_axes.items():
            self.core_frontiers[core] = min(axes)

    def begin_lif_context(self, *, conv_shape=None, fc_m=None, fc_pe_rows=3,
                          last_t=False, channel_mask=None, threshold=None, leak=0,
                          reset_potential=0, with_lif=True):
        """开始一个时间步或输出行分块；保持前一上下文遍历result array以完成LIF计算并反压新上下文的退休请求。"""
        if len(self.lif_contexts) >= 2:
            raise RuntimeError('Two output contexts are already occupied')
        self.submit_epoch += 1
        mode = 'fc' if fc_m is not None else 'conv'
        if self.mode not in (None, mode):
            raise ValueError('CONV/FC switch requires completed LIF contexts')
        self.mode = mode
        if self.result_array is None:
            banks, slots = (self.bank_w, self.bank_h) if mode == 'fc' else (self.bank_h, self.bank_w)
            self.result_array = [[[0]*self.num_pus for _ in range(slots)] for _ in range(banks)]
        scan = LIFScan(self.bank_h, self.bank_w, conv_shape=conv_shape,
                       fc_m=fc_m, fc_pe_rows=fc_pe_rows, points=self.drain_words_per_cycle)
        context = dict(epoch=self.submit_epoch, scan=scan, backend_done=False,
                       dispatched=False, cores=(), early=False, last_t=last_t,
                       channels=channel_mask, threshold=threshold, leak=leak,
                       reset=reset_potential, pending_values={}, read_cycles=0,
                       hidden_cycles=0, spikes=[], started=False, with_lif=with_lif)
        self.lif_contexts.append(context)
        self.bank_stats['lif_context_peak'] = max(self.bank_stats['lif_context_peak'],
                                                 len(self.lif_contexts))
        self.stats['drain_requests'] += 1
        return context

    def set_lif_progress(self, cores, *, dispatched, early=True, epoch=None,
                         overlap=False, during_compute=True):
        # 原循环提供本拍是否仍有正常计算，避免将纯等待误算为隐藏周期。
        self._lif_during_compute = during_compute
        epoch = self.submit_epoch if epoch is None else epoch
        context = next((c for c in self.lif_contexts if c['epoch'] == epoch), None)
        if not overlap:
            context.update(cores=tuple(cores), dispatched=bool(dispatched), early=bool(early))
            return
        # 派发完成信号保持有效；等待下一上下文容量时不能撤销上一轮的完成信号。
        if context is not None:
            context['dispatched'] |= bool(dispatched)
        for context in self.lif_contexts:
            context.update(cores=tuple(cores), early=bool(early))
            outstanding = any(
                (not core.is_finished and core.lif_epoch == context['epoch']) or
                (core.pending_tile is not None and core.pending_lif_epoch == context['epoch'])
                for core in cores)
            if context['dispatched'] and not outstanding:
                self.finish_backend(context['epoch'])
        # 沿用原有重叠统计，仅将信号更新集中在此处，不再另设执行循环。
        live_epochs = {core.lif_epoch for core in cores if not core.is_finished}
        self.bank_stats['mixed_epoch_compute_cycles'] += int(len(live_epochs) > 1)
        head_epoch = self.lif_contexts[0]['epoch'] if self.lif_contexts else None
        old_active = any(not core.is_finished and core.lif_epoch == head_epoch for core in cores)
        next_issue = any(core.lif_epoch != head_epoch and
            core.compute_issue_cycles + core.express_issue_cycles > self._issue_before.get(
                core, core.compute_issue_cycles + core.express_issue_cycles) for core in cores)
        self.bank_stats['next_epoch_issue_while_old_active_cycles'] += int(old_active and next_issue)

    def finish_backend(self, epoch=None):
        epoch = self.submit_epoch if epoch is None else epoch
        context = next((c for c in self.lif_contexts if c['epoch'] == epoch), None)
        if context is not None:
            context.update(backend_done=True, dispatched=True)

    def start_core_task(self, core_id, epoch=None):
        self.core_frontiers[core_id] = 0
        self.core_epochs[core_id] = self.submit_epoch if epoch is None else epoch

    def _lif_safe_axis(self, context):
        """选择最后一条当前上下文写通路，再比较其请求地址。"""
        selected_core = None
        remaining_tasks = 0
        if not context['backend_done']:
            if not context['dispatched'] or not context['early']:
                return 0
            if any(getattr(core, 'has_old_banks', False)
                   and getattr(core, 'lif_epoch', context['epoch']) == context['epoch']
                   for core in context['cores']):
                return 0
            active = [i for i, core in enumerate(context['cores'])
                      if not core.is_finished and
                      getattr(core, 'lif_epoch', context['epoch']) == context['epoch']]
            pending = [core for core in context['cores'] if core.pending_tile is not None and
                       getattr(core, 'pending_lif_epoch', context['epoch']) == context['epoch']]
            remaining_tasks = len(active) + len(pending)
            if remaining_tasks > 1 or pending:
                return 0
            if active:
                selected_core = active[0]
        # 排空判断只检查通路归属；地址最小值只在选中的两个缓冲表项及写回流水内计算。
        bundles = self.pipeline + [b for q in self.queues for b in q if b is not None]
        bundles += [r[0] for r in self._pending_requests if r[0].epoch == context['epoch']]
        bundles = [bundle for bundle in bundles if bundle.epoch == context['epoch']]
        write_cores = {bundle.core for bundle in bundles}
        if remaining_tasks == 0:
            if len(write_cores) > 1:
                return 0
            selected_core = next(iter(write_cores), None)
        elif write_cores - {selected_core}:
            return 0
        safe = self.bank_h if remaining_tasks == 0 else self.core_frontiers[selected_core]
        if remaining_tasks:
            safe = min(safe, getattr(context['cores'][selected_core], 'psum_safe_axis', safe))
        for bundle in bundles:
            if bundle.core != selected_core:
                continue
            axis = min(s for s, _ in bundle.points) if context['scan'].fc else bundle.bank
            safe = min(safe, axis)
        return safe

    def _advance_lif(self):
        if not self.lif_contexts:
            return
        context = self.lif_contexts[0]
        if not context['with_lif']:
            pending = self.pipeline + [bundle for queue in self.queues for bundle in queue if bundle is not None]
            pending += [request[0] for request in self._pending_requests]
            if context['backend_done'] and not any(bundle.epoch == context['epoch'] for bundle in pending):
                # 中间矩阵不执行泄漏积分发放计算；写通路排空后释放物理存储，允许后继输出分块退休。
                for bank in self.result_array:
                    for slot in bank:
                        slot[:] = [0] * self.num_pus
                self.lif_contexts.popleft()
            return
        scan = context['scan']
        if not context['started']:
            if context['dispatched']:
                context['started'] = True
            # 与硬件描述符锁存启动沿一致；当前时间步任务尚未全部调度时不能发读。
            scan.tick(0)
            return
        safe = self._lif_safe_axis(context)
        read, writes = scan.tick(safe)
        if read is not None:
            context['read_cycles'] += 1
            context['hidden_cycles'] += int(self._lif_during_compute)
            self.stats['drain_cycles'] += 1
            self.stats['drained_words'] += len(read.banks)
            self.stats['drained_bits'] += len(read.banks) * self.word_width_bits
            values = []
            for bank in read.banks:
                lanes = list(self.result_array[bank][read.slot])
                for lane, value in enumerate(lanes):
                    bit = lane*3 + read.slot%3 if scan.fc else lane
                    enabled = context['channels'] is None or bool(context['channels'] & (1 << bit))
                    if enabled and context['threshold'] is not None:
                        lanes[lane], spike = lif_value(value, context['threshold'], context['leak'],
                                                       context['reset'], self.mp_size)
                        context['spikes'].append((bank, read.slot, lane, spike))
                    if context['last_t']:
                        lanes[lane] = 0
                values.append(lanes)
            context['pending_values'][read] = values
        for beat in writes:
            for bank, lanes in zip(beat.banks, context['pending_values'].pop(beat)):
                self.result_array[bank][beat.slot][:] = lanes
        if scan.done:
            self.completed_lif.append(self.lif_contexts.popleft())
            self.bank_stats['lif_contexts'] += 1
            self.bank_stats['lif_read_cycles'] += context['read_cycles']
            self.bank_stats['lif_hidden_read_cycles'] += context['hidden_cycles']

    def tick(self):
        was_draining = self.drain_words_remaining > 0
        super().tick()
        self._opened = False
        self._clock += 1
        if was_draining and not self.drain_words_remaining:
            self.mode = None
            self.result_array = None
            self.rr = [0] * len(self.rr)

    def is_empty(self):
        return (self.drain_words_remaining == 0 and not self.pipeline
                and not self._pending_requests and not self._staged_requests
                and all(all(item is None for item in q) for q in self.queues))

    def report(self):
        result = dict(self.bank_stats)
        result.update(occupancy_hist=dict(self.occupancy_hist),
                      wait_hist=dict(self.wait_hist), bank_services=dict(self.bank_services),
                      core_full_cycles=dict(self.core_full_cycles))
        return result
