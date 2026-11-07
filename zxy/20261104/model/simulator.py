"""1104 新设计的整网络顶层：连续 tile 计算、旧 bank 退休及跨空间tiles预计算。"""
from collections import Counter
import csv
import json
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .Accumulator import Accumulator
from .streaming_core import StreamingCore
from .tile_input_buffer import TileInputBuffer
from .reporting import cycle_breakdown, paper_cycle_breakdown, OTHER_CYCLE_REASONS, PAPER_COMPONENTS, SPLIT_WAIT_REASONS
from model.utils import Stats

SPLIT_FIFO_DEPTH = 8


def on_accept(accepted):
    """核对拍初握手，入队时完整复制的数据不再依赖源 bank。"""
    assert accepted, '拍初 ready 与 accumulator 接收结果不一致'


def psum_safe_axis(core, fully_connected):
    """计算拍初未完成贡献的最小地址，防止按最近提交位置越过驻留部分和。"""
    pending = []
    for entry in core.unit.entries:
        if entry is not None:
            columns = entry.modified - entry.retired
            if columns:
                pending.append(min(columns) if fully_connected else entry.row)
    split = core.split
    if split is not None and not split.is_finished:
        if fully_connected:
            package = split.peek_mode0_packet()
            pending.append(max(0, int(package['r'])-2) if package is not None else 0)
        else:
            rows = [split.mode0_front_row(), split.express_front_row()]
            rows += [int(item['packet'].row) for item in core.unit.express_queue]
            pending.append(max(0, min(rows)-2))
    elif core.unit.express_queue:
        pending.append(max(0, min(int(item['packet'].row) for item in core.unit.express_queue)-2))
    return min(pending, default=12)


class Cluster:
    """同一 result_array 上执行派发、计算、完成响应及神经元计算。"""
    def __init__(self, fully_connected=False, prefetch_cycles=2,
                 dispatch_lanes=None, pool_banks=8, num_cores=6,
                 bundle_buffer_depth=3, bank_lookahead_entries=3, split_fifo_depth=8, lif_units=48):
        # 派发 lane 数默认取环境变量，便于同一模型对比单 lane / 双 lane 调度。
        if dispatch_lanes is None:
            dispatch_lanes = int(os.environ.get('EVA_DISPATCH_LANES', 2))
        # 两拍由任务启动与输入接收状态实现；禁止再在 splitunit 内重复计费。
        assert prefetch_cycles == 2
        assert dispatch_lanes in (1, 2, 3)
        self.num_cores = num_cores
        if lif_units not in (48, 64):
            raise ValueError('LIF数量当前支持48或64')
        self.lif_units = lif_units
        self.pool_banks = pool_banks
        self.bundle_buffer_depth = bundle_buffer_depth
        self.bank_lookahead_entries = bank_lookahead_entries
        if split_fifo_depth not in (4, 8):
            raise ValueError('package buffer 深度当前支持四或八')
        self.split_fifo_depth = split_fifo_depth
        self.performance = Counter()
        self.drain_bank_histogram = Counter()
        self.prefetch_window_switches = 0
        self.prefetch_window_resident_peak = 0
        self.prefetch_window_trace = []
        self.prefetch_cycles = prefetch_cycles
        self.dispatch_lanes = dispatch_lanes
        self.finish_histogram = Counter()
        self.state_counts = Counter()
        self.fc = fully_connected
        self.accumulator = Accumulator(
            num_cores=self.num_cores, bank_h=12, bank_w=12, num_pus=16,
            input_width=13, buffer_depth=bundle_buffer_depth, lookahead=True,
            bank_lookahead_entries=bank_lookahead_entries,
            drain_words_per_cycle=lif_units//16)
        self.cycles = 0
        self.lif_exposed = 0
        self.modes = Counter()
        self.compute_regions = 0
        self.zero_channels = 0
        self.retirements = 0
        self.express_consecutive_cycles = 0
        self.express_issue_cycles = 0
        self.issue_cycles = 0

    def record_other(self, reason, count=1):
        """其他分类的子项互斥计数，子项之和始终等于原有其他计数。"""
        assert reason in OTHER_CYCLE_REASONS
        self.performance['other_partition_core_cycles'] += count
        self.performance[f'other_{reason}_core_cycles'] += count

    def record_split_wait(self, scope, diagnosis):
        """互斥子项与条件交集分别保存；条件交集不能作为周期相加。"""
        assert diagnosis is not None
        reason, flags = diagnosis
        assert reason in SPLIT_WAIT_REASONS
        self.performance[f'{scope}_split_wait_{reason}_core_cycles'] += 1
        for flag, present in flags.items():
            self.performance[f'{scope}_split_flag_{flag}_core_cycles'] += int(present)

    def drain_lif(self, capacity=0):
        """等待输出上下文容量；神经元计算读请求及两拍写回均由原模型推进。"""
        acc = self.accumulator
        guard = 0
        while len(acc.lif_contexts) > capacity:
            acc.begin_cycle()
            writeback_pending = not acc.is_empty()
            acc.tick()
            self.cycles += 1
            if writeback_pending:
                self.performance['accumulator_tail_only_core_cycles'] += self.num_cores
                self.record_other('accumulator_tail', self.num_cores)
                self.performance['paper_current_accumulating_sub_unit_core_cycles'] += self.num_cores
                self.performance['nine_other_core_cycles'] += self.num_cores
            else:
                self.lif_exposed += 1
                self.performance['lif_wait_core_cycles'] += self.num_cores
                self.performance['paper_LIF_core_cycles'] += self.num_cores
                self.performance['nine_LIF_core_cycles'] += self.num_cores
            guard += 1
            assert guard < 10000, '遍历result array以完成LIF计算未能排空'

    def compute(self, tile, context=None):
        """保持调度流水，计算完成后释放任务位置，独立等待空间tiles的 psum 排空。"""
        acc = self.accumulator
        conv_shape = ((12, 12) if self.fc else
                      context['scan_shape'] if context is not None else (tile.shape[1]-2, tile.shape[2]-2))
        cores = [StreamingCore(self.fc, self.pool_banks, self.split_fifo_depth, conv_shape=conv_shape)
                 for _ in range(self.num_cores)]
        epoch = context['epoch'] if context is not None else 0
        if context is not None:
            # 累加器按真实 core 对象采样发射计数；拍初安全查询仍使用只读快照。
            context['core_objects'] = tuple(cores)
        for core in cores:
            core.lif_epoch = core.pending_lif_epoch = epoch
        # 全零 tile 在测试平台发送前过滤，scheduler 不接收全零 tile。
        nonzero = np.any(tile, axis=(1, 2))
        self.zero_channels += int(len(tile) - np.count_nonzero(nonzero))
        tile = tile[nonzero]
        channel_count = len(tile)
        buffers = TileInputBuffer(tile)
        if channel_count == 0:
            # 全零 tile 不进入 scheduler，也不占用计算周期。
            return 0
        # 三级流水：SELECT(当拍登记) -> START(下一拍起核) -> SEND(送整块 tile)。
        # 与 accelerator_dispatcher 一一对应：核在 START 那一拍就能收下整块 tile 时走同拍旁路，
        # 该 lane 因此每拍都能登记新任务；核收不下时任务才落进 SEND 级等下一拍。
        stage_start = [None]*self.dispatch_lanes
        stage_send = [None]*self.dispatch_lanes
        next_channel = 0
        slots = [0]*self.num_cores
        for cycle in range(2000000):
            acc.begin_cycle()
            old_done = [core.done for core in cores]
            self.finish_histogram[sum(old_done)] += 1
            old_slots = slots[:]
            # 两个发射口的拍初握手：SEND 级同样要求本核能收下整块 tile。
            ready = [core.ready for core in cores]
            send_fire = [stage_send[lane] is not None and ready[stage_send[lane][0]]
                         for lane in range(self.dispatch_lanes)]
            start_fire = [stage_start[lane] is not None and ready[stage_start[lane][0]]
                          for lane in range(self.dispatch_lanes)]
            # START 级能被核当拍收下（同拍旁路）就不必落进 SEND 级。
            start_move = [stage_start[lane] is not None and
                          (stage_send[lane] is None or send_fire[lane])
                          for lane in range(self.dispatch_lanes)]
            start_slot_free = [stage_start[lane] is None or start_move[lane]
                               for lane in range(self.dispatch_lanes)]
            # 下一拍 SEND 级必须为空，这一拍取的整块 tile 才不会被后续读覆盖。
            send_free_next = [(not start_move[lane] or start_fire[lane]) and
                              (stage_send[lane] is None or send_fire[lane])
                              for lane in range(self.dispatch_lanes)]
            starts, loads = {}, {}
            for lane in range(self.dispatch_lanes):
                if start_fire[lane]:
                    index, channel = stage_start[lane]
                    starts[index] = channel
                    loads[index] = channel
                elif send_fire[lane]:
                    index, channel = stage_send[lane]
                    loads[index] = channel
            old_cores = tuple(SimpleNamespace(is_finished=core.is_finished,
                pending_tile=core.pending_tile, lif_epoch=epoch, pending_lif_epoch=epoch,
                has_old_banks=getattr(core, 'has_old_banks', False),
                psum_safe_axis=psum_safe_axis(core, self.fc))
                for core in cores)
            # 神经元计算安全性使用拍初 core 状态，不能看本拍提前更新的完成状态。
            if context is not None and context in acc.lif_contexts:
                acc.set_lif_progress(cores, dispatched=next_channel >= channel_count,
                                     early=True, epoch=epoch, during_compute=False)
            any_computation = False
            pool_empty_before = all(core.unit.idle for core in cores)
            for index, core in enumerate(cores):
                start = index in starts
                load = index in loads
                if start and core.is_finished:
                    acc.start_core_task(index, epoch)
                free = len(acc._free_at_start[index])
                epoch_ready = acc._accept_epoch is None or acc._accept_epoch == epoch
                old_channel = core.channel
                entries_before = [(entry.row, entry.current_tile) for entry in core.unit.entries
                                  if entry is not None and entry.modified - entry.retired]
                computation_finished = ((core.split is None or core.split.is_finished)
                                        and not core.unit.express_queue)
                frontier_before = core.unit.front
                dispatch_channel = starts.get(index, loads.get(index, 0))
                result = core.step(start, load, buffers.read(dispatch_channel) if load else None, dispatch_channel,
                    (epoch_ready and free > 0, epoch_ready and free > 1))
                if load:
                    buffers.release_input(dispatch_channel)
                if core.channel != old_channel:
                    acc.start_core_task(index, epoch)
                self.modes[0] += int(result['normal_fire'])
                if result['express_fire']:
                    self.modes[int(result['express_mode'])] += 1
                computing = result['compute_active']
                allocation_stall = result['normal_alloc_stall'] or result['express_alloc_stall']
                any_computation |= computing
                self.performance['compute_active_core_cycles'] += int(computing)
                self.performance['allocation_stall_core_cycles'] += int(allocation_stall)
                self.performance['allocation_stall_without_compute_core_cycles'] += int(allocation_stall and not computing)
                old_resident = any(not current for _, current in entries_before)
                self.performance['compute_with_old_bank_resident_core_cycles'] += int(computing and old_resident)
                self.performance['compute_with_old_retirement_core_cycles'] += int(
                    computing and any(item['old_tile'] for item in result['submissions']))
                self.performance['next_space_backpressure_core_cycles'] += int(not epoch_ready and bool(entries_before))
                pure_drain = bool(not computing and computation_finished and entries_before)
                self.performance['pure_pool_drain_core_cycles'] += int(pure_drain)
                self.performance['final_pool_drain_core_cycles'] += int(
                    pure_drain and next_channel >= channel_count and core.pending is None)
                category = ('compute_partition_core_cycles' if computing else
                            'allocation_partition_core_cycles' if allocation_stall else
                            'drain_partition_core_cycles' if pure_drain else 'other_partition_core_cycles')
                # 论文分类与开发分类分别计数：前者保留无包、分配和退休等待的归因顺序。
                retirement_blocked = bool(result['retire_candidate_ports']
                    and result['retire_blocked_ports'] == result['retire_candidate_ports'])
                scheduler_wait = bool(not result['split_present']
                    and not result['express_busy_before'] and not entries_before)
                paper_category = ('current_accumulating_sub_unit' if computing else
                    'split_unit' if result['split_supply_stall'] else
                    'current_accumulating_sub_unit' if allocation_stall else
                    'accumulator' if retirement_blocked else
                    'scheduler' if scheduler_wait else 'current_accumulating_sub_unit')
                self.performance[f'paper_{paper_category}_core_cycles'] += 1
                nine_category = ('pe_ex_compute' if computing else
                    'splitunit' if paper_category == 'split_unit' else
                    'allocation_stall' if allocation_stall else
                    'accumulator' if paper_category == 'accumulator' else
                    'pure_drain' if pure_drain else
                    'scheduler_idle' if paper_category == 'scheduler' else 'other')
                self.performance[f'nine_{nine_category}_core_cycles'] += 1
                if paper_category == 'split_unit':
                    self.record_split_wait('paper', result['split_wait_diagnosis'])
                if category == 'other_partition_core_cycles':
                    reason = ('split_unit' if result['split_supply_stall'] else
                        'accumulator_backpressure' if retirement_blocked else
                        'tile_handoff' if result['tile_handoff'] else
                        'row_dependency' if result['row_dependency_blocked'] else
                        'scheduler' if scheduler_wait else 'control')
                    self.record_other(reason)
                    if reason == 'split_unit':
                        self.record_split_wait('other', result['split_wait_diagnosis'])
                else:
                    self.performance[category] += 1
                if pure_drain:
                    for row, current in entries_before:
                        label = ('old_tile' if not current else 'y_lt_r_minus_2' if row < frontier_before-2
                                 else 'y_eq_r_minus_2' if row == frontier_before-2
                                 else 'y_eq_r_minus_1' if row == frontier_before-1
                                 else 'y_eq_r' if row == frontier_before else 'other')
                        self.drain_bank_histogram[label] += 1
                for submission in result['submissions']:
                    values = [[v & 8191 for v in lane] for lane in submission['values']]
                    # 新电流累积单元已给出物理 bank；全连接旧接口还会转置一次，故先反向转换。
                    rows, columns = ([submission['bank']]*len(submission['columns']),
                                     list(submission['columns']))
                    if self.fc:
                        rows, columns = columns, rows
                    acc.request_bundle(index, rows, columns, values,
                        kind='fc' if self.fc else 'conv', on_result=on_accept)
                    self.retirements += 1
            # 对安全地址的查询发生在更新前状态的副本上，避免 core_done 提前影响读出。
            if context is not None and context in acc.lif_contexts:
                context['cores'] = old_cores
            acc._lif_during_compute = any_computation
            self.performance['accumulator_tail_only_core_cycles'] += self.num_cores * int(
                pool_empty_before and not any_computation and not any(old_slots) and not acc.is_empty())
            acc.tick()
            for index in range(self.num_cores):
                if old_slots[index] and old_done[index]:
                    slots[index] -= 1
            # 本拍启动的任务占用 slot；与同拍完成按原有口径覆盖计数。
            for core_index in starts:
                slots[core_index] = (1 if old_done[core_index] and old_slots[core_index] == 1
                                     else old_slots[core_index] + 1)
            # 流水前移：只有没能同拍被收下的 START 级任务进入下一拍的 SEND 级；
            # 若 SEND 级被占住，该 lane 的 START 级原地保持，下一拍才能再看是否能让位。
            advanced = [None]*self.dispatch_lanes
            kept = [None]*self.dispatch_lanes
            for lane in range(self.dispatch_lanes):
                if stage_start[lane] is None:
                    continue
                if start_move[lane]:
                    if not start_fire[lane]:
                        advanced[lane] = stage_start[lane]
                else:
                    kept[lane] = stage_start[lane]
            stage_send = advanced
            buffers.refill()
            # SELECT：每拍最多登记 dispatch_lanes 个任务，核与输入通道都不重复；
            # 该 lane 的 START 位与下一拍的 SEND 位必须都空着。
            reserved = {entry[0] for entry in stage_send + kept if entry is not None}
            stage_start = kept
            for lane in range(self.dispatch_lanes):
                if next_channel >= channel_count or not buffers.available(next_channel):
                    break
                if not (start_slot_free[lane] and send_free_next[lane]):
                    continue
                choice = next((core_index for core_index in range(self.num_cores)
                               if core_index not in reserved
                               and slots[core_index] < 2 and cores[core_index].ready), None)
                if choice is None:
                    break
                stage_start[lane] = (choice, next_channel)
                reserved.add(choice)
                next_channel += 1
            for core in cores:
                self.state_counts[core.state] += 1
                self.state_counts['pending' if core.pending is not None else 'no_pending'] += 1
            self.cycles += 1
            if (next_channel >= channel_count and all(x is None for x in stage_start + stage_send)
                    and not any(slots)
                    and all(core.unit.idle and core.is_finished for core in cores)):
                if context is not None:
                    acc.finish_backend(epoch)
                self.prefetch_window_switches += max(0, buffers.window_count-1)
                self.prefetch_window_resident_peak = max(self.prefetch_window_resident_peak, buffers.resident_peak)
                self.prefetch_window_trace.extend(buffers.trace)
                self.compute_regions += 1
                # 发射统计是各核的累计量，只能在整段结束后汇总一次，不能逐拍累加。
                for core in cores:
                    self.express_consecutive_cycles += core.statistics['express_consecutive_cycles']
                    self.express_issue_cycles += core.statistics['express_issue_cycles']
                    self.issue_cycles += core.statistics['issue_cycles']
                return cycle+1
        raise RuntimeError('最新模块集成未能完成任务')

    def execute(self, tiles, shapes, with_lif=True):
        """有界输入窗口连续供给；psum 排空后允许下一空间tiles预计算。"""
        for tile, shape in zip(tiles, shapes):
            # 全零 tile 在进入物理任务分块前过滤，保证 scheduler 看不到零 tile。
            zero_mask = ~np.any(tile, axis=(1, 2))
            self.zero_channels += int(np.count_nonzero(zero_mask))
            tile = tile[~zero_mask]
            if len(tile) == 0:
                # 整块全零的空间 tile 同样不发送，不推进 LIF 上下文。
                continue
            self.drain_lif(capacity=1)
            # 无论是否执行 LIF 都建立上下文：with_lif=False 只跳过膜电位扫描，数据流与退休照常。
            context = self.accumulator.begin_lif_context(with_lif=with_lif, **shape)
            context['scan_shape'] = shape.get('conv_shape', (12, 12))
            self.compute(tile, context)
        self.drain_lif()
        return self


class OutProductSimulator:
    """EVA 端到端接口：真实逐拍计算，代表 Cout 组缩放和原片外访存重叠。"""
    def __init__(self, num_cores=6, num_pus=16, oh=12, ow=12,
                 split_fifo_depth=SPLIT_FIFO_DEPTH, psum_pool_rows=8,
                 bundle_buffer_depth=3, bank_lookahead_entries=3, lif_units=48, **kwargs):
        assert isinstance(num_cores, int) and num_cores > 0
        assert (num_pus, oh, ow) == (16, 12, 12)
        assert split_fifo_depth in (4, 8)
        self.split_fifo_depth = split_fifo_depth
        if lif_units not in (48, 64):
            raise ValueError('LIF数量当前支持48或64')
        self.lif_units = lif_units
        assert psum_pool_rows in (8, 9)
        self.pool_banks = psum_pool_rows
        if not 1 <= bank_lookahead_entries <= bundle_buffer_depth:
            raise ValueError('仲裁候选数必须位于一和 bundle buffer 深度之间')
        self.bundle_buffer_depth = bundle_buffer_depth
        self.bank_lookahead_entries = bank_lookahead_entries
        assert kwargs.get('retire_packets_per_cycle', 2) == 2
        # 用户历史端到端配置为逐任务两拍；当前集成通过显式状态推进实现。
        assert kwargs.get('prefetch_cycles', 2) == 2
        self.num_cores, self.num_pus, self.oh, self.ow = num_cores, num_pus, oh, ow
        self.bandwidth, self.spike_size, self.weight_size = 1024, 2, 8
        self.GBsize_spike = 128*14*14*2
        self.GBsize_weight = 128*16*9*8
        self.calls = 0
        self.current_case = 0
        self.event_file = None
        self.records = []

    def set_enabled_modes(self, modes):
        assert tuple(modes) == (0, 1, 2)

    def configure_reporting(self, output, metadata):
        """保存逐调用证据及实际源文件摘要，防止把其他版本结果误作本设计结果。"""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.event_file = output/'calls.jsonl'
        if self.event_file.exists():
            raise FileExistsError(f'已有逐调用结果，请选择新的 --output-root：{output}')
        sources = list(Path(__file__).parent.glob('*.py'))
        runner = Path(__file__).resolve().parents[3]/'model/test_outerproduct_simulator/simulator_runner.py'
        sources.append(runner)
        manifest = dict(design='1104_streaming', simulator=str(Path(__file__).resolve()),
            configuration=metadata, num_cores=self.num_cores, pool_banks=self.pool_banks,
            bundle_buffer_depth=self.bundle_buffer_depth, bank_lookahead_entries=self.bank_lookahead_entries,
            split_fifo_depth=self.split_fifo_depth,
            lif_units=self.lif_units,
            dispatch_lanes=int(os.environ.get('EVA_DISPATCH_LANES', 2)),
            baseline_simulation=not metadata.get('eva_only', False),
            sources={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
        (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    def begin_test_case(self, index):
        self.current_case = index

    def finish_reporting(self):
        """合计原始计数后计算比例，不能平均各层百分比。"""
        if self.event_file is None:
            return
        summary = {}
        for kind in ('conv', 'fc', 'all'):
            selected = [record for record in self.records if kind == 'all' or record['kind'] == kind]
            counters, histogram = Counter(), Counter()
            for record in selected:
                counters.update(record['retirement_stats']['counters'])
                histogram.update(record['retirement_stats']['drain_bank_wait_histogram'])
            denominator = sum(record['retirement_stats']['denominator_core_cycles'] for record in selected)
            summary[kind] = dict(calls=len(selected), total_cycles=sum(record['total_cycles'] for record in selected),
                backend_cycles=sum(record['backend_cycles'] for record in selected),
                denominator_core_cycles=denominator, counters=dict(counters),
                drain_bank_wait_histogram=dict(histogram),
                ratios={key: value/denominator if denominator else 0 for key, value in counters.items()})
            components = {key: sum(record['paper_cycle_breakdown'][key] for record in selected)
                          for key in PAPER_COMPONENTS}
            assert math.isclose(sum(components.values()), summary[kind]['total_cycles'], abs_tol=1e-6, rel_tol=1e-12)
            other = counters['other_partition_core_cycles']
            summary[kind]['other_breakdown'] = dict(
                denominator_core_cycles=other,
                counters={reason: counters[f'other_{reason}_core_cycles'] for reason in OTHER_CYCLE_REASONS},
                ratios={reason: counters[f'other_{reason}_core_cycles']/other if other else 0
                        for reason in OTHER_CYCLE_REASONS})
            summary[kind]['cycles_breakdown'] = components
            summary[kind]['split_wait_breakdown'] = {}
            for scope, parent in (('paper', 'paper_split_unit_core_cycles'), ('other', 'other_split_unit_core_cycles')):
                count = counters[parent]
                values = {reason: counters[f'{scope}_split_wait_{reason}_core_cycles'] for reason in SPLIT_WAIT_REASONS}
                assert sum(values.values()) == count
                summary[kind]['split_wait_breakdown'][scope] = dict(
                    denominator_core_cycles=count, counters=values,
                    ratios={reason: value/count if count else 0 for reason, value in values.items()},
                    flags={key: value for key, value in counters.items() if key.startswith(f'{scope}_split_flag_')})
            total = summary[kind]['total_cycles']
            summary[kind]['cycles_breakdown_ratios'] = {key: value/total if total else 0
                                                       for key, value in components.items()}
        output = self.event_file.parent
        (output/'performance_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        lines = [f'比例分母：对应通路合计 backend 周期 × {self.num_cores}，包含未隐藏的 LIF 周期。', '']
        for kind, result in summary.items():
            lines.append(f"{kind}: total_cycles={result['total_cycles']}, denominator={result['denominator_core_cycles']}")
            lines.extend(f"  {key}: {value} ({result['ratios'][key]:.6%})" for key, value in result['counters'].items())
        (output/'performance_summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        with (output/'cycles_breakdown_calls.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=('test_case', 'call', 'kind', *PAPER_COMPONENTS, 'total_cycles'))
            writer.writeheader()
            for record in self.records:
                writer.writerow(dict(test_case=record['test_case'], call=record['call'], kind=record['kind'],
                                     **record['paper_cycle_breakdown'], total_cycles=record['total_cycles']))
        (output/'COMPLETE.json').write_text(json.dumps(dict(status='pass', design='1104_streaming', calls=len(self.records))), encoding='utf-8')

    def _finish(self, engine, scale, reads, writes, initial, shape, kind, extra=None):
        stats = Stats()
        backend = engine.cycles*scale
        lif_exposed = engine.lif_exposed*scale
        lif_reads = engine.accumulator.bank_stats['lif_read_cycles']*scale
        hidden = engine.accumulator.bank_stats['lif_hidden_read_cycles']*scale
        tails = sum(2 for ctx in engine.accumulator.completed_lif if ctx['scan'].beats)*scale
        if extra is not None:
            # 全连接末组仅有一行或两行时，重新推进时序，保留其真实可见开销。
            backend += extra.cycles-engine.cycles
            lif_exposed += extra.lif_exposed-engine.lif_exposed
            lif_reads += extra.accumulator.bank_stats['lif_read_cycles']-engine.accumulator.bank_stats['lif_read_cycles']
            hidden += extra.accumulator.bank_stats['lif_hidden_read_cycles']-engine.accumulator.bank_stats['lif_hidden_read_cycles']
        if engine.compute_regions == 0:
            # 整次调用的输入均被跳零过滤，没有权重load、结果写回或访存启动。
            assert backend == 0
            reads = writes = initial = 0
        dram = math.ceil((reads+writes)*scale/self.bandwidth)
        total = max(backend, dram)+initial
        stats.total_cycles, stats.compute_cycles = total, backend
        stats.reads['dram'], stats.writes['dram'] = reads*scale, writes*scale
        stats.cycle_breakdown = cycle_breakdown(backend, total, lif_exposed, dram, initial, lif_reads, hidden, tails)
        if kind == 'conv':
            stats.conv_mode_split_counts = dict(engine.modes)
        self.compute_cycles, self.total_cycles, self.global_stats = backend, total, stats
        record = dict(test_case=self.current_case, call=self.calls, kind=kind, shape=shape,
            scale=scale, compute_regions=engine.compute_regions, zero_channels=engine.zero_channels,
            retired_bundles=engine.retirements, mode_counts=dict(engine.modes),
            prefetch_cycles=engine.prefetch_cycles, finish_histogram=dict(engine.finish_histogram),
            extra_finish_histogram=dict(extra.finish_histogram) if extra is not None else None,
            reads_bits=reads*scale, writes_bits=writes*scale,
            express_consecutive_cycles=engine.express_consecutive_cycles,
            express_issue_cycles=engine.express_issue_cycles,
            issue_cycles=engine.issue_cycles, **stats.cycle_breakdown)
        counters = Counter({key: value*scale for key, value in engine.performance.items()})
        histogram = Counter({key: value*scale for key, value in engine.drain_bank_histogram.items()})
        if extra is not None:
            counters.subtract(engine.performance)
            counters.update(extra.performance)
            histogram.subtract(engine.drain_bank_histogram)
            histogram.update(extra.drain_bank_histogram)
        denominator = backend * self.num_cores
        partition = sum(counters[key] for key in ('compute_partition_core_cycles',
            'allocation_partition_core_cycles', 'drain_partition_core_cycles',
            'other_partition_core_cycles', 'lif_wait_core_cycles'))
        assert partition == denominator, (partition, denominator)
        nine_names = ('LIF', 'scheduler_idle', 'splitunit', 'accumulator',
                      'pe_ex_compute', 'allocation_stall', 'pure_drain', 'other')
        assert sum(counters[f'nine_{name}_core_cycles'] for name in nine_names) == denominator
        record['nine_cycle_breakdown'] = {name: counters[f'nine_{name}_core_cycles']/self.num_cores
                                          for name in nine_names}
        record['nine_cycle_breakdown']['DRAM'] = stats.cycle_breakdown['dram_exposed_cycles']
        assert math.isclose(sum(record['nine_cycle_breakdown'].values()), total, abs_tol=1e-6)
        record['paper_cycle_breakdown'] = paper_cycle_breakdown(counters, self.num_cores, stats.cycle_breakdown)
        record['retirement_stats'] = dict(denominator_core_cycles=denominator,
            denominator_basis='backend_cycles_times_num_cores_including_exposed_lif',
            counters=dict(counters), drain_bank_wait_histogram=dict(histogram),
            ratios={key: value/denominator if denominator else 0 for key, value in counters.items()})
        record['input_window_switches'] = engine.prefetch_window_switches
        record['input_resident_tile_peak'] = engine.prefetch_window_resident_peak
        record['pool_banks'] = self.pool_banks
        record['bundle_buffer_depth'] = self.bundle_buffer_depth
        record['bank_lookahead_entries'] = self.bank_lookahead_entries
        record['split_fifo_depth'] = self.split_fifo_depth
        record['lif_units'] = self.lif_units
        record['lif_points_per_cycle'] = self.lif_units//self.num_pus
        record['dispatch_lanes'] = engine.dispatch_lanes
        record['accumulator_statistics_representative'] = engine.accumulator.report()
        record['design'] = '1104_streaming'
        print('分配与排空统计 ' + json.dumps(record['retirement_stats'], ensure_ascii=False), flush=True)
        # 沿用既有六类日志接口，历史提取入口无需猜测新字段或将缺失值补零。
        print('\n[Conv 路径 Core 平均周期占比]' if kind == 'conv' else '\n[全连接路径计算核心互斥周期]')
        print(f'分母: {backend} cycles x {self.num_cores} cores = {denominator} core-cycles')
        labels = (('scheduler', '空闲/调度'),
                  ('split_unit', 'SplitUnit 供数不足' if kind == 'conv' else '取不到 mode0 包'),
                  ('current_accumulating_sub_unit', '电流累积单元'),
                  ('accumulator', 'Accumulator 反压等待'),
                  ('LIF', 'Accumulator/LIF 排空' if kind == 'conv' else '泄漏积分发放等待'))
        for key, label in labels:
            print(f"{label}: {counters[f'paper_{key}_core_cycles']} core-cycles")
        print(f"总周期分解: {total} = 正常计算 {stats.cycle_breakdown['normal_compute_cycles']}"
              f" + 未隐藏 LIF {lif_exposed} + 未隐藏 DRAM {total-backend}", flush=True)
        self.records.append(record)
        if self.event_file is not None:
            with Path(self.event_file).open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False)+'\n')
        self.calls += 1
        print(f'最新 EVA：总周期 {total}；计算及神经元计算 {backend}；未隐藏神经元计算 {lif_exposed}；片外访存 {total-backend}', flush=True)
        print('状态计数 ' + json.dumps({k: engine.state_counts[k] for k in sorted(engine.state_counts)},
                                      ensure_ascii=False), flush=True)
        ratio = (engine.express_consecutive_cycles / engine.issue_cycles
                 if engine.issue_cycles else 0.0)
        print('连续express拍数 %d；express发射拍数 %d；任一发射拍数 %d；连续比例 %.2f%%'
              % (engine.express_consecutive_cycles, engine.express_issue_cycles,
                 engine.issue_cycles, ratio*100), flush=True)
        return None, stats

    def run_convolution(self, input_tensor, kernel_tensor, padding=1, stride=1, groups=1, spike_in_buffer=True):
        data = input_tensor.detach().cpu().numpy()
        if data.ndim == 4:
            data = data[:, None]
        T, B, Cin, height, width = data.shape
        Cout, _, kh, kw = kernel_tensor.shape
        assert (kh, kw) == (3, 3) and stride in (1, (1, 1))
        ph, pw = padding if isinstance(padding, tuple) else (padding, padding)
        oh, ow = height+2*ph-2, width+2*pw-2
        padded = np.pad(data, ((0, 0), (0, 0), (0, 0), (ph, ph), (pw, pw)))
        cin, cout = Cin//groups, Cout//groups
        wave, scale = min(cout, 16), math.ceil(cout/16)
        tiles, shapes = [], []
        reads = writes = 0
        for batch in range(B):
            for group in range(groups):
                for top in range(0, oh, 12):
                    for left in range(0, ow, 12):
                        vh, vw = min(12, oh-top), min(12, ow-left)
                        weight_bits = cin*kh*kw*self.weight_size*wave
                        reads += weight_bits*(1 if weight_bits <= self.GBsize_weight else T)
                        for time in range(T):
                            cut = padded[time, batch, group*cin:(group+1)*cin, top:top+vh+2, left:left+vw+2]
                            nonzero = int(np.count_nonzero(np.any(cut, axis=(1, 2))))
                            if not spike_in_buffer:
                                reads += nonzero*(vh+2)*(vw+2)*self.spike_size
                            # 有效输出保持原层尺寸，物理计算输入补齐十四乘十四。
                            tiles.append(np.pad(cut, ((0, 0), (0, 14-cut.shape[1]), (0, 14-cut.shape[2]))))
                            shapes.append(dict(conv_shape=(vh, vw), last_t=time == T-1,
                                               channel_mask=(1 << wave)-1))
                            writes += vh*vw*wave
        engine = Cluster(pool_banks=self.pool_banks, num_cores=self.num_cores,
                         bundle_buffer_depth=self.bundle_buffer_depth,
                         bank_lookahead_entries=self.bank_lookahead_entries,
                         split_fifo_depth=self.split_fifo_depth, lif_units=self.lif_units).execute(tiles, shapes)
        initial_bits = (min(14, height)*min(14, width)*self.spike_size+9*self.weight_size)*min(self.num_cores, Cin)
        initial = math.ceil(initial_bits/self.bandwidth)+2
        return self._finish(engine, scale, reads, writes, initial,
                            dict(input=list(data.shape), cout=Cout, groups=groups), 'conv')

    def run_fc(self, input_tensor, weight_matrix, *, with_lif=True):
        data = input_tensor.detach().cpu().numpy()
        M, K = data.shape
        assert K == weight_matrix.shape[0]
        Cout = weight_matrix.shape[1]
        wave, scale = min(Cout, 48), math.ceil(Cout/48)
        weight_bits = K*self.weight_size*wave
        repeat_weights = weight_bits >= self.GBsize_weight
        reads = 0 if repeat_weights else weight_bits
        writes = 0
        tiles, shapes = [], []
        # 保持 lookahead 的矩阵展平和四十八行分块，不在本次比较中另改时间步映射。
        for start in range(0, M, 48):
            cut = data[start:start+48]
            count = len(cut)
            padded = np.pad(cut, ((0, 48-count), (0, (-K) % 3)))
            tile = padded.reshape(48, -1, 3).transpose(1, 0, 2)
            nonzero = int(np.count_nonzero(np.any(tile, axis=(1, 2))))
            reads += nonzero*count*3*self.spike_size
            if repeat_weights:
                reads += nonzero*9*wave*self.weight_size
            writes += count*wave
            tiles.append(tile)
            shapes.append(dict(fc_m=count, fc_pe_rows=min(3, wave), last_t=True,
                               channel_mask=(1 << wave)-1))
        engine = Cluster(True, pool_banks=self.pool_banks, num_cores=self.num_cores,
                         bundle_buffer_depth=self.bundle_buffer_depth,
                         bank_lookahead_entries=self.bank_lookahead_entries,
                         split_fifo_depth=self.split_fifo_depth, lif_units=self.lif_units).execute(tiles, shapes, with_lif=with_lif)
        extra = None
        last_rows = min(3, Cout-(scale-1)*48)
        if with_lif and scale > 1 and last_rows < min(3, wave):
            extra = Cluster(True, pool_banks=self.pool_banks, num_cores=self.num_cores,
                         bundle_buffer_depth=self.bundle_buffer_depth,
                         bank_lookahead_entries=self.bank_lookahead_entries,
                         split_fifo_depth=self.split_fifo_depth, lif_units=self.lif_units).execute(tiles, [dict(shape, fc_pe_rows=last_rows) for shape in shapes])
        initial_bits = (14*14*self.spike_size+9*wave*self.weight_size)*min(self.num_cores, K//3)
        initial = math.ceil(initial_bits/self.bandwidth)+2
        return self._finish(engine, scale, reads, writes, initial,
                            dict(input=list(data.shape), cout=Cout, with_lif=with_lif), 'fc', extra)
