# FC 模式，core映射为不同的k，pe映射为不同的cout

import torch
from .reference_accumulator import Accumulator
from .core import Core
from model.utils import ceil_a_by_b, Stats
import collections

CORE_CYCLE_LABELS = {
    'compute': 'PE 计算',
    'express': 'Express 独立执行/等待',
    'frontend': 'SplitUnit 供数不足',
    'psum_alloc': 'Psum 分配停顿',
    'backpressure': 'Accumulator 反压等待',
    'writeback': 'Psum 写回排空',
    'lif_drain': 'Accumulator/LIF 排空',
    'idle': '空闲/调度',
}


class OutProductSimulator:
    def __init__(
        self,
        num_cores=8,
        num_pus=64,
        oh=32,
        ow=32,
        fetch_rows=4,
        split_fifo_depth=16,
        num_sub_banks=None,
        retire_column=3,
        enable_reduce_tree=True,
        psum_pool_rows=8,
        enabled_modes=(0, 1, 2),
        num_split=2,
        num_reduce_trees=None,
        record_tree_trace=False,
        psum_alloc_mode="halfrow",
        retire_packets_per_cycle=2,
        retire_iface="scatter",
        retire_pipeline_stages=0,
    ):
        self.num_cores = num_cores
        self.num_pus = num_pus
        self.GBsize_MP = 0
        self.GBsize_weight = 0
        self.bandwidth = 2048
        self.weight_size = 8
        self.mp_size = 16
        self.lif_num = 3 * self.num_pus

        self.ow = ow
        self.oh = oh
        self.kernel_size = 3
        self.fetch_rows = fetch_rows  
        self.split_fifo_depth = split_fifo_depth  
        # Accepted for CLI compatibility. The accumulator is modeled as a
        # point-level register array and does not use sub-banks.
        self.num_sub_banks = None if num_sub_banks is None else int(num_sub_banks)
        self.retire_column = int(retire_column)
        self.enable_reduce_tree = bool(enable_reduce_tree)
        self.num_reduce_trees = None if num_reduce_trees is None else int(num_reduce_trees)
        if self.num_reduce_trees is not None and self.num_reduce_trees < 0:
            raise ValueError("num_reduce_trees must be nonnegative or None")
        self.record_tree_trace = bool(record_tree_trace)
        # CONV 每个 T 后执行一次 LIF 扫描：每拍三个不同 bank 各一表项。
        # 默认 12×12、16 PU、48 LIF 是 48 拍扫描，另计 2 拍流水排空。
        # 排空的墙钟拍数（与按 core 累加的 lif_drain 计数区分开）
        self.drain_wall_cycles = 0
        self.psum_pool_rows = int(psum_pool_rows)
        self.enabled_modes = tuple(sorted(int(mode) for mode in enabled_modes))
        self.num_split = max(1, int(num_split))
        # Psum allocation granularity: "point" (historical point array) or
        # "halfrow" (one slot per contiguous half of an output row). Conv only;
        # FC shares CONV allocation, but retires three rows at one column.
        self.psum_alloc_mode = str(psum_alloc_mode)
        if self.psum_alloc_mode not in ("point", "halfrow"):
            raise ValueError(f"unknown psum allocation mode: {psum_alloc_mode!r}")
        # Retire read ports per core per cycle; None = unlimited.
        self.retire_packets_per_cycle = (
            None if retire_packets_per_cycle is None
            else int(retire_packets_per_cycle))
        self.retire_iface = str(retire_iface)
        if self.retire_iface not in ("scatter", "positional"):
            raise ValueError(f"unknown retire interface: {retire_iface!r}")
        # ---- retire 流水（默认关，基线逐位不变）----
        self.retire_pipeline_stages = int(retire_pipeline_stages)
        if self.retire_pipeline_stages not in (0, 1):
            raise ValueError("retire_pipeline_stages must be 0 or 1")

        self.cores = [
            Core(
                kernel_size=self.kernel_size,
                num_pus=num_pus,
                psum_w=ow,
                bank_h=oh,
                fetch_rows=fetch_rows,
                split_fifo_depth=split_fifo_depth,
                retire_column=self.retire_column,
                psum_pool_rows=self.psum_pool_rows,
                enabled_modes=self.enabled_modes,
                num_split=self.num_split,
                psum_alloc_mode=self.psum_alloc_mode,
                retire_packets_per_cycle=self.retire_packets_per_cycle,
                retire_iface=self.retire_iface,
            )
            for _ in range(num_cores)
        ]

        self.pe_cycles = torch.zeros((num_cores, 3))  
        self.bitstream_lengths = []  
        self.hazard_num = 0  
        

    def set_enabled_modes(self, enabled_modes):
        modes = tuple(sorted({int(mode) for mode in enabled_modes}))
        if 0 not in modes or any(mode not in (0, 1, 2) for mode in modes):
            raise ValueError("enabled_modes must contain mode 0 and only use modes 0, 1, and 2")
        self.enabled_modes = modes
        for core in self.cores:
            core.enabled_modes = modes
            core.split_unit.enabled_modes = frozenset(modes)
            for current_split in core.split_unit.splits:
                current_split.enabled_modes = frozenset(modes)

    def _reset_performance_counters(self):
        self.pe_cycles.zero_()
        self.tile_assignment_counts = [0] * self.num_cores
        self.all_zero_channel_counts = 0
        self.fc_all_zero_tiles = 0
        self.retire_trace_cycle = 0
        self.conv_core_cycle_counts = {
            core: collections.Counter({state: 0 for state in CORE_CYCLE_LABELS})
            for core in self.cores
        }
        for core in self.cores:
            core.reset_performance_counters()
            core.is_finished = True

    def _publish_performance_counters(
        self,
        representative_compute_cycles,
        conv_cout_scale=None,
    ):
        for index, core in enumerate(self.cores):
            self.pe_cycles[index] = core.pe_cycles
        self.representative_compute_cycles = representative_compute_cycles
        self.global_stats.representative_compute_cycles = representative_compute_cycles
        self.global_stats.pe_cycles = self.pe_cycles.clone()
        if conv_cout_scale is not None:
            # 原始计数覆盖代表性 Cout-group；各项与 compute_cycles 使用相同
            # 倍率，不能将未放大的 Core 停顿除以已经放大的层周期。
            per_core = []
            representative = collections.Counter()
            for core, counts in self.conv_core_cycle_counts.items():
                assert sum(counts.values()) == representative_compute_cycles
                assert counts['frontend'] == core.frontend_stalls
                assert counts['psum_alloc'] == core.psum_buffer_alloc_stalls
                assert counts['compute'] == core.compute_issue_cycles
                representative.update(counts)
                per_core.append({key: value * conv_cout_scale for key, value in counts.items()})
            scaled = {key: value * conv_cout_scale for key, value in representative.items()}
            denominator = self.compute_cycles * len(per_core)
            assert sum(scaled.values()) == denominator
            self.global_stats.conv_core_cycles = scaled
            self.global_stats.conv_core_cycles_per_core = per_core
            self.global_stats.conv_core_cycles_representative = dict(representative)
            self.global_stats.conv_core_cycle_denominator = denominator
            self.global_stats.conv_core_cycle_scale = conv_cout_scale
            self.global_stats.conv_core_cycle_ratios = {
                key: value / denominator if denominator else 0.0
                for key, value in scaled.items()
            }
            print('\n[Conv 路径 Core 平均周期占比]')
            print(f'  分母: {self.compute_cycles} cycles x {len(per_core)} cores = {denominator} core-cycles')
            for key, label in CORE_CYCLE_LABELS.items():
                print(f'  {label}: {scaled[key]} core-cycles ({self.global_stats.conv_core_cycle_ratios[key]:.2%})')
        self.global_stats.psum_alloc_mode = self.psum_alloc_mode
        self.global_stats.psum_point_capacity_per_core = self.psum_pool_rows * ((self.ow + 1) // 2)
        # Under half-row allocation the point count is unchanged but the pool only
        # has psum_pool_rows slots, so the binding limit is slots, not points.
        self.global_stats.psum_units_per_core = (
            self.psum_pool_rows if self.psum_alloc_mode == "halfrow"
            else self.global_stats.psum_point_capacity_per_core)
        self.global_stats.row_barrier_stall_cycles = sum(c.row_barrier_stalls for c in self.cores)
        self.global_stats.express_unit_stats = {
            'issued_packets': sum(c.express_issue_cycles for c in self.cores),
            'busy_cycles': sum(c.express_unit.express_unit_busy_cycles for c in self.cores),
            'packets_done': sum(c.express_unit.express_packets_done for c in self.cores),
            'write_cols': sum(c.express_unit.express_write_cols for c in self.cores),
            'pe_conflict_stalls': sum(c.express_unit.express_pe_conflict_stalls for c in self.cores),
            'queue_full_stalls': sum(c.express_queue_full_stalls for c in self.cores),
        }
        # issue 空转：本拍允许发射（已过 row barrier）却取不到包，导致 PE 或
        # Express 空转的拍数。按 core 累加，与 conv_core_cycle_counts 同口径
        # （每个墙钟拍里每个 core 各记一次），因此可与 Core 周期占比表用
        # 同一个分母（compute_cycles × num_cores）比较。
        # 计数器是代表性 Cout 组的原始值，须与 conv_core_cycles 一样按
        # conv_cout_scale 放大，才能和 compute_cycles × num_cores 的分母对齐。
        scale = 1 if conv_cout_scale is None else conv_cout_scale
        starved_mode0 = sum(c.issue_starved_mode0 for c in self.cores) * scale
        starved_express = sum(c.issue_starved_express for c in self.cores) * scale
        starved_both = sum(c.issue_starved_both for c in self.cores) * scale
        self.global_stats.issue_starvation_stats = {
            'mode0_starved': starved_mode0,
            'express_starved': starved_express,
            'both_starved': starved_both,
            'mode0_only': starved_mode0 - starved_both,
            'express_only': starved_express - starved_both,
            'any_starved': starved_mode0 + starved_express - starved_both,
        }
        starve_denom = self.compute_cycles * len(self.cores)
        self.global_stats.issue_starvation_ratios = {
            key: (value / starve_denom if starve_denom else 0.0)
            for key, value in self.global_stats.issue_starvation_stats.items()
        }
        print('\n[issue 空转统计]')
        print(f'  PE 取不到 mode0 包:               {starved_mode0} core-cycles '
              f'({self.global_stats.issue_starvation_ratios["mode0_starved"]:.2%})')
        print(f'  Express 取不到 mode1/2 包且空闲:  {starved_express} core-cycles '
              f'({self.global_stats.issue_starvation_ratios["express_starved"]:.2%})')
        print(f'  两者同拍都空:                     {starved_both} core-cycles '
              f'({self.global_stats.issue_starvation_ratios["both_starved"]:.2%})')

        # ------------------------------------------------------------------
        # Psum 停顿理由 + retire 利用率（按 core 分解）
        # ------------------------------------------------------------------
        # 这些计数来自各 Core 的 PsumPool，天然是 per-core 的；池每个 tile 重建，
        # Core 换池时已把旧池计数折进累计器，所以这里是整层的全量。
        # 注意：它们**不进入**上面那套互斥周期分类（七类之和必须等于
        # compute_cycles × cores），是与之平行的独立视角。
        per_core_stall = [{k: v * scale for k, v in c.snapshot_pool_stall_reasons().items()}
                          for c in self.cores]
        reason_keys = ('alloc_no_free_slot', 'alloc_point_in_flight',
                       'express_floor_block', 'retire_rejected', 'retire_backoff_wait')
        self.global_stats.psum_stall_reasons_per_core = per_core_stall
        self.global_stats.psum_stall_reasons = {
            key: sum(d.get(key, 0) for d in per_core_stall) for key in reason_keys}

        per_core_retire = []
        for core in self.cores:
            per_core_retire.append({
                'submitted': core.retire_submitted_points * scale,
                'rejected': core.retire_rejected_points * scale,
                'backoff_wait': core.backoff_wait_cycles * scale,
                'busy_cycles': core.retire_busy_cycles * scale,
                'retired': core.retired_points() * scale,
            })
        self.global_stats.retire_per_core = per_core_retire
        retired_total = sum(r['retired'] for r in per_core_retire)
        # 分母：一拍最多能 retire 的点数 = num_cores × retire_column
        retire_capacity = self.num_cores * self.retire_column * self.compute_cycles
        self.global_stats.retire_capacity_points = retire_capacity
        self.global_stats.retire_utilization = (
            retired_total / retire_capacity if retire_capacity else 0.0)
        self.global_stats.retire_rejected_total = sum(r['rejected'] for r in per_core_retire)
        self.global_stats.retire_backoff_wait_total = sum(
            r['backoff_wait'] for r in per_core_retire)

        print('\n[Retire 利用率]')
        print(f'  实际退休点数:   {retired_total}')
        print(f'  理论上限:       {retire_capacity} '
              f'= num_cores({self.num_cores}) x retire_column({self.retire_column}) '
              f'x compute_cycles({self.compute_cycles})')
        print(f'  利用率:         {self.global_stats.retire_utilization:.2%}')
        print(f'  accumulator 反压: 被拒 {self.global_stats.retire_rejected_total} 点, '
              f'错开等待 {self.global_stats.retire_backoff_wait_total} 拍')
        print('\n[各 Core retire 统计]')
        print('  Core  提交点  被拒点  退避等待拍  提交拍  实际退休点  利用率')
        for idx, r in enumerate(per_core_retire):
            util = r['retired'] / (self.retire_column * self.compute_cycles) \
                if self.compute_cycles else 0.0
            print(f'  {idx:^4}  {r["submitted"]:^7} {r["rejected"]:^7} '
                  f'{r["backoff_wait"]:^10} {r["busy_cycles"]:^7} '
                  f'{r["retired"]:^10} {util:>7.2%}')
        print('\n[Psum 停顿理由（按 core）]')
        head = '  ' + '理由'.ljust(26) + ''.join(
            f'{"C"+str(i):^9}' for i in range(self.num_cores)) + '  合计'
        print(head)
        label = {
            'alloc_no_free_slot': '点槽耗尽',
            'alloc_point_in_flight': '该点在途(等写回/FC)',
            'express_floor_block': '被 Express 欠账占住',
            'retire_rejected': 'accumulator 反压被拒',
            'retire_backoff_wait': '被拒后错开等待拍',
        }
        for key in reason_keys:
            row = '  ' + label[key].ljust(26)
            row += ''.join(f'{d.get(key, 0):^9}' for d in per_core_stall)
            row += f'{self.global_stats.psum_stall_reasons[key]:^6}'
            print(row)
        representative_mode_counts = collections.Counter()
        for core in self.cores:
            representative_mode_counts.update(core.conv_mode_split_counts)
        for mode in (0, 1, 2):
            representative_mode_counts.setdefault(mode, 0)
        # Cout groups reuse the same input feature maps, so mode ratios should
        # stay at the representative cout-group count instead of being scaled.
        self.global_stats.conv_mode_split_counts = {
            mode: int(representative_mode_counts[mode])
            for mode in (0, 1, 2)
        }
        total_mode_splits = sum(self.global_stats.conv_mode_split_counts.values())
        self.global_stats.conv_mode_split_fractions = {
            mode: (
                self.global_stats.conv_mode_split_counts[mode] / total_mode_splits
                if total_mode_splits
                else 0.0
            )
            for mode in (0, 1, 2)
        }

    def _publish_tree_counters(self, accumulator, scale):
        # Every scheduled cycle, including the final one, closes in tick().
        assert accumulator.stats['tree_sampled_cycles'] == self.representative_compute_cycles
        self.global_stats.num_reduce_trees = self.num_reduce_trees if self.enable_reduce_tree else 0
        self.global_stats.reduce_tree_stats_representative = dict(accumulator.stats)
        self.global_stats.reduce_tree_stats = {
            key: value if key.endswith('_peak') else value * scale
            for key, value in accumulator.stats.items()
        }
        self.global_stats.reduce_tree_demand_hist = {
            needed: cycles * scale for needed, cycles in sorted(accumulator.tree_demand_hist.items())
        }
        self.global_stats.reduce_tree_used_hist = {
            used: cycles * scale for used, cycles in sorted(accumulator.tree_used_hist.items())
        }
        self.reduce_tree_trace = accumulator.tree_trace
        self.global_stats.reduce_tree_cout_scale = scale
        self.global_stats.frontend_stall_cycles = sum(c.frontend_stalls for c in self.cores) * scale
        self.global_stats.psum_buffer_alloc_stall_cycles = sum(c.psum_buffer_alloc_stalls for c in self.cores) * scale
        self.global_stats.bank_conflict_cycles = accumulator.stats['conflict_core_cycles'] * scale
        print(f"  Reduce trees: {self.global_stats.num_reduce_trees}, "
              f"demand peak={accumulator.stats['tree_demand_peak']}, "
              f"used peak={accumulator.stats['tree_used_peak']}")
        print(f"  Tree demand histogram (needed: cycles): {self.global_stats.reduce_tree_demand_hist}")

    def _clear_accumulator_state(self, accumulator):
        """FC TBL tile 之间清空阵列状态，使下一块 tile 能复用。

        FC 当前不计这段周期；CONV 已改为每个 T 后执行计费排空。
        _cycle_open 置 False 可让 end_cycle() 提前返回，从而不增加
        tree_sampled_cycles —— 该计数必须等于 representative_compute_cycles，
        而后者不含这里的清空拍数。
        """
        assert all(c.is_finished and (c.pool is None or not c.pool.has_pending())
                   for c in self.cores)
        # 流水暂存区也必须为空。accumulator 是跨 b/g 复用的，一个滞留包会静默
        # 漏进下一组，表现为**数值错误而不是周期错误**——所以这里炸掉比放过去好。
        assert not accumulator._staged_requests
        assert not accumulator._pending_requests
        # 清空是零成本的簿记，不应计入排空统计——否则 drain_requests 会把它
        # 和被计费的那一次排空混在一起，使 drain_cycles 与 drain_requests 的
        # 倍数关系失真。这里临时快照并在结束后还原这四个计数器。
        saved = {key: accumulator.stats[key] for key in
                 ('drain_requests', 'drain_cycles', 'drained_words', 'drained_bits')}
        accumulator.request_drain()
        while not accumulator.is_empty():
            accumulator._cycle_open = False
            accumulator.tick()
        accumulator.stats.update(saved)

    def _drain_accumulator(self, accumulator, *, conv=False):
        """当前 T 后执行阻塞扫描，计入周期。

        扫满整个物理阵列（bank_h × bank_w 个 word，每拍 drain_words_per_cycle
        个），阻塞全部 Core；返回的拍数由调用方计入 compute_cycles 与
        group_total_cycles。
        """
        assert all(c.is_finished and (c.pool is None or not c.pool.has_pending())
                   for c in self.cores)
        # 流水暂存区也必须为空。accumulator 是跨 b/g 复用的，一个滞留包会静默
        # 漏进下一组，表现为**数值错误而不是周期错误**——所以这里炸掉比放过去好。
        assert not accumulator._staged_requests
        assert not accumulator._pending_requests
        accumulator.request_drain()
        cycles = 0
        while not accumulator.is_empty():
            accumulator.begin_cycle()
            for core in self.cores:
                if core.pool is not None:
                    core.pool.set_cycle(self.retire_trace_cycle)
                if conv:
                    self.conv_core_cycle_counts[core]['lif_drain'] += 1
            accumulator.tick()
            self.retire_trace_cycle += 1
            cycles += 1
        # 对齐 RTL 的同步读、LIF 处理与最终写回屏障。
        for _ in range(2):
            accumulator.begin_cycle()
            for core in self.cores:
                if conv:
                    self.conv_core_cycle_counts[core]['lif_drain'] += 1
            accumulator.tick()
            self.retire_trace_cycle += 1
            cycles += 1
        self.drain_wall_cycles += cycles
        return cycles

    def _print_performance_counters(self):
        print("\n[代表性 Cout-group 性能统计]")
        mode_counts = getattr(self.global_stats, "conv_mode_split_counts", {})
        mode_fractions = getattr(self.global_stats, "conv_mode_split_fractions", {})
        total_mode_splits = sum(mode_counts.get(mode, 0) for mode in (0, 1, 2))
        if total_mode_splits:
            print(
                "  Conv mode 比例: "
                f"M0 {mode_counts.get(0, 0)} ({mode_fractions.get(0, 0.0):.2%}), "
                f"M1 {mode_counts.get(1, 0)} ({mode_fractions.get(1, 0.0):.2%}), "
                f"M2 {mode_counts.get(2, 0)} ({mode_fractions.get(2, 0.0):.2%})"
            )
        else:
            print("  Conv mode 比例: N/A（该层不走 Conv Split 路径）")
        if self.representative_compute_cycles:
            pe_util = self.pe_cycles.to(torch.float32) / self.representative_compute_cycles
            print(f"  各 Core PE 利用率 [W0,W1,W2]: {pe_util.tolist()}")

    def run_convolution(self, input_tensor, kernel_tensor, padding=1, stride=1, groups=1):
            """
            基于全局时钟推进的周期精确模拟 (Cycle-Accurate Simulation)
            支持：任意矩形卷积核(如1x3, 3x1)、分组卷积(Groups)、Depthwise卷积、非对称Padding/Stride
            """        
            self.global_stats = Stats()
            self.total_cycles = 0
            self.compute_cycles = 0
            self._reset_performance_counters()
            
            # ==========================================================
            # 🚨 状态重置：彻底清空所有 Core 在上一个测试用例的残留统计数据
            # ==========================================================
            # 初始化隐形周期监控变量
            self.total_tail_draining = [0] * self.num_cores
            self.total_tile_idle = [0] * self.num_cores   # cin 队列为空后，core 的等待周期，即调度周期的一部分
            self.total_sync_barriers = 0
            # ==========================================================

            # 1. 维度解析与张量预处理
            if len(input_tensor.shape) == 5:
                T, B, in_c, in_h, in_w = input_tensor.shape
            else:
                T, in_c, in_h, in_w = input_tensor.shape
                B = 1
                input_tensor = input_tensor.unsqueeze(1) 

            # 动态获取卷积核的真实形状
            Cout, in_c_kernel, k_h, k_w = kernel_tensor.shape
            
            # 分组卷积参数解析
            Cin_per_group = in_c // groups
            Cout_per_group = Cout // groups
            assert in_c == in_c_kernel, f"Group Cin mismatch: {in_c} vs {in_c_kernel}"

            # 兼容 tuple 格式的 padding 和 stride
            p_h, p_w = padding if isinstance(padding, (tuple, list)) else (padding, padding)
            s_h, s_w = stride if isinstance(stride, (tuple, list)) else (stride, stride)

            # 使用解耦后的参数计算输出特征图尺寸
            out_h = (in_h + 2 * p_h - k_h) // s_h + 1
            out_w = (in_w + 2 * p_w - k_w) // s_w + 1
            output_tensor = torch.zeros((T, B, Cout, out_h, out_w), dtype=kernel_tensor.dtype, device=input_tensor.device)

            # 使用不对称的 padding 构造输入特征图
            padded_input = torch.zeros((T, B, in_c, in_h + 2 * p_h, in_w + 2 * p_w), 
                                    dtype=input_tensor.dtype, device=input_tensor.device)
            padded_input[:, :, :, p_h:p_h+in_h, p_w:p_w+in_w] = input_tensor

            # 计算分块数量（注意：此处是分组卷积中每group的 Cout 需要被self.num_pus拆分成多少个 Tile）
            Cout_tiles_per_g = ceil_a_by_b(Cout_per_group, self.num_pus)
            Cout_group_nums_per_g = Cout_tiles_per_g
            H_tile_num = ceil_a_by_b(out_h, self.oh)
            W_tile_num = ceil_a_by_b(out_w, self.ow)

            # 初始化带有反压机制的全局 Bank
            accumulator = Accumulator(
                retire_column=self.retire_column,
                bank_h=self.oh,
                bank_w=self.ow,
                enable_reduce_tree=self.enable_reduce_tree,
                num_reduce_trees=self.num_reduce_trees,
                record_tree_trace=self.record_tree_trace,
                num_pus=self.num_pus,
                mp_size=self.mp_size,
                retire_pipeline_stages=self.retire_pipeline_stages,
            )

            # 将动态识别到的 kernel_h 下发给 Core，用于 1x3 卷积的硬件门控
            for core in self.cores:
                core.kernel_h = k_h

            # ==========================================================
            #物理引擎启动：B -> Group -> Cout(根据num_pu进行分组，每组的cycle都是一样的) -> H_tile -> W_tile -> T -> Cin
            # ==========================================================
            for b in range(B):
                for g in range(groups):
                    # for cout in range(Cout_group_nums_per_g):
                    group_total_cycles = 0
                    
                    # 定位当前 Group 负责的输入和输出通道范围
                    cin_start = g * Cin_per_group
                    cin_end = (g + 1) * Cin_per_group
                    cout_start = g * Cout_per_group
                    group_cout_end = (g + 1) * Cout_per_group
                    # kernel_tensor is [Cout, Cin, kH, kW] in *global* channel
                    # numbering, so both slices must be taken group-locally.
                    # Slicing rows from the global cout offset would wrap into the
                    # next group's filters (and can go past the end of Cout).
                    tile_end = min(group_cout_end, cout_start + self.num_pus)
                    local_cout_end = tile_end - cout_start
                    wave_kernel = kernel_tensor[cout_start:tile_end,
                                                cin_start:cin_end, :, :]
                    valid_pu = local_cout_end
                    wave_cout = valid_pu
                    
                    # --- 载入权重的开销 ---
                    # wave_kernel is [valid_pu, Cin_per_group, k_h, k_w]。
                    # 这里算的是**单个 group** 的滤波器字节数；"每个 group 都要
                    # 载入自己的权重" 这一事实已经由外层 `for g in range(groups)`
                    # 通过累加 group_total_cycles 体现，不能再乘 groups，
                    # 否则每组开销和访存统计都会被放大 groups 倍。
                    weight_data = (Cin_per_group * k_h * k_w * self.weight_size
                                   * wave_cout)
                    self.global_stats.reads['g_wgt'] += weight_data 
                    self.global_stats.data_moved['kernel'] += weight_data 
                    load_kernel_cycle = ceil_a_by_b(weight_data, self.bandwidth)
                    group_total_cycles += load_kernel_cycle 
                    
                    for h_idx in range(H_tile_num):
                        out_h_start = h_idx * self.oh
                        out_h_end = min(out_h, out_h_start + self.oh)
                        valid_oh = out_h_end - out_h_start

                        for w_idx in range(W_tile_num):
                            out_w_start = w_idx * self.ow
                            out_w_end = min(out_w, out_w_start + self.ow)
                            valid_ow = out_w_end - out_w_start

                            in_h_start = out_h_start * s_h
                            in_h_end = (out_h_end - 1) * s_h + k_h
                            in_w_start = out_w_start * s_w
                            in_w_end = (out_w_end - 1) * s_w + k_w

                            for t in range(T):
                                # 切片获取当前组专属的输入特征图
                                current_input_tile = padded_input[t, b, cin_start:cin_end, in_h_start:in_h_end, in_w_start:in_w_end]
                                
                                # --- 载入特征图的开销 ---
                                act_data = Cin_per_group * (in_h_end - in_h_start) * (in_w_end - in_w_start)
                                self.global_stats.reads['g_act'] += act_data 
                                self.global_stats.data_moved['act'] += act_data 
                                load_act_cycle = ceil_a_by_b(act_data, self.bandwidth)
               
                                self.total_sync_barriers += 1  # 记录一次硬件同步屏障
                                
                                # 建立任务队列：只包含当前 Group 对应的输入通道
                                cin_queue = list(range(cin_start, cin_end))
                                time_step_compute_cycles = 0
                                
                                for core in self.cores:
                                    core.is_finished = True

                                # 开启全局心跳
                                while (
                                    cin_queue
                                    or any(not core.is_finished for core in self.cores)
                                    or not accumulator.is_empty()
                                ):

                                    # 任务派发, 每个core每拍都有任务
                                    for i, core in enumerate(self.cores):
                                        if not core.is_finished:
                                            continue

                                        # 每个空闲 Core 每拍只检查一个 Cin tile，零通道也占一次派发。
                                        while cin_queue:
                                            global_cin = cin_queue.pop(0)
                                            # current_input_tile and wave_kernel are both
                                            # group-local, so the group-local index must
                                            # drive the map lookup *and* the weight
                                            # selection. Passing the global index here
                                            # indexed past the group's filters.
                                            local_cin = global_cin - cin_start
                                            current_cin_map = current_input_tile[local_cin, :, :]

                                            # 全0通道直接快速跳过
                                            if torch.sum(current_cin_map) == 0:
                                                self.all_zero_channel_counts += 1
                                                continue

                                            self.tile_assignment_counts[i] += 1
                                            core.configure_conv_weights(
                                                cin=local_cin,
                                                kernel=wave_kernel,
                                            )
                                            core.init_for_new_tile(
                                                accumulator=accumulator,
                                                core_id=i,
                                                if_map=current_cin_map,
                                                H=valid_oh,
                                                W=valid_ow,
                                            )
                                            break

                                    # 每拍每个 Core 都采样，包括已完成、等待其他 Core 的空闲拍。
                                    # 在写回前记录排空状态，避免漏计最后一次写回/释放的周期。
                                    cycle_before = {
                                        core: (
                                            core.compute_issue_cycles,
                                            core.frontend_stalls,
                                            core.psum_buffer_alloc_stalls,
                                            core.express_unit.express_unit_busy_cycles,
                                            # accumulator 反压：本拍该 core 的池因错开退避
                                            # （_retry_after）未能提交退休。用池内计数增量判定。
                                            (core.pool.backoff_wait_cycles
                                             if core.pool is not None else 0),
                                            not core.is_finished
                                            and core.split_unit.is_finished
                                            and not core.has_pending_issue()
                                            and core.pool is not None
                                            and core.pool.has_pending(),
                                        )
                                        for core in self.conv_core_cycle_counts
                                    }

                                    accumulator.begin_cycle()
                                    # One cycle: clock -> Split -> old retire requests
                                    # -> PE/Express issue/write -> whole-cycle arbitration.
                                    for core in self.cores:
                                        if core.pool is not None:
                                            core.pool.set_cycle(self.retire_trace_cycle)
                                        if not core.is_finished:
                                            core.split_unit.tick()
                                    for core in self.cores:
                                        if core.pool is not None:
                                            core.pool.try_flush_to_accumulator()
                                    for core in self.cores:
                                        if not core.is_finished:
                                            core.tick_compute()
                                    accumulator.tick()

                                    # 各 Core 独立互斥分类，不要求其他 Core 同时停顿。
                                    # 成功计算与前端/分配失败由本拍计数增量判断；完成计算后
                                    # 首次 retire_old([]) 新产生的排空状态也计入写回等待。
                                    for core, (compute, frontend, psum_alloc, express_busy,
                                               backoff_before, was_draining) in cycle_before.items():
                                        if core.compute_issue_cycles > compute:
                                            state = 'compute'
                                        elif core.frontend_stalls > frontend:
                                            state = 'frontend'
                                        elif core.psum_buffer_alloc_stalls > psum_alloc:
                                            state = 'psum_alloc'
                                        elif core.express_unit.express_unit_busy_cycles > express_busy:
                                            state = 'express'
                                        elif (core.pool is not None
                                              and core.pool.backoff_wait_cycles > backoff_before):
                                            state = 'backpressure'
                                        elif was_draining or (
                                            not core.is_finished
                                            and core.split_unit.is_finished
                                            and not core.has_pending_issue()
                                            and core.pool is not None
                                            and core.pool.has_pending()
                                        ):
                                            state = 'writeback'
                                        else:
                                            state = 'idle'
                                        self.conv_core_cycle_counts[core][state] += 1

                                    
                                    # splitunit在FIFO中所有数据都给pu后，才会finish
                                    # core算完了，而且发现cin队列为空，那就只需要等待其它core的计算了。
                                    for i, core in enumerate(self.cores):
                                        if core.is_finished and not cin_queue:
                                            self.total_tile_idle[i] += 1

                                    time_step_compute_cycles += 1
                                    self.retire_trace_cycle += 1
                                
                                self.compute_cycles += time_step_compute_cycles
                                group_total_cycles += max(time_step_compute_cycles, load_act_cycle)
                                drain = self._drain_accumulator(accumulator, conv=True)
                                self.compute_cycles += drain
                                group_total_cycles += drain

                                # 每个时间步算完后，bank中的内容读取给LIF层一次，更新膜电位后再写回bank
                                spike_write_data = valid_oh * valid_ow * wave_cout * self.mp_size
                                self.global_stats.writes['g_psum'] += spike_write_data 
                                self.global_stats.data_moved['act'] += spike_write_data
                             
                            # 算完所有时间步后，bank中的结果可以写回dram
                            spike_write_data = valid_oh * valid_ow * wave_cout * self.mp_size
                            self.global_stats.writes['dram'] += spike_write_data 
                            self.global_stats.data_moved['mp'] += spike_write_data
                    
                    self.total_cycles += group_total_cycles

            # --- 根据每个group中对num_pu的tile数，放大计算周期和总周期 ---
            # 放大计算周期和总周期
            representative_compute_cycles = self.compute_cycles
            self.compute_cycles *= Cout_group_nums_per_g
            self.total_cycles = self.total_cycles * Cout_group_nums_per_g

            # 放大全局访存数据
            for key in ['g_wgt', 'g_act']:
                self.global_stats.reads[key] *= Cout_group_nums_per_g
            self.global_stats.writes['g_psum'] *= Cout_group_nums_per_g
            for key in ['kernel', 'act', 'mp']:
                self.global_stats.data_moved[key] *= Cout_group_nums_per_g
                
            # 放大拥塞与硬件停顿画像
            self._publish_performance_counters(
                representative_compute_cycles,
                conv_cout_scale=Cout_group_nums_per_g,
            )
            self._print_performance_counters()
            self._publish_tree_counters(accumulator, Cout_group_nums_per_g)

            self.global_stats.total_cycles = self.total_cycles 
            self.global_stats.compute_cycles = self.compute_cycles 

            # 打印必要的逐层信息；五类周期占比已由性能统计统一输出。
            print("\n" + "="*60)
            print("--- 全局 Cycle-Accurate (CAS) 仿真报告 ---")
            print(f"参与协同计算 Core 数量: {self.num_cores} | Group 切分: {groups}")
            print(f"总计全局执行周期: {self.total_cycles}")
            print(f"纯内核计算周期: {self.compute_cycles}")

            print("\n[一个 cout_group 中，各 Core 微架构统计]")
            print(f"  全零通道数目: {self.all_zero_channel_counts} 个")
            for i, core in enumerate(self.cores):
                print(
                    f"  -> Core {i}: Psum分配失败: {core.psum_buffer_alloc_stalls:4d} 拍 | "
                    f"Split供数不足: {core.frontend_stalls:4d} 拍 | "
                    f"分配tile: {self.tile_assignment_counts[i]:4d}块"
                )
            print("="*60 + "\n")

            return output_tensor, self.global_stats

    # confim by yp in 2026/05/17
    def run_fc(self, input_tensor, weight_matrix):
        """
        input_tensor: [TBL, N_in] 
        """
        self.global_stats = Stats()
        self.total_cycles = 0
        self.compute_cycles = 0
        self._reset_performance_counters()
        
        # 清空 Core 历史状态
        TBL, Cin = input_tensor.shape
        Cin_w, Cout = weight_matrix.shape
        assert Cin == Cin_w, f"FC Mapping Error: Cin mismatch {Cin} vs {Cin_w}"

        # ==========================================================
        # 🚨 核心切分逻辑：计算 Tile 的规模,每个PU放3个PE，每个PE表示一个post-nu
        # ==========================================================
        Cout_tiles = ceil_a_by_b(Cout, self.num_pus * 3)
        Cout_group_nums = Cout_tiles
        
        # 根据 Bank 高度安全红线，计算 TBL 维度能容纳的最大行数
        TBL_tile_size = (self.oh // 3) * self.ow
        TBL_tile_nums = ceil_a_by_b(TBL, TBL_tile_size)
        # print(f"[*] 计算 TBL Tile 数量: {TBL_tile_nums}")
        # ==========================================================
        
        accumulator = Accumulator(
            retire_column=self.retire_column,
            bank_h=self.oh,
            bank_w=self.ow,
            enable_reduce_tree=self.enable_reduce_tree,
            num_reduce_trees=self.num_reduce_trees,
            record_tree_trace=self.record_tree_trace,
            num_pus=self.num_pus,
            mp_size=self.mp_size,
            retire_pipeline_stages=self.retire_pipeline_stages,
        )


        # 最外层：Cout 分组 (避免权重重复搬运)
        # for cout_idx in range(Cout_group_nums):
        # ==========================================================
        # 🚀 极速仿真优化：因为每组负载等价，我们仅对【第一组】进行真实时钟推演
        # ==========================================================   
        cout_start = 0
        cout_end = min(Cout, self.num_pus * 3)
        wave_weight = weight_matrix[:, cout_start:cout_end]
        wave_cout = cout_end - cout_start
        valid_pu = ceil_a_by_b(wave_cout, 3)
        # systolic_delay = valid_pu
        
        weight_data = Cin * self.weight_size * wave_cout
        self.global_stats.reads['g_wgt'] += weight_data
        self.global_stats.data_moved['kernel'] += weight_data
        load_kernel_cycle = ceil_a_by_b(weight_data, self.bandwidth)
        
        group_total_cycles = load_kernel_cycle

        # 中层循环：对 TBL 序列进行安全分块 (TBL Tile)
        for TBL_idx in range(TBL_tile_nums):
            TBL_start = TBL_idx * TBL_tile_size
            TBL_end = min(TBL, TBL_start + TBL_tile_size)
            valid_TBL = TBL_end - TBL_start
            
            # 当前 TBL Tile 的完整输入特征图
            current_input_tile = input_tensor[TBL_start:TBL_end, :]

            # --- 加载激活数据---
            act_data = valid_TBL * Cin * 2
            self.global_stats.reads['g_act'] += act_data
            self.global_stats.data_moved['act'] += act_data
            load_act_cycle = ceil_a_by_b(act_data, self.bandwidth)

            # FC 模式下，每次从 Cin 中取 3 列给一个 Core
            cin_queue = list(range(0, Cin, 3))
            time_step_compute_cycles = 0

            for core in self.cores:
                core.is_finished = True

            # 内层心跳：彻底遍历完所有的 Cin (累加完整的 Psum)
            while (
                cin_queue
                or any(not core.is_finished for core in self.cores)
                or not accumulator.is_empty()
            ):

                # 1. 任务派发
                for i, core in enumerate(self.cores):
                    while core.is_finished and cin_queue:
                        cin_start_idx = cin_queue.pop(0)
                        cin_end_idx = min(Cin, cin_start_idx + 3)
                        
                        core_input = current_input_tile[:, cin_start_idx:cin_end_idx]
                        
                        # 补齐 3 列，防止越界
                        if core_input.shape[1] < 3:
                            pad_len = 3 - core_input.shape[1]
                            core_input = torch.nn.functional.pad(core_input, (0, pad_len))
                            
                        if torch.sum(core_input) == 0:
                            # 统计代表性 Cout-group 中被调度器跳过的全零 FC 输入 tile。
                            self.fc_all_zero_tiles += 1
                            continue

                        self.tile_assignment_counts[i] += 1

                        # 3*48 of core weight
                        core_weight = wave_weight[cin_start_idx:cin_end_idx, :]
                        if core_weight.shape[0] < 3:
                            pad_len = 3 - core_weight.shape[0]
                            core_weight = torch.nn.functional.pad(core_weight, (0, 0, 0, pad_len))

                        core.configure_fc_weights(core_weight)
                        core.init_for_linear_tile(
                            accumulator,
                            i,
                            core_input,
                            valid_pu,
                        )
                        break

                accumulator.begin_cycle()
                for core in self.cores:
                    if core.pool is not None:
                        core.pool.set_cycle(self.retire_trace_cycle)
                    if not core.is_finished:
                        core.split_unit.tick()
                for core in self.cores:
                    if core.pool is not None:
                        core.pool.try_flush_to_accumulator()
                for core in self.cores:
                    if not core.is_finished:
                        core.tick_compute_linear()
                accumulator.tick()

                time_step_compute_cycles += 1
                self.retire_trace_cycle += 1

            # All Cin contributions are committed; drain before reusing the array.
            self._clear_accumulator_state(accumulator)
            post_nu_num = valid_TBL * wave_cout
            spike_write_data = post_nu_num * self.ow
            self.global_stats.writes['g_psum'] += spike_write_data
            self.global_stats.data_moved['mp'] += spike_write_data
            
            # 无需加上脉动阵列排空的尾部延迟
            self.compute_cycles += (time_step_compute_cycles )
            tile_compute_cycles = time_step_compute_cycles
            if TBL_idx == 0:
                group_total_cycles += load_act_cycle + tile_compute_cycles
            else:
                group_total_cycles += max(tile_compute_cycles, load_act_cycle)

        
        # ==========================================================
        # 📈 状态缩放：将单组代表的统计结果，按 Cout_group_nums 等比放大
        # ==========================================================
        # FC 没有 group 循环，"每组一次" 对应每次 FC 调用排空一次。
        # 逐 TBL tile 之间只清状态、不计费。
        drain = self._drain_accumulator(accumulator)
        self.compute_cycles += drain
        group_total_cycles += drain
        self.total_cycles = group_total_cycles * Cout_group_nums
        representative_compute_cycles = self.compute_cycles
        self.compute_cycles *= Cout_group_nums
        
        # 放大全局访存数据
        for key in ['g_wgt', 'g_act']:
            self.global_stats.reads[key] *= Cout_group_nums
        self.global_stats.writes['g_psum'] *= Cout_group_nums
        for key in ['kernel', 'act', 'mp']:
            self.global_stats.data_moved[key] *= Cout_group_nums
            
        # 放大拥塞与硬件停顿画像
        self._publish_performance_counters(
            representative_compute_cycles,
        )
        self._print_performance_counters()
        self._publish_tree_counters(accumulator, Cout_group_nums)

        # 同步给 global_stats 对象
        self.global_stats.total_cycles = self.total_cycles 
        self.global_stats.compute_cycles = self.compute_cycles 
        self.global_stats.fc_all_zero_tiles = self.fc_all_zero_tiles

        # 打印必要的逐层信息；FC 映射、派发和周期计算保持不变。
        print("\n" + "="*60)
        print("--- CAS report (linear layer) ---")
        print(f"张量切分状态: Cout 被分为 {Cout_group_nums} 组 | TBL 被切分为 {TBL_tile_nums} 个块")
        print(f"总计全局执行周期: {self.total_cycles}")
        print(f"纯内核计算周期: {self.compute_cycles}")
        print(f"全零输入 tile: {self.fc_all_zero_tiles} 个（代表性 Cout-group）")

        print("\n[各 Core 微架构统计]")
        for i, core in enumerate(self.cores):
            print(
                f"  -> Core {i}: Psum分配失败: {core.psum_buffer_alloc_stalls:4d} 拍 | "
                f"Split供数不足: {core.frontend_stalls:4d} 拍 | "
                f"分配tile: {self.tile_assignment_counts[i]:4d}块"
            )
        print("="*60 + "\n")

        # 注意：由于我们在 core.py 中去掉了全量 PU 驱动，导致返回的输出张量全是 0。
        # 既然我们只做周期评估，这里直接返回 None 即可。
        return None, self.global_stats
