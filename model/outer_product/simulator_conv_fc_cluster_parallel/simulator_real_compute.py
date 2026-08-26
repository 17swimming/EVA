import torch
from model.outer_product.simulator_conv_fc_cluster_parallel.Accumulator import Accumulator
from model.outer_product.simulator_conv_fc_cluster_parallel.core import Core
from model.utils import ceil_a_by_b, Stats
import collections



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
        num_clusters=1,
        retire_column=3,
        enable_reduce_tree=True,
        psum_pool_rows=6,
        enabled_modes=(0, 1, 2),
    ):
        self.num_cores = num_cores
        self.num_core_per_cluster = num_cores
        self.num_clusters = max(1, int(num_clusters))
        self.num_simulated_clusters = 1
        self.num_pus = num_pus
        self.GBsize_MP = 0
        self.GBsize_weight = 0
        self.bandwidth = 2048
        self.weight_size = 16
        self.mp_size = 32
        self.lif_num = 32

        self.ow = ow
        self.oh = oh
        self.kernel_size = 3
        self.fetch_rows = fetch_rows  
        self.split_fifo_depth = split_fifo_depth  
        # Accepted for CLI/backward compatibility. Cluster accumulators are
        # modeled as point-level register arrays and do not use sub-banks.
        self.num_sub_banks = None if num_sub_banks is None else int(num_sub_banks)
        self.retire_column = int(retire_column)
        self.enable_reduce_tree = bool(enable_reduce_tree)
        self.psum_pool_rows = int(psum_pool_rows)
        self.enabled_modes = tuple(sorted(int(mode) for mode in enabled_modes))

        self.clusters = [
            [
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
                )
                for _ in range(num_cores)
            ]
        ]
        self.cores = self.clusters[0]

        self.pe_cycles = torch.zeros((num_cores, 3))  
        self.cluster_pe_cycles = torch.zeros((self.num_simulated_clusters, num_cores, 3))
        self.bitstream_lengths = []  
        self.hazard_num = 0  
        
        # 新增：全局 Bank0 写流量追踪字典 {绝对周期索引 : 总写请求数}
        self.accumulator_write_trace = collections.defaultdict(list)
        # 新增：用于统计 r 跳跃的跨度
        self.r_jump_stats = collections.defaultdict(int)
        self.total_r_transitions = 0

    def set_enabled_modes(self, enabled_modes):
        modes = tuple(sorted({int(mode) for mode in enabled_modes}))
        if 0 not in modes or any(mode not in (0, 1, 2) for mode in modes):
            raise ValueError("enabled_modes must contain mode 0 and only use modes 0, 1, and 2")
        self.enabled_modes = modes
        for cluster in self.clusters:
            for core in cluster:
                core.enabled_modes = modes
                core.split_unit.enabled_modes = frozenset(modes)
                core.split_unit.split.enabled_modes = frozenset(modes)

    def _reset_performance_counters(self):
        self.pe_cycles.zero_()
        self.cluster_pe_cycles.zero_()
        self.issued_packets = 0
        self.post_split_issue_cycles = [0] * self.num_cores
        self.writeback_drain_cycles = [0] * self.num_cores
        self.post_split_issue_wall_cycles = 0
        self.writeback_drain_wall_cycles = 0
        self.writeback_drain_reason_cycles = collections.Counter()
        self.writeback_drain_occupancy_totals = collections.Counter()
        self.retire_trace_cycle = 0
        for cluster in self.clusters:
            for core in cluster:
                core.reset_performance_counters()
                core.is_finished = True

    def _sample_post_split_phases(self):
        any_issue = False
        any_writeback = False
        for index, core in enumerate(self.cores):
            if core.is_finished or not core.split_unit.is_finished:
                continue
            if core.has_pending_issue():
                self.post_split_issue_cycles[index] += 1
                any_issue = True
            elif core.pool is not None and core.pool.has_pending():
                self.writeback_drain_cycles[index] += 1
                summary = core.pool.pending_summary()
                self.writeback_drain_reason_cycles[self._classify_pending_summary(summary)] += 1
                for key, value in summary.items():
                    self.writeback_drain_occupancy_totals[key] += int(value)
                any_writeback = True
        self.post_split_issue_wall_cycles += int(any_issue)
        self.writeback_drain_wall_cycles += int(any_writeback)

    @staticmethod
    def _classify_pending_summary(summary):
        if (
            summary.get("pending_fc_bundles", 0)
            or summary.get("active_fc_pending", 0)
            or summary.get("pending_wb_fc_pending", 0)
        ):
            return "fc_pending_retire"
        if summary.get("pending_wb", 0):
            return "row_pending_wb"
        if summary.get("active_stream_pending", 0):
            return "stream_pending_retire"
        if summary.get("modified_buffers", 0):
            return "modified_active_buffers"
        return "unknown"

    def _publish_performance_counters(
        self,
        accumulators,
        representative_compute_cycles,
    ):
        if not isinstance(accumulators, (list, tuple)):
            accumulators = [accumulators]
        for cluster_idx, cluster in enumerate(self.clusters):
            for index, core in enumerate(cluster):
                self.cluster_pe_cycles[cluster_idx, index] = core.pe_cycles
                self.issued_packets += core.issued_packets
        self.pe_cycles = self.cluster_pe_cycles[0].clone()
        self.representative_compute_cycles = representative_compute_cycles
        self.global_stats.representative_compute_cycles = representative_compute_cycles
        self.global_stats.num_clusters = self.num_clusters
        self.global_stats.num_simulated_clusters = self.num_simulated_clusters
        self.global_stats.frontend_stall_cycles = sum(
            core.frontend_stalls for cluster in self.clusters for core in cluster
        )
        self.global_stats.compute_issue_core_cycles = sum(
            getattr(core, 'compute_issue_cycles', 0)
            for cluster in self.clusters
            for core in cluster
        )
        self.global_stats.psum_buffer_alloc_stall_cycles = sum(
            getattr(core, 'psum_buffer_alloc_stalls', core.stalls)
            for cluster in self.clusters
            for core in cluster
        )
        self.global_stats.retire_backpressure_stall_cycles = sum(
            getattr(core, 'retire_backpressure_stalls', 0)
            for cluster in self.clusters
            for core in cluster
        )
        self.global_stats.psum_buffer_stall_cycles = self.global_stats.psum_buffer_alloc_stall_cycles
        self.global_stats.bank_conflict_requests = sum(
            accumulator.stats["stalled_requests"] for accumulator in accumulators
        )
        self.global_stats.bank_conflict_cycles = sum(
            accumulator.stats["conflict_cycles"] for accumulator in accumulators
        )
        self.global_stats.bank_conflict_core_cycles = sum(
            accumulator.stats.get("conflict_core_cycles", 0) for accumulator in accumulators
        )
        self.global_stats.post_split_issue_core_cycles = sum(self.post_split_issue_cycles)
        self.global_stats.post_split_issue_wall_cycles = self.post_split_issue_wall_cycles
        self.global_stats.writeback_drain_core_cycles = sum(self.writeback_drain_cycles)
        self.global_stats.writeback_drain_wall_cycles = self.writeback_drain_wall_cycles
        self.global_stats.pe_cycles = self.pe_cycles.clone()
        self.global_stats.cluster_pe_cycles = self.cluster_pe_cycles.clone()
        self.global_stats.issued_packets = self.issued_packets
        representative_mode_counts = collections.Counter()
        for cluster in self.clusters:
            for core in cluster:
                representative_mode_counts.update(core.conv_mode_split_counts)
        for mode in (0, 1, 2):
            representative_mode_counts.setdefault(mode, 0)
        self.global_stats.conv_mode_split_counts_representative = dict(representative_mode_counts)
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
        # Backward-compatible aliases for any downstream CSV/log readers that
        # still look for the old issue-based names.
        self.global_stats.conv_mode_issue_counts_representative = dict(
            self.global_stats.conv_mode_split_counts_representative
        )
        self.global_stats.conv_mode_issue_counts = dict(self.global_stats.conv_mode_split_counts)
        self.global_stats.conv_mode_issue_fractions = dict(self.global_stats.conv_mode_split_fractions)
        self.global_stats.retire_column_cycle_stats = [
            {int(cycle): dict(values) for cycle, values in core.retire_column_cycle_stats.items()}
            for core in self.cores
        ]
        self.global_stats.retire_column_total_stats = [
            dict(core.retire_column_total_stats)
            for core in self.cores
        ]
        self.global_stats.retire_column_total_stats_by_cluster = [
            [dict(core.retire_column_total_stats) for core in cluster]
            for cluster in self.clusters
        ]
        self.global_stats.accumulator_stats = [
            dict(accumulator.stats) for accumulator in accumulators
        ]
        aggregate_accumulator_stats = collections.Counter()
        for accumulator in accumulators:
            aggregate_accumulator_stats.update(accumulator.stats)
        self.global_stats.accumulator_aggregate_stats = dict(aggregate_accumulator_stats)
        pool_stats = []
        aggregate_pool_stats = collections.Counter()
        aggregate_request_col_hist = collections.Counter()
        aggregate_request_row_hist = collections.Counter()
        aggregate_fc_bundle_row_hist = collections.Counter()
        aggregate_fc_pending_wait_hist = collections.Counter()
        psum_alloc_stall_reasons = collections.Counter()
        psum_alloc_stall_occupancy_totals = collections.Counter()
        psum_alloc_stall_active_rows_hist = collections.Counter()
        for cluster in self.clusters:
            cluster_pool_stats = []
            for core in cluster:
                if core.pool is None:
                    cluster_pool_stats.append({})
                    continue
                stats = dict(core.pool.stats)
                stats["request_col_hist"] = dict(core.pool.request_col_hist)
                stats["request_row_hist"] = dict(core.pool.request_row_hist)
                stats["fc_bundle_row_hist"] = dict(core.pool.fc_bundle_row_hist)
                stats["fc_pending_wait_hist"] = dict(core.pool.fc_pending_wait_hist)
                stats["final_occupancy"] = core.pool.occupancy_summary()
                cluster_pool_stats.append(stats)
                aggregate_pool_stats.update(core.pool.stats)
                aggregate_request_col_hist.update(core.pool.request_col_hist)
                aggregate_request_row_hist.update(core.pool.request_row_hist)
                aggregate_fc_bundle_row_hist.update(core.pool.fc_bundle_row_hist)
                aggregate_fc_pending_wait_hist.update(core.pool.fc_pending_wait_hist)
                psum_alloc_stall_reasons.update(core.psum_alloc_stall_reasons)
                psum_alloc_stall_occupancy_totals.update(core.psum_alloc_stall_occupancy_totals)
                psum_alloc_stall_active_rows_hist.update(core.psum_alloc_stall_active_rows_hist)
            pool_stats.append(cluster_pool_stats)
        self.global_stats.pool_stats = pool_stats
        self.global_stats.retire_pool_stats = dict(aggregate_pool_stats)
        self.global_stats.retire_request_col_hist = dict(aggregate_request_col_hist)
        self.global_stats.retire_request_row_hist = dict(aggregate_request_row_hist)
        self.global_stats.fc_bundle_row_hist = dict(aggregate_fc_bundle_row_hist)
        self.global_stats.fc_pending_wait_hist = dict(aggregate_fc_pending_wait_hist)
        self.global_stats.psum_alloc_stall_reasons = dict(psum_alloc_stall_reasons)
        self.global_stats.psum_alloc_stall_occupancy_totals = dict(psum_alloc_stall_occupancy_totals)
        self.global_stats.psum_alloc_stall_active_rows_hist = dict(psum_alloc_stall_active_rows_hist)
        self.global_stats.writeback_drain_reason_cycles = dict(self.writeback_drain_reason_cycles)
        self.global_stats.writeback_drain_occupancy_totals = dict(self.writeback_drain_occupancy_totals)

    def _print_performance_counters(self):
        print("\n[Representative Cout-group performance counters]")
        print(
            f"  Bank conflicts: {self.global_stats.bank_conflict_requests} rejected requests, "
            f"{self.global_stats.bank_conflict_cycles} wall-clock cycles"
        )
        print(
            f"  Accumulator conflict occupancy: "
            f"{self.global_stats.bank_conflict_core_cycles} rejected core-cycles"
        )
        print(
            f"  Accumulator conflict denominator: {self.num_cores} cores x "
            f"{self.representative_compute_cycles} representative compute cycles"
        )
        print(
            f"  Post-Split issue/compute: {self.global_stats.post_split_issue_core_cycles} core-cycles, "
            f"{self.global_stats.post_split_issue_wall_cycles} wall-clock cycles"
        )
        print(
            f"  Compute issue: {self.global_stats.compute_issue_core_cycles} core-cycles"
        )
        mode_counts = getattr(self.global_stats, "conv_mode_split_counts", {})
        mode_fractions = getattr(self.global_stats, "conv_mode_split_fractions", {})
        representative_mode_counts = getattr(
            self.global_stats, "conv_mode_split_counts_representative", {}
        )
        total_mode_splits = sum(mode_counts.get(mode, 0) for mode in (0, 1, 2))
        if total_mode_splits:
            print(
                "  Conv mode split stats (representative cout-group): "
                f"M0 {mode_counts.get(0, 0)}/{representative_mode_counts.get(0, 0)} "
                f"({mode_fractions.get(0, 0.0):.2%}), "
                f"M1 {mode_counts.get(1, 0)}/{representative_mode_counts.get(1, 0)} "
                f"({mode_fractions.get(1, 0.0):.2%}), "
                f"M2 {mode_counts.get(2, 0)}/{representative_mode_counts.get(2, 0)} "
                f"({mode_fractions.get(2, 0.0):.2%})"
            )
        else:
            print("  Conv mode split stats: N/A (this layer uses no Conv split path)")
        print(
            f"  Pure Psum writeback drain: {self.global_stats.writeback_drain_core_cycles} core-cycles, "
            f"{self.global_stats.writeback_drain_wall_cycles} wall-clock cycles"
        )
        print(
            f"  Issue stall split: psum alloc {self.global_stats.psum_buffer_alloc_stall_cycles} core-cycles, "
            f"frontend {self.global_stats.frontend_stall_cycles} core-cycles; "
            f"retire backpressure events {self.global_stats.retire_backpressure_stall_cycles} core-cycles"
        )
        if self.representative_compute_cycles:
            pe_util = self.pe_cycles.to(torch.float32) / self.representative_compute_cycles
            print(f"  PE utilization per core [W0,W1,W2]: {pe_util.tolist()}")
        self._print_retire_diagnostics()

    def _format_counter(self, counter, limit=8):
        if not counter:
            return "{}"
        items = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
        return "{" + ", ".join(f"{key}: {value}" for key, value in items) + "}"

    def _print_retire_diagnostics(self):
        pool_stats = getattr(self.global_stats, "retire_pool_stats", {})
        if not pool_stats:
            return
        pending_events = int(pool_stats.get("pending_wait_events", 0))
        avg_pending_wait = (
            pool_stats.get("pending_wait_cycles", 0) / pending_events
            if pending_events
            else 0.0
        )
        print("  Retire diagnostics:")
        print(
            "    requests/accept/reject: "
            f"conv_stream {pool_stats.get('conv_stream_requests', 0)}/"
            f"{pool_stats.get('conv_stream_accepts', 0)}/"
            f"{pool_stats.get('conv_stream_rejects', 0)}, "
            f"conv_row {pool_stats.get('conv_row_requests', 0)}/"
            f"{pool_stats.get('conv_row_accepts', 0)}/"
            f"{pool_stats.get('conv_row_rejects', 0)}, "
            f"fc {pool_stats.get('fc_requests', 0)}/"
            f"{pool_stats.get('fc_accepts', 0)}/"
            f"{pool_stats.get('fc_rejects', 0)}"
        )
        print(
            "    pending wait: "
            f"events {pending_events}, avg {avg_pending_wait:.2f} cycles, "
            f"max {pool_stats.get('pending_wait_max', 0)}, "
            f"retry attempts {pool_stats.get('pending_retry_attempts', 0)}"
        )
        print(
            "    fc partial rows: "
            f"partial requests {pool_stats.get('fc_partial_requests', 0)}, "
            f"accepted rows {pool_stats.get('fc_accepted_rows', 0)}, "
            f"rejected rows {pool_stats.get('fc_rejected_rows', 0)}, "
            f"pending wait rows {pool_stats.get('pending_wait_rows', 0)}"
        )
        accumulator_stats = getattr(self.global_stats, "accumulator_aggregate_stats", {})
        if accumulator_stats:
            print(
                "    accumulator request overlap: "
                f"conv {self._format_overlap_hist(accumulator_stats, 'conv_request_overlap')}; "
                f"fc {self._format_overlap_hist(accumulator_stats, 'fc_request_overlap')}; "
                f"conflict fc {self._format_overlap_hist(accumulator_stats, 'fc_conflict_overlap')}"
            )
            print(
                "    accumulator reduce tree: "
                f"requests {accumulator_stats.get('reduced_requests', 0)}, "
                f"writes {accumulator_stats.get('reduced_writes', 0)}, "
                f"fc writes {accumulator_stats.get('fc_reduced_writes', 0)}"
            )
        print(
            "    request cols hist "
            f"{self._format_counter(getattr(self.global_stats, 'retire_request_col_hist', {}))}; "
            "rows hist "
            f"{self._format_counter(getattr(self.global_stats, 'retire_request_row_hist', {}))}; "
            "fc rows hist "
            f"{self._format_counter(getattr(self.global_stats, 'fc_bundle_row_hist', {}))}"
        )
        print(
            "    psum alloc stall reasons "
            f"{self._format_counter(getattr(self.global_stats, 'psum_alloc_stall_reasons', {}))}; "
            "active rows hist "
            f"{self._format_counter(getattr(self.global_stats, 'psum_alloc_stall_active_rows_hist', {}))}"
        )
        print(
            "    writeback drain reasons "
            f"{self._format_counter(getattr(self.global_stats, 'writeback_drain_reason_cycles', {}))}"
        )
        drain_samples = int(getattr(self.global_stats, "writeback_drain_core_cycles", 0))
        drain_totals = getattr(self.global_stats, "writeback_drain_occupancy_totals", {})
        if drain_samples and drain_totals:
            print(
                "    writeback drain avg occupancy: "
                f"pending_wb {drain_totals.get('pending_wb', 0) / drain_samples:.2f}, "
                f"stream_pending {drain_totals.get('active_stream_pending', 0) / drain_samples:.2f}, "
                f"fc_bundles {drain_totals.get('pending_fc_bundles', 0) / drain_samples:.2f}, "
                f"fc_buffers {drain_totals.get('active_fc_pending', 0) / drain_samples:.2f}, "
                f"modified_buffers {drain_totals.get('modified_buffers', 0) / drain_samples:.2f}"
            )

    def _format_overlap_hist(self, stats, prefix):
        return "{" + ", ".join(
            f"{overlap}: {int(stats.get(f'{prefix}_{overlap}_requests', 0))}"
            for overlap in range(4)
        ) + "}"

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
            self.total_tile_idle = [0] * self.num_cores
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
            Cout_group_nums_per_g = ceil_a_by_b(Cout_tiles_per_g, self.num_clusters)
            active_clusters = min(self.num_clusters, Cout_tiles_per_g)
            H_tile_num = ceil_a_by_b(out_h, self.oh)
            W_tile_num = ceil_a_by_b(out_w, self.ow)

            # 初始化带有反压机制的全局 Bank
            accumulators = [
                Accumulator(
                    retire_column=self.retire_column,
                    bank_h=self.oh,
                    bank_w=self.ow,
                    enable_reduce_tree=self.enable_reduce_tree,
                )
                for _ in range(self.num_simulated_clusters)
            ]

            # 将动态识别到的 kernel_h 下发给 Core，用于 1x3 卷积的硬件门控
            for cluster in self.clusters:
                for core in cluster:
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
                    wave_kernels = []
                    valid_pus = []
                    for cluster_idx in range(self.num_clusters):
                        tile_start = cout_start + cluster_idx * self.num_pus
                        tile_end = min(group_cout_end, tile_start + self.num_pus)
                        if tile_start < tile_end:
                            wave_kernels.append(kernel_tensor[tile_start:tile_end, :, :, :])
                            valid_pus.append(tile_end - tile_start)
                        else:
                            wave_kernels.append(None)
                            valid_pus.append(0)
                    valid_pu = max(valid_pus) if valid_pus else 0
                    wave_cout = sum(valid_pus)
                    
                    # --- 载入权重的开销 ---
                    weight_data = Cin_per_group * k_h * k_w * self.weight_size * wave_cout
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
                                
                                for cluster in self.clusters:
                                    for core in cluster:
                                        core.is_finished = True

                                # 开启全局心跳
                                while (
                                    cin_queue
                                    or any(
                                        not core.is_finished
                                        for cluster in self.clusters
                                        for core in cluster
                                    )
                                    or any(not acc.is_empty() for acc in accumulators)
                                ):

                                    # 任务派发
                                    for i, core in enumerate(self.cores):
                                        if core.is_finished and cin_queue:
                                            local_cin = cin_queue[0]
                                            # [c,h,w]
                                            current_cin_map = current_input_tile[local_cin - cin_start, :, :]
                                            
                                            # 全0通道直接快速跳过
                                            if torch.sum(current_cin_map) == 0:
                                                cin_queue.pop(0)
                                                continue 
                                                
                                            cin_queue.pop(0)
                                            for cluster_idx, cluster in enumerate(self.clusters):
                                                cluster_kernel = wave_kernels[cluster_idx]
                                                cluster_core = cluster[i]
                                                if cluster_kernel is None:
                                                    cluster_core.is_finished = True
                                                    continue
                                                cluster_core.configure_conv_weights(
                                                    cin=local_cin,
                                                    kernel=cluster_kernel,
                                                )
                                                cluster_core.init_for_new_tile(
                                                    accumulator=accumulators[cluster_idx],
                                                    core_id=i,
                                                    if_map=current_cin_map,
                                                    H=current_cin_map.shape[0],
                                                )

                                    # (A) accumulator consumes requests from
                                    # the previous cycle first.
                                    for accumulator in accumulators:
                                        accumulator.tick()

                                    # (B) 内存总线传输阶段：写回 Psum
                                    for cluster in self.clusters:
                                        for core in cluster:
                                            if not core.is_finished or (core.pool is not None and core.pool.has_pending()):
                                                if core.pool is not None:
                                                    core.pool.set_cycle(self.retire_trace_cycle)
                                                core.pool.try_flush_to_accumulator()

                                    # (C) Core consumes packages already in the
                                    # FIFO at the start of this cycle.
                                    for cluster in self.clusters:
                                        for core in cluster:
                                            core.retire_trace_cycle = self.retire_trace_cycle

                                            if not core.is_finished:
                                                core.tick_compute()

                                    # (D) SplitUnit produces the next package
                                    # after Core; it is visible next cycle.
                                    for cluster in self.clusters:
                                        for core in cluster:
                                            if not core.is_finished:
                                                core.split_unit.tick()
                                    self._sample_post_split_phases()
                                    
                                    # === 隐形周期tick监控 ===
                                    for i, core in enumerate(self.cores):
                                        # splitunit在FIFO中所有数据都给pu后，才会finish
                                        # core算完了，而且发现cin队列为空，那就只需要等待其它core的计算了。
                                        if core.is_finished and not cin_queue:
                                            self.total_tile_idle[i] += 1

                                    time_step_compute_cycles += 1
                                    self.retire_trace_cycle += 1
                                
                                compute_with_systolic = time_step_compute_cycles + valid_pu   # 如果PU级联，cycles是这个
                                self.compute_cycles += time_step_compute_cycles
                                group_total_cycles += max(time_step_compute_cycles, load_act_cycle)

                                # 每个时间步算完后，bank中的内容读取给LIF层一次，更新膜电位后再写回bank
                                spike_write_data = valid_oh * valid_ow * wave_cout * self.mp_size
                                self.global_stats.writes['g_psum'] += spike_write_data 
                                self.global_stats.data_moved['act'] += spike_write_data
                             
                            # 算完所有时间步后，bank中的结果可以写回dram
                            spike_write_data = valid_oh * valid_ow * wave_cout * self.mp_size
                            self.global_stats.writes['dram'] += spike_write_data 
                            self.global_stats.data_moved['mp'] += spike_write_data
                    
                    self.total_cycles += group_total_cycles 
                
            # finish cin loop , send to lif ,but pipline ,so ,only last lif cycle need to add
            lif_cycle = ceil_a_by_b(valid_oh * valid_ow * valid_pu, self.lif_num)

            # --- 根据每个group中对num_pu的tile数，放大计算周期和总周期 ---
            # 放大计算周期和总周期
            representative_compute_cycles = self.compute_cycles
            self.compute_cycles *= Cout_group_nums_per_g
            self.total_cycles = self.total_cycles * Cout_group_nums_per_g + lif_cycle

            # 放大全局访存数据
            for key in ['g_wgt', 'g_act']:
                self.global_stats.reads[key] *= Cout_group_nums_per_g
            self.global_stats.writes['g_psum'] *= Cout_group_nums_per_g
            for key in ['kernel', 'act', 'mp']:
                self.global_stats.data_moved[key] *= Cout_group_nums_per_g
                
            # 放大拥塞与硬件停顿画像
            self._publish_performance_counters(
                accumulators,
                representative_compute_cycles,
            )
            self._print_performance_counters()

            self.global_stats.total_cycles = self.total_cycles 
            self.global_stats.compute_cycles = self.compute_cycles 

            # =======================================================
            # --- 打印全局周期精确仿真报告 ---
            # =======================================================
            print("\n" + "="*60)
            print("--- 🔬 全局 Cycle-Accurate (CAS) 仿真与拥塞报告 ---")
            print(f"参与协同计算 Core 数量: {self.num_cores} | Group 切分: {groups}")
            print(f"总计全局执行周期 (Total): {self.total_cycles}")
            print(f"纯内核计算周期 (Compute): {self.compute_cycles}")
            
            
            stalls_total = sum(acc.stats['stalled_requests'] for acc in accumulators)
            print(f"\n[后端存储反压 (Backend Stall) 评估]")
            print(f"  bank拒收的总次数: **{stalls_total}** 次")
            # if stalls_total > 0:
            #     print("  ⚠️ 警告：系统发生后端拥塞，部分 Core 被迫停顿！建议增加 Sub-bank 数量或 FIFO 深度。")
            # else:
            #     print("  ✅ 存储流水线畅通：未发生任何 Bank 写入冲突和反压阻塞。")
                
            print("\n[各 Core 微架构停顿画像]")
            total_frontend_stalls = 0
            total_psum_alloc_stalls = 0
            total_retire_backpressure_stalls = 0
            for i, core in enumerate(self.cores):
                core_stalls = getattr(core, 'psum_buffer_alloc_stalls', getattr(core, 'stalls', 0))
                retire_stalls = getattr(core, 'retire_backpressure_stalls', 0)
                frontend_stalls = getattr(core, 'frontend_stalls', 0)
                total_psum_alloc_stalls += core_stalls
                total_retire_backpressure_stalls += retire_stalls
                total_frontend_stalls += frontend_stalls
                print(
                    f"  -> Core {i}: [后端] Psum分配失败停顿: {core_stalls:4d} 拍 | "
                    f"[后端] Retire写回反压: {retire_stalls:4d} 拍 | "
                    f"[前端] Split 供数不足空转: {frontend_stalls:4d} 拍"
                )

            print(f"\n[前端解析性能 (Frontend Bubble) 评估]")
            print(f"  代表 cluster 累计因 PsumPool 无可用 buffer 导致的计算停顿: **{total_psum_alloc_stalls}** 拍")
            print(f"  代表 cluster 累计 retire 写回反压后台挂起事件: **{total_retire_backpressure_stalls}** 拍")
            print(f"  所有 Core 累计因 SplitUnit 解析过慢导致的饥饿空转: **{total_frontend_stalls}** 拍")
                
            # 隐形周期揭秘打印
            print(f"\n[🕵️ 隐形周期揭秘 (Hidden Cycles Breakdown)]")
            print("  说明: [排空写回] = Split结束但等待Psum写回bank的时间; [Tile尾部闲置] = cin任务耗尽后等待其他Core完成当前Tile的时间")
            print(f"  ▶ 一个cout_group累计触发的 H-W-T 同步 (Sync Barriers) 次数: **{self.total_sync_barriers}** 次")
            
            # 因为每个core的统计值total_tile_idle没有乘cout_group_nums，所以这里要乘以cout_group_nums
            total_drain = sum(self.writeback_drain_cycles)
            total_idle = sum(self.total_tile_idle) 
            
            for i in range(self.num_cores):
                avg_idle = self.total_tile_idle[i] / self.total_sync_barriers if self.total_sync_barriers > 0 else 0
                avg_drain = self.writeback_drain_cycles[i] / self.total_sync_barriers if self.total_sync_barriers > 0 else 0
                print(f"  -> Core {i}: [纯Psum写回等待] {self.writeback_drain_cycles[i]:4d} 拍 (均值 {avg_drain:.1f}拍/次) | "
                    f"[Tile尾部闲置] {self.total_tile_idle[i]:4d} 拍 (均值 {avg_idle:.1f} 拍/次)")
                
            print(f"  ▶ 一个cout_group累计发生的 [排空写回] 周期总和 (包含多个core同时等待的重叠，所以实际并没有等待这么多cycle): {total_drain} 拍")
            print(f"  ▶ 一个cout_group累计发生的 [Tile尾部闲置] 周期总和 (包含多个core同时等待的重叠，所以实际并没有等待这么多cycle): {total_idle} 拍")
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
        Cout_group_nums = ceil_a_by_b(Cout_tiles, self.num_clusters)
        
        # 根据 Bank 高度安全红线，计算 TBL 维度能容纳的最大行数
        TBL_tile_size = (self.oh // 3) * self.ow
        TBL_tile_nums = ceil_a_by_b(TBL, TBL_tile_size)
        # print(f"[*] 计算 TBL Tile 数量: {TBL_tile_nums}")
        # ==========================================================
        
        accumulators = [
            Accumulator(
                retire_column=self.retire_column,
                bank_h=self.oh,
                bank_w=self.ow,
                enable_reduce_tree=self.enable_reduce_tree,
            )
            for _ in range(self.num_simulated_clusters)
        ]

        # 1D 脉动阵列流水线延迟：数据跑完整个阵列需要的额外周期

        # 最外层：Cout 分组 (避免权重重复搬运)
        # for cout_idx in range(Cout_group_nums):
        # ==========================================================
        # 🚀 极速仿真优化：因为每组负载等价，我们仅对【第一组】进行真实时钟推演
        # ==========================================================   
        cout_start = 0
        wave_weights = []
        valid_pus = []
        for cluster_idx in range(self.num_clusters):
            tile_start = cout_start + cluster_idx * self.num_pus * 3
            tile_end = min(Cout, tile_start + self.num_pus * 3)
            if tile_start < tile_end:
                wave_weights.append(weight_matrix[:, tile_start:tile_end])
                valid_pus.append(ceil_a_by_b(tile_end - tile_start, 3))
            else:
                wave_weights.append(None)
                valid_pus.append(0)
        valid_pu = max(valid_pus) if valid_pus else 0
        wave_cout = sum(weight.shape[1] for weight in wave_weights if weight is not None)
        systolic_delay = valid_pu
        
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
            act_data = valid_TBL * Cin * self.mp_size
            self.global_stats.reads['g_act'] += act_data
            self.global_stats.data_moved['act'] += act_data
            load_act_cycle = ceil_a_by_b(act_data, self.bandwidth)

            # FC 模式下，每次从 Cin 中取 3 列给一个 Core
            cin_queue = list(range(0, Cin, 3))
            time_step_compute_cycles = 0

            for cluster in self.clusters:
                for core in cluster:
                    core.is_finished = True

            # 内层心跳：彻底遍历完所有的 Cin (累加完整的 Psum)
            while (
                cin_queue
                or any(
                    not core.is_finished
                    for cluster in self.clusters
                    for core in cluster
                )
                or any(not acc.is_empty() for acc in accumulators)
            ):
                # 1. 任务派发
                for i, core in enumerate(self.cores):
                    if core.is_finished and cin_queue:
                        cin_start_idx = cin_queue.pop(0)
                        cin_end_idx = min(Cin, cin_start_idx + 3)
                        
                        core_input = current_input_tile[:, cin_start_idx:cin_end_idx]
                        
                        # 补齐 3 列，防止越界
                        if core_input.shape[1] < 3:
                            pad_len = 3 - core_input.shape[1]
                            core_input = torch.nn.functional.pad(core_input, (0, pad_len))
                            
                        if torch.sum(core_input) == 0:
                            continue

                        # 3*48 of core weight
                        for cluster_idx, cluster in enumerate(self.clusters):
                            cluster_weight = wave_weights[cluster_idx]
                            cluster_core = cluster[i]
                            if cluster_weight is None:
                                cluster_core.is_finished = True
                                continue

                            core_weight = cluster_weight[cin_start_idx:cin_end_idx, :]
                            if core_weight.shape[0] < 3:
                                pad_len = 3 - core_weight.shape[0]
                                core_weight = torch.nn.functional.pad(core_weight, (0, 0, 0, pad_len))

                            cluster_core.configure_fc_weights(core_weight)
                            cluster_core.init_for_linear_tile(
                                accumulators[cluster_idx],
                                i,
                                core_input,
                                valid_pus[cluster_idx],
                            )

                # 2. 后端 accumulator 先处理上一拍提交的请求。
                for accumulator in accumulators:
                    accumulator.tick()

                # 3. Bank 写回机制
                for cluster in self.clusters:
                    for core in cluster:
                        if not core.is_finished or (core.pool is not None and core.pool.has_pending()):
                            if core.pool is not None:
                                core.pool.set_cycle(self.retire_trace_cycle)
                            core.pool.try_flush_to_accumulator()

                for cluster in self.clusters:
                    for core in cluster:
                        core.retire_trace_cycle = self.retire_trace_cycle

                # 4. 驱动计算单元 (底层已优化为仅驱动 pu0)
                for cluster in self.clusters:
                    for core in cluster:
                        if not core.is_finished:
                            core.tick_compute_linear()

                # 5. SplitUnit produces a package after Core; it becomes
                # visible to Core in the next cycle.
                for cluster in self.clusters:
                    for core in cluster:
                        if not core.is_finished:
                            core.split_unit.tick()

                self._sample_post_split_phases()
                time_step_compute_cycles += 1
                self.retire_trace_cycle += 1

            # 退出 While 循环代表：当前 TBL Tile 已经完整地遍历了所有 Cin，Bank 中的 Psum 全部就绪并清空
            post_nu_num = valid_TBL * wave_cout
            spike_write_data = post_nu_num * self.ow
            self.global_stats.writes['g_psum'] += spike_write_data
            self.global_stats.data_moved['mp'] += spike_write_data
            
            lif_cycle = ceil_a_by_b(post_nu_num, self.lif_num)
            
            # 加上脉动阵列排空的尾部延迟
            self.compute_cycles += (time_step_compute_cycles + systolic_delay)
            tile_compute_cycles = time_step_compute_cycles + systolic_delay
            if TBL_idx == 0:
                group_total_cycles += load_act_cycle + tile_compute_cycles
            else:
                group_total_cycles += max(tile_compute_cycles, load_act_cycle)

        
        # ==========================================================
        # 📈 状态缩放：将单组代表的统计结果，按 Cout_group_nums 等比放大
        # ==========================================================
        # compute and lif is pipelined,so only the last lif cycle is counted
        self.total_cycles = group_total_cycles * Cout_group_nums + lif_cycle
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
            accumulators,
            representative_compute_cycles,
        )
        self._print_performance_counters()

        # 同步给 global_stats 对象
        self.global_stats.total_cycles = self.total_cycles 
        self.global_stats.compute_cycles = self.compute_cycles 

        # 统计打印信息
        print("\n" + "="*60)
        print("--- Cluster CAS report (linear layer) ---")
        # print(f"参与计算 Core 数量: {self.num_cores} (PU 架构: 1D 脉动阵列)")
        print(f"张量切分状态: Cout 被分为 {Cout_group_nums} 组 | TBL 被切分为 {TBL_tile_nums} 个块")
        print(f"总计全局执行周期: {self.total_cycles}")
        print(f"纯内核计算周期: {self.compute_cycles}")
        
        stalls_total = sum(acc.stats['stalled_requests'] for acc in accumulators)
        print(f"\n[后端存储反压 (Backend Stall) 评估]")
        print(f"  因 FIFO 满导致总线拒收的总次数: **{stalls_total}** 次")
            
        print("\n[各 Core 微架构停顿画像]")
        total_frontend_stalls = 0
        total_psum_alloc_stalls = 0
        total_retire_backpressure_stalls = 0
        for i, core in enumerate(self.cores):
            core_stalls = getattr(core, 'psum_buffer_alloc_stalls', getattr(core, 'stalls', 0))
            retire_stalls = getattr(core, 'retire_backpressure_stalls', 0)
            frontend_stalls = getattr(core, 'frontend_stalls', 0)
            total_psum_alloc_stalls += core_stalls
            total_retire_backpressure_stalls += retire_stalls
            total_frontend_stalls += frontend_stalls
            print(
                f"  -> Core {i}: [后端] Psum分配失败累积停顿: {core_stalls:4d} 拍 | "
                f"[后端] Retire写回反压累积: {retire_stalls:4d} 拍 | "
                f"[前端] Split 供数累积空转: {frontend_stalls:4d} 拍"
            )
        print(f"  代表 cluster 累计 PsumPool 无可用 buffer 计算停顿: **{total_psum_alloc_stalls}** 拍")
        print(f"  代表 cluster 累计 retire 写回反压后台挂起事件: **{total_retire_backpressure_stalls}** 拍")
        print(f"  代表 cluster 累计 SplitUnit 供数不足空转: **{total_frontend_stalls}** 拍")

        print("="*60 + "\n")

        # 注意：由于我们在 core.py 中去掉了全量 PU 驱动，导致返回的输出张量全是 0。
        # 既然我们只做周期评估，这里直接返回 None 即可。
        return None, self.global_stats
