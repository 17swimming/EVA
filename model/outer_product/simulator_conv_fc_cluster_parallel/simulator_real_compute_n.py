# FC 模式，core+pe都映射为不同的cout

import torch
from model.outer_product.simulator_conv_fc_cluster_parallel.Accumulator import Accumulator
from model.outer_product.simulator_conv_fc_cluster_parallel.core import Core
from model.utils import ceil_a_by_b, Stats
import collections

CORE_CYCLE_LABELS = {
    'compute': '正常计算',
    'frontend': 'SplitUnit 供数不足',
    'psum_alloc': 'Psum 分配停顿',
    'writeback': 'Psum 写回排空',
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
        psum_pool_rows=6,
        enabled_modes=(0, 1, 2),
        num_split=2,
    ):
        self.num_cores = num_cores
        self.num_pus = num_pus
        self.GBsize_MP = 0
        self.GBsize_weight = 0
        self.bandwidth = 2048  # 从片外到片上的带宽
        self.config_weight_bandwidth = 1152  # 从weight BUFFER到core的带宽. 一拍完成一个core的权重配置
        self.config_act_bandwidth = 24  # 从act BUFFER到core的带宽，一次写一行
        self.weight_size = 8
        self.mp_size = 16
        self.lif_num = 32

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
        self.psum_pool_rows = int(psum_pool_rows)
        self.enabled_modes = tuple(sorted(int(mode) for mode in enabled_modes))
        self.num_split = max(1, int(num_split))

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
            )
            for _ in range(num_cores)
        ]

        self.pe_cycles = torch.zeros((num_cores, 3))  

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

            # 初始化带有反压机制的全局 累加器
            accumulator = Accumulator(
                retire_column=self.retire_column,
                bank_h=self.oh,
                bank_w=self.ow,
                enable_reduce_tree=self.enable_reduce_tree,
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
                    tile_end = min(group_cout_end, cout_start + self.num_pus)
                    wave_kernel = kernel_tensor[cout_start:tile_end, :, :, :]
                    valid_pu = tile_end - cout_start
                    wave_cout = valid_pu
                    
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
               
                                # 建立任务队列：只包含当前 Group 对应的输入通道
                                cin_queue = list(range(cin_start, cin_end))
                                time_step_compute_cycles = 0
                                
                                for core in self.cores:
                                    core.is_finished = True

                                # 开启全局心跳
                                while (
                                    cin_queue
                                    or any(
                                        not core.is_finished
                                        for core in self.cores
                                    )
                                    or not accumulator.is_empty()
                                ):

                                    # 任务派发
                                    for i, core in enumerate(self.cores):
                                        if not core.is_finished:
                                            continue

                                        # 全零 Cin tile 不占用本拍的派发机会。当前 Core
                                        # 会继续检查队列，直到找到一个非零 tile 或队列耗尽。
                                        while cin_queue:
                                            local_cin = cin_queue.pop(0)
                                            # 输入已按 group 切片；全局 Cin 编号需换算成组内索引。
                                            # 权重仍使用下方传递的全局 local_cin，不改变数据位宽。
                                            current_cin_map = current_input_tile[local_cin - cin_start, :, :]
                                            
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
                                                H=current_cin_map.shape[0],
                                            )
                                            break

                                    # 每拍每个 Core 都采样，包括已完成、等待其他 Core 的空闲拍。
                                    # 在写回前记录排空状态，避免漏计最后一次写回/释放的周期。
                                    cycle_before = {
                                        core: (
                                            core.compute_issue_cycles,
                                            core.frontend_stalls,
                                            core.psum_buffer_alloc_stalls,
                                            not core.is_finished
                                            and core.split_unit.is_finished
                                            and not core.has_pending_issue()
                                            and core.pool is not None
                                            and core.pool.has_pending(),
                                        )
                                        for core in self.conv_core_cycle_counts
                                    }

                                    # (A) accumulator consumes requests from
                                    # the previous cycle first.
                                    accumulator.tick()

                                    # (B) 内存总线传输阶段：写回 Psum
                                    for core in self.cores:
                                        if not core.is_finished or (core.pool is not None and core.pool.has_pending()):
                                            core.pool.try_flush_to_accumulator()

                                    # (C) Core consumes packages already in the
                                    # FIFO at the start of this cycle.
                                    for core in self.cores:
                                        if not core.is_finished:
                                            core.tick_compute()

                                    # 各 Core 独立互斥分类，不要求其他 Core 同时停顿。
                                    # 成功计算与前端/分配失败由本拍计数增量判断；完成计算后
                                    # 首次 retire_old([]) 新产生的排空状态也计入写回等待。
                                    for core, (compute, frontend, psum_alloc, was_draining) in cycle_before.items():
                                        if core.compute_issue_cycles > compute:
                                            state = 'compute'
                                        elif core.frontend_stalls > frontend:
                                            state = 'frontend'
                                        elif core.psum_buffer_alloc_stalls > psum_alloc:
                                            state = 'psum_alloc'
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

                                    # (D) SplitUnit produces the next package
                                    # after Core; it is visible next cycle.
                                    for core in self.cores:
                                        if not core.is_finished:
                                            core.split_unit.tick()

                                    time_step_compute_cycles += 1
                                
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
                representative_compute_cycles,
                conv_cout_scale=Cout_group_nums_per_g,
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
            
            
            print("\n[一个cout_group中，各 Core 微架构停顿画像]")
            print(f" 全零通道数目: {self.all_zero_channel_counts} 个")
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
        input_tensor: [M, N_in] 
        every pe process one cout, all core process atmost 288 cout per tiles
        """
        self.global_stats = Stats()
        self.total_cycles = 0
        self.compute_cycles = 0
        self._reset_performance_counters()
        
        # 清空 Core 历史状态
        M, K = input_tensor.shape
        Cin_w, Cout = weight_matrix.shape
        assert K == Cin_w, f"FC Mapping Error: K mismatch {K} vs {Cin_w}"

        # ==========================================================
        # 🚨 核心切分逻辑：计算 Tile 的规模,每个PU放3个PE，每个PE表示一个cout
        # ==========================================================
        Cout_group_nums = ceil_a_by_b(Cout, self.num_cores * self.num_pus * 3)
        
        # psum阵列3x12个寄存器，
        m_tile = self.ow
        M_tile_nums = ceil_a_by_b(M, m_tile)
        # print(f"[*] 计算 M Tile 数量: {TBL_tile_nums}")
        # ==========================================================
        
        accumulator = Accumulator(
            retire_column=self.retire_column,
            bank_h=self.oh,
            bank_w=self.ow,
            enable_reduce_tree=self.enable_reduce_tree,
            num_pus=self.num_pus,
            mp_size=self.mp_size,
        )


        # 最外层：Cout 分组 (避免权重重复搬运)
        # for cout_idx in range(Cout_group_nums):
        # ==========================================================
        # 每组cout负载等价，我们仅对【第一组】进行真实时钟推演
        # 第一组内所有core的行为都一致，因此仅看一个core就行。
        # 但是这种方式，如果cout不满足288的倍数，就会有core空闲，cycles虽然正确，
        # 但是没法真实模拟core的利用率
        # ==========================================================   
        cout_start = 0
        tile_end = min(Cout, self.num_cores * self.num_pus * 3)
        valid_pu = min(self.num_pus, ceil_a_by_b(tile_end - cout_start, 3))
        # 每个m_tile和cout_tile都会把权重重新加载一次，
        # weight_data = K * self.weight_size * wave_cout
        # self.global_stats.reads['g_wgt'] += weight_data
        # self.global_stats.data_moved['kernel'] += weight_data

        # 每个cout_group的总周期
        cout_group_total_cycles = 0

        # 中层循环：对 M 序列进行安全分块 (M Tile)
        for M_idx in range(M_tile_nums):
            M_start = M_idx * m_tile
            M_end = min(M, M_start + m_tile)
            valid_M = M_end - M_start
            
            # --- 加载激活数据---
            act_data = valid_M * K * self.mp_size
            self.global_stats.reads['g_act'] += act_data
            self.global_stats.data_moved['act'] += act_data
            load_act_cycle = ceil_a_by_b(act_data, self.config_act_bandwidth)

            # -----每个m_tile都会完整的再加载一次权重----
            # 但是这里模拟的是从片外加载权重的时间
            valid_cout = tile_end - cout_start
            weight_data = self.weight_size * valid_cout * self.kernel_size
            # self.global_stats.reads['g_wgt'] += weight_data
            # self.global_stats.data_moved['kernel'] += weight_data
            load_kernel_cycle = ceil_a_by_b(weight_data, self.config_weight_bandwidth)
            cout_group_total_cycles += load_kernel_cycle



            # assign cout to core,all the cores process the same act_tile
            # then cores start compute
            # FC 模式下，所有core的输入都是12x3的tile，沿着k维度去取
            k_queue = list(range(0, K, 3))
            #
            m_tile_compute_cycles = 0
            # 各 Core 沿 N 维处理不同输出，但共享同一个 M/K 输入 tile。
            # 只要任意 K tile 非零，当前 M tile 才需要启动 accumulator/LIF。
            m_tile_has_work = False

            # 由于所有core的行为一致，因此仅看一个core就行。
            core = self.cores[0]
            # 内层循环，彻底遍历完所有的 K (累加完整的 Psum)
            while (
                k_queue
                or not core.is_finished
            ):
                # 1. 任务派发
                # 全零 K tile 直接跳过，并在同一拍
                # 继续为当前 Core 寻找下一个可执行的非零 tile。
                while core.is_finished and k_queue:
                    k_start_idx = k_queue.pop(0)
                    k_end_idx = min(K, k_start_idx + 3)

                    core_input = input_tensor[M_start:M_end, k_start_idx:k_end_idx]

                    # 补齐 3 列，防止越界
                    if core_input.shape[1] < 3:
                        pad_len = 3 - core_input.shape[1]
                        core_input = torch.nn.functional.pad(core_input, (0, pad_len))
                        
                    if torch.sum(core_input) == 0:
                        continue

                    m_tile_has_work = True

                    # 统计每个核被分配的tile的数量，每个都一样
                    for core_id in range(self.num_cores):
                        self.tile_assignment_counts[core_id] += 1

                    # 每次将三个 K 值映射到每个 PU 内的三个 PE。
                    core_weight = weight_matrix[k_start_idx:k_end_idx, cout_start:cout_start+self.num_pus*3]
                    # 补齐 3 行
                    if core_weight.shape[0] < 3:
                        pad_len = 3 - core_weight.shape[0]
                        core_weight = torch.nn.functional.pad(core_weight, (0, 0, 0, pad_len))

                    core.configure_fc_weights(core_weight)
                    # core_input直接维持12x3的shape就行，
                    # split会按照每ifmap的ow行对应psum的一行的方式split
                    core.init_for_linear_tile(
                        accumulator,
                        0,
                        core_input,
                        valid_pu,
                    )
                    break

                # 2. 后端 accumulator 先处理上一拍提交的请求。
                accumulator.tick()

                # 3. 流式退休机制
                if not core.is_finished or (core.pool is not None and core.pool.has_pending()):
                    core.pool.try_flush_to_accumulator()

                # 4. 驱动计算单元 (底层已优化为仅驱动 pu0)
                if not core.is_finished:
                    core.tick_compute_linear()

                # 5. SplitUnit produces a package after Core; it becomes
                # visible to Core in the next cycle.
                if not core.is_finished:
                    core.split_unit.tick()

                m_tile_compute_cycles += 1

            drain_cycles = 0
            if m_tile_has_work:
                # All K tasks and their Psum retire operations have completed.
                # Drain the full physical accumulator through 4 * num_pus LIFs.
                # The signal itself takes no cycle; the first transfer is next tick.
                accumulator.request_drain()
                while not accumulator.is_empty():
                    accumulator.tick()
                    drain_cycles += 1

                post_nu_num = valid_M * self.num_cores * self.num_pus * 3
                # self.global_stats.writes['g_psum'] += post_nu_num * self.mp_size
                self.global_stats.data_moved['mp'] += post_nu_num * self.mp_size
            
            self.compute_cycles += m_tile_compute_cycles
            # 三个阶段，加载激活，计算，lif计算，
            # 一个m_tile计算后得等lif,但是可以加载下一个m_tile的act
            if M_idx == 0:
                cout_group_total_cycles += load_act_cycle + m_tile_compute_cycles + drain_cycles
            else:
                cout_group_total_cycles += max(m_tile_compute_cycles + drain_cycles, load_act_cycle)

        # ==========================================================
        # 完成所有m_tile后，将单组代表的统计结果，按 Cout_group_nums 等比放大
        # ==========================================================
        # Each M tile waits for its drain before reusing the accumulator.
        self.total_cycles = cout_group_total_cycles * Cout_group_nums
        representative_compute_cycles = self.compute_cycles
        self.compute_cycles *= Cout_group_nums
        self.global_stats.accumulator_drain_cycles = accumulator.stats['drain_cycles'] * Cout_group_nums
        self.global_stats.accumulator_drain_requests = accumulator.stats['drain_requests'] * Cout_group_nums
        self.global_stats.accumulator_drained_words = accumulator.stats['drained_words'] * Cout_group_nums
        self.global_stats.accumulator_drained_bits = accumulator.stats['drained_bits'] * Cout_group_nums
        self.global_stats.accumulator_lif_units = accumulator.lif_units
        
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

        # 同步给 global_stats 对象
        self.global_stats.total_cycles = self.total_cycles 
        self.global_stats.compute_cycles = self.compute_cycles 

        # 统计打印信息
        print("\n" + "="*60)
        print("--- CAS report (linear layer) ---")
        print(f"张量切分状态: Cout 被分为 {Cout_group_nums} 组 | M 被切分为 {M_tile_nums} 个块")
        print(f"总计全局执行周期: {self.total_cycles}")
        print(f"纯内核计算周期: {self.compute_cycles}")
        print(f"Accumulator 排空/LIF 周期: {self.global_stats.accumulator_drain_cycles}")
        
        print("\n[各 Core 微架构停顿画像]")
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
