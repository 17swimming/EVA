"""
   通用模拟器运行脚本 - 支持命令行参数配置
   
   使用方式:
   python simulator_runner.py --network SDTrack_FE108 --num_cores 8 --num_pus 16 --bank_h 32 --bank_w 32 --fetch_rows 8 --split_fifo_depth 2
   
   参数说明:
   --network: 指定要运行的网络名称 (如 SDTrack_FE108, TIM_NCARS, SLTNet_DDD17, QKFormer_DVSGesture, EMSYOLO_GEN1)
   --num_cores: 核心数量 (默认: 8)
   --num_pus: PU数量 (默认: 16)
   --bank_h: bank高度 (默认: 32)
   --bank_w: bank宽度 (默认: 32)
   --fetch_rows: fetch行数 (默认: 8)
   --split_fifo_depth: split fifo深度 (默认: 2)
"""

import os
import sys
import time
import argparse
import importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))))
# 设置环境变量以解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
# # 这个是shift中使用if_reg做多行预取的模拟器
# from model.outer_product.simulator_conv_fc_tile_PsumPool_4bank_DEBUG.simulator_real_compute import OutProductSimulator
# # 这个是shift中对ifmap使用了bitmask的模拟器
# from model.outer_product.simulator_conv_fc_tile_PsumPool_4bank_plus.simulator_real_compute import OutProductSimulator
# # 这个是理想模型，已经使用了3种model，且默认没有split和bank的读写限制
# from model.outer_product.simulator_conv_fc_tile_full1.simulator_real_compute import OutProductSimulator
import math
import csv
import ast
from model.baselines import run_prosperity_conv, run_prosperity_fc, run_gustavSNN_fc, run_gustavSNN_conv, run_PTB_layer_with_lif, run_prosperity_layer_with_lif, run_DLA_layer
from model.baselines import PROSPERITY_MEM_IF_WIDTH
from model.dla_cycle_model import DLA_MEMORY_BITS_PER_CYCLE
from model.utils import ceil_a_by_b, img2col, parse_spike_field , expand_integer_spike_to_binary
from model.outer_product.simulator_conv_fc_cluster_parallel.accumulator_bank_lookahead_study.reporting import result_columns

SELF_ATTENTION_TYPES = {'SSA', 'Spiking_Self_Attention'}


# ==========================================
# 🚀 双向日志记录器 (终端 + 文件)
# ==========================================
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='通用模拟器运行脚本')
    
    # 网络配置
    parser.add_argument('--network', type=str, required=True,
                        choices=['SpikeET_FE108', 'SDTrack_FE108', 'TIM_NCARS', 'SpikeBRGNet_DDD17', 'QKFormer_DVSGesture', 'EMSYOLO_GEN1'],
                        help='指定要运行的网络名称')
    
    parser.add_argument('--simulator', type=str, default='full1',
                        choices=['DEBUG', 'plus', 'pro', 'cluster', 'cluster_parallel', 'vector', 'full1', 'full1_without_model2'],
                        help='Select simulator variant')
    parser.add_argument('--accumulator-variant', choices=['k', 'lookahead', '1104', '1106'], default='k',
                        help='cluster_parallel: select the original _k or bank-lookahead accumulator')
    parser.add_argument('--retire-packets-per-cycle', type=int, choices=[1, 2], default=2,
                        help='lookahead: maximum bundles submitted by each Core per cycle')
    parser.add_argument('--prefetch-cycles', type=int, default=0,
                        help='lookahead: tile/weight prefetch latency in cycles')
    parser.add_argument('--output-root', type=str, default=None,
                        help='Override simulator output root directory')
    parser.add_argument('--eva-only', action='store_true', help='仅运行 EVA，不执行其他加速器基线')
    parser.add_argument('--max-test-cases', type=int, default=None, help='只运行清单前N项，保留原始测试编号')
    parser.add_argument('--bundle-buffer-depth', type=int, default=None,
                        help='1104: 每个 core 的 accumulator bundle buffer 深度，默认3')
    parser.add_argument('--bank-lookahead-entries', type=int, default=None,
                        help='1104: 每个 core 的 bank 仲裁候选数，默认3')
    parser.add_argument('--lif-units', type=int, choices=(48,64), default=None,
                        help='1104: LIF神经元并行数量，默认48，64对应每拍4个点')
    
    # Simulator配置
    parser.add_argument('--num_cores', type=int, default=8, help='核心数量')
    parser.add_argument('--num_pus', type=int, default=16, help='PU数量')
    parser.add_argument('--bank_h', type=int, default=32, help='bank高度')
    parser.add_argument('--bank_w', type=int, default=32, help='bank宽度')
    parser.add_argument('--fetch_rows', type=int, default=8, help='fetch行数')
    parser.add_argument('--split_fifo_depth', type=int, default=2, help='split fifo深度')
    parser.add_argument('--num_split', type=int, default=2, help='cluster_parallel中并行Split数量')
    parser.add_argument('--num_sub_banks', type=int, default=4, help='global bank中的子bank数目；cluster simulator会忽略该参数')
    parser.add_argument('--num_clusters', type=int, default=1, help='cluster simulator cluster count')
    parser.add_argument('--retire_column', type=int, default=3, help='columns retired per psum row each cycle')
    parser.add_argument('--psum_pool_rows', type=int, default=6, help='cluster simulator psum pool rows per core')
    parser.add_argument('--disable_reduce_tree', action='store_true',
                        help='cluster ablation: reject same-point accumulator writes instead of reducing them')
    parser.add_argument('--num_reduce_trees', type=int, default=None,
                        help='cluster_parallel: shared trees per cycle; omitted=unlimited, 0=no reduction')
    parser.add_argument(
        '--enabled_modes',
        type=str,
        default='0,1,2',
        help='comma-separated Conv split modes; mode 0 is mandatory (for example: 0,1,2)',
    )
    
    # 时间步和批次配置
    parser.add_argument('--T', type=int, default=4, help='时间步数量')
    parser.add_argument('--B', type=int, default=1, help='批次大小')
    
    args = parser.parse_args()
    if args.max_test_cases is not None and args.max_test_cases < 1:
        parser.error('--max-test-cases必须为正整数')
    if args.accumulator_variant in ('1104','1106'):
        args.lif_units = 48 if args.lif_units is None else args.lif_units
        args.bundle_buffer_depth = 3 if args.bundle_buffer_depth is None else args.bundle_buffer_depth
        args.bank_lookahead_entries = 3 if args.bank_lookahead_entries is None else args.bank_lookahead_entries
        if not 1 <= args.bank_lookahead_entries <= args.bundle_buffer_depth:
            parser.error('1104 requires 1 <= bank-lookahead-entries <= bundle-buffer-depth')
    elif args.bundle_buffer_depth is not None or args.bank_lookahead_entries is not None:
        parser.error('bundle buffer 参数当前仅适用于 --accumulator-variant 1104')
    if args.accumulator_variant not in ('1104','1106') and args.lif_units is not None:
        parser.error('--lif-units当前仅适用于 --accumulator-variant 1104')
    if args.accumulator_variant != 'k' and args.simulator != 'cluster_parallel':
        parser.error('--accumulator-variant lookahead/1104 requires --simulator cluster_parallel')
    if args.prefetch_cycles < 0:
        parser.error('--prefetch-cycles must be nonnegative')
    if args.accumulator_variant in ('lookahead', '1104', '1106') and args.num_reduce_trees not in (None, 0):
        parser.error('lookahead has no cross-Core reduce trees; use --num_reduce_trees 0')
    if args.num_reduce_trees is not None:
        if args.num_reduce_trees < 0:
            parser.error('--num_reduce_trees must be nonnegative')
        if args.simulator != 'cluster_parallel':
            parser.error('--num_reduce_trees currently requires --simulator cluster_parallel')
    try:
        enabled_modes = tuple(
            sorted({int(mode.strip()) for mode in args.enabled_modes.split(',') if mode.strip()})
        )
    except ValueError as error:
        parser.error(f"invalid --enabled_modes value: {error}")
    if not enabled_modes or 0 not in enabled_modes or any(mode not in (0, 1, 2) for mode in enabled_modes):
        parser.error('--enabled_modes must contain mode 0 and only use 0, 1, and 2')
    args.enabled_modes = enabled_modes
    return args


def cycle_speedup(reference_cycles, eva_cycles):
    """跳过全零输入后周期可以为零，此时加速比记为未定义。"""
    return reference_cycles / eva_cycles if eva_cycles > 0 else float('nan')


def supports_eva_spike(test_case):
    """本项目加速器支持二值脉冲和最大值不超过四的整数脉冲。"""
    kind = test_case.get('spike_kind', 'skip')
    factor = test_case.get('spike_factor')
    return kind == 'binary' or (kind == 'integer' and factor is not None and 1 <= factor <= 4)


def load_test_cases(network_name):
    """加载指定网络的测试用例"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_cases_file = os.path.join(current_dir, f'{network_name}.csv')
    
    if not os.path.exists(test_cases_file):
        raise FileNotFoundError(f"测试用例文件不存在: {test_cases_file}")
    
    test_cases = []
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 跳过空行或无效行（检查type字段是否为空）
            if not row.get('type') or not row['type'].strip():
                continue
            
            test_case = {}
            test_case['type'] = row['type']
            # 处理spike字段：TRUE、数字或者xx_float
            test_case['spike'] = row['spike'].strip()
            test_case.update(parse_spike_field(test_case['spike']))

            # 处理Cout/dim字段，处理NONE值
            cout_val = row.get('Cout/dim', 'NONE').strip().upper()
            if cout_val == 'NONE':
                test_case['Cout'] = 0  # 设置默认值
            else:
                test_case['Cout'] = int(cout_val)
            # 处理file字段，转换为绝对路径（相对于CSV文件所在目录）
            file_path = row['file']
            if file_path.startswith('./'):
                file_path = file_path[2:]
            # 将相对路径转换为绝对路径
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            abs_file_path = os.path.join(project_root, file_path)
            test_case['file'] = os.path.normpath(abs_file_path)
            # 处理shape字段（如果存在）
            if 'shape' in row:
                test_case['shape'] = ast.literal_eval(row['shape'])
            # 处理sparsity字段
            test_case['sparsity'] = float(row['sparsity'])            
            test_cases.append(test_case)
    
    return test_cases


def run():
    args = parse_args()
    if args.eva_only:
        # 保留原有网络映射和结果格式，明确关闭全部基线计算及其脉冲展开。
        for name in ('run_prosperity_conv', 'run_prosperity_fc', 'run_gustavSNN_fc',
                     'run_gustavSNN_conv', 'run_PTB_layer_with_lif'):
            globals()[name] = lambda *values, **options: float('nan')
        globals()['run_prosperity_layer_with_lif'] = lambda *values, **options: (float('nan'), None)
        globals()['run_DLA_layer'] = lambda *values, **options: (float('nan'), None, None)
        globals()['expand_integer_spike_to_binary'] = lambda input_tensor, factor: input_tensor
    
    T = args.T
    B = args.B
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_root = (
        os.path.abspath(args.output_root)
        if args.output_root
        else os.path.join(project_root, 'output/simulator_outproduct')
    )
    
        # 根据命令行参数动态导入模拟器
    if args.simulator == 'DEBUG':
        from model.outer_product.simulator_conv_fc_tile_PsumPool_4bank_DEBUG.simulator_real_compute import OutProductSimulator
        print(f"使用模拟器: simulator_conv_fc_tile_PsumPool_4bank_DEBUG (if_reg多行预取)")
         # 创建模拟器实例
        simulator_sim = OutProductSimulator(
            num_cores=args.num_cores,
            num_pus=args.num_pus,
            oh=args.bank_h,
            ow=args.bank_w,
            fetch_rows=args.fetch_rows,
            split_fifo_depth=args.split_fifo_depth,
            num_sub_banks=args.num_sub_banks
        )
        # 创建输出文件夹
        output_folder = os.path.join(output_root, f'{args.network}_DEBUG')
        os.makedirs(output_folder, exist_ok=True)


    elif args.simulator == 'plus':
        from model.outer_product.simulator_conv_fc_tile_PsumPool_4bank_plus.simulator_real_compute import OutProductSimulator
        print(f"使用模拟器: simulator_conv_fc_tile_PsumPool_4bank_plus (bitmask)")
         # 创建模拟器实例
        simulator_sim = OutProductSimulator(
            num_cores=args.num_cores,
            num_pus=args.num_pus,
            oh=args.bank_h,
            ow=args.bank_w,
            fetch_rows=args.fetch_rows,
            split_fifo_depth=args.split_fifo_depth,
            num_sub_banks=args.num_sub_banks
        )
        # 创建输出文件夹
        output_folder = os.path.join(output_root, f'{args.network}_plus')
        os.makedirs(output_folder, exist_ok=True)


    elif args.simulator == 'pro':
        from model.outer_product.simulator_conv_fc_tile_PsumPool_4bank_pro.simulator_real_compute import OutProductSimulator
        print(f"使用模拟器: simulator_conv_fc_tile_PsumPool_4bank_pro (bitmask+mode2)")
         # 创建模拟器实例
        simulator_sim = OutProductSimulator(
            num_cores=args.num_cores,
            num_pus=args.num_pus,
            oh=args.bank_h,
            ow=args.bank_w,
            fetch_rows=args.fetch_rows,
            split_fifo_depth=args.split_fifo_depth,
            num_sub_banks=args.num_sub_banks
        )
        # 创建输出文件夹
        output_folder = os.path.join(output_root, f'{args.network}_pro')
        os.makedirs(output_folder, exist_ok=True)


    elif args.simulator == 'cluster':
        from model.outer_product.simulator_conv_fc_cluster.simulator_real_compute import OutProductSimulator
        print("使用模拟器: simulator_conv_fc_cluster (cluster+stream retire)")
        simulator_sim = OutProductSimulator(
            num_cores=args.num_cores,
            num_pus=args.num_pus,
            oh=args.bank_h,
            ow=args.bank_w,
            fetch_rows=args.fetch_rows,
            split_fifo_depth=args.split_fifo_depth,
            num_clusters=args.num_clusters,
            retire_column=args.retire_column,
            enable_reduce_tree=not args.disable_reduce_tree,
            psum_pool_rows=args.psum_pool_rows,
        )
        output_folder = os.path.join(output_root, f'{args.network}_cluster')
        os.makedirs(output_folder, exist_ok=True)


    elif args.simulator == 'cluster_parallel':
        if args.accumulator_variant in ('1104','1106'):
            version = '20261104' if args.accumulator_variant == '1104' else '20261106'
            implementation = importlib.import_module(f'zxy.{version}.model.simulator')
            OutProductSimulator = implementation.OutProductSimulator
            print(f'使用模拟器: zxy/{version}/model/simulator.py')
            accumulator_kwargs = dict(retire_packets_per_cycle=args.retire_packets_per_cycle,
                                      prefetch_cycles=args.prefetch_cycles,
                                      bundle_buffer_depth=args.bundle_buffer_depth,
                                      bank_lookahead_entries=args.bank_lookahead_entries,
                                      lif_units=args.lif_units)
        elif args.accumulator_variant == 'lookahead':
            from model.outer_product.simulator_conv_fc_cluster_parallel.accumulator_bank_lookahead_study.simulator_real_compute import OutProductSimulator
            print('使用模拟器: cluster_parallel + bank_lookahead')
            accumulator_kwargs = dict(
                psum_alloc_mode='halfrow',
                retire_packets_per_cycle=args.retire_packets_per_cycle,
                bank_lookahead=True,
                prefetch_cycles=args.prefetch_cycles,
            )
        else:
            from model.outer_product.simulator_conv_fc_cluster_parallel._k.simulator_real_compute import OutProductSimulator
            print('使用模拟器: simulator_conv_fc_cluster_parallel._k')
            accumulator_kwargs = {}
        simulator_sim = OutProductSimulator(
            num_cores=args.num_cores,
            num_pus=args.num_pus,
            oh=args.bank_h,
            ow=args.bank_w,
            fetch_rows=args.fetch_rows,
            split_fifo_depth=args.split_fifo_depth,
            retire_column=args.retire_column,
            enable_reduce_tree=not args.disable_reduce_tree,
            psum_pool_rows=args.psum_pool_rows,
            num_split=args.num_split,
            num_reduce_trees=args.num_reduce_trees,
            **accumulator_kwargs,
        )
        output_folder = os.path.join(output_root, f'{args.network}_cluster_parallel')
        if args.accumulator_variant in ('1104','1106'):
            output_folder = os.path.join(output_root, f'{args.network}_{args.accumulator_variant}')
        os.makedirs(output_folder, exist_ok=True)


    elif args.simulator == 'vector':
        from model.outer_product.simulator_conv_fc_tile_vector.simulator_real_compute import OutProductSimulator
        print(f"使用模拟器: simulator_conv_fc_tile_vector (local psum SRAM + MP SRAM)")
         # 创建模拟器实例
        simulator_sim = OutProductSimulator(
            num_cores=args.num_cores,
            num_pus=args.num_pus,
            oh=args.bank_h,
            ow=args.bank_w,
            fetch_rows=args.fetch_rows,
            split_fifo_depth=args.split_fifo_depth,
            num_sub_banks=args.num_sub_banks
        )
        # 创建输出文件夹
        output_folder = os.path.join(output_root, f'{args.network}_vector')
        os.makedirs(output_folder, exist_ok=True)


    elif args.simulator == 'full1': # full1
        from model.outer_product.simulator_conv_fc_tile_full1.simulator_real_compute import OutProductSimulator
        print(f"使用模拟器: simulator_conv_fc_tile_full1 (理想模型)")
         # 创建模拟器实例
        simulator_sim = OutProductSimulator(
            num_cores=args.num_cores,
            num_pus=args.num_pus,
            oh=args.bank_h,
            ow=args.bank_w
        )
        # 创建输出文件夹
        output_folder = os.path.join(output_root, f'{args.network}_full1')
        os.makedirs(output_folder, exist_ok=True)
    
    elif args.simulator == 'full1_without_model2': # full1
        from model.outer_product.simulator_conv_fc_tile_full1.simulator_real_compute_without_model2 import OutProductSimulator
        print(f"使用模拟器: simulator_conv_fc_tile_full1 (理想模型)")
         # 创建模拟器实例
        simulator_sim = OutProductSimulator(
            num_cores=args.num_cores,
            num_pus=args.num_pus,
            oh=args.bank_h,
            ow=args.bank_w
        )
        # 创建输出文件夹
        output_folder = os.path.join(output_root, f'{args.network}_full1_without_model2')
        os.makedirs(output_folder, exist_ok=True)
   


    # 🚀 生成日志文件名（使用配置参数）
    if args.simulator in ('pro', 'cluster', 'cluster_parallel'):
        simulator_sim.set_enabled_modes(args.enabled_modes)
    if hasattr(simulator_sim, 'configure_reporting'):
        simulator_sim.configure_reporting(output_folder, vars(args))

    log_filename = os.path.join(output_folder, 
        f"simulation_log_{args.network}_T{args.T}_B{args.B}_cores{args.num_cores}_pus{args.num_pus}_bank{args.bank_h}x{args.bank_w}_fetch{args.fetch_rows}_fifo{args.split_fifo_depth}{'' if args.simulator in ('cluster', 'cluster_parallel') else '_subbanks' + str(args.num_sub_banks)}.txt")
    
    # 接管标准输出
    logger = Logger(log_filename)
    sys.stdout = logger
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 仿真任务启动！")
    print(f"网络名称: {args.network}")
    print(f"配置: {args.num_cores} Core，{args.num_pus} lane，"
          f"result_array {args.bank_h}×{args.bank_w}，mode {args.enabled_modes}")

    # 加载测试用例
    test_cases = load_test_cases(args.network)
    if args.max_test_cases is not None:
        test_cases = test_cases[:args.max_test_cases]
    print(f"已加载 {len(test_cases)} 个测试用例\n")

    # 准备存储结果的列表
    results = []
    ours_total_cycle_num = 0
    dla_total_cycle_num = 0
    prosperity_total_cycle_num = 0
    ptb_total_cycle_num = 0
    gustav_total_cycle_num = 0


    # 🚀 优化：全局关闭梯度计算，极大节省内存和提速
    with torch.no_grad():
        # 记录总开始时间
        total_start_time = time.time()

        # 上一层的输出脉冲是否能直接存在片上
        spike_in_buffer = False
        # 遍历测试数据集
        for i, test_case in enumerate(test_cases):
            start_time = time.time()
            
            # 解析 groups 参数与算子类型
            type_str = test_case['type']
            groups = 1
            if '_g' in type_str:
                parts = type_str.split('_g_')
                groups = int(parts[-1])
                clean_type = parts[0]
            else:
                clean_type = type_str
            
            # ==========================================================
            # 对于spike字段，如果True，支持，如果是xx_float，跳过该层，如果是数字。
            # simulator可以直接支持，但是由于prosperity和GustavSNN只能支持0/1脉冲，
            # 所以就需要根据spike字段作时间步展开，比如spike=4，就需要把这个npy文件的时间步x4，展开为0/1后，再作为gustav和prosperity的输入。
            # ==========================================================
            spike_val = test_case['spike']
            spike_kind = test_case.get('spike_kind', 'skip')
            spike_factor = test_case.get('spike_factor')
            valid_spike = supports_eva_spike(test_case)
            if not valid_spike:
                print(
                    f"\n=== 跳过测试用例 {i+1}/{len(test_cases)}: "
                    f"spike={spike_val}，当前只支持 TRUE 或 1–4 的整数 spike ==="
                )
                continue
            # ==========================================================
            
            # 加载输入张量，并根据spike_kind决定是否展开为二进制脉冲
            #todo，对于attention类型的算子，不是input_tensor，而是根据file字段加载q_act,将file中Q换成K和V，也可以取到k_act和v_act
            if clean_type in SELF_ATTENTION_TYPES:
                act_q = np.load(test_case['file'])
                file_path_k = test_case['file'].replace('_Q_', '_K_')
                act_k = np.load(file_path_k)
                file_path_v = test_case['file'].replace('_Q_', '_V_')
                act_v = np.load(file_path_v)
                act_q = torch.from_numpy(act_q).to(torch.int8)
                act_k = torch.from_numpy(act_k).to(torch.int8)
                act_v = torch.from_numpy(act_v).to(torch.int8)
            elif clean_type == 'Token_QK_Attention':
                act_q = np.load(test_case['file'])
                file_path_k = test_case['file'].replace('_Q_', '_K_')
                act_k = np.load(file_path_k)
                act_q = torch.from_numpy(act_q).to(torch.int8)
                act_k = torch.from_numpy(act_k).to(torch.int8)
            else:
                if not os.path.exists(test_case['file']):
                    print(f"警告：文件不存在，跳过测试用例: {test_case['file']}")
                    continue
                input_tensor = np.load(test_case['file']) 
                input_tensor = torch.from_numpy(input_tensor).to(torch.int8)
                # 一维卷积未提供批次维度时，统一按单批次处理。
                if clean_type.startswith('conv1d_') and input_tensor.ndim == 3:
                    input_tensor = input_tensor.unsqueeze(1)
                if spike_kind == 'integer':
                    input_tensor_baseline = expand_integer_spike_to_binary(input_tensor, spike_factor)
                else:
                    input_tensor_baseline = input_tensor

            input_num_elements = input_tensor.numel()
            nonzero_elements = torch.sum(input_tensor != 0).item()
            dense = nonzero_elements / input_num_elements
            # 检查input_tensor的稠密度是否与sparsity一致
            # if dense != (1-test_case['sparsity']):
            #     print(f"警告：input_tensor的稠密度与保存输入时统计得到的稠密度不一致，跳过测试用例: {test_case['file']}")
            #     continue

            #开始测试
            print(f"\n=== 测试用例 {i+1}/{len(test_cases)}: {os.path.basename(test_case['file'])} ===")
            if hasattr(simulator_sim, 'begin_test_case'):
                simulator_sim.begin_test_case(i+1)
            # 根据'clean_type'选择simulator中的函数
            if clean_type == 'conv2d_(3, 3)_(1, 1)_(1, 1)' and valid_spike:
                DLA_cycles, _, _ = run_DLA_layer(
                    input_tensor, clean_type, test_case['Cout'], groups=groups)
                Cout = test_case['Cout']
                # Some QKFormer convolution inputs are stored as [T, Cin, H, W].
                # The cluster simulator accepts this form as a batch-one input.
                if input_tensor.ndim == 4:
                    T, Cin, h, w = input_tensor.shape
                    B = 1
                    input_tensor_baseline_conv = input_tensor_baseline.unsqueeze(1)
                elif input_tensor.ndim == 5:
                    T, B, Cin, h, w = input_tensor.shape
                    input_tensor_baseline_conv = input_tensor_baseline
                else:
                    raise ValueError(f"Unsupported 3x3 conv input shape: {input_tensor.shape}")

                oh = h
                ow = w
                Cin_g = Cin // groups
                Cout_g = Cout // groups
                kernel = torch.zeros(Cout, Cin, 3, 3).to(torch.int8)
                _ , stats = simulator_sim.run_convolution(input_tensor, kernel, padding=1, stride=1, groups=groups, spike_in_buffer=spike_in_buffer)

                # Prosperity: conv 后必接 LIF，主体随 GEMM 流水隐藏，只追加排空尾巴
                prosperity_cycles = run_prosperity_conv(Cout, input_tensor_baseline_conv, 3, 1, 1, groups=groups)
                T_lif = T * (spike_factor or 1)
                num_lif_neurons = Cout * oh * ow
                prosperity_cycles, lif_stats = run_prosperity_layer_with_lif(
                    prosperity_cycles, num_lif_neurons, T_lif, B)
                print(f"Prosperity cycles: {prosperity_cycles}")
            
                gustav_cycles = run_gustavSNN_conv(Cout, input_tensor_baseline_conv, 3, 1, 1, groups=groups)
                print(f"Gustav cycles: {gustav_cycles}")

                PTB_cycles = run_PTB_layer_with_lif(
                    input_tensor_baseline_conv.flatten(0,1), "conv", 3, 1, 1, Cout, oh, ow,
                    time_steps=T_lif, num_neuron=num_lif_neurons * B)
                print(f"PTB cycles: {PTB_cycles}")

            
                # 判断上一层所有时间步的结果能否都存在片上
                # 只有conv 层 后面仍是conv层时，用这个功能，因为我的fc需要拼装。
                if (T * oh * ow * Cout) <= simulator_sim.GBsize_spike:
                    spike_in_buffer = True

                direct_conv_cycles = oh * (ow+2*1) * T * B * math.ceil(Cout_g/args.num_pus) * math.ceil(Cin_g/args.num_cores) * groups
                # DLA_cycles = h * w * 9 * T * B * ceil_a_by_b(Cout_g, 16) * ceil_a_by_b(Cin_g, 64) * groups
            
                dla_speedup = cycle_speedup(DLA_cycles, stats.total_cycles)
                gustav_speedup = cycle_speedup(gustav_cycles, stats.total_cycles)
                prosperity_speedup = cycle_speedup(prosperity_cycles, stats.total_cycles)
                ptb_speedup = cycle_speedup(PTB_cycles, stats.total_cycles)
            
                dla_total_cycle_num += DLA_cycles
                ours_total_cycle_num += stats.total_cycles
                gustav_total_cycle_num += gustav_cycles
                prosperity_total_cycle_num += prosperity_cycles
                ptb_total_cycle_num += PTB_cycles
            

                elapsed_time = time.time() - start_time
            
                eva_report = result_columns(stats)
                results.append({
                    'test_case': i+1,
                    'file': test_case['file'],
                    'type': test_case['type'],
                    'input_shape': input_tensor.shape,
                    'dense': dense,
                    'simulator_cycles': stats.total_cycles,
                    'num_mode0': eva_report['num_mode0'],
                    'ratio_mode0': eva_report['ratio_mode0'],
                    'num_mode1': eva_report['num_mode1'],
                    'ratio_mode1': eva_report['ratio_mode1'],
                    'num_mode2': eva_report['num_mode2'],
                    'ratio_mode2': eva_report['ratio_mode2'],
                    'EVA': eva_report['EVA'],
                    'LIF': eva_report['LIF'],
                    'DRAM': eva_report['DRAM'],
                    'ptb_cycles': PTB_cycles,
                    'ptb_speedup': ptb_speedup,
                    'direct_conv_cycles': direct_conv_cycles,
                    'dla_cycles': DLA_cycles,
                    'dla_speedup': dla_speedup,
                    'prosperity_cycles': prosperity_cycles,
                    'prosperity_speedup': prosperity_speedup,   
                    'gustav_cycles': gustav_cycles,
                    'gustav_speedup': gustav_speedup,
                    'elapsed_time': elapsed_time
                })

        
            elif clean_type == 'conv2d_(3, 3)_(2, 2)_(1, 1)' and valid_spike:
                DLA_cycles, _, _ = run_DLA_layer(
                    input_tensor, clean_type, test_case['Cout'], groups=groups)
                Cout = test_case['Cout']
                T,B, Cin, h, w = input_tensor.shape
            
                input_tensor_f = input_tensor.flatten(0,1) # (T*B, Cin, h, w)
                # 通过img2col转为矩阵[TB,M,K]
                input_matrix = img2col(input_tensor_f, kernel_size=3, stride=2, padding=1)
                input_matrix_our = input_matrix.flatten(0,1)   #[TBM,K]
                # 权重矩阵
                weight_matrix = torch.zeros(Cin*3*3, Cout).to(torch.int8)   #[K,Cout]

                output_tensor, stats = simulator_sim.run_fc(input_matrix_our, weight_matrix)
                            
                gustav_cycles = run_gustavSNN_conv(Cout, input_tensor_baseline, 3, 2, 1, groups=groups)
                print(f"Gustav cycles: {gustav_cycles}")

            
                oh = (h-3+2*1)//2 +1
                ow = (w-3+2*1)//2 +1
                Cin_g = Cin // groups
                Cout_g = Cout // groups

                # Prosperity: conv 后必接 LIF，主体随 GEMM 流水隐藏，只追加排空尾巴
                prosperity_cycles = run_prosperity_conv(Cout, input_tensor_baseline, 3, 2, 1, groups=groups)
                T_lif = T * (spike_factor or 1)
                num_lif_neurons = Cout * oh * ow
                prosperity_cycles, lif_stats = run_prosperity_layer_with_lif(
                    prosperity_cycles, num_lif_neurons, T_lif, B)

                PTB_cycles = run_PTB_layer_with_lif(
                    input_tensor_baseline.flatten(0,1), "conv", 3, 2, 1, Cout, oh, ow,
                    time_steps=T_lif, num_neuron=num_lif_neurons * B)

                direct_conv_cycles = oh * (ow+2*1) * T * B * math.ceil(Cout_g/args.num_pus) * math.ceil(Cin_g/args.num_cores) * groups
                # DLA_cycles = oh * ow * 9 * T * B * ceil_a_by_b(Cout_g, 16) * ceil_a_by_b(Cin_g, 64) * groups
            
                dla_speedup = cycle_speedup(DLA_cycles, stats.total_cycles)
                gustav_speedup = cycle_speedup(gustav_cycles, stats.total_cycles)
                prosperity_speedup = cycle_speedup(prosperity_cycles, stats.total_cycles)
                ptb_speedup = cycle_speedup(PTB_cycles, stats.total_cycles)
            
                dla_total_cycle_num += DLA_cycles
                ours_total_cycle_num += stats.total_cycles
                gustav_total_cycle_num += gustav_cycles
                prosperity_total_cycle_num += prosperity_cycles
                ptb_total_cycle_num += PTB_cycles
            

                elapsed_time = time.time() - start_time
            
                eva_report = result_columns(stats)
                results.append({
                    'test_case': i+1,
                    'file': test_case['file'],
                    'type': test_case['type'],
                    'input_shape': input_tensor.shape,
                    'dense': dense,
                    'simulator_cycles': stats.total_cycles,
                    'num_mode0': eva_report['num_mode0'],
                    'ratio_mode0': eva_report['ratio_mode0'],
                    'num_mode1': eva_report['num_mode1'],
                    'ratio_mode1': eva_report['ratio_mode1'],
                    'num_mode2': eva_report['num_mode2'],
                    'ratio_mode2': eva_report['ratio_mode2'],
                    'EVA': eva_report['EVA'],
                    'LIF': eva_report['LIF'],
                    'DRAM': eva_report['DRAM'],
                    'ptb_cycles': PTB_cycles,
                    'ptb_speedup': ptb_speedup,
                    'direct_conv_cycles': direct_conv_cycles,
                    'dla_cycles': DLA_cycles,
                    'dla_speedup': dla_speedup,
                    'prosperity_cycles': prosperity_cycles,
                    'prosperity_speedup': prosperity_speedup,   
                    'gustav_cycles': gustav_cycles,
                    'gustav_speedup': gustav_speedup,
                    'elapsed_time': elapsed_time
                })


            elif (clean_type == 'conv2d_(1, 3)_(1, 1)_(0, 1)' or clean_type == 'conv2d_(1, 3)_(1, 1)_(0, 2)') and valid_spike:
                DLA_cycles, _, _ = run_DLA_layer(
                    input_tensor, clean_type, test_case['Cout'], groups=groups)
                Cout = test_case['Cout']
                T,B, Cin, h, w = input_tensor.shape
            
                kernel = torch.zeros(Cout, Cin, 1, 3, device=input_tensor.device).to(torch.int8)
                padding = (0,1) if clean_type == 'conv2d_(1, 3)_(1, 1)_(0, 1)' else (0,2)
            
                output_tensor, stats = simulator_sim.run_convolution(input_tensor, kernel, padding=padding, stride=(1,1), groups=groups)
            
                prosperity_cycles = run_prosperity_conv(Cout, input_tensor_baseline, kernel_size=(1,3), stride=(1,1), padding=padding, groups=groups)
            
                gustav_cycles = run_gustavSNN_conv(Cout, input_tensor_baseline, kernel_size=(1,3), stride=(1,1), padding=padding, groups=groups)

            
            
                oh = (h - 1 + 2 * padding[0]) // 1 + 1
                ow = (w - 3 + 2 * padding[1]) // 1 + 1
                Cin_g = Cin // groups
                Cout_g = Cout // groups

                # Prosperity: conv 后必接 LIF，主体随 GEMM 流水隐藏，只追加排空尾巴
                T_lif = T * (spike_factor or 1)
                num_lif_neurons = Cout * oh * ow
                prosperity_cycles, lif_stats = run_prosperity_layer_with_lif(
                    prosperity_cycles, num_lif_neurons, T_lif, B)

                direct_conv_cycles = oh * (ow+2*padding[1]) * T * B * math.ceil(Cout_g/args.num_pus) * math.ceil(Cin_g/args.num_cores) * groups
                PTB_cycles = run_PTB_layer_with_lif(
                    input_tensor_baseline.flatten(0,1), "conv", (1, 3), (1, 1), padding, Cout, oh, ow,
                    time_steps=T_lif, num_neuron=num_lif_neurons * B)
                # DLA_cycles = oh * ow * 3 * T * B * ceil_a_by_b(Cout_g, 16) * ceil_a_by_b(Cin_g, 64) * groups
            
                dla_speedup = cycle_speedup(DLA_cycles, stats.total_cycles)
                gustav_speedup = cycle_speedup(gustav_cycles, stats.total_cycles)
                prosperity_speedup = cycle_speedup(prosperity_cycles, stats.total_cycles)
                ptb_speedup = cycle_speedup(PTB_cycles, stats.total_cycles)
            
                dla_total_cycle_num += DLA_cycles
                ours_total_cycle_num += stats.total_cycles
                gustav_total_cycle_num += gustav_cycles
                prosperity_total_cycle_num += prosperity_cycles
                ptb_total_cycle_num += PTB_cycles
            

                elapsed_time = time.time() - start_time
            
                eva_report = result_columns(stats)
                results.append({
                    'test_case': i+1,
                    'file': test_case['file'],
                    'type': test_case['type'],
                    'input_shape': input_tensor.shape,
                    'dense': dense,
                    'simulator_cycles': stats.total_cycles,
                    'num_mode0': eva_report['num_mode0'],
                    'ratio_mode0': eva_report['ratio_mode0'],
                    'num_mode1': eva_report['num_mode1'],
                    'ratio_mode1': eva_report['ratio_mode1'],
                    'num_mode2': eva_report['num_mode2'],
                    'ratio_mode2': eva_report['ratio_mode2'],
                    'EVA': eva_report['EVA'],
                    'LIF': eva_report['LIF'],
                    'DRAM': eva_report['DRAM'],
                    'ptb_cycles': PTB_cycles,
                    'ptb_speedup': ptb_speedup,
                    'direct_conv_cycles': direct_conv_cycles,
                    'dla_cycles': DLA_cycles,
                    'dla_speedup': dla_speedup,
                    'prosperity_cycles': prosperity_cycles,
                    'prosperity_speedup': prosperity_speedup,   
                    'gustav_cycles': gustav_cycles,
                    'gustav_speedup': gustav_speedup,
                    'elapsed_time': elapsed_time
                })

            elif (clean_type == 'conv2d_(3, 1)_(1, 1)_(1, 0)' or clean_type == 'conv2d_(3, 1)_(1, 1)_(2, 0)') and valid_spike:
                DLA_cycles, _, _ = run_DLA_layer(
                    input_tensor, clean_type, test_case['Cout'], groups=groups)
                Cout = test_case['Cout']
                T,B, Cin, h, w = input_tensor.shape

                k_h, k_w = 3, 1
                s_h, s_w = 1, 1
                if clean_type == 'conv2d_(3, 1)_(1, 1)_(1, 0)':
                    p_h, p_w = 1, 0
                elif clean_type == 'conv2d_(3, 1)_(1, 1)_(2, 0)':
                    p_h, p_w = 2, 0   

                input_tensor_transposed = input_tensor.transpose(-2, -1)
                input_tensor_baseline_transposed = input_tensor_baseline.transpose(-2, -1)
                k_transposed = (k_w, k_h)  
                s_transposed = (s_w, s_h)
                p_transposed = (p_w, p_h)  
            
                kernel_transposed = torch.zeros(Cout, Cin, k_transposed[0], k_transposed[1], device=input_tensor.device).to(torch.int8)



                output_tensor, stats = simulator_sim.run_convolution(
                    input_tensor_transposed, 
                    kernel_transposed, 
                    padding=p_transposed, 
                    stride=s_transposed,
                    groups=groups
                )

            
                prosperity_cycles = run_prosperity_conv(
                    Cout, input_tensor_baseline_transposed, kernel_size=k_transposed, stride=s_transposed, padding=p_transposed, groups=groups
                )
            
                gustav_cycles = run_gustavSNN_conv(
                    Cout, input_tensor_baseline_transposed, kernel_size=k_transposed, stride=s_transposed, padding=p_transposed, groups=groups
                )

                oh = (h - k_h + 2 * p_h) // s_h + 1
                ow = (w - k_w + 2 * p_w) // s_w + 1
                Cin_g = Cin // groups
                Cout_g = Cout // groups

                # Prosperity: conv 后必接 LIF，主体随 GEMM 流水隐藏，只追加排空尾巴
                T_lif = T * (spike_factor or 1)
                num_lif_neurons = Cout * oh * ow
                prosperity_cycles, lif_stats = run_prosperity_layer_with_lif(
                    prosperity_cycles, num_lif_neurons, T_lif, B)

                direct_conv_cycles = oh * (ow + 2 * p_w) * T * B * math.ceil(Cout_g/args.num_pus) * math.ceil(Cin_g/args.num_cores) * groups
                PTB_cycles = run_PTB_layer_with_lif(
                    input_tensor_baseline_transposed.flatten(0,1), "conv", k_transposed, s_transposed,
                    p_transposed, Cout, ow, oh,
                    time_steps=T_lif, num_neuron=num_lif_neurons * B)
                # DLA_cycles = h * w * 3 * T * B * ceil_a_by_b(Cout_g, 16) * ceil_a_by_b(Cin_g, 64) * groups
            
                dla_speedup = cycle_speedup(DLA_cycles, stats.total_cycles)
                gustav_speedup = cycle_speedup(gustav_cycles, stats.total_cycles)
                prosperity_speedup = cycle_speedup(prosperity_cycles, stats.total_cycles)
                ptb_speedup = cycle_speedup(PTB_cycles, stats.total_cycles)
            
                dla_total_cycle_num += DLA_cycles
                ours_total_cycle_num += stats.total_cycles
                gustav_total_cycle_num += gustav_cycles
                prosperity_total_cycle_num += prosperity_cycles
                ptb_total_cycle_num += PTB_cycles
            

                elapsed_time = time.time() - start_time
            
                eva_report = result_columns(stats)
                results.append({
                    'test_case': i+1,
                    'file': test_case['file'],
                    'type': test_case['type'],
                    'input_shape': input_tensor.shape,
                    'dense': dense,
                    'simulator_cycles': stats.total_cycles,
                    'num_mode0': eva_report['num_mode0'],
                    'ratio_mode0': eva_report['ratio_mode0'],
                    'num_mode1': eva_report['num_mode1'],
                    'ratio_mode1': eva_report['ratio_mode1'],
                    'num_mode2': eva_report['num_mode2'],
                    'ratio_mode2': eva_report['ratio_mode2'],
                    'EVA': eva_report['EVA'],
                    'LIF': eva_report['LIF'],
                    'DRAM': eva_report['DRAM'],
                    'ptb_cycles': PTB_cycles,
                    'ptb_speedup': ptb_speedup,
                    'direct_conv_cycles': direct_conv_cycles,
                    'dla_cycles': DLA_cycles,
                    'dla_speedup': dla_speedup,
                    'prosperity_cycles': prosperity_cycles,
                    'prosperity_speedup': prosperity_speedup,   
                    'gustav_cycles': gustav_cycles,
                    'gustav_speedup': gustav_speedup,
                    'elapsed_time': elapsed_time
                })

            elif clean_type == 'conv2d_(1, 1)_(1, 1)_(0, 0)' and valid_spike:
                DLA_cycles, _, _ = run_DLA_layer(
                    input_tensor, clean_type, test_case['Cout'], groups=groups)
                Cout = test_case['Cout']
                if len(input_tensor.shape) == 4:
                    TB, Cin, h, w = input_tensor.shape
                    T = TB
                    B = 1
                    input_tensor_for_sim = input_tensor
                    input_tensor_baseline_conv = input_tensor_baseline.unsqueeze(1)
                else:
                    T, B, Cin, h, w = input_tensor.shape
                    input_tensor_for_sim = input_tensor.view(T*B, Cin, h, w)
                    input_tensor_baseline_conv = input_tensor_baseline

                spike_matrix = input_tensor_for_sim.permute(0, 2, 3, 1).flatten(0, 2).to(torch.int8)
                weight_matrix = torch.zeros(Cin, Cout, device=input_tensor.device).to(torch.int8)
                output_tensor, stats = simulator_sim.run_fc(spike_matrix, weight_matrix)
            
                prosperity_cycles = run_prosperity_conv(Cout, input_tensor_baseline_conv, kernel_size=1, stride=1, padding=0)
                T_lif = T * (spike_factor or 1)
                num_lif_neurons = Cout * h * w
                prosperity_cycles, lif_stats = run_prosperity_layer_with_lif(
                    prosperity_cycles, num_lif_neurons, T_lif, B)
            
                gustav_cycles = run_gustavSNN_conv(Cout, input_tensor_baseline_conv, kernel_size=1, stride=1, padding=0)

                PTB_cycles = run_PTB_layer_with_lif(
                    input_tensor_baseline_conv.flatten(0,1), "conv", 1, 1, 0, Cout, h, w,
                    time_steps=T_lif, num_neuron=num_lif_neurons * B)

            
            
                # DLA_cycles = h * w * T * B * ceil_a_by_b(Cout, 16) * ceil_a_by_b(Cin, 64)

            
                dla_speedup = cycle_speedup(DLA_cycles, stats.total_cycles)
                gustav_speedup = cycle_speedup(gustav_cycles, stats.total_cycles)
                prosperity_speedup = cycle_speedup(prosperity_cycles, stats.total_cycles)
                ptb_speedup = cycle_speedup(PTB_cycles, stats.total_cycles)
            
                dla_total_cycle_num += DLA_cycles
                ours_total_cycle_num += stats.total_cycles
                gustav_total_cycle_num += gustav_cycles
                prosperity_total_cycle_num += prosperity_cycles
                ptb_total_cycle_num += PTB_cycles
            

                elapsed_time = time.time() - start_time
            
                eva_report = result_columns(stats)
                results.append({
                    'test_case': i+1,
                    'file': test_case['file'],
                    'type': test_case['type'],
                    'input_shape': input_tensor.shape,
                    'dense': dense,
                    'simulator_cycles': stats.total_cycles,
                    'num_mode0': eva_report['num_mode0'],
                    'ratio_mode0': eva_report['ratio_mode0'],
                    'num_mode1': eva_report['num_mode1'],
                    'ratio_mode1': eva_report['ratio_mode1'],
                    'num_mode2': eva_report['num_mode2'],
                    'ratio_mode2': eva_report['ratio_mode2'],
                    'EVA': eva_report['EVA'],
                    'LIF': eva_report['LIF'],
                    'DRAM': eva_report['DRAM'],
                    'ptb_cycles': PTB_cycles,
                    'ptb_speedup': ptb_speedup,
                    'dla_cycles': DLA_cycles,
                    'dla_speedup': dla_speedup,
                    'prosperity_cycles': prosperity_cycles,
                    'prosperity_speedup': prosperity_speedup,
                    'gustav_cycles': gustav_cycles,
                    'gustav_speedup': gustav_speedup,
                    'elapsed_time': elapsed_time
                })

            # 没有分组卷积层
            elif clean_type == 'conv1d_1_1' and valid_spike:
                DLA_cycles, _, _ = run_DLA_layer(
                    input_tensor, clean_type, test_case['Cout'], groups=groups)
                if len(input_tensor.shape) == 4:
                    T, B, Cin, N = input_tensor.shape
                    input_tensor_for_sim = input_tensor.view(T*B, Cin, N)
                    Tb, Bb, Cin_b, N_b = input_tensor_baseline.shape
                    input_tensor_baseline_fc = input_tensor_baseline.view(Tb*Bb, Cin_b, N_b)
                elif len(input_tensor.shape) == 3:
                    T, Cin, N = input_tensor.shape
                    B = 1
                    input_tensor_for_sim = input_tensor
                    input_tensor_baseline_fc = input_tensor_baseline
                else:
                    raise ValueError(f"Unsupported conv1d_1_1 input shape: {input_tensor.shape}")

                dim = test_case['Cout']
                Cin_g = Cin // groups
                dim_g = dim // groups

                input_matrix = input_tensor_for_sim.permute(0, 2, 1).to(torch.int8)
                input_matrix_sim = input_matrix.flatten(0,1).to(torch.int8) # [TB*N, Cin]
                weight_matrix = torch.zeros(Cin, dim,device=input_tensor.device).to(torch.int8)  # [Cin,dim]
                output_tensor, stats = simulator_sim.run_fc(input_matrix_sim, weight_matrix)

                input_matrix_baseline = input_tensor_baseline_fc.permute(0, 2, 1).flatten(0,1).to(torch.int8)
                prosperity_cycles = run_prosperity_fc(dim, input_matrix_baseline)
                T_lif = T * (spike_factor or 1)
                num_lif_neurons = N * dim
                prosperity_cycles, lif_stats = run_prosperity_layer_with_lif(
                    prosperity_cycles, num_lif_neurons, T_lif, B)

                gustav_cycles = run_gustavSNN_fc(input_tensor_baseline_fc, dim)

                PTB_cycles = run_PTB_layer_with_lif(
                    input_tensor_baseline_fc.permute(0, 2, 1), "fc", cout=dim,
                    M=N, K=Cin,
                    time_steps=T_lif, num_neuron=num_lif_neurons * B)

                direct_conv_cycles = T*B * N * Cin_g * dim_g * groups
                # DLA_cycles = N * T*B * ceil_a_by_b(dim_g, 16) * ceil_a_by_b(Cin_g, 64) * groups
            
                dla_speedup = cycle_speedup(DLA_cycles, stats.total_cycles)
                gustav_speedup = cycle_speedup(gustav_cycles, stats.total_cycles)
                prosperity_speedup = cycle_speedup(prosperity_cycles, stats.total_cycles)
                ptb_speedup = cycle_speedup(PTB_cycles, stats.total_cycles)
            
                dla_total_cycle_num += DLA_cycles
                ours_total_cycle_num += stats.total_cycles
                gustav_total_cycle_num += gustav_cycles
                prosperity_total_cycle_num += prosperity_cycles
                ptb_total_cycle_num += PTB_cycles


                elapsed_time = time.time() - start_time

                eva_report = result_columns(stats)
                results.append({
                    'test_case': i+1,
                    'file': test_case['file'],
                    'type': test_case['type'],
                    'input_shape': input_tensor.shape,
                    'dense': dense,
                    'simulator_cycles': stats.total_cycles,
                    'num_mode0': eva_report['num_mode0'],
                    'ratio_mode0': eva_report['ratio_mode0'],
                    'num_mode1': eva_report['num_mode1'],
                    'ratio_mode1': eva_report['ratio_mode1'],
                    'num_mode2': eva_report['num_mode2'],
                    'ratio_mode2': eva_report['ratio_mode2'],
                    'EVA': eva_report['EVA'],
                    'LIF': eva_report['LIF'],
                    'DRAM': eva_report['DRAM'],
                    'ptb_cycles': PTB_cycles,
                    'ptb_speedup': ptb_speedup,
                    'direct_conv_cycles': direct_conv_cycles,
                    'dla_cycles': DLA_cycles,
                    'dla_speedup': dla_speedup,
                    'prosperity_cycles': prosperity_cycles,
                    'prosperity_speedup': prosperity_speedup,   
                    'gustav_cycles': gustav_cycles,
                    'gustav_speedup': gustav_speedup,
                    'elapsed_time': elapsed_time
                })
        
            # 没有分组卷积层
            elif clean_type == 'conv1d_3_1' and valid_spike:
                DLA_cycles, _, _ = run_DLA_layer(
                    input_tensor, clean_type, test_case['Cout'], groups=groups)
                T, B, Cin, N = input_tensor.shape
                Cout = test_case['Cout']

                # 构建 1x3 的权重传入
                kernel = torch.zeros(Cout, Cin, 1, 3,device=input_tensor.device).to(torch.int8)
                padding = 1
                # 扩展维度，看成H=1的5D张量
                input_tensor_for_sim = input_tensor.unsqueeze(3)
                input_tensor_baseline_conv = input_tensor_baseline.unsqueeze(3)
        
                output_tensor, stats = simulator_sim.run_convolution(input_tensor_for_sim, kernel, padding=padding, stride=1, groups=groups)
                # pe_utilization = simulator_sim.pe_cycles / stats.total_cycles
            
                oh = 1
                ow = (N - 3 + 2 * padding) // 1 + 1
                prosperity_cycles = run_prosperity_conv(Cout, input_tensor_baseline_conv, kernel_size=(1,3), stride=1, padding=padding, groups=groups)
                # Prosperity: conv 后必接 LIF，主体随 GEMM 流水隐藏，只追加排空尾巴
                T_lif = T * (spike_factor or 1)
                num_lif_neurons = Cout * oh * ow
                prosperity_cycles, lif_stats = run_prosperity_layer_with_lif(
                    prosperity_cycles, num_lif_neurons, T_lif, B)
            
                gustav_cycles = run_gustavSNN_conv(Cout, input_tensor_baseline_conv, kernel_size=(1,3), stride=1, padding=padding, groups=groups)

                PTB_cycles = run_PTB_layer_with_lif(
                    input_tensor_baseline_conv, "conv", (1, 3), 1, padding, Cout, oh, ow,
                    time_steps=T_lif, num_neuron=Cout * oh * ow * B)

                Cin_g = Cin // groups
                Cout_g = Cout // groups

                direct_conv_cycles = oh * (ow+2*padding) * T * B * math.ceil(Cout_g/args.num_pus) * math.ceil(Cin_g/args.num_cores) * groups
                # DLA_cycles = oh * ow * 3 * T * B * ceil_a_by_b(Cout_g,16) * ceil_a_by_b(Cin_g,64) * groups
            
                dla_speedup = cycle_speedup(DLA_cycles, stats.total_cycles)
                gustav_speedup = cycle_speedup(gustav_cycles, stats.total_cycles)
                prosperity_speedup = cycle_speedup(prosperity_cycles, stats.total_cycles)
                ptb_speedup = cycle_speedup(PTB_cycles, stats.total_cycles)
            
                dla_total_cycle_num += DLA_cycles
                ours_total_cycle_num += stats.total_cycles
                gustav_total_cycle_num += gustav_cycles
                prosperity_total_cycle_num += prosperity_cycles
                ptb_total_cycle_num += PTB_cycles
            
                input_tensor = input_tensor_for_sim
            
                elapsed_time = time.time() - start_time
            
                eva_report = result_columns(stats)
                results.append({
                    'test_case': i+1,
                    'file': test_case['file'],
                    'type': test_case['type'],
                    'input_shape': input_tensor.shape,
                    'dense': dense,
                    'simulator_cycles': stats.total_cycles,
                    'num_mode0': eva_report['num_mode0'],
                    'ratio_mode0': eva_report['ratio_mode0'],
                    'num_mode1': eva_report['num_mode1'],
                    'ratio_mode1': eva_report['ratio_mode1'],
                    'num_mode2': eva_report['num_mode2'],
                    'ratio_mode2': eva_report['ratio_mode2'],
                    'EVA': eva_report['EVA'],
                    'LIF': eva_report['LIF'],
                    'DRAM': eva_report['DRAM'],
                    'ptb_cycles': PTB_cycles,
                    'ptb_speedup': ptb_speedup,
                    'direct_conv_cycles': direct_conv_cycles,
                    'dla_cycles': DLA_cycles,
                    'dla_speedup': dla_speedup,
                    'prosperity_cycles': prosperity_cycles,
                    'prosperity_speedup': prosperity_speedup,   
                    'gustav_cycles': gustav_cycles,
                    'gustav_speedup': gustav_speedup,
                    'elapsed_time': elapsed_time
                })

            # TIM有
            elif clean_type == 'conv1d_5_1' and valid_spike:
                DLA_cycles, _, _ = run_DLA_layer(
                    input_tensor, clean_type, test_case['Cout'], groups=groups)
                T, B, Cin, N = input_tensor.shape
                input_tensor_for_im2col = input_tensor.view(T*B, Cin, N).unsqueeze(2)
                    # 统计input_tensor的稠密度
                # total_elements = input_tensor.numel()
                # nonzero_elements = torch.sum(input_tensor != 0).item()
                # dense = nonzero_elements / total_elements
                # print(f"conv1d_5_1，input_tensor的稠密度: {dense:.4f}")
            
                dim = test_case['Cout']  
                dim_g = dim // groups
                Cin_g = Cin // groups
                # img2col 接收[B,C,H,W]的input_tensor，得到[B,M,K]的input_matrix     
                input_matrix = img2col(input_tensor_for_im2col, kernel_size=(1,5), stride=1, padding=(0,2)).to(torch.int8)
                # print(f"conv1d_5_1，im2col后input_matrix的shape: {input_matrix.shape}")
                # 统计input_matrix的稠密度
                # total_elements = input_matrix.numel()
                # nonzero_elements = torch.sum(input_matrix != 0).item()
                # dense = nonzero_elements / total_elements
                # print(f"conv1d_5_1，经过im2col处理后input_matrix的稠密度: {dense:.4f}")

                # 展平[B,M,K]的input_matrix，得到[B*M,K]的input_matrix_sim
                input_matrix_sim = input_matrix.flatten(0,1)
                # print(f"conv1d_5_1扩展后input_matrix的shape: {input_matrix_sim.shape}")

                weight_matrix = torch.zeros(Cin*5, dim, device=input_tensor.device).to(torch.int8)  # [Cin*5,dim]
                output_tensor, stats = simulator_sim.run_fc(input_matrix_sim, weight_matrix)
            
                prosperity_cycles = run_prosperity_fc(dim, input_matrix_sim)
                # Prosperity: 卷积后必接 LIF（输出长度不变，dim 个通道），
                # 主体随 GEMM 流水隐藏，只追加排空尾巴
                T_lif = T * (spike_factor or 1)
                num_lif_neurons = N * dim
                prosperity_cycles, lif_stats = run_prosperity_layer_with_lif(
                    prosperity_cycles, num_lif_neurons, T_lif, B)
                # img2col 输出 [T*B, N, D]，GustavSNN 的右矩阵必须是
                # [T*B, D, N]；转置后才按输入通道 D 做 NRV 行跳过。
                gustav_cycles = run_gustavSNN_fc(
                    input_matrix.transpose(1, 2), dim)

                PTB_cycles = run_PTB_layer_with_lif(
                    input_matrix, "fc", cout=dim,
                    M=input_matrix.shape[0], K=input_matrix.shape[1],
                    time_steps=T_lif, num_neuron=num_lif_neurons * B)


                direct_conv_cycles = T*B * N * Cin_g * dim_g * groups
                # 5x1 的一维卷积，
                # 旧式遗漏输出长度N，保留为历史比较；不能作为新模型的原子操作数。
                # DLA_cycles = 5 * T*B * ceil_a_by_b(dim_g, 16) * ceil_a_by_b(Cin_g, 64) * groups
            
                dla_speedup = cycle_speedup(DLA_cycles, stats.total_cycles)
                gustav_speedup = cycle_speedup(gustav_cycles, stats.total_cycles)
                prosperity_speedup = cycle_speedup(prosperity_cycles, stats.total_cycles)
                ptb_speedup = cycle_speedup(PTB_cycles, stats.total_cycles)
            
                dla_total_cycle_num += DLA_cycles
                ours_total_cycle_num += stats.total_cycles
                gustav_total_cycle_num += gustav_cycles
                prosperity_total_cycle_num += prosperity_cycles
                ptb_total_cycle_num += PTB_cycles


                elapsed_time = time.time() - start_time

                eva_report = result_columns(stats)
                results.append({
                    'test_case': i+1,
                    'file': test_case['file'],
                    'type': test_case['type'],
                    'input_shape': input_tensor.shape,
                    'dense': dense,
                    'simulator_cycles': stats.total_cycles,
                    'num_mode0': eva_report['num_mode0'],
                    'ratio_mode0': eva_report['ratio_mode0'],
                    'num_mode1': eva_report['num_mode1'],
                    'ratio_mode1': eva_report['ratio_mode1'],
                    'num_mode2': eva_report['num_mode2'],
                    'ratio_mode2': eva_report['ratio_mode2'],
                    'EVA': eva_report['EVA'],
                    'LIF': eva_report['LIF'],
                    'DRAM': eva_report['DRAM'],
                    'ptb_cycles': PTB_cycles,
                    'ptb_speedup': ptb_speedup,
                    'direct_conv_cycles': direct_conv_cycles,
                    'dla_cycles': DLA_cycles,
                    'dla_speedup': dla_speedup,
                    'prosperity_cycles': prosperity_cycles,
                    'prosperity_speedup': prosperity_speedup,   
                    'gustav_cycles': gustav_cycles,
                    'gustav_speedup': gustav_speedup,
                    'elapsed_time': elapsed_time
                })
        
            # TIM叫SSA，QKformer叫Spiking_Self_Attention & Token_QK_Attention
            elif clean_type in SELF_ATTENTION_TYPES and valid_spike:
                """
                Self attention Q/K/V: [T, B, heads, N, dim_per_head].
                If N > dim_per_head:
                    S = K^T @ V, then Y = Q @ S
                Otherwise:
                    A = Q @ K^T, then Y = A @ V
                Each matrix multiply is mapped to the same FC simulator path.
                """
                # for DLA,GEMM is h=w=1 special conv, if Gemm.shape = [M,K]*[K,N],
                # cin = K,cout = N, DLA_cycles = M * ceil_a_by_b(N,16) * ceil_a_by_b(M,64)
                T, B, num_heads, sequence_length, dim_per_head = act_q.shape
                EVA_layer_cycles = 0
                DLA_cycles = 0
                prosperity_cycles = 0
                PTB_cycles = 0
                for t in range(T):
                    for b in range(B):
                        for h in range(num_heads):
                            q_mat = act_q[t,b,h]
                            k_mat = act_k[t,b,h]
                            v_mat = act_v[t,b,h]
                            s_mat = torch.rand(dim_per_head, dim_per_head, device=act_q.device).to(torch.int8)  # Placeholder for S
                            a_mat = torch.rand(sequence_length, sequence_length, device=act_q.device).to(torch.int8)  # Placeholder for Y

                        # if sequence_length > dim_per_head:
                            # S = K^T @ V: [D, N] x [N, D] -> [D, D]
                            output_tensor,states= simulator_sim.run_fc(k_mat.T.contiguous(), v_mat, with_lif=False)
                            EVA_layer_cycles += states.total_cycles
                            prosperity_cycles += run_prosperity_fc(dim_per_head, k_mat.T)
                            dla_cycles, _, _ = run_DLA_layer(
                                k_mat.reshape(1, 1, sequence_length, dim_per_head),
                                'conv1d_1_1', dim_per_head, with_lif=False)
                            DLA_cycles += dla_cycles
                            # Y = Q @ S: [N, D] x [D, D] -> [N, D]
                            output_tensor,states= simulator_sim.run_fc(q_mat.contiguous(), s_mat)
                            EVA_layer_cycles += states.total_cycles    
                            prosperity_layer_cycles, _ = run_prosperity_layer_with_lif(
                                run_prosperity_fc(dim_per_head, q_mat),
                                num_post_neuron=sequence_length * dim_per_head, T_lif=1)
                            prosperity_cycles += prosperity_layer_cycles
                            dla_cycles, _, _ = run_DLA_layer(
                                q_mat.T.reshape(1, 1, dim_per_head, sequence_length),
                                'conv1d_1_1', dim_per_head)
                            DLA_cycles += dla_cycles
                        # else:       # 虽然prosperity中实现了这个，但是我觉得如果要做这个，需要转置，开销太大
                        #     # A = Q @ K^T: [N, D] x [D, N] -> [N, N]
                        #     output_tensor,states= simulator_sim.run_fc(q_mat.contiguous(), k_mat.T.contiguous())
                        #     EVA_layer_cycles += states.total_cycles
                        #     prosperity_cycles += run_prosperity_fc(sequence_length, q_mat)
                        #     DLA_cycles += sequence_length * ceil_a_by_b(dim_per_head,64) * ceil_a_by_b(sequence_length,16)
                        #     # Y = A @ V: [N, N] x [N, D] -> [N, D]
                        #     output_tensor,states= simulator_sim.run_fc(a_mat.contiguous(), v_mat)
                        #     EVA_layer_cycles += states.total_cycles
                        #     prosperity_cycles += run_prosperity_fc(sequence_length, a_mat)
                        #     DLA_cycles += sequence_length * ceil_a_by_b(dim_per_head,64) * ceil_a_by_b(sequence_length,16)
                for b in range(B):
                    for h in range(num_heads):
                        # 时间压缩至少需要四个时间步；补零只用于矩阵计算，神经元计算仍使用实际时间步数。
                        ptb_k = torch.nn.functional.pad(act_k[:, b, h], (0, 0, 0, 0, 0, max(0, 4-T)))
                        ptb_q = torch.nn.functional.pad(act_q[:, b, h], (0, 0, 0, 0, 0, max(0, 4-T)))
                        PTB_cycles += run_PTB_layer_with_lif(
                            ptb_k.transpose(1, 2).contiguous(), type='fc',
                            cout=dim_per_head, with_lif=False)
                        PTB_cycles += run_PTB_layer_with_lif(
                            ptb_q.contiguous(), type='fc', cout=dim_per_head,
                            time_steps=T, num_neuron=sequence_length * dim_per_head)
                        
                        # gustav需要保留T维度，且脉冲矩阵是作为GEMM的右矩阵，K正好不用转置了，但是Q需要转置
                        gustav_q = act_q[:, b, h]
                        gustav_k = act_k[:, b, h]
                        GustavSNN_cycles = (run_gustavSNN_fc(gustav_k, dim_per_head, with_lif=False)
                                        + run_gustavSNN_fc(gustav_q.permute(0, 2, 1), dim_per_head))
                    
                direct_conv_cycles= T*B*num_heads * sequence_length * dim_per_head * sequence_length  + T*B*num_heads * sequence_length * dim_per_head * dim_per_head
                dla_speedup = cycle_speedup(DLA_cycles, EVA_layer_cycles)
                gustav_speedup = cycle_speedup(GustavSNN_cycles, EVA_layer_cycles)
                prosperity_speedup =  prosperity_cycles / EVA_layer_cycles
                ptb_speedup = cycle_speedup(PTB_cycles, EVA_layer_cycles)
            
                dla_total_cycle_num += DLA_cycles
                ours_total_cycle_num += EVA_layer_cycles
                gustav_total_cycle_num += GustavSNN_cycles
                prosperity_total_cycle_num += prosperity_cycles
                ptb_total_cycle_num += PTB_cycles


                print(f"DLA加速比：{dla_speedup:.2f}")
                print(f"prosperity加速比：{prosperity_speedup:.2f}")
                print(f"gustavSNN加速比：{gustav_speedup:.2f}")
                print(f"PTB加速比：{ptb_speedup:.2f}")

                elapsed_time = time.time() - start_time
                print(f"测试用例用时: {elapsed_time:.4f} 秒")

                results.append({
                    'test_case': i+1,
                    'file': test_case['file'],
                    'type': test_case['type'],
                    'input_shape': act_q.shape,
                    'dense': dense,
                    'simulator_cycles': EVA_layer_cycles,
                    'ptb_cycles': PTB_cycles,
                    'ptb_speedup': ptb_speedup,
                    'direct_conv_cycles': direct_conv_cycles,
                    'dla_cycles': DLA_cycles,
                    'dla_speedup': dla_speedup,
                    'prosperity_cycles': prosperity_cycles,
                    'prosperity_speedup': prosperity_speedup,   
                    'gustav_cycles': GustavSNN_cycles,
                    'gustav_speedup': gustav_speedup,
                    'elapsed_time': elapsed_time
                })


            # elif clean_type == 'Token_QK_Attention' and valid_spike:
            #     """
            #     Token_QK_Attention 则没有 V，而Q/K ：[T, B, heads, N, dim_per_head]，然后
            #     gate = LIF(sum(Q, dim=dim_per_head))  # [T, B, heads, N, 1]
            #     Y = gate * K                      # 广播到每个 dim_per_head
            #     """
            #     #act_q就是gate，所以此处直接用act_q作为输入，act_k作为权重
            #     T, B, num_heads, sequence_length, dim_per_head = act_k.shape
            #     EVA_layer_cycles = 0
            #     prosperity_cycles = 0
            #     PTB_cycles = 0
            #     DLA_cycles = 0
            #     for t in range(T):
            #         for b in range(B):
            #             for h in range(num_heads):
            #                 q_mat = act_q[t, b, h]  # [N, 1]，但是需要转置才能执行向量矩阵乘法
            #                 k_mat = act_k[t, b, h]  # [N, dim_per_head]
            #                 # q需要转置
            #                 output,states= simulator_sim.run_fc(q_mat.t(), k_mat.contiguous())
            #                 EVA_layer_cycles += states.total_cycles
            #                 prosperity_cycles += run_prosperity_fc(dim_per_head, q_mat)
            #                 DLA_cycles += sequence_length * 1 * ceil_a_by_b(dim_per_head, 16)

            #     for b in range(B):
            #         for h in range(num_heads):
            #             # 时间压缩至少需要四个时间步；补零只用于矩阵计算，神经元计算仍使用实际时间步数。
            #             ptb_k = torch.nn.functional.pad(act_k[:, b, h], (0, 0, 0, 0, 0, max(0, 4-T)))
            #             ptb_q = torch.nn.functional.pad(act_q[:, b, h], (0, 0, 0, 0, 0, max(0, 4-T)))
            #             # q 被gate后，执行向量矩阵乘，如何映射到PTB呢？
            #             PTB_cycles += run_PTB_layer_with_lif(
            #                 ptb_q.contiguous(), type='fc', cout=dim_per_head,
            #                 time_steps=T, num_neuron=sequence_length * dim_per_head)



            #             gustav_q = act_q[:, b, h]
            #             gustav_k = act_k[:, b, h]
            #             # 直接将q看作权重，则完美符合GustavSNN的要求，gustav中权重在左边，脉冲矩阵在右边,
            #             # Q 被alone D求和后，再与K相乘，因此output_dim = 1，但是Q需要转置
            #             GustavSNN_cycles = run_gustavSNN_fc(gustav_k.permute(0, 2, 1), 1)

            #     dla_speedup = DLA_cycles / EVA_layer_cycles
            #     gustav_speedup = GustavSNN_cycles / EVA_layer_cycles
            #     prosperity_speedup =  prosperity_cycles / EVA_layer_cycles
            
            #     dla_total_cycle_num += DLA_cycles
            #     ours_total_cycle_num += EVA_layer_cycles
            #     gustav_total_cycle_num += GustavSNN_cycles
            #     prosperity_total_cycle_num += prosperity_cycles

            #     input_tensor = input_tensor_for_sim
            #     print(f"模拟器输入形状: {input_tensor.shape}")
            #     print(f"DLA加速比：{dla_speedup:.2f}")
            #     print(f"prosperity加速比：{prosperity_speedup:.2f}")
            #     print(f"gustavSNN加速比：{gustav_speedup:.2f}")

            #     elapsed_time = time.time() - start_time
            #     print(f"测试用例用时: {elapsed_time:.4f} 秒")

            #     results.append({
            #         'test_case': i+1,
            #         'file': test_case['file'],
            #         'type': test_case['type'],
            #         'input_shape': input_tensor.shape,
            #         'dense': dense,
            #         'simulator_cycles': EVA_layer_cycles,
            #         'direct_conv_cycles': direct_conv_cycles,
            #         'dla_cycles': DLA_cycles,
            #         'dla_speedup': dla_speedup,
            #         'prosperity_cycles': prosperity_cycles,
            #         'prosperity_speedup': prosperity_speedup,   
            #         'gustav_cycles': GustavSNN_cycles,
            #         'gustav_speedup': gustav_speedup,
            #         'elapsed_time': elapsed_time
            #     })
        
            elif clean_type == 'maxpool':
                # 参考了DLA中SDP的实现方式，对生成的脉冲直接做或操作即可，而且是可以流水的，只会在最后增加一拍的周期。
                # 如果是maxpooling，所有考虑的DRAM的加速器的总周期还需要减，因为上一层算DRAM写回的数据量时按照maxpooing之前shape的算的，多算了3/4
                # 用多的数据量除以各自的DRAM带宽。
                over_dram_datas = input_tensor.numel() * 3 / 4
                ours_total_cycle_num += 1 - over_dram_datas * simulator_sim.spike_size / simulator_sim.bandwidth
                dla_total_cycle_num += 1 - over_dram_datas * 8 / DLA_MEMORY_BITS_PER_CYCLE
                prosperity_total_cycle_num += 1 - over_dram_datas / PROSPERITY_MEM_IF_WIDTH
                gustav_total_cycle_num += 1 - over_dram_datas / (256 / 5)
                # 沿用现有时间压缩基线的输出字节量及每拍 1024 个数据单位的访存公式。
                ptb_total_cycle_num += 1 - over_dram_datas / 8 / 1024
                continue

            else:
                logger.silent = False
                print(f"暂时不支持的type: {test_case['type']}_spike{test_case['spike']}")  
                continue

            # 比较列使用总周期；保留周期分解与卷积代表输出通道组的数据包比例。
            results[-1].update({key: value for key, value in eva_report.items()
                                if key not in results[-1]})


            row = results[-1]
            print(f"总周期: EVA {row['simulator_cycles']:,}; DLA {row['dla_cycles']:,}; "
                  f"Prosperity {row['prosperity_cycles']:,}; "
                  f"GustavSNN {row['gustav_cycles']:,}; PTB {row['ptb_cycles']:,}")
            if row['EVA'] is not None:
                print(f"EVA分解: {row['EVA']:,} + LIF {row['LIF']:,} + DRAM {row['DRAM']:,}")

    # 计算总用时
    total_elapsed_time = time.time() - total_start_time
    
    # 打印总体统计
    print(f"\n{'='*60}")
    print(f"总体统计 (网络: {args.network})")
    print(f"{'='*60}")
    print(f"完成所有测试用例总用时: {total_elapsed_time:.4f} 秒")
    print(f"总DLA周期: {dla_total_cycle_num:,.0f}")
    print(f"总EVA周期: {ours_total_cycle_num:,.0f}")
    print(f"总prosperity周期: {prosperity_total_cycle_num:,.0f}")
    print(f"总PTB周期: {ptb_total_cycle_num:,.0f}")
    print(f"总gustavSNN周期: {gustav_total_cycle_num:,.0f}")
    if ours_total_cycle_num > 0:
        print(f"总体DLA加速比: {dla_total_cycle_num / ours_total_cycle_num:.2f}")
        print(f"总体prosperity加速比: {prosperity_total_cycle_num / ours_total_cycle_num:.2f}")
        print(f"总体PTB加速比: {ptb_total_cycle_num / ours_total_cycle_num:.2f}")
        print(f"总体gustavSNN加速比: {gustav_total_cycle_num / ours_total_cycle_num:.2f}")
    print(f"{'='*60}")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 仿真任务完成！")

    dla_speedup = cycle_speedup(dla_total_cycle_num, ours_total_cycle_num)
    # 保存结果到CSV文件
    if results:
        # 将结果保存为CSV文件，文件名包含加速比结果
        csv_filename = f'{output_folder}/core{args.num_cores}_pu{args.num_pus}_dla{dla_speedup:.2f}_results.csv'
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            # 固定常用列的顺序；不同算子缺少的字段由 DictWriter 留空。
            preferred_fieldnames = [
                'test_case',
                'file',
                'type',
                'input_shape',
                'dense',
                'simulator_cycles',
                'num_mode0',
                'ratio_mode0',
                'num_mode1',
                'ratio_mode1',
                'num_mode2',
                'ratio_mode2',
                'EVA',
                'LIF',
                'DRAM',
                'direct_conv_cycles',
                'dla_cycles',
                'dla_speedup',
                'prosperity_cycles',
                'prosperity_speedup',
                'gustav_cycles',
                'gustav_speedup',
                'ptb_cycles',
                'ptb_speedup',
                'elapsed_time',
            ]
            available_fields = {
                fieldname
                for result in results
                for fieldname in result
            }
            fieldnames = [
                fieldname
                for fieldname in preferred_fieldnames
                if fieldname in available_fields
            ]
            # 将未来新增字段按首次出现顺序追加，避免遗漏字段。
            known_fields = set(fieldnames)
            for result in results:
                for fieldname in result:
                    if fieldname not in known_fields:
                        fieldnames.append(fieldname)
                        known_fields.add(fieldname)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n结果已保存到: {csv_filename}")

    # 总体加速比只能在全部测试结束后确定，因此先关闭临时日志，再将其
    # 重命名为带 DLA 加速比的最终文件名，使日志和 CSV 可以一一对应。
    # 注意：log_filename 是完整绝对路径，必须用 dirname/basename 分离目录和文件名，
    # 否则前缀会被加到整个路径字符串前面，产生非法路径（WinError 123）。
    log_dir = os.path.dirname(log_filename)       
    log_basename = os.path.basename(log_filename)  
    final_log_filename = os.path.join(log_dir, f"dla{dla_speedup:.2f}_{log_basename}")
    print(f"保存至日志文件: {final_log_filename}")
    logger.flush()
    sys.stdout = logger.terminal
    logger.log.close()
    os.replace(log_filename, final_log_filename)
    if hasattr(simulator_sim, 'finish_reporting'):
        simulator_sim.finish_reporting()


if __name__ == "__main__":
    run()
