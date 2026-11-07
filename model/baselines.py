import torch
from model.utils import Stats, pad_to_power_of_2, img2col, ceil_a_by_b
from model.networks import Conv2D, FC, LIFNeuron, Attention, LayerNorm, conv2d_2_fc
from collections import OrderedDict
import numpy as np
from typing import Union
from model.dla_cycle_model import estimate_dla_runner_layer
import networkx as nx
import torch.nn.functional as F


def expand_nonbinary_timesteps(input_tensor: torch.Tensor) -> torch.Tensor:
    """Expand integer spike counts into unary binary time steps."""
    if input_tensor.numel() == 0:
        return input_tensor
    if torch.any(input_tensor < 0):
        raise ValueError("Temporal expansion does not support negative spikes")

    max_value = int(torch.max(input_tensor).item())
    if max_value <= 1:
        return input_tensor

    levels_shape = [1, max_value] + [1] * (input_tensor.ndim - 1)
    levels = torch.arange(
        1,
        max_value + 1,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    ).view(*levels_shape)
    expanded = input_tensor.unsqueeze(1) >= levels
    return expanded.reshape(-1, *input_tensor.shape[1:]).to(input_tensor.dtype)

 
# def run_gustavSNN_conv(Cout,input_tensor, kernel_size=3, stride=1, padding=1):
#     """
#     input_tensor[T, B, Cin, H, W]
#     kernel_size: kernel_size x kernel_size
#     """
#     # print("gustavNN:",input_tensor.shape,kernel_size,stride,padding)
#     # gustavNN: torch.Size([16, 1, 16, 128, 128]) 3 1 1
#     T,B,Cin,H,W = input_tensor.shape
#     D = kernel_size * kernel_size * Cin
#     oH = (H - kernel_size + 2 * padding) // stride  + 1
#     oW = (W - kernel_size + 2 * padding) // stride  + 1
#     N = oH * oW
#     spike_tensor = torch.zeros((T,D,N))
#     for t in range(T):
#         # 此处img2col的输出是按照将脉冲矩阵作为左矩阵的shape【B,M，K】
#         input_matrix = img2col(input_tensor[t, :, :, :, :], kernel_size, stride, padding)  #B,M,K
#         input_matrix = torch.flatten(input_matrix, start_dim=0, end_dim=1)  # BM,K
#         # 但是gustavGnn中是将脉冲矩阵作为右矩阵，所以需要转置
#         spike_tensor[t] = input_matrix.T
    
#     compute_cycle = run_gustavSNN_fc(spike_tensor, Cout)
#     return compute_cycle       

def run_gustavSNN_conv(Cout, input_tensor, kernel_size=3, stride=1, padding=1, groups=1):
    """
    支持分组卷积的 GustavSNN 评估
    input_tensor: [T, B, Cin, H, W]
    """
    input_tensor = expand_nonbinary_timesteps(input_tensor)
    T, B, Cin, H, W = input_tensor.shape
    
    # 兼容 tuple 参数
    k_h, k_w = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
    s_h, s_w = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    p_h, p_w = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    
    # 计算分组参数
    Cin_g = Cin // groups
    Cout_g = Cout // groups
    D_g = k_h * k_w * Cin_g  # 每组展平后的向量维度
    
    oH = (H - k_h + 2 * p_h) // s_h + 1
    oW = (W - k_w + 2 * p_w) // s_w + 1
    N = oH * oW
    
    total_cycles = 0
    
    # ==========================================
    # 核心：拆分为 g 个独立的小 GEMM
    # ==========================================
    for g in range(groups):
        cin_start = g * Cin_g
        cin_end = (g + 1) * Cin_g
        
        # 提取当前组的输入特征图
        input_group = input_tensor[:, :, cin_start:cin_end, :, :]
        
        # 展平 T 和 B 维度送入 img2col得到GEMM中的[B,M,K],但是由于GustavSNN中是将脉冲矩阵作为右矩阵，所以需要转置。
        input_group_tb = input_group.view(T * B, Cin_g, H, W)
        input_matrix_g = img2col(input_group_tb, kernel_size, stride, padding)  # [T*B, N, D_g]
        
        # Reshape 为 GustavSNN fc 要求的形状 [T, D_g, B*N]
        spike_tensor_g = input_matrix_g.view(T, B * N, D_g).transpose(1, 2)
        
        # 计算当前组的周期并累加
        cycles = run_gustavSNN_fc(spike_tensor_g, Cout_g)
        total_cycles += cycles
        
    return total_cycles

def run_gustavSNN_fc(input_tensor, M, *, weight_read_ports=2, return_details=False, with_lif=True):
    """
    T, D, N = input_tensor.shape
    GustavSNN 总周期：权重取数仲裁、处理单元执行及片外读写流水。

    with_lif=False 关闭泄漏积分发放计算，片外访存沿用原模型。

    论文配置:8 tile，每 tile 8 PE，每 PE P=8 列，
    Nr=4 个非零行/批，稠密 INT8 权重；计算阶段只接受二值脉冲 [T,D,N]。
    多值脉冲在函数入口展开，供现有 runner 复用；不是硬件直接处理多值脉冲。
    双端口建模假设（已确认）：每拍两个独立读请求，轮询仲裁，读延迟 1 拍，
    不合并同址请求、不广播；每 PE 每拍最多发一个权重请求。
    每 PE 保存当前执行批和下一预取批，两批各最多 Nr 行；fetch/execution
    重叠。不同 tile 的同 ID PE 在批边界同步，而非同 tile 的不同 ID PE
    每四个原始 D 行统一同步。没有权重矩阵输入，暂不支持零权重跳过；
    稠密权重下各 tile 同 ID 的请求序列相同，用一个 tile 的轨迹代表它们。

    总周期保留三段流水：输入读入、PE 执行/LIF、输出写回；片外读写共用
    LPDDR3 6.4 GB/s @ 1 GHz (=51.2 bit/拍)，输出为 1 bit/神经元位图。
    NRV 每记录 P 位数据 + ceil(log2(D)) 位行索引；LIF 每 PE 每拍 1 元素。
    片外流水采用整个 (一个时间步,64列) 分组的阶段级近似，输入整组读完
    才启动 PE；未模拟 G-BUF 容量/逐记录供数、DRAM 固定延迟和权重跨 M
    组预取，故这是含端口争用的参考模型，不是论文 RTL 的精确复现。
    return_details=True 返回总周期、PE 周期及端口争用统计，供定向验证使用。
    """
    if input_tensor.ndim != 3:
        raise ValueError('GustavSNN input must have shape [T, D, N]')
    if M < 0:
        raise ValueError('GustavSNN output channels M must be nonnegative')
    if not isinstance(weight_read_ports, int) or not 1 <= weight_read_ports <= 8:
        raise ValueError('weight_read_ports must be an integer in [1, 8]')
    # 在展开前验证整数计数，避免 2.5 等值被展开过程静默截成两次脉冲。
    if input_tensor.is_floating_point() and bool((input_tensor != input_tensor.floor()).any().item()):
        raise ValueError('GustavSNN requires integer spike counts before temporal expansion')
    input_tensor = expand_nonbinary_timesteps(input_tensor)
    # 计算阶段只能是二值脉冲，不能用非零判断替代脉冲计数展开。
    if bool(((input_tensor != 0) & (input_tensor != 1)).any().item()):
        raise ValueError('GustavSNN requires binary spikes after temporal expansion')
    T, D, N = input_tensor.shape
    P, Nr, num_pe, num_tile = 8, 4, 8, 8
    weight_bits, weight_read_latency = 8, 1
    group_count = ceil_a_by_b(N, num_pe * P)
    m_group_count = ceil_a_by_b(M, num_tile)
    row_index_bits = max(1, (D - 1).bit_length())
    units_per_cycle, units_per_bit = 256, 5  # 精确表示 51.2 bit/拍，不逐小块取整。
    details = dict(total_cycles=0, pe_cycles=0, weight_requests_per_tile=0,
                   weight_wait_pe_cycles_per_tile=0, weight_port_extra_pe_cycles=0)
    if M == 0 or T == 0 or N == 0:
        return details if return_details else 0

    def simulate_tile(row_counts, ports):
        # 每条非零行只请求一次权重；其多个脉冲在本地 W REG 复用。
        # 不保存完整脉冲坐标：无广播、稠密权重、每拍一脉冲的假设下，
        # 每行 popcount 足以确定执行时长及权重请求的依赖关系。
        batches = []
        for counts in row_counts:
            nonzero = [int(count) for count in counts if count]
            batches.append([nonzero[start:start + Nr]
                            for start in range(0, len(nonzero), Nr)])
        batch_index = [0] * num_pe
        issued_rows = [0] * num_pe
        response_at = [0] * num_pe
        execute_end = [0] * num_pe
        busy = [False] * num_pe
        cycle, rr_head, requests, wait_pe_cycles = 0, 0, 0, 0
        while True:
            # 先处理本拍返回和执行完成，再启动就绪批，最后仲裁本拍请求。
            for pe in range(num_pe):
                if busy[pe] and execute_end[pe] <= cycle:
                    busy[pe] = False
                index = batch_index[pe]
                if (not busy[pe] and index < len(batches[pe])
                        and issued_rows[pe] == len(batches[pe][index])
                        and response_at[pe] <= cycle):
                    execute_end[pe] = cycle + sum(batches[pe][index])
                    busy[pe] = True
                    batch_index[pe] += 1
                    issued_rows[pe] = 0
                    # 当前批进入执行寄存器后，下一批即可占用预取寄存器。
            pending = []
            for offset in range(num_pe):
                pe = (rr_head + offset) % num_pe
                if (batch_index[pe] < len(batches[pe])
                        and issued_rows[pe] < len(batches[pe][batch_index[pe]])
                        and response_at[pe] <= cycle):
                    pending.append(pe)
            grants = pending[:ports]
            wait_pe_cycles += len(pending) - len(grants)
            for pe in grants:
                issued_rows[pe] += 1
                response_at[pe] = cycle + weight_read_latency
                requests += 1
            if grants:
                rr_head = (grants[-1] + 1) % num_pe
                cycle += 1
            else:
                # 没有请求时跳到最近的返回/执行完成事件，避免逐累加拍循环。
                # 有待仲裁请求时绝不跳拍，所以端口争用等待仍逐拍精确统计。
                events = [time for time in response_at if time > cycle]
                events += [execute_end[pe] for pe in range(num_pe)
                           if busy[pe] and execute_end[pe] > cycle]
                if not events:
                    break
                cycle = min(events)
        return max(execute_end), requests, wait_pe_cycles

    chunk_stats = []
    for column_group in range(group_count):
        chunk = input_tensor[:, :, column_group * num_pe * P:(column_group + 1) * num_pe * P]
        columns = chunk.shape[-1]
        padded = F.pad(chunk, (0, num_pe * P - columns))
        # 每组只转换一次 CPU 标量，避免在每次权重仲裁时 Tensor.item()。
        counts_by_time = padded.reshape(T, D, num_pe, P).sum(dim=3).permute(0, 2, 1).tolist()
        for counts in counts_by_time:
            tile_cycles, requests, wait = simulate_tile(counts, weight_read_ports)
            execute_cycles = tile_cycles + (min(P, columns) if with_lif else 0)
            read_bits = requests * (P + row_index_bits)
            extra_cycles = 0
            if return_details:
                # 8 端口代表每 PE 一个独立读端口，仅用于隔离共享端口影响。
                extra_cycles = tile_cycles - simulate_tile(counts, num_pe)[0]
            chunk_stats.append((execute_cycles, read_bits, columns, requests, wait, extra_cycles))

    # 顺序为 M group -> column group -> T，保持 column-major time-second。
    # 同 ID PE 跨 tile 批同步由稠密权重下相同轨迹满足，不添加全 PE 的 Nr
    # 分块屏障；时间步末的组级屏障是阶段级流水近似的一部分。
    dram_end = pe_end = 0
    for m_group in range(m_group_count):
        active_tiles = min(num_tile, M - m_group * num_tile)
        # 下一 M 组等上组排空再读权重；未假定跨 M 组权重预取为零成本。
        dram_end = max(dram_end, pe_end)
        dram_end += active_tiles * D * weight_bits * units_per_bit
        previous_end = None
        previous_write_bits = 0
        for execute_cycles, read_bits, columns, requests, wait, extra in chunk_stats:
            dram_end += read_bits * units_per_bit
            current_end = max(pe_end, dram_end) + execute_cycles * units_per_cycle
            if previous_end is not None:
                # 同一 DRAM 总线先读当前块、后写前一块，均可与 PE 运算重叠。
                dram_end = max(dram_end, previous_end) + previous_write_bits * units_per_bit
            pe_end = previous_end = current_end
            previous_write_bits = active_tiles * columns
            details['pe_cycles'] += execute_cycles
            details['weight_requests_per_tile'] += requests
            details['weight_wait_pe_cycles_per_tile'] += wait
            details['weight_port_extra_pe_cycles'] += extra
        dram_end = max(dram_end, pe_end) + previous_write_bits * units_per_bit
    details['total_cycles'] = (dram_end + units_per_cycle - 1) // units_per_cycle
    return details if return_details else details['total_cycles']
    


# def run_gustavSNN_fc(input_tensor, M):
#     """    
#     param input_tensor[T,D,N]
#     param M= cout(conv) or N_out(fc)

#     kernel_matrix[M,D]   * input_tensor[T,D,N]
#     """    
#     P = 8
#     Nr = 4
#     num_pe = 8
#     num_tile = 8  # 对应文章中的M'
#     T,D,N = input_tensor.shape

#     # along N
#     pe_group = ceil_a_by_b(N, P * num_pe)
#     tile_compute_cycles = 0
#     tile_total_cycles = 0
#     lif_cycles = 1
#     MEM_IF_WIDTH = 6.4
#     # gustav中，spike和权重都是直接从DRAM中获取的，因此直接增加这个周期就行。
#     # 第一步是cout，即代码中的M，给8个tile，但是不知道gustav的DRAM带宽，也和prosperity对齐吧
#     # 每个cout都会把整个[T,D,N]的特征图处理完，才会切换下一个cout
#     # 片上weight g-buffer只有128KB，即使num_tile个cout的权重超过128KB，就得从D维度截断，但是这不影响访存cycles
#     weight_DRAM_cycles_per_tile = ceil_a_by_b(num_tile * D * 8 , PROSPERITY_MEM_IF_WIDTH)

#     # 由于所有core的脉冲tile共享的原因，所以不存在复杂的异步调度，都是同步
#     # 虽然片外spike 使用了NRV格式进行压缩，也就是全零行跳过，和P有关
#     # 即Nr*P*num_pe大小的脉冲矩阵中，这Nr行中，如果第i行全为0，那么第i行的脉冲矩阵就可以跳过，不进行访存
#     # 那么每组m_tile的 脉冲 DRAM 开销需要结合具体的脉冲矩阵来定。
#     # one tile
#     for pe_group_idx in range(pe_group):
#         N_start = pe_group_idx * P * num_pe
#         # along T
#         for t in range(T):
#             # along D
#             nr_group = ceil_a_by_b(D,Nr)
#             for n in range(nr_group):      # 所有tile的PE[0]都共享相同的spike tile，所以需要同步，DISCUSSION章节也写了
#                 D_start = n * Nr
#                 pe_total_cycles = [0]*num_pe
#                 pe_compute_cycles = [0]*num_pe
#                 pe_load_spike_cycles = [0]*num_pe
#                 # 遍历num_pe个PE
#                 # 每个PE都对应一个spike G-BUFFER，每个G-BUFFER的DRAM访存cycles是不一样的
#                 # 虽然每个spike G-BUFFER的DRAM访存cycles是不一样的，但是得由于共享权重，因此
#                 # 每个m_tile遍历整个D维度时，每个Nr的cyels是由max（计算周期+DRAM中读spike周期）决定的
#                 for i in range(num_pe):
#                     # 每个PE一次计算[Nr,P]大小的输入
#                     pe_compute_cycles[i] += (input_tensor[t,D_start:D_start+Nr, (N_start + i*P) : (N_start + i*P + P)]).sum().item()
#                     # 计算 [Nr, P] 子矩阵中的非零行数量（NRV格式压缩时全零行可跳过访存）
#                     rows = torch.any(input_tensor[t, D_start:D_start+Nr, (N_start + i*P) : (N_start + i*P + P)] != 0, dim=-1).sum().item()
#                     pe_load_spike_cycles[i] += ceil_a_by_b(rows * P, MEM_IF_WIDTH)
#                     # 根据 fig9，取脉冲和计算是流水化的
#                     pe_total_cycles[i] += max(pe_compute_cycles[i], pe_load_spike_cycles[i])
#                 #tag 每次切换nr时都需要同步
#                 tile_total_cycles += max(pe_total_cycles)
#                 tile_compute_cycles += max(pe_compute_cycles)

#             #遍历完D维度后，需要执行lif, 根据论文中fig9，我可以认为P个神经元的膜电位更新是串行的, 但是计算和lif肯定是串行的
#             # 而且由于是lif，所有神经元都要leak/spike
#             #当然，生成脉冲后也就写回片外了，假设lif和写回也是流水的，因此额外再增加最最后的一次写回周期就行
#             tile_total_cycles += lif_cycles * P + ceil_a_by_b(P, MEM_IF_WIDTH)
    
#     # 根据weight G-BUFFER的大小可知是PINGPONG结构，因此除了初始化时的加载周期，后续权重加载和计算是流水话的。
#     # 而且由于m_tile之间计算cycles完全相同，因此无需额外对M进行一个for循环
#     #tag 当然，这里建模没有考虑spike G-BUFFER是pingpong结构
#     #tag 不仅如此，我还忽略了从 g-buffer中取数据到每个tile中的cycles
#     total_comput_cycles = weight_DRAM_cycles_per_tile + max(tile_total_cycles, weight_DRAM_cycles_per_tile) * ceil_a_by_b(M,num_tile)
#     return total_comput_cycles
    
# 优化执行速度版
# def run_gustavSNN_fc(input_tensor, M):
#     """
#     保持原周期公式，仅将逐 PE、逐 Nr 分块的统计改为张量归约以加快代码运行数据
#     """
#     P = 8
#     Nr = 4
#     num_pe = 8
#     num_tile = 8
#     T, D, N = input_tensor.shape
#     pe_group = ceil_a_by_b(N, P * num_pe)
#     tile_total_cycles = 0
#     lif_cycles = 1
#     MEM_IF_WIDTH = 6.4
#     weight_DRAM_cycles_per_tile = ceil_a_by_b(num_tile * D * 8, PROSPERITY_MEM_IF_WIDTH)
#     nr_group = ceil_a_by_b(D, Nr)

#     # rows 只能取 0..Nr；查表复用原 ceil_a_by_b，避免改变浮点带宽取整。
#     load_cycles = torch.tensor(
#         [ceil_a_by_b(rows * P, MEM_IF_WIDTH) for rows in range(Nr + 1)],
#         device=input_tensor.device,
#     )
#     # 每次仅处理 64 列，限制临时张量大小，不为整个 N 维同时分配 padding。
#     for pe_group_idx in range(pe_group):
#         N_start = pe_group_idx * P * num_pe
#         chunk = input_tensor[:, :, N_start:N_start + P * num_pe]
#         chunk = F.pad(chunk, (0, P * num_pe - chunk.shape[-1], 0, nr_group * Nr - D))
#         blocks = chunk.reshape(T, nr_group, Nr, num_pe, P)
#         pe_compute = blocks.sum(dim=(2, 4))
#         rows = (blocks != 0).any(dim=4).sum(dim=2)
#         # 仍然先在每个 PE 取 max(计算,读 spike)，再对 PE 取 max，
#         # 最后沿 T/Nr 求和；不可交换 max 和 sum，否则就改变同步边界。
#         tile_total_cycles += torch.maximum(pe_compute, load_cycles[rows]).amax(dim=2).sum().item()
#     # 原版对尾块也处理 P 个神经元，且 M=0 时仍计权重初始化；原样保留。
#     tile_total_cycles += pe_group * T * (lif_cycles * P + ceil_a_by_b(P, MEM_IF_WIDTH))
#     return (weight_DRAM_cycles_per_tile
#             + max(tile_total_cycles, weight_DRAM_cycles_per_tile) * ceil_a_by_b(M, num_tile))
  














def run_prosperity_conv(Cout, input_tensor, kernel_size=3, stride=1, padding=1, groups=1 , spike_stored_in_buffer=False):
    """
    支持分组卷积的 Prosperity 评估
    input_tensor: [T, B, Cin, H, W]
    """
    input_tensor = expand_nonbinary_timesteps(input_tensor)
    T, B, Cin, H, W = input_tensor.shape
    
    # 兼容 tuple 参数
    k_h, k_w = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
    s_h, s_w = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    p_h, p_w = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    
    # 计算分组参数
    Cin_g = Cin // groups
    Cout_g = Cout // groups
    K_g = k_h * k_w * Cin_g  # Prosperity 中的 K
    N_g = Cout_g             # Prosperity 中的 N (即 Cout)
    
    oH = (H - k_h + 2 * p_h) // s_h + 1
    oW = (W - k_w + 2 * p_w) // s_w + 1
    
    total_cycles = 0

    # ==========================================
    # 核心：拆分为 g 个独立的小 GEMM
    # ==========================================
    for g in range(groups):
        cin_start = g * Cin_g
        cin_end = (g + 1) * Cin_g
        
        # 提取当前组的输入特征图
        input_group = input_tensor[:, :, cin_start:cin_end, :, :]
        
        # 展平 T 和 B 维度送入 img2col
        input_group_flat = torch.flatten(input_group, start_dim=0, end_dim=1)
        
        # img2col 提取 Patch [TB, oh*ow, K_g]
        input_matrix_g = img2col(input_group_flat, kernel_size, stride, padding)
        
        # Prosperity 要求的 Spike Tensor 形状 [TB*oh*ow, K_g]
        spike_tensor_g = torch.flatten(input_matrix_g, start_dim=0, end_dim=1)
        
        # 计算当前组的周期并累加
        cycles = run_prosperity_fc(N_g, spike_tensor_g, spike_stored_in_buffer)
        total_cycles += cycles
        
    return total_cycles

# Prosperity DRAM 接口位宽（bit/cycle），默认 1024b ———— 500MHZ下就是64GB/s的带宽，
# 注意：该假设意味着一拍可向片上 SRAM 写入 1024 bit。以 act tile 为例
# （256 行 x 16 bit = 4096 bit），4 拍即可填满整个 act buffer，隐含要求
# SRAM 写口也有 1024 bit/拍（行宽 1024 或 64 bank 并行），若实际 SRAM 行宽
# 只有 16 bit，应将此值改小以匹配物理带宽。
PROSPERITY_MEM_IF_WIDTH = 1024

# 优化版，去除冗余的 n 循环
# 适配版（无 accelerator/operator 对象）：配置常量与 Prosperity accelerator 默认值一致，
# 权重 8 bit、tile 256x16x128、mem_if_width 1024；DRAM/buffer 流量口径与
# Prosperiy/simulator.py 的 run_fc 完全一致，只是把最内层 n 循环折叠（乘以 tile_num_N
# 或直接利用 Σ_n cur_tile_size_N == N）。
def run_prosperity_fc(N_out, input_tensor, spike_stored_in_buffer=False,
                      weight_stored_in_buffer=False):
    """
    input_tensor[M,K]
    param N_out: 输出维度
    """
    # 初始化统计信息对象
    stats = Stats()
    # 由于只能处理二进制输入，所以需要展开非二进制输入
    input_tensor = expand_nonbinary_timesteps(input_tensor)
    M, K = input_tensor.shape
    N = N_out
    issue_type = 1  # 1: search over tree through prefix 2: sorting based issue
    mem_if_width = PROSPERITY_MEM_IF_WIDTH
    weight_nbits = 8

    tile_size_M = 256
    tile_size_K = 16
    tile_size_N = 128   # 即adder_array_size
    num_popcnt = 8
    sram_size = {}
    sram_size['wgt'] = tile_size_K * tile_size_N * weight_nbits  # global buffer 16 * 128 * 8 bit
    sram_size['act'] = tile_size_M * tile_size_K * 1             # global buffer 256 * 16 bit
    sram_size['out'] = tile_size_M * tile_size_N * 16            # global buffer 256 * 128 * 16 bit

    # 计算各维度需要的分块数量
    tile_num_M = ceil_a_by_b(M, tile_size_M)
    tile_num_K = ceil_a_by_b(K, tile_size_K)
    tile_num_N = ceil_a_by_b(N, tile_size_N)

    # 计算激活值和权重的分块大小（以位为单位）
    act_tile_size = tile_size_M * tile_size_K * 1
    wgt_tile_size = tile_size_K * tile_size_N * weight_nbits

    # 判断激活值缓冲区状态（根据SRAM大小）
    if M * K <= sram_size['act']:
        buffer_state_act = "store all"
    elif act_tile_size * tile_num_K <= sram_size['act']:
        buffer_state_act = "store row"
    elif act_tile_size <= sram_size['act']:
        buffer_state_act = "store single tile"
    else:
        raise Exception("single tile cannot fit in sram act buffer")

    # 判断权重缓冲区状态（根据SRAM大小）
    if K * N * weight_nbits <= sram_size['wgt']:
        buffer_state_wgt = "store all"
    elif wgt_tile_size * tile_num_K <= sram_size['wgt']:
        buffer_state_wgt = "store col"
    elif wgt_tile_size <= sram_size['wgt']:
        buffer_state_wgt = "store single tile"
    else:
        raise Exception("single tile cannot fit in sram wgt buffer")

    # 初始化周期计数器
    compute_cycles = 0
    preprocess_cycles = 0

    # 移除冗余的 n 循环：cur_act 仅由 m 和 k 决定。
    # 原版 m->n->k 三层循环里与 n 有关的累加按以下恒等式折叠：
    #   - 按 tile 重复的流量直接乘 tile_num_N；
    #   - Σ_n cur_tile_size_N == N（含 N 维度最后一个不满块）。
    for m in range(tile_num_M):
        for k in range(tile_num_K):
            # 计算当前分块的实际大小
            cur_tile_size_M = min(tile_size_M, M - m * tile_size_M)
            cur_tile_size_K = min(tile_size_K, K - k * tile_size_K)

            # 提取当前激活值分块
            cur_act = input_tensor[m * tile_size_M: m * tile_size_M + cur_tile_size_M,
                                   k * tile_size_K: k * tile_size_K + cur_tile_size_K]

            # 如果脉冲未存储在缓冲区中，需要从DRAM读取
            if not spike_stored_in_buffer:
                act_load = cur_tile_size_M * cur_tile_size_K
                if buffer_state_act == "store single tile":
                    # 每个 n 列块都要重新从 DRAM 流式读取同一块激活
                    stats.reads['dram'] += act_load * tile_num_N
                    stats.writes['g_act'] += act_load * tile_num_N
                elif buffer_state_act == "store row":
                    stats.reads['dram'] += act_load * tile_num_N
                    stats.writes['g_act'] += act_load * tile_num_N
                # 整个激活矩阵都可以存在缓冲区中：原版条件 n == 0，
                # 折叠 n 循环后每个 (m, k) 对计一次。
                elif buffer_state_act == "store all":
                    stats.reads['dram'] += act_load
                    stats.writes['g_act'] += act_load
            # 如果权重未存储在缓冲区中，需要从DRAM读取
            if not weight_stored_in_buffer:
                # Σ_n cur_tile_size_N == N
                wgt_load = cur_tile_size_K * N * weight_nbits
                if buffer_state_wgt == "store single tile":
                    stats.reads['dram'] += wgt_load
                    stats.writes['g_wgt'] += wgt_load
                # 权重按列缓冲：m == 0 时每个 k 行块的所有 n 列块各加载一次
                elif buffer_state_wgt == "store col" and m == 0:
                    stats.reads['dram'] += wgt_load
                    stats.writes['g_wgt'] += wgt_load
                # 全部权重可缓冲：m == 0 时加载一遍
                elif buffer_state_wgt == "store all" and m == 0:
                    stats.reads['dram'] += wgt_load
                    stats.writes['g_wgt'] += wgt_load

            # 启用 product sparsity
            preprocess_act, prefix_array = find_product_sparsity(cur_act)
            # get_prosparsity_cycles(cur_act) 返回需要搜索前缀的行数；
            # ceil(cur_M / num_popcnt) 是所有行 popcount 的固定开销。
            prosparsity_cycles = get_prosparsity_cycles(cur_act) + ceil_a_by_b(cur_tile_size_M, num_popcnt)
            nnz_preprocess = torch.sum(preprocess_act != 0).item()
            # g_act 读取每个 n 列块都发生一次；g_wgt 仅读取非零行对应权重，Σ_n cur_N == N
            stats.reads['g_act'] += cur_tile_size_M * cur_tile_size_K * tile_num_N
            stats.reads['g_wgt'] += nnz_preprocess * N * weight_nbits
            # 查找预处理激活值中全零行，如果原来不是全零则需要一个周期
            nnz_each_row = torch.sum(preprocess_act != 0, dim=-1)
            nnz_each_row_ori = torch.sum(cur_act != 0, dim=-1)
            all_zero_row = nnz_each_row == 0
            all_zero_row_ori = nnz_each_row_ori == 0
            if issue_type == 1:
                forest = construct_prosparsity_forest(prefix_array)
                depth = nx.dag_longest_path_length(forest)
                cur_issue_cycles = ceil_a_by_b(depth, 4) * cur_tile_size_M
            else:
                cur_issue_cycles = 0     # 默认选项，我理解的是排序所需周期
            # 当前计算周期 = 处理后的(非零值数+全零行数) - 原始全零行数
            cur_compute_cycles = (nnz_preprocess   # 没有被匹配上的点的数量
                                  + torch.sum(all_zero_row).item()   # 完全匹配的
                                  - torch.sum(all_zero_row_ori).item())  # 全零行直接跳过
            stats.num_ops += nnz_preprocess * N

            # 部分和读写：原版每个 (m,n,k) 计 cur_M*cur_N，折叠后 Σ_n cur_N == N
            stats.reads['g_psum'] += cur_tile_size_M * N * 16
            stats.writes['g_psum'] += cur_tile_size_M * N * 16

            # 将原来在 n 循环中的累加，改为直接乘以 tile_num_N
            compute_cycles += max(cur_issue_cycles, cur_compute_cycles) * tile_num_N
            preprocess_cycles += prosparsity_cycles * tile_num_N

    # 这部分延迟无法与计算重叠，必须等待数据加载完成才能开始计算
    init_mem_access = 0
    if not weight_stored_in_buffer:
        init_mem_access += min(tile_size_K, K) * min(tile_size_N, N) * weight_nbits
    if not spike_stored_in_buffer:
        init_mem_access += min(tile_size_K, K) * min(tile_size_M, M)

    total_mem_access = stats.reads['dram'] + stats.writes['dram']
    middle_mem_access = total_mem_access - init_mem_access
    init_latency = ceil_a_by_b(init_mem_access, mem_if_width)
    middle_latency = ceil_a_by_b(middle_mem_access, mem_if_width)
    stats.compute_cycles = max(compute_cycles, preprocess_cycles)
    stats.preprocess_stall_cycles = max(0, preprocess_cycles - compute_cycles)
    stats.mem_stall_cycles = init_latency + max(0, middle_latency - stats.compute_cycles)
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles

    # 返回含访存停顿的总周期，与 Prosperity 网络级累加口径一致
    return stats.total_cycles

def run_LIF(tensor, T, num_post_neuron, batch_size=1,
            last_tile_size_M=256, last_tile_size_N=128, LIF_array_size=32):
    """
    tensor: 膜电位特征图。可以是 torch.Tensor（取 numel），也可以直接传 int
            （元素总数 = num_post_neuron * T * batch_size）。
    T: 时间步数；整数 spike 已展开为 0/1 时传展开后的 T。
    num_post_neuron: 单个时间步、单个 batch 的输出神经元数（conv 为 Cout*oH*oW）。
    batch_size: batch 大小。

    返回 (stats, spike_stored_in_buffer)：
      stats.compute_cycles : LIF 全部工作量（随前层 GEMM 流水时主体被隐藏）
      stats.LIF_latency    : 流水线排空延迟，即一个前层 GEMM 输出 tile 对应的
                             LIF 尾巴，网络级只在 GEMM 周期之外追加这一项
    """
    stats = Stats()
    mem_if_width = PROSPERITY_MEM_IF_WIDTH

    # 与 run_prosperity_fc 一致的 on-chip global buffer 容量（bit）
    sram_size = {}
    sram_size['wgt'] = 16 * 128 * 8          # global buffer 16 * 128 * 8 bit
    sram_size['act'] = 256 * 16 * 1          # global buffer 256 * 16 bit
    sram_size['out'] = 256 * 128 * 16        # global buffer 256 * 128 * 16 bit

    if torch.is_tensor(tensor):
        size_of_tensor = tensor.numel()
    else:
        size_of_tensor = int(tensor)
    stats.reads['g_psum'] = size_of_tensor * 16   # 膜电位位宽是16位
    spike_stored_in_buffer = False
    # 计算整层的lif工作量
    num_round = ceil_a_by_b(num_post_neuron * batch_size, LIF_array_size)
    compute_cycles = num_round * T * 2 # one cycle for addition one cycle for mutiplication
    stats.compute_cycles = compute_cycles

    # 最后一个tile的lif latency
    num_neuron_in_last_tile = last_tile_size_M * last_tile_size_N 
    latency = ceil_a_by_b(num_neuron_in_last_tile, LIF_array_size) * T * 2
    latency = min(latency, compute_cycles)
    stats.LIF_latency = latency


    # 如果经过lif后，这一层的output tensor可以放在SRAM中
    if size_of_tensor < sram_size['act']:
        stats.writes['g_act'] = size_of_tensor
        spike_stored_in_buffer = True
    # output tensor不能放在SRAM中，需要写入DRAM
    else:
        stats.writes['dram'] = size_of_tensor
        mem_cycles = size_of_tensor // mem_if_width
        stats.mem_stall_cycles += max(0, mem_cycles - stats.compute_cycles)
        # no extra memory latency since spike is generated in streaming manner, each time 32 bits
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles

    return stats, spike_stored_in_buffer


def construct_prosparsity_forest(prefix_array: torch.Tensor):
    forest = nx.DiGraph()
    for i in range(prefix_array.shape[0]):
        if prefix_array[i].item() == -1:
            continue
        forest.add_edge(prefix_array[i].item(), i)
    return forest


def get_prosparsity_cycles(act: torch.Tensor):
    ##如果某行非零元素数量大于1，则该行需要搜索前缀行
    nnz_each_row = torch.sum(act != 0, dim=-1)    #形状为[行数M]
    num_searched_row = torch.sum(nnz_each_row > 1).item() 

    return num_searched_row

#tag，关键函数，act是二维张量[256,16]，得到被处理后的act，非零元素数量就是所需cycles
# done,confim by yuping in 20260606
def find_product_sparsity(act: torch.Tensor):
    preprocessed_act = act.clone()
    # 初始化前缀数组，用于记录行之间的依赖关系
    prefix_array = torch.ones(act.shape[0])
    prefix_array = -prefix_array # 将所有元素初始化为-1，表示没有找到匹配的前缀行
    for i in range(act.shape[0]):
        cur_row = act[i]
        nnz = torch.sum(cur_row != 0).item()
        # 如果非零元素少于2个，则跳过该行（没有足够的数据进行匹配）
        if nnz < 2:
            continue
        # 计算当前行与所有行的逻辑与操作，用于查找相似的激活模式
        and_result = torch.logical_and(cur_row, act)
        # and_result 形状: [行数M, 特征数K] - 当前行与所有行的逻辑与结果
        equalities = torch.eq(and_result, act)
        # equalities 形状: [行数M, 特征数K] - 逻辑与结果是否等于原数据
        # 对于每个位置(i,j)，equalities[i,j] = (and_result[i,j] == act[i,j])即True或False
        # 对于每个行向量，如果逻辑与结果等于原向量，则说明原向量是当前行的子集
        # torch.all(equalities, dim=-1) 在最后一个维度（特征维度）上检查所有元素是否都为True
        is_subset = torch.all(equalities, dim=-1)
        # is_subset 形状: [行数M] - 每一行是否为当前行的子集的布尔值
        # 如果is_subset[i]为True，则表示act[i]是cur_row的子集
        # 检查每行是否与当前行完全相等
        equalities = torch.eq(cur_row, act)
        is_equal = torch.all(equalities, dim=-1)
        # 创建索引条件，排除当前行及之前已经处理过的行
        is_bigger_index = torch.arange(act.shape[0], device=act.device) >= i
        # 排除与当前行完全相等且索引大于等于当前行的行
        is_excluded = torch.logical_and(is_equal, is_bigger_index)
        # 确定真正的子集行（是子集但不完全相等且索引大于当前行），形状为[行数M]
        is_real_subset = torch.logical_and(is_subset, ~is_excluded)
        # 如果没有找到任何子集行，则继续处理下一行
        if torch.sum(is_real_subset) == 0:
            # CAM no match
            continue
        # 提取所有子集行
        subset_row = act[is_real_subset]
        # 计算每个子集行中非零元素的数量，形状为[行数M]
        subset_row_nnz = torch.sum(subset_row != 0, dim=-1)
        # 找到非零元素最多的子集行大小
        max_subset_size = torch.max(subset_row_nnz).item()
        # 获取非零元素最多的子集行
        max_subset = subset_row[torch.argmax(subset_row_nnz)]
        # 获取子集行的索引
        subset_index = torch.nonzero(is_real_subset).flatten()
        # 计算每个子集行的激活总和
        subset_size = torch.sum(act[is_real_subset], dim=-1)
        # 找到激活总和最大的子集行及其索引
        max_subset_size, max_subset_index = torch.max(subset_size, dim=-1)
        # 如果最大子集大小小于1，则继续处理下一行
        if max_subset_size.item() < 1:
            continue
        # 记录当前行的前缀行索引
        prefix_array[i] = subset_index[max_subset_index].item()
        # 使用异或操作更新预处理后的激活数据，消除重复计算
        # if max_subset_size > 1: # can also reuse even when the size is 1
        preprocessed_act[i] = torch.logical_xor(preprocessed_act[i], max_subset)
    
    return preprocessed_act, prefix_array


def run_DLA_layer(input_tensor, operator_type, cout, groups=1, *, with_lif=True):
    """返回按 with_lif 选择的串行总周期、原始阶段统计和旧公式周期。"""
    estimate, legacy_cycles = estimate_dla_runner_layer(
        operator_type, tuple(input_tensor.shape), cout, groups)
    total_cycles = estimate.total_cycles if with_lif else estimate.total_cycles - estimate.activation_cycles
    return total_cycles, estimate, legacy_cycles


def run_prosperity_layer_with_lif(gemm_cycles, num_post_neuron, T_lif, batch_size=1):
    """
    LIF 主体随前层 GEMM tile 流式并行被overlap，总cycles= GEMM 周期 + 尾部 tile 的 LIF_latency
    
    T_lif 必须是整数 spike 展开为 0/1 后的实际时间步数。
    num_post_neuron: 单个时间步、单个 batch 的输出神经元数
                     （conv: Cout*oH*oW；线性投影: N_pos*dim）。
    """
    total_membrane_elements = num_post_neuron * T_lif * batch_size
    lif_stats, _ = run_LIF(total_membrane_elements, T_lif, num_post_neuron,
                           batch_size=batch_size)
    return gemm_cycles + lif_stats.LIF_latency, lif_stats



# PTB 的核心架构设计
def StSAP(act: torch.Tensor):
    # systolic array of size 16x8
    # vgg16的第一个卷积层就是[1024,576,8]
    whole_act = act.clone()
    total_length = 0
    
    # 遍历张量的第一个维度,即每个输出神经元 以及 时间步, output_H * output_H * timesteps/8
    for i in range(whole_act.shape[0]): #1024
        processed_act = whole_act[i]   #[576,8]
        # find row with all zero，dim=-1确保是timestemp维度
        all_zero_row = torch.sum(processed_act != 0, dim=-1) == 0
        # find row with all nonzero，dim=-1确保是timestemp维度
        all_nonzero_row = torch.sum(processed_act != 0, dim=-1) == processed_act.shape[-1]
        # 合并全零tensor和全非零tensor，这些tensor不需要特殊处理
        excepted_row = torch.logical_or(all_zero_row, all_nonzero_row)
        # get all the row except all zero row and all nonzero row
        processed_act = processed_act[~excepted_row]
        # 初始化当前处理块的大小
        cur_size = processed_act.shape[0]  # 576
        searched_row = torch.zeros(processed_act.shape[0], dtype=torch.bool) #[576]
        
        # 遍历张量第二个维度，k*k*Cin行，压缩无重叠的行，得到实际需要计算的行数目cur_size
        # for i in range(processed_act.shape[0] - 1):   
        for j in range(processed_act.shape[0]):   #Cin*k*k，576
            # 如果当前行已经被处理过，则跳过
            if searched_row[j]:
                continue
            cur_row = processed_act[j]
            rest_row = processed_act[j+1:]
            # 计算当前行与剩余行的逻辑与操作，找出重叠部分
            and_result = torch.logical_and(cur_row, rest_row)
            # 查找与当前行无重叠的行（可以合并处理的行）
            non_overlap_row = torch.sum(and_result, dim=-1) == 0
            # pad the left of non_overlap_row to the original shape
            non_overlap_row = torch.cat([torch.zeros(j + 1, dtype=torch.bool), non_overlap_row])
            # 确保这些行未被搜索过
            non_overlap_row = torch.logical_and(non_overlap_row, ~searched_row)
            # find the first nonzero in non_overlap_row
            first_nonzero = torch.argmax(non_overlap_row.to(torch.int)).item()
            if first_nonzero != 0:
                cur_size -= 1
                searched_row[first_nonzero] = True
        
        processed_size = cur_size + torch.sum(all_nonzero_row)
        total_length += processed_size.item()
    return total_length


def run_eyeriss_conv_fc(operator: Union[FC, Conv2D, LIFNeuron]):
    # a 14 * 12 systolic array, weight stationary
    stats = Stats()
    if isinstance(operator, FC):
        stats.compute_cycles += ceil_a_by_b(operator.output_dim, 12) * (operator.input_dim * operator.batch_size * operator.time_steps * operator.sequence_length // 14)
    elif isinstance(operator, Conv2D):
        eq_fc = conv2d_2_fc(operator)
        stats.compute_cycles += ceil_a_by_b(eq_fc.output_dim, 12) * (eq_fc.input_dim * eq_fc.batch_size * eq_fc.time_steps * eq_fc.sequence_length // 14)
    elif isinstance(operator, LIFNeuron):
        stats.compute_cycles += (operator.time_steps // 12) * operator.num_neuron * operator.batch_size // 14
    else:
        pass
    stats.total_cycles = stats.compute_cycles
    print(operator.name)
    print("total cycles: ", stats.total_cycles)
    return stats


def run_sato_lif(operator: LIFNeuron):
    stats = Stats()
    stats.compute_cycles = operator.num_neuron * (torch.log2(torch.tensor(operator.time_steps)).item() + 1) # 1 for leaf node comparator
    stats.total_cycles = stats.compute_cycles
    return stats






def run_PTB_convfc(input_tensor, type="conv", kernel_size=3, stride=1, padding=1, cout=0,  OH=None, OW=None, M = None,  K = None):
    """
    运行PTB架构的卷积/全连接层仿真
    
    Args:
        input_tensor (torch.Tensor): 输入张量，形状为 [time_steps, sequence_length, input_dim] 或 [time_steps, input_channel, input_H, input_W]
    
    Returns:
        Stats: 统计信息对象，包含周期数、内存访问等信息
    """
    print("----------run PTB_convfc ------------\n")
    # 使用16x8或21x8的脉动阵列
    stats = Stats()
    time_window_size = 4
    if (type == "fc"):
        T, M, K = input_tensor.shape
        stats.reads['dram'] += T * M * K 
        stats.reads['dram'] += cout * K * 8   # weight is 8bit
        stats.writes['dram'] += T * M * cout  
    elif (type == "conv"):
        T, C, H, W = input_tensor.shape  
        stats.reads['dram'] += T * C * H * W
        stats.reads['dram'] += cout * C * kernel_size * kernel_size * 8
        stats.writes['dram'] += T * cout * OH * OW 


    input_tensor = pad_to_power_of_2(input_tensor, 0)
    input_shape = input_tensor.shape
    # 卷积输入已经没有独立的批次维度；高度为 1 时仍必须保留高度维度，
    # 否则时间窗口压缩会生成三维张量，导致 img2col 的六维重排失败。
    # 卷积路径只压缩时间步维度；全连接路径保持原有维度处理。
    new_shape = [-1, time_window_size]
    new_shape.extend(input_shape[1:])  # 保持dim0之外的其他维度不变
    input_tensor = input_tensor.reshape(new_shape)  # 把时间步维度按照time_window，展开为dim0和dim1，[T/4,4,M,K]
    input_tensor = input_tensor.sum(dim=1)  # 完成time_window的压缩，这个时候就只有[T/4,M,K]

    if (type == "fc"):
        if len(input_tensor.shape) == 2:    
            input_tensor = input_tensor.permute(1, 0).contiguous()
        elif len(input_tensor.shape) == 3:
            # [T,M,K]
            # 只有压缩后的time batch大于等于8时，才能填满脉动阵列
            # 根据压缩后的 timestep再去reshape，第一个维度最大为8，但是仅能从压缩后的时间步维度中reshape。
            T_compressed = input_tensor.shape[0]
            # 时间列数最多为 8；不足 8 个压缩时间步时使用实际数量，
            # 避免把不同点误当作时间列，或在单个点的输入上产生非法重排。
            colume = min(T_compressed, 8)
            input_tensor = input_tensor.reshape([colume, -1, input_tensor.shape[-1]])
            input_tensor = input_tensor.permute(1, 2, 0).contiguous()
        else:
            raise Exception("unsupported input shape")
        sequence_length = M
        input_dim = K
        output_dim = cout
        
    elif (type == "conv"):
        input_tensor = img2col(input_tensor, kernel_size, stride, padding)  
        # [T,M,K]
        # the first dimension should be 8 to fully utilize the systolic array, change the second dimension to make first dimension 8
        # 只有压缩后的time batch大于等于8时，才能填满脉动阵列
        # 根据压缩后的 timestep再去reshape，第一个维度最大为8，但是仅能从压缩后的时间步维度中reshape。
        T_compressed = input_tensor.shape[0]
        # 时间列数由压缩时间步数量决定，不足 8 时不能强制填满阵列，
        # 否则小尺寸输出特征图无法重排，或不同点会被合并到时间列中。
        colume = min(T_compressed, 8)
        input_tensor = input_tensor.reshape([colume, -1, input_tensor.shape[-1]])
        input_tensor = input_tensor.permute(1, 2, 0).contiguous()
        sequence_length = OH * OW
        input_dim = C * kernel_size * kernel_size
        output_dim = cout   
    
    if True:
        input_length = StSAP(input_tensor)
    else:
        input_length = np.prod(input_tensor.shape[:-1])
    
    repeate_times = ceil_a_by_b(output_dim, 16)     
    stats.compute_cycles += input_length * repeate_times * (time_window_size)
    num_cold_start = sequence_length
    stats.compute_cycles += num_cold_start * 16 * 2    

    
    init_mem_access = 16 * 8 * (8 + 4)  
    total_mem_access = stats.reads['dram'] + stats.writes['dram']  
    middle_mem_access = total_mem_access - init_mem_access  #
    init_latency = init_mem_access // 1024  
    middle_latency = middle_mem_access // 1024  
    
    stats.mem_stall_cycles = init_latency + max(0, middle_latency - stats.compute_cycles)
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles

    return stats

def run_PTB_lif(num_neuron, time_steps):
    stats = Stats()
    stats.compute_cycles = num_neuron * time_steps * 3 // 16
    stats.total_cycles = stats.compute_cycles
    return stats


def run_PTB_layer_with_lif(input_tensor, type="conv", kernel_size=3, stride=1, padding=1, cout=0,  OH=None, OW=None, M = None,  K = None, time_steps=None, num_neuron=None, *, with_lif=True):
    """
    运行 PTB 架构的卷积或全连接层仿真；with_lif=False 跳过泄漏积分发放计算。
    
    Args:
        input_tensor (torch.Tensor): 输入张量，形状为 [time_steps, sequence_length, input_dim] 或 [time_steps, input_channel,input_H, input_W]
        type (str): 层类型，"conv"或"fc"
        kernel_size (int): 卷积核大小
        stride (int): 卷积步长
        padding (int): 卷积填充
        cout (int): 输出通道数
        OH (int): 输出高度
        OW (int): 输出宽度
        M (int): 输入序列长度
        K (int): 输入通道数
        time_steps (int): 时间步长
        num_neuron (int): LIF神经元数
    
    Returns:
        int: 总周期数
       """
    stats = run_PTB_convfc(input_tensor, type, kernel_size, stride, padding, cout, OH, OW, M, K)
    if not with_lif:
        return stats.total_cycles
    lif_stats = run_PTB_lif(num_neuron, time_steps)
    total_cycles = stats.total_cycles + lif_stats.total_cycles
    return total_cycles

