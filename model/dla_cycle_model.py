"""NVDLA INT8 周期模型；runner 使用 RTL 控制规则推导的事件调度模型。

本项目已确认的配置与调度假设（集中在此处，便于后续修改）：
1. 八位有符号整数直接卷积特征图模式，Atomic-C=64、Atomic-K=16。不启用 Winograd、
   image-input、通道后扩展或权重压缩，2-bit 脉冲仍以 INT8 特征图输入。
2. 按源码 BATCH_DISABLE，不启用 multi-batch；T、B 和卷积分组分别
   映射为独立硬件层任务，不能把不同 T/B 拼进同一 stripe 隐藏权重加载。
3. 空间条带长度随输出通道并行度配置：剩余点数不少于 64 时取 32，
   33 至 63 时取 16，不超过 32 时全部合并；最短发射窗口为 16 拍。
4. CSC 每拍向 CMAC 装载一个输出通道的 atomic_c 个权重；每个 stripe
   都重新加载权重。CMAC 寄存器双缓冲：当前 stripe 开始后才可
   装载下一个 stripe 的权重，不能无限提前预取。片上权重加载不是 DRAM 读取。
5. 同一结果的相邻累加至少间隔 11 拍；仅同一空间 stripe 内的 R*S/C
   分块存在此读后写相关，不在不同空间 stripe/输出通道组间强加停顿。
6. 计算区间：CSC接受配置且当前所需输入/权重在CBUF就绪，到CACC最后
   一个entry被SDP接收。CSB配置、DRAM等待另计；不同T/B/组不交织。
   计入SG启动、DL/CMAC/CACC固定流水、64-entry信用和信用返回延迟。
   不设置拟合用固定倍率。尚无同配置RTL仿真校准，不能宣称周期精确。
7. 结果交付缓冲区不允许读出和发送重叠。本配置每个表项包含 16 个元素，
   读出占一拍，再以每拍 16 元素发送一拍，完整表项服务时间为两拍。
   泄漏积分发放计算仍保持每拍 16 元素。
8. 采用 1024 个乘累加单元和 256 二进制千字节共享卷积缓冲区，
   频率为 1 吉赫兹，片外访存总带宽为每秒 64 吉字节，即每拍 64 字节。
   激活吞吐为每拍 16 元素、结果为 16 位；膜电位的片外访存读写由
   activation_extra_* 显式计入，默认 0 不表示自动计入。
9. 缓冲容量采用有效字节数近似：全部驻留时复用权重，否则流式加载两组权重；
   输入仍放不下时按容量比例分块。忽略存储体粒度、地址对齐和重叠输入行，
   不模拟逐拍片外访存仲裁。计算阶段仍假设片上操作数供给无停顿，
   访存时间单独统计，避免与串行阶段和重复计费。
10.建模假设SDP 能执行 LIF，LIF按T逐步执行，不能合并不同T的尾拍。

参考源码固定为 nvdla/hw 提交1a65f1f5b48268accaa47c95f95c2601918be095。
每个流水参数的来源及限制见 dla_rtl_cycle_model.md。

依据：原子操作/stripe/双缓冲/11 拍累加相关与 CACC→SDP 吞吐：
https://nvdla.org/hw/v1/ias/unit_description.html
卷积映射与 stripe 合并策略：
https://nvdla.org/hw/v1/ias/programming_guide.html
CSC 权重加载顺序（WaitStripeBeginHasSent 后逐 kernel 向 CMAC 发送）：
https://github.com/nvdla/hw/blob/master/cmod/csc/NV_NVDLA_csc.cpp
源码配置（BATCH_DISABLE）：
https://github.com/nvdla/hw/blob/master/spec/defs/nv_large.spec
可配置 Atomic-C/K：https://nvdla.org/hw/v2/scalability.html

标准 NVDLA 特征图采用 INT8/INT16/FP16；默认按 INT8 传输这些值。
函数同时返回三个阶段的串行和、以及理想流水重叠时的阶段瓶颈周期。
根据已确认的比较口径，总周期采用计算、片外访存与泄漏积分发放计算的串行阶段和。
该总周期不包含片外访存启动延迟及阶段间流水气泡，并非实际硬件的严格上界。
"""

from dataclasses import dataclass, replace
from collections import deque
import re


# 默认值对应上面的已确认口径；短 stripe 的代价由调度推导，而非固定倍率。
DLA_ATOMIC_C = 64
DLA_ATOMIC_K = 16
DLA_STRIPE_POSITIONS = DLA_ATOMIC_K
DLA_MAX_STRIPE_POSITIONS = 2 * DLA_ATOMIC_K
DLA_WEIGHT_KERNELS_PER_CYCLE = 1
DLA_ACCUMULATOR_REVISIT_CYCLES = 11
DLA_HARDWARE_LAYER_OVERHEAD_CYCLES = 0
# 按项目配置使用 1 吉赫兹频率和每秒 64 吉字节片外访存带宽。
# 带宽表示整个共享接口的读写总吞吐，不是每个方向分别拥有该吞吐。
DLA_CLOCK_HZ = 1_000_000_000
DLA_MEMORY_BYTES_PER_SECOND = 64_000_000_000
DLA_MEMORY_BITS_PER_CYCLE = DLA_MEMORY_BYTES_PER_SECOND * 8 // DLA_CLOCK_HZ
DLA_CBUF_CAPACITY_BYTES = 256 * 1024
DLA_ACTIVATION_ELEMENTS_PER_CYCLE = 16
DLA_ACTIVATION_EXTRA_READ_BITS = 32  # 16-bit电流结果回读 + 16-bit膜电位读取。
DLA_ACTIVATION_EXTRA_WRITE_BITS = 16  # 膜电位写回；输出脉冲另按INT8写回。
DLA_RTL_SOURCE_COMMIT = "1a65f1f5b48268accaa47c95f95c2601918be095"


@dataclass(frozen=True)
class DlaBufferPlan:
    """容量级近似；不模拟存储体分配、地址对齐和逐拍访存仲裁。"""

    mode: str
    capacity_bytes: int
    full_input_bytes: int
    full_weight_bytes: int
    resident_weight_bytes: int
    output_positions_per_tile: tuple[int, ...]
    weight_reusable: bool
    weight_fetches_per_group: int


@dataclass(frozen=True)
class DlaRtlTiming:
    """寄存器级延迟：cycle为周期起点，握手完成后才更新下一拍状态。

    默认配置没有跨分区RT模块。DC旁路PRA，不能给DL末端再加5拍PRA。
    输入端平衡流水仍保留5拍，以便数据与权重的启动延迟匹配。
    """

    sg_first_weight_pop_cycle: int = 3  # IDLE→BUSY、package寄存、FIFO输入寄存。
    sg_output_register_cycles: int = 1
    dl_input_balance_cycles: int = 5  # CSC_DL_PRA_LATENCY，只在DL输入端保留。
    dl_package_cycles: int = 4  # DC sub_h_total=1，选择dl_pvld_d4。
    dl_request_register_cycles: int = 2
    cbuf_read_cycles: int = 6
    dl_response_cycles: int = 1  # DC选择dat_rsp_pvld_d1。
    dl_output_register_cycles: int = 3  # dat_out、dl_out、sc2mac三个寄存级。
    cmac_input_retime_cycles: int = 2
    cmac_status_cycles: int = 2  # MAC_PD_LATENCY；提前3拍给CACC预读。
    cmac_output_retime_cycles: int = 2
    cacc_status_align_cycles: int = 3  # calculator.calc_valid三级延迟。
    cacc_final_cell_cycles: int = 2
    cacc_final_register_cycles: int = 1
    delivery_write_cycles: int = 1
    delivery_counter_cycles: int = 1
    delivery_read_cycles: int = 1
    delivery_read_issue_interval: int = 2  # data_left_mask禁止同拍发送/下一读。
    credit_capacity: int = 64  # 两种配置分别为 4×16 和 2×32，信用容量均为 64 个表项。
    credit_return_cycles: int = 3  # CACC返回寄存、SG跨分区寄存、credit_cnt。
    csc_done_flush_cycles: int = 48  # CSC_SG_DONE_FLUSH=6'h30。

    @property
    def data_pop_to_cmac_cycles(self) -> int:
        return (self.sg_output_register_cycles + self.dl_input_balance_cycles
                + self.dl_package_cycles + self.dl_request_register_cycles
                + self.cbuf_read_cycles + self.dl_response_cycles
                + self.dl_output_register_cycles)

    @property
    def data_pop_to_sdp_cycles(self) -> int:
        return (self.data_pop_to_cmac_cycles + self.cmac_input_retime_cycles
                + self.cmac_status_cycles + self.cmac_output_retime_cycles
                + self.cacc_status_align_cycles + self.cacc_final_cell_cycles
                + self.cacc_final_register_cycles + self.delivery_write_cycles
                + self.delivery_counter_cycles + self.delivery_read_cycles)


DLA_RTL_TIMING = DlaRtlTiming(delivery_read_issue_interval=1 + DLA_ATOMIC_K // 16)


@dataclass(frozen=True)
class DlaStripeTrace:
    """单硬件层中一个stripe operation的周期记录；默认不保存以节省内存。"""

    kernel_group: int
    spatial_stripe: int
    reduction_step: int
    length: int
    weight_pop_cycle: int
    data_pop_cycle: int
    first_cmac_cycle: int
    last_cmac_cycle: int
    final_partial_sum: bool
    last_sdp_cycle: int | None


@dataclass(frozen=True)
class DlaDcComputeEstimate:
    """DC feature 片上调度分解；权重加载总服务时间不能再加到计算总周期。"""

    compute_cycles: int
    atomic_cycles: int
    weight_load_cycles: int
    weight_startup_cycles: int
    weight_stall_cycles: int
    accumulator_stall_cycles: int
    hardware_layer_overhead_cycles: int
    hardware_layer_count: int
    kernel_groups_per_hardware_layer: int
    input_channel_blocks: int
    spatial_stripe_lengths: tuple[int, ...]
    stripe_operation_count: int
    model_kind: str = "stripe_analytic"
    controller_startup_cycles: int = 0
    controller_stall_cycles: int = 0
    credit_stall_cycles: int = 0
    pipeline_cycles: int = 0
    delivery_tail_cycles: int = 0
    first_cmac_cycle: int = 0
    last_cmac_cycle: int = 0
    last_sdp_cycle: int = 0
    csc_done_cycle: int = 0
    rtl_waveform_calibrated: bool = False
    trace: tuple[DlaStripeTrace, ...] = ()


def _positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def plan_dla_buffer(
    *, input_elements_per_sample: int, output_positions_per_sample: int,
    input_channels: int, output_channels: int, kernel_elements: int,
    time_steps: int, batch_size: int, groups: int = 1,
    capacity_bytes: int = DLA_CBUF_CAPACITY_BYTES,
    physical_activation_bits: int = 8, weight_bits: int = 8,
    atomic_k: int = DLA_ATOMIC_K,
) -> DlaBufferPlan:
    """按有效字节数估算驻留和分块，不进行精确的存储体容量规划。

    输入与权重共用容量。先尝试全部驻留，否则保留两个输出通道组的权重，
    其余容量留给输入。输入仍放不下时，按容量比例估算空间分块数。
    全部权重驻留时，同一卷积分组的不同时间步与批次复用权重；流式加载时，
    每个时间步、批次、空间分块重新读取权重，但不因此重复读取整个输入。
    不计空间分块的重叠行、地址对齐和存储体粒度，因此是近似模型而非严格上界。
    """
    for name, value in dict(input_elements_per_sample=input_elements_per_sample,
                            output_positions_per_sample=output_positions_per_sample,
                            input_channels=input_channels, output_channels=output_channels,
                            kernel_elements=kernel_elements, time_steps=time_steps,
                            batch_size=batch_size, groups=groups, capacity_bytes=capacity_bytes,
                            physical_activation_bits=physical_activation_bits,
                            weight_bits=weight_bits, atomic_k=atomic_k).items():
        _positive_integer(name, value)
    if (input_channels % groups or output_channels % groups
            or input_elements_per_sample % input_channels):
        raise ValueError("Invalid channel/group/input element dimensions")
    cin, cout = input_channels // groups, output_channels // groups
    data_bytes = _ceil_div(input_elements_per_sample // groups * physical_activation_bits, 8)
    weight_bytes = _ceil_div(cin * cout * kernel_elements * weight_bits, 8)
    double_weight_bytes = _ceil_div(cin * min(cout, 2 * atomic_k)
                                   * kernel_elements * weight_bits, 8)
    samples = time_steps * batch_size
    if data_bytes + weight_bytes <= capacity_bytes:
        mode, resident, reusable = "full_input_full_weight", weight_bytes, True
        tiles = 1
    else:
        resident = double_weight_bytes
        available = capacity_bytes - resident
        if available <= 0:
            raise ValueError("Double-buffered weights exhaust convolution buffer capacity")
        tiles = _ceil_div(data_bytes, available)
        if tiles > output_positions_per_sample:
            raise ValueError("Capacity estimate requires unsupported input-channel splitting")
        reusable = resident == weight_bytes
        mode = "full_input_partial_weight" if tiles == 1 else "spatial_split_approximate"
    # 均分输出点仅用于估算每块的启动和排空开销，不声称是编译器实际高度划分。
    quotient, remainder = divmod(output_positions_per_sample, tiles)
    positions = tuple(quotient + (i < remainder) for i in range(tiles))
    fetches = 1 if reusable else samples * tiles
    return DlaBufferPlan(
        mode=mode, capacity_bytes=capacity_bytes, resident_weight_bytes=resident,
        full_input_bytes=data_bytes, full_weight_bytes=weight_bytes,
        output_positions_per_tile=positions,
        weight_reusable=reusable, weight_fetches_per_group=fetches,
    )


def estimate_dla_dc_analytic_cycles(
    *,
    output_positions_per_sample: int,
    input_channels: int,
    output_channels: int,
    kernel_elements: int,
    time_steps: int,
    batch_size: int,
    groups: int = 1,
    atomic_c: int = DLA_ATOMIC_C,
    atomic_k: int = DLA_ATOMIC_K,
    stripe_positions: int = DLA_STRIPE_POSITIONS,
    max_stripe_positions: int = DLA_MAX_STRIPE_POSITIONS,
    weight_kernels_per_cycle: int = DLA_WEIGHT_KERNELS_PER_CYCLE,
    accumulator_revisit_cycles: int = DLA_ACCUMULATOR_REVISIT_CYCLES,
    hardware_layer_overhead_cycles: int = DLA_HARDWARE_LAYER_OVERHEAD_CYCLES,
) -> DlaDcComputeEstimate:
    """按 kernel group→空间 stripe→C 分块→R*S→atomic 顺序建模。

    每个 atomic 处理一组 atomic_c 输入通道、至多 atomic_k 输出通道，
    产生一个空间位置的部分和。每个 stripe 在多个位置上复用同一组权重；
    C 分块和 R*S 各引入一个 stripe operation，CACC 累加同一组位置。
    一个 T/B/卷积分组映射一个独立硬件层任务，任务间不重叠计算或权重加载。

    调度递推：下一 stripe 的开始时刻为
    max(当前 stripe 结束，当前开始+下一组权重加载时间，相关累加的最早时刻)。
    RAW 约束仅用于相同结果，权重和 RAW 停顿取 max，不能相加重复计费。
    返回的是 CSC/CMAC 发射时间及显式层开销，不含未校准的固定 CACC
    排空时间、SDP 反压或软件任务切换延迟。调整 Atomic-K 大于16时，
    尤其不能用这里的发射估算代替16元素/拍 CACC 输出的完整时序模型。
    """
    for name, value in {
        "output_positions_per_sample": output_positions_per_sample,
        "input_channels": input_channels, "output_channels": output_channels,
        "kernel_elements": kernel_elements, "time_steps": time_steps,
        "batch_size": batch_size, "groups": groups, "atomic_c": atomic_c,
        "atomic_k": atomic_k, "stripe_positions": stripe_positions,
        "max_stripe_positions": max_stripe_positions,
        "weight_kernels_per_cycle": weight_kernels_per_cycle,
        "accumulator_revisit_cycles": accumulator_revisit_cycles,
    }.items():
        _positive_integer(name, value)
    if (not isinstance(hardware_layer_overhead_cycles, int)
            or isinstance(hardware_layer_overhead_cycles, bool)
            or hardware_layer_overhead_cycles < 0):
        raise ValueError("hardware_layer_overhead_cycles must be a nonnegative integer")
    if input_channels % groups or output_channels % groups:
        raise ValueError("input_channels and output_channels must be divisible by groups")
    if max_stripe_positions < stripe_positions:
        raise ValueError("max_stripe_positions must be >= stripe_positions")

    # 末 stripe 合并避免把例如17个位置拆成16+1，产生不必要的短 stripe。
    # 使用元素数，不把 D_ATOMICS 寄存器的“元素数减1”编码混入周期计算。
    lengths = []
    remaining = output_positions_per_sample
    while remaining > max_stripe_positions:
        lengths.append(stripe_positions)
        remaining -= stripe_positions
    lengths.append(remaining)

    cin_blocks = _ceil_div(input_channels // groups, atomic_c)
    cout = output_channels // groups
    kernel_groups = _ceil_div(cout, atomic_k)
    operations_per_spatial_stripe = cin_blocks * kernel_elements
    atomic_cycles = weight_load_cycles = weight_startup = 0
    weight_stalls = accumulator_stalls = 0
    previous_length = None

    # 同长度 stripe operation 的服务时间可解析合并，无需逐拍或逐 MAC
    # 循环，因此运行开销与空间 stripe 数、输出通道组数有关，而非 MAC 数。
    for kernel_group in range(kernel_groups):
        valid_kernels = min(atomic_k, cout - kernel_group * atomic_k)
        # 尾输出通道组只装载有效 kernel；输入通道尾块仍需逐 kernel 发送，
        # 不因为有效 Cin 少就假定可在一拍装入多个 kernel。
        load_cycles = _ceil_div(valid_kernels, weight_kernels_per_cycle)
        for length in lengths:
            atomic_cycles += length * operations_per_spatial_stripe
            weight_load_cycles += load_cycles * operations_per_spatial_stripe
            if previous_length is None:
                weight_startup = load_cycles
            else:
                # 新空间 stripe 或新 kernel group 的结果独立，没有 RAW；
                # 但影子权重装载仍需与上一 stripe 的计算竞争时间窗口。
                weight_stalls += max(0, load_cycles - previous_length)

            repeated = operations_per_spatial_stripe - 1
            weight_gap = max(0, load_cycles - length)
            # 权重加载已提供的间隔同时满足一部分 RAW 延迟，只计剩余停顿。
            raw_gap = max(0, accumulator_revisit_cycles - max(length, load_cycles))
            weight_stalls += repeated * weight_gap
            accumulator_stalls += repeated * raw_gap
            previous_length = length

    layers = time_steps * batch_size * groups
    overhead = hardware_layer_overhead_cycles * layers
    return DlaDcComputeEstimate(
        compute_cycles=(atomic_cycles + weight_startup + weight_stalls
                        + accumulator_stalls) * layers + overhead,
        atomic_cycles=atomic_cycles * layers,
        weight_load_cycles=weight_load_cycles * layers,
        weight_startup_cycles=weight_startup * layers,
        weight_stall_cycles=weight_stalls * layers,
        accumulator_stall_cycles=accumulator_stalls * layers,
        hardware_layer_overhead_cycles=overhead,
        hardware_layer_count=layers,
        kernel_groups_per_hardware_layer=kernel_groups,
        input_channel_blocks=cin_blocks,
        spatial_stripe_lengths=tuple(lengths),
        stripe_operation_count=(kernel_groups * len(lengths)
                                * operations_per_spatial_stripe * layers),
    )


def estimate_dla_dc_compute_cycles(
    *,
    output_positions_per_sample: int,
    input_channels: int,
    output_channels: int,
    kernel_elements: int,
    time_steps: int,
    batch_size: int,
    groups: int = 1,
    atomic_c: int = DLA_ATOMIC_C,
    atomic_k: int = DLA_ATOMIC_K,
    stripe_positions: int = DLA_STRIPE_POSITIONS,
    max_stripe_positions: int = DLA_MAX_STRIPE_POSITIONS,
    weight_kernels_per_cycle: int = DLA_WEIGHT_KERNELS_PER_CYCLE,
    accumulator_revisit_cycles: int = DLA_ACCUMULATOR_REVISIT_CYCLES,
    hardware_layer_overhead_cycles: int = DLA_HARDWARE_LAYER_OVERHEAD_CYCLES,
    timing: DlaRtlTiming | None = None,
    collect_trace: bool = False,
) -> DlaDcComputeEstimate:
    """按CSC发射计数器及CACC信用/读出握手进行离散事件周期建模。

    使用RTL允许的事件时刻跳转，等价于逐拍递减pop_cnt，避免百万拍空转。
    权重与数据package的顺序关系对应wt_pop_ready/dat_pop_ready；输入
    FIFO在本配置下1包/拍且至少4深，稳态16/32拍发射间隔不致供给不足。
    第一个weight pop的启动延迟单独保留，不能假设首拍直接出现数据包。

    信用在channel_end data pop时按stripe长度一次预留，在对应entry
    完成CACC→SDP握手后才返回。尾输出通道仍占一个完整物理entry。
    不把“RTL源码推导的事件模型”标成“通过RTL波形校准”的模型。
    """
    # 复用旧解析模型的参数检查及原子操作计数，旧模型本身仍可独立调用。
    analytic = estimate_dla_dc_analytic_cycles(
        output_positions_per_sample=output_positions_per_sample,
        input_channels=input_channels, output_channels=output_channels,
        kernel_elements=kernel_elements, time_steps=time_steps, batch_size=batch_size,
        groups=groups, atomic_c=atomic_c, atomic_k=atomic_k,
        stripe_positions=stripe_positions, max_stripe_positions=max_stripe_positions,
        weight_kernels_per_cycle=weight_kernels_per_cycle,
        accumulator_revisit_cycles=accumulator_revisit_cycles,
        hardware_layer_overhead_cycles=hardware_layer_overhead_cycles,
    )
    if (atomic_c != 64 or atomic_k not in (16, 32) or stripe_positions != atomic_k
            or max_stripe_positions != 2 * atomic_k or weight_kernels_per_cycle != 1):
        raise ValueError("Event model requires C64, K16/K32, stripe K/2K and one kernel/cycle")
    if accumulator_revisit_cycles > atomic_k:
        raise ValueError("Minimum stripe window does not cover requested revisit latency")
    if timing is None:
        timing = DlaRtlTiming(delivery_read_issue_interval=1 + atomic_k // 16)
    if not isinstance(timing, DlaRtlTiming):
        raise TypeError("timing must be a DlaRtlTiming")
    for name, value in vars(timing).items():
        _positive_integer(name, value)
    if timing.credit_capacity < max_stripe_positions:
        raise ValueError("credit_capacity cannot hold the maximum stripe")
    if not isinstance(collect_trace, bool):
        raise TypeError("collect_trace must be bool")

    # 对应CSC_sg.cur_stripe：>=2*upper取upper，>upper取lower，尾部全取。
    lengths = []
    remaining = output_positions_per_sample
    while remaining:
        length = (max_stripe_positions if remaining >= 2 * max_stripe_positions
                  else stripe_positions if remaining > max_stripe_positions else remaining)
        lengths.append(length)
        remaining -= length

    reduction_steps = analytic.input_channel_blocks * kernel_elements
    cout = output_channels // groups
    free_credits = timing.credit_capacity
    # 返回事件严格按输出先后排列；同拍返回在对应寄存器已生效后可使用。
    credit_returns = deque()
    weight_pop = timing.sg_first_weight_pop_cycle
    # wt_max_cycles对1个kernel也至少取2；启动期不能直接缩为1拍。
    first_weight_window = max(min(atomic_k, cout), 2)
    data_pop = weight_pop + first_weight_window
    first_data_pop = data_pop
    controller_stalls = credit_stalls = weight_load = 0
    previous_length = None
    last_send = -1
    trace = []
    stripe_count = 0

    for kg in range(analytic.kernel_groups_per_hardware_layer):
        kernels = min(atomic_k, cout - kg * atomic_k)
        for spatial, length in enumerate(lengths):
            for reduction in range(reduction_steps):
                final = reduction == reduction_steps - 1
                if previous_length is not None:
                    # 下一weight package在上一data pop时并行发射。RTL的
                    # dat_max_cycles=max(长度,CSC_MIN_STRIPE)，不是RAW=11。
                    weight_pop = data_pop
                    interval = max(previous_length, atomic_k, kernels, 2)
                    controller_stalls += interval - previous_length
                    data_pop += interval
                if final:
                    while credit_returns and credit_returns[0] <= data_pop:
                        credit_returns.popleft()
                        free_credits += 1
                    credit_wait_start = data_pop
                    while free_credits < length:
                        # 仅跳到下一个信用更新时刻，不能越过一个可发射周期。
                        data_pop = credit_returns[0]
                        while credit_returns and credit_returns[0] <= data_pop:
                            credit_returns.popleft()
                            free_credits += 1
                    credit_stalls += data_pop - credit_wait_start
                    free_credits -= length
                    for point in range(length):
                        # 固定流水延迟对应首个发送拍；信用在完整表项发送结束后返回。
                        # 输出通道并行度为 32 时，末组不足 32 通道仍占完整物理表项。
                        ready_cycle = (data_pop + point + timing.data_pop_to_sdp_cycles
                                       + atomic_k // 16 - 1)
                        # delivery_buffer必须等data_left_mask清零才能发下一读，
                        # 因而不能把一串最终结果简单按1 entry/拍排空。
                        last_send = max(ready_cycle,
                                        last_send + timing.delivery_read_issue_interval)
                        credit_returns.append(last_send + timing.credit_return_cycles)
                weight_load += kernels
                stripe_count += 1
                if collect_trace:
                    first = data_pop + timing.data_pop_to_cmac_cycles
                    trace.append(DlaStripeTrace(
                        kg, spatial, reduction, length, weight_pop, data_pop,
                        first, first + length - 1, final, last_send if final else None,
                    ))
                previous_length = length

    issue_end = data_pop + lengths[-1]
    # 周期0为任务配置接受周期；握手所在周期计入完成区间，故最后+1。
    compute_per_layer = last_send + 1
    delivery_tail = compute_per_layer - issue_end - timing.data_pop_to_sdp_cycles
    layers = analytic.hardware_layer_count
    return DlaDcComputeEstimate(
        compute_cycles=compute_per_layer * layers + analytic.hardware_layer_overhead_cycles,
        atomic_cycles=analytic.atomic_cycles,
        weight_load_cycles=weight_load * layers,
        weight_startup_cycles=first_weight_window * layers,
        weight_stall_cycles=0,  # 权重加载等待已经计入控制发射停顿，不能重复统计。
        accumulator_stall_cycles=0,  # 最短发射窗口覆盖十一拍的累加相关延迟。
        hardware_layer_overhead_cycles=analytic.hardware_layer_overhead_cycles,
        hardware_layer_count=layers,
        kernel_groups_per_hardware_layer=analytic.kernel_groups_per_hardware_layer,
        input_channel_blocks=analytic.input_channel_blocks,
        spatial_stripe_lengths=tuple(lengths),
        stripe_operation_count=stripe_count * layers,
        model_kind="rtl_control_event",
        controller_startup_cycles=timing.sg_first_weight_pop_cycle * layers,
        controller_stall_cycles=controller_stalls * layers,
        credit_stall_cycles=credit_stalls * layers,
        pipeline_cycles=timing.data_pop_to_sdp_cycles * layers,
        delivery_tail_cycles=delivery_tail * layers,
        first_cmac_cycle=first_data_pop + timing.data_pop_to_cmac_cycles,
        last_cmac_cycle=issue_end - 1 + timing.data_pop_to_cmac_cycles,
        last_sdp_cycle=last_send,
        # 独立列出SG done的timer，主计算边界不使用这个保守flush计数。
        # FIFO清空→DONE入口按2拍计，仍需要完整RTL波形校准此控制边界。
        csc_done_cycle=data_pop + lengths[-1] + timing.csc_done_flush_cycles + 2,
        trace=tuple(trace),
    )


@dataclass(frozen=True)
class DlaCycleEstimate:
    compute_cycles: int
    memory_cycles: int
    activation_cycles: int
    input_read_bits: int
    weight_read_bits: int
    result_write_bits: int
    activation_output_write_bits: int
    activation_extra_bits: int
    serial_stage_sum_cycles: int
    ideal_overlap_floor_cycles: int
    compute_schedule: DlaDcComputeEstimate
    buffer_plan: DlaBufferPlan
    compute_tile_schedules: tuple[DlaDcComputeEstimate, ...]

    @property
    def total_cycles(self) -> int:
        """总周期采用已确认的计算、片外访存与泄漏积分发放计算串行和。"""
        return self.serial_stage_sum_cycles


def estimate_dla_layer_cycles(
    *,
    input_elements_per_sample: int,
    output_positions_per_sample: int,
    input_channels: int,
    output_channels: int,
    kernel_elements: int,
    time_steps: int,
    batch_size: int,
    effective_memory_bits_per_cycle: int,
    activation_elements_per_cycle: int,
    groups: int = 1,
    atomic_c: int = DLA_ATOMIC_C,
    atomic_k: int = DLA_ATOMIC_K,
    stripe_positions: int = DLA_STRIPE_POSITIONS,
    max_stripe_positions: int = DLA_MAX_STRIPE_POSITIONS,
    weight_kernels_per_cycle: int = DLA_WEIGHT_KERNELS_PER_CYCLE,
    accumulator_revisit_cycles: int = DLA_ACCUMULATOR_REVISIT_CYCLES,
    hardware_layer_overhead_cycles: int = DLA_HARDWARE_LAYER_OVERHEAD_CYCLES,
    timing: DlaRtlTiming | None = None,
    logical_activation_bits: int = 2,
    physical_activation_bits: int = 8,
    weight_bits: int = 8,
    result_bits: int = 16,
    activation_output_storage_bits: int = 8,
    input_reloads: int = 1,
    weight_reloads: int = 1,
    result_writes: int = 1,
    activation_output_writes: int = 1,
    activation_extra_read_bits_per_element: int = 0,
    activation_extra_write_bits_per_element: int = 0,
    cbuf_capacity_bytes: int = DLA_CBUF_CAPACITY_BYTES,
) -> DlaCycleEstimate:
    """估算一层卷积或 FC 的计算、片外访存和逐元素激活周期。

    ``input_elements_per_sample`` 是一个时间步、一个 batch 的输入元素数；
    ``output_positions_per_sample`` 是对应的输出空间位置数。FC 可将每个
    向量视为一个位置，并设 ``kernel_elements=1``。通道数为整个分组层的
    总通道数，不是每组通道数。

    计算调用 RTL控制事件模型，不再直接将理想原子操作数当成周期；
    ``compute_schedule`` 单独给出原子操作、CSC发射、信用和输出排空开销。
    禁用 multi-batch，首任务权重加载不能被前一 T/B 的计算隐藏。层固定开销
    可通过 ``hardware_layer_overhead_cycles`` 校准，其默认值0不是实测值。

    片外访存以未压缩数据量及卷积缓冲区容量估算；权重无法全部驻留时，
    每个时间步、批次和空间分块分别读取权重，不再无条件假定全层只读一次。
    ``input_reloads`` 和 ``weight_reloads`` 是在容量估算基础上的额外读取倍数，
    默认 1 表示不再追加重复读取，0 表示由调用方排除对应流量。
    ``result_writes``、``activation_output_writes`` 用来表达是否把 16-bit
    计算结果和激活输出分别执行片外访存写入。例如融合后处理时可以将前者设为 0。
    2-bit 激活输出若以 NVDLA 原生特征图供下一层使用，默认按 8 bit 存储。

    LIF 等带状态激活若需额外读取计算结果、读取/写回膜电位，可通过两个
    ``activation_extra_*_bits_per_element`` 参数显式计入。这里不推断
    NVDLA 的 SDP 能直接执行有状态 LIF，也不暗中假定膜电位驻留位置。

    ``effective_memory_bits_per_cycle`` 必须由实际系统带宽给出；本项目
    默认每拍 512 位，对应 1 吉赫兹下每秒 64 吉字节，不等同于物理接口宽度。
    所有流量按共享接口相加，仅是忽略对齐、burst、压缩和读启动延迟的
    带宽下限。串行阶段和与理想重叠瓶颈均不是精确执行周期；真正的调度
    方式确定后，再选择或细化总周期公式。
    """
    positive = {
        "input_elements_per_sample": input_elements_per_sample,
        "output_positions_per_sample": output_positions_per_sample,
        "input_channels": input_channels,
        "output_channels": output_channels,
        "kernel_elements": kernel_elements,
        "time_steps": time_steps,
        "batch_size": batch_size,
        "effective_memory_bits_per_cycle": effective_memory_bits_per_cycle,
        "activation_elements_per_cycle": activation_elements_per_cycle,
        "groups": groups,
        "atomic_c": atomic_c,
        "atomic_k": atomic_k,
        "logical_activation_bits": logical_activation_bits,
        "physical_activation_bits": physical_activation_bits,
        "weight_bits": weight_bits,
        "result_bits": result_bits,
        "activation_output_storage_bits": activation_output_storage_bits,
    }
    nonnegative = {
        "input_reloads": input_reloads,
        "weight_reloads": weight_reloads,
        "result_writes": result_writes,
        "activation_output_writes": activation_output_writes,
        "activation_extra_read_bits_per_element":
            activation_extra_read_bits_per_element,
        "activation_extra_write_bits_per_element":
            activation_extra_write_bits_per_element,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in nonnegative.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if input_channels % groups or output_channels % groups:
        raise ValueError("input_channels and output_channels must divide by groups")
    if logical_activation_bits > physical_activation_bits:
        raise ValueError("logical activation precision exceeds physical precision")

    def ceil_div(numerator: int, denominator: int) -> int:
        return (numerator + denominator - 1) // denominator

    cin_per_group = input_channels // groups
    samples = time_steps * batch_size
    output_elements = output_positions_per_sample * output_channels * samples
    weight_elements = output_channels * cin_per_group * kernel_elements

    buffer_plan = plan_dla_buffer(
        input_elements_per_sample=input_elements_per_sample,
        output_positions_per_sample=output_positions_per_sample,
        input_channels=input_channels, output_channels=output_channels,
        kernel_elements=kernel_elements, time_steps=time_steps, batch_size=batch_size,
        groups=groups, capacity_bytes=cbuf_capacity_bytes,
        physical_activation_bits=physical_activation_bits, weight_bits=weight_bits, atomic_k=atomic_k,
    )
    # 每块独立调用既有计算模型，以计入额外启动、短条带和流水排空。
    compute_tiles = tuple(estimate_dla_dc_compute_cycles(
        output_positions_per_sample=positions,
        input_channels=input_channels, output_channels=output_channels,
        kernel_elements=kernel_elements, time_steps=time_steps,
        batch_size=batch_size, groups=groups, atomic_c=atomic_c, atomic_k=atomic_k,
        stripe_positions=stripe_positions, max_stripe_positions=max_stripe_positions,
        weight_kernels_per_cycle=weight_kernels_per_cycle,
        accumulator_revisit_cycles=accumulator_revisit_cycles,
        hardware_layer_overhead_cycles=hardware_layer_overhead_cycles,
        timing=timing,
    ) for positions in buffer_plan.output_positions_per_tile)
    compute_schedule = compute_tiles[0]
    if len(compute_tiles) > 1:
        sum_fields = (
            "compute_cycles", "atomic_cycles", "weight_load_cycles", "weight_startup_cycles",
            "weight_stall_cycles", "accumulator_stall_cycles", "hardware_layer_overhead_cycles",
            "hardware_layer_count", "stripe_operation_count", "controller_startup_cycles",
            "controller_stall_cycles", "credit_stall_cycles", "pipeline_cycles", "delivery_tail_cycles",
        )
        # 汇总对象不提供虚假的全层事件时刻；逐块时刻保存在 compute_tile_schedules。
        compute_schedule = replace(compute_schedule,
            **{name: sum(getattr(part, name) for part in compute_tiles) for name in sum_fields},
            spatial_stripe_lengths=tuple(length for part in compute_tiles
                                         for length in part.spatial_stripe_lengths),
            model_kind="rtl_control_event_capacity_approximate",
            first_cmac_cycle=0, last_cmac_cycle=0, last_sdp_cycle=0, csc_done_cycle=0)
    compute_cycles = compute_schedule.compute_cycles
    input_read_bits = (
        input_elements_per_sample * samples * physical_activation_bits
        * input_reloads
    )
    weight_read_bits = (weight_elements * weight_bits * weight_reloads
                        * buffer_plan.weight_fetches_per_group)
    result_write_bits = output_elements * result_bits * result_writes
    activation_output_write_bits = (
        output_elements * activation_output_storage_bits
        * activation_output_writes
    )
    activation_extra_bits = output_elements * (
        activation_extra_read_bits_per_element
        + activation_extra_write_bits_per_element
    )
    total_memory_bits = (
        input_read_bits + weight_read_bits + result_write_bits
        + activation_output_write_bits + activation_extra_bits
    )
    memory_cycles = ceil_div(total_memory_bits, effective_memory_bits_per_cycle)
    # 有状态LIF的T维存在依赖：不能把不同T的不足一拍元素合并到同一拍。
    activation_cycles = time_steps * ceil_div(
        output_positions_per_sample * output_channels * batch_size,
        activation_elements_per_cycle,
    )

    return DlaCycleEstimate(
        compute_cycles=compute_cycles,
        memory_cycles=memory_cycles,
        activation_cycles=activation_cycles,
        input_read_bits=input_read_bits,
        weight_read_bits=weight_read_bits,
        result_write_bits=result_write_bits,
        activation_output_write_bits=activation_output_write_bits,
        activation_extra_bits=activation_extra_bits,
        serial_stage_sum_cycles=(compute_cycles + memory_cycles + activation_cycles),
        ideal_overlap_floor_cycles=max(
            compute_cycles, memory_cycles, activation_cycles
        ),
        compute_schedule=compute_schedule,
        buffer_plan=buffer_plan,
        compute_tile_schedules=compute_tiles,
    )


def estimate_dla_runner_layer(operator: str, input_shape: tuple[int, ...],
                              output_channels: int, groups: int = 1
                              ) -> tuple[DlaCycleEstimate, int]:
    """把runner原始算子形状映射到DC feature，而不是映射EVA的im2col形状。

    返回RTL规则模型和原runner公式的历史值。后者故意保留历史公式的
    空间尺寸口径，便于定位公式修正与控制流水开销，不能视为正确的基准。
    尤其conv1d_5_1旧公式遗漏输出长度N，新模型不能沿用该遗漏。
    不展开多值脉冲的T维：DLA本身以INT8输入这些值。
    """
    shape = tuple(int(x) for x in input_shape)
    match = re.fullmatch(
        r"conv2d_\((\d+),\s*(\d+)\)_\((\d+),\s*(\d+)\)_\((\d+),\s*(\d+)\)",
        operator,
    )
    if match:
        kh, kw, sh, sw, ph, pw = map(int, match.groups())
        if len(shape) == 4:
            t, cin, h, w = shape
            b = 1
        elif len(shape) == 5:
            t, b, cin, h, w = shape
        else:
            raise ValueError(f"Unsupported convolution shape: {shape}")
        oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
        input_elements = cin * h * w
        kernel = kh * kw
        # 原runner的3x1分支用h*w；padding=2时没有计入额外两行输出。
        old_positions = h * w if (kh, kw) == (3, 1) else oh * ow
        positions = oh * ow
    else:
        match = re.fullmatch(r"conv1d_(1|3|5)_1", operator)
        if not match or len(shape) != 4:
            raise ValueError(f"Unsupported DLA runner operator/shape: {operator}/{shape}")
        t, b, cin, n = shape
        kernel = int(match.group(1))
        # 与现有runner的算子定义一致：3点核不补零，5点核padding=2。
        padding = 2 if kernel == 5 else 0
        positions = n + 2 * padding - kernel + 1
        input_elements = cin * n
        old_positions = 1 if kernel == 5 else positions
    estimate = estimate_dla_layer_cycles(
        input_elements_per_sample=input_elements,
        output_positions_per_sample=positions, input_channels=cin,
        output_channels=int(output_channels), kernel_elements=kernel,
        time_steps=t, batch_size=b, groups=groups,
        effective_memory_bits_per_cycle=DLA_MEMORY_BITS_PER_CYCLE,
        activation_elements_per_cycle=DLA_ACTIVATION_ELEMENTS_PER_CYCLE,
        activation_extra_read_bits_per_element=DLA_ACTIVATION_EXTRA_READ_BITS,
        activation_extra_write_bits_per_element=DLA_ACTIVATION_EXTRA_WRITE_BITS,
    )
    legacy = (old_positions * kernel * t * b * groups
              * _ceil_div(output_channels // groups, 16)
              * _ceil_div(cin // groups, 64))
    if operator == 'conv2d_(1, 1)_(1, 1)_(0, 0)':
        # 该旧分支没有groups项；保留它的原样数值，而非悄悄修正历史列。
        legacy = (positions * t * b * _ceil_div(output_channels, 16)
                  * _ceil_div(cin, 64))
    return estimate, legacy
