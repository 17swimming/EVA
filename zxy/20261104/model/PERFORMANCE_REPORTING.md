# EVA 周期与卷积数据包报告

## 1104新增的逐拍归因

### 九类逐层周期表

`calls.jsonl`新增`nine_cycle_breakdown`，只增加计数，不改变计算或原六类论文归因。DRAM和LIF使用原未隐藏周期；计算循环内每core每拍按实际PE／Express计算、splitunit等待、无计算分配停顿、显式accumulator反压、纯排空、空闲／调度、其它的优先级互斥计数，再除以core数。最后仅有LIF工作的周期计入LIF，accumulator最终写入尾等待计入其它。九项之和逐调用等于总周期。

该表的纯排空排除了已归到splitunit、分配停顿或显式accumulator反压的周期，不等于原始`pure_pool_drain_core_cycles`诊断。不能直接混加两套独立互斥分类来生成九项。

正式入口`model/test_outerproduct_simulator/run_nine_way_layer_report.py`统一运行4 core、单调度通道、64 LIF、深度2／候选2。TIM、BRGNet、QKFormer使用完整清单；SDTrack和EMSYOLO当前清单按FLOPs排序的前10项通过`--max-test-cases 10`运行。`build_nine_way_layer_csv.py`核对137个有效层的九项周期，`author_nine_way_csv.mjs`使用表格运行库生成带UTF-8 BOM的单张CSV。四位百分比经最大余数法处理，显示值逐行合计100%。

### splitunit供数不足的进一步拆分

FC空队列路径现在允许已结束启动等待的首个package直接参与本拍握手。握手成功时推进一次输入索引，并禁止本拍tick再生产第二个package；下游不ready时，原package正常进入队列，保持输出稳定。保留已有启动等待和结束确认时序。RTL的FC接口本身直接连接fc_row_encoder输出，本次只修正Python队列的可见性，不新增RTL的FC FIFO。

`paper_split_wait_*_core_cycles`之和等于`paper_split_unit_core_cycles`；`other_split_wait_*_core_cycles`之和等于`other_split_unit_core_cycles`。两个范围各自守恒，不相互混加。`performance_summary.json`中`split_wait_breakdown`按CONV、FC和合计导出各子项。

只在当前mode0接口无包时，对输入计算单元前的拍初状态进行只读采样。互斥优先级依次为：启动等待`startup`；输入编码和全部FIFO均结束后的完成确认`finish_confirmation`；mode0行序已越过输入末行但任务尚未结束`mode0_stream_complete`；期望行所属编码器的FIFO满`fifo_backpressure`；后续行已有mode0包或期望行已编码完但尚未推进`row_order_wait`；期望行未派发`row_dispatch`；所属编码器尚未处于期望行`row_handoff`；mode0窗口仍有未知输入位置`window_assembly`；其余当前行编码供给`encoder_supply`；FC输入列表尚待入队`linear_supply`；剩余`unclassified`。

这些是观察到的等待状态归因，多个条件可以同时成立，不能将优先分类当成独立消融实验的因果结论。`*_split_flag_*`另行记录后续行已有包、mode0或Express FIFO非空、所属编码器FIFO满、期望行已编码完成等条件交集；它们可能重叠，不能相加作为时间分解。startup只使用当前倒计时，不猜测已经重叠掉的启动开销；完成确认与编码吞吐严格区分。`row_order_wait`不表示可以安全乱序发射，现有行序仍服务于退休安全条件。

所有细项跟随原有Cout代表组缩放及FC末组重放修正。新增统计不消费package、不推进编码器、不改变FIFO或ready条件。

LIF并行度可通过统一runner的`--lif-units 48`或`--lif-units 64`指定。默认48保持已有周期校准配置，64表示每拍处理4个点、每点16个lane；不仅改变计算数量，也使CONV和FC的实际读取扫描按4个bank分组，仍保留安全地址约束和两拍尾流水。完整12×12输出块无安全等待时，48 LIF需要48次读请求，64 LIF需要36次。此参数独立于`fetch_rows`，记录在manifest.json和calls.jsonl中；本轮没有修改RTL的LIF数量。

当前`simulator.py`同时保留五类后端互斥统计和论文六类归因。五类中的`other_partition_core_cycles`继续保留，新子项为scheduler空闲或派发等待、splitunit供数不足、显式accumulator反压、tile交接、行进度约束、最终accumulator写入流水等待、其余控制等待。子项仅在原分类为其他时计数，子项之和严格等于原其他计数；所有旧计数和时序控制保持原样。

论文六类仍为DRAM、scheduler、split_unit、current_accumulating_sub_unit、accumulator、LIF。每个core每拍单独归因：实际计算优先，其次splitunit无包、psum pool分配停顿、显式退休请求反压、空闲或任务派发，剩余计算、退休、行进度及交接等待归入电流累积单元。独立LIF等待单独归因。分配停顿和排空仍属于current_accumulating_sub_unit，不能将其替换为纯PE或Express计算周期。

CONV供数不足表示存在尚未完成的splitunit任务、取不到mode0 package，且Express拍初没有在执行的package、本拍也没有新接收Express package。FC沿用取不到mode0 package的定义。两者均在实际供给接口采样，不通过FIFO总占用量推断。新设计没有旧版定长退休退避状态；显式accumulator反压依据有可退休候选、但本拍没有任何候选完成入队握手判断，不将拒绝点数转换为等待周期。

论文归因和开发统计有各自的优先级，不能直接用其他的子项替代六类的全部计数。例如退休反压也可能发生在开发统计中的纯排空周期。各自独立校验守恒。

六类中的core-cycles除以core数得到平均归因周期，未隐藏LIF和片外访存使用已有墙钟周期。每次调用的六类之和等于调用总周期。FC输出通道末组的重放计数与原统计采用相同修正，不能只按完整组放大。

日志保留与lookahead提取入口兼容的互斥计数和总周期分解。`calls.jsonl`新增`paper_cycle_breakdown`，`performance_summary.json`新增各通路的`other_breakdown`、`cycles_breakdown`和`cycles_breakdown_ratios`；结束后生成`cycles_breakdown_calls.csv`。五网络批次入口另外生成逐层六类表格和批次汇总，不覆盖既有论文图片。

可复用批次入口为`model/test_outerproduct_simulator/run_1104_workload_batch.py --output-root <独立结果目录>`。默认并行运行五个网络，4 core、EVA_only、8个psum pool bank、bundle buffer深度3、候选3、四个package FIFO各深度4。时间步参数与现有激励一致：TIM 10、QKFormer 16、SpikeBRGNet 5、EMSYOLO 5、SDTrack 4；不截断原始激励。批次`COMPLETE.json`区分结束与通过，要求finished_workloads=5、passed_workloads=5、failures为空；单网络完成标志不代表整个批次结束。

保留现有重叠模型及 `Stats.compute_cycles`、`Stats.total_cycles` 的数值。
`compute_cycles` 是计算与泄漏积分发放计算重叠后的后端墙钟周期，包含调度、退休、写回与反压等待。
它已经包含未隐藏的泄漏积分发放计算，不能再直接加上完整的泄漏积分发放计算周期。

主 runner 的结果表按指定名称依次导出以下九列，并保留已实现的额外报告字段。

| 结果表列名 | 含义 |
| --- | --- |
| `num_mode0` | 代表输出通道组的 mode 0 卷积数据包数量 |
| `ratio_mode0` | mode 0 数据包数量除以三种 mode 的数据包总数 |
| `num_mode1` | 代表输出通道组的 mode 1 卷积数据包数量 |
| `ratio_mode1` | mode 1 数据包数量除以三种 mode 的数据包总数 |
| `num_mode2` | 代表输出通道组的 mode 2 卷积数据包数量 |
| `ratio_mode2` | mode 2 数据包数量除以三种 mode 的数据包总数 |
| `EVA` | 正常计算周期，包含调度、退休、写回与反压等待 |
| `LIF` | 未隐藏的泄漏积分发放计算周期 |
| `DRAM` | 未隐藏的动态随机访问存储器访问周期，包含启动开销 |

`EVA + LIF + DRAM = eva_total_cycles`。比例是 0 到 1 的数值。
`Stats.cycle_breakdown` 保留以下内部字段；前三个周期分量在结果表中分别命名为 `EVA`、`LIF`、`DRAM`，其余字段带 `eva_` 前缀。

| 字段 | 含义 |
| --- | --- |
| `total_cycles` | 正常计算 + 未隐藏泄漏积分发放计算 + 未隐藏动态随机访问存储器访问 |
| `normal_compute_cycles` | 后端墙钟周期减去仅有泄漏积分发放计算工作的周期；包含计算期间的反压等待 |
| `lif_exposed_cycles` | 后端计算、调度与退休写通路空闲后，仍等待遍历result array以完成LIF计算的墙钟周期 |
| `dram_exposed_cycles` | 启动开销 + 超出后端墙钟时间的数据传输周期 |
| `backend_cycles` | 原有 `compute_cycles`，等于正常计算 + 未隐藏泄漏积分发放计算 |
| `lif_work_cycles` | 原始工作周期：有效读请求周期 + 每个非空输出上下文的两拍尾流水 |
| `lif_read_cycles` | 全层有效读请求周期 |
| `lif_hidden_read_cycles` | 与后端计算重叠的有效读请求周期，仅统计读请求 |
| `lif_pipeline_tail_cycles` | 每个非空输出上下文的两拍尾流水之和 |
| `dram_work_cycles` | 原始数据传输周期：原有读写数据量除以带宽后向上取整，不含启动开销 |
| `dram_startup_cycles` | 原有启动开销：首次加载估算 + 两拍全局缓冲到分块缓冲延迟 |

正常计算的口径由当前时间线决定，并非关闭泄漏积分发放计算后重新仿真的周期。
`lif_work_cycles` 是无安全边界等待时顺序执行各输出上下文的工作量；不包含描述符装载与等待。
因此不能把原始工作周期与未隐藏周期之差一概解释成实际隐藏的工作周期。
全连接最后输出通道组的读请求、尾流水和未隐藏周期，均采用已有末组重放时间线修正。

现有总周期公式保持为：

```text
backend_cycles = normal_compute_cycles + lif_exposed_cycles
dram_exposed_cycles = dram_startup_cycles + max(0, dram_work_cycles - backend_cycles)
total_cycles = normal_compute_cycles + lif_exposed_cycles + dram_exposed_cycles
```

没有改变数据量、带宽、首次加载估算及输入输出分块映射。

## 三种 mode 的数据包比例

`shift.py` 已按实际生成的数据包计数：mode 0 每个输出列的数据包计一次，mode 1 和 mode 2 每次成功生成数据包计一次。
计算核心在卷积任务完成后汇总，模型提供 `conv_mode_split_counts` 和 `conv_mode_split_fractions`。
这是数据包比例，不是输入脉冲比例，也不是计算周期比例。

主 runner 使用 `num_mode0`、`num_mode1`、`num_mode2` 及对应的 `ratio_mode0`、`ratio_mode1`、`ratio_mode2`。
三种数据包总数仍保存在额外字段 `eva_conv_packets_representative` 中。
数量保留代表输出通道组口径，涵盖该代表组执行的全部输入通道、时间步、批次和空间分块。
输出通道组复用输入，因此重复输出通道组不影响逐层比例。
全连接不使用这套卷积 mode 分类，三个比例在结果表中留空；没有卷积数据包的层也留空。
各层代表输出通道组的数据包总数可用于加权汇总，不能解释为按全部输出通道组放大后的网络数据包比例。

## runner 使用

主 runner 需要选择当前模型：

```powershell
& 'D:/APPs/Anaconda/python.exe' model/test_outerproduct_simulator/simulator_runner.py --network TIM_NCARS --simulator cluster_parallel --accumulator-variant lookahead --num_cores 6 --num_pus 16 --bank_h 12 --bank_w 12
```

逐层日志只保留各架构总周期及 EVA 的三个周期分量，网络汇总报告总周期和加速比；详细字段保留在结果表中。
`simulator_cycles` 与 `eva_total_cycles` 均使用 `Stats.total_cycles`，所有加速比以 EVA 总周期为分母。
原后端周期仍保存在额外字段 `eva_backend_cycles`，不用于总周期比较。
DLA 使用已确认的串行总周期，即计算 + 片外访存 + 泄漏积分发放计算；Prosperity、GustavSNN 和 PTB 使用现有函数返回的总周期。
其他模型没有这份周期分解时，结果表只导出它们已有的总周期和后端周期，不推断未提供的分解。

## 验证

2026-10-07：当前模型完整回归 121 项通过、1 项跳过。
新增测试覆盖卷积、全连接、多时间步、多批次、全零输入、末输出通道组及动态随机访问存储器瓶颈。
实际数据包计数测试比较生成直至发射结束的数据包，覆盖深度为 1 和 4 的缓冲。
主 runner 使用小型卷积及映射到全连接的步长为 2 的卷积完成日志和结果表验证。
两层总周期为 `233 = 166 + 50 + 17`，分别对应正常计算、未隐藏泄漏积分发放计算和未隐藏动态随机访问存储器访问。
该数据仅用于报告功能验证，不是完整网络性能结果。
