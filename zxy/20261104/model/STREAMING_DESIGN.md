# 1104 Python 模拟器

唯一整网络顶层为 `simulator.py`。`core.py` 导出 `streaming_core.py` 中的新设计；
`simulator_real_compute.py` 仅兼容旧导入路径，不包含另一套实现。
1104 不保留旧同步模式或切换开关；旧设计对照使用 `zxy/20261030`。
本次修改 Python 性能模型，尚未将新交接机制实现并验证到 RTL。

## 统一运行入口

在仓库根目录的 PowerShell 中运行：

```powershell
& D:/APPs/Anaconda/python.exe -B -X utf8 model/test_outerproduct_simulator/simulator_runner.py --network TIM_NCARS --T 10 --simulator cluster_parallel --accumulator-variant 1104 --eva-only --num_cores 6 --num_pus 16 --bank_h 12 --bank_w 12 --split_fifo_depth 8 --psum_pool_rows 8 --prefetch-cycles 2 --output-root output/performance_analysis/tim_1104_bank8_new_run
```

bundle buffer 深度和仲裁候选数通过 `--bundle-buffer-depth`、`--bank-lookahead-entries` 指定，1104 默认均为3。候选数不能超过深度；每个 core 每拍仍最多提交两个 bundle、发射一个 bundle，每个 result_array bank 每拍最多接受一个仲裁发射。九个 bank 使用 `--psum_pool_rows 9`，同时更换输出目录。`--eva-only` 关闭全部其他加速器基线。
每个输出目录保存逐调用 `calls.jsonl`、汇总 `performance_summary.json/txt`、配置与源文件摘要
`manifest.json`、成功标志 `COMPLETE.json`，以及统一运行器生成的日志和逐层结果。
已有逐调用记录时拒绝覆盖。原 lookahead 选择方式保持不变。

## 交接和退休

- 当前 tile 的计算全部写入 psum pool 后，标记剩余 bank 为旧 bank，释放计算任务位置。
- 当前 tile 的写入只匹配当前标志及逻辑位置；旧 bank 只退休。无跨 Cin 合并、无 tile 年龄信息。
- 退休优先级：旧 bank、当前 tile 的 `y < r-2`、满足安全条件的 `y = r-2`；同级按逻辑行号，再按物理 bank 编号。
- 退休请求携带物理 psum pool bank 编号，握手只更新该 bank。accumulator 入队完整复制数据后源 bank 可释放；队列和写入流水仍计入空间tiles收尾。
- 下一 tile 计算不等待旧 psum 排空，仍受真实分配容量及计算资源反压。
- 全部 Cin 计算及所有 psum 提交结束后，cores 可以预计算下一空间tiles。accumulator 完成旧写入、LIF 及必要清理前拒绝新空间tiles请求。
- CONV、FC 共用上述控制；地址分别映射为半行、半列。边界卷积仅写入有效输出点。
- 提前 LIF 使用拍初快照；旧 bank 驻留时禁止依据当前 tile 进度推进，同时检查未退休点的最小地址及 accumulator 待写入数据。
- 核数可配置；本次实验为六个 core。其他参数固定为十六个 lane、12×12 result_array、splitunit FIFO 深度可选四或八（默认八）、两个退休端口、accumulator bundle buffer 默认深度三与候选数三；可用参数设置为二作容量对比。

## 输入窗口与建模边界

`tile_input_buffer.py` 限制两组 ping-pong 窗口，每组最多128个 tile，并按奇偶拆分。
只有输入已进入窗口才可派发；整块输入复制到 core 后释放其读取占用。
窗口切换不触发 psum 或 accumulator 排空。

沿用现有整网络片外访存公式、带宽与权重load开销；窗口准备采用原模型的数据就绪假设，
没有新增逐事务的片外访存完成时序。整网络仍用代表输出通道组及全一权重评估性能，
并非任意真实权重的完整数值推理器；单元检查另用变化权重验证交接。

## 周期统计

所有比例分母为对应通路合计 backend 周期乘以 core 数量，包含未隐藏的 LIF 周期，
不包含未隐藏的片外访存。输出通道缩放及 FC 末组修正同时作用于分子、分母。

- `compute_active_core_cycles`：PE 或 Express 实际计算写入的周期。
- `compute_with_old_bank_resident_core_cycles`：实际计算与未退休旧部分和驻留重叠。
- `compute_with_old_retirement_core_cycles`：实际计算与旧 bank 退休握手同拍发生。
- `allocation_stall_core_cycles`：任一路分配请求失败的并集，可能与另一路计算重叠。
- `allocation_stall_without_compute_core_cycles`：分配失败且两路均没有实际计算推进。
- `pure_pool_drain_core_cycles`：拍初当前计算已结束、仍有未退休点、当拍无实际计算。
  不含空 bank 等待释放；也不把旧部分和退休但下一 tile 正在计算的周期计为纯排空。
- `final_pool_drain_core_cycles`：上述周期中，全部任务已派发且该 core 无预取任务的部分。
- `next_space_backpressure_core_cycles`：新空间tiles已有未退休点，但 accumulator 仍处理旧空间tiles。
- `accumulator_tail_only_core_cycles`：没有计算推进时的 accumulator 最终写入等待。
- `lif_wait_core_cycles`：独立等待 LIF；旧写入尚未完成的等待另列，不重复计入此项。

计算、无计算的分配失败、纯排空、其他、独立 LIF 等待五项互斥分类之和必须等于分母。
其他分析计数为交集或子集，不可直接相加。行号分布按未退休 bank 驻留周期计数；旧 tile 单列，
不使用新 tile 的行进度解释旧 bank。纯排空占比不等于可直接从端到端周期中扣除的比例。

本轮结果集中在 `output/performance_analysis/tim_1104_streaming_20261105`；以其中 REPORT.md 的正式目录索引为准。

## package buffer 的现有实现

当前 `shift.py` 的卷积路径是两个 split 各拥有一个 mode0 FIFO 和一个 Express FIFO，
共四个深度八的队列。两个 split 可同时向各自队列写入；不要求低行编码结束后高行才能写入。
`expected_mode0_row` 和 `expected_express_row` 在 issue 端分别保证本通路行号单调。
只有当前行编码结束、且本通路该行 package 已取完，才推进本通路期望行。
两路期望行可以不同，退休安全前沿仍取两路最小未完成行，并计入 Express 后端队列。
这不同于两个 split 共享两个深度八的 package buffer、按行顺序交接写入权的设计。
本次只改变 accumulator bundle buffer 深度和候选数，不修改上述 package buffer 组织。

论文的六类周期归因定义保持不变。本文五类后端互斥计数及重叠计数仅用于开发分析，
当前开发计数仍不能直接代替论文六类归因所需的细项。
本次容量实验结果位于 `output/performance_analysis/tim_1104_bundle3_candidates3_20261105`。

package buffer 继续保留四队列组织，可用 `--split_fifo_depth 4` 测试减小容量；深度八作为默认对照。面积分析按用户要求推迟到1104 RTL实现后。
