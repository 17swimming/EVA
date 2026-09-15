# Accumulator：共享 Reduce Tree 与逐拍需求统计

更新日期：2026-09-07。本文对应 `simulator_conv_fc_cluster_parallel` 当前实际执行代码。

## 1. 存储地址与树的单位

Accumulator 按寄存器阵列建模：共有 `bank_h × bank_w` 个存储地址，每个地址由
`(row, col)` 标识，硬件数据宽度为 `num_pus × mp_size` bit。

例如 `bank_h=12、bank_w=12` 时，共有 144 个地址。这里“一行使用一棵树”中的行，
指展平后的一个存储字；同一二维 row 的不同 col 是不同地址，不能同拍共用一棵树。
一棵树覆盖该地址全部 PU 的归约数据，模型假设树输入数足够接收所有 Core 的贡献。

原实现没有显式实例化 144 棵树，而是对所有同点请求无条件合并，等效于没有归约资源限制。
当前实现增加共享树池，允许限制同拍可以归约多少个不同地址。

## 2. 参数

`Accumulator` 和 `OutProductSimulator` 均支持以下参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `enable_reduce_tree` | `True` | 是否允许同地址请求合并 |
| `num_reduce_trees` | `None` | 同拍可用的共享树数；None 保持原无限制行为 |
| `record_tree_trace` | `False` | 是否保存逐拍需求/使用轨迹；直方图始终统计 |

`num_reduce_trees=0` 时，冲突地址的全部请求被拒绝；正整数 N 表示同拍最多为 N 个不同的
冲突地址分配树。`enable_reduce_tree=False` 优先于树数参数，行为相当于 0 棵树。
按当前硬件规则，0 棵树遇到重复地址可能一直等待；它不再代表“每地址串行接收一个请求”。
Conv 整包接收还要求树预算能覆盖至少一个待处理 bundle 的全部冲突地址，否则也可能无法前进。

通用 runner 已增加 `--num_reduce_trees N`，目前仅支持 `--simulator cluster_parallel`。
省略该参数保持无限制。更改树数时使用不同的 `--output-root` 保存结果。

```powershell
python -B -X utf8 model/test_outerproduct_simulator/simulator_runner.py --network TIM_NCARS --simulator cluster_parallel --num_cores 6 --num_pus 16 --bank_h 12 --bank_w 12 --split_fifo_depth 4 --psum_pool_rows 6 --retire_column 3 --num_split 2 --enabled_modes 0 --T 10 --B 1 --num_reduce_trees 8 --output-root output/tim_trees8
```

## 3. 每拍仲裁

每拍开始清空上一拍的地址访问表和树分配状态。先收集所有 Core 的本拍请求，
再统一调用 `arbitrate()`。请求提交时不写 accumulator，也不提前清空源 Psum。
Conv 在收齐所有后台写回请求后仲裁，再进行计算；FC 将后台重试与本拍计算产生的退休
请求一起收集，在所有 Core 计算后统一仲裁。

对于一个地址：

1. 本拍只有一个请求：不需要树，可以直接写入（仍受 Conv 整包接收约束）。
2. 本拍有多个请求：该地址需要一棵树，所有接收的贡献一起归约。
3. 没分到树：该地址本拍所有请求都等待，包括最先提交到 Python 的请求。
4. 请求被连带拒绝后，不重新将原本冲突的地址解释为单请求旁路地址。
5. 下一拍重新收集和分配，上一拍地址不继续占用树。

例如，6 个 Core 同拍写同一个地址，只需要 1 棵树；同拍 3 个不同地址分别有多个 Core
请求，则需要 3 棵树。即使树全部占用，无冲突地址仍可接收单个写请求。

模型采用动态共享分配，没有将树固定绑定到地址，也没有新增共享选择网络延迟。
因此结果评估的是归约资源数量对现有周期模型的影响，不包含共享网络的面积和时序开销。

树分配采用固定的 core_id、地址顺序优先级，与 Python 提交顺序无关。
Conv 按 bundle 预留全部所需树，预算不足时不预留；FC 按地址依次分配。
所有接收决定完成后，统一通过回调更新 Pool 和 Core 状态。

## 4. 树不足时的数据与反压

### Conv：整包接收

`request_bundle()` 对整个 bundle 先进行资源判断。如果其中有点缺树，整个 bundle 拒绝，
不会写入部分数据，也不会为失败的 bundle 预占树。Psum Pool 保留数据，后续周期重试。

### FC：部分接收

`request_bundle_partial()` 提交请求，仲裁后通过 `on_result` 回调返回逐点接收掩码：

- 已接收点清空对应 Psum 列。
- 未接收点的数据复制到 `pending_fc_bundles`，由后台队列继续重试。
- Core 再次检查当前退休操作时，不重复提交已经清空、且数据已转入 pending 的源列。

树不足首先表现为退休请求等待。它可以被其他计算或后台写回隐藏；只有影响 Psum buffer
分配、Core 后续 issue 或最终写回完成时，才会影响计算循环的长度。
因此“缺树周期数”不等于“总周期增加数”，两者需要分别观察。

## 5. 每拍到底需要多少棵树

设本拍对地址 a 提交的请求数为 `requests[a]`，则：

```text
needed_trees = 地址集合中 requests[a] >= 2 的地址数
used_trees   = 本拍实际分配到树的地址数
```

`needed_trees` 基于本拍全部提交请求，包含被拒绝的请求，不受树数量上限截断。
一个地址被 6 个 Core 同拍请求，需求仍只计 1。零需求、前端停顿、空闲和 Psum 写回排空周期
也会进入统计，避免只统计有冲突周期而高估需求频率。

`end_cycle()` 在本拍 `arbitrate()` 完成后记录一次；主模拟器在每个计算循环末尾显式调用，
保证最后一拍不丢失。该方法可重复调用而不会重复计数。

需要区分两种分布：

- **无限制基线需求**：没有树不足引起的重试，适合观察原始工作负载需要多少棵树。
- **限流运行需求**：包含重试和调度变化，是限制后的实际请求分布。不能将它作为未限流需求的替代。

逐拍统计覆盖实际执行的计算循环，不包含顶层用公式估算的额外访存/LIF 周期。

## 6. 统计字段与缩放

### Accumulator 内部

| 字段 | 含义 |
|---|---|
| `tree_demand_hist[n]` | 本拍需要 n 棵树的周期数 |
| `tree_used_hist[n]` | 本拍实际使用 n 棵树的周期数 |
| `stats['tree_sampled_cycles']` | 已采样的计算循环周期数 |
| `stats['tree_demand_peak']` | 每拍需求峰值 |
| `stats['tree_used_peak']` | 每拍实际使用峰值 |
| `stats['tree_exhausted_cycles']` | 本拍出现至少一个不能合并的点的周期数；每拍最多加 1 |
| `stats['tree_rejected_points']` | 不能合并而拒绝的点请求数；重试后再次拒绝会再次计数 |
| `tree_trace` | 可选逐拍 `(cycle, needed, used, rejected)` 轨迹 |

Conv 整包中被连带拒绝的非冲突点不计入 `tree_rejected_points`。
关闭 reduce tree 时仍记录需求和拒绝，实际使用树数为 0。

### 顶层输出

| 字段 | 口径 |
|---|---|
| `global_stats.reduce_tree_stats_representative` | 原始代表性 Cout-group 的 accumulator 统计 |
| `global_stats.reduce_tree_stats` | 累计计数按 Cout 倍率放大，峰值不放大 |
| `global_stats.reduce_tree_demand_hist` | 按 Cout 倍率放大的需求分布 |
| `global_stats.reduce_tree_used_hist` | 按 Cout 倍率放大的使用分布 |
| `global_stats.reduce_tree_cout_scale` | 本层 Cout 缩放倍率 |
| `simulator.reduce_tree_trace` | 原始代表性逐拍轨迹，不复制 Cout 的等价轨迹 |

校验关系：

```text
sum(reduce_tree_demand_hist.values()) == compute_cycles
sum(reduce_tree_used_hist.values())   == compute_cycles
len(reduce_tree_trace)               == representative_compute_cycles  # 开启轨迹时
```

## 7. TIM_NCARS 扫描脚本

脚本：`sweep_reduce_trees.py`。

用途：在相同的 TIM_NCARS 保存输入、相同调度与硬件配置下，扫描树数并输出逐层性能和逐拍需求。
它使用通用 runner 对 TIM 的实际映射：3x3 Conv 走卷积路径，conv1d_1_1 和 conv1d_5_1 走 FC 路径。

固定配置：6 cores、16 PUs、12x12 accumulator、6 行 Psum Pool、retire_column=3、FIFO=4、两个 Split。
默认 `enabled_modes=(0,)`，可以通过脚本的 `--enabled_modes` 修改；同一次扫描各档保持一致。

```powershell
python -B -X utf8 -m model.outer_product.simulator_conv_fc_cluster_parallel.sweep_reduce_trees --output output/reduce_tree_tim --trees unlimited,144,64,32,16,12,8,6,4,2,1,0 --workers 3
```

`--workers` 表示并发运行的独立仿真进程数，不改变模拟硬件的 Core 数量。

输出内容：

- `manifest.json`：配置、输入处理口径和代码哈希。
- `summary.csv`：每档树数的整体、Conv、FC 周期与下降比例。
- `baseline_demand.csv`：基线树需求直方图、累计覆盖率和超出该树数的周期数。
- `<树数>/layers.csv`：逐层周期、需求峰值、使用峰值和拒绝统计。
- `<树数>/histograms.csv`：逐层需求/使用分布。
- `<树数>/case_XX.trace.csv.gz`：逐拍原始轨迹，列为 `representative_cycle,needed_trees,used_trees,rejected_points`。
- `<树数>/case_XX.log`：该层模拟器日志。
- `<树数>/skipped.csv`：未进入模拟的算子及原因。
- `report.md`：简要汇总报告。

重复 CSV 条目如果文件、算子、Cout 相同，复用一次仿真结果，汇总时仍按条目分别计入。
`reused_case` 记录复用来源，轨迹文件对应来源 case。

## 8. 2026-09-07 TIM 历史结果（旧顺序仲裁）

本节保留修正前的数据供追溯：旧实现允许缺树地址的首个请求写入，其余请求才等待。
它与当前“未分到树则该地址全部请求等待”的规则不同，以下树数建议已不适用于当前模型。
修正后的重测结果见第 10 节。

### 范围与基线

本次沿用上述配置及完整保存张量，主干输入为 T=10、B=1；TIM interactor 保存数组的首维为 16，
没有额外截断，和通用 runner 的实际读入方式一致。

共统计 34 个 CSV 条目：4 个 Conv 条目、30 个 FC 路径条目。
SSA 和浮点输入算子沿用通用 runner 当前排除范围，因此这里是被模拟条目的周期和，
不代表包含全部算子的完整网络运行时间。

无限制和 144 棵树的每层 `compute_cycles`、`total_cycles` 均一致：

```text
compute_cycles 合计 = 280,016
total_cycles 合计   = 313,084
```

本次保留当前顶层的访存、LIF 和排空计费，只改变树数量及对应仲裁。

### 性能对比

周期增加比例为 `cycles_N / cycles_144 - 1`；吞吐下降比例为 `1 - cycles_144 / cycles_N`。

| 树数 | Total cycles 合计 | 周期增加 | 吞吐下降 |
|---:|---:|---:|---:|
| 无限制 / 144 / 64 / 32 / 16 / 12 | 313,084 | 0% | 0% |
| 8 | 313,084 | 0% | 0% |
| 6 | 313,132 | 0.01533% | 0.01533% |
| 4 | 313,710 | 0.19995% | 0.19955% |
| 2 | 318,070 | 1.59254% | 1.56758% |
| 1 | 318,838 | 1.83785% | 1.80468% |
| 0 | 320,382 | 2.33100% | 2.27791% |

6 棵树的额外 48 cycles 全部来自 Conv；FC 路径在 6 棵树下仍与基线一致。
8 棵树出现 8 个按 Cout 缩放后的缺树周期，但各层最终完成时间均未增加。
12 棵树没有缺树拒绝。

### 基线每拍需求分布

分母为 280,016 个按 Cout 缩放后的计算循环周期，包含零需求周期。

| 本拍需要的树数 | 周期数 |
|---:|---:|
| 0 | 203,950 |
| 1 | 25,856 |
| 2 | 5,560 |
| 3 | 38,186 |
| 4 | 1,184 |
| 5 | 412 |
| 6 | 4,524 |
| 7 | 272 |
| 8 | 64 |
| 12 | 8 |

需求峰值为 **12 棵**；本次未出现 9、10、11 棵的需求。

| 配置树数 | 基线需求不超过此数量的周期比例 |
|---:|---:|
| 4 | 98.11439% |
| 6 | 99.87715% |
| 8 | 99.99714% |
| 12 | 100% |

### 如何选数量

- 希望本次工作负载完全不因缺树而拒绝请求：选择 **12 棵**。
- 希望在已测档位中尽量减少树数，同时保持层总周期不变：选择 **8 棵**。
- 可以接受约 0.20% 的周期增加：**4 棵**是更小的候选。

8 棵是本次已测档位中的最小零周期损失配置，没有测试 5、7 等所有整数档位，
因此不能把它表述为严格的最小需求。其他网络、Core 数量、FIFO、Psum 行数和输入分布改变后需要重新评估。

### 结果位置

主结果目录：`output/reduce_tree_tim_20260907/`。
6、12 棵树的原始层日志和轨迹位于 `output/reduce_tree_tim_20260907_supplement/`；
主目录的 `summary.csv`、`report.md` 已合并补测结果。
主目录额外提供 `layer_comparison.csv`，包含相对 144 棵基线的逐层周期差值和原始数据目录。

## 9. 已完成验证与模型边界

已验证同地址共用树、无冲突写入绕过树、Conv 原子拒绝不泄露资源、FC 部分接收、
零需求及最后一拍统计，以及非零权重下 FC 多 K 分块重试的数据守恒。
旧顺序仲裁的无限制与 144 棵基线在 TIM 全部被模拟条目上一致，408 个历史条目/配置组合
的分布总数均与层计算周期一致。当前同步仲裁重新验证了缺树时全部请求等待、单请求旁路、
提交顺序不改变接收结果，以及有反压时 Conv/FC 非零权重的数值守恒。

该模拟器主要评估周期和反压。Accumulator 保存写入记录并合并同拍数据，没有维护完整输出张量；
PU0 代表其他 PU 的行为。共享树选择网络的实际延迟和物理可实现频率需要后续硬件验证。

## 10. 同拍统一仲裁后的 TIM 重测

以下结果使用第 3 节的同步规则，保留原 TIM 配置：6 cores、16 PUs、12×12 accumulator、
FIFO=4、Psum rows=6、retire_column=3、num_split=2、enabled_modes=0。
使用已有扫描脚本运行 144/4/6/8 棵，共 34 个被测条目；整数输入与历史 TIM 测试一致。
这里没有执行 Prosperity/GustavSNN。

```powershell
python -B -u -X utf8 -m model.outer_product.simulator_conv_fc_cluster_parallel.sweep_reduce_trees --output output/reduce_tree_tim_simultaneous_20260907 --trees 144,4,6,8 --workers 3 --enabled_modes 0
```

| 树数 | compute_cycles | total_cycles | 总周期增加 | 总周期增加比例 | 缺树周期 |
|---:|---:|---:|---:|---:|---:|
| 144（重测基线） | 280,016 | 313,084 | 0 | 0% | 0 |
| 4 | 281,544 | 314,612 | 1,528 | 0.488048% | 6,090 |
| 6 | 280,052 | 313,120 | 36 | 0.011499% | 320 |
| 8 | 280,016 | 313,084 | 0 | 0% | 8 |

144 棵重测的逐层 compute_cycles 和 total_cycles 均与旧 144 棵基线一致。
4 棵增加的 1,528 周期中，Conv 占 616，FC 占 912；6 棵增加的 36 周期全部来自 Conv。
8 棵仍出现 8 个缺树周期、64 个被拒绝点请求，但没有增加最终周期。

只看 compute_cycles 时，4/6/8 棵相对基线分别增加 0.545683%、0.012856%、0%。
不要混用 compute_cycles 与包含额外访存/LIF 估算的 total_cycles。

重测基线逐拍需求峰值为 12 棵，按计算周期统计：

| 可用树数 | 覆盖的基线周期比例 | 需求超过该数量的基线周期 |
|---:|---:|---:|
| 4 | 98.114393% | 5,280 |
| 6 | 99.877150% | 344 |
| 8 | 99.997143% | 8 |

在本次 TIM 配置下，8 棵可保持基线周期；6 棵以很小的周期代价减少两棵树。
QKFormer/SpikeBRGNet 现有 4/6/8 棵结果使用旧顺序仲裁，尚不能用来确认其他网络也有相同结论。

结果目录：`output/reduce_tree_tim_simultaneous_20260907/`。
`summary.csv`、`baseline_demand.csv` 保存总体性能与基线需求；各树数子目录的 `layers.csv`、
`histograms.csv` 和压缩逐拍 trace 保存逐层证据。逐拍 trace 是代表性 Cout 组，直方图按 Cout 缩放。
