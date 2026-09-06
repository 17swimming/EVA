# simulator_conv_fc_cluster

代码风格尽量保持 plus 版本的直观写法：模块少、路径清楚、注释简洁且使用英文，避免过度拆函数带来的阅读压力。

## 设计目标

1. 同时支持 `mode0`、`mode1`、`mode2`。
2. 实现 stream retire：根据 PE0/PE1/PE2 的计算顺序，某个 psum pool 行在被 PE2 最后一次使用后，可以把安全前缀列 retire 到后级 accumulator。
3. `retire_column` 当前默认 `3`，可配置。conv 中表示每拍每个 psum pool 的一行最多 retire 3 列；FC 中表示每拍把同一个 column 的 3 行打包 retire。
4. 所有 psum pool 都可以在同一拍提出 retire 请求，行为一致；cluster accumulator 被建模为 `bank_h * bank_w` 点级寄存器阵列，同一拍可访问任意 `(row, col)`。
5. 把 `num_core_per_cluster` 个 core 加一个共享 accumulator 称为一个 cluster。当前 Python 参数中：
   - `num_cores` = `num_core_per_cluster`
   - `num_pus` = `num_kernel_per_core`
   - `num_clusters` = 逻辑 cluster 数量，用于 Cout tile 并行度和代表性缩放
6. 多个 cluster 之间只有 kernel/Cout tile 不同，输入 activation 和 scheduler 决策相同。因此当前 Python 只实例化 1 个代表性 cluster，真实推演这一份行为；`num_clusters` 只用于计算一个 wave 覆盖多少 Cout tile，以及后续总 cycles/数据搬运的缩放。
7. accumulator 对应之前 Python 模型中说的 MP bank/GlobalSharedBank 后端，现在统一命名为 accumulator。
8. 本 README 记录最新设计；Python 代码已按这里描述的代表性 cluster、stream retire、点级 accumulator 和 reduce-tree 语义实现。

## 架构

```text
OutProductSimulator
  +-- representative cluster[0]
  |     +-- core[0..num_cores-1]
  |     |     +-- SplitUnit
  |     |     +-- PE W0/W1/W2
  |     |     +-- SharedPsumPool
  |     +-- Accumulator
```

调度方式：

```text
for each representative input/Cout wave:
  scheduler watches representative_cluster.core[i].is_finished
  when core i is free:
    pop one Cin task
    send it to representative_cluster.core[i]
  scale total cycles/data by logical Cout waves derived from num_clusters
```

因此 cluster 数量增加时，硬件语义上同一个 activation tile 会被复用到多个 kernel tile；Python 中不重复实例化这些等价 cluster，只保留一个代表性 cluster 的周期行为。一个 cluster 内仍然由 `num_cores` 个 core 并行消化 Cin 任务。

## Stream Retire

### Conv

3x3 conv 中同一个 output row 会被三个 PE 在不同输入行上更新：

```text
W0 updates output row r
W1 updates output row r - 1
W2 updates output row r - 2
```

对某个 output row 来说，PE2 是最后一次使用。因此在 PE2 每拍消耗指令列 `c` 后，psum pool 中该 row 的安全前缀列可以 retire。

设计中使用 `retire_limit` 和 `retire_cursor` 记录每个 psum buffer 行的可 retire 前缀：

- `retire_limit`：当前已经确认安全、允许 retire 的右边界，左闭右开。
- `retire_cursor`：该行下一次从哪里继续扫描 retire。
- `modified`：该列是否真的被写过 psum。没有被修改的列不会产生 accumulator 写请求。

当 PE2 消耗一条指令后，core 会根据 mode 计算可推进到的边界：

- `mode0`：本拍普通滑窗输出列为 `c`，可推进到 `c + 1`。
- `mode1`：孤立点会影响 `[c - 2, c]`，可推进到 `c + 1`。
- `mode2`：连续 run 长度为 `L`，影响 `[c - 2, c + L - 1]`，可推进到 `c + L`。

这里的设计使用左闭右开边界，因此 `request_stream_retire(..., end_exclusive)` 表示 `[0, end_exclusive)` 已经安全。

### FC

FC 路径也采用 stream retire，不再使用整行 retire。

FC 每拍只修改 psum pool 中的一个 column。由于 `retire_column=3`，FC 的 retire 策略改为把同一个 column 上的 3 行结果打包成一个 row bundle，同时写回 accumulator：

```text
FC retire bundle:
  rows = [base_r, base_r + 1, base_r + 2]
  cols = [c, c, c]
```

因此 `retire_column=3` 在 FC 中不是“同一行 retire 3 个 column”，而是“同一个 column 打包 3 个 row”。这和 conv 共享同一个 accumulator 仲裁思想，只是 bundle 的形状不同。

### Core 发起 accumulator 写回的条件

core 不会在 PE 计算后直接把结果写到 accumulator。PE 每拍只把部分和累加到本 core 的 `SharedPsumPool`；只有 psum pool 中某些结果已经满足 retire 条件时，core 才通过 psum pool 向 accumulator 发起写回请求。

全局时序上，每拍先推进 `SplitUnit.tick()`，再执行 `SharedPsumPool.try_flush_to_accumulator()`，然后执行 `Core.tick_compute()` / `Core.tick_compute_linear()`，最后 `Accumulator.tick()` 清空本拍访问记录。因此 conv 中 PE2 本拍刚推进的 stream retire 边界，最早在下一拍的 flush 阶段写 accumulator；FC 中一列计算完成后会在同一个 compute tick 里尝试提交 3-row bundle。

conv 路径有两类 accumulator 写回请求：

- `conv_stream`：当 PE2 消耗一条指令后，core 根据该指令的 `mode` 和列 `c` 计算 `end_exclusive`，调用 `request_stream_retire(buf, end_exclusive)` 推进该 psum row 的 `retire_limit`。下一拍 `try_flush_to_accumulator()` 会从 `retire_cursor` 扫到 `retire_limit`，只选择真正被 `modified` 标记过的列，最多取 `retire_column` 列，形成 `rows=[target_r]`、`cols=[c0,c1,c2]` 的 bundle 写 accumulator。
- `conv_row`：当某个 psum buffer 的 `target_r` 不再属于当前 conv 垂直窗口的 active rows 时，`retire_old(active_target_rs)` 将其置为 `PENDING_WB`，并把 `retire_limit` 推到整行宽度 `bank_w`。后续 flush 阶段会继续按每拍最多 `retire_column` 个 modified column 写回，直到该 buffer 没有未写回列后释放。

FC 路径的 accumulator 写回请求由一条 linear 指令完成后触发：core 同拍更新 `r`、`r-1`、`r-2` 三个 psum row 的同一列 `c`，随后形成 `rows=[r-2,r-1,r]`、`cols=[c,c,c]` 的 bundle，并通过 `try_retire_fc_bundle()` / `request_bundle_partial()` 写 accumulator。默认点级寄存器阵列和 reduce tree 开启时，该 bundle 应被立即接收；如果 ablation 关闭 reduce tree 且出现同点冲突，则已接收点先清空，未接收点进入 pending/retry 路径。

没有被 `modified` 标记的 psum 列不会产生 accumulator 写请求。accumulator 写回被挂起本身只表示 retire 后台没完成；只有后续 PE 申请新的 psum buffer 时发现没有可用 buffer，才计入真正的计算 stall。

### 带宽

`retire_column = 3` 的含义按路径解释。

对 conv 来说，它表示每个 psum_pool buffer row 每拍最多 retire 3 个 column：

```text
conv bundle:
  rows = [target_r]
  cols = [c0, c1, c2]   # 最多 retire_column 个 column
```

对 FC 来说，它表示每拍把 3 个 row 的同一个 column 打包写回：

```text
FC bundle:
  rows = [r0, r1, r2]   # 最多 retire_column 个 row
  cols = [c, c, c]
```

不是整个 cluster 总共 3 列，也不是 accumulator 全局只有 3 个写入 lane。所有 psum pool 都可以同拍提出 retire 请求，最终由 accumulator 的点级寄存器阵列模型接收。硬件上如果每个元素数据宽度是 `num_pus * psum_width`，那么 conv 的一行三列写回宽度为：

```text
retire_column * num_pus * psum_width
```

例如 `retire_column=3`、`num_pus=4`、`psum_width=16bit`，则 conv 一行每拍 `3 * 4 * 16bit = 192bit`。FC 则是 3 个 row 各写同一个 column，总数据量相同，但地址维度从 column lane 变成 row lane。

## Accumulator 寄存器阵列模型

cluster 版本的 accumulator 不再使用 `num_sub_banks` 做 row-bank 仲裁，而是按硬件中的寄存器阵列建模：

```text
bank_h * bank_w 个 accumulator point
每个 point = (row, col)
同一拍任意 point 都可以被访问
```

同一拍内，多个 core 写同一 row 的不同 column 时都可以接收。例如 core0 写 row `[0, 1, 2]` 的 column 4，core1 写 row `[0, 1, 2]` 的 column 5，这两个 FC bundle 不冲突，都应在本拍被接受。

### 点级接收规则

每个 core 发出的 retire 请求仍然是一个 bundle：

```text
bundle = {
  core_id,
  rows,
  cols,
  data
}
```

Accumulator 会把 bundle 展开成若干个点级写：

```text
(row, col, data)
```

仲裁不再检查 row set 是否重叠，而是检查完全相同的 `(row, col)`：

- 如果本拍还没有访问过该 `(row, col)`，直接接收，形成一个 accumulator 写。
- 如果本拍已有其他 core 请求同一个 `(row, col)`，则在 accumulator 前通过 reduce tree 合并数据，本拍仍然接收，不产生 psum pool 写回拒绝。
- 同 row 不同 col 永远不是冲突。

为了做 ablation，Python 模型中保留 `enable_reduce_tree` 开关。默认值为 `True`，对应上面的硬件设计；关闭时，同一拍访问同一个 `(row, col)` 的后续请求会被拒绝，FC 路径仍通过 `request_bundle_partial()` 部分接收非冲突点，并把冲突点留到后续 retry。

因此，旧设计中的 `num_sub_banks=1`、`num_sub_banks=oh` 只适用于 SRAM/row-bank 建模，不再是 cluster simulator 的 accumulator 参数。runner 里 `--num_sub_banks` 仍保留给 pro/旧版本和旧命令兼容；cluster 版本内部会忽略它。

### 统计含义

- `processed_bundles`：本拍被 accumulator 接收的逻辑 bundle 数。
- `processed_writes` / `column_writes`：实际进入 accumulator 的点级物理写数量。被 reduce tree 合并的同点请求不会增加物理写数量。
- `reduced_requests` / `reduced_writes`：有多少请求、多少点级写被 reduce tree 合并。
- `*_request_overlap_*`：仍统计 row overlap，用来观察多个 core 的访问形状。
- `*_conflict_overlap_*`：在点级寄存器阵列模型下通常应为 0，因为 row overlap 不再导致拒绝。


## 文件说明

### `Accumulator.py`

点级寄存器阵列 accumulator 模型。

- `Accumulator.request_bundle(core_id, rows, cols, data)`：接收一个 retired psum bundle 写请求，展开成 `(row, col)` 点级写并立即接收。
- `Accumulator.request_bundle_partial(core_id, rows, cols, data)`：保留 FC pending 兼容接口；在点级寄存器阵列模型下正常返回全 True。
- `Accumulator.request_write(core_id, target_r, col, data)`：可以保留为单 row/single column bundle 的兼容包装。
- `retire_column`：定义 conv 的同 row 多 column bundle 宽度，也定义 FC 的同 column 多 row bundle 宽度。
- `enable_reduce_tree`：默认开启，同点写回在 accumulator 前合并；关闭时用于 ablation，同点写回产生 point conflict，`request_bundle_partial()` 只接收不冲突的点。
- `tick()`：推进一个周期，清除本拍点级访问表和 row-overlap 诊断状态。
- `stats`：记录写入数量、row overlap 形状、同点 reduce 数量等；cluster 模型中 `num_sub_banks` 不参与仲裁。

该模型主要用于 cycle/backpressure 统计，没有把数据真正累加成完整 output tensor。

### `SharedPsumPool.py`

core 内共享 psum pool 模型。默认每个 core 有 6 个全相联 buffer 行；`Core(psum_pool_rows=...)` 可以改成 3 行等其他配置，用于评估面积/性能权衡。

每个 buffer 维护：

- `state`：`FREE`、`ACTIVE`、`PENDING_WB`
- `target_r`：当前 buffer 对应的 output row
- `data`：该 row 的 psum 列数据
- `modified`：每列是否写过
- `retire_cursor`：stream retire 扫描指针
- `retire_limit`：stream retire 当前安全边界

主要函数：

- `get_or_allocate(target_r)`：命中 active row 或分配 free buffer。
- `retire_old(active_target_rs)`：conv 中兜底处理不再 active 的 row；FC 目标设计中不再依赖整行 retire。
- `request_stream_retire(buf, end_exclusive)`：PE2 最后使用后推进 conv 安全 retire 边界。
- `try_flush_to_accumulator()`：conv 每拍对每个非空 buffer 行最多发 `retire_column` 个 column；FC 每拍把同一 column 的 3 个 row 打成一个 bundle。
- `try_retire_fc_bundle(row_buffers, col)`：FC 一条 linear 指令完成后立即尝试写回同一 column 的 3 个 row；默认应被 accumulator 点级接收，若被部分拒绝则把剩余点保留到 pending retry。
- `has_pending()`：判断是否还有 stream retire 或整行 retire 没完成。

### `core.py`

单个 core 的计算模型，包含 split 前端、PE 和 psum pool。

功能：

- `configure_conv_weights(cin, kernel)`：给 conv 路径配置当前 Cin 的 kernel。
- `configure_fc_weights(weights)`：给 FC 路径配置 3-Cin 输入对应的权重。
- `init_for_new_tile(accumulator, core_id, if_map, H)`：初始化 conv tile。
- `tick_compute()`：conv 每拍 issue 一条 split 指令，驱动 W0/W1/W2，并在 PE2 后触发 stream retire。
- `init_for_linear_tile(accumulator, core_id, if_map, valid_pu)`：初始化 FC tile。
- `tick_compute_linear()`：FC 每拍 issue 一条线性指令，并按 3-row bundle 做 stream retire。

mode 支持：

- `mode0`：普通滑窗 MAC。
- `mode1`：孤立非零点展开到受影响输出列。
- `mode2`：连续全 1 run 展开到受影响输出列；非 0/1 spike 继续由 mode0/mode1 携带真实数值。

stream retire 相关：

- `_mark_modified_columns(...)` 标记某条指令实际影响的 psum 列。
- `_record_pe2_retire_columns(...)` 在 PE2 后推进 conv retire 边界，并记录统计。
- FC 路径会在一列结果完成后形成 3-row retire bundle，交给 accumulator 做点级写入；同 row 不同 col 可同拍接收，同点写入由 reduce tree 合并。
- conv 中 core 申请 accumulator 写回的直接触发点不是 PE0/PE1，而是 PE2 推进安全前缀后由 psum pool 的下一拍 flush 发出；FC 中 direct trigger 是 `tick_compute_linear()` 完成一列计算后的 `_try_commit_linear_retire()`。

### `PE.py`

单个 PE 的数值更新逻辑。

- `set_weights(weights)`：设置 3 个横向 kernel 权重。
- `process_v2(mask, c, target_r, psum_buffer_ref, mode)`：根据 mode 把贡献累加进 psum buffer。
- `_accumulate_run(...)`：mode1/mode2 共用的 run 展开逻辑。

`target_r` 在当前实现里只保留接口兼容，PE 内部不依赖它。

### `shift.py`

split 前端和稀疏行编码。这个文件描述的是硬件前端的周期级行为，不只是离线编码函数。

包含两个类：

- `split`：单行参考编码器，保留给测试和兼容路径使用，负责把 ifmap row 编成 `bitstream/r/c/mode`。
- `SplitUnit`：cycle-level split 前端。`row_fifo` 是 core 可见的 issue fifo；encoder 在 cycle N 生成的 packet 先进入 `pending_packet`，cycle N+1 才能写入 `row_fifo`，因此不会出现同拍写入又同拍 issue。

conv 编码：

- `mode0`：普通滑窗。
- `mode1`：孤立非零点。
- `mode2`：长度至少 4 的连续全 1 run，长度用 3 bit 编码，最大处理 7。

conv 前端时序：

- `init_stream(if_map, mode='conv')` 接收整张 ifmap 时，先对每一行做全零检测，生成 `row_nonzero_bitmask` 和 `row_nonzero_flags`。后续只处理非全零行。
- `_tick_conv_encoder()` 每拍只处理当前非零行中的一个非零值，相当于硬件 encoder 一拍完成一次 priority encode + mode 判断。
- 处理完一行后，下一拍才开始下一个非全零行。
- `mode1/mode2` 可以在当前非零值分类完成后立即形成 issue packet。
- `mode0` 不等待整个 mode0 segment 结束。当前实现使用小 lookahead 状态：
  - `mode0_known_bits`：记录当前行中哪些输入位置已经被确认。普通 mode0 位置记录真实值；被 `mode1/mode2` 截获的位置在 mode0 视角记录为 0，避免贡献重复计算。
  - `mode0_pending_cols`：记录已经发现、但还在等待 3-bit mask 成熟的输出列 `c`。
  - 当 `[c, c+1, c+2]` 三个输入位置都已确认后，立即生成一条 `mode0` issue packet `{bitstream=[x[c], x[c+1], x[c+2]], r, c, mode=0}`。
- 这个 lookahead 让 mode0 指令按成熟列流出，避免旧实现等到遇到 `mode1/mode2` 或行结束才 flush 整段 mode0，从而减少 core 因 issue fifo 为空产生的前端空转。

linear 编码：

- FC 路径目前只生成 `mode0`，并保持原有行为：初始化时跳过全零 vector，只为非零 vector 生成 `linear_entries`；本次 lookahead 优化只作用于 conv 模式。

### `simulator_real_compute.py`

顶层 simulator。

参数：

- `num_cores`：每个 cluster 内 core 数，即 `num_core_per_cluster`。
- `num_pus`：每个 core 内 kernel/PU 数，即 `num_kernel_per_core`。
- `num_clusters`：逻辑 cluster 数量；当前只实例化 1 个代表性 cluster。
- `retire_column`：conv 中是每个 psum pool 行每拍最多 retire 的列数；FC 中是同一个 column 打包 retire 的 row 数。
- `num_sub_banks`：cluster 版本保留兼容参数，但 accumulator 不再使用它做仲裁；pro/旧版本仍可使用该参数。
- `enable_reduce_tree`：传给每个 cluster accumulator。默认开启；runner 的 `--disable_reduce_tree` 会把它关掉，用于评估没有同点 reduce tree 时的性能损失。
- `psum_pool_rows`：每个 core 的 psum pool buffer 行数，默认 6；runner 的 `--psum_pool_rows 3` 用于测试只保留 3 行 buffer 的设计。

功能：

- 构建 `self.clusters[0][core_id]` 这一份代表性 cluster。
- `self.cores = self.clusters[0]` 作为 scheduler 监测对象。
- conv 路径中，`num_clusters` 仍表示一个逻辑 wave 同时覆盖多少个 Cout tile；Python 只推演第 0 个 Cout tile 的代表性行为。
- FC 路径中，`num_clusters` 仍表示一个逻辑 wave 同时覆盖多少个 Cout tile；Python 只推演第 0 个 Cout tile 的代表性行为。
- 每当代表性 cluster 的 core 空闲，就取一个 Cin 任务并派发给该 core。
- 统计代表性 cluster 的 PE cycles 和 accumulator stats，同时在 `global_stats` 中记录 `num_clusters` 与 `num_simulated_clusters=1`。

当前仍沿用 pro/plus 的代表性缩放方式：只真实模拟一个 cluster wave，然后按 Cout wave 数缩放总 cycles 和数据搬运。

### `test_cluster.py`

cluster 版本的基本测试。

覆盖：

- mixed mode row 与 dense reference 对齐。
- SplitUnit 能发出 mode2。
- core 的 mode0 bitstream pointer 推进规则。
- `retire_column=3` 时，conv 一个 psum row 每拍只 retire 3 列，FC 则把同一个 column 的 3 行打包 retire。
- accumulator 按 `(row, col)` 点级访问建模：同 row 不同 col 接收，完全相同的点由 reduce tree 合并。
- `num_clusters=2` 时，仍只实例化一个代表性 cluster，并在 stats 中记录 `num_simulated_clusters=1`。

运行：

```bash
python -B model/outer_product/simulator_conv_fc_cluster/test_cluster.py
```

### `area_com.py`

面积估算脚本。

当前根据：

- `num_cluster`
- `num_core_per_cluster`
- `num_kernel_per_core`
- `psum_pool_rows`
- `psum_width`
- `oh/ow`

估算 psum pool 和 accumulator 的容量及面积。这里的 `num_kernel_per_core` 对应 simulator 中的 `num_pus`。

## 当前 SplitUnit 优化评估

配置：

```bash
python model/test_outerproduct_simulator/simulator_runner.py \
  --network TIM_NCARS \
  --num_cores 4 \
  --num_pus 4 \
  --num_clusters 2 \
  --bank_h 16 \
  --bank_w 16 \
  --fetch_rows 8 \
  --split_fifo_depth 2 \
  --retire_column 3 \
  --T 16 \
  --simulator cluster
```

在仅修改 conv SplitUnit、FC 路径保持原行为的前提下，mode0 lookahead 优化前后的 TIM_NCARS 聚合统计如下：

| 版本 | total cycle | 正常流水 | SplitUnit 慢/issue 空 | psum_pool 不够 stall | 计算结束等 psum 写 accumulator |
|---|---:|---:|---:|---:|---:|
| conv streaming split，mode0 整段 flush | 741,565 | 80.94% | 10.03% | 0.04% | 9.00% |
| conv streaming split，mode0 lookahead | 716,997 | 84.53% | 7.54% | 0.03% | 7.90% |

结论：

- `Compute issue` 保持 104,753 core-cycles，说明计算工作量没有改变。
- `SplitUnit 慢/issue 空` 从 12,975 降到 9,349 core-cycles，降低约 27.95%。
- total cycle 从 741,565 降到 716,997，降低 24,568 cycles，约 3.31%。
- 优化的收益来自 mode0 成熟列即时发射；它减少了 core 等待 mode0 segment flush 的时间。

## Psum Pool 行数评估

配置同上，并保持 accumulator 为点级寄存器阵列、reduce tree 开启。只改变每个 core 的 psum pool buffer 行数：

| psum_pool_rows | total cycle | 正常流水 | SplitUnit 慢/issue 空 | psum_pool 不够 stall | 计算结束等 psum 写 accumulator | bank conflicts | retire backpressure |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 716,997 | 84.53% | 7.54% | 0.03% | 7.90% | 0 | 0 |
| 3 | 805,674 | 74.47% | 5.94% | 12.34% | 7.24% | 0 | 0 |

结论：reg-array accumulator 和 reduce tree 基本消除了后端写回反压，但 3 行 psum pool 会让真计算 stall 明显增加。TIM_NCARS 上 total cycle 增加 88,677 cycles，约 12.37%，因此 3 行不是当前配置下的性能无损面积优化。

## Runner 使用

`model/test_outerproduct_simulator/simulator_runner.py` 已加入：

```bash
--simulator cluster
--num_clusters <N>
--retire_column <C>
--psum_pool_rows <R>
--disable_reduce_tree   # 可选，仅用于 accumulator 无 reduce tree 的 ablation
```

示例：

```bash
python model/test_outerproduct_simulator/simulator_runner.py \
  --network QKFormer_DVSGesture \
  --num_cores 4 \
  --num_pus 4 \
  --num_clusters 2 \
  --retire_column 3 \
  --bank_h 128 \
  --bank_w 128 \
  --fetch_rows 8 \
  --split_fifo_depth 2 \
  --T 16 \
  --simulator cluster
```

输出目录：

```text
output/simulator_outproduct/<network>_cluster
```
