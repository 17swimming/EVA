# 测试
以下命令都在项目根目录 `D:\desktop\SCNN_acc` 下运行。

# 消融研究（不启动Prosperity和Gustav）
消融 runner 不启动 Prosperity 和 Gustav。由于它复用了旧 runner 的 `cluster` 入口，需要先设置环境变量，强制加载 `cluster_parallel`：
```powershell
$env:EVA_CLUSTER_PARALLEL='1'
```
默认开启 reduce tree。统一配置为6 core、16 PU、12x12、FIFO深度4、PsumPool深度6、每拍退休3列。

SDTrack，读取 `model/test_outerproduct_simulator/SDTrack_FE108 copy.csv`：

```powershell
python -X utf8 model/outer_product/ablation_study/simulator_runner_ablation.py --network SDTrack_FE108 --num_cores 6 --num_pus 16 --num_clusters 1 --bank_h 12 --bank_w 12 --retire_column 3 --psum_pool_rows 6 --T 4 --simulator cluster --enabled_modes 0,1,2 --split_fifo_depth 4 --output-root model/outer_product/ablation_study/cluster_parallel
```

SpikeBRGNet，读取 `model/test_outerproduct_simulator/SpikeBRGNet_DDD17 copy.csv`：

```powershell
python -X utf8 model/outer_product/ablation_study/simulator_runner_ablation.py --network SpikeBRGNet_DDD17 --num_cores 6 --num_pus 16 --num_clusters 1 --bank_h 12 --bank_w 12 --retire_column 3 --psum_pool_rows 6 --T 5 --simulator cluster --enabled_modes 0,1,2 --split_fifo_depth 4 --output-root model/outer_product/ablation_study/cluster_parallel
```

EMS-YOLO，读取 `model/test_outerproduct_simulator/EMSYOLO_GEN1 copy.csv`：

```powershell
python -X utf8 model/outer_product/ablation_study/simulator_runner_ablation.py --network EMSYOLO_GEN1 --num_cores 6 --num_pus 16 --num_clusters 1 --bank_h 12 --bank_w 12 --retire_column 3 --psum_pool_rows 6 --T 5 --simulator cluster --enabled_modes 0,1,2 --split_fifo_depth 4 --output-root model/outer_product/ablation_study/cluster_parallel
```

TIM，读取 `model/test_outerproduct_simulator/TIM_NCARS.csv`：

```powershell
python -X utf8 model/outer_product/ablation_study/simulator_runner_ablation.py --network TIM_NCARS --num_cores 6 --num_pus 16 --num_clusters 1 --bank_h 12 --bank_w 12 --retire_column 3 --psum_pool_rows 6 --T 10 --simulator cluster --enabled_modes 0,1,2 --split_fifo_depth 4 --output-root model/outer_product/ablation_study/cluster_parallel
```

QKFormer，读取 `model/test_outerproduct_simulator/QKFormer_DVSGesture.csv`：

```powershell
python -X utf8 model/outer_product/ablation_study/simulator_runner_ablation.py --network QKFormer_DVSGesture --num_cores 6 --num_pus 16 --num_clusters 1 --bank_h 12 --bank_w 12 --retire_column 3 --psum_pool_rows 6 --T 16 --simulator cluster --enabled_modes 0,1,2 --split_fifo_depth 4 --output-root model/outer_product/ablation_study/cluster_parallel
```

## 正常启动（会启动Prosperity和Gustav）：
到test_outerproduct_simulator目录下，
这里以TIM为列子：
python -X utf8 simulator_runner.py --network TIM_NCARS --num_cores 6 --num_pus 16 --num_clusters 1 --bank_h 12 --bank_w 12 --retire_column 3 --T 10 --simulator cluster_parallel --enabled_modes 0,1,2  --split_fifo_depth 4

# 性能统计
统计所有 Core 的平均值。比如 compute cycle=10，6个 Core 累积前端 stall 12次，则前端 stall 比例为 `12/(10x6)=20%`。

每拍对每个 Core 依次判断：
- 成功处理 package：正常计算。
- Split 尚未结束、取不到 package：SplitUnit 供数不足。
- Psum 分配失败：Psum 分配停顿。
- Split 已结束、没有 package，但 PsumPool 仍有待写回数据：Psum 写回排空。
- 以上均不满足：空闲/调度。

空闲/调度主要包括 Core 提前完成后等待其他 Core，以及完成状态确认等控制周期。

## SplitUnit 供数不足

不只由“前一行稀疏、后一行稠密”导致，还包括：
- 每个 Cin tile 启动时，Core 先读 FIFO，Split 后生成第一个 package。
- mode1/2 只检测、不生成 package，对应处理拍可能使 FIFO 读空。
- 严格行序会阻塞后续行，即使另一个 Split 的 FIFO 已有 package 也不能读取。
- 全零行、没有 mode0 的行以及等待凑齐 mode0 的3个输入位置都可能产生空拍。

## Psum 写回排空

该开销在每个非零 Cin tile 完成后都会发生，不是整层只发生一次：
- 最后由 W0/W1 更新的两行没有后续 W2 经过，不能通过 PE2 完成 stream retire。
- 尾部全零行和被丢弃的 mode1/2 不生成 package，不能推动 Core 的退休窗口。
- `retire_column=3` 时，12列残留数据最多需要约4拍排空。
- PsumPool 变为 `PENDING_WB` 后，最早下一拍才能开始写回。

FC 的 accumulator 排空是另一项开销。每个 M tile 完成后读取完整的12x12阵列，每拍读取4个word，该周期直接计入 FC 的 total cycle，不属于上述 Conv 五类统计。

# shift.py: 

卷积编码class splitunit：
接收if_map后，在接收的时候就需要判断这一整行是否全0，并用一个bitmask记录哪些行全零，eg： bitmask = 00100就表示只有第2行非全零。不再需要预取逻辑，在启动计算时，根据bitmask，取非全零行给split进行处理，比如当前tick发现这一行都处理完了，下一个tick就开始处理下一个非全零行。

在处理非零行时，使用split.process的while循环内的逻辑，一次循环就表示每拍处理一个非零值，单需要修改为，每处理一个非零值，编码后的结果直接压入issue fifo。

mode1和mode2检测完后可以直接写入issue fifo，但对 mode0 来说，PE 消费的是某个输出列 c 对应的 3 个输入位置：
‘’‘mask = input[c], input[c+1], input[c+2]’‘’
所以只要 split 已经扫描/确认到 c+2 的信息，c 这一条 mode0 指令就可以放入issue fifo了。

(根据idx分类mode的代码主要是split.process的while循环内的逻辑。)


FC编码：
预处理（OR 归约树）：将 一整行SRAM 数据按每 3 个spike分组，经过 x 个简单的或门（OR Gate），生成一个 x-bit 的 vector_valid 掩码。
寻找非零向量：将这 x-bit 掩码输入到一个迷你的 x-bit 优先编码器中，直接输出第一个非零向量的索引 vec_idx。
坐标计算（纯连线/加法）：r 和 c 直接通过 vec_idx 和基础偏移量计算，完全不需要除法和取余。
数据提取与屏蔽：用 vec_idx 控制一个标准的 x 选 1 的  MUX 提取数据；下一拍，只需将 x-bit 掩码中的这一位清零，或将 SRAM 中已经处理的3个spike清零即可。


# core.py：
只要issue fifo非空，就可以issue，cycle+1
pe每拍计算后写回psum_pool，直至psum_pool没有可用的行，计算才会暂停

# SharedPsumPool.py
pe的写回和stream retire互不影响。
pe根据mode写回，mode0写入列c，mode1写入[c-2:c],mode2写入至多9列。


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
然后下一拍，才能根据 `retire_limit` 和 `retire_cursor` 执行 retire。比如cycle1，PE2消耗了列c，那么retire_limit = c + 1，cycle2时才能retire[retire_cursor:retire_limit]之间的retire_column列，目前参数是3列。

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


# accumulator.py
cluster 版本的 accumulator 按硬件寄存器阵列建模，整个 accumulator 视为 `bank_h * bank_w` 个点，每个点是 `(row, col)`，同一拍任意点都可以被访问。

仲裁：只看完全相同的 `(row, col)`。同 row 不同 col 不冲突，例如 core0 请求 row 0/1/2 的 col 4，core1 请求 row 0/1/2 的 col 5，二者都可以同拍接收。

累加：如果多个 core 在同一拍请求完全相同的 `(row, col)`，在 accumulator 前用 reduce tree 合并数据，然后对该点写一次。因此最坏情况下 4 个 core 都 overlap 到相同 3 个点，也可以通过 reduce tree 接收，不应导致 psum_pool pending wb。
