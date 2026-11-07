# FC 退休方式修改记录

## 当前状态

按用户最新要求，`_k_zxy/_k`（即本工程中的 zxy_k）与本实验目录的 FC 均恢复为 `_k` 的包形态：**同一列，至多三个输出行**。不修改 `config_n_m`、原 `_k` 或 RTL。TIM 性能测试仍暂停；此前中断运行的输出不能用于评价本次 FC 修改。

## 修改范围与接口

- CONV 保持同行不同列，每包至多三个有效点。
- FC 在同一个 PE 行组 `[3g, 3g+1, 3g+2]` 内，按列构造 `fc_column` 包。不跨行组拼包；只有一至两个有效点时也允许提交。
- FC 采用 `request_bundle_partial`，每点独立确认。已接受点不重复提交；未接受点保留在原 Psum 槽内重试，不另加无界队列。在途点不能重复提交。
- 一个 K tile 的线性输入流对每个行组的每列最多产生一个 packet，因此三个 PE 完成本列后即可声明该列结果就绪并退休，恢复 `_k` 的计算后提交方式。前后两次 flush 共享每拍 bundle 数量限制。
- 保留 zxy_k 的 Psum 分配、任务映射、每拍退休包数参数与 LIF 周期模型；并非整体替换成旧 `_k` 的 Core/Psum 实现。
- FC 固定采用上述稀疏点接口；`retire_iface=positional` 仅影响 CONV，不将 FC 拆成三个行包。

实验版 result array 的地址组织为：CONV `bank=row, slot=col`；FC `bank=col, slot=row`。默认均为 12 个 bank、每 bank 12 个结果表项；非方形配置下 FC 为 `ow` 个 bank，每 bank `oh` 个表项。每 bank 每拍最多服务一个 Core 的一个包。

实验版 Core 的回调表示缓冲**入队接收**；buffer 获得 bank 服务后进入写回流水级，随后更新结果。完成判断包含全部 buffer 和在途写回。因同包各点属于同 bank，实验版当前一起接受或一起拒绝，但仍返回逐点 mask。zxy_k 的接口支持不一致的逐点接受结果。

本目录 `reference_psum.py` 和 `core.py` 已同步此次 zxy_k 修改，用于后续公平对照；`reference_simulator.py` 保留其原 accumulator/调度行为。后续四组对照应全部使用三行一列 FC，不能混用先前同行多列的性能数据。

## config_n_m 检查结论（仅检查，未修改）

实际 FC 路径不调用普通的 Psum→accumulator 退休接口：

1. Core 在一个 M tile 内跨全部 K tile 复用本地 Psum。K tile 完成只表示这一轮计算结束，既不按列定稿，也不释放 Psum。
2. **M 映射**：全部 K tile 完成后，Core 排队使用共享 LIF；周期按 `ceil(3 × valid_m × valid_pu / lif_num)` 计算，目前 `lif_num=128`。处理结束后才释放 pool、领取下一 M tile。
3. **N 映射**：保留“各 Core 并行向 LIF 提供每拍三个 M 位置”的时序假设，周期按 `ceil(valid_M / 3)` 计算；数据从本地 pool 到 LIF，不经过 accumulator。

因此不能把 config_n_m 描述成已经实现了逐拍“三行一列”或“同行三列”的 accumulator 退休包：它对 FC 的最终排空主要是 LIF 周期统计，没有逐包的地址、grant 和 buffer 仲裁模型。文件中保留的 `fc_row` 分支并非当前 FC 调度使用的路径。M/N 的 LIF 吞吐假设也不同，此次按要求保留，未统一。

## 验证

在仓库根目录执行：

```powershell
python -B -m pytest model/outer_product/simulator_conv_fc_cluster_parallel/accumulator_bank_lookahead_study/test_bank.py model/outer_product/simulator_conv_fc_cluster_parallel/accumulator_bank_lookahead_study/test_fc_retirement.py -q -p no:cacheprovider
```

结果：24 passed、1 skipped。跳过项是给实验版额外开启基线退休流水参数的组合；实验版本身已有固定写回流水级，不支持该组合。其正常流水时序已有测试覆盖。

覆盖：逐点 grant、拒绝点重试、在途去重、稀疏/尾包、行组隔离、两种 Psum 分配方式、实际 FC 请求形态、基线延迟响应、CONV 周期和请求序列回归，以及 config_n_m 的 M/N 路径无 accumulator 请求。尚未恢复 TIM，因此这里不提供性能退化结论。
