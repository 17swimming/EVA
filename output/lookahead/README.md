# 论文实验配图统一入口

当前八项论文绘图的实现、数据提取工具和配置均在 `plot/`。旧脚本路径保留兼容入口，今后修改 `plot/` 内的实现即可。交付说明见 [plot/README.md](plot/README.md)，依赖清单为 `plot/requirements.txt`。项目根目录 `experiment_colors.py` 继续作为共享配色的编辑源；统一入口运行时会刷新 `plot/experiment_colors.py` 交付副本。

在项目根目录运行：

```powershell
python output/lookahead/plot_all_figures.py
```

在本目录运行：

```powershell
python plot_all_figures.py
```

脚本根据自身位置定位项目，工作目录不影响默认数据和输出路径。需要安装 `matplotlib`、`numpy` 和 `openpyxl`。

当前任务：

| 任务 | 数据来源 | 生成图片 | 同步到论文目录的文件 |
| --- | --- | --- | --- |
| `dse_tile_size` | `output/dse_tile_size/tile{8x8,12x12,16x16}/` 下五个工作负载的完整日志与逐层结果表；重新读取日志末尾总周期 | `output/dse_tile_size/figures/tile_size_dse.{pdf,png,svg}` 和 `tile_size_speedups.csv`，包含五个工作负载与几何平均 | `2_tile_size_dse.pdf` |
| `ablation_study` | `output/lookahead/ablation_study/` 下 R、R_S、R_C、NVDLA+R_C_S 的完整日志和逐层表；递归支持时间戳目录；主频来自 `end_to_end.xlsx` | `ablation_study/figures/fig_ablation_cycles_patterns.{pdf,png,svg}` 及状态、延迟核对表；模式数量保存到 `pattern_breakdown/` | `1_mode_reduction.pdf` |
| `cycles_breakdown` | 五个工作负载目录中的完整模拟日志 | 汇总周期分解图、各网络周期分解图和核对数据 | `EVA_cycles_breakdown.pdf` |
| `end_to_end` | `end_to_end.xlsx` 的原始周期数、主频、操作数、实测延迟和功耗 | 吞吐率、归一化延迟与能效三子图，单栏与双栏版本及核对数据 | `end_to_end_combined.pdf`，同时同步 `4_end_to_end_combined.pdf` |
| `dse_core_number` | `output/dse_core_number/core{6,8,10}/` 下五个工作负载的完整日志与层结果表；每次重新汇总日志末尾的总周期 | `output/dse_core_number/figures/core_count_dse.{pdf,png,svg}`，以六个计算核心为基准的加速比及几何平均 | `3_core_count_dse.pdf` |
| `dse_pu_number` | 当前五个工作负载目录的六个计算核心、十六个处理单元逐层表格及完成日志；输出通道数来自运行器网络清单 | `output/dse_SIMD_Width/` 的网络汇总、逐层缩放明细、分组原因汇总及 `figures/pu_number_dse.{pdf,png,svg}`，包含五个网络与几何平均 | `5_pu_number_dse.pdf` |
| `area_power` | `plot/draw_area_power_breakdown.py` 内现有面积、功耗字典 | `figures/fig_arch_area_power_breakdown.{pdf,png}` | `5_area_power_breakdown.pdf` |
| `dse_combined` | 上述三种实验的原始总周期；`end_to_end.xlsx` 的主频；`plot/dse_plot_config.json` 中用户面积表的总面积 | `output/lookahead/figures/dse_combined.{pdf,png,svg}` 与 `dse_combined_data.json`，三张延迟子图纵向排列 | `dse_combined.pdf` |

计算核心数量图采用与端到端图一致的单栏字体、带边框且留有间隔的柱子、水平虚线和紧凑单行图例，并突出显示 1× 基准线。配色读取项目根目录 `experiment_colors.py` 的 `CONFIGURATION_COLORS`。单独运行 `output/lookahead/plot/draw_core_count_dse.py` 也会重新汇总原始日志；传入 `--input` 可改为直接使用指定汇总表的总周期。

计算核心数量图的纵坐标采用 `0、1、1.2、1.4、1.6` 五个等距刻度，区间内线性插值，属于分段非线性尺度；柱顶标注保留真实加速比。超过显示范围时脚本报错，避免静默截断。

分块尺寸图沿用计算核心数量图的尺寸、字号、柱间留白、边框、配色和等距刻度。三组配置为 8×8、12×12、16×16，以同主频下的 8×8 总周期除以当前配置总周期得到加速比，包含五个工作负载和几何平均。默认重新生成 `output/dse_tile_size/tile_size_dse_summary.csv`，采用完成日志末尾总周期（保留 TIM 的池化修正），并核对各尺寸的输入文件名、层编号、算子类型和输入形状；跨机器绝对路径前缀不参与比较。未完成日志不参与选择，多个完成日志会报错。12×12 的历史运行包含基线结果，绘图只读取其 EVA 周期。

分项脚本为 `output/lookahead/plot/draw_tile_size_dse.py`，可独立运行；`--input` 可指定总周期汇总表而跳过日志提取。共享配色来自 `experiment_colors.py`。

处理单元数量实验采用比例估算，配置为 16、32、64，计算核心数量固定为六个，同主频下以十六个处理单元为基准。按运行器实际执行路径，`run_convolution` 使用 `groups × ceil((Cout / groups) / num_pus)`，`run_fc` 使用 `ceil(Cout / (num_pus × 3))`。每层完整 `total_cycles` 乘以新旧组数之比，包含未隐藏神经元计算和片外访存，不重新计算重叠。注意力层的输出通道数取每个注意力头的维度。TIM 在逐层表格之外的池化修正保持原值，因此基准配置与原日志总周期一致。分项入口为 `output/lookahead/plot/draw_pu_number.py`，每次重新读取原始数据，详细口径与各网络加速比变化原因保存在同目录的 `README.md`。

三张设计空间探索图的压缩合并入口为 `plot/draw_dse_combined.py`，按 SIMD Width、计算核心数量、分块尺寸的顺序纵向排列，分别标注为 (a)、(b)、(c)。总画布为 3.5×3.4 英寸，相当于三张独立图总高度的三分之二，保留共享配色和底部网络标签。三个纵坐标均为 `Latency (ms)`，按 `latency_ms = total_cycles / (frequency_MHz × 1000)` 换算，主频读取 `end_to_end.xlsx` 的 `config` 表（当前 EVA 为 500 MHz），所有配置使用同一主频。纵坐标采用真正的十进制对数尺度，刻度为 `0.1、1、10、33.3、100 ms`，不包含零；纵轴标题简化为 `Latency (ms)`。顶部预留文字空间。33.3 ms 刻度及水平虚线加粗，超过显示范围时报错。`Gmean` 是五个网络绝对延迟的几何平均。柱顶标注小于 10 ms 保留一位小数，其余取整；与 33.3 ms 参考线相交的标注自动向较近一侧错开并留出两磅间隔，相邻文字重叠时缩小字号；完整精度记录在 `dse_combined_data.json`。每个配置的图例显示总面积，单位为 `mm²`，数值采用用户提供截图的总面积列，保存在 `plot/dse_plot_config.json`；SIMD Width 的三个数值仍来自上述比例估算。可独立运行 `python output/lookahead/plot/draw_dse_combined.py`，通过 `--height-scale` 调整高度、`--workbook` 指定主频来源、`--config` 指定刻度及面积配置。分项独立图保留原有加速比口径。

周期分解汇总图整体压缩为 3.5×0.92 英寸，横向柱子的实际高度按消融组合图下半部分的绘图区域、纵坐标范围和柱高换算，使两图的柱子高度一致。两行图例保留完整分类名称，图例与柱内百分比统一使用 6.5 磅加粗字体，工作负载字号为 7。统一入口同时生成 `cycles_breakdown/EVA_cycles_breakdown.{pdf,png,svg}` 与 `figures/EVA_cycles_breakdown.{pdf,png,svg}`，逐层分项数据仍保存在原目录。

扩展性图的面积已更新为 2026 年 10 月 12 日截图中的估算总面积，图例保留四位小数。基准配置为 1.7252 mm²；各配置数值见 `plot/dse_plot_config.json` 和 `plot/README.md` 的面积表。

能效采用工作簿的计算口径：`GOP/J = GOP/s × 1000 ÷ config 中的功耗(mW)`；两种 Jetson 实现共用 Jetson 功耗。缺失功耗的实现不绘制能效柱，核对数据保留 `n/a`。能效纵轴刻度 `0、100、500、1000、4000` 等距排列，超过上限的柱子用断口及真实值表示。三幅子图均使用水平虚线，归一化延迟的 1× 线与刻度加粗。

默认同步至 `paper/hpca2027-latex-template/figures/`，端到端图使用当前论文的单栏布局。工作簿不被修改，修改周期数后再次运行即可重新计算端到端结果。周期分解每次重新提取日志；TIM 的逐层分项合计与日志末尾汇总差异继续记录在核对数据中。

端到端组合图单栏画布为 3.5×4.35 英寸，双栏画布为 7.125×5.2 英寸，三个绘图区按吞吐率、归一化延迟、能效的顺序采用 1:1.05:0.65 的高度比例。保留现有字号和刻度，能效子图进一步压缩。尺寸与高度比例分别由 `plot/plot_workbook_end_to_end.py` 的 `SINGLE_COLUMN_SIZE`、`FIGURE_SIZE` 和 `PANEL_HEIGHT_RATIOS` 控制。

只更新指定实验或查看登记任务：

```powershell
python output/lookahead/plot_all_figures.py --only end_to_end
python output/lookahead/plot_all_figures.py --only cycles_breakdown
python output/lookahead/plot_all_figures.py --only dse_core_number
python output/lookahead/plot_all_figures.py --only dse_pu_number
python output/lookahead/plot_all_figures.py --only dse_combined
python output/lookahead/plot_all_figures.py --only dse_tile_size
python output/lookahead/plot_all_figures.py --only ablation_study
python output/lookahead/plot_all_figures.py --list
```

可以通过 `--workbook`、`--log-root`、`--paper-dir` 指定其他数据或输出位置。`--layout double` 选择双栏端到端图片；在论文中使用该选项时，图片环境和宽度也需相应采用双栏布局。

某个任务失败时，会明确打印失败信息并返回非零退出码，其余任务仍会执行；失败任务不会由入口脚本同步图片。

## 后续绘图接入约定

### 端到端实验的英伟达深度学习加速器周期重算

当前片外访存总带宽为每秒 64 吉字节，时钟为 1 吉赫兹，对应每拍 512 位。
`update_dla_results.py` 按五个 `*_cluster_parallel` 目录中现有结果表的层集合重新调用周期模型。
默认只核对并显示结果；加 `--apply` 会先备份至项目的 `_codex_tmp/dla_bandwidth_backup_*`，
再更新结果表的 `dla_cycles`、`dla_speedup`、日志中的对应数值，以及包含加速比的文件名。
其他加速器的结果保持原值。TIM 的网络总计额外包含未列入逐层表格的池化访存修正。

```powershell
python -X utf8 output/lookahead/update_dla_results.py
python -X utf8 output/lookahead/update_dla_results.py --apply
```

此命令更新原始周期数据；使用工作簿作为来源的论文配图仍从工作簿读取数值。

### 绘图登记

消融组合图保留当前单栏样式，直接读取 `ablation_study/` 的最新完整运行日志。R、R_S、R_C 分别对应 R、R+S、R+C，`NVDLA+R_C_S` 同时提供 R+S+C 和 DLA 基准。支持 `ablation_study/<时间戳>/<配置>/<网络>_cluster_parallel/` 和直接配置目录，以日志内完成时间选取最新结果，核对层编号、输入文件名、算子类型、输入形状与启用模式。主频从工作簿 `config` 读取，先换算延迟再归一化。未完成配置标为 N/A，不回填旧工作簿数据；任一配置缺失时核对表中的五个网络几何平均统一标为 Pending。补入完整日志和逐层表后重新运行会自动纳入。`ablation_study/figures/ablation_status.csv` 记录每种配置的完成状态和来源，`latency_vs_dla.csv` 保留精确值。

当前五个网络的四种配置均已完成，消融图包含五个网络的全部配置，不绘制几何平均，几何平均仅保留在核对表中。子图 a 不再逐柱标注加速比：五个工作负载均用黑色箭头标注 R 相较于 DLA 的延迟加速比；QKFormer 和 BRGNet 用蓝色虚线箭头标注 R+S 相较于 R 的延迟下降百分比；SDTrack 和 EMSYOLO 用红色箭头标注 R+C 相较于 R 的延迟下降百分比。箭头连接真实柱顶，纵坐标尺度保持不变。数字靠近对应箭头；图例显示 NVDLA，其实际延迟在柱右侧空白处水平显示，TIM、QKFormer 和 BRGNet 的 R+S+C 柱顶竖排显示实际延迟。

下图读取 `ablation_study/NVDLA+R_C_S/` 中完整 R+S+C 配置的五个工作负载横向模式分布，组合图标签使用 TIM、QKFormer、BRGNet、SDTrack、EMSYOLO，独立图保留完整网络和数据集名称，图例使用 Regular、Solitary、Consecutive 全称。按日志逐层相加三种模式的实际计数后归一化，每根柱子之和为 1；不平均四舍五入的层比例，不构造不同配置间的数量变化。计数口径为卷积层代表性输出通道组汇总，排除不经过卷积编码路径的层；子图图例稍向下移动，右侧以 K/M 显示总计数并省略 N 表头，精确值和来源在 `pattern_breakdown/pattern_distribution.csv`，逐层数据在 `pattern_breakdown/pattern_distribution_layers.csv`。小于 5% 的色块保留但不强行标注。

所有后续为本论文新增或接入的实验绘图脚本，必须在 `plot_all_figures.py` 的 `FIGURE_TASKS` 中登记，并补充本说明中的任务表。登记内容包括脚本路径、运行参数和论文图片的同步文件名。分项脚本保留独立运行能力和样式参数；默认统一命令必须能够重新生成新增图片，而不是只复制已有文件。

端到端子图编号在绘图区顶部上方偏移 6 磅；归一化延迟子图的数值使用常规字重，保留归一化基准线的粗虚线。能效子图从 Prosperity 向 EVA 绘制箭头，在箭头中点右侧加粗标注 EVA 能效除以 Prosperity 能效的倍数，允许文字覆盖黄色柱体。每根有效能效柱顶竖排显示原始值，小于 100 时保留一位小数，其余取整；精确值保留在核对数据中。原始值使用常规字重，原始值和倍数的字号由 `ENERGY_VALUE_FONT_SIZE` 控制。截断仅影响柱体显示高度，数值与倍数始终根据工作簿重新计算。
