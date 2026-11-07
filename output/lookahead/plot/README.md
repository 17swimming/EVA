# 论文实验绘图交付

所有当前论文绘图实现和数据提取工具集中在此目录。旧位置仅保留兼容入口。
运行环境为 Python 3.10 或以上，不需要模拟器、神经网络模型或 GPU。

在项目根目录安装依赖并重绘：

```bash
python -m pip install -r output/lookahead/plot/requirements.txt
python output/lookahead/plot_all_figures.py
```

统一入口根据文件位置定位项目，可从任意工作目录运行。也可运行：

```bash
python output/lookahead/plot_all_figures.py --list
python output/lookahead/plot_all_figures.py --only ablation_study
python output/lookahead/plot_all_figures.py --only dse_combined end_to_end
python output/lookahead/plot/plot_ablation_cycles_patterns.py
```

## 实现和输入

| 任务 | 实现 | 输入 |
| --- | --- | --- |
| 消融与模式分布 | `plot_ablation_cycles_patterns.py`、`collect_ablation_results.py` | `output/lookahead/ablation_study/` 的完成日志及逐层表，`end_to_end.xlsx` 主频 |
| Tile size | `draw_tile_size_dse.py`、`collect_tile_dse_results.py` | `output/dse_tile_size/tile{8x8,12x12,16x16}/` |
| Core number | `draw_core_count_dse.py`、`collect_core_count_dse_results.py` | `output/dse_core_number/core{6,8,10}/` |
| SIMD width | `draw_pu_number.py`、`scale_pu_number.py` | `output/lookahead/*_cluster_parallel/` 日志及逐层表，`manifests/` 网络清单 |
| 三张设计空间探索合并图 | `draw_dse_combined.py` | 上述三项输入、`end_to_end.xlsx`、`dse_plot_config.json` |
| 周期分解 | `extract_cycles_breakdown.py` | `output/lookahead/*_cluster_parallel/` 完成日志 |
| 端到端三子图 | `plot_workbook_end_to_end.py`、`plot_cluster_normalized_speedup.py` | `output/lookahead/end_to_end.xlsx` |
| 面积功耗 | `draw_area_power_breakdown.py` | 文件中的现有 `AREA_MM2`、`POWER_MW` 字典 |

图像会生成到各实验输出目录，并由统一入口同步到 `paper/hpca2027-latex-template/figures/`。
可用 `--paper-dir` 指定同步目录。默认八项均会执行，失败会返回非零状态。

## 配色与样式

项目根目录的 `experiment_colors.py` 是共享配色的编辑源，继续在该文件调整颜色。
统一入口每次运行会将其复制到本目录作为交付副本，内容不会反向覆盖项目源文件。
交付包包含根目录配色源，所有图使用同一份配置。各脚本保留原有字号、尺寸和留白参数。
周期分解的分类配色由 `BREAKDOWN_COLORS` 控制，采用浅色柱段和黑色粗体百分比，兼顾彩色显示与黑白打印。
设计空间探索合并图的顺序为 (a) SIMD width、(b) Core number、(c) Tile size；
刻度与面积编辑本目录 `dse_plot_config.json`。扩展性组合图采用真正的十进制对数延迟轴，显示 0.1、1、10、33.3、100 毫秒，纵轴标题简化为 Latency (ms)。与 33.3 毫秒参考线相交的柱顶数字自动向较近一侧错开，保留两磅间隔；顶部预留文字空间。
扩展性图面积按 2026 年 10 月 12 日用户提供的估算总面积表更新，图例保留四位小数，单位为平方毫米：

| 实验 | 配置 | 估算总面积 |
|---|---|---|
| 计算核心数量 | 6 / 8 / 10 | 1.7252 / 2.1095 / 2.4937 |
| 单指令多数据宽度 | 16 / 32 / 64 | 1.7252 / 3.3403 / 6.5704 |
| 分块尺寸 | 8×8 / 12×12 / 16×16 | 1.2342 / 1.7252 / 2.3690 |


端到端组合图在 `plot_workbook_end_to_end.py` 中设置画布尺寸与 `PANEL_HEIGHT_RATIOS`：单栏为 3.5×4.35 英寸，双栏为 7.125×5.2 英寸，三个绘图区高度比例为 1:1.05:0.65，能效子图进一步压缩，字号保持不变。

## 最新消融数据

配置目录为 `R`、`R_S`、`R_C`、`NVDLA+R_C_S`。也识别时间戳目录中的这些配置。
只使用含完成标记、有效末尾总周期及对应逐层结果表的运行，以日志内完成时间选择最新结果。
跨机器文件路径使用文件名核对，并验证层编号、算子类型、输入形状和启用模式。
`NVDLA+R_C_S` 提供全模式周期、DLA 基准与子图 b 的模式计数。

当前五个网络的四种配置均已完成，图中显示五个网络的全部柱子，不绘制几何平均。
后续如有配置缺失，该柱显示 N/A，不混用旧消融工作簿；几何平均仅保留在核对表中。
放入新日志和逐层表后重跑统一入口，会自动读取最新完整结果。
精确值与来源见 `ablation_study/figures/latency_vs_dla.csv` 和 `ablation_status.csv`。
模式计数及逐层明细仍输出到 `output/lookahead/pattern_breakdown/`。
图例使用 NVDLA；其实际延迟按主频换算，在基准柱右侧空白处水平显示，格式为 `xx.xx ms`。TIM、QKFormer 和 BRGNet 的 R+S+C 柱顶另外竖排标注实际延迟。加速比和下降百分比靠近对应箭头，避免遮挡箭头和延迟标注。
延迟标注使用黑色粗体，基准柱对共享颜色作浅色处理，以便黑白打印。
子图 b 的单行图例为 Regular Pattern、Solitary Pattern、Consecutive Pattern。
子图 a 仅保留指定对比标注：五个工作负载的黑色箭头表示 R 相较于 DLA 的延迟加速比；QKFormer 和 BRGNet 的蓝色虚线箭头表示 R+S 相较于 R 的延迟下降百分比；SDTrack 和 EMSYOLO 的红色箭头表示 R+C 相较于 R 的延迟下降百分比。箭头连接真实柱顶，不修改纵坐标尺度。子图 b 的图例下移，计数列保留数值并删除 N 表头。

## 给同事的交付包

运行 `python output/lookahead/plot/package_plots.py`，生成
`output/lookahead/plot_handoff.zip`。包内含本目录全部实现、统一入口、共享配色、
工作簿以及当前八项任务必需的原始日志和逐层数据，保留项目相对目录结构。
解压后在包的根目录运行上面的安装和重绘命令。无需复制原模拟器源码。
Windows 与 Linux 均可运行；没有 Times New Roman 时会使用字体后备选项。
原始日志中的绝对路径只作为实验来源记录，不会被当作本机输入路径访问。

端到端子图编号在绘图区顶部上方偏移 6 磅；归一化延迟子图的数值使用常规字重，保留归一化基准线的粗虚线。能效子图新增从 NVDLA 到 EVA 的箭头，比值按工作簿重算后的 EVA 能效除以 NVDLA 能效计算，截断仅影响显示高度。
