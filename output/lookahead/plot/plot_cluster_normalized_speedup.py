"""绘制以 NVDLA GOPS 为基准的端到端归一化性能图。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiment_colors import ACCELERATOR_COLORS

import matplotlib.pyplot as plt
import numpy as np


# 当前脚本所在目录，用于定位默认输入数据和输出图目录。
THIS_DIR = Path(__file__).resolve().parents[3] / "model/outer_product/end_to_end_eval"
# SDTrack 的端到端结果尚未返回，暂时不参与绘图；结果完成后取消下一行注释即可。
NETWORK_ORDER = [
    "TIM_NCARS",
    "QKFormer_DVSGesture",
    "SpikeBRGNet_DDD17",
    "SDTrack_FE108",
    # "SpikeET_FE108",
    "EMSYOLO_GEN1",
]
# 横坐标命名与 tile-size DSE 图保持一致；换行符控制两行标签排版。
NETWORK_LABELS = {
    "TIM_NCARS": "TIM\nNCARS",
    "QKFormer_DVSGesture": "QKFormer\nDVS128Gesture",
    "SpikeBRGNet_DDD17": "SpikeBRGNet\nDDD17",
    # "SpikeET_FE108": "SpikeET\nFE108",
    "SDTrack_FE108": "SDTrack\nFE108",
    "EMSYOLO_GEN1": "EMSYOLO\nGEN1",
}
# 柱状图的显示顺序与图例顺序。
METHODS = ("NVDLA", "GustavSNN", "Prosperity", "EVA",)
# 端到端性能实验使用论文共用的加速器配色。
COLORS = ACCELERATOR_COLORS

# 字体层级与 tile-size DSE 图一致。
AXIS_LABEL_FONT_SIZE = 26
AXIS_TICK_FONT_SIZE = 16
# 横坐标字体大小与 tile-size DSE 图一致。
WORKLOAD_FONT_SIZE = 24
# 六个 workload 同图时，两个较长名称略小一档，避免相邻双行标签相互覆盖。
QKFORMER_FONT_SIZE = 24
SPIKEBRGNET_FONT_SIZE = 24
LEGEND_FONT_SIZE = 22
VALUE_FONT_SIZE = 16
# 相邻 workload 柱组的横向间距；沿用 tile-size DSE 的留白方式，避免双行标签重叠。
WORKLOAD_SPACING = 1.12
# 是否额外显示柱顶绝对值。默认关闭，避免与 EVA/Prosperity 提升倍率标签重叠。
SHOW_ABSOLUTE_BAR_VALUE = False
# 仅当绝对值标签开启时，达到该阈值的柱才会标注。
ABSOLUTE_VALUE_LABEL_THRESHOLD = 2.5

# NVDLA 归一化基准线：数值 1.0、颜色、线宽和虚线样式均可在此调整。
BASELINE_VALUE = 1.0
BASELINE_COLOR = "#303030"
BASELINE_LINE_WIDTH = 1.8
BASELINE_LINE_STYLE = "--"

# 在最后一个 workload 与 GMean 之间绘制分隔虚线，突出几何平均结果。
GMEAN_SEPARATOR_COLOR = "#8A8A8A"
GMEAN_SEPARATOR_LINE_WIDTH = 1.2
GMEAN_SEPARATOR_LINE_STYLE = "--"

# EVA 相对 Prosperity 提升箭头的外观与布局参数。
IMPROVEMENT_ARROW_COLOR = "#303030"
IMPROVEMENT_ARROW_LINE_WIDTH = 1.6
IMPROVEMENT_ARROW_STYLE = "-|>"
IMPROVEMENT_ARROW_HEAD_SIZE = 16
# 箭头端点相对两个柱左上顶点的垂直偏移；默认 0.0 表示严格连接柱顶。
IMPROVEMENT_ARROW_Y_OFFSET = 0.0
# 倍率文字相对箭头几何中点的横向和纵向偏移；默认 0.0 表示位于箭头中央。
IMPROVEMENT_LABEL_X_OFFSET = 0.0
IMPROVEMENT_LABEL_Y_OFFSET = 0.0
IMPROVEMENT_LABEL_FONT_SIZE = 24
IMPROVEMENT_LABEL_FONT_WEIGHT = "bold"
# 用白色底框遮住穿过文字的箭头，保证中央倍率标签可读。
IMPROVEMENT_LABEL_BACKGROUND_COLOR = "white"
IMPROVEMENT_LABEL_BACKGROUND_PADDING = 0.08


def parse_args() -> argparse.Namespace:
    """解析输入统计 CSV 与图表输出目录。"""
    parser = argparse.ArgumentParser(description="Plot end-to-end speedup normalized to NVDLA GOPS.")
    parser.add_argument(
        "--input",
        type=Path,
        default=THIS_DIR / "results" / "end_to_end_cluster_cycles.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "figures")
    return parser.parse_args()


def geomean(values: list[float]) -> float:
    """计算正数列表的几何平均值，用于 GMean 柱。"""
    return math.exp(sum(math.log(value) for value in values) / len(values))


def load_speedups(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    """读取周期统计，并转换为相对于 NVDLA 的有效 GOPS 比值。"""
    # 使用 network 字段建立索引，避免 CSV 行顺序影响绘图顺序。
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = {row["network"]: row for row in csv.DictReader(f)}
    missing = [network for network in NETWORK_ORDER if network not in rows]
    if missing:
        raise ValueError(f"Missing results for: {', '.join(missing)}")

    values = {method: [] for method in METHODS}
    labels = []
    for network in NETWORK_ORDER:
        row = rows[network]
        # 四类加速器分别记录其端到端总周期数。
        dla_cycles = float(row["total_dla_cycles"])
        eva_cycles = float(row["total_sim_cycles"])
        prosperity_cycles = float(row["total_prosperity_cycles"])
        gustav_cycles = float(row["total_gustav_cycles"])
        if min(dla_cycles, eva_cycles, prosperity_cycles, gustav_cycles) <= 0:
            raise ValueError(f"Non-positive cycle count for {network}")
        # 不使用 CSV 中的短标签，统一使用 DSE 图的两行 workload 名称。
        labels.append(NETWORK_LABELS[network])
        # EVA、NVDLA、Prosperity 均以 500 MHz 运行，GOPS 比等于周期反比。
        values["EVA"].append(dla_cycles / eva_cycles)
        values["NVDLA"].append(1.0)
        values["Prosperity"].append(dla_cycles / prosperity_cycles)
        # GustavSNN 以 1 GHz 运行，NVDLA 以 500 MHz 运行，需补偿 2 倍频率。
        values["GustavSNN"].append(2.0 * dla_cycles / gustav_cycles)

    # 每种加速器对五个 workload 计算几何平均值。
    for method in METHODS:
        values[method].append(geomean(values[method]))
    return labels + ["GMean"], values


def draw(labels: list[str], values: dict[str, list[float]], output_dir: Path) -> None:
    """绘制四种加速器的分组柱状图，并导出 PDF/PNG。"""
    # 字体和嵌入式 TrueType 输出设置参考 tile-size DSE 绘图脚本。
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 20,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "axes.labelweight": "bold",
            "xtick.labelsize": WORKLOAD_FONT_SIZE,
            "ytick.labelsize": AXIS_TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # 每组对应一个 workload，最后一组为五个 workload 的几何平均值。
    # 使用额外的 workload 组间距，给较长的双行横坐标标签留出空间。
    x = np.arange(len(labels), dtype=float) * WORKLOAD_SPACING
    width = 0.19
    fig, ax = plt.subplots(figsize=(16.0, 6.2), constrained_layout=True)
    max_value = max(max(series) for series in values.values())
    # 保存每种加速器的 BarContainer，后续用于定位 Prosperity 和 EVA 柱顶。
    bars_by_method: dict[str, object] = {}
    for index, method in enumerate(METHODS):
        offset = (index - 1.5) * width
        bars = ax.bar(
            x + offset,
            values[method],
            width=width,
            label=method,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        bars_by_method[method] = bars
        # 可选的柱顶绝对值标签；默认关闭以突出 EVA/Prosperity 倍率箭头。
        for bar, value in zip(bars, values[method]):
            if SHOW_ABSOLUTE_BAR_VALUE and value >= ABSOLUTE_VALUE_LABEL_THRESHOLD:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + max_value * 0.025,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=VALUE_FONT_SIZE,
                )

    # 连接同一 workload 中的 Prosperity 与 EVA 柱，标出 EVA 的相对性能提升。
    prosperity_values = values["Prosperity"]
    eva_values = values["EVA"]
    prosperity_bars = bars_by_method["Prosperity"]
    eva_bars = bars_by_method["EVA"]
    for prosperity_bar, eva_bar, prosperity_value, eva_value in zip(
        prosperity_bars,
        eva_bars,
        prosperity_values,
        eva_values,
    ):
        # 箭头端点严格取两根柱的左上顶点；可通过 Y 偏移整体上移或下移。
        arrow_start = (
            prosperity_bar.get_x(),
            prosperity_value + IMPROVEMENT_ARROW_Y_OFFSET,
        )
        arrow_end = (
            eva_bar.get_x(),
            eva_value + IMPROVEMENT_ARROW_Y_OFFSET,
        )
        ax.annotate(
            "",
            xy=arrow_end,
            xytext=arrow_start,
            arrowprops={
                "arrowstyle": IMPROVEMENT_ARROW_STYLE,
                "color": IMPROVEMENT_ARROW_COLOR,
                "lw": IMPROVEMENT_ARROW_LINE_WIDTH,
                "mutation_scale": IMPROVEMENT_ARROW_HEAD_SIZE,
            },
            zorder=6,
        )
        # 在箭头两个端点的几何中点显示 EVA/Prosperity 倍率。
        ax.text(
            (arrow_start[0] + arrow_end[0]) / 2 + IMPROVEMENT_LABEL_X_OFFSET,
            (arrow_start[1] + arrow_end[1]) / 2 + IMPROVEMENT_LABEL_Y_OFFSET,
            f"{eva_value / prosperity_value:.2f}x",
            ha="center",
            va="center",
            color=IMPROVEMENT_ARROW_COLOR,
            fontsize=IMPROVEMENT_LABEL_FONT_SIZE,
            fontweight=IMPROVEMENT_LABEL_FONT_WEIGHT,
            bbox={
                "facecolor": IMPROVEMENT_LABEL_BACKGROUND_COLOR,
                "edgecolor": "none",
                "pad": IMPROVEMENT_LABEL_BACKGROUND_PADDING,
            },
            zorder=7,
        )

    # 纵轴仅给出用户指定的归一化性能名称，NVDLA 柱固定为 1。
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=WORKLOAD_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=AXIS_TICK_FONT_SIZE)
    # QKFormer 与 SpikeBRGNet 名称较长，沿用 DSE 图的单独字号以避免拥挤。
    workload_tick_labels = ax.get_xticklabels()
    workload_tick_labels[1].set_fontsize(QKFORMER_FONT_SIZE)
    workload_tick_labels[2].set_fontsize(SPIKEBRGNET_FONT_SIZE)
    ax.set_ylabel("Norm. Speedup", fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold")
    # 预留箭头及倍率文字上方空间；系数可按版式需要调整。
    ax.set_ylim(0, max_value * 1.30)
    ax.grid(axis="y", color="#D0D3D8", linewidth=0.6, alpha=0.8, zorder=0)
    # 强调归一化位置 1.0，方便直接比较各加速器是否超过 NVDLA 基准。
    ax.axhline(
        BASELINE_VALUE,
        color=BASELINE_COLOR,
        linewidth=BASELINE_LINE_WIDTH,
        linestyle=BASELINE_LINE_STYLE,
        zorder=2,
    )
    # 分隔真实 workload 与最后一组 GMean。
    if len(x) >= 2:
        ax.axvline(
            (x[-2] + x[-1]) / 2,
            color=GMEAN_SEPARATOR_COLOR,
            linewidth=GMEAN_SEPARATOR_LINE_WIDTH,
            linestyle=GMEAN_SEPARATOR_LINE_STYLE,
            zorder=1,
        )
    ax.legend(
        loc="upper left",
        ncol=4,
        frameon=False,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.15,
    )

    # 同时输出论文常用的矢量 PDF 与预览 PNG。
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        output = output_dir / f"fig_end_to_end_performance_nvdla_baseline.{extension}"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Wrote {output}")
    plt.close(fig)


def main() -> None:
    """加载结果并生成端到端性能图。"""
    args = parse_args()
    labels, values = load_speedups(args.input)
    draw(labels, values, args.output_dir)


if __name__ == "__main__":
    main()
