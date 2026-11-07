"""绘制计算核心数量实验，以六个计算核心的执行时间为归一化基准。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiment_colors import CONFIGURATION_COLORS
from output.lookahead.plot.collect_core_count_dse_results import collect_results

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "output/dse_core_number"
WORKLOADS = (
    ("TIM_NCARS", "TIM"),
    ("QKFormer_DVSGesture", "QKFormer"),
    ("SpikeBRGNet_DDD17", "BRGNet"),
    ("SDTrack_FE108", "SDTrack"),
    ("EMSYOLO_GEN1", "EMSYOLO"),
)
CORE_COUNTS = (6, 8, 10)
COLORS = dict(zip(CORE_COUNTS, CONFIGURATION_COLORS))

# 按论文单栏的实际宽度设置字体；以下常量可独立调整字号与柱间留白。
FIGURE_SIZE = (3.5, 1.65)
AXIS_FONT_SIZE = 9
TICK_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8
VALUE_FONT_SIZE = 8
BAR_WIDTH = 0.19
BAR_CENTER_SPACING = 0.245
WORKLOAD_SPACING = 0.94
# 相邻刻度在图中等距排列，数值区间内采用线性插值。
SPEEDUP_TICKS = np.array([0.0, 1.0, 1.2, 1.4, 1.6])
SPEEDUP_TICK_POSITIONS = np.arange(len(SPEEDUP_TICKS), dtype=float)


def speedup_axis_forward(values):
    """把真实加速比映射到分段非线性纵坐标。"""
    return np.interp(values, SPEEDUP_TICKS, SPEEDUP_TICK_POSITIONS)


def speedup_axis_inverse(values):
    """把纵坐标显示位置还原为真实加速比。"""
    return np.interp(values, SPEEDUP_TICK_POSITIONS, SPEEDUP_TICKS)


def load_rows(path):
    """读取汇总总周期并重新计算相对性能，避免使用替换周期前的加速比。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    cycles = {(row["network"], int(row["num_cores"])): float(row["eva_total_cycles"])
              for row in records}
    if len(cycles) != len(records):
        raise ValueError("Duplicate workload/core-count results")
    missing = [(network, cores) for network, _ in WORKLOADS for cores in CORE_COUNTS
               if (network, cores) not in cycles]
    if missing:
        raise ValueError(f"Missing experiment results: {missing}")
    if any(not math.isfinite(value) or value <= 0 for value in cycles.values()):
        raise ValueError("Total cycles must be finite and positive")
    return {(network, cores): cycles[(network, 6)] / cycles[(network, cores)]
            for network, _ in WORKLOADS for cores in CORE_COUNTS}


def draw(rows, output_dir=None, *, axis=None, compact=False):
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": TICK_FONT_SIZE, "axes.labelsize": AXIS_FONT_SIZE,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    labels = [label for _, label in WORKLOADS] + ["Gmean"]
    positions = np.arange(len(labels)) * WORKLOAD_SPACING
    standalone = axis is None
    if standalone:
        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        figure.subplots_adjust(left=0.12, right=0.995, bottom=0.12, top=0.87)
    else:
        figure = axis.figure
    maximum = 1.0
    for index, cores in enumerate(CORE_COUNTS):
        values = [rows[(network, cores)] for network, _ in WORKLOADS]
        values.append(math.exp(sum(math.log(value) for value in values) / len(values)))
        maximum = max(maximum, max(values))
        bars = axis.bar(positions + (index - 1) * BAR_CENTER_SPACING, values, BAR_WIDTH,
                        color=COLORS[cores], label=f"{cores} cores", zorder=3,
                        edgecolor="#333333", linewidth=0.45)
        # 合并图采用横向数值标注，减少标注占用的高度；独立图保留原来的样式。
        labels_on_bars = [f"{value:.2f}" for value in values] if compact else [f"{value:.2f}×" for value in values]
        if not compact or index != 0:
            axis.bar_label(bars, labels=labels_on_bars, padding=1 if compact else 3,
                           fontsize=6 if compact else VALUE_FONT_SIZE,
                           fontweight="bold", rotation=0 if compact else 90)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Speedup over\n6 cores" if compact else "Speedup over 6 cores", labelpad=0)
    if maximum > SPEEDUP_TICKS[-1]:
        raise ValueError("Speedup exceeds the displayed range; extend SPEEDUP_TICKS")
    axis.set_yscale("function", functions=(speedup_axis_forward, speedup_axis_inverse))
    axis.set_ylim(SPEEDUP_TICKS[0], SPEEDUP_TICKS[-1])
    axis.set_xlim(positions[0] - 0.43, positions[-1] + 0.43)
    axis.set_yticks(SPEEDUP_TICKS)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.tick_params(axis="x", length=0, pad=3)
    axis.tick_params(axis="y", length=2, pad=0.5)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#B8B8B8", linewidth=0.55, linestyle=(0, (3, 3)))
    axis.axhline(1, color="#222222", linewidth=1.25, linestyle=(0, (4, 2)), zorder=4)
    for tick, label in zip(axis.get_yticks(), axis.get_yticklabels()):
        if tick == 1:
            label.set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                borderaxespad=0, fontsize=LEGEND_FONT_SIZE, handlelength=1.1,
                handletextpad=0.35, columnspacing=1.0)
    if not standalone:
        return axis
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        output = output_dir / f"core_count_dse.{extension}"
        figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.01)
        print(f"Wrote {output}")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="绘制计算核心数量设计空间探索结果")
    parser.add_argument("--raw-root", type=Path, default=OUTPUT_ROOT,
                        help="默认重新汇总此目录下各配置的日志与层结果")
    parser.add_argument("--input", type=Path,
                        help="直接读取指定总周期汇总表，跳过原始日志汇总")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "figures")
    args = parser.parse_args()
    if args.input is None:
        collect_results(args.raw_root)
        args.input = args.raw_root / "core_count_dse_summary.csv"
    draw(load_rows(args.input), args.output_dir)


if __name__ == "__main__":
    main()
