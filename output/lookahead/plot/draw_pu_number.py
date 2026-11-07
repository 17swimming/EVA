"""重新提取逐层数据并绘制处理单元数量比例估算的加速比。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiment_colors import CONFIGURATION_COLORS
from output.lookahead.plot.scale_pu_number import LABELS, NETWORKS, PU_COUNTS, collect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def draw(rows, output_dir=None, *, axis=None, compact=False):
    """沿用其他设计空间探索图的单栏尺寸、字体和共享配置配色。"""
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    lookup = {(row["network"], row["num_pus"]): float(row["speedup_vs_pu16"]) for row in rows}
    positions = np.arange(len(NETWORKS) + 1) * 0.94
    standalone = axis is None
    if standalone:
        figure, axis = plt.subplots(figsize=(3.5, 1.8))
        figure.subplots_adjust(left=0.11, right=0.995, bottom=0.14, top=0.85)
    else:
        figure = axis.figure
    maximum = 1.0
    for index, pus in enumerate(PU_COUNTS):
        values = [lookup[(network, pus)] for network in NETWORKS]
        values.append(math.exp(sum(math.log(value) for value in values) / len(values)))
        maximum = max(maximum, max(values))
        bars = axis.bar(positions + (index - 1) * 0.245, values, 0.19,
                        color=CONFIGURATION_COLORS[index], label=str(pus), zorder=3,
                        edgecolor="#333333", linewidth=0.45)
        # 合并图采用横向数值标注，减少标注占用的高度；独立图保留原来的样式。
        labels_on_bars = [f"{value:.2f}" for value in values] if compact else [f"{value:.2f}×" for value in values]
        if not compact or index != 0:
            axis.bar_label(bars, labels=labels_on_bars, padding=1 if compact else 2,
                           fontsize=6 if compact else 7,
                           fontweight="bold", rotation=0 if compact else 90)
    axis.set_xticks(positions, [*LABELS, "Gmean"])
    axis.set_ylabel("Estimated\nspeedup" if compact else "Estimated speedup", labelpad=1)
    axis.set_ylim(0, max(4.5, maximum + 0.8))
    axis.set_xlim(positions[0] - 0.43, positions[-1] + 0.43)
    axis.set_yticks([0, 1, 2, 3, 4])
    axis.tick_params(axis="x", length=0, pad=3)
    axis.tick_params(axis="y", length=2, pad=0.5)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#B8B8B8", linewidth=0.55, linestyle=(0, (3, 3)))
    axis.axhline(1, color="#222222", linewidth=1.25, linestyle=(0, (4, 2)), zorder=4)
    axis.get_yticklabels()[1].set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    handles, labels = axis.get_legend_handles_labels()
    # 共用一项图例说明，后面的三项只显示各配置的数据通路宽度。
    axis.legend([Line2D([], [], linestyle="none"), *handles], ["SIMD Width:", *labels],
                frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                borderaxespad=0, fontsize=8, handlelength=1.1,
                handletextpad=0.35, columnspacing=1.0)
    if not standalone:
        return axis
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output_dir / f"pu_number_dse.{extension}", dpi=300,
                       bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "output/lookahead")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/dse_SIMD_Width")
    args = parser.parse_args()
    rows = collect(args.results_root.resolve(), args.output_dir.resolve(), Path(__file__).resolve().parent / "manifests")
    draw(rows, args.output_dir / "figures")
    print(f"Generated processing-unit scaling tables and figures: {args.output_dir}")


if __name__ == "__main__":
    main()
