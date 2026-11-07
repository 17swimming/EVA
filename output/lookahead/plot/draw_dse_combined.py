"""重新读取设计空间探索周期，按主频换算延迟并纵向绘制三张压缩子图。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiment_colors import CONFIGURATION_COLORS
from output.lookahead.plot import draw_tile_size_dse as tile_plot
from output.lookahead.plot import draw_core_count_dse as core_plot
from output.lookahead.plot import draw_pu_number as pu_plot

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
from openpyxl import load_workbook


def load_cycles(path, configuration_field):
    """采用完成日志的网络总周期，保留逐层表格之外的修正。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = {}
    for row in rows:
        key = (row["network"], int(row[configuration_field]))
        value = float(row["eva_total_cycles"])
        if key in values or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid or duplicate total cycles: {path}: {key}")
        values[key] = value
    return values


def load_frequency(path):
    """从端到端工作簿读取设计的主频，不修改工作簿。"""
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book["config"]
        columns = {cell.value: index for index, cell in enumerate(sheet[1], 1)}
        frequency = float(sheet.cell(4, columns["EVA"]).value)
        if sheet.cell(4, 1).value != "Freq.(MHz)":
            raise ValueError("Workbook frequency row does not match Freq.(MHz)")
    finally:
        book.close()
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("Frequency must be positive")
    return frequency


def draw_panel(axis, cycles, configurations, names, areas, ticks, reference, frequency):
    positions = np.arange(len(pu_plot.NETWORKS) + 1) * 0.94
    records = []
    for index, configuration in enumerate(configurations):
        values = [cycles[(network, configuration)] / (frequency * 1000)
                  for network in pu_plot.NETWORKS]
        values.append(math.exp(sum(math.log(value) for value in values) / len(values)))
        if min(values) <= ticks[0] or max(values) > ticks[-1]:
            raise ValueError(f"Latency outside the logarithmic axis limits: {min(values)}--{max(values)} ms")
        area = float(areas[str(configuration)])
        bars = axis.bar(positions + (index - 1) * 0.245, np.asarray(values) - ticks[0], 0.19,
                        bottom=ticks[0],
                        color=CONFIGURATION_COLORS[index],
                        label=f"{names[index]} ({area:.4f} mm²)", zorder=3,
                        edgecolor="#333333", linewidth=0.45)
        axis.bar_label(bars, labels=[f"{value:.1f}" if value < 10 else f"{value:.0f}"
                                    for value in values],
                       padding=1, fontsize=5.5, fontweight="bold")
        for network, value in zip([*pu_plot.NETWORKS, "Gmean"], values):
            records.append({"network": network, "configuration": configuration,
                            "total_cycles": cycles.get((network, configuration)),
                            "latency_ms": value, "total_area_mm2": area})
    # 使用真正的十进制对数坐标，柱顶仍表示以毫秒为单位的实际延迟。
    axis.set_yscale("log", base=10)
    axis.minorticks_off()
    # 顶部预留文字空间，避免较高柱子的数值与上边框相交。
    axis.set_ylim(ticks[0], ticks[-1] * 2)
    axis.set_yticks(ticks)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_ylabel("Latency (ms)", labelpad=1, fontsize=8)
    axis.set_xticks(positions, [*pu_plot.LABELS, "Gmean"])
    axis.set_xlim(positions[0] - 0.43, positions[-1] + 0.43)
    axis.tick_params(axis="x", labelsize=7.5, length=0, pad=3)
    axis.tick_params(axis="y", labelsize=7, length=2, pad=2)
    axis.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.55, color="#b6b6b6")
    axis.set_axisbelow(True)
    axis.axhline(reference, color="#222222", linewidth=1.25,
                 linestyle=(0, (4, 2)), zorder=4)
    for label, value in zip(axis.get_yticklabels(), ticks):
        if math.isclose(value, reference):
            label.set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    return records


def draw(tile_cycles, core_cycles, pu_rows, output_dir, frequency, config, height_scale=2 / 3):
    """使用实际总周期与面积，保持三个子图的共享配色和紧凑布局。"""
    if height_scale <= 0:
        raise ValueError("Height scale must be positive")
    ticks = np.asarray(config["latency_ticks_ms"], dtype=float)
    reference = float(config["reference_latency_ms"])
    if (config.get("latency_scale") != "log" or len(ticks) < 2
            or not np.all(np.isfinite(ticks)) or ticks[0] <= 0 or not np.all(np.diff(ticks) > 0)
            or not np.any(np.isclose(ticks, reference))):
        raise ValueError("Use a logarithmic scale with positive increasing ticks including the reference")
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    heights = [1.8, core_plot.FIGURE_SIZE[1], tile_plot.FIGURE_SIZE[1]]
    figure, axes = plt.subplots(3, 1, figsize=(3.5, sum(heights) * height_scale),
                                sharex=True, gridspec_kw={"height_ratios": heights})
    figure.subplots_adjust(left=0.14, right=0.995, bottom=0.08, top=0.94, hspace=0.42)
    pu_cycles = {(row["network"], int(row["num_pus"])): float(row["total_cycles"])
                 for row in pu_rows}
    panels = [
        ("simd_width", pu_cycles, (16, 32, 64), ("16", "32", "64")),
        ("core_number", core_cycles, (6, 8, 10), ("6 cores", "8 cores", "10 cores")),
        ("tile_size", tile_cycles, (8, 12, 16), ("8×8", "12×12", "16×16")),
    ]
    audit = {"frequency_mhz": frequency, "latency_scale": "log", "log_base": 10,
             "latency_ticks_ms": ticks.tolist(),
             "reference_latency_ms": reference, "area_source": config["area_source"],
             "panels": {}}
    for index, (name, cycles, configurations, labels) in enumerate(panels):
        axis = axes[index]
        audit["panels"][name] = draw_panel(
            axis, cycles, configurations, labels, config["total_area_mm2"][name],
            ticks, reference, frequency)
        handles, legend_labels = axis.get_legend_handles_labels()
        if name == "simd_width":
            handles.insert(0, Line2D([], [], linestyle="none"))
            legend_labels.insert(0, "SIMD Width:")
        axis.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 1.015),
                    ncol=len(legend_labels), frameon=False, fontsize=6, handlelength=0.8,
                    handletextpad=0.25, columnspacing=0.7, borderaxespad=0)
        axis.text(-0.13, 1.055, f"({chr(ord('a') + index)})", transform=axis.transAxes,
                  fontsize=8, fontweight="bold", va="bottom", ha="left")
        axis.tick_params(axis="x", labelbottom=index == 2)
    # 将与参考线相交的柱顶文字移至较近的一侧，保留两磅间隔，再检查相邻文字是否重叠。
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in axes:
        annotations = [text for text in axis.texts if text.get_text()[0].isdigit()]
        reference_y = axis.transData.transform((0, reference))[1]
        clearance = figure.dpi * 2 / 72
        for annotation in annotations:
            bounds = annotation.get_window_extent(renderer)
            if bounds.y0 < reference_y + clearance and bounds.y1 > reference_y - clearance:
                horizontal, vertical = annotation.get_position()
                upwards = reference_y + clearance - bounds.y0
                downwards = reference_y - clearance - bounds.y1
                displacement = upwards if abs(upwards) <= abs(downwards) else downwards
                annotation.set_position((horizontal, vertical + displacement * 72 / figure.dpi))
        for fontsize in (5.5, 5.25, 5):
            for annotation in annotations:
                annotation.set_fontsize(fontsize)
            bounds = [text.get_window_extent(renderer).expanded(1.08, 1) for text in annotations]
            if not any(first.overlaps(second) for index, first in enumerate(bounds)
                       for second in bounds[index + 1:]):
                break
        else:
            raise ValueError("Bar labels overlap at the minimum readable font size")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dse_combined_data.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for extension in ("png", "pdf", "svg"):
        path = output_dir / f"dse_combined.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Wrote {path}")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-root", type=Path, default=ROOT / "output/dse_tile_size")
    parser.add_argument("--core-root", type=Path, default=ROOT / "output/dse_core_number")
    parser.add_argument("--pu-root", type=Path, default=ROOT / "output/dse_SIMD_Width")
    parser.add_argument("--log-root", type=Path, default=ROOT / "output/lookahead")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/lookahead/figures")
    parser.add_argument("--workbook", type=Path, default=ROOT / "output/lookahead/end_to_end.xlsx")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("dse_plot_config.json"))
    parser.add_argument("--height-scale", type=float, default=2 / 3,
                        help="每张子图相对独立图的高度比例，默认三分之二")
    args = parser.parse_args()
    tile_plot.collect_lookahead_results(args.tile_root, [name for name, _ in tile_plot.WORKLOADS],
                                       tile_plot.TILE_SIZES, require_no_baselines=False)
    core_plot.collect_results(args.core_root)
    pu_rows = pu_plot.collect(args.log_root.resolve(), args.pu_root.resolve(),
                              Path(__file__).resolve().parent / "manifests")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    draw(load_cycles(args.tile_root / "tile_size_dse_summary.csv", "tile_h"),
         load_cycles(args.core_root / "core_count_dse_summary.csv", "num_cores"),
         pu_rows, args.output_dir, load_frequency(args.workbook), config, args.height_scale)


if __name__ == "__main__":
    main()
