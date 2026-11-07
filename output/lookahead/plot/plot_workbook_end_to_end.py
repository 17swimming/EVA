"""从端到端结果工作簿绘制共用图例的吞吐率、归一化延迟和能效图。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import FuncFormatter
import numpy as np
from openpyxl import load_workbook

from output.lookahead.plot.plot_cluster_normalized_speedup import (
    COLORS, NETWORK_LABELS, NETWORK_ORDER, WORKLOAD_SPACING, geomean,
    IMPROVEMENT_ARROW_COLOR, IMPROVEMENT_ARROW_STYLE,
)


ROOT = Path(__file__).resolve().parents[3]
METHODS = (
    "PTB", "Jetson (original)", "Jetson (optimized)",
    "GustavSNN", "NVDLA", "Prosperity", "EVA",
)
INPUT_NAMES = ("TIM", "QKFormer", "SpikeBRGNet", "SDTrack", "EMSYOLO")
DISPLAY_NAMES = {
    "Jetson (original)": "Jetson (orig.)",
    "Jetson (optimized)": "Jetson (opt.)",
}

# 柱子宽度小于中心间距，保留同组柱子之间的白色留白。
BAR_WIDTH = 0.09
BAR_CENTER_SPACING = 0.12

# 按论文双栏宽度设置画布；以下字号分别控制图例、坐标轴、刻度和数值标注。
FIGURE_SIZE = (7.125, 5.2)
LEGEND_FONT_SIZE = 8.5
AXIS_FONT_SIZE = 10
TICK_FONT_SIZE = 9
VALUE_FONT_SIZE = 8
# 能效柱顶数值采用常规字重，单独控制字号以保证单栏打印时可读。
ENERGY_VALUE_FONT_SIZE = 7
LATENCY_LIMIT = 10.0
THROUGHPUT_LIMIT = 1000
THROUGHPUT_SCALE_BOUNDARY = 600
THROUGHPUT_LOWER_GRID_INTERVAL = 150
# 单栏画布保持文字的实际字号，通过缩短组间留白和使用工作负载简称节省宽度。
SINGLE_COLUMN_SIZE = (3.5, 4.35)
# 三个绘图区依次为吞吐率、归一化延迟和能效；能效图使用更小的高度。
PANEL_HEIGHT_RATIOS = (1.0, 1.05, 0.65)
ENERGY_TICKS = np.array([0, 100, 500, 1000, 4000], dtype=float)
SINGLE_COLUMN_SPACING = 0.98
SINGLE_COLUMN_BAR_CENTER_SPACING = 0.13
SINGLE_COLUMN_LABELS = ["TIM", "QKFormer", "BRGNet", "SDTrack", "EMSYOLO"]


def positive(value: object, address: str) -> float:
    """检查原始操作数、周期数、主频和实测延迟是否为正数。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected a numeric input in {address}, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Expected a finite positive input in {address}: {number}")
    return number


def load_results(path: Path):
    """读取原始输入并重算延迟，确保替换周期数后无需依赖旧的公式缓存。"""
    workbook = load_workbook(path, read_only=True, data_only=False)
    sheet, config = workbook["performance"], workbook["config"]
    actual = tuple(sheet.cell(row, 1).value for row in range(3, 8))
    if actual != INPUT_NAMES:
        raise ValueError(f"Unexpected workload order: {actual}")
    config_columns = {config.cell(1, column).value: column for column in range(2, 11)}
    cycle_columns = {sheet.cell(2, column).value: column for column in range(2, 7)}
    jetson_columns = {sheet.cell(13, column).value: column for column in (7, 8)}
    throughput = {method: [] for method in METHODS}
    latency_ms = {method: [] for method in METHODS}
    for index, row in enumerate(range(3, 8)):
        operations = positive(sheet.cell(row, 8).value, f"performance!H{row}")
        latencies = {}
        for method in METHODS:
            if method.startswith("Jetson"):
                key = "jetson_ori" if method == "Jetson (original)" else "jetson_opt"
                column = jetson_columns[key]
                latency = positive(sheet.cell(index + 14, column).value,
                                   f"performance!{sheet.cell(index + 14, column).coordinate}")
            else:
                column = cycle_columns[method]
                cycles = positive(sheet.cell(row, column).value,
                                  f"performance!{sheet.cell(row, column).coordinate}")
                frequency_cell = config.cell(4, config_columns[method])
                frequency = positive(frequency_cell.value, f"config!{frequency_cell.coordinate}")
                latency = cycles / (frequency * 1000)
            latencies[method] = latency
            throughput[method].append(operations / (latency * 1e6))
        for method in METHODS:
            latency_ms[method].append(latencies[method])
    energy_efficiency = {}
    for method in METHODS:
        power_name = "Jetson" if method.startswith("Jetson") else method
        power_cell = config.cell(6, config_columns[power_name])
        if power_cell.value is None or str(power_cell.value).strip().lower() in ("", "n/a", "none"):
            energy_efficiency[method] = [float("nan")] * len(INPUT_NAMES)
            print(f"Energy efficiency unavailable for {method}: missing power in config!{power_cell.coordinate}")
        else:
            power_mw = positive(power_cell.value, f"config!{power_cell.coordinate}")
            energy_efficiency[method] = [value * 1000 / power_mw for value in throughput[method]]
    workbook.close()
    labels = [NETWORK_LABELS[network] for network in NETWORK_ORDER]
    return labels, throughput, latency_ms, energy_efficiency


def energy_axis_forward(values):
    """将用户指定的能效刻度映射到等距位置，并在线性末段延伸。"""
    values = np.asarray(values)
    mapped = np.interp(values, ENERGY_TICKS, np.arange(len(ENERGY_TICKS)))
    last_position = len(ENERGY_TICKS) - 1
    last_interval = ENERGY_TICKS[-1] - ENERGY_TICKS[-2]
    first_interval = ENERGY_TICKS[1] - ENERGY_TICKS[0]
    return np.where(values > ENERGY_TICKS[-1],
                    last_position + (values - ENERGY_TICKS[-1]) / last_interval,
                    np.where(values < ENERGY_TICKS[0],
                             (values - ENERGY_TICKS[0]) / first_interval, mapped))


def energy_axis_inverse(values):
    """将能效纵轴位置还原为每焦耳十亿次操作数。"""
    values = np.asarray(values)
    mapped = np.interp(values, np.arange(len(ENERGY_TICKS)), ENERGY_TICKS)
    last_position = len(ENERGY_TICKS) - 1
    last_interval = ENERGY_TICKS[-1] - ENERGY_TICKS[-2]
    first_interval = ENERGY_TICKS[1] - ENERGY_TICKS[0]
    return np.where(values > last_position,
                    ENERGY_TICKS[-1] + (values - last_position) * last_interval,
                    np.where(values < 0, ENERGY_TICKS[0] + values * first_interval, mapped))



def throughput_axis_forward(values):
    """将六个指定刻度等距排列，使纵轴均分为五个区间。"""
    values = np.asarray(values)
    return np.where(values <= THROUGHPUT_SCALE_BOUNDARY,
                    values / THROUGHPUT_LOWER_GRID_INTERVAL,
                    4 + (values - THROUGHPUT_SCALE_BOUNDARY)
                    / (THROUGHPUT_LIMIT - THROUGHPUT_SCALE_BOUNDARY))


def throughput_axis_inverse(values):
    """将纵轴位置还原为吞吐率，用于刻度和箭头文字定位。"""
    values = np.asarray(values)
    return np.where(values <= 4, values * THROUGHPUT_LOWER_GRID_INTERVAL,
                    THROUGHPUT_SCALE_BOUNDARY + (values - 4)
                    * (THROUGHPUT_LIMIT - THROUGHPUT_SCALE_BOUNDARY))


def add_prosperity_improvement(axis, bars, values):
    """在吞吐率图中保留相对于 Prosperity 的提升箭头和加速比。"""
    for workload_index, (pb, eb, pv, ev) in enumerate(zip(
        bars["Prosperity"], bars["EVA"], values["Prosperity"], values["EVA"]
    )):
        start = (pb.get_x() + BAR_WIDTH / 2, min(pv, THROUGHPUT_LIMIT * 0.92))
        end = (eb.get_x(), min(ev, THROUGHPUT_LIMIT))
        axis.annotate("", xy=end, xytext=start, arrowprops={
            "arrowstyle": IMPROVEMENT_ARROW_STYLE, "color": IMPROVEMENT_ARROW_COLOR,
            "lw": 0.9, "mutation_scale": 9, "shrinkB": 0}, zorder=6)
        middle = throughput_axis_inverse((throughput_axis_forward(start[1])
                                          + throughput_axis_forward(end[1])) / 2)
        # 将提升数值放在其他实现的柱顶上方，避免文字遮挡柱体。
        highest_baseline = max(values[method][workload_index] for method in METHODS if method != "EVA")
        minimum_label_position = throughput_axis_inverse(throughput_axis_forward(highest_baseline) + 0.22)
        middle = max(middle, minimum_label_position)
        axis.text(start[0] - 0.07, middle, f"{ev / pv:.2f}×",
                  ha="right", va="center", fontsize=VALUE_FONT_SIZE, fontweight="bold",
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.1}, zorder=7)


def add_energy_improvement(axis, bars, values):
    """连接 Prosperity 与本项目加速器的能效柱，并在箭头右侧加粗标注真实能效比值。"""
    for baseline_bar, eva_bar, baseline, eva in zip(
            bars["Prosperity"], bars["EVA"], values["Prosperity"], values["EVA"]):
        if not all(math.isfinite(value) and value > 0 for value in (baseline, eva)):
            continue
        start = (baseline_bar.get_x() + baseline_bar.get_width(), min(baseline, ENERGY_TICKS[-1]))
        end = (eva_bar.get_x(),
               min(eva, float(energy_axis_inverse(len(ENERGY_TICKS) - 1.15))))
        axis.annotate("", xy=end, xytext=start, arrowprops={
            "arrowstyle": IMPROVEMENT_ARROW_STYLE, "color": IMPROVEMENT_ARROW_COLOR,
            "lw": 0.9, "mutation_scale": 8}, zorder=6)
        # 按显示坐标计算箭头中点，将倍数放在右侧，允许文字覆盖黄色柱体。
        middle = float(energy_axis_inverse(
            (energy_axis_forward(start[1]) + energy_axis_forward(end[1])) / 2))
        axis.annotate(f"{eva / baseline:.2f}×",
                      xy=((start[0] + end[0]) / 2, middle),
                      xytext=(2, 0), textcoords="offset points",
                      ha="left", va="center", fontsize=ENERGY_VALUE_FONT_SIZE,
                      fontweight="bold", annotation_clip=False, zorder=7)


def mark_clipped_bar(axis, position, value, limit, is_latency, stagger=False, label=True):
    """用白色波浪断口表示截断，标注未经截断的实际数值。"""
    wave_x = np.linspace(position - BAR_WIDTH / 2, position + BAR_WIDTH / 2, 40)
    wave_y = limit * (0.955 + 0.008 * np.sin(np.linspace(0, 2 * np.pi, len(wave_x))))
    axis.fill_between(wave_x, wave_y - limit * 0.007, wave_y + limit * 0.007, color="white", zorder=5)
    for offset in (-limit * 0.007, limit * 0.007):
        axis.plot(wave_x, wave_y + offset, color="#333333", linewidth=0.65, zorder=6)
    if not label:
        return
    if is_latency:
        label_top = limit * (0.64 if stagger else 0.915)
        axis.text(position, label_top, f"{value:.1f}×", rotation=90, ha="center", va="top",
                  fontsize=VALUE_FONT_SIZE, fontweight="normal", zorder=7,
                  path_effects=[path_effects.withStroke(linewidth=0.35, foreground="white")])
    else:
        axis.text(position, limit * 1.025, f"{value:.0f}", ha="center", va="bottom",
                  fontsize=VALUE_FONT_SIZE, fontweight="bold", clip_on=False, zorder=7)


def draw(labels, throughput, normalized_latency, latency_ms, energy_efficiency, output, layout="single"):
    """绘制共享图例、各自显示横坐标的上下子图，导出矢量图和预览图。"""
    single = layout == "single"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": TICK_FONT_SIZE,
        "axes.labelsize": 9 if single else AXIS_FONT_SIZE,
        "xtick.labelsize": 8 if single else TICK_FONT_SIZE,
        "ytick.labelsize": 8 if single else TICK_FONT_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(3, 1, figsize=SINGLE_COLUMN_SIZE if single else FIGURE_SIZE, sharex=True,
                                gridspec_kw={"height_ratios": PANEL_HEIGHT_RATIOS})
    figure.subplots_adjust(left=0.10 if single else 0.08, right=0.993,
                           bottom=0.035 if single else 0.055,
                           top=0.89 if single else 0.91, hspace=0.40 if single else 0.35)
    labels = SINGLE_COLUMN_LABELS if single else labels
    x = np.arange(len(labels), dtype=float) * (SINGLE_COLUMN_SPACING if single else WORKLOAD_SPACING)
    eva_latency_ms = latency_ms["EVA"]
    upper = THROUGHPUT_LIMIT
    handles = []
    for panel_index, (axis, data) in enumerate(zip(axes, (throughput, normalized_latency, energy_efficiency))):
        is_latency = panel_index == 1
        is_energy = panel_index == 2
        bars = {}
        panel_methods = [method for method in METHODS
                         if not is_energy or any(math.isfinite(value) for value in data[method])]
        for index, method in enumerate(panel_methods):
            center_spacing = SINGLE_COLUMN_BAR_CENTER_SPACING if single else BAR_CENTER_SPACING
            positions = x + (index - (len(panel_methods) - 1) / 2) * center_spacing
            limit = LATENCY_LIMIT if is_latency else (ENERGY_TICKS[-1] if is_energy else upper)
            heights = np.minimum(data[method], limit)
            bars[method] = axis.bar(
                positions, heights, width=BAR_WIDTH, color=COLORS[method],
                edgecolor="#333333", linewidth=0.45, zorder=3,
                label=DISPLAY_NAMES.get(method, method))
            if panel_index == 0:
                handles.append(bars[method])
            for workload_index, (position, value) in enumerate(zip(positions, data[method])):
                if not math.isfinite(value):
                    continue
                elif is_energy:
                    if value > limit:
                        mark_clipped_bar(axis, position, value, limit, False, label=False)
                    axis.annotate(f"{value:.1f}" if value < 100 else f"{value:.0f}",
                                  xy=(position, min(value, limit)), xytext=(0, 2),
                                  textcoords="offset points", rotation=90,
                                  ha="center", va="bottom", fontsize=ENERGY_VALUE_FONT_SIZE,
                                  fontweight="normal", annotation_clip=False, zorder=7)
                elif value > limit:
                    mark_clipped_bar(axis, position, value, limit, is_latency,
                                     stagger=single and index % 2 == 1)
                elif is_latency and method == "EVA":
                    axis.text(position, value + 0.22, f"{eva_latency_ms[workload_index]:.2f} ms",
                              rotation=90, ha="center", va="bottom",
                              fontsize=VALUE_FONT_SIZE, fontweight="normal", zorder=7,
                              path_effects=[path_effects.withStroke(linewidth=0.35, foreground="white")])
                elif not is_latency and method == "EVA":
                    label_height = value + limit * 0.025
                    if is_energy and energy_axis_forward(limit) - energy_axis_forward(label_height) < 0.4:
                        label_height = limit * 1.025
                    axis.text(position, label_height, f"{value:.0f}",
                              ha="center", va="bottom", fontsize=VALUE_FONT_SIZE, fontweight="bold")
        axis.set_xlim(x[0] - (0.50 if single else 0.55), x[-1] + (0.51 if single else 0.65))
        if is_energy:
            axis.set_yscale("function", functions=(energy_axis_forward, energy_axis_inverse))
        elif not is_latency:
            axis.set_yscale("function", functions=(throughput_axis_forward, throughput_axis_inverse))
        axis.set_ylim(0, limit)
        axis.set_yticks([0, 1, 2, 4, 6, 8, 10] if is_latency
                        else (ENERGY_TICKS if is_energy else [0, 150, 300, 450, 600, 1000]))
        axis.set_ylabel("Normalized Latency" if is_latency else ("GOP/J" if is_energy else "GOP/s"), labelpad=-2)
        axis.annotate(("(a)", "(b)", "(c)")[panel_index],
                      xy=(-0.10 if single else -0.065, 1), xycoords="axes fraction",
                      xytext=(0, 6), textcoords="offset points",
                      ha="left", va="bottom", fontsize=10, fontweight="bold")
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", length=0, pad=2, labelbottom=True)
        axis.tick_params(axis="y", length=2, pad=0.5)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#B8B8B8", linewidth=0.55, linestyle=(0, (3, 3)))
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333")
            spine.set_linewidth(0.8)
        if is_latency:
            axis.axhline(1, color="#222222", linewidth=1.25, linestyle=(0, (4, 2)), zorder=4)
        elif is_energy:
            add_energy_improvement(axis, bars, data)
        else:
            add_prosperity_improvement(axis, bars, data)
    order = [0, 4, 1, 5, 2, 6, 3] if single else list(range(len(METHODS)))
    figure.legend([handles[index] for index in order],
                  [DISPLAY_NAMES.get(METHODS[index], METHODS[index]) for index in order],
                  ncol=4 if single else len(METHODS), loc="upper center",
                  bbox_to_anchor=(0.52 if single else 0.54, 1.005 if single else 0.99),
                  frameon=False, fontsize=8 if single else LEGEND_FONT_SIZE,
                  labelspacing=0.3,
                  columnspacing=0.7 if single else 0.85,
                  handlelength=1.1, handletextpad=0.35)
    for suffix in ("pdf", "svg", "png"):
        # 紧贴所有文字和图形导出，避免便携式文档底部白边叠加论文图题间距。
        figure.savefig(output.with_suffix("." + suffix), dpi=300,
                       bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)


def main():
    """每次直接读取工作簿原始输入，重新计算并生成论文合并图及核对数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/lookahead/end_to_end.xlsx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/lookahead/figures")
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper/hpca2027-latex-template/figures")
    parser.add_argument("--layout", choices=("single", "double"), default="single")
    args = parser.parse_args()
    labels, throughput, latency_ms, energy_efficiency = load_results(args.input)
    normalized_latency = {
        method: [baseline / eva for baseline, eva in zip(latency_ms[method], latency_ms["EVA"])]
        for method in METHODS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "end_to_end_combined"
    for layout in ("single", "double"):
        draw(labels, throughput, normalized_latency, latency_ms, energy_efficiency,
             args.output_dir / f"end_to_end_combined_{layout}", layout)
    for suffix in ("pdf", "svg", "png"):
        shutil.copy2(args.output_dir / f"end_to_end_combined_{args.layout}.{suffix}",
                     output.with_suffix("." + suffix))
    shutil.copy2(output.with_suffix(".pdf"), args.paper_dir / "4_end_to_end_combined.pdf")
    shutil.copy2(output.with_suffix(".pdf"), args.paper_dir / "end_to_end_combined.pdf")
    for metric, data in (("gops", throughput), ("latency_ms", latency_ms),
                         ("normalized_latency", normalized_latency), ("energy_efficiency", energy_efficiency)):
        with (args.output_dir / f"end_to_end_{metric}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Workload", *METHODS])
            for index, name in enumerate(INPUT_NAMES):
                writer.writerow([name, *(data[method][index] if math.isfinite(data[method][index]) else "n/a" for method in METHODS)])
            writer.writerow(["GMean", *(geomean(data[method]) if all(math.isfinite(v) for v in data[method]) else "n/a" for method in METHODS)])
    summary = {
        "normalization_reference": "EVA", "normalized_latency_unit": "dimensionless",
        "throughput_gmean": {method: geomean(throughput[method]) for method in METHODS},
        "energy_efficiency_gmean": {method: geomean(energy_efficiency[method])
                                    if all(math.isfinite(v) for v in energy_efficiency[method]) else None
                                    for method in METHODS},
        "normalized_latency_gmean": {method: geomean(normalized_latency[method]) for method in METHODS},
        "eva_relative_gmean": {method: geomean(normalized_latency[method]) for method in METHODS},
        "eva_peak_gops": max(throughput["EVA"]),
        "eva_max_speedup_over_nvdla": max(normalized_latency["NVDLA"])}
    (args.output_dir / "end_to_end_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
