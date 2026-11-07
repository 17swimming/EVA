"""从最新完整日志绘制消融延迟与完整配置下的横向编码模式分布。"""
import argparse
import csv
import math
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle
import numpy as np
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "output/lookahead/ablation_study"
sys.path.insert(0, str(ROOT))
from experiment_colors import ABLATION_COLORS, CONFIGURATION_COLORS, ACCELERATOR_COLORS
from output.lookahead.plot.collect_ablation_results import collect, completed_result

WORKLOADS = (
    ("TIM", "TIM", "TIM_NCARS", "T"),
    ("QKFORMER", "QKFormer", "QKFormer_DVSGesture", "Q"),
    ("BRGNET", "BRGNet", "SpikeBRGNet_DDD17", "B"),
    ("SDTRACK", "SDTrack", "SDTrack_FE108", "S"),
    ("EMSYOLO", "EMSYOLO", "EMSYOLO_GEN1", "E"),
)
VARIANTS = (("m0", "R"), ("m0+m1", "R+S"), ("m0_m2", "R+C"), ("m0+m1+m2", "R+S+C"))
PLOT_VARIANTS = (("DLA", "NVDLA"),) + VARIANTS
PATTERN_NAMES = ("Regular Pattern", "Solitary Pattern", "Consecutive Pattern")
PATTERN_COLORS = CONFIGURATION_COLORS[:3]
# 以下字号按论文单栏实际尺寸设置；柱子高度和标注阈值可独立调整。
FIGURE_SIZE = (3.5, 2.50)
FONT_SIZE = 8
VALUE_FONT_SIZE = 8
# 基准柱沿用共享配色的色相并混入白色，使黑色延迟标注适合黑白打印。
DLA_WHITE_BLEND = 0.55
LATENCY_TEXT_COLOR = '#000000'
STACK_HEIGHT = 0.78
MIN_LABEL_FRACTION = 0.05
# 箭头连接真实延迟柱顶；弧线仅为小幅下降留出可见的箭头空间。
COMPARISONS = {
    "TIM": (("DLA", "m0", "#000000", "-"),),
    "QKFORMER": (("DLA", "m0", "#000000", "-"),
                 ("m0", "m0+m1", "#2166AC", (0, (2, 2)))),
    "BRGNET": (("DLA", "m0", "#000000", "-"),
               ("m0", "m0+m1", "#2166AC", (0, (2, 2)))),
    "SDTRACK": (("DLA", "m0", "#000000", "-"),
                ("m0", "m0_m2", "#C92323", "-")),
    "EMSYOLO": (("DLA", "m0", "#000000", "-"),
                ("m0", "m0_m2", "#C92323", "-")),
}
MODE_RE = re.compile(r"Conv mode 比例:\s*M0 (\d+) \([^)]*\),\s*M1 (\d+) \([^)]*\),\s*M2 (\d+) \([^)]*\)")
CASE_RE = re.compile(r"^=== 测试用例 (\d+)/\d+: (.+?) ===", re.MULTILINE)


def load_frequencies(path):
    """读取当前论文工作簿中的兆赫主频，用于换算完成日志的周期。"""
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book["config"]
        columns = {cell.value: cell.column for cell in sheet[1] if cell.value}
        values = {name: float(sheet.cell(4, columns[name]).value) for name in ("EVA", "NVDLA")}
        if any(not math.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("Clock frequencies must be finite and positive")
        return values
    finally:
        book.close()


def load_patterns(log_root):
    """逐层汇总代表性输出通道组的计数，排除不经过卷积编码路径的层。"""
    totals, layers = [], []
    for key, label, network, initial in WORKLOADS:
        result = completed_result(log_root, "NVDLA+R_C_S", network)
        if result is None:
            raise ValueError(f"Missing completed pattern reference: {network}")
        paths = [result["path"]]
        text = paths[0].read_text(encoding="utf-8-sig")
        if "仿真任务完成" not in text:
            raise ValueError(f"Incomplete log: {paths[0]}")
        cases, current = list(CASE_RE.finditer(text)), []
        for index, case in enumerate(cases):
            block = text[case.end():cases[index + 1].start() if index + 1 < len(cases) else len(text)]
            matches = MODE_RE.findall(block)
            if not matches:
                continue
            if len(matches) != 1:
                raise ValueError(f"Duplicate pattern counters: {network}, {case[1]}")
            counts = tuple(map(int, matches[0]))
            if not sum(counts):
                continue
            row = dict(network=network, test_case=int(case[1]), layer=case[2],
                       pattern0=counts[0], pattern1=counts[1], pattern2=counts[2],
                       total=sum(counts), source_log=str(paths[0].relative_to(log_root)))
            current.append(row)
            layers.append(row)
        counts = [sum(r[f"pattern{i}"] for r in current) for i in range(3)]
        total = sum(counts)
        if total <= 0:
            raise ValueError(f"No positive pattern counts: {network}")
        totals.append(dict(network=network, initial=initial, configuration="R+S+C", conv_layers=len(current),
                           pattern0=counts[0], pattern1=counts[1], pattern2=counts[2], total=total,
                           fraction0=counts[0] / total, fraction1=counts[1] / total, fraction2=counts[2] / total,
                           scope="sum of representative output-channel-group counts across Conv layers",
                           source_log=str(paths[0].relative_to(log_root))))
    return totals, layers


def draw_cycles(axis, rows, compact=False):
    lookup = {(r["workload"], r["variant"]): r for r in rows}
    groups = [(key, label) for key, label, _, _ in WORKLOADS]
    centers = np.arange(len(groups), dtype=float)
    for i, (variant, label) in enumerate(PLOT_VARIANTS):
        selected = [lookup[(key, variant)] for key, _ in groups]
        color = ACCELERATOR_COLORS["NVDLA"] if variant == "DLA" else ABLATION_COLORS[variant]
        if variant == "DLA":
            color = tuple(channel * (1 - DLA_WHITE_BLEND) + DLA_WHITE_BLEND
                          for channel in to_rgb(color))
        bars = axis.bar(centers + (i - 2) * 0.17, [r["normalized_latency"] for r in selected],
                        width=0.135, color=color, edgecolor="#333333",
                        linewidth=0.45, label=label, zorder=3)
        if variant == "DLA":
            for bar, row in zip(bars, selected):
                if not math.isfinite(row["normalized_latency"]):
                    continue
                latency = row["dla_latency_ms"]
                if latency == "":
                    values = [r["dla_latency_ms"] for r in selected if r["dla_latency_ms"] != ""]
                    latency = math.exp(sum(math.log(v) for v in values) / len(values))
                axis.text(bar.get_x() + bar.get_width() + 0.06, 0.98, f"{latency:.2f} ms",
                          ha="left", va="top", color=LATENCY_TEXT_COLOR,
                          fontsize=6.5 if compact else VALUE_FONT_SIZE, fontweight="bold")
        if variant == "m0+m1+m2":
            for bar, row in zip(bars, selected):
                if row["workload"] not in ("TIM", "QKFORMER", "BRGNET"):
                    continue
                if not math.isfinite(row["normalized_latency"]):
                    continue
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                          f"{row['latency_ms']:.2f} ms", ha="center", va="bottom", rotation=90,
                          color=LATENCY_TEXT_COLOR, fontsize=6.5 if compact else VALUE_FONT_SIZE,
                          fontweight="bold", zorder=7)
        for bar, row in zip(bars, selected):
            if not math.isfinite(row["normalized_latency"]) and row["workload"] != "GMEAN":
                axis.text(bar.get_x() + bar.get_width() / 2, 0.025, "N/A", rotation=90,
                          ha="center", va="bottom", fontsize=6.5, color="#666666")
    variant_positions = {variant: (i - 2) * 0.17 for i, (variant, _) in enumerate(PLOT_VARIANTS)}
    comparisons = ((center, key, comparison) for center, (key, _) in zip(centers, groups)
                   for comparison in COMPARISONS[key])
    for center, key, (source, target, color, linestyle) in comparisons:
        before, after = lookup[(key, source)], lookup[(key, target)]
        y0, y1 = before["normalized_latency"], after["normalized_latency"]
        if not all(math.isfinite(value) and value > 0 for value in (y0, y1)):
            continue
        x0, x1 = center + variant_positions[source], center + variant_positions[target]
        if source == "DLA":
            label = f"×{before['latency_ms'] / after['latency_ms']:.2f}"
            start, finish = (x0 + 0.06, y0 - 0.015), (x1 - 0.045, y1 + 0.012)
            text_position = (center - 0.17, (y0 + y1) / 2 - (0.02 if key == "SDTRACK" else 0))
            connection = "arc3,rad=0"
        else:
            reduction = 100 * (1 - after["latency_ms"] / before["latency_ms"])
            label = f"−{reduction:.1f}%"
            start, finish = (x0, y0 + 0.012), (x1, y1 + 0.012)
            text_position = (center + 0.07, max(y0, y1) + 0.045)
            connection = "arc3,rad=-0.65"
            if target == "m0+m1":
                start, finish = (x0 - 0.045, y0 + 0.012), (x1 + 0.045, y1 + 0.012)
                connection = "arc3,rad=-0.75"
                text_position = (center + 0.04, max(y0, y1) + 0.065)
        axis.annotate("", xy=finish, xytext=start,
                      arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.3,
                                      linestyle=linestyle, mutation_scale=6 if target == "m0+m1" else 8,
                                      connectionstyle=connection, shrinkA=0, shrinkB=0), zorder=6)
        axis.text(*text_position, label, color=color, ha="left" if source == "DLA" else "center", va="bottom",
                  fontsize=(6.5 if target == "m0+m1" else 7) if compact else VALUE_FONT_SIZE,
                  fontweight="bold", zorder=7,
                  bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    axis.set(xlim=(-0.46, centers[-1] + 0.46), ylim=(0, 1), ylabel="Normalized Latency")
    axis.set_xticks(centers, [label for _, label in groups])
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#B8B8B8", linewidth=0.55, linestyle=(0, (3, 3)))
    axis.axhline(1, color="#222222", linewidth=1.1, linestyle=(0, (4, 2)), zorder=4)
    axis.tick_params(axis="x", length=0, pad=2)
    axis.tick_params(axis="y", length=2, pad=0.5)
    axis.yaxis.labelpad = 0
    if compact:
        axis.tick_params(labelsize=7)
        axis.yaxis.label.set_size(8)
    for tick, label in zip(axis.get_yticks(), axis.get_yticklabels()):
        if tick == 1:
            label.set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    axis.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                borderaxespad=0, handlelength=1.0, handletextpad=0.3,
                columnspacing=0.6 if compact else 0.9, fontsize=7 if compact else FONT_SIZE)


def draw_patterns(axis, rows, compact=False):
    """每根柱子总和为一，显示代表性计数总和，过窄色块不挤入百分比文字。"""
    for position, row in enumerate(rows):
        left = 0.0
        for i, color in enumerate(PATTERN_COLORS):
            fraction = row[f"fraction{i}"]
            if fraction:
                patch = axis.barh(position, fraction, left=left, height=STACK_HEIGHT, color=color,
                                  edgecolor="black", linewidth=0.6, zorder=3)[0]
                patch.set_path_effects([effects.Stroke(linewidth=1.7, foreground="white"), effects.Normal()])
            if fraction >= MIN_LABEL_FRACTION:
                brightness = np.dot(to_rgb(color), [0.2126, 0.7152, 0.0722])
                axis.text(left + fraction / 2, position, f"{fraction:.0%}", ha="center", va="center",
                          color="white" if brightness < 0.48 else "#222222", fontsize=6.5 if compact else FONT_SIZE,
                          fontweight="bold", zorder=5)
            left += fraction
        if not math.isclose(left, 1, abs_tol=1e-12):
            raise ValueError(f"Fractions do not sum to one: {row['network']}")
        axis.add_patch(Rectangle((0, position - STACK_HEIGHT / 2), 1, STACK_HEIGHT,
                                 fill=False, edgecolor="black", linewidth=0.8, zorder=4))
        total = row["total"]
        label = f"{total / 1e6:.2f}M" if total >= 1e6 else f"{total / 1000:.1f}K"
        axis.text(1.025, position, label, ha="left", va="center", fontsize=7 if compact else FONT_SIZE)
    axis.set(xlim=(0, 1.22), ylim=(len(rows) - 0.5, -0.9))
    workload_labels = {network: label for _, label, network, _ in WORKLOADS}
    axis.set_yticks(range(len(rows)), [workload_labels[r["network"]] if compact else r["network"] for r in rows])
    axis.tick_params(axis="y", length=0, pad=2)
    for label in axis.get_yticklabels():
        label.set_fontweight("bold")
        if compact:
            label.set_fontsize(7)
            label.set_linespacing(0.95)
    axis.xaxis.set_visible(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.6,
                     label=n) for c, n in zip(PATTERN_COLORS, PATTERN_NAMES)]
    axis.legend(handles=handles, ncol=3, frameon=False, loc="lower center",
                bbox_to_anchor=(0.44 if compact else 0.48, 0.95 if compact else 0.96),
                borderaxespad=0, prop={"size": 7 if compact else FONT_SIZE, "weight": "bold"}, handlelength=0.8,
                handletextpad=0.3, columnspacing=0.7 if compact else 0.9)


def save_figure(figure, stem):
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(stem.with_suffix("." + suffix), dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)
    print(f"Wrote {stem}.pdf/.png/.svg")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=ROOT / "output/lookahead")
    parser.add_argument("--ablation-log-root", type=Path, default=DIRECTORY)
    parser.add_argument("--frequency-workbook", type=Path, default=ROOT / "output/lookahead/end_to_end.xlsx")
    parser.add_argument("--output-dir", type=Path, default=DIRECTORY / "figures")
    parser.add_argument("--pattern-output-dir", type=Path, default=ROOT / "output/lookahead/pattern_breakdown")
    args = parser.parse_args()
    cycles, original_cycles, status = collect(args.ablation_log_root, WORKLOADS,
                                              load_frequencies(args.frequency_workbook))
    patterns, layers = load_patterns(args.ablation_log_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.pattern_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ablation_status.csv", status)
    for name, rows in (("cycle_reduction_data", original_cycles), ("latency_vs_dla", cycles)):
        write_csv(args.output_dir / f"{name}.csv", rows)
    for name, rows in (("pattern_distribution", patterns), ("pattern_distribution_layers", layers)):
        write_csv(args.pattern_output_dir / f"{name}.csv", rows)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": FONT_SIZE, "axes.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, gridspec_kw={"height_ratios": [1.1, 0.5]})
    figure.subplots_adjust(left=0.13, right=0.995, bottom=0.025, top=0.92, hspace=0.30)
    draw_cycles(axes[0], cycles, compact=True)
    box = axes[1].get_position()
    axes[1].set_position([0.18, box.y0, 0.815, box.height])
    draw_patterns(axes[1], patterns, compact=True)
    for i, axis in enumerate(axes):
        axis.text(-0.14 if i == 0 else -0.21, 1.02, f"({chr(97 + i)})", transform=axis.transAxes,
                  ha="left", va="bottom", fontsize=8, fontweight="bold")
    save_figure(figure, args.output_dir / "fig_ablation_cycles_patterns")
    figure, axis = plt.subplots(figsize=(7.125, 1.85))
    figure.subplots_adjust(left=0.105, right=0.995, bottom=0.12, top=0.86)
    draw_cycles(axis, cycles)
    save_figure(figure, args.output_dir / "fig_latency_vs_dla")
    figure, axis = plt.subplots(figsize=(7.125, 1.35))
    figure.subplots_adjust(left=0.26, right=0.995, bottom=0.02, top=0.80)
    draw_patterns(axis, patterns)
    save_figure(figure, args.pattern_output_dir / "fig_pattern_distribution")


if __name__ == "__main__":
    main()
