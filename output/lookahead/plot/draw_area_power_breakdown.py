"""Draw EVA Engine area and power pie charts from manually entered inputs.

Fill the AREA_MM2 and POWER_MW dictionaries below with values from the latest
DC/Excel report.  The script derives one Core as one Split Unit plus 16 Neuron
Potential Processing Units, then expands this to six Cores at the EVA Engine
top level.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiment_colors import PALETTE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "figures"

# ---------------------------------------------------------------------------
# User-editable architecture inputs.
# Unit: mm^2 for area, mW for power.  Set every value before running.
# Dispatcher should include whichever scheduling/issue/FIFO logic you intend
# to account for at the EVA Engine level.  Accumulator is the global result
# storage/accumulation block, not part of an individual Core.
# ---------------------------------------------------------------------------
NUM_CORES = 6
NUM_NPPU_PER_CORE = 16

AREA_MM2 = {
    "Activation BUFFER": 0.006417,
    "Weight BUFFER": 0.037527,
    # Dispatcher = Scheduler (0.003039) + ISSUE/FIFO (0.002456).
    "Dispatcher": 0.005495,
    "Accumulator": 0.147456,
    "Split Unit": 0.00696686,
    "Neuron Potential Processing Unit": 0.007728,
}

POWER_MW = {
    "Activation BUFFER": 3.43,
    "Weight BUFFER": 21.688,
    # Dispatcher = Scheduler (0.134) + ISSUE/FIFO (0.196).
    "Dispatcher": 0.33,
    "Accumulator": 29.4912,
    "Split Unit": 6.8,
    "Neuron Potential Processing Unit": 1.0456,
}

# 面积和功耗实验使用论文共用的蓝粉配色。
TOP_LEVEL_COLORS = {
    "Activation BUFFER": PALETTE[3],
    "Weight BUFFER": PALETTE[1],
    "Dispatcher": PALETTE[6],
    "6 Cores": PALETTE[2],
    "Accumulator": PALETTE[5],
}
INNER_CORE_COLORS = {
    # 拆分单元的面积比十六个神经元膜电位处理单元小，使用深蓝色便于识别。
    "Splitting Unit": PALETTE[0],
    "Current Accumulating Unit": PALETTE[4],
}

# Small top-level slices use an external label.  Both values are deliberately
# centralized so their placement can be adjusted without touching pie logic.
SMALL_SLICE_THRESHOLD = 0.025
SMALL_SLICE_LABEL_RADIUS = 1.14
SMALL_SLICE_LABEL_STAGGER = 0.13
SMALL_SLICE_DARKEN_FACTOR = 0.72


def pol2xy(radius: float, theta_deg: float) -> tuple[float, float]:
    """Convert polar chart coordinates to Cartesian coordinates."""
    theta = np.deg2rad(theta_deg)
    return radius * np.cos(theta), radius * np.sin(theta)


def validate_inputs(metric_values: dict[str, float | None], unit: str) -> None:
    """Fail early when a user-editable area/power value has not been filled."""
    missing = [name for name, value in metric_values.items() if value is None]
    if missing:
        raise ValueError(f"Fill {unit} values for: {', '.join(missing)}")
    invalid = [name for name, value in metric_values.items() if value is not None and value < 0]
    if invalid:
        raise ValueError(f"{unit} values must be non-negative: {', '.join(invalid)}")


def metric_values(metric: str) -> dict[str, float | str]:
    """Expand user inputs into the EVA Engine hierarchy for one metric."""
    if metric == "area":
        values = AREA_MM2
        unit = "mm$^2$"
        title_prefix = "(a) Area"
    elif metric == "power":
        values = POWER_MW
        unit = "mW"
        title_prefix = "(b) Power"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    validate_inputs(values, metric)
    split_unit = float(values["Split Unit"])
    nppu = float(values["Neuron Potential Processing Unit"])
    one_core = split_unit + NUM_NPPU_PER_CORE * nppu
    six_cores = NUM_CORES * one_core
    total = (
        float(values["Activation BUFFER"])
        + float(values["Weight BUFFER"])
        + float(values["Dispatcher"])
        + float(values["Accumulator"])
        + six_cores
    )
    if total <= 0:
        raise ValueError(f"Total {metric} must be positive")

    return {
        "title": f"{title_prefix}: {total:.3f}{unit}",
        "total": total,
        "activation": float(values["Activation BUFFER"]),
        "weight": float(values["Weight BUFFER"]),
        "dispatcher": float(values["Dispatcher"]),
        "accumulator": float(values["Accumulator"]),
        "one_core": one_core,
        "six_cores": six_cores,
        "split": split_unit,
        "nppu": nppu,
    }


def add_wedge(
    axis,
    theta1: float,
    theta2: float,
    color: str,
    edge: str = "white",
    line_width: float = 0.9,
    zorder: int = 1,
) -> Wedge:
    """Add one filled pie wedge."""
    wedge = Wedge(
        (0, 0),
        1.0,
        theta1,
        theta2,
        facecolor=color,
        edgecolor=edge,
        linewidth=line_width,
        zorder=zorder,
    )
    axis.add_patch(wedge)
    return wedge


def add_radial_line(axis, theta: float, color: str = PALETTE[0]) -> None:
    """Add a radial boundary for the representative Core decomposition."""
    x0, y0 = pol2xy(0.0, theta)
    x1, y1 = pol2xy(1.0, theta)
    axis.plot([x0, x1], [y0, y1], color=color, linewidth=0.8, zorder=8)


def pct_text(value: float, total: float) -> str:
    """Format a component's percentage of total engine area or power."""
    percentage = 100.0 * value / total
    return "<0.1%" if 0 < percentage < 0.1 else f"{percentage:.1f}%"


def darken_color(color: str, factor: float = SMALL_SLICE_DARKEN_FACTOR) -> str:
    """Keep a component's hue while using a deeper shade for a small slice."""
    rgb = np.asarray(to_rgb(color))
    return to_hex(rgb * factor)


def add_pct_label(
    axis,
    theta1: float,
    theta2: float,
    value: float,
    total: float,
    radius: float = 0.62,
    xytext: tuple[float, float] | None = None,
    fontsize: float = 6.4,
    color: str = "black",
) -> None:
    """Place a percentage inside a wedge or at an explicitly chosen position."""
    theta = (theta1 + theta2) / 2
    text = pct_text(value, total)
    if xytext is None:
        x, y = pol2xy(radius, theta)
        axis.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color, zorder=20)
        return
    axis.annotate(
        text,
        xy=pol2xy(0.96, theta),
        xytext=xytext,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        annotation_clip=False,
        zorder=20,
        arrowprops={"arrowstyle": "-", "lw": 0.45, "color": "0.35"},
    )


def add_nearby_external_label(
    axis,
    theta1: float,
    theta2: float,
    value: float,
    total: float,
    stagger_index: int,
) -> None:
    """Place a small-slice label just outside its wedge with a short leader."""
    theta = (theta1 + theta2) / 2
    radius = SMALL_SLICE_LABEL_RADIUS + stagger_index * SMALL_SLICE_LABEL_STAGGER
    xytext = pol2xy(radius, theta)
    horizontal_alignment = "left" if xytext[0] >= 0 else "right"
    axis.annotate(
        pct_text(value, total),
        xy=pol2xy(0.98, theta),
        xytext=xytext,
        ha=horizontal_alignment,
        va="center",
        fontsize=6.4,
        color="black",
        annotation_clip=False,
        zorder=20,
        arrowprops={"arrowstyle": "-", "lw": 0.45, "color": "0.35"},
    )


def add_core_outline(axis, theta1: float, theta2: float) -> None:
    """Use a dashed outline to identify the aggregate six-Core region."""
    outline = Wedge(
        (0, 0),
        1.012,
        theta1,
        theta2,
        facecolor="none",
        edgecolor=PALETTE[0],
        linewidth=1.0,
        linestyle=(0, (2.4, 1.6)),
        zorder=28,
    )
    axis.add_patch(outline)


def draw_architecture_pie(axis, metric: str) -> None:
    """Draw one hierarchical EVA Engine pie for area or power."""
    values = metric_values(metric)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(str(values["title"]), fontsize=8.2, fontweight="bold", pad=1.0)

    # EVA Engine top level: three shared blocks, six equal Cores, and Accumulator.
    top_items = [
        ("Activation BUFFER", float(values["activation"])),
        ("Weight BUFFER", float(values["weight"])),
        ("Dispatcher", float(values["dispatcher"])),
        ("6 Cores", float(values["six_cores"])),
        ("Accumulator", float(values["accumulator"])),
    ]
    total = float(values["total"])
    current_theta = 0.0
    spans: dict[str, tuple[float, float]] = {}
    # Darken proportionally small top-level components while keeping the
    # original hue family.  This improves visual separation from large Cores.
    for name, value in top_items:
        next_theta = current_theta + 360.0 * value / total
        base_color = TOP_LEVEL_COLORS[name]
        color = (
            darken_color(base_color)
            if value / total < SMALL_SLICE_THRESHOLD
            else base_color
        )
        add_wedge(axis, current_theta, next_theta, color)
        spans[name] = (current_theta, next_theta)
        current_theta = next_theta
    axis.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="#303030", linewidth=0.7, zorder=10))

    # Select one Core-sized angular segment from the aggregate six-Core wedge.
    # The selected Core is decomposed into one Split Unit and 16 NPPUs.
    cores_theta1, cores_theta2 = spans["6 Cores"]
    one_core_span = (cores_theta2 - cores_theta1) / NUM_CORES
    representative_theta1 = cores_theta2 - one_core_span
    representative_theta2 = cores_theta2
    split_theta2 = representative_theta1 + one_core_span * float(values["split"]) / float(values["one_core"])
    add_wedge(axis, representative_theta1, split_theta2, INNER_CORE_COLORS["Splitting Unit"], zorder=4)
    add_wedge(
        axis,
        split_theta2,
        representative_theta2,
        INNER_CORE_COLORS["Current Accumulating Unit"],
        zorder=4,
    )
    for theta in (representative_theta1, split_theta2, representative_theta2):
        add_radial_line(axis, theta)
    add_core_outline(axis, cores_theta1, cores_theta2)

    # Top-level percentage labels always use the complete EVA Engine total.
    small_label_index = 0
    for name, value in top_items:
        theta1, theta2 = spans[name]
        if value / total < SMALL_SLICE_THRESHOLD:
            add_nearby_external_label(
                axis,
                theta1,
                theta2,
                value,
                total,
                small_label_index,
            )
            small_label_index += 1
        elif name == "6 Cores":
            add_pct_label(axis, theta1, theta2, value, total, radius=0.63)
        else:
            add_pct_label(
                axis,
                theta1,
                theta2,
                value,
                total,
                radius=0.62,
                color="white" if name == "Activation BUFFER" else "black",
            )

    # Representative-Core labels also use engine total, so their values show
    # the global contribution of one Split Unit or the 16 NPPUs in one Core.
    add_pct_label(
        axis,
        representative_theta1,
        split_theta2,
        float(values["split"]),
        total,
        xytext=(-0.36, -1.16),
    )
    add_pct_label(
        axis,
        split_theta2,
        representative_theta2,
        NUM_NPPU_PER_CORE * float(values["nppu"]),
        total,
        xytext=(0.38, -1.16),
    )
    axis.set_xlim(-1.16, 1.16)
    axis.set_ylim(-1.23, 1.10)


def draw(output_dir: Path = SCRIPT_DIR) -> tuple[Path, Path]:
    """Draw and save the area/power breakdown pair using the manual inputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Times New Roman", "pdf.fonttype": 42, "ps.fonttype": 42})

    fig = plt.figure(figsize=(5.35, 2.3), dpi=300)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.84], wspace=0.10)
    area_axis = fig.add_subplot(grid[0, 0])
    power_axis = fig.add_subplot(grid[0, 1])
    legend_axis = fig.add_subplot(grid[0, 2])
    legend_axis.axis("off")
    draw_architecture_pie(area_axis, "area")
    draw_architecture_pie(power_axis, "power")

    legend_items = [
        ("Activation BUFFER", TOP_LEVEL_COLORS["Activation BUFFER"]),
        ("Weight BUFFER", TOP_LEVEL_COLORS["Weight BUFFER"]),
        ("Dispatcher", TOP_LEVEL_COLORS["Dispatcher"]),
        ("6 Cores", TOP_LEVEL_COLORS["6 Cores"]),
        ("Accumulator", TOP_LEVEL_COLORS["Accumulator"]),
        ("Splitting Unit", INNER_CORE_COLORS["Splitting Unit"]),
        ("Current Accumulating Unit", INNER_CORE_COLORS["Current Accumulating Unit"]),
    ]
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=5.2, markerfacecolor=color, markeredgecolor=color)
        for _, color in legend_items
    ]
    legend_axis.legend(
        handles,
        [name for name, _ in legend_items],
        loc="center left",
        frameon=False,
        fontsize=8,
        handlelength=0.8,
        handletextpad=0.45,
        labelspacing=0.42,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.08)

    png_path = output_dir / "fig_arch_area_power_breakdown.png"
    pdf_path = output_dir / "fig_arch_area_power_breakdown.pdf"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    png, pdf = draw()
    print(f"Saved {png}")
    print(f"Saved {pdf}")
