"""统一更新论文实验配图，从当前日志和工作簿读取数据并同步论文图片。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]


@dataclass(frozen=True)
class FigureTask:
    """保存分项绘图脚本、参数和成功生成后需要同步的论文图片。"""

    name: str
    description: str
    script: str
    arguments: tuple[str, ...]
    paper_files: tuple[tuple[str, str], ...]


# 后续新增或接入的论文绘图脚本必须登记在此处，默认运行会执行全部任务。
# 路径模板由主函数提供，分项脚本继续负责各自的数据计算和绘图样式。
FIGURE_TASKS = (
    FigureTask(
        name="dse_tile_size",
        description="Tile-size scalability normalized to 8x8 tiles",
        script="output/lookahead/plot/draw_tile_size_dse.py",
        arguments=("--raw-root", "{tile_size_root}", "--output-dir", "{tile_size_dir}"),
        paper_files=(("{tile_size_dir}/tile_size_dse.pdf", "2_tile_size_dse.pdf"),),
    ),
    FigureTask(
        name="ablation_study",
        description="Current completed ablation logs and pattern distribution",
        script="output/lookahead/plot/plot_ablation_cycles_patterns.py",
        arguments=("--log-root", "{log_root}",
                   "--ablation-log-root", "{ablation_root}", "--output-dir", "{ablation_root}/figures",
                   "--pattern-output-dir", "{pattern_dir}", "--frequency-workbook", "{workbook}"),
        paper_files=(("{ablation_root}/figures/fig_ablation_cycles_patterns.pdf", "1_mode_reduction.pdf"),),
    ),
    FigureTask(
        name="dse_core_number",
        description="Core-count scalability normalized to six cores",
        script="output/lookahead/plot/draw_core_count_dse.py",
        arguments=("--raw-root", "{core_count_root}", "--output-dir", "{core_count_dir}"),
        paper_files=(("{core_count_dir}/core_count_dse.pdf", "3_core_count_dse.pdf"),),
    ),
    FigureTask(
        name="dse_pu_number",
        description="Estimated processing-unit scalability normalized to sixteen units",
        script="output/lookahead/plot/draw_pu_number.py",
        arguments=("--results-root", "{log_root}", "--output-dir", "{pu_count_dir}"),
        paper_files=(("{pu_count_dir}/figures/pu_number_dse.pdf", "5_pu_number_dse.pdf"),),
    ),
    FigureTask(
        name="dse_combined",
        description="Compact latency panels: (a) SIMD width, (b) core count, (c) tile size",
        script="output/lookahead/plot/draw_dse_combined.py",
        arguments=("--tile-root", "{tile_size_root}", "--core-root", "{core_count_root}",
                   "--pu-root", "{pu_count_dir}", "--log-root", "{log_root}",
                   "--workbook", "{workbook}", "--output-dir", "{figure_dir}"),
        paper_files=(("{figure_dir}/dse_combined.pdf", "dse_combined.pdf"),),
    ),
    FigureTask(
        name="area_power",
        description="Area and power breakdown from the existing component inputs",
        script="output/lookahead/plot/draw_area_power_breakdown.py",
        arguments=(),
        paper_files=(("{figure_dir}/fig_arch_area_power_breakdown.pdf", "5_area_power_breakdown.pdf"),),
    ),
    FigureTask(
        name="cycles_breakdown",
        description="Cycle breakdown",
        script="output/lookahead/plot/extract_cycles_breakdown.py",
        arguments=("--all", "--log-root", "{log_root}", "--output-dir", "{breakdown_dir}",
                   "--figure-dir", "{figure_dir}"),
        paper_files=(("{figure_dir}/EVA_cycles_breakdown.pdf", "EVA_cycles_breakdown.pdf"),),
    ),
    FigureTask(
        name="end_to_end",
        description="End-to-end throughput, normalized latency, and energy efficiency",
        script="output/lookahead/plot/plot_workbook_end_to_end.py",
        arguments=("--input", "{workbook}", "--output-dir", "{figure_dir}",
                   "--paper-dir", "{paper_dir}", "--layout", "{layout}"),
        paper_files=(("{figure_dir}/end_to_end_combined.pdf", "4_end_to_end_combined.pdf"),
                     ("{figure_dir}/end_to_end_combined.pdf", "end_to_end_combined.pdf")),
    ),
)


def run_task(task: FigureTask, paths: dict[str, str]) -> None:
    """调用当前解释器执行分项脚本，成功后检查并同步论文图片。"""
    script = ROOT / task.script
    if not script.is_file():
        raise FileNotFoundError(f"Plot script does not exist: {script}")
    command = [sys.executable, "-X", "utf8", str(script)]
    command.extend(argument.format_map(paths) for argument in task.arguments)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    print(f"\n[RUN] {task.name}: {task.description}", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    for source_template, destination_name in task.paper_files:
        source = Path(source_template.format_map(paths))
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Expected nonempty figure was not generated: {source}")
        destination = Path(paths["paper_dir"]) / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[PAPER] {destination}", flush=True)
    print(f"[DONE] {task.name}", flush=True)


def main() -> int:
    """默认更新所有已登记配图，也可列出任务或只更新指定实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="列出已登记的绘图任务，不生成图片")
    parser.add_argument("--only", nargs="+", choices=[task.name for task in FIGURE_TASKS],
                        help="只更新指定任务；省略时更新全部配图")
    parser.add_argument("--log-root", type=Path, default=DIRECTORY, help="周期分解日志的根目录")
    parser.add_argument("--workbook", type=Path, default=DIRECTORY / "end_to_end.xlsx",
                        help="端到端结果工作簿")
    parser.add_argument("--paper-dir", type=Path,
                        default=ROOT / "paper/hpca2027-latex-template/figures",
                        help="论文图片输出目录")
    parser.add_argument("--layout", choices=("single", "double"), default="single",
                        help="端到端合并图的排版，默认单栏；同时保留两种宽度的对比图")
    args = parser.parse_args()
    if args.list:
        for task in FIGURE_TASKS:
            print(f"{task.name}: {task.description}")
        return 0
    # 交付副本随统一重绘刷新，项目根目录仍是共享配色的唯一编辑来源。
    palette = ROOT / "experiment_colors.py"
    if palette.is_file():
        shutil.copy2(palette, DIRECTORY / "plot/experiment_colors.py")
    paths = {
        "log_root": str(args.log_root.resolve()),
        "workbook": str(args.workbook.resolve()),
        "breakdown_dir": str(DIRECTORY / "cycles_breakdown"),
        "figure_dir": str(DIRECTORY / "figures"),
        "core_count_root": str(ROOT / "output/dse_core_number"),
        "tile_size_root": str(ROOT / "output/dse_tile_size"),
        "tile_size_dir": str(ROOT / "output/dse_tile_size/figures"),
        "core_count_dir": str(ROOT / "output/dse_core_number/figures"),
        "pu_count_dir": str(ROOT / "output/dse_SIMD_Width"),
        "ablation_root": str(DIRECTORY / "ablation_study"),
        "pattern_dir": str(DIRECTORY / "pattern_breakdown"),
        "paper_dir": str(args.paper_dir.resolve()),
        "layout": args.layout,
    }
    tasks = [task for task in FIGURE_TASKS if not args.only or task.name in args.only]
    failures = []
    for task in tasks:
        try:
            run_task(task, paths)
        except (OSError, subprocess.CalledProcessError) as error:
            failures.append(task.name)
            print(f"[FAILED] {task.name}: {error}", file=sys.stderr, flush=True)
    if failures:
        print(f"\nFigure update failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\nUpdated {len(tasks)} figure tasks. Paper figures: {paths['paper_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
