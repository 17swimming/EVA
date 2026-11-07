"""按实际输出通道分组数，估算处理单元数量变化后的端到端总周期。"""

from __future__ import annotations

import argparse
import ast
import csv
from fractions import Fraction
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
NETWORKS = ("TIM_NCARS", "QKFormer_DVSGesture", "SpikeBRGNet_DDD17",
            "SDTrack_FE108", "EMSYOLO_GEN1")
LABELS = ("TIM", "QKFormer", "BRGNet", "SDTrack", "EMSYOLO")
PU_COUNTS = (16, 32, 64)
CONV_TYPES = {"conv2d_(3, 3)_(1, 1)_(1, 1)",
              "conv2d_(1, 3)_(1, 1)_(0, 1)",
              "conv2d_(3, 1)_(1, 1)_(1, 0)", "conv1d_3_1"}
FC_TYPES = {"conv2d_(3, 3)_(2, 2)_(1, 1)",
            "conv2d_(1, 1)_(1, 1)_(0, 0)", "conv1d_1_1", "conv1d_5_1"}
ATTENTION_TYPES = {"SSA", "Spiking_Self_Attention"}


def read_csv(path):
    """保留层编号，以便与运行器的原始网络清单逐项核对。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def basename(value):
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def mapping_for(row, manifest_row):
    """分类依据运行器实际调用的方法，不根据卷积名称猜测执行路径。"""
    operator = row["type"]
    clean_type, separator, suffix = operator.partition("_g_")
    groups = int(suffix) if separator else 1
    if clean_type in ATTENTION_TYPES:
        # 运行器的两个矩阵乘法都以每个注意力头的维度作为输出通道数。
        shape = ast.literal_eval(row["input_shape"].removeprefix("torch.Size(").removesuffix(")"))
        if len(shape) != 5:
            raise ValueError(f"Unexpected attention shape: {shape}")
        return "run_fc_attention", int(shape[-1]), 1
    cout = int(manifest_row["Cout/dim"])
    if clean_type in CONV_TYPES:
        if cout % groups:
            raise ValueError(f"Output channels not divisible by convolution groups: {operator}/{cout}")
        return "run_convolution", cout, groups
    if clean_type in FC_TYPES:
        # 此路径的运行器传入完整权重矩阵，没有按原卷积分组再次调用。
        return "run_fc", cout, 1
    raise ValueError(f"Unsupported executed operator: {operator}")


def output_groups(mapping, cout, groups, pus):
    """所有配置保持六个计算核心，只改变处理单元数量。"""
    if mapping == "run_convolution":
        return groups * ceil_div(cout // groups, pus)
    return ceil_div(cout, pus * 3)


def load_source(network, results_root, manifests_root):
    """只读取完成的六个计算核心、十六个处理单元实验，排除未完成日志。"""
    folder = results_root / f"{network}_cluster_parallel"
    tables = list(folder.glob("core6_pu16_*_results.csv"))
    if len(tables) != 1:
        raise ValueError(f"Expected one source table in {folder}, found {len(tables)}")
    completed = []
    for path in folder.glob("*simulation_log_*.txt"):
        text = path.read_text(encoding="utf-8-sig")
        totals = re.findall(r"^总EVA周期:\s*([\d,]+(?:\.\d+)?)", text, re.MULTILINE)
        if totals and "仿真任务完成！" in text:
            completed.append((path, Fraction(totals[-1].replace(",", ""))))
    if len(completed) != 1:
        raise ValueError(f"Expected one completed source log in {folder}, found {len(completed)}")
    log, total = completed[0]
    if "cores6_pus16_bank12x12" not in log.name:
        raise ValueError(f"Unexpected source configuration: {log}")
    manifest = read_csv(manifests_root / f"{network}.csv")
    layers, seen = [], set()
    for row in read_csv(tables[0]):
        case = int(row["test_case"])
        if case in seen or not 1 <= case <= len(manifest):
            raise ValueError(f"Invalid or duplicate layer: {tables[0]} / {case}")
        seen.add(case)
        source = manifest[case - 1]
        if row["type"] != source["type"] or basename(row["file"]) != basename(source["file"]):
            raise ValueError(f"Layer does not match manifest: {network} / {case}")
        mapping, cout, groups = mapping_for(row, source)
        layers.append((row, mapping, cout, groups))
    if not layers:
        raise ValueError(f"Empty source table: {tables[0]}")
    layer_sum = sum(Fraction(row["simulator_cycles"]) for row, *_ in layers)
    correction = total - layer_sum
    # 当前只有该网络在逐层表格之外保留最大池化访存修正，增加处理单元不会改变它。
    if correction and network != "TIM_NCARS":
        raise ValueError(f"Unexplained network correction: {network} / {correction}")
    return layers, total, correction, tables[0], log


def number(value):
    """比例估算保留小数，不将每层周期提前取整后再累加。"""
    if isinstance(value, Fraction):
        return value.numerator if value.denominator == 1 else float(value)
    return value


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: number(value) for key, value in row.items()} for row in rows)


def collect(results_root, output_dir, manifests_root):
    """每层总周期乘以新旧分组数之比，再逐层相加得到网络总周期。"""
    summary, details, reasons = [], [], []
    for network in NETWORKS:
        layers, base_total, correction, table, log = load_source(network, results_root, manifests_root)
        base_layer_sum = base_total - correction
        for pus in PU_COUNTS:
            network_rows = []
            for row, mapping, cout, groups in layers:
                old_groups = output_groups(mapping, cout, groups, 16)
                new_groups = output_groups(mapping, cout, groups, pus)
                scale = Fraction(new_groups, old_groups)
                base_cycles = Fraction(row["simulator_cycles"])
                channels_per_group = cout // groups if mapping == "run_convolution" else cout
                group_width = pus if mapping == "run_convolution" else pus * 3
                record = {
                    "network": network, "test_case": int(row["test_case"]),
                    "file": basename(row["file"]), "type": row["type"],
                    "input_shape": row["input_shape"], "execution_path": mapping,
                    "Cout": cout, "convolution_groups": groups,
                    "num_cores": 6, "num_pus": pus,
                    "base_output_groups": old_groups, "output_groups": new_groups,
                    "scale_factor": scale, "base_total_cycles": base_cycles,
                    "total_cycles": base_cycles * scale,
                    "speedup_vs_pu16": 1 / scale,
                    "output_channel_utilization": Fraction(channels_per_group, ceil_div(channels_per_group, group_width) * group_width),
                    "cycles_saved": base_cycles * (1 - scale),
                }
                network_rows.append(record)
            details.extend(network_rows)
            layer_sum = sum(row["total_cycles"] for row in network_rows)
            total = layer_sum + correction
            if pus == 16 and total != base_total:
                raise AssertionError(f"Base configuration mismatch: {network}")
            if not 0 < total <= base_total:
                raise ValueError(f"Invalid scaled total: {network}/{pus}/{total}")
            unchanged = [row for row in network_rows if row["output_groups"] == row["base_output_groups"]]
            unchanged_cycles = sum(row["base_total_cycles"] for row in unchanged)
            summary.append({
                "network": network, "num_cores": 6, "num_pus": pus,
                "layers": len(layers), "base_total_cycles": base_total,
                "layer_total_cycles_sum": layer_sum,
                "network_correction_cycles": correction, "total_cycles": total,
                "speedup_vs_pu16": base_total / total,
                "ideal_speedup_vs_pu16": Fraction(pus, 16),
                "scaling_efficiency": base_total / total / Fraction(pus, 16),
                "unchanged_layers": len(unchanged),
                "unchanged_base_cycles": unchanged_cycles,
                "unchanged_base_cycles_ratio": unchanged_cycles / base_layer_sum,
                "raw_csv": table.relative_to(ROOT).as_posix() if table.is_relative_to(ROOT) else str(table),
                "raw_log": log.relative_to(ROOT).as_posix() if log.is_relative_to(ROOT) else str(log),
            })
            # 按实际执行路径、输出通道数和分组变化汇总，定位无法同比例加速的层。
            buckets = {}
            for row in network_rows:
                key = (row["execution_path"], row["Cout"], row["convolution_groups"],
                       row["base_output_groups"], row["output_groups"])
                bucket = buckets.setdefault(key, {"layers": 0, "base_cycles": Fraction(0), "cycles": Fraction(0)})
                bucket["layers"] += 1
                bucket["base_cycles"] += row["base_total_cycles"]
                bucket["cycles"] += row["total_cycles"]
            for (mapping, cout, groups, old_groups, new_groups), bucket in sorted(buckets.items()):
                reasons.append({
                    "network": network, "num_pus": pus, "execution_path": mapping,
                    "Cout": cout, "convolution_groups": groups,
                    "base_output_groups": old_groups, "output_groups": new_groups,
                    "layers": bucket["layers"], "base_total_cycles": bucket["base_cycles"],
                    "total_cycles": bucket["cycles"], "speedup_vs_pu16": Fraction(old_groups, new_groups),
                    "base_cycles_ratio": bucket["base_cycles"] / base_layer_sum,
                    "cycles_saved": bucket["base_cycles"] - bucket["cycles"],
                })
    write_csv(output_dir / "pu_number_summary.csv", summary)
    write_csv(output_dir / "pu_number_layers.csv", details)
    write_csv(output_dir / "pu_number_scaling_reasons.csv", reasons)
    write_analysis(output_dir, summary, details)
    return summary


def write_analysis(output_dir, summary, details):
    """记录分组取整、单组下限及各层周期权重对端到端加速比的影响。"""
    lines = ["# 处理单元数量比例估算", "",
             "基于当前六个计算核心、十六个处理单元的端到端结果，所有配置采用相同主频。",
             "处理单元数量为 32、64 时的结果是按分组数推算的估计值。", "",
             "## 计算口径", "",
             "- `run_convolution`：`groups × ceil((Cout / groups) / num_pus)`。",
             "- `run_fc`：`ceil(Cout / (num_pus × 3))`。",
             "- 注意力层：当前运行器的两个矩阵乘法都使用每个注意力头的维度作为 `Cout`，因此可对整层采用同一比例。",
             "- `新 total_cycles = 原 total_cycles × 新分组数 / 原分组数`。整层总周期一起缩放，包含未隐藏神经元计算和片外访存，不重新计算重叠。",
             "- 网络周期为逐层估算之和，加上原日志的网络修正。TIM 的池化修正保持原值。",
             "- 使用精确分数累加，导出时保留小数；没有将每层周期提前取整。", "",
             "## 汇总", "",
             "| 网络 | 十六个处理单元周期 | 三十二个处理单元周期 | 加速比 | 六十四个处理单元周期 | 加速比 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for network, label in zip(NETWORKS, LABELS):
        records = {row["num_pus"]: row for row in summary if row["network"] == network}
        lines.append(f"| {label} | {float(records[16]['total_cycles']):,.0f} | {float(records[32]['total_cycles']):,.3f} | "
                     f"{float(records[32]['speedup_vs_pu16']):.3f}× | {float(records[64]['total_cycles']):,.3f} | "
                     f"{float(records[64]['speedup_vs_pu16']):.3f}× |")
    lines.extend(["", "## 加速比变化的原因", "",
                  "增加处理单元只有在输出通道分组数减少时才会缩短估算周期。分组数向上取整，最低为一组。",
                  "例如全连接路径的输出通道数为 64 时，三种配置的分组数为 2、1、1，所以从十六增加到三十二个处理单元可以获得两倍层加速，从三十二继续增加到六十四则不再加速。",
                  "输出通道数为 128 时，全连接路径的分组数为 3、2、1，对应层加速比为 1、1.5、3，低于处理单元数量的两倍和四倍比例。",
                  "端到端加速比按原层周期的权重累加得到，不能对各层加速比直接求算术平均。", ""])
    for network, label in zip(NETWORKS, LABELS):
        lines.extend([f"### {label}", ""])
        for pus in (32, 64):
            record = next(row for row in summary if row["network"] == network and row["num_pus"] == pus)
            lines.append(f"- {pus} 个处理单元：加速比 {float(record['speedup_vs_pu16']):.3f}×。"
                         f"分组数不变的 {record['unchanged_layers']} 层占原逐层周期的 "
                         f"{float(record['unchanged_base_cycles_ratio']):.2%}。")
        selected = [row for row in details if row["network"] == network and row["num_pus"] == 64]
        previous = {row["test_case"]: row for row in details if row["network"] == network and row["num_pus"] == 32}
        saturated = [row for row in selected if row["output_groups"] == previous[row["test_case"]]["output_groups"]]
        saturated_cycles = sum(previous[row["test_case"]]["total_cycles"] for row in saturated)
        previous_summary = next(row for row in summary if row["network"] == network and row["num_pus"] == 32)
        current_summary = next(row for row in summary if row["network"] == network and row["num_pus"] == 64)
        lines.append(f"- 从三十二增加到六十四个处理单元，{len(saturated)} 层的分组数不再减少，"
                     f"占三十二个处理单元逐层周期的 {float(saturated_cycles / previous_summary['layer_total_cycles_sum']):.2%}。"
                     f"这一阶段的端到端加速比为 {float(previous_summary['total_cycles'] / current_summary['total_cycles']):.3f}×。")
        stalled = sorted((row for row in selected if row["scale_factor"] == 1),
                         key=lambda row: row["base_total_cycles"], reverse=True)[:3]
        if stalled:
            lines.append("- 六十四个处理单元时，周期较多且没有缩短的层：" + "；".join(
                f"第 {row['test_case']} 层 `{row['type']}`，输出通道 {row['Cout']}，原周期 {float(row['base_total_cycles']):,.0f}"
                for row in stalled) + "。")
        losses = sorted(selected, key=lambda row: row["total_cycles"] - row["base_total_cycles"] / 4, reverse=True)[:3]
        lines.append("- 相对理想四倍加速，周期差额最大的层：" + "；".join(
            f"第 {row['test_case']} 层，分组数 {row['base_output_groups']}→{row['output_groups']}，差额 "
            f"{float(row['total_cycles'] - row['base_total_cycles'] / 4):,.3f} 拍"
            for row in losses) + "。")
        correction = next(row["network_correction_cycles"] for row in summary if row["network"] == network)
        if correction:
            lines.append(f"- 原日志的池化修正为 {float(correction):g} 拍，所有处理单元数量配置保留这一修正。")
        lines.append("")
    lines.extend(["## 文件与重现", "",
                  "- `pu_number_summary.csv`：五个网络、三种处理单元数量的总周期与加速比。",
                  "- `pu_number_layers.csv`：每层的实际执行路径、通道数、分组数、缩放比例和周期。",
                  "- `pu_number_scaling_reasons.csv`：按通道数、执行路径和分组变化汇总周期权重。", "",
                  "```powershell", "python output/dse_SIMD_Width/scale_pu_number.py",
                  "python output/lookahead/plot_all_figures.py --only dse_pu_number", "```", "",
                  "独立绘图入口为 `output/dse_SIMD_Width/draw_pu_number.py`，每次重新提取原始结果并生成上述表格。",
                  "图片保存为 `figures/pu_number_dse.{pdf,png,svg}`。统一绘图入口同步至论文的 `5_pu_number_dse.pdf`。", "",
                  "此估算采用用户指定的每组周期相同假设，未重新模拟处理单元数量变化后的退休带宽、访存或重叠变化。"])
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "output/lookahead")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/dse_SIMD_Width")
    parser.add_argument("--manifests-root", type=Path, default=Path(__file__).resolve().parent / "manifests")
    args = parser.parse_args()
    rows = collect(args.results_root.resolve(), args.output_dir.resolve(), args.manifests_root.resolve())
    for row in rows:
        print(f"{row['network']:25s} num_pus={row['num_pus']:2d}: "
              f"{float(row['total_cycles']):14,.3f} cycles, {float(row['speedup_vs_pu16']):.4f}x")


if __name__ == "__main__":
    main()
