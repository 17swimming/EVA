"""汇总分 bank 前瞻仲裁的计算核心数量实验，并以六个计算核心为性能基准。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NETWORKS = ("TIM_NCARS", "QKFormer_DVSGesture", "SpikeBRGNet_DDD17",
            "SDTrack_FE108", "EMSYOLO_GEN1")
CORE_COUNTS = (6, 8, 10)


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_results(output_root, networks=NETWORKS, cores=CORE_COUNTS, require_complete=True):
    """总周期取运行日志的总体统计，包含最大池化修正；层周期另保留原始结果表之和。

    总周期与六个计算核心的总周期直接比较，所有配置使用同一主频，无需主频换算。
    旧实验的对照加速器加速比与冲突比例不再混入本轮汇总，避免采用不同仲裁规则的数据。
    """
    output_root = Path(output_root).resolve()
    rows, incomplete, coverage = [], [], {}
    for network in networks:
        for count in cores:
            folder = output_root / f"core{count}" / f"{network}_cluster_parallel"
            tables = sorted(folder.glob(f"core{count}_pu16_*_results.csv"))
            logs = sorted(folder.glob("*simulation_log_*.txt"))
            if len(tables) != 1 or len(logs) != 1:
                incomplete.append((network, count))
                continue
            log_text = logs[0].read_text(encoding="utf-8")
            totals = re.findall(r"^总EVA周期:\s*([\d,]+(?:\.\d+)?)", log_text, re.MULTILINE)
            if not totals or "仿真任务完成！" not in log_text:
                incomplete.append((network, count))
                continue
            with tables[0].open(encoding="utf-8", newline="") as stream:
                layers = list(csv.DictReader(stream))
            # 不同计算核心数量必须处理相同层及输入形状，才能公平比较总体周期。
            layer_keys = tuple((layer.get("test_case"), layer.get("file"),
                                layer.get("type"), layer.get("input_shape")) for layer in layers)
            if network in coverage and coverage[network] != layer_keys:
                raise RuntimeError(f"Unequal workload coverage: {network}, cores={count}")
            coverage[network] = layer_keys
            rows.append({
                "network": network, "num_cores": count, "num_pus": 16,
                "layers": len(layers),
                "eva_total_cycles": float(totals[-1].replace(",", "")),
                "layer_total_cycles_sum": sum(float(row["simulator_cycles"]) for row in layers),
                "speedup_vs_core6": "",
                "prefetch_cycles": 2, "retire_packets_per_cycle": 2,
                "psum_pool_rows": 8, "num_split": 2, "bank_h": 12, "bank_w": 12,
                "fetch_rows": 8, "split_fifo_depth": 4, "retire_column": 3,
                "num_reduce_trees": 0, "enabled_modes": "0,1,2",
                "skipped_cases": len(re.findall(r"=== 跳过测试用例", log_text)),
                "missing_input_cases": len(re.findall(r"警告：文件不存在，跳过测试用例", log_text)),
                "unsupported_cases": len(re.findall(r"暂时不支持的type", log_text)),
                "raw_csv": str(tables[0].relative_to(output_root)),
                "raw_log": str(logs[0].relative_to(output_root)),
            })
    baselines = {row["network"]: row["eva_total_cycles"] for row in rows if row["num_cores"] == 6}
    for row in rows:
        if row["network"] in baselines and row["eva_total_cycles"] > 0:
            row["speedup_vs_core6"] = baselines[row["network"]] / row["eva_total_cycles"]
    if rows:
        write_table(output_root / "core_count_dse_summary.csv", rows)
        aggregate = []
        for count in cores:
            values = [row["speedup_vs_core6"] for row in rows
                      if row["num_cores"] == count and row["speedup_vs_core6"] != ""]
            if len(values) == len(networks):
                aggregate.append({"num_cores": count,
                                  "speedup_vs_core6_gmean": math.exp(sum(math.log(v) for v in values) / len(values)),
                                  "workloads": len(values)})
        if aggregate:
            write_table(output_root / "core_count_dse_aggregate_summary.csv", aggregate)
        print(f"Summary: {output_root / 'core_count_dse_summary.csv'}", flush=True)
    if incomplete:
        message = f"Incomplete experiment configurations: {incomplete}"
        if require_complete:
            raise RuntimeError(message)
        print(message, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="汇总计算核心数量设计空间探索结果")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "output/dse_core_number")
    parser.add_argument("--networks", nargs="+", choices=NETWORKS, default=list(NETWORKS))
    parser.add_argument("--cores", nargs="+", type=int, choices=CORE_COUNTS, default=list(CORE_COUNTS))
    args = parser.parse_args()
    collect_results(args.raw_root, args.networks, args.cores)


if __name__ == "__main__":
    main()
