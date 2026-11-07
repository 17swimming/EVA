"""读取最新完整消融日志，保留未完成配置的缺失状态和数据来源。"""
import csv
import math
from pathlib import Path
import re

VARIANT_FOLDERS = {"m0": "R", "m0+m1": "R_S", "m0_m2": "R_C", "m0+m1+m2": "NVDLA+R_C_S"}
MODES = {"m0": {0}, "m0+m1": {0, 1}, "m0_m2": {0, 2}, "m0+m1+m2": {0, 1, 2}}


def completed_result(root, folder_name, network):
    """支持直接配置目录及带时间戳的运行目录，只选择日志末尾完成的结果。"""
    candidates = []
    for path in root.rglob(f"{network}_cluster_parallel/*simulation_log_*.txt"):
        if path.parent.parent.name != folder_name:
            continue
        text = path.read_text(encoding="utf-8-sig")
        total = re.findall(r"^总EVA周期:\s*([\d,]+(?:\.\d+)?)", text, re.MULTILINE)
        finished = re.findall(r"\[([^\]]+)\]\s*仿真任务完成！", text)
        if not total or not finished or "Traceback" in text:
            continue
        tables = list(path.parent.glob("core6_pu16_*_results.csv"))
        if len(tables) != 1:
            raise ValueError(f"Expected one layer table for completed run: {path}")
        with tables[0].open(encoding="utf-8-sig", newline="") as stream:
            layers = list(csv.DictReader(stream))
        if not layers:
            raise ValueError(f"Empty completed table: {tables[0]}")
        coverage = tuple((r["test_case"], r["file"].replace("\\", "/").rsplit("/", 1)[-1],
                          r["type"], r["input_shape"]) for r in layers)
        value = float(total[-1].replace(",", ""))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid completed total: {path}")
        candidates.append(dict(timestamp=finished[-1], path=path, table=tables[0], text=text,
                               cycles=value, coverage=coverage))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["timestamp"], item["path"].as_posix()))
    newest = candidates[-1]
    peers = [item for item in candidates if item["timestamp"] == newest["timestamp"]]
    if any(item["cycles"] != newest["cycles"] or item["coverage"] != newest["coverage"] for item in peers):
        raise ValueError(f"Conflicting runs at the same completion time: {network}, {folder_name}")
    return newest


def collect(root, workloads, frequencies):
    root = Path(root).resolve()
    rows, reductions, status = [], [], []
    for key, _, network, _ in workloads:
        reference = completed_result(root, "NVDLA+R_C_S", network)
        if reference is None:
            raise ValueError(f"Missing completed full-configuration reference: {network}")
        totals = re.findall(r"^总DLA周期:\s*([\d,]+(?:\.\d+)?)", reference["text"], re.MULTILINE)
        if not totals or float(totals[-1].replace(",", "")) <= 0:
            raise ValueError(f"Missing positive reference cycles: {reference['path']}")
        baseline = float(totals[-1].replace(",", ""))
        runs = {variant: completed_result(root, folder, network) for variant, folder in VARIANT_FOLDERS.items()}
        for variant, result in runs.items():
            if result is None:
                print(f"[PENDING] {network} {VARIANT_FOLDERS[variant]}: no completed result")
            else:
                if result["coverage"] != reference["coverage"]:
                    raise ValueError(f"Unequal workload coverage: {network}, {variant}")
                match = re.search(r"mode\s*\(([^)]*)\)", result["text"])
                if not match or set(map(int, re.findall(r"\d+", match[1]))) != MODES[variant]:
                    raise ValueError(f"Unexpected pattern configuration: {result['path']}")
            status.append(dict(network=network, configuration=VARIANT_FOLDERS[variant],
                               status="complete" if result else "pending",
                               source_log=str(result["path"].relative_to(root)) if result else ""))
        for variant in ("DLA", *VARIANT_FOLDERS):
            result = reference if variant == "DLA" else runs[variant]
            value = baseline if variant == "DLA" else result["cycles"] if result else math.nan
            frequency = frequencies["NVDLA" if variant == "DLA" else "EVA"]
            latency = value / (frequency * 1000)
            baseline_latency = baseline / (frequencies["NVDLA"] * 1000)
            rows.append(dict(workload=key, variant=variant, cycles=value, dla_cycles=baseline,
                             normalized_cycles=value / baseline, cycle_speedup=baseline / value,
                             frequency_mhz=frequency, dla_frequency_mhz=frequencies["NVDLA"],
                             latency_ms=latency, dla_latency_ms=baseline_latency,
                             normalized_latency=latency / baseline_latency, latency_speedup=baseline_latency / latency,
                             status="complete" if result else "pending",
                             source_logs=str(result["path"].relative_to(root)) if result else "",
                             baseline_log=str(reference["path"].relative_to(root))))
        regular = runs["m0"]["cycles"] if runs["m0"] else math.nan
        for variant, result in runs.items():
            value = result["cycles"] if result else math.nan
            reductions.append(dict(workload=key, variant=variant, cycles=value,
                                   normalized_cycles=value / regular, reduction_percent=(1 - value / regular) * 100))
    all_complete = all(row["status"] == "complete" for row in status)
    for variant in ("DLA", *VARIANT_FOLDERS):
        selected = [r for r in rows if r["variant"] == variant]
        row = dict(selected[0], workload="GMEAN", cycles="", dla_cycles="", latency_ms="", dla_latency_ms="",
                   source_logs="", baseline_log="", status="complete" if all_complete else "pending")
        for field in ("normalized_cycles", "cycle_speedup", "normalized_latency", "latency_speedup"):
            row[field] = math.exp(sum(math.log(r[field]) for r in selected) / len(selected)) if all_complete else math.nan
        rows.append(row)
    for variant in VARIANT_FOLDERS:
        values = [r["normalized_cycles"] for r in reductions if r["variant"] == variant]
        value = math.exp(sum(math.log(v) for v in values) / len(values)) if all_complete else math.nan
        reductions.append(dict(workload="GMEAN", variant=variant, cycles="", normalized_cycles=value,
                               reduction_percent=(1 - value) * 100))
    return rows, reductions, status
