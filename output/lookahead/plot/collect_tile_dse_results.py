from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re


THIS_DIR = Path(__file__).resolve().parent
NETWORKS = [
    "TIM_NCARS",
    "QKFormer_DVSGesture",
    "SpikeET_FE108",
    "SDTrack_FE108",
    "SpikeBRGNet_DDD17",
    "EMSYOLO_GEN1",
]
TILES = [(8, 8), (12, 12), (16, 16), (24, 24), (32, 32)]
FIELDS = [
    "network",
    "tile_h",
    "tile_w",
    "layers",
    "eva_cycles",
    "dla_cycles",
    "eva_over_dla_speedup",
    "raw_csv",
]


def collect_lookahead_results(output_root, networks, tiles, *, require_complete=True, require_no_baselines=True):
    """汇总当前前瞻仲裁实验；保留旧汇总入口，不混用旧模型或旧工作负载的结果。

    总周期取日志最终统计，包含最大池化周期修正；逐层结果另行求和。
    不同分块尺寸必须完成相同用例、输入文件、算子类型及输入形状，才计算相对性能。
    """
    output_root = Path(output_root).resolve()
    rows, incomplete, coverage = [], [], {}
    for network in networks:
        for tile in tiles:
            folder = output_root / f"tile{tile}x{tile}" / f"{network}_cluster_parallel"
            tables = list(folder.glob("core6_pu16_*_results.csv"))
            logs = list(folder.glob("*simulation_log_*.txt"))
            logs = [path for path in logs if "仿真任务完成！" in path.read_text(encoding="utf-8-sig")]
            if len(tables) != 1 or len(logs) != 1:
                incomplete.append((network, tile))
                continue
            log = logs[0].read_text(encoding="utf-8")
            totals = re.findall(r"^总EVA周期:\s*([\d,]+(?:\.\d+)?)", log, re.MULTILINE)
            if not totals or "仿真任务完成！" not in log or "Traceback" in log:
                incomplete.append((network, tile))
                continue
            with tables[0].open(encoding="utf-8", newline="") as stream:
                layers = list(csv.DictReader(stream))
            if not layers:
                raise RuntimeError(f"No completed layers: {network}, tile={tile}")
            # 跨机器结果以输入文件名识别同一层，避免绝对路径的目录前缀影响覆盖核对。
            layer_keys = tuple((layer.get("test_case"),
                                (layer.get("file") or "").replace("\\", "/").rsplit("/", 1)[-1],
                                layer.get("type"), layer.get("input_shape")) for layer in layers)
            if network in coverage and coverage[network] != layer_keys:
                raise RuntimeError(f"Unequal workload coverage: {network}, tile={tile}")
            coverage[network] = layer_keys
            # 未执行的基线周期必须为非数值，不能把零周期误解释为执行结果。
            for layer in layers:
                for field in ("dla_cycles", "prosperity_cycles", "gustav_cycles", "ptb_cycles"):
                    if require_no_baselines and not math.isnan(float(layer[field])):
                        raise RuntimeError(f"Baseline model executed: {network}, tile={tile}, {field}")
            missing = len(re.findall(r"警告：文件不存在，跳过测试用例", log))
            if missing:
                raise RuntimeError(f"Missing workload inputs: {network}, tile={tile}, count={missing}")
            def column_sum(field):
                return sum(float(layer[field]) for layer in layers if layer.get(field))
            rows.append({
                "network": network, "tile_h": tile, "tile_w": tile,
                "num_cores": 6, "num_pus": 16, "layers": len(layers),
                "eva_total_cycles": float(totals[-1].replace(",", "")),
                "layer_total_cycles_sum": column_sum("simulator_cycles"),
                "eva_compute_cycles_sum": column_sum("EVA"),
                "lif_exposed_cycles_sum": column_sum("LIF"),
                "dram_exposed_cycles_sum": column_sum("DRAM"),
                "speedup_vs_tile8": "", "retire_packets_per_cycle": 2,
                "prefetch_cycles": 2, "num_split": 2, "fetch_rows": 8,
                "split_fifo_depth": 4, "psum_pool_rows": 8, "retire_column": 3,
                "num_reduce_trees": 0, "enabled_modes": "0,1,2",
                "skipped_cases": len(re.findall(r"=== 跳过测试用例", log)),
                "unsupported_cases": len(re.findall(r"暂时不支持的type", log)),
                "missing_input_cases": missing,
                "raw_csv": str(tables[0].relative_to(output_root)),
                "raw_log": str(logs[0].relative_to(output_root)),
            })
    references = {row["network"]: row["eva_total_cycles"] for row in rows if row["tile_h"] == 8}
    for row in rows:
        if row["network"] in references and row["eva_total_cycles"] > 0:
            row["speedup_vs_tile8"] = references[row["network"]] / row["eva_total_cycles"]
    if rows:
        output = output_root / "tile_size_dse_summary.csv"
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary: {output}", flush=True)
    if incomplete:
        message = f"Incomplete experiment configurations: {incomplete}"
        if require_complete:
            raise RuntimeError(message)
        print(message, flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect tile-size DSE raw runner CSVs.")
    parser.add_argument("--raw-root", type=Path, default=THIS_DIR / "results" / "raw")
    parser.add_argument("--output", type=Path, default=THIS_DIR / "results" / "tile_dse_cycles_summary.csv")
    return parser.parse_args()


def latest_result_csv(raw_root: Path, network: str, tile_h: int, tile_w: int) -> Path | None:
    folder = raw_root / f"{network}_tile{tile_h}x{tile_w}_cluster"
    if not folder.exists():
        return None
    csvs = sorted(folder.glob("*_results.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def sum_cycles(path: Path) -> tuple[int, float, float]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    eva_cycles = sum(float(row.get("simulator_cycles") or 0.0) for row in rows)
    dla_cycles = sum(float(row.get("dla_cycles") or 0.0) for row in rows)
    return len(rows), eva_cycles, dla_cycles


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for network in NETWORKS:
        for tile_h, tile_w in TILES:
            raw_csv = latest_result_csv(args.raw_root, network, tile_h, tile_w)
            if raw_csv is None:
                rows.append({
                    "network": network,
                    "tile_h": tile_h,
                    "tile_w": tile_w,
                    "layers": 0,
                    "eva_cycles": "",
                    "dla_cycles": "",
                    "eva_over_dla_speedup": "",
                    "raw_csv": "",
                })
                continue
            layers, eva_cycles, dla_cycles = sum_cycles(raw_csv)
            rows.append({
                "network": network,
                "tile_h": tile_h,
                "tile_w": tile_w,
                "layers": layers,
                "eva_cycles": eva_cycles,
                "dla_cycles": dla_cycles,
                # EVA speedup over DLA: values above 1 mean EVA is faster.
                "eva_over_dla_speedup": dla_cycles / eva_cycles if eva_cycles else "",
                "raw_csv": str(raw_csv.relative_to(THIS_DIR)),
            })

    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
