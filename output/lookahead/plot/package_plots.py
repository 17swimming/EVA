"""打包论文绘图入口、实现、配置与原始输入，保留可重绘的项目相对路径。"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/lookahead/plot_handoff.zip")
    args = parser.parse_args()
    paths = {ROOT / "experiment_colors.py", ROOT / "output/lookahead/plot_all_figures.py",
             ROOT / "output/lookahead/README.md", ROOT / "output/lookahead/end_to_end.xlsx"}
    for path in Path(__file__).resolve().parent.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            paths.add(path)
    for directory in (ROOT / "output/dse_core_number", ROOT / "output/dse_tile_size",
                      ROOT / "output/lookahead/ablation_study"):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in (".csv", ".txt") and path.parent.name.endswith("_cluster_parallel"):
                paths.add(path)
    for directory in (ROOT / "output/lookahead").glob("*_cluster_parallel"):
        paths.update(path for path in directory.iterdir() if path.is_file() and path.suffix in (".csv", ".txt"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory = []
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            archive.writestr(relative, data)
            inventory.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        archive.writestr("handoff_manifest.json", json.dumps(inventory, ensure_ascii=False, indent=2))
        archive.writestr("README.txt", "See output/lookahead/plot/README.md.\n"
                         "Install: python -m pip install -r output/lookahead/plot/requirements.txt\n"
                         "Run: python output/lookahead/plot_all_figures.py\n")
    print(f"Packaged {len(inventory)} files: {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
