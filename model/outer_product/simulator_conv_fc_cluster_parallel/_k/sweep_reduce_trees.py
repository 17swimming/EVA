"""Sweep shared reduce trees on stored TIM_NCARS inputs.

Run from the repository root:
python -B -X utf8 -m model.outer_product.simulator_conv_fc_cluster_parallel.sweep_reduce_trees --output output/reduce_tree_tim

Mappings and input shapes follow simulator_runner.py's active TIM paths.
"""
import argparse
import collections
import concurrent.futures
import contextlib
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from model.utils import img2col
from .simulator_real_compute import OutProductSimulator

ROOT = Path(__file__).resolve().parents[3]


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_budget(budget, config, output):
    torch.set_num_threads(1)
    label = 'unlimited' if budget is None else str(budget)
    folder = Path(output) / label
    folder.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'model/test_outerproduct_simulator/TIM_NCARS.csv').open(encoding='utf-8') as file:
        cases = list(csv.DictReader(file))
    rows, skipped, hist_rows = [], [], []
    cache = {}
    for case_id, case in enumerate(cases, 1):
        kind = case['type'].strip()
        if '_float' in case['spike'].lower() or kind == 'SSA':
            skipped.append({'case': case_id, 'type': kind, 'reason': 'Excluded by active regular runner paths'})
            continue
        key = (case['file'], kind, case['Cout/dim'])
        if key in cache:
            old_row, old_hist = cache[key]
            rows.append(dict(old_row, case=case_id, reused_case=old_row['case']))
            hist_rows.extend(dict(h, case=case_id) for h in old_hist)
            continue
        x = torch.from_numpy(np.load(ROOT / case['file'])).to(torch.int8)
        cout = int(case['Cout/dim'])
        shape = list(x.shape)
        sim = OutProductSimulator(**config, num_reduce_trees=budget, record_tree_trace=True)
        start = time.perf_counter()
        with (folder / f'case_{case_id:02d}.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
            if kind == 'conv2d_(3, 3)_(1, 1)_(1, 1)':
                _, stats = sim.run_convolution(x, torch.zeros((cout, x.shape[2], 3, 3), dtype=torch.int8))
                path = 'conv'
            elif kind == 'conv1d_1_1':
                cin = x.shape[-2]
                matrix = x.reshape(-1, cin, x.shape[-1]).permute(0, 2, 1).reshape(-1, cin)
                _, stats = sim.run_fc(matrix, torch.zeros((cin, cout), dtype=torch.int8))
                path = 'fc'
            elif kind == 'conv1d_5_1':
                cin = x.shape[-2]
                matrix = img2col(x.reshape(-1, cin, x.shape[-1]).unsqueeze(2),
                                 kernel_size=(1, 5), stride=1, padding=1).to(torch.int8).flatten(0, 1)
                _, stats = sim.run_fc(matrix, torch.zeros((cin * 5, cout), dtype=torch.int8))
                path = 'fc'
            else:
                raise ValueError(f'Unsupported TIM operator: {kind}')
        counters = stats.reduce_tree_stats
        trace_name = f'case_{case_id:02d}.trace.csv.gz'
        with gzip.open(folder / trace_name, 'wt', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['representative_cycle', 'needed_trees', 'used_trees', 'rejected_points'])
            writer.writerows(sim.reduce_tree_trace)
        row = dict(case=case_id, reused_case='', trees=label, path=path, file=case['file'],
                   input_shape=json.dumps(shape), total_cycles=stats.total_cycles,
                   compute_cycles=stats.compute_cycles, representative_cycles=stats.representative_compute_cycles,
                   cout_scale=stats.reduce_tree_cout_scale, demand_peak=counters['tree_demand_peak'],
                   used_peak=counters['tree_used_peak'], exhausted_cycles=counters['tree_exhausted_cycles'],
                   rejected_points=counters['tree_rejected_points'],
                   psum_alloc_core_cycles=stats.psum_buffer_alloc_stall_cycles,
                   frontend_core_cycles=stats.frontend_stall_cycles,
                   conflict_core_cycles=counters['conflict_core_cycles'],
                   seconds=round(time.perf_counter() - start, 3), trace=trace_name)
        hist = [dict(case=case_id, trees=label, count=n,
                     demand_cycles=stats.reduce_tree_demand_hist.get(n, 0),
                     used_cycles=stats.reduce_tree_used_hist.get(n, 0))
                for n in sorted(set(stats.reduce_tree_demand_hist) | set(stats.reduce_tree_used_hist))]
        assert sum(h['demand_cycles'] for h in hist) == row['compute_cycles']
        assert len(sim.reduce_tree_trace) == row['representative_cycles']
        rows.append(row)
        hist_rows.extend(hist)
        cache[key] = row, hist
        write_csv(folder / 'layers.csv', rows)
        write_csv(folder / 'histograms.csv', hist_rows)
        print(f'trees={label} case={case_id} {path} cycles={row["compute_cycles"]} '
              f'peak={row["demand_peak"]} seconds={row["seconds"]}', flush=True)
    write_csv(folder / 'layers.csv', rows)
    write_csv(folder / 'histograms.csv', hist_rows)
    write_csv(folder / 'skipped.csv', skipped)
    return label, rows, hist_rows


def summarize(output, results):
    output = Path(output)
    by_budget = {label: rows for label, rows, _ in results}
    reference = by_budget['144']
    if 'unlimited' in by_budget:
        assert len(reference) == len(by_budget['unlimited'])
        for a, b in zip(reference, by_budget['unlimited']):
            assert (a['case'], a['compute_cycles'], a['total_cycles']) == (b['case'], b['compute_cycles'], b['total_cycles'])
    summary = []
    for label, rows, _ in results:
        assert [r['case'] for r in rows] == [r['case'] for r in reference]
        for path in ('all', 'conv', 'fc'):
            subset = [r for r in rows if path == 'all' or r['path'] == path]
            baseline = [r for r in reference if path == 'all' or r['path'] == path]
            total = sum(r['total_cycles'] for r in subset)
            compute = sum(r['compute_cycles'] for r in subset)
            base_total = sum(r['total_cycles'] for r in baseline)
            base_compute = sum(r['compute_cycles'] for r in baseline)
            summary.append(dict(trees=label, path=path, layers=len(subset), total_cycles=total,
                                compute_cycles=compute, total_increase_pct=100*(total/base_total-1),
                                throughput_loss_pct=100*(1-base_total/total),
                                compute_increase_pct=100*(compute/base_compute-1),
                                exhausted_cycles=sum(r['exhausted_cycles'] for r in subset),
                                rejected_points=sum(r['rejected_points'] for r in subset),
                                psum_alloc_core_cycles=sum(r['psum_alloc_core_cycles'] for r in subset)))
    write_csv(output / 'summary.csv', summary)
    baseline_hist = next(hist for label, _, hist in results if label == ('unlimited' if 'unlimited' in by_budget else '144'))
    histogram = collections.Counter()
    for row in baseline_hist:
        histogram[row['count']] += row['demand_cycles']
    total = sum(histogram.values())
    running = 0
    distribution = []
    for count, cycles in sorted(histogram.items()):
        running += cycles
        distribution.append(dict(needed_trees=count, cycles=cycles, fraction=cycles/total,
                                 coverage=running/total, exceed_cycles=total-running))
    write_csv(output / 'baseline_demand.csv', distribution)
    report = ['# TIM_NCARS reduce-tree sweep', '',
              'Shared dynamic trees; one conflicting (row,col) address per tree per cycle. '
              'All cycle requests are collected before arbitration. Only singleton addresses bypass reduction; '
              'every writer to a conflicting address waits unless that address receives a tree. '
              'Trees use fixed core/address priority, with atomic Conv bundles and partial FC bundles.', '',
              f'{len(reference)} simulated CSV entries; SSA and float-input operators are excluded, matching the regular runner. '
              'Repeated identical entries reuse a simulation and are counted separately. '
              'Totals are sums of simulated layer cycles, not a measured full-network runtime.', '',
              '| Trees | Total cycles | Increase | Throughput loss |',
              '|---:|---:|---:|---:|']
    for row in summary:
        if row['path'] == 'all':
            report.append(f'| {row["trees"]} | {row["total_cycles"]:,} | {row["total_increase_pct"]:.3f}% | {row["throughput_loss_pct"]:.3f}% |')
    report += ['', f'Baseline demand peak: {max(histogram)} trees.',
               'Histograms include every compute-loop cycle, including zero-demand cycles, and apply Cout scaling. '
               'Per-cycle gzip traces contain representative cycles before Cout scaling. '
               'Limited-run demand includes retries and changed scheduling; use the baseline for sizing.']
    (output / 'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trees', default='unlimited,144,64,32,16,8,4,2,1,0')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--enabled_modes', default='0')
    args = parser.parse_args()
    budgets = [None if value == 'unlimited' else int(value) for value in args.trees.split(',')]
    if 144 not in budgets or any(b is not None and b < 0 for b in budgets):
        parser.error('Include baseline 144 and only nonnegative tree counts or unlimited')
    config = dict(num_cores=6, num_pus=16, oh=12, ow=12, split_fifo_depth=4,
                  psum_pool_rows=6, retire_column=3, num_split=2,
                  enabled_modes=tuple(int(m) for m in args.enabled_modes.split(',')))
    args.output.mkdir(parents=True, exist_ok=True)
    sources = ['Accumulator.py', 'SharedPsumPool.py', 'core.py', 'shift.py', 'simulator_real_compute.py', 'sweep_reduce_trees.py']
    hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}
    manifest = dict(network='TIM_NCARS', config=config, trees=budgets, source_sha256=hashes,
                    input_policy='Full stored tensors, matching simulator_runner.py. Main layers T=10/B=1; TIM interactor arrays first dimension=16.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_budget, b, config, str(args.output.resolve())) for b in budgets]
        results = [future.result() for future in futures]
    summarize(args.output, results)
    print(f'Complete: {args.output / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
