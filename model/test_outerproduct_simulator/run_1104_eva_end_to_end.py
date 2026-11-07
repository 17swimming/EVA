"""先核对完整 TIM 周期，通过后运行其余四个完整工作负载。"""
import os
os.environ['EVA_DISPATCH_LANES'] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import json
import re
from pathlib import Path
import shutil
import sys
import time
import traceback

import run_1104_workload_batch as batch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REFERENCE = ROOT/'output/performance_analysis/nine_way_layers_core4_dispatch1_lif64_bundle2_20261106'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compare_tim(output, result):
    """比较端到端和逐层周期，保留网络末尾池化修正。"""
    reference = next(r for r in read_json(REFERENCE/'summary.json') if r['network'] == 'TIM_NCARS')
    differences = []
    for field in ('calls', 'layers', 'network_total_cycles', 'component_total_cycles', 'network_adjustment_cycles'):
        if result.get(field) != reference[field]:
            differences.append(dict(field=field, expected=reference[field], actual=result.get(field)))
    for filename in ('cycles_breakdown_layers.csv',):
        def rows(root):
            with (root/'TIM_NCARS_1104'/filename).open(encoding='utf-8-sig', newline='') as handle:
                return [(r['test_case'], r['layer'], r['total_cycles']) for r in csv.DictReader(handle)]
        expected, actual = rows(REFERENCE), rows(output)
        if expected != actual:
            differences.append(dict(field=filename, expected=expected, actual=actual))
    comparison = dict(status='pass' if not differences else 'failed',
                      expected_network_cycles=474113, actual_network_cycles=result.get('network_total_cycles'),
                      differences=differences, reference=str(REFERENCE), checked=batch.now())
    batch.save(output/'TIM_comparison.json', comparison)
    assert not differences, 'TIM comparison failed; remaining workloads were not started'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--skip-tim-reference', action='store_true',
                        help='独立复现实验时跳过本地历史参考比较，仍完整运行五个工作负载')
    parser.add_argument('--resume-after-tim', action='store_true',
                        help='接受已核对的 TIM 差异，保留结果并续跑其余四个完整工作负载')
    args = parser.parse_args()
    if args.skip_tim_reference and args.resume_after_tim:
        parser.error('--skip-tim-reference cannot be combined with --resume-after-tim')
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output/'source_snapshot'
    results = []
    if args.resume_after_tim:
        manifest = read_json(output/'batch_manifest.json')
        previous = read_json(output/'COMPLETE.json')
        assert previous['status'] == 'failed'
        results = read_json(output/'summary.json')
        assert len(results) == 1 and results[0]['network'] == 'TIM_NCARS'
        assert results[0]['status'] == 'pass' and results[0]['network_total_cycles'] == 473025
        diagnosis = read_json(output/'TIM_difference_diagnosis.json')
        assert diagnosis['all_changes_zero_input_dram'] and diagnosis['delta'] == -1088
        for relative, digest in manifest['hashes'].items():
            if relative == Path(__file__).name:
                continue
            source = (ROOT/'zxy/20261104'/relative if Path(relative).parts[0] == 'model'
                      else Path(__file__).parent/relative)
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, f'Source changed: {source}'
        for network in batch.NETWORKS:
            if network != 'TIM_NCARS':
                assert not (output/f'{network}_1104').exists(), f'Existing results: {network}'
        archive = output/'TIM_gate_COMPLETE.json'
        assert not archive.exists()
        (output/'COMPLETE.json').rename(archive)
        shutil.copy2(Path(__file__), output/'resume_runner_snapshot.py')
        batch.save(output/'resume_manifest.json', dict(resumed=batch.now(), parent_pid=os.getpid(),
            accepted_tim_cycles=473025, reference_tim_cycles=474113,
            reason='User accepted the diagnosed difference and requested the remaining four workloads',
            source_hashes_verified=True, max_test_cases=None))
    else:
        assert not (output/'batch_manifest.json').exists(), 'Refusing to overwrite existing results'
        shutil.copytree(ROOT/'zxy/20261104/model', snapshot/'model', ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('simulator_runner.py', 'run_1104_workload_batch.py', Path(__file__).name,
                     *[n+'.csv' for n in batch.NETWORKS]):
            shutil.copy2(Path(__file__).parent/name, snapshot/name)
    catalog = {}
    for network in batch.NETWORKS:
        with (snapshot/(network+'.csv')).open(encoding='utf-8-sig') as handle:
            catalog[network] = len([r for r in csv.DictReader(handle) if r.get('type', '').strip()])
    if not args.resume_after_tim:
        batch.save(output/'batch_manifest.json', dict(started=batch.now(), parent_pid=os.getpid(),
        expected_workloads=5, num_cores=4, num_pus=16, tile=[12, 12], lif_units=64,
        bundle_buffer_depth=2, bank_lookahead_entries=2, dispatch_lanes=1, frequency_mhz=500,
        split_fifo_depth=4, pool_banks=8, eva_only=True, max_test_cases=None,
        workloads=catalog, tim_reference_cycles=474113,
        workbook=str(ROOT/'output/20261106/end_to_end.xlsx'), workbook_status='pending_simulation_and_review',
        hashes={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in snapshot.rglob('*') if p.is_file()}))
    state = {n: dict(status='queued', total_test_cases=catalog[n]) for n in batch.NETWORKS}
    if results:
        state['TIM_NCARS'].update(results[0], reused=True)
    started = time.time()

    def progress(stage):
        with batch.LOCK:
            current = json.loads(json.dumps(state))
        for network, item in current.items():
            log_path = output/f'{network}.console.log'
            if log_path.exists():
                with log_path.open('rb') as handle:
                    handle.seek(max(0, log_path.stat().st_size-131072))
                    tail = handle.read().decode('utf-8', errors='replace')
                matches = re.findall(r'=== (?:跳过)?测试用例 (\d+)/(\d+):', tail)
                if matches:
                    item['latest_test_case'], item['total_test_cases'] = map(int, matches[-1])
        batch.save(output/'progress.json', dict(updated=batch.now(), stage=stage,
            parent_pid=os.getpid(), elapsed_seconds=time.time()-started, expected=5,
            completed=len(results), workloads=current))
        batch.save(output/'summary.json', results)

    def run(network):
        return batch.run_network(network, batch.NETWORKS[network], output, state,
                                 bundle_depth=2, lif_units=64)

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            if not args.resume_after_tim:
                pending = {pool.submit(run, 'TIM_NCARS')}
                while pending:
                    done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                    results.extend(f.result() for f in done)
                    progress('TIM_comparison')
                assert results[0]['status'] == 'pass', results[0]
                if not args.skip_tim_reference:
                    compare_tim(output, results[0])
                else:
                    batch.save(output/'TIM_comparison.json', dict(status='skipped',
                        reason='Standalone reproduction without the historical local reference',
                        actual_network_cycles=results[0]['network_total_cycles']))
            progress('starting_remaining_workloads')
            pending = {pool.submit(run, n) for n in batch.NETWORKS if n != 'TIM_NCARS'}
            while pending:
                done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                results.extend(f.result() for f in done)
                progress('remaining_workloads')
        failures = [r for r in results if r['status'] != 'pass']
        assert len(results) == 5 and not failures, failures
        batch.save(output/'eva_cycle_updates.json', dict(frequency_mhz=500,
            cycles={r['network']:r['network_total_cycles'] for r in results},
            workbook=str(ROOT/'output/20261106/end_to_end.xlsx'), status='ready_for_workbook_update'))
        progress('simulation_complete_workbook_pending')
        batch.save(output/'COMPLETE.json', dict(status='pass', finished=batch.now(),
            expected_workloads=5, finished_workloads=5, passed_workloads=5, failures=[],
            workbook_updated=False, elapsed_seconds=time.time()-started))
    except Exception:
        progress('failed')
        batch.save(output/'COMPLETE.json', dict(status='failed', finished=batch.now(),
            expected_workloads=5, finished_workloads=len(results),
            passed_workloads=sum(r['status'] == 'pass' for r in results),
            failures=[traceback.format_exc()], workbook_updated=False))
        raise


if __name__ == '__main__':
    main()
