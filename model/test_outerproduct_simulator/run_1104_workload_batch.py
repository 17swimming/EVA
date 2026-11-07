"""运行1104五网络周期分析，保存进度、六类周期归因和独立完成标志。"""
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
NETWORKS = {'TIM_NCARS': 10, 'QKFormer_DVSGesture': 16,
            'SpikeBRGNet_DDD17': 5, 'EMSYOLO_GEN1': 5, 'SDTrack_FE108': 4}
COMPONENTS = ('DRAM', 'scheduler', 'split_unit', 'current_accumulating_sub_unit', 'accumulator', 'LIF')
LOCK = threading.Lock()


def now():
    """记录带时区的本机时间，便于关联后台进度与日志。"""
    return datetime.now().astimezone().isoformat(timespec='seconds')


def save(path, data):
    """通过同目录替换发布完整进度，避免读取到半个文件。"""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def validate_network(directory, network, *, bundle_depth=3, lif_units=None):
    """校验runner已结束、调用统计守恒以及六类分解与逐层周期一致。"""
    from output.lookahead.plot.extract_cycles_breakdown import parse_log, save_csv

    complete = json.loads((directory/'COMPLETE.json').read_text(encoding='utf-8'))
    assert complete['status'] == 'pass'
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['num_cores'] == 4
    assert manifest['configuration']['eva_only']
    assert manifest['split_fifo_depth'] == 4
    assert manifest['bundle_buffer_depth'] == manifest['bank_lookahead_entries'] == bundle_depth
    if lif_units is not None:
        assert manifest['lif_units'] == lif_units
    logs = list(directory.glob('dla*_simulation_log_*.txt'))
    assert len(logs) == 1, logs
    rows, warnings = parse_log(logs[0], allow_network_summary_mismatch=network == 'TIM_NCARS')
    save_csv(rows, directory/'cycles_breakdown_layers.csv')
    total = rows[-1]
    text = logs[0].read_text(encoding='utf-8')
    assert '仿真任务完成' in text
    network_total = int(re.search(r'^总EVA周期:\s*([\d,]+)\s*$', text, re.MULTILINE)[1].replace(',', ''))
    summary = json.loads((directory/'performance_summary.json').read_text(encoding='utf-8'))
    assert total['total_cycles'] == summary['all']['total_cycles']
    for key in COMPONENTS:
        assert abs(float(total[key])-summary['all']['cycles_breakdown'][key]) < 1e-6
    assert complete['calls'] == summary['all']['calls']
    result = dict(network=network, status='pass', calls=complete['calls'], layers=len(rows)-1,
                  network_total_cycles=network_total, component_total_cycles=total['total_cycles'],
                  network_adjustment_cycles=network_total-total['total_cycles'],
                  components={key: float(total[key]) for key in COMPONENTS},
                  percentages={key: float(total[key]/total['total_cycles']*100) for key in COMPONENTS},
                  warnings=warnings, directory=str(directory))
    assert result['network_adjustment_cycles'] == (-417 if network == 'TIM_NCARS' else 0)
    save(directory/'cycles_breakdown_summary.json', result)
    return result


def run_network(network, timesteps, output, state, *, bundle_depth=3, lif_units=48, max_test_cases=None,
                variant='1104'):
    """每个网络使用独立进程、日志和结果目录，只运行EVA。"""
    arguments = [sys.executable, '-B', '-u', '-X', 'utf8',
                 str(ROOT/'model/test_outerproduct_simulator/simulator_runner.py'),
                 '--network', network, '--simulator', 'cluster_parallel',
                 '--accumulator-variant', variant, '--eva-only', '--num_cores', '4',
                 '--num_pus', '16', '--bank_h', '12', '--bank_w', '12',
                 '--num_split', '2', '--num_clusters', '1', '--fetch_rows', '8',
                 '--split_fifo_depth', '4', '--psum_pool_rows', '8', '--retire_column', '3',
                 '--retire-packets-per-cycle', '2', '--prefetch-cycles', '2',
                 '--bundle-buffer-depth', str(bundle_depth), '--bank-lookahead-entries', str(bundle_depth),
                 '--lif-units', str(lif_units),
                 '--num_reduce_trees', '0', '--enabled_modes', '0,1,2',
                 '--T', str(timesteps), '--B', '1', '--output-root', str(output)]
    if max_test_cases is not None:
        arguments.extend(['--max-test-cases', str(max_test_cases)])
    log_path = output/f'{network}.console.log'
    with LOCK:
        state[network] = dict(status='starting', timesteps=timesteps,
                              arguments=arguments, console_log=str(log_path))
    try:
        with log_path.open('w', encoding='utf-8') as handle:
            process = subprocess.Popen(arguments, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            with LOCK:
                state[network].update(status='running', pid=process.pid, started=now())
            code = process.wait()
        assert code == 0, f'runner exit code {code}; see {log_path}'
        result = validate_network(output/f'{network}_{variant}', network, bundle_depth=bundle_depth, lif_units=lif_units)
    except Exception:
        result = dict(network=network, status='failed', error=traceback.format_exc(), console_log=str(log_path))
    with LOCK:
        state[network].update(result, finished=now())
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    """启动五个完整网络并在全部结束后发布批次完成与验证结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=5, choices=range(1, 6))
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not (output/'batch_manifest.json').exists(), 'Refusing to overwrite an existing batch'
    import torch
    torch.set_num_threads(1)
    from model.test_outerproduct_simulator.simulator_runner import load_test_cases, supports_eva_spike

    catalog = {}
    for network, timesteps in NETWORKS.items():
        cases = load_test_cases(network)
        eligible = [case for case in cases if supports_eva_spike(case)]
        missing = [case['file'] for case in eligible if not Path(case['file']).is_file()]
        assert not missing, (network, missing)
        catalog[network] = dict(timesteps_argument=timesteps, input_timesteps='原始激励中的全部时间步',
                                cases=len(cases), spike_eligible=len(eligible))
    snapshot = output/'source_snapshot'
    shutil.copytree(ROOT/'zxy/20261104/model', snapshot/'model', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['simulator_runner.py', 'run_1104_workload_batch.py', *[network+'.csv' for network in NETWORKS]]:
        shutil.copy2(ROOT/'model/test_outerproduct_simulator'/name, snapshot/name)
    manifest = dict(started=now(), parent_pid=os.getpid(), expected_workloads=5,
                    num_cores=4, eva_only=True, workers=args.workers, workloads=catalog,
                    source_hashes={str(path.relative_to(snapshot)):hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in snapshot.rglob('*') if path.is_file()})
    save(output/'batch_manifest.json', manifest)
    state = {network: dict(status='queued') for network in NETWORKS}
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_network, name, steps, output, state) for name, steps in NETWORKS.items()}
        while pending:
            done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            results.extend(future.result() for future in done)
            with LOCK:
                progress = json.loads(json.dumps(state))
            for network, item in progress.items():
                directory = output/f'{network}_1104'
                item['result_directory'] = str(directory)
                item['last_result_update'] = (datetime.fromtimestamp((directory/'calls.jsonl').stat().st_mtime)
                    .astimezone().isoformat(timespec='seconds') if (directory/'calls.jsonl').exists() else None)
                log_path = output/f'{network}.console.log'
                if log_path.exists():
                    with log_path.open('rb') as handle:
                        handle.seek(max(0, log_path.stat().st_size-131072))
                        tail = handle.read().decode('utf-8', errors='replace')
                    matches = re.findall(r'=== (?:跳过)?测试用例 (\d+)/(\d+):', tail)
                    if matches:
                        item['latest_test_case'] = int(matches[-1][0])
                        item['total_test_cases'] = int(matches[-1][1])
            save(output/'progress.json', dict(updated=now(), expected=5, completed=len(results), workloads=progress))
            save(output/'summary.json', results)
    with (output/'cycles_breakdown_summary.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=('network', 'status', 'layers', 'calls',
             'network_total_cycles', 'component_total_cycles', 'network_adjustment_cycles', *COMPONENTS))
        writer.writeheader()
        for result in results:
            row = {key: result[key] for key in writer.fieldnames if key in result}
            row.update(result.get('components', {}))
            writer.writerow(row)
    failures = [result for result in results if result['status'] != 'pass']
    save(output/'COMPLETE.json', dict(status='pass' if not failures and len(results)==5 else 'failed',
        expected_workloads=5, finished_workloads=len(results), passed_workloads=len(results)-len(failures),
        failures=failures, finished=now()))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        output_argument = sys.argv.index('--output-root')+1
        output = Path(sys.argv[output_argument]).resolve()
        if output.is_dir():
            save(output/'COMPLETE.json', dict(status='failed', finished=now(), error=traceback.format_exc()))
        raise
