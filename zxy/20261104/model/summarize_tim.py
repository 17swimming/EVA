"""Validate completed TIM runs and write an auditable performance comparison."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VARIANTS = ('zxy_native', 'zxy_1bundle', 'bank_fifo2', 'bank_lookahead2')


OUTPUT = ROOT / 'output/accumulator_bank_lookahead_TIM_20260929'
STUDY = Path(__file__).resolve().parent


def read(variant):
    directory = OUTPUT / variant
    with (directory / 'operations.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    assert summary['complete'] and summary['layers'] == 18 and summary['operations'] == 20
    assert len(rows) == 20 and len({(r['layer'], r['operation']) for r in rows}) == 20
    for category in ('CONV', 'FC', 'SSA', 'ALL'):
        for metric in ('compute_cycles', 'total_cycles'):
            actual = sum(int(r[metric]) for r in rows if category == 'ALL' or r['category'] == category)
            assert actual == summary[category][metric]
    return rows, summary, metadata


def percent(value):
    return f'{value:+.2f}%'


def main():
    runs = {variant: read(variant) for variant in VARIANTS}
    source_hashes = {json.dumps(item[2]['source_hashes'], sort_keys=True) for item in runs.values()}
    manifest_hashes = {item[2]['manifest_sha256'] for item in runs.values()}
    assert len(source_hashes) == len(manifest_hashes) == 1
    baseline = runs['zxy_native'][1]
    restricted = runs['zxy_1bundle'][1]
    fifo = runs['bank_fifo2'][1]
    lookahead = runs['bank_lookahead2'][1]
    lines = [
        '# TIM_NCARS：双表项前瞻仲裁性能结果', '',
        '数据范围：最终 `zxy_k` 映射下 TIM 清单全部 18 层；SSA 各包含两次 FC 模拟，共 20 次调用。',
        '`run_tim.py` 是本实验专用驱动，读取原 `simulator_runner_end_to_end.py` 所用的 `TIM_NCARS.csv` 与 `.npy`，逐算子调用复制的 zxy 模拟器或 bank 实验版；没有直接执行原 end-to-end runner。原 runner 的 `k` 分支仍指向 `model._k`，且 SSA 的第二次 FC 调用存在 K 维不匹配。以下是统一实验驱动结果，不应与未经修改的原 runner 输出混称。',
        '四组使用相同输入、6 Core、16 PU、12×12 result array、8 Psum 单元、split FIFO 深度 4、两路 split，模式 0/1/2。',
        'FC 均按同列最多三个输出行退休，CONV 均按同行最多三个不同列退休。', '',
        '## 总周期', '',
        '| 方案 | CONV | FC | SSA | 总周期 | 相对 zxy 原版 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for variant in VARIANTS:
        s = runs[variant][1]
        delta = (s['ALL']['total_cycles'] / baseline['ALL']['total_cycles'] - 1) * 100
        lines.append(f'| `{variant}` | {s["CONV"]["total_cycles"]:,} | '
                     f'{s["FC"]["total_cycles"]:,} | {s["SSA"]["total_cycles"]:,} | '
                     f'{s["ALL"]["total_cycles"]:,} | {percent(delta)} |')
    lines += ['',
        '所有表内周期均为 simulator 报告的 `total_cycles`，包含现有 LIF 和访存时间模型。',
        '原版每 Core 每拍最多提交两个 bundle；`zxy_1bundle` 只把此上限改为一个；'
        '另外两版均每 Core 两个独立物理缓冲表项、每 bank 每拍服务一个 bundle，'
        '其中前瞻版在队首仲裁后允许落选 Core 的第二表项竞争空闲 bank。', '',
        '## 开销分解', '',
    ]
    for category in ('CONV', 'FC', 'SSA', 'ALL'):
        base = baseline[category]['total_cycles']
        restricted_delta = restricted[category]['total_cycles'] - base
        bank_delta = lookahead[category]['total_cycles'] - restricted[category]['total_cycles']
        fifo_recovery = fifo[category]['total_cycles'] - lookahead[category]['total_cycles']
        lines.append(f'- {category}：每 Core 一包上限相对原版 {restricted_delta:+,} 拍；'
                     f'前瞻 bank 方案相对一包上限 {bank_delta:+,} 拍；'
                     f'前瞻相对普通 FIFO 节约 {fifo_recovery:,} 拍。')
    recovery = fifo['ALL']['total_cycles'] - lookahead['ALL']['total_cycles']
    penalty = lookahead['ALL']['total_cycles'] - baseline['ALL']['total_cycles']
    lines += ['',
        f'前瞻仲裁比普通 FIFO 少 {recovery:,} 拍，即普通 FIFO 总周期的 '
        f'{100 * recovery / fifo["ALL"]["total_cycles"]:.2f}%；'
        f'但相对原版仍多 {penalty:,} 拍。',
        '两阶段比较能看到：CONV 的较大损失已在限制每 Core 一包时出现；'
        'FC/SSA 的差异主要出现在采用 bank、缓冲和写回流水的实验版。'
        '这些同时改变了多个时序条件，不能把差额单独归因为 bank 冲突。', '',
        '## 逐层总周期', '',
        '| 层 | 类型 | zxy 原版 | zxy 一包 | bank FIFO | bank 前瞻 | 前瞻相对原版 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    per_variant = {}
    for variant, (rows, _, _) in runs.items():
        grouped = {}
        for row in rows:
            group = grouped.setdefault(int(row['layer']), {'category': row['category'], 'total': 0, 'file': row['file']})
            group['total'] += int(row['total_cycles'])
        per_variant[variant] = grouped
    for layer in range(1, 19):
        values = [per_variant[variant][layer]['total'] for variant in VARIANTS]
        category = per_variant['zxy_native'][layer]['category']
        lines.append(f'| {layer} | {category} | ' + ' | '.join(f'{value:,}' for value in values)
                     + f' | {percent((values[-1] / values[0] - 1) * 100)} |')
    lines += ['', '每层名称、两段 SSA 的独立调用结果、缓冲和仲裁计数均在四组的 `operations.csv` 中。',
              '', '## 请求缓冲和前瞻使用量', '']
    for variant in ('bank_fifo2', 'bank_lookahead2'):
        rows = runs[variant][0]
        metrics = {key: sum(int(r[key]) for r in rows) for key in
                   ('enqueued_bundles', 'committed_bundles', 'committed_points',
                    'bypass_bundles', 'enqueue_rejected_bundles',
                    'two_entries_same_bank', 'two_entries_different_bank')}
        assert metrics['enqueued_bundles'] == metrics['committed_bundles']
        lines.append(f'- `{variant}`：{metrics["enqueued_bundles"]:,} 包入队并写回、'
                     f'{metrics["committed_points"]:,} 点写回、'
                     f'{metrics["enqueue_rejected_bundles"]:,} 次因满而拒绝；'
                     f'第二表项旁路 {metrics["bypass_bundles"]:,} 包。'
                     '这些是各层代表性运行的原始计数之和，未按 Cout 比例放大。')
    lines += ['', '## 可比性与限制', '',
        '- SSA 采用原 end-to-end runner 的两次 FC 调用方式。原 runner 在 `N<=D` 分支把 '
        '`A` 占位矩阵设为 `N×N`，却把 `V` 展开成 `(T×B×heads×N)×D`，第二次 FC 的 K 维不匹配。'
        '本实验把仍为全零的 `A` 占位矩阵设为 `N×(T×B×heads×N)`，以匹配原 runner 的展开 `V`。'
        '这是使该占位调用可执行的维度修正，不是逐 head 的数值正确 SSA 模型。'
        '四组使用完全一致的修正；单独解读 SSA 数值时须保留此限制。',
        '- zxy_k 的数值计算路径只计算代表性 PU0；实验 result array 对真实标量只更新 PU0，'
        '其余 PU 的全宽写回功能已由单元测试检查。本报告是周期比较，不是端到端数值精度验证。',
        '- 当前版本的 LIF 周期模型保持 zxy_k 的既有假设；用户要求暂不重新建模 LIF。',
        '- 已有结果是 Python 模拟周期，没有在此轮对新架构做 DC 面积或 500 MHz 时序综合。',
        '', '运行数据：`output/accumulator_bank_lookahead_TIM_20260929/<variant>/operations.csv`、'
        '`summary.json`、`metadata.json`、`simulation.log`。',
    ]
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
