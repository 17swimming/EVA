"""报告已有重叠时间线及代表输出通道组的卷积数据包比例。"""
from fractions import Fraction

OTHER_CYCLE_REASONS = ('scheduler', 'split_unit', 'accumulator_backpressure',
                       'tile_handoff', 'row_dependency', 'accumulator_tail', 'control')
PAPER_COMPONENTS = ('DRAM', 'scheduler', 'split_unit',
                    'current_accumulating_sub_unit', 'accumulator', 'LIF')
SPLIT_WAIT_REASONS = ('startup', 'finish_confirmation', 'mode0_stream_complete',
    'fifo_backpressure', 'row_order_wait', 'row_dispatch', 'row_handoff',
    'window_assembly', 'encoder_supply', 'linear_supply', 'unclassified')


def paper_cycle_breakdown(counters, num_cores, breakdown):
    """计算核心平均周期归因；分配停顿和排空仍归入电流累积单元。"""
    other = counters.get('other_partition_core_cycles', 0)
    assert sum(counters.get(f'other_{reason}_core_cycles', 0)
               for reason in OTHER_CYCLE_REASONS) == other
    for scope, parent in (('paper', 'paper_split_unit_core_cycles'),
                          ('other', 'other_split_unit_core_cycles')):
        assert sum(counters.get(f'{scope}_split_wait_{reason}_core_cycles', 0)
                   for reason in SPLIT_WAIT_REASONS) == counters.get(parent, 0)
    backend = breakdown['backend_cycles']
    assert sum(counters.get(f'paper_{key}_core_cycles', 0)
               for key in PAPER_COMPONENTS if key != 'DRAM') == backend * num_cores
    assert counters.get('paper_LIF_core_cycles', 0) == breakdown['lif_exposed_cycles'] * num_cores
    values = {key: Fraction(counters.get(f'paper_{key}_core_cycles', 0), num_cores)
              for key in PAPER_COMPONENTS if key != 'DRAM'}
    values['DRAM'] = breakdown['dram_exposed_cycles']
    assert sum(values.values()) == breakdown['total_cycles']
    assert all(value >= 0 for value in values.values())
    return {key: float(values[key]) for key in PAPER_COMPONENTS}


def cycle_breakdown(backend_cycles, total_cycles, lif_exposed_cycles,
                    dram_work_cycles, dram_startup_cycles, lif_read_cycles,
                    lif_hidden_read_cycles, lif_pipeline_tail_cycles):
    """仅分解已有墙钟周期；正常计算包含调度、退休和反压等待。"""
    normal_compute_cycles = backend_cycles - lif_exposed_cycles
    dram_exposed_cycles = total_cycles - backend_cycles
    if normal_compute_cycles < 0 or dram_exposed_cycles < 0:
        raise AssertionError('Exposed cycles exceed the existing timeline')
    if dram_exposed_cycles != dram_startup_cycles + max(0, dram_work_cycles - backend_cycles):
        raise AssertionError('DRAM decomposition disagrees with the overlap model')
    return {
        'normal_compute_cycles': normal_compute_cycles,
        'lif_exposed_cycles': lif_exposed_cycles,
        'dram_exposed_cycles': dram_exposed_cycles,
        'backend_cycles': backend_cycles,
        'total_cycles': total_cycles,
        'lif_work_cycles': lif_read_cycles + lif_pipeline_tail_cycles,
        'lif_read_cycles': lif_read_cycles,
        'lif_hidden_read_cycles': lif_hidden_read_cycles,
        'lif_pipeline_tail_cycles': lif_pipeline_tail_cycles,
        'dram_work_cycles': dram_work_cycles,
        'dram_startup_cycles': dram_startup_cycles,
    }


def result_columns(stats):
    """按指定名称导出九列，并保留已有的额外报告字段；缺少的统计留空。"""
    result = {}
    counts = getattr(stats, 'conv_mode_split_counts', None)
    total = sum(counts.get(mode, 0) for mode in (0, 1, 2)) if counts is not None else 0
    for mode in (0, 1, 2):
        count = counts.get(mode, 0) if counts is not None else None
        result[f'num_mode{mode}'] = count
        result[f'ratio_mode{mode}'] = count / total if total else None
    breakdown = getattr(stats, 'cycle_breakdown', None)
    for column, key in (('EVA', 'normal_compute_cycles'),
                        ('LIF', 'lif_exposed_cycles'),
                        ('DRAM', 'dram_exposed_cycles')):
        result[column] = breakdown[key] if breakdown else None
    result.update(eva_total_cycles=stats.total_cycles, eva_backend_cycles=stats.compute_cycles)
    if breakdown:
        renamed = {'normal_compute_cycles', 'lif_exposed_cycles', 'dram_exposed_cycles'}
        result.update((f'eva_{key}', value) for key, value in breakdown.items() if key not in renamed)
    if counts is not None:
        result['eva_conv_packets_representative'] = total
    return result
