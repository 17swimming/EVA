"""从端到端实验日志提取逐层周期归因，并绘制网络总周期的横向堆叠图。"""

import argparse
import csv
from fractions import Fraction
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiment_colors import BREAKDOWN_COLORS


COMPONENTS = ('DRAM', 'Scheduler', 'SplitUnit',
              'Current_Accumulating', 'Accumulator', 'LIF')
COMPONENT_KEYS = dict(zip(COMPONENTS, (
    'DRAM', 'scheduler', 'split_unit', 'current_accumulating_sub_unit', 'accumulator', 'LIF')))
COUNTER_COMPONENTS = tuple(COMPONENT_KEYS.values())
NUMBER = r'([\d,]+)'
BREAKDOWN = re.compile(
    rf'总周期分解:\s*{NUMBER}\s*=\s*正常计算\s*{NUMBER}'
    rf'\s*\+\s*未隐藏 LIF\s*{NUMBER}\s*\+\s*未隐藏 DRAM\s*{NUMBER}')
CASES = re.compile(r'^=== 测试用例 (\d+)/\d+: (.+?) ===\s*$', re.MULTILINE)
NETWORKS = (
    ('TIM_NCARS', 'TIM'),
    ('QKFormer_DVSGesture', 'QKFormer'),
    ('SpikeBRGNet_DDD17', 'BRGNet'),
    ('SDTrack_FE108', 'SDTrack'),
    ('EMSYOLO_GEN1', 'EMSYOLO'),
)
COMPONENT_LABELS = {key: component for component, key in COMPONENT_KEYS.items()}
WORKLOAD_INITIALS = dict(zip((network for network, _ in NETWORKS), ('T', 'Q', 'B', 'S', 'E')))

# 左侧工作负载名称的字号；所有名称均使用粗体。
WORKLOAD_FONT_SIZE = 7
# 柱内百分比的字号；所有比例均使用粗体并带百分号。
PERCENT_FONT_SIZE = 6.5
# 百分比统一使用黑色粗体，与浅色柱段搭配，保证黑白打印时的可读性。
PERCENT_TEXT_COLOR = '#000000'
# 顶端两行图例与柱内百分比使用相同字号，修改柱内字号时同步调整。
LEGEND_FONT_SIZE = PERCENT_FONT_SIZE
# 百分比的小数位数，以及允许在柱内标注的最小百分比。
PERCENT_DECIMALS = 0
MIN_LABEL_PERCENT = 4
# 图片宽度和高度使用英寸；压缩整张图，并使柱子实际高度与消融组合图一致。
FIGURE_WIDTH = 3.5
FIGURE_HEIGHT = 0.92
BAR_HEIGHT = 0.78
# 分类边框、分类之间的白色分隔和整根柱子的外框宽度，单位为磅。
SEGMENT_BORDER_WIDTH = 0.6
SEGMENT_SEPARATOR_WIDTH = 1.8
BAR_BORDER_WIDTH = 0.9


def integer(text):
    return int(text.replace(',', ''))


def core_count(section, label):
    match = re.search(rf'^\s*{re.escape(label)}:\s*{NUMBER} core-cycles', section, re.MULTILINE)
    if match is None:
        raise ValueError(f'日志缺少计数：{label}。请使用补齐全连接统计后的日志。')
    return integer(match[1])


def invocation_components(text, breakdown):
    """一次模拟调用按计算核心平均归因，各分项严格等于此次总周期。"""
    total, normal, lif, dram = map(integer, breakdown.groups())
    if total != normal + lif + dram:
        raise ValueError('总周期与正常计算、泄漏积分发放、访存之和不一致。')
    conv_marker = '[Conv 路径 Core 平均周期占比]'
    linear_marker = '[全连接路径计算核心互斥周期]'
    if conv_marker in text:
        section = text.split(conv_marker, 1)[1].split('[issue 空转统计]', 1)[0]
        split_label, lif_label = 'SplitUnit 供数不足', 'Accumulator/LIF 排空'
    elif linear_marker in text:
        section = text.split(linear_marker, 1)[1].split('[issue 空转统计]', 1)[0]
        split_label, lif_label = '取不到 mode0 包', '泄漏积分发放等待'
    else:
        raise ValueError('日志缺少全连接空闲/调度统计；请补齐统计并重跑，不能将缺失值当作零。')
    denominator = re.search(rf'分母:\s*{NUMBER} cycles x (\d+) cores = {NUMBER} core-cycles', section)
    if denominator is None:
        raise ValueError('日志缺少计算核心数量或周期分母。')
    backend, cores, count = map(integer, denominator.groups())
    counts = [integer(value) for value in re.findall(r':\s*([\d,]+) core-cycles', section)]
    if backend != normal + lif or count != backend * cores or sum(counts) != count:
        raise ValueError('互斥分类之和、周期分母或计算周期不一致。')
    if Fraction(core_count(section, lif_label), cores) != lif:
        raise ValueError('泄漏积分发放等待与总周期分解不一致。')
    values = dict.fromkeys(COUNTER_COMPONENTS, Fraction(0))
    values.update(DRAM=Fraction(dram), LIF=Fraction(lif))
    values['scheduler'] = Fraction(core_count(section, '空闲/调度'), cores)
    values['split_unit'] = Fraction(core_count(section, split_label), cores)
    values['accumulator'] = Fraction(core_count(section, 'Accumulator 反压等待'), cores)
    values['current_accumulating_sub_unit'] = (
        normal - values['scheduler'] - values['split_unit'] - values['accumulator'])
    if min(values.values()) < 0 or sum(values.values()) != total:
        raise ValueError('周期分类存在负数或重复计数。')
    return values


def parse_log(path, allow_network_summary_mismatch=False):
    text = path.read_text(encoding='utf-8-sig')
    cases = list(CASES.finditer(text))
    rows, warnings = [], []
    for index, case in enumerate(cases):
        block = text[case.end():cases[index + 1].start() if index + 1 < len(cases) else len(text)]
        layer_total = re.search(rf'总周期:\s*EVA\s*{NUMBER};', block)
        calls = list(BREAKDOWN.finditer(block))
        if layer_total is None:
            if calls:
                raise ValueError(f'测试用例 {case[1]} 缺少层总周期，日志可能未完成。')
            continue
        if not calls:
            raise ValueError(f'测试用例 {case[1]} 缺少模拟调用周期分解。')
        values = dict.fromkeys(COUNTER_COMPONENTS, Fraction(0))
        previous_end = 0
        # 自注意力含多次矩阵乘法，必须逐次累加，不能读取可能沿用上一层的末尾报告。
        for call in calls:
            try:
                current = invocation_components(block[previous_end:call.end()], call)
            except ValueError as error:
                raise ValueError(f'测试用例 {case[1]}：{error}') from error
            for name in COUNTER_COMPONENTS:
                values[name] += current[name]
            previous_end = call.end()
        total = integer(layer_total[1])
        if sum(values.values()) != total:
            raise ValueError(f'测试用例 {case[1]} 的模拟调用之和与层总周期不一致。')
        last_report = re.search(rf'EVA分解:\s*{NUMBER}\s*\+ LIF {NUMBER}\s*\+ DRAM {NUMBER}', block)
        if last_report:
            reported = tuple(map(integer, last_report.groups()))
            expected = (total - values['LIF'] - values['DRAM'], values['LIF'], values['DRAM'])
            if reported != expected:
                warnings.append(f'测试用例 {case[1]} 的末尾 EVA分解不一致，已使用内部模拟调用之和。')
        rows.append(dict(test_case=int(case[1]), layer=case[2], **values, total_cycles=total))
    if not rows:
        raise ValueError('没有找到完整的层周期报告。')
    total_row = dict(test_case='', layer='TOTAL',
                     **{key: sum(row[key] for row in rows) for key in COUNTER_COMPONENTS},
                     total_cycles=sum(row['total_cycles'] for row in rows))
    network_total = re.search(rf'^总EVA周期:\s*{NUMBER}\s*$', text, re.MULTILINE)
    if network_total is None:
        raise ValueError('日志缺少网络总周期，日志可能尚未完成。')
    if integer(network_total[1]) != total_row['total_cycles']:
        if not allow_network_summary_mismatch:
            raise ValueError('逐层之和与网络总周期不一致；请检查未报告的层或网络级修正。')
        warnings.append(f'网络末尾汇总 {integer(network_total[1]):,} 与分项合计 '
                        f'{total_row["total_cycles"]:,} 不一致；比例使用分项合计，未缩放原始计数。')
    return rows + [total_row], warnings


def save_csv(rows, path):
    # 用有理数完成核对后才转为小数；保留足够精度，避免平均周期逐层取整。
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=('test_case', 'layer', *COUNTER_COMPONENTS, 'total_cycles'))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (str(value.numerator) if value.denominator == 1
                                   else f'{float(value):.9f}') if isinstance(value, Fraction)
                             else value for key, value in row.items()})


def plot_total(total_row, prefix, network):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10})
    total = total_row['total_cycles']
    if total <= 0:
        raise ValueError('总周期必须为正数。')
    fig, axis = plt.subplots(figsize=(13, 3.0))
    left = 0.0
    for component in COMPONENTS:
        name = COMPONENT_KEYS[component]
        color = BREAKDOWN_COLORS[name]
        ratio = float(total_row[name]) / total * 100
        axis.barh(0, ratio, left=left, height=0.5, color=color, edgecolor='white',
                  linewidth=0.5, label=f'{COMPONENT_LABELS[name]}  {ratio:.2f}%')
        if ratio >= 4:
            axis.text(left + ratio / 2, 0, f'{ratio:.1f}%', ha='center', va='center', fontsize=10,
                      color=PERCENT_TEXT_COLOR, fontweight='bold')
        left += ratio
    label = dict(NETWORKS).get(network, network)
    axis.set(xlim=(0, 100), ylim=(-0.55, 0.65), yticks=[0], yticklabels=[label])
    axis.xaxis.set_major_formatter(PercentFormatter())
    axis.set_title(f'Total cycles: {total:,}', loc='left', fontsize=13, pad=56)
    axis.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.02),
                frameon=False, fontsize=10, columnspacing=1.4, handlelength=1.3)
    for side in ('top', 'right', 'left'):
        axis.spines[side].set_visible(False)
    axis.tick_params(axis='y', length=0)
    fig.tight_layout()
    for suffix in ('.png', '.svg'):
        fig.savefig(prefix.with_suffix(suffix), dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_networks(results, prefix, figure_dir=None):
    """以逐层分项之和为分母，绘制五个网络的计算核心平均周期归因。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import Patch, Rectangle

    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'], 'font.size': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    # 消融组合图的下半图采用相同柱高；将实际高度换算为英寸后匹配周期分解图。
    from output.lookahead.plot.plot_ablation_cycles_patterns import FIGURE_SIZE, STACK_HEIGHT
    reference, reference_axes = plt.subplots(2, 1, figsize=FIGURE_SIZE,
                                             gridspec_kw={"height_ratios": [1.1, 0.5]})
    reference.subplots_adjust(left=0.13, right=0.995, bottom=0.025, top=0.92, hspace=0.30)
    bar_height_inches = reference_axes[1].get_position().height * FIGURE_SIZE[1] * STACK_HEIGHT / 5.4
    plt.close(reference)
    axis_height_inches = bar_height_inches * len(results) / BAR_HEIGHT
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.035,
                           top=0.035 + axis_height_inches / FIGURE_HEIGHT)
    positions = list(range(len(results)))
    left = [0.0] * len(results)
    for component in COMPONENTS:
        name = COMPONENT_KEYS[component]
        values = [float(item['total'][name]) / item['total']['total_cycles'] * 100
                  for item in results]
        for index, value in enumerate(values):
            if value > 0:
                bars = axis.barh(index, value, left=left[index], height=BAR_HEIGHT,
                                 color=BREAKDOWN_COLORS[name],
                                 linewidth=SEGMENT_BORDER_WIDTH, edgecolor='black', zorder=3)
                bars[0].set_path_effects([
                    path_effects.Stroke(linewidth=SEGMENT_SEPARATOR_WIDTH, foreground='white'),
                    path_effects.Normal(),
                ])
            if value >= MIN_LABEL_PERCENT:
                axis.text(left[index] + value / 2, index,
                          f'{value:.{PERCENT_DECIMALS}f}%',
                          color=PERCENT_TEXT_COLOR,
                          ha='center', va='center', fontsize=PERCENT_FONT_SIZE,
                          fontweight='bold', zorder=5)
            left[index] += value
    for position in positions:
        axis.add_patch(Rectangle((0, position - BAR_HEIGHT / 2), 100, BAR_HEIGHT,
                                 fill=False, edgecolor='black', linewidth=BAR_BORDER_WIDTH,
                                 clip_on=False, zorder=4))
    axis.set_yticks(positions, [WORKLOAD_INITIALS[item['network']] for item in results])
    axis.set(xlim=(0, 100), ylim=(-0.5, len(results) - 0.5))
    axis.invert_yaxis()
    axis.xaxis.set_visible(False)
    axis.tick_params(axis='y', length=0, labelsize=WORKLOAD_FONT_SIZE, pad=2)
    for label in axis.get_yticklabels():
        label.set_fontweight('bold')
    for spine in axis.spines.values():
        spine.set_visible(False)
    handles = [Patch(facecolor=BREAKDOWN_COLORS[COMPONENT_KEYS[component]],
                     edgecolor='black', linewidth=SEGMENT_BORDER_WIDTH, label=component)
               for component in COMPONENTS]
    # 两行分别居中，完整分类名称无需通过缩小整张图片来容纳。
    for row, row_handles in enumerate((handles[:3], handles[3:])):
        figure.legend(handles=row_handles, ncol=3, loc='upper center',
                      bbox_to_anchor=(0.52, 1.0 - row * 0.16 / FIGURE_HEIGHT), borderaxespad=0,
                      frameon=False, prop={'size': LEGEND_FONT_SIZE, 'weight': 'bold'},
                      columnspacing=0.8, handlelength=0.8, handletextpad=0.3)
    prefixes = [prefix]
    if figure_dir is not None:
        figure_dir.mkdir(parents=True, exist_ok=True)
        prefixes.append(figure_dir / prefix.name)
    for destination in prefixes:
        for extension in ('.png', '.pdf', '.svg'):
            figure.savefig(destination.with_suffix(extension), dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close(figure)


def extract_all(log_root, output_dir, figure_dir=None):
    """从本轮五个网络的完整日志提取分项，记录来源和网络末尾汇总差异。"""
    results = []
    for network, label in NETWORKS:
        candidates = sorted((log_root / f'{network}_cluster_parallel').glob('dla*_simulation_log_*.txt'))
        if len(candidates) != 1:
            raise ValueError(f'{network} 需要唯一完整日志，找到 {len(candidates)} 个。')
        path = candidates[0]
        rows, warnings = parse_log(path, allow_network_summary_mismatch=network == 'TIM_NCARS')
        report = re.search(rf'^总EVA周期:\s*{NUMBER}\s*$',
                           path.read_text(encoding='utf-8-sig'), re.MULTILINE)
        results.append(dict(network=network, label=label, rows=rows, total=rows[-1],
                            reported_total_cycles=integer(report[1]), source_log=str(path.resolve()),
                            warnings=warnings))
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ('network', 'valid_layers', *COUNTER_COMPONENTS, 'total_cycles', 'reported_total_cycles',
              'difference_cycles', *(f'{name}_pct' for name in COUNTER_COMPONENTS), 'source_log', 'notes')
    with (output_dir / 'EVA_cycles_breakdown_summary.csv').open(
            'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in results:
            total = item['total']
            save_csv(item['rows'], output_dir / f'{item["network"]}_cycles_breakdown.csv')
            plot_total(total, output_dir / f'{item["network"]}_cycles_breakdown', item['network'])
            writer.writerow(dict(network=item['network'], valid_layers=len(item['rows']) - 1,
                                 **{name: f'{float(total[name]):.9f}' for name in COUNTER_COMPONENTS},
                                 total_cycles=total['total_cycles'],
                                 reported_total_cycles=item['reported_total_cycles'],
                                 difference_cycles=total['total_cycles'] - item['reported_total_cycles'],
                                 **{f'{name}_pct': f'{float(total[name]) / total["total_cycles"] * 100:.9f}'
                                    for name in COUNTER_COMPONENTS},
                                 source_log=item['source_log'], notes=' | '.join(item['warnings'])))
            print(f'{item["network"]}: {len(item["rows"]) - 1} layers; '
                  f'component_total={total["total_cycles"]:,}; summary={item["reported_total_cycles"]:,}')
            for warning in item['warnings']:
                print(warning)
    plot_networks(results, output_dir / 'EVA_cycles_breakdown', figure_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path, nargs='?', help='包含全连接互斥周期统计的完整日志')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / "cycles_breakdown")
    parser.add_argument('--network', default='TIM_NCARS')
    parser.add_argument('--all', action='store_true', help='绘制本轮五个网络的汇总图')
    parser.add_argument('--figure-dir', type=Path, help='额外保存五网络汇总图片的目录')
    parser.add_argument('--log-root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--allow-network-summary-mismatch', action='store_true',
                        help='保留分项计数，并明确记录其与网络末尾汇总的差异')
    args = parser.parse_args()
    if args.all:
        if args.log:
            parser.error('--all 与单个日志参数不能同时使用。')
        extract_all(args.log_root, args.output_dir, args.figure_dir)
        return
    if args.log is None:
        parser.error('请提供日志路径或使用 --all。')
    rows, warnings = parse_log(args.log, args.allow_network_summary_mismatch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f'{args.network}_cycles_breakdown'
    save_csv(rows, prefix.with_suffix('.csv'))
    plot_total(rows[-1], prefix, args.network)
    print(f'{len(rows)-1} layers; total_cycles={rows[-1]["total_cycles"]:,}')
    for warning in warnings:
        print(warning)
    print(prefix.with_suffix('.csv'))
    print(prefix.with_suffix('.png'))


if __name__ == '__main__':
    main()
