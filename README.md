# EVA end-to-end experiments

本仓库包含 2026-11-06 端到端 EVA_only 实验的运行入口、1104 周期模型、共享依赖、五个网络清单及原有论文绘图脚本。

## 环境与输入

使用 Python 3.11，在仓库根目录执行 `python -m pip install -r requirements-end-to-end.txt`。依赖未锁定版本，应根据本机环境安装合适的 PyTorch；本次上传未重新运行完整网络回归。

输入张量不上传。请将 `conv/`、`atten/`、`other/` 放在仓库根目录，并保持 `model/test_outerproduct_simulator/` 下五个网络 CSV 的相对路径。CSV 是算子、形状和输入路径清单，不包含张量。现有工作簿及实验结果也不包含在本次脚本上传中。

## 运行五个完整工作负载

统一入口是 `model/test_outerproduct_simulator/simulator_runner.py`，通过 `--accumulator-variant 1104` 直接调用 `zxy/20261104/model/simulator.py`。五个网络都使用完整清单，不传 `--max-test-cases`。

固定配置为 4 Core、16 PU、12×12 tile、64 LIF、accumulator bundle buffer 深度 2、仲裁候选 2、一个 scheduler 派发通道、8 个 psum pool bank、四个 splitunit FIFO 各深度 4、模式 0/1/2。延迟换算主频为 500 MHz，`latency_ms = cycles / 500000`。各网络时间步参数为 TIM 10、QKFormer 16、SpikeBRGNet 5、EMSYOLO 5、SDTrack 4；保留激励实际时间步及运行器原有算子支持范围。

下面的 PowerShell 命令直接为每个网络启动独立的 `simulator_runner.py` 后台进程，无需额外 Python 编排脚本。结果目录必须为全新目录，已有本轮结果时请更换 `$evaOutput`：

```powershell
$evaPython = (Get-Command python).Source
$evaRoot = (Get-Location).Path
$evaOutput = 'output/20261104/EVA_only'
if (Test-Path -LiteralPath $evaOutput) { throw 'Use a new result directory' }
New-Item -ItemType Directory -Path $evaOutput -Force | Out-Null
$env:EVA_DISPATCH_LANES = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$evaCommon = @(
    '-B','-u','-X','utf8','model/test_outerproduct_simulator/simulator_runner.py',
    '--simulator','cluster_parallel','--accumulator-variant','1104','--eva-only',
    '--num_cores','4','--num_pus','16','--bank_h','12','--bank_w','12',
    '--num_split','2','--num_clusters','1','--fetch_rows','8',
    '--split_fifo_depth','4','--psum_pool_rows','8','--retire_column','3',
    '--retire-packets-per-cycle','2','--prefetch-cycles','2',
    '--bundle-buffer-depth','2','--bank-lookahead-entries','2','--lif-units','64',
    '--num_reduce_trees','0','--enabled_modes','0,1,2','--B','1',
    '--output-root',$evaOutput
)
$evaNetworks = [ordered]@{
    TIM_NCARS = 10
    QKFormer_DVSGesture = 16
    SpikeBRGNet_DDD17 = 5
    SDTrack_FE108 = 4
    EMSYOLO_GEN1 = 5
}
$evaProcesses = foreach ($entry in $evaNetworks.GetEnumerator()) {
    $evaArguments = $evaCommon + @('--network',$entry.Key,'--T',[string]$entry.Value)
    Start-Process -FilePath $evaPython -WorkingDirectory $evaRoot -ArgumentList $evaArguments -WindowStyle Hidden -RedirectStandardOutput "$evaOutput/$($entry.Key).stdout.log" -RedirectStandardError "$evaOutput/$($entry.Key).stderr.log" -PassThru
}
Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in $evaProcesses.Id } | Select-Object ProcessId,ParentProcessId,Name,CommandLine
Get-Content "$evaOutput/*.stderr.log"
Get-Content "$evaOutput/*.stdout.log" -Tail 8
```

如只运行一个网络，保留公共参数和环境变量，只对相应网络执行一次 `Start-Process`。需要先核对 TIM 时，先仅启动 TIM，核对后再启动其余四个网络。

每个网络结果位于 `<output-root>/<network>_1104/`，包含 `manifest.json`、`calls.jsonl`、`performance_summary.json`、逐层结果 CSV 和模拟日志。日志中的测试用例编号表示当前进度。完成需同时核对进程退出码为零、日志出现“仿真任务完成”、网络 `COMPLETE.json` 的 `status=pass`，以及调用数和逐层结果。五个网络需要分别核对。直接运行不会生成批次级 `progress.json`、`summary.json` 或批次级 `COMPLETE.json`，也不会自动更新工作簿。端到端周期取各网络完成日志中的“总EVA周期”，TIM 不能直接以逐调用周期之和替代。

## 已完成实验的端到端周期

| Workload | EVA cycles |
| --- | ---: |
| TIM_NCARS | 473025 |
| QKFormer_DVSGesture | 2971535 |
| SpikeBRGNet_DDD17 | 3456406 |
| SDTrack_FE108 | 40655684 |
| EMSYOLO_GEN1 | 14925184 |

TIM 保留网络末尾的 −417 池化修正。历史参考为 474113；最新模型将 76 次全零输入调用的片外访存开销归零，减少 1088 周期，计算和 LIF 周期不变。

## 绘图

`output/lookahead/plot_all_figures.py` 与 `output/lookahead/plot/` 按原样保留。准备包含已有 baseline 数据的工作簿，只更新 EVA 周期后执行：

```powershell
python output/lookahead/plot_all_figures.py --only end_to_end --workbook output/20261106/end_to_end.xlsx
```

其他绘图任务需要各自实验数据，详见 `output/lookahead/README.md`。本次上传不修改 baseline 周期或绘图样式。
