# EVA end-to-end experiments

本仓库包含 2026-11-06 端到端 EVA_only 实验的运行入口、1104 周期模型、共享依赖、五个网络清单及原有论文绘图脚本。

## 环境与输入

使用 Python 3.11，在仓库根目录执行 `python -m pip install -r requirements-end-to-end.txt`。依赖未锁定版本，应根据本机环境安装合适的 PyTorch；本次上传未重新运行完整网络回归。

输入张量不上传。请将 `conv/`、`atten/`、`other/` 放在仓库根目录，并保持 `model/test_outerproduct_simulator/` 下五个网络 CSV 的相对路径。CSV 是算子、形状和输入路径清单，不包含张量。现有工作簿及实验结果也不包含在本次脚本上传中。

## 运行五个完整工作负载

```powershell
python -B -u -X utf8 model/test_outerproduct_simulator/run_1104_eva_end_to_end.py --output-root output/20261104/EVA_only --skip-tim-reference
```

此入口调用 `simulator_runner.py`，先运行 TIM，然后并行运行其余四个完整清单，不使用前十层截断。固定配置为 4 Core、16 PU、12×12 tile、64 LIF、accumulator bundle buffer 深度 2、仲裁候选 2、一个 scheduler 派发通道、8 个 psum pool bank、四个 splitunit FIFO 各深度 4、模式 0/1/2。延迟换算主频为 500 MHz，`latency_ms = cycles / 500000`。各网络时间步参数为 TIM 10、QKFormer 16、SpikeBRGNet 5、EMSYOLO 5、SDTrack 4；保留激励实际时间步及运行器原有算子支持范围。

`--skip-tim-reference` 是为独立仓库增加的选项，仅跳过依赖原机器历史结果的比较，不改变周期模型和实验配置。没有此选项时，原脚本仍要求 `output/performance_analysis/nine_way_layers_core4_dispatch1_lif64_bundle2_20261106/`；`--resume-after-tim` 仅用于原批次已记录差异的续跑。历史参考不随本次上传。

正式长时间实验应以独立后台进程运行。Windows 示例（使用全新结果目录）：

```powershell
$evaPython = (Get-Command python).Source
$evaRoot = (Get-Location).Path
New-Item -ItemType Directory -Path output/20261104/EVA_only -Force | Out-Null
$evaProcess = Start-Process -FilePath $evaPython -WorkingDirectory $evaRoot -ArgumentList @('-B','-u','-X','utf8','model/test_outerproduct_simulator/run_1104_eva_end_to_end.py','--output-root','output/20261104/EVA_only','--skip-tim-reference') -WindowStyle Hidden -RedirectStandardOutput output/20261104/EVA_only/batch.stdout.log -RedirectStandardError output/20261104/EVA_only/batch.stderr.log -PassThru
Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -eq $evaProcess.Id -or $_.ParentProcessId -eq $evaProcess.Id } | Select-Object ProcessId,ParentProcessId,Name
Get-Content output/20261104/EVA_only/batch.stderr.log
```

启动后检查父进程、模拟子进程以及各网络 console 日志。`progress.json` 记录进度，`summary.json` 记录逐网络周期及有效层数，网络子目录包含逐层 CSV 和 `calls.jsonl`。`COMPLETE.json` 表示流程结束；成功还必须满足 `status=pass`、`finished_workloads=5`、`passed_workloads=5`、`failures=[]`。完成后 `eva_cycle_updates.json` 给出工作簿所需周期，脚本不会自动改写工作簿。

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

其他绘图任务需要各自实验数据，详见 `output/lookahead/README.md`。本次上传不修改 baseline 周期或绘图样式。`run_1104_workload_batch.py` 提供共用运行和校验函数，其独立运行默认参数不同；复现本次配置请使用上述 `run_1104_eva_end_to_end.py`。
