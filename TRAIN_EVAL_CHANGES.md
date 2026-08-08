# UniMRG 训练与评测记录

## 五数据集 Single-Test 结果（2026-08-03）

以下结果使用 VLMEvalKit 官方答案提取器，并统一为逐样本百分比。禁止使用自定义宽松解析结果替换本表。

| 测试脚本 | 模型 | MMBench_DEV_EN | RealWorldQA | MMVP | HallusionBench | VSR-zeroshot | Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| `eval_unimrg_ar10k.sh` | HarmonAR10k | 63.78 | 55.42 | 61.33 | 43.53 | 60.07 | **56.83** |
| `eval_unimrg_dllm.sh` | HarmonDLLM10k_BD | **64.06** | 50.98 | 57.33 | 42.59 | 53.19 | 53.63 |
| `eval_unimrg_dllm_lora.sh` | HarmonLoRA10k_BD | 63.57 | 51.24 | **62.33** | 40.80 | 58.10 | 55.21 |
| `eval_unimrg_lora_opt20k.sh` | HarmonLoRAOpt20k_BD | 63.78 | 53.20 | 61.00 | 43.64 | 59.00 | 56.12 |
| `eval_unimrg_lora_opt_mlp20k.sh` | HarmonLoRAOptMlp20k_BD | 58.60 | 48.63 | 60.00 | **46.79** | 58.18 | 54.44 |

### 固定评测口径

- Judge 固定为 `exact_matching`，答案提取只使用当前仓库 VLMEvalKit 官方实现，不增加首字母、`Answer: X`、选项文本或 Yes/No 的自定义宽松补偿。
- **MMBench_DEV_EN**：使用官方 `can_infer` 对原始预测文件的 4329 条记录逐条计分，即 `vanilla_all`；不使用 circular `Overall`。
- **RealWorldQA**：使用官方 `*_RealWorldQA_acc.csv` 的 `Overall`。
- **MMVP**：使用官方 `*_MMVP_acc.csv` 的 `Average`；不使用 pairwise `Overall`。
- **HallusionBench**：使用官方 `*_HallusionBench_score.csv` 中 `split=Overall` 的 `aAcc`。
- **VSR-zeroshot**：使用官方 `*_VSR-zeroshot_score.csv` 的 `acc`。
- **Avg**：先对五个未四舍五入的百分比分数做等权宏平均，再显示两位小数。

### 当前结果来源

- HarmonAR10k：`/nvmedata/xiexu/uni/work_dirs/eval_unimrg_ar10k/HarmonAR10k/T20260802-145256/`
- HarmonDLLM10k_BD：`/nvmedata/xiexu/uni/work_dirs/eval_unimrg_dllm/HarmonDLLM10k_BD/T20260802-151229/`
- HarmonLoRA10k_BD：`/nvmedata/xiexu/uni/work_dirs/eval_unimrg_dllm_lora/HarmonLoRA10k_BD/T20260802-154255/`
- HarmonLoRAOpt20k_BD：`/nvmedata/xiexu/uni/work_dirs/eval_unimrg_lora_opt15k/HarmonLoRAOpt20k_BD/T20260802-162221/`
- HarmonLoRAOptMlp20k_BD：`/nvmedata/xiexu/uni/work_dirs/eval_unimrg_lora_opt_mlp20k/HarmonLoRAOptMlp20k_BD/T20260802-170450/`

说明：LoRAOpt20k 的历史结果目录误命名为 `eval_unimrg_lora_opt15k`，但内部模型名和 checkpoint 均为 20k。当前评测脚本已把默认目录修正为 `eval_unimrg_lora_opt20k`。

### 非规范结果

旧汇总曾使用自定义宽松解析，将 HarmonLoRAOptMlp20k_BD 的 MMBench single-test 分数从官方口径的 `58.60` 提高到 `63.20`。该结果只反映解析规则变化，不用于模型横向比较。

## 可追溯评测入口

五个模型脚本现在统一调用 `Harmon/scripts/eval_unimrg_common.sh`。正常运行示例：

```bash
bash Harmon/scripts/eval_unimrg_ar10k.sh
bash Harmon/scripts/eval_unimrg_dllm.sh
bash Harmon/scripts/eval_unimrg_dllm_lora.sh
bash Harmon/scripts/eval_unimrg_lora_opt20k.sh
bash Harmon/scripts/eval_unimrg_lora_opt_mlp20k.sh
```

每次运行写入独立目录：

```text
<WORK_ROOT>/runs/<UTC时间戳-PID>/
```

可通过 `RUN_ID` 指定稳定标识，通过 `WORK_ROOT`、`CHECKPOINT`、`MODEL_CONFIG`、`LMU_ROOT`、`CUDA_VISIBLE_DEVICES` 等环境变量覆盖默认值。若指定的 `RUN_ID` 已存在，脚本直接失败，不覆盖旧结果。

每个运行目录保存：

- 实际 `config.json`、`command.txt`、`eval.log` 和 `status.json`；
- Git HEAD/status/diff、Python/CUDA/GPU/依赖信息；
- wrapper、公共 runner、模型配置、汇总器和官方匹配器的源码副本及 SHA256；
- checkpoint 与五个数据文件的 SHA256；
- VLMEvalKit 原始结果文件清单；
- 自动生成的 `single_test_results.json`、`.csv`、`.md` 和 `result_inputs.sha256`。

只检查配置和溯源产物而不启动 GPU 推理：

```bash
DRY_RUN=1 RUN_ID=check-config \
  WORK_ROOT=/tmp/unimrg-eval-check \
  bash Harmon/scripts/eval_unimrg_lora_opt20k.sh
```

对已有的单次完整运行重新生成规范汇总：

```bash
python Harmon/scripts/summarize_single_test.py \
  --run-dir /path/to/one/complete/run \
  --model-name HarmonLoRAOpt20k_BD \
  --output-dir /path/to/summary
```

汇总器要求每个数据集恰好存在一份原始预测和一份对应官方评分文件，并校验固定样本数。缺失文件、重复文件、样本数变化、非有限分数或字段变化均返回非零退出码。
