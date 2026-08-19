# KTV 重录打分实验归档-Cqy-所有改动都在audioscore目录下

## 数据与划分

完整数据池共 3,760 条：

| 数据来源 | 数量 | 分数范围 | 用途 |
| --- | ---: | ---: | --- |
| `2310` KTV 用户录音 | 2,310 | 30–92 | 1,925 条训练，385 条测试 |
| `2000_enhanced` KTV 重录扩充 | 1,450 | 59–99 | 只进入 mixed 训练集 |

“加入 GT85 数据”是加入 `2000/2000_enhanced` 重录来源的历史简称，不表示所有样本都大于 85 分。

| 设置 | 训练集 | 测试集 |
| --- | --- | --- |
| mixed | 3,375 条：1,925 KTV + 1,450 重录 | 385 条纯 KTV，30–84 分 |
| KTV-only | 1,925 条 KTV | 385 条纯 KTV，30–84 分 |

Pairwise 使用分组 JSON，直接回归使用相同样本的扁平 JSON：

| 用途 | Pairwise | 直接回归 |
| --- | --- | --- |
| mixed 训练 | `dataset/ktv_rerecord_3760/pairwise/train_mixed_3375.json` | `dataset/ktv_rerecord_3760/regression/train_mixed_3375.json` |
| KTV-only 训练 | `dataset/ktv_rerecord_3760/pairwise/train_ktv_only_1925.json` | `dataset/ktv_rerecord_3760/regression/train_ktv_only_1925.json` |
| 纯 KTV 测试 | `dataset/ktv_rerecord_3760/pairwise/test_ktv_lt85_385.json` | `dataset/ktv_rerecord_3760/regression/test_ktv_lt85_385.json` |

详细划分规则见 `dataset/ktv_rerecord_3760/split_note.md`。训练代码为兼容历史接口仍使用参数名 `--val_json`，本轮传入的是上述固定测试清单。

## 实验结果

下表只保留同一份 385 条纯 KTV 测试集上的结果；使用 mixed 测试集得到的约 `0.78` Pairwise accuracy 和约 `0.86` Pearson 不列入比较。Pairwise 均使用组内所有异分有序对。

| checkpoint | 初始化 | 训练集加入 mixed 重录数据 | 训练方法 | Pairwise 比较方向 | 训练步数 | Pearson |
| --- | --- | --- | --- | --- | ---: | ---: |
| `step_3_extended_audio/best_model_step_2000`（仅本地历史，未上传） | step2 续训 | 是：旧 mixed 3,758 | Pairwise | 反向：高分 raw `<` 低分 raw；推理用 `5 - raw` | 2,000 | 0.489811 |
| `pairwise_from_muq_mixed_step_9000` | 原始 MuQ + fresh LoRA | 是：mixed 3,375 | Pairwise | 正向：高分 raw `>` 低分 raw | 9,000 | 0.479695 |
| `regression_from_muq_mixed_r128_step_14200` | 原始 MuQ + fresh 回归头 | 是：mixed 3,375 | 直接回归 r128 | — | 14,200 | 0.4733 |
| `regression_from_muq_mixed_r32_step_13200` | 原始 MuQ + fresh 回归头 | 是：mixed 3,375 | 直接回归 r32 | — | 13,200 | 0.4813 |
| `regression_from_muq_ktv_only_r32_step_6600` | 原始 MuQ + fresh 回归头 | 否：KTV-only 1,925 | 直接回归 r32 | — | 6,600 | **0.514830** |
| `step_2_al_audio/best_model_step_132000`（线上旧版） | —：线上基线 | 原训练清单缺失 | Pairwise | 反向：高分 raw `<` 低分 raw；推理用 `5 - raw` | 132,000 | **0.640597** |

表中的 `r128` 和 `r32` 分别表示 MuQ 编码器使用的 LoRA rank 为 128 和 32；两者的回归头结构相同，`r` 不是回归头维度。

`step_2_al_audio/best_model_step_132000` 的原始训练清单缺失，因此无法排除上述 385 条纯 KTV 测试样本与其训练数据重叠的可能。该模型的 `0.640597` 只作为同集参考，不能视为已排除数据泄露的泛化结果。

## 最终方案

- 本轮新实验选用 KTV-only 直接回归 step 6,600，归档为 `step_3_ktv_rerecord_highscore`。
- 该模型是本轮新方案中 Pearson 最高的版本，但没有超过当前线上旧版的同集结果，因此不替换线上权重。

## 训练命令

以下命令均在 `audioscore/` 目录执行：

```bash
export AUDIOSCORE_ARCHIVE=/archive/private_data/cqy/audio-score
```

Pairwise mixed：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 --nnodes=1 \
  scripts/train/train_sort_audio.py \
  --train_json dataset/ktv_rerecord_3760/pairwise/train_mixed_3375.json \
  --val_json dataset/ktv_rerecord_3760/pairwise/test_ktv_lt85_385.json \
  --cache_base_dir /cache/cqy_cache/audio-score \
  --cache_dir "$AUDIOSCORE_ARCHIVE/pre_store_3760_full_24k" \
  --segment_duration 60 --epochs 10
```

最终 KTV-only 直接回归：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 --nnodes=1 \
  scripts/train/train_regression_audio.py \
  --train_json dataset/ktv_rerecord_3760/regression/train_ktv_only_1925.json \
  --val_json dataset/ktv_rerecord_3760/regression/test_ktv_lt85_385.json \
  --cache_base_dir /cache/cqy_cache/audio-score \
  --cache_dir "$AUDIOSCORE_ARCHIVE/pre_store_3760_full_24k" \
  --epochs 50 --lora_r 32 --lora_alpha 32 \
  --warmup_max_steps 500 --segment_duration 90
```

## 推理命令

当前线上旧版：

```bash
python scripts/train/infer_sort_audio.py \
  --model_path ckpts/SongEvalGenerator/step_2_al_audio/best_model_step_132000 \
  --val_json dataset/ktv_rerecord_3760/pairwise/test_ktv_lt85_385.json \
  --cache_base_dir /cache/cqy_cache/audio-score \
  --val_cache_dir "$AUDIOSCORE_ARCHIVE/pre_store_3760_full_24k" \
  --val_cache_mode read --val_cache_kind full \
  --val_cache_cut_mode stable --segment_duration 90 --seed 1234 --inverse
```

最终 Pairwise：

```bash
python scripts/train/infer_sort_audio.py \
  --model_path ckpts/SongEvalGenerator/ktv_rerecord_experiments/pairwise_from_muq_mixed_step_9000 \
  --val_json dataset/ktv_rerecord_3760/pairwise/test_ktv_lt85_385.json \
  --cache_base_dir /cache/cqy_cache/audio-score \
  --val_cache_dir "$AUDIOSCORE_ARCHIVE/pre_store_3760_full_24k" \
  --val_cache_mode read --val_cache_kind full \
  --val_cache_cut_mode stable --segment_duration 90 --seed 1234
```

最终直接回归：

```bash
python scripts/train/infer_regression_audio.py \
  --model_path ckpts/SongEvalGenerator/step_3_ktv_rerecord_highscore \
  --val_json dataset/ktv_rerecord_3760/regression/test_ktv_lt85_385.json \
  --cache_base_dir /cache/cqy_cache/audio-score \
  --val_cache_dir "$AUDIOSCORE_ARCHIVE/pre_store_3760_full_24k" \
  --val_cache_cut_mode stable --segment_duration 90 --seed 1234
```

## Checkpoint 下载

### 最终 checkpoint

本轮最终 checkpoint 是 KTV-only 直接回归 `step_3_ktv_rerecord_highscore`：

```text
audioscore/ckpts/SongEvalGenerator/step_3_ktv_rerecord_highscore/
├── config.json
├── muq_lora.pt
├── regression_head.pt
└── type.txt
```

Hugging Face 下载地址：

<https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/tree/main/audioscore/ckpts/SongEvalGenerator/step_3_ktv_rerecord_highscore>

在项目根目录运行以下命令，下载后文件将位于本地
`audioscore/ckpts/SongEvalGenerator/step_3_ktv_rerecord_highscore/`：

```bash
hf download karl-wang/QwenFeat-Vocal-Score \
  --include "audioscore/ckpts/SongEvalGenerator/step_3_ktv_rerecord_highscore/*" \
  --local-dir .
```

### 实验 checkpoint

除未上传的 step2 续训历史模型外，其余本轮 LoRA 实验 checkpoint 归档在：

<https://huggingface.co/karl-wang/vocalverse-highscore-expand-train-cqy/tree/main/audioscore/ckpts/SongEvalGenerator/ktv_rerecord_experiments>

在项目根目录运行：

```bash
hf download karl-wang/vocalverse-highscore-expand-train-cqy \
  --include "audioscore/ckpts/SongEvalGenerator/ktv_rerecord_experiments/**" \
  --local-dir .
```

下载后的本地目录为：

```text
audioscore/ckpts/SongEvalGenerator/ktv_rerecord_experiments/
├── pairwise_from_muq_mixed_step_9000/
├── regression_from_muq_mixed_r128_step_14200/
├── regression_from_muq_mixed_r32_step_13200/
└── regression_from_muq_ktv_only_r32_step_6600/
```

