# Audioscore Dataset JSON 说明

本目录只保留当前 MuQ 歌唱打分排序模型需要的数据 JSON。旧版 `4176`、`4176_90_10`、`example` 以及低到高重复训练集已清理。

## 保留文件

| 路径 | 用途 | 数据形态 | 说明 |
| --- | --- | --- | --- |
| `4171.json` | 当前权威全量清单 | 1 组，4171 条，高分到低分 | 由旧 `4176` 去重得到，分数范围 30-99。用于重新切分或核对全集，不直接作为默认训练入口。 |
| `grouped_4171_95_5_h2l/train_4171_95.json` | 默认训练集 | 112 组，3960 条，每组高分到低分 | 当前推荐训练 JSON。对应 4171 全量数据的 95% train split。 |
| `grouped_4171_95_5_h2l/val_4171_05.json` | 默认验证集 | 6 组，211 条，每组高分到低分 | 当前推荐验证 JSON。对应 4171 全量数据的 5% val split。 |
| `grouped_4171_95_5/val_4171_05_cleaned_lt85.json` | 低分段验证诊断 | 6 组，124 条，低分到高分 | 从验证集拆出的 `<85` 分样本，用于单独观察低分段表现。不是默认训练/验证入口。 |
| `grouped_4171_95_5/val_4171_05_cleaned_gt85.json` | 高分段验证诊断 | 6 组，81 条，高分到低分 | 从验证集拆出的 `>85` 分样本，用于单独观察高分段表现。不是默认训练/验证入口。 |

## 数据格式

当前排序模型使用 grouped JSON：

```json
[
  [
    {
      "audio_id": 467614,
      "score": 98.0,
      "audio_file": "/cache/cqy_cache/audio-score/2000/467614.m4a"
    }
  ]
]
```

字段含义：

- `audio_id`：音频 ID。
- `score`：人工标注绝对分。
- `audio_file`：音频文件绝对路径。

训练代码当前按 `score_i > score_j` 生成 pairwise label，因此 label 不依赖 JSON 顺序；但为了和 README 中的扩大训练指南保持一致，默认训练/验证集仍保留高分到低分的 `_h2l` 版本。

## 默认入口

`scripts/train/train_sort_audio.py` 当前默认使用：

- train: `dataset/grouped_4171_95_5_h2l/train_4171_95.json`
- val: `dataset/grouped_4171_95_5_h2l/val_4171_05.json`

`scripts/cache_audio_features.py` 的默认验证缓存 JSON 也已切到同一个验证集。
