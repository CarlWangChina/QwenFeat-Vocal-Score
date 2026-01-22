# 人声录音打分模型  
## 使用方法：  

### 安装环境  

```bash
conda create -n audioscore python=3.10
conda activate audioscore
pip install -r requirements.txt
```

### 下载模型  
下载下面的目录，放在当前位置  
https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/tree/main/audioscore/ckpts  
直接git clone [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score](https://huggingface.co/karl-wang/QwenFeat-Vocal-Score) 也可以

### 调用  
(可执行的代码见`python tests/test_generate_score.py`)  

```python
import os,sys
import audioscore.model

if __name__=="__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = audioscore.model.SongEvalGenerator_audio_lora()
    model.load_model(os.path.join(ROOT_DIR,"ckpts", "SongEvalGenerator", "step_2_al_audio", "best_model_step_132000")) #加载模型
    model = model.half() #可选
    model = model.cuda()

    score = model.generate_tag("/data/nfs/audioscore/data/sort/data/audio/203887_250822105518005501.m4a") #执行打分

    print("score:", score)

```

可以直接执行下面的代码进行测试  

```bash
python tests/test_generate_score.py
```

## 训练  

直接训练  
`torchrun --nproc_per_node=4 --nnodes=1  scripts/train/train_sort_audio.py`  

对抗训练（使用samoye的spk encoder进行解耦）  
`torchrun --nproc_per_node=4 --nnodes=1  scripts/train/train_sort_audio_grl.py`  

# MuQ 歌唱打分模型：增加数据与扩大训练指南

本说明用于记录如何增加原曲分离清唱数据（高分段）以优化打分模型，供备份参考。

## 1. 数据处理格式

需将新增数据与原有数据混合，处理成 `data/sort/data/index/train.json` 格式。

* **逻辑**：第一级是一组数据，第二级按 **分数从高到低** 排序。
* 注意（关于分组逻辑）：
1. 绝对分数逻辑：由于当前数据采用的是“绝对分数”（即用户给出的客观打分，而非组内相对排序），该分数在任何组别下含义一致。
2. 取消多组划分：在处理新数据时，无需再按批次（如 KTV 批次）划分为多个第一级列表。只需将所有数据混合，整体按分数从高到低排序后放入同一个组（即 JSON 的第一层级仅保留一个主列表）即可。
3. 历史背景：早期版本采用多组划分（Pairwise 训练）是因为当时各组指标不统一，现在统一为绝对分数后，合并为一组训练更有利于模型学习全局分布。

* **JSON 示例**（严格遵循原始格式，注意字段末尾的英文逗号 `,`）：

```json
[
    [
        {
            "audio_file": "/data/nfs/audioscore/data/sort/data/audio/20732755_final.m4a",
        },
        {
            "audio_file": "/data/nfs/audioscore/data/sort/data/audio/463674_250811122315010877.m4a",
        }
    ]
]

```

## 2. 训练脚本说明

本次任务使用 **排序模型** 进行训练，不修改模型结构，仅扩大训练数据。

* **训练代码**：`scripts/train/train_sort_audio.py`
* 关键注意事项（收敛问题）：
1. 数据顺序反转：在代码实现与 README 文档中，关于“高到低”还是“低到高”可能存在表述冲突。根据实测经验，必须严格按照分数从高到低排列。
2. 避坑指南：前期测试发现，如果按分数“正向”排列（低到高），训练过程会出现不收敛的情况。若遇到模型不收敛，请优先检查 JSON 中的音频序列是否已按高分在前的方式重排。
* **模型选择逻辑**：
* 当时是因为绝对分数的数据数量不够，所以使用了排序模型。
* **关于 `scripts/train/train_grl.py**`：虽然支持标量分数带对抗解耦，但其 MuQ 未参与训练且输出为五分类（需改造成 songeval 那样的连续数据），且与当前排序模型结构不同，故不在此任务中使用。

### **模型保存说明**

在 `src/audioscore/model.py` 第 **1787** 行附近，如果硬盘空间充足，建议改为保存完整的模型权重，以防止某些情况下数据保存不全。

**修改方案如下：**

```python
# 只保存 LoRA 权重 (非常小)
# lora_sd = get_lora_state_dict(self.muq)
# torch.save(lora_sd, os.path.join(path, "muq_lora.pt"))

# 替换为保存完整的模型 state_dict
torch.save(self.muq.state_dict(), os.path.join(path, "muq_lora.pt"))
```

## 3. 详细操作步骤

在 `scripts/train/train_sort_audio.py` 脚本中进行以下修改：

1. **加载模型**（约第 **33** 行）：在此位置先加载之前版本MuQ打分模型权重。推荐权重路径：
建议加载 step_2_al_audio/best_model_step_132000 版本的模型，该版本在实测中表现更优。
弃用旧版权重（如 step_1_mul_audio_grl/best_model_step_74000）。
2. **更换路径**（约第 **42**、**43** 行）：将两个路径更换为新生成的训练集和测试集路径。

## 4. 任务背景

目前已由数据同学处理好数据，音乐同学完成标分。任务是将这 2000 首原曲分离出的清唱（85-99 分）作为高分段，与之前的 KTV 唱歌数据（50-85 分）混合，以此优化此版本的打分效果。
数据配比建议：
组内样本个数不强制要求统一，模型支持非固定长度的组内序列训练。
核心目标是通过引入高分段（85分以上）的清唱数据，补全模型在高质量歌声判别上的短板，使模型能够建立起从 50 分到 99 分的完整评分尺度。

# 任务说明： CAM++和MuQ多特征融合音色打分器训练

## 1. 背景与动机
根据工程侧测试反馈，数据标注分数与 3D-Speaker 项目中 CAM++ 模型（达摩院开源）提取的音色特质呈现极强的相关性。为了提升打分器的准确度，拟引入 CAM++ 特征进行融合实验。

## 2. 实验设计
本项目需要维护并对比以下两种方案：
- **方案 A (Baseline)**: 使用 MuQ 作为单一特征提取器。
- **方案 B (Hybrid)**: 使用 MuQ 特征与 CAM++ 特征进行拼接（Concat）作为联合特征提取器。

## 3. 技术细节
- **特征提取**:
  - MuQ: 原有特征提取流程。
  - CAM++: 采用 3D-Speaker 库（https://github.com/modelscope/3D-Speaker）。
- **融合方式**: 特征向量水平拼接。
- **训练策略**: 
  - 打分器（Scorer）权重需从零开始训练（Scratch），以适配新的特征维度与表达。

