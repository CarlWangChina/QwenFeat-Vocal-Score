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
* **模型选择逻辑**：
* 当时是因为绝对分数的数据数量不够，所以使用了排序模型。
* **关于 `scripts/train/train_grl.py**`：虽然支持标量分数带对抗解耦，但其 MuQ 未参与训练且输出为五分类（需改造成 songeval 那样的连续数据），且与当前排序模型结构不同，故不在此任务中使用。



## 3. 详细操作步骤

在 `scripts/train/train_sort_audio.py` 脚本中进行以下修改：

1. **加载模型**（约第 **33** 行）：在此位置先加载之前版本MuQ打分模型权重。
2. **更换路径**（约第 **42**、**43** 行）：将两个路径更换为新生成的训练集和测试集路径。

## 4. 任务背景

目前已由数据同学处理好数据，音乐同学完成标分。任务是将这 2000 首原曲分离出的清唱（85-99 分）作为高分段，与之前的 KTV 唱歌数据（50-85 分）混合，以此优化此版本的打分效果。
