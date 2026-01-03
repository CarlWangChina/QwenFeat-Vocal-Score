# VocalVerse Project Overview

This work sincerely acknowledges the support of previous works such as QwenAudio, SongEval, and MuQ.

## 📜 License & Copyright

1. **License Framework**: This project is licensed under the **[CC BY-NC-ND 4.0](http://creativecommons.org/licenses/by-nc-nd/4.0/) (Attribution-NonCommercial-NoDerivs 4.0 International)** license.
2. **Non-Commercial Use**: The code, model weights, and documentation in this project are free for academic exchange and personal educational purposes only.
3. **Commercial Use Strictly Prohibited**: Without written authorization, it is strictly forbidden to use any part of this project for any form of commercial profit (including but not limited to integration into commercial software, providing commercial AI services, etc.).
4. **Commercial Licensing Inquiries**: For commercial cooperation or to obtain a commercial license, please contact: **3156018231@qq.com**.

---

## 🌍 Version Notice & Legacy Archive

**Legacy Repository**:  
[https://github.com/CarlWangChina/Singing-Aesthetic-Assessment](https://github.com/CarlWangChina/Singing-Aesthetic-Assessment)


## 📖 Paper Citation & Instructions

For detailed technical solutions, experimental results, and theoretical support regarding this implementation, please refer to our research paper. If you use the code or models from this repository in your research or work, please cite our paper.

### Paper Information

**Title**: Singing Timbre Popularity Assessment Based on Multimodal Large Foundation Model
**Conference**: Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)

* **ACM Digital Library (Official Version)**: [https://doi.org/10.1145/3746027.3758148](https://doi.org/10.1145/3746027.3758148)
* **arXiv (Free Preview Version)**: [https://www.arxiv.org/abs/2512.06999](https://www.arxiv.org/abs/2512.06999) 
  *(Note: The arXiv version is identical in content to the official ACM DL version)*

### Citation Format

#### ACM Reference Format
> Zihao Wang, Ruibin Yuan, Ziqi Geng, Hengjia Li, Xingwei Qu, Xinyi Li, Songye Chen, Haoying Fu, Roger B. Dannenberg, and Kejun Zhang. 2025. Singing Timbre Popularity Assessment Based on Multimodal Large Foundation Model. In Proceedings of the 33rd ACM International Conference on Multimedia (MM '25). Association for Computing Machinery, New York, NY, USA, 12227–12236. https://doi.org/10.1145/3746027.3758148

#### BibTeX
```bibtex
@inproceedings{10.1145/3746027.3758148,
author = {Wang, Zihao and Yuan, Ruibin and Geng, Ziqi and Li, Hengjia ox and Qu, Xingwei and Li, Xinyi and Chen, Songye and Fu, Haoying and Dannenberg, Roger B. and Zhang, Kejun},
title = {Singing Timbre Popularity Assessment Based on Multimodal Large Foundation Model},
year = {2025},
isbn = {9798400720352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {[https://doi.org/10.1145/3746027.3758148](https://doi.org/10.1145/3746027.3758148)},
doi = {10.1145/3746027.3758148},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {12227–12236},
numpages = {10},
keywords = {computational music aesthetics, descriptive feedback, multi-dimensional evaluation, multimodal foundation models, singing timbre popularity, singing voice assessment},
location = {Dublin, Ireland},
series = {MM '25}
}
```

## 📂 Data Description

The VocalVerse_Datasets-human_labels folder contains three core files, providing two-level evaluation data (amateur and professional) for singing performance.

| File Name | Description |
| :--- | :--- |
| **`Amateur_overall_mos_avg5.xlsx`** | **Amateur Consensus Scores:** Contains overall "pleasantness" ratings from 165 amateur annotators. Each recording was rated by 5 independent annotators on a 1-5 Likert scale, and this file provides the final Mean Opinion Scores (MOS). |
| **`Professional_multidim_annotations_raw_...xlsx`** | **Expert Multi-dimensional Labels:** Detailed annotations provided by two professional vocal coaches. It includes 1-5 integer scores across four core dimensions—**Timbre, Breath, Emotion, and Technique**—along with accompanying **textual critiques**. |
| **`Professional_scoring_rubric.xlsx`** | **Scoring Standards:** The formal criteria (Rubric) used by experts to ensure consistency and high quality across the four dimensions. |

### Data Scale

According to the research paper:
* **Original Data Pool:** We initially collected over **100,000** raw a cappella recordings.
* **Pre-screened Set:** Following automated preliminary screening via manual and RuleSignal scoring, a dataset of **10,000** clips was formed.
* **Current Open-source Subset:** The files provided in this repository represent the **top 10% (approximately 1,000 recordings with high technical proficiency)**, which were subsequently subjected to intensive professional multi-dimensional annotation.

### Annotation Methodology

1. **Amateur Phase:** A total of 165 non-music major annotators participated. They employed a "forced distribution" method to ensure score differentiation and capture general aesthetic preferences.
2. **Professional Phase:** Two senior vocal teachers provided dual-modality annotations (scores + descriptive text) to support the training of descriptive Multimodal Large Language Models (MLLMs).
3. **Evaluation Dimensions:**
    * **Timbre Quality:** The uniqueness, texture, and layering of the voice.
    * **Breath Control:** Support and stability for complex phrases.
    * **Emotional Expression:** The infectiousness and resonance of the performance.
    * **Vocal Technique:** Mastery of singing skills.


# VocalVerse1: Singing Evaluation Model based on QwenAudio

[qwenaudio](./qwenaudio/README.md) **Qwen Comment Generation + Scoring Module**: This includes a Lora-trained version of Qwen-audio. It takes audio as input and outputs comments on issues in the singing voice (equivalent to descriptive tagging). These comments are then used as input for a "deep thinking" phase to perform the final timbre scoring. Finally, a TTS system with a singer's voice is used to generate the vocal critique. The full workflow is:

1. The fine-tuned Qwen model generates comments on singing issues.
2. Comments and audio are fed back into the Qwen scoring model to assist in evaluation (acting as a "deep thinking" step).
3. Another LLM call polishes the comments, generates a summary, and provides vocal suggestions.
4. Finally, the summary is read aloud using the corresponding singer's voice.

All weights for this section are preserved. Model directory link: Download the full repository containing models from Hugging Face: [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/] 


## Usage

### Download Models
Download the following directory and place it in the current path:
https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/tree/main/qwenaudio/ckpts
Or simply run: `git clone https://huggingface.co/karl-wang/QwenFeat-Vocal-Score`

### Start Service

`python scripts/infer_service.py`  
Call via CURL:  
`curl -X POST http://localhost:8080/score -F "file=@/path/to/audio"`

### CLI Usage
`python scripts/infer.py path_to_audio.wav output_file.txt`

### Python API

Refer to `tests/test_score.py` for implementation.

```python
import qwenaudio.processor
import qwenaudio.prompts
import librosa

# Initialize model and processor
processor = qwenaudio.processor.create_processor(
            "ckpts/generator-lora-32-16-scoreonly-f16/best_model_epoch_13/lora_weights",
            "ckpts/generator-lora-32-16-textonly-simple-v2-int4/best_model_epoch_16/lora_weights"
        )

# Scoring
audio_path = "/home/w-4090/cutted_score_audio_separated/446892/344523004.wav"
data, sr = librosa.load(audio_path, sr=processor.processor.feature_extractor.sampling_rate, mono=True)
data = data[:processor.processor.feature_extractor.sampling_rate * 30] # Use first 30 seconds
print(data.shape)

for i in range(4):
    print("test:", qwenaudio.prompts.prompt_mapper_reverse[i]) # Meanings of parameters 0-4
    score = processor.generate(data, i) # Generate score
```

If model loading hangs, try using a Hugging Face mirror:
`export HF_ENDPOINT=https://hf-mirror.com`

# VocalVerse2: Vocal Recording Scoring Model based on MuQ

[audioscore](./audioscore/README.md) **MuQ Scoring & Ranking Module**: Contains two versions (with and without decoupling) using the same codebase, organized into a single directory.

1. **Scoring using MuQ as encoder + scoring head**: No decoupling. This architecture is basically the same as the SongEval work. The unfrozen MuQ Lora weights and the scoring head weights are preserved. Link: [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/] 

2. **Decoupling Experiment**: Uses the speaker encoder from SaMoye-SVC for reverse gradient training to decouple speaker identity features, aiming to improve aesthetic understanding accuracy. The code is compatible with both versions. Note: Decoupling weights were lost due to a server move. However, experiments using SaMoye’s or Wespeaker’s encoder for reverse gradient training showed that while training and convergence are difficult (easier with smaller batch sizes), the final aesthetic assessment accuracy slightly improved, proving that decoupling is effective.

**Note**: This repository does not contain the model weights. Researchers should download the full repository including models from Hugging Face: [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/] 

## Usage

### Environment Setup

```bash
conda create -n audioscore python=3.10
conda activate audioscore
pip install -r requirements.txt
```

### Download Models
Download the following directory and place it in the current path:
https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/tree/main/audioscore/ckpts
Or `git clone https://huggingface.co/karl-wang/QwenFeat-Vocal-Score`

### Inference
(Runnable code is available in `python tests/test_generate_score.py`)

```python
import os, sys
import audioscore.model

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = audioscore.model.SongEvalGenerator_audio_lora()
    # Load model
    model.load_model(os.path.join(ROOT_DIR, "ckpts", "SongEvalGenerator", "step_2_al_audio", "best_model_step_132000")) 
    model = model.half() # Optional
    model = model.cuda()

    # Perform scoring
    score = model.generate_tag("/data/nfs/audioscore/data/sort/data/audio/203887_250822105518005501.m4a")

    print("score:", score)
```

Run tests directly:
```bash
python tests/test_generate_score.py
```

### Training

Direct Training:
`torchrun --nproc_per_node=4 --nnodes=1 scripts/train/train_sort_audio.py`

Adversarial Training (Decoupling with SaMoye spk encoder):
`torchrun --nproc_per_node=4 --nnodes=1 scripts/train/train_sort_audio_grl.py`

## Additional Notes for VocalVerse1 (QwenAudio based)

- Use the `qwenaudio` conda environment; weights for both models have been included.
- Added `infer_service.py` and `infer.py` scripts; instructions are in the README. `test/test_score.py` is functional.
- The prompts are located at `/home/w-4090/projects/qwenaudio/src/qwenaudio/prompts.py`.
- Test scripts are available at `/home/w-4090/projects/qwenaudio/tests/test_processor_v2.py` and `/home/w-4090/projects/qwenaudio/tests/test_processor_v3.py`.
- For the code in Figures 1 and 2: when loading models locally, change the model paths in `model.py` and `processor.py` to your local paths.

<img width="1601" height="315" alt="77b6f8227c1d4af030fd7e2e0af8c8ab" src="https://github.com/user-attachments/assets/191e8e3f-2b67-43c0-b51e-f0b50f5d7dd6" />
<img width="1335" height="165" alt="76943b4c05d12bf090aecc602c8430b4" src="https://github.com/user-attachments/assets/8b058f65-de12-44c6-b041-e7db5ef97095" />

- Figure 3 shows the two trained Lora models. These are ready and should be uploaded.
<img width="418" height="56" alt="image" src="https://github.com/user-attachments/assets/e07494e7-eedc-44bc-b885-bf825de3d3dc" />

- The entire project including code and Lora weights is included, but the base model must be downloaded from the internet.
- If only the final classifier is active and the Lora on the model body is not, accuracy will be very low. Once Lora weights are active, accuracy reaches 80%-90%. (Encoder + LLM Decode + Lora + Classifier).
- Figure 4 shows that the `Qwen2-Audio-7B-Instruct/` base model is required. It was previously downloaded automatically to `.cache`; if downloaded manually, re-specify the path in the code.

<img width="785" height="858" alt="32e73d7a89338a602f350225e859b67f" src="https://github.com/user-attachments/assets/26b3f4f9-2595-4567-934d-10d089785464" />

# VocalVerse项目总览

本工作郑重感谢 QwenAudio, SongEval, MuQ等先前工作的支持.

## 📜 许可协议与版权声明 | License & Copyright

1. **协议框架**：本项目采用 **[CC BY-NC-ND 4.0](http://creativecommons.org/licenses/by-nc-nd/4.0/) (署名-非商业性使用-禁止演绎)** 国际许可协议。
2. **非商业用途**：本项目代码、模型权重及文档仅限学术交流和个人教育用途免费使用。
3. **严禁商用**：未经书面授权，严禁将本项目任何部分用于任何形式的商业营利目的（包括但不限于集成至商业软件、提供营利性 AI 服务等）。
4. **商业授权申请**：如有商业合作意向或需获得商业使用许可，请务必联系邮箱：**3156018231@qq.com**。
---

## 🌍 版本说明与旧版存档 | Version Notice & Legacy Archive

**Legacy Repository / 旧版项目地址**: 
[https://github.com/CarlWangChina/Singing-Aesthetic-Assessment](https://github.com/CarlWangChina/Singing-Aesthetic-Assessment)


## 📖 论文引用与说明

关于本代码实现的详细技术方案、实验结果及理论支撑，请参考我们的研究论文。如果您在研究或工作中使用了本仓库的代码或模型，欢迎引用我们的论文。

### 论文信息

**标题**：Singing Timbre Popularity Assessment Based on Multimodal Large Foundation Model
**会议**：Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)

* **ACM Digital Library (官方版本)**: [https://doi.org/10.1145/3746027.3758148](https://doi.org/10.1145/3746027.3758148)
* **arXiv (免费预览版)**: [https://www.arxiv.org/abs/2512.06999](https://www.arxiv.org/abs/2512.06999) 
  *(注：arXiv 版本内容与 ACM DL 官方版本完全一致)*

### 引用格式

#### ACM Reference Format
> Zihao Wang, Ruibin Yuan, Ziqi Geng, Hengjia Li, Xingwei Qu, Xinyi Li, Songye Chen, Haoying Fu, Roger B. Dannenberg, and Kejun Zhang. 2025. Singing Timbre Popularity Assessment Based on Multimodal Large Foundation Model. In Proceedings of the 33rd ACM International Conference on Multimedia (MM '25). Association for Computing Machinery, New York, NY, USA, 12227–12236. https://doi.org/10.1145/3746027.3758148

#### BibTeX
```bibtex
@inproceedings{10.1145/3746027.3758148,
author = {Wang, Zihao and Yuan, Ruibin and Geng, Ziqi and Li, Hengjia and Qu, Xingwei and Li, Xinyi and Chen, Songye and Fu, Haoying and Dannenberg, Roger B. and Zhang, Kejun},
title = {Singing Timbre Popularity Assessment Based on Multimodal Large Foundation Model},
year = {2025},
isbn = {9798400720352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {[https://doi.org/10.1145/3746027.3758148](https://doi.org/10.1145/3746027.3758148)},
doi = {10.1145/3746027.3758148},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {12227–12236},
numpages = {10},
keywords = {computational music aesthetics, descriptive feedback, multi-dimensional evaluation, multimodal foundation models, singing timbre popularity, singing voice assessment},
location = {Dublin, Ireland},
series = {MM '25}
}
``` 
## 📂 VocalVerse数据集

VocalVerse_Datasets-human_labels文件夹包含三个核心文件，提供了针对演唱表现的业余和专业两级评价数据。

| 文件名 | 说明 |
| :--- | :--- |
| **`Amateur_overall_mos_avg5.xlsx`** | **业余共识评分：** 包含 165 名业余标注者的整体“听感愉悦度”评分。每段录音由 5 名独立的标注者按 1-5 分 Likert 量表打分，该文件提供最终的平均意见分 (MOS)。 |
| **`Professional_multidim_annotations_raw_...xlsx`** | **专家多维度标签：** 由两名专业声乐教练提供的详细标注。包含 **音色、气息、情感、技巧** 四个核心维度的 1-5 分整数评分及配套的**文字评语**。 |
| **`Professional_scoring_rubric.xlsx`** | **评分标准：** 专家标注时使用的正式准则（Rubric），确保四个维度的标注具有一致性和高质量。 |

### 数据规模

根据论文研究：
* **原始数据池：** 我们最初收集了超过 **100,000** 段原始清唱录音。
* **预筛选集合：** 经过人工和RuleSignal评分自动初步筛选，形成了包含 **10,000** 段片段的数据集。
* **当前开源子集：** 本仓库提供的文件代表了其中 **前 10%（约 1,000 段歌唱技术精湛的录音）**，这些录音随后, 经过了密集的专业多维度标注。

### 标注方法

1.  **业余阶段：** 共有 165 名非音乐专业标注者参与。他们采用“强制分布法”以确保分数具有区分度，从而捕捉大众的审美偏好。
2.  **专业阶段：** 两名资深声乐教师提供双模态标注（分数 + 描述性文字），以支持描述性多模态大模型（MLLM）的训练。
3.  **评价维度：**
    * **音色 (Timbre Quality)：** 声音的独特度、质感和层次感。
    * **气息 (Breath Control)：** 对复杂句子的支撑力和稳定性。
    * **情感 (Emotional Expression)：** 演唱的感染力与共鸣感。
    * **技巧 (Vocal Technique)：** 演唱技巧的熟练度。

# VocalVerse1: 基于qwenaudio的歌唱评价模型

[qwenaudio](./qwenaudio/README.md) Qwen评语生成+打分部分. 包含输入音频给Qwen-audio我们用Lora训练后的版本, 输出针对歌声的存在问题的评语 相当于对音频做描述性打标, 然后评语再作为输入作为深度思考部分, 最终进行音色打分, 然后用歌手音色的TTS念出来生成点评语音, 这一整套的流程:

1.qwen微调训练后的模型生成歌唱问题的评语.
2.评语+音频一起再次给qwen打分模型来辅助打分(相当于做了一次深度思考). 
3.最后再调一次大模型来对评语进行润色,  生成总结 , 然后给出唱法建议. 
4.在最后把总结用对应歌手的声音念出来. 

这一块的所有权重都保留了, 模型对应目录链接: 前往huggingface下载包含模型的完整仓库： [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/]  

## 调用方法

### 下载模型  
下载下面的目录，放在当前位置  
https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/tree/main/qwenaudio/ckpts  
直接git clone [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score](https://huggingface.co/karl-wang/QwenFeat-Vocal-Score) 也可以

### 启用服务  

`python scripts/infer_service.py`  
调用：  
`curl -X POST http://localhost:8080/score   -F "file=@音频路径"`  

### 命令行调用
`python scripts/infer.py 音频路径.wav 输出文件.txt`

### python调用

代码参考 `tests/test_score.py`

```python
import qwenaudio.processor
import qwenaudio.prompts
import librosa

# 初始化模型和处理器
processor = qwenaudio.processor.create_processor(
            "ckpts/generator-lora-32-16-scoreonly-f16/best_model_epoch_13/lora_weights",
            "ckpts/generator-lora-32-16-textonly-simple-v2-int4/best_model_epoch_16/lora_weights"
        )

# 打分
audio_path= "/home/w-4090/cutted_score_audio_separated/446892/344523004.wav"
data, sr = librosa.load(audio_path, sr=processor.processor.feature_extractor.sampling_rate, mono=True)
data = data[:processor.processor.feature_extractor.sampling_rate * 30] # 截取30秒
print(data.shape)

for i in range(4):
    print("test:", qwenaudio.prompts.prompt_mapper_reverse[i]) # 输出参数0到4的含义
    score = processor.generate(data, i)# 生成分数
```

如果在加载模型时卡住，可启用huggingface镜像
`export HF_ENDPOINT=https://hf-mirror.com`

# VocalVerse2: 基于MuQ的人声录音打分模型 

[audioscore](./audioscore/README.md) MuQ打分、排序部分. 包含 加解耦 和 不加解耦 两个版本, 使用的同一套代码，只分了一个目录. 

1、使用MuQ作为encoder+后接打分器进行打分的代码,不加解耦,这一块的架构跟SongEval工作基本相同.  其中MuQ解冻用lora的权重、后面打分器的权重, 都保留了,权重对应目录链接:  [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/]  

2、加解耦 , 采用SaMoye-SVC的 spk encoder作为反向梯度训练、对说话人身份特征进行解耦合, 来提升对美学理解的准确度的实验部分. 
二者项目代码是同一个、兼容的. 加解耦部分因为机房退租时没来得及拷贝下来, 导致模型权重丢失了. 但是使用SaMoye的spk encoder或者用wespeaker, 进行反向梯度训练来解耦合, 看看效果是否会变好的实验,实验结果大致如下: 使用SaMoye的spk encoder或者 wespeaker的encoder, 分别作为反向梯度来解耦合, 然后打分, 都特别难训练、难收敛. 但是batchsize小一点也能收敛. 最后评价美学等级的准确率, 效果比原来好一点点. 说明这个解耦部分确实是有用的. 

最后注意: 本仓库不包含模型部分, 研究者需要前往huggingface下载包含模型的完整仓库： [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/]  

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

### 训练  

直接训练  
`torchrun --nproc_per_node=4 --nnodes=1  scripts/train/train_sort_audio.py`  

对抗训练（使用samoye的spk encoder进行解耦）  
`torchrun --nproc_per_node=4 --nnodes=1  scripts/train/train_sort_audio_grl.py`  

## 其他注意事项 of VocalVerse1: 基于qwenaudio的歌唱评价模型

conda环境用qwenaudio, 两个模型权重都复制进去了

我加了infer_service.py和infer.py两个脚本，使用方法我写README.md里面了. test/test_score.py可以运行

/home/w-4090/projects/qwenaudio/src/qwenaudio/prompts.py 这个是prompt所在的位置.

测试脚本在/home/w-4090/projects/qwenaudio/tests/test_processor_v2.py和/home/w-4090/projects/qwenaudio/tests/test_processor_v3.py.

附图1和2的代码, 加载本地模型把model.py和processor.py中的模型路径改成本地的就行了

<img width="1601" height="315" alt="77b6f8227c1d4af030fd7e2e0af8c8ab" src="https://github.com/user-attachments/assets/191e8e3f-2b67-43c0-b51e-f0b50f5d7dd6" />
<img width="1335" height="165" alt="76943b4c05d12bf090aecc602c8430b4" src="https://github.com/user-attachments/assets/8b058f65-de12-44c6-b041-e7db5ef97095" />

附图3中两个是训练的lora模型. 这个是训好的. 这两个要上传
<img width="418" height="56" alt="image" src="https://github.com/user-attachments/assets/e07494e7-eedc-44bc-b885-bf825de3d3dc" />

整个工程包括代码和lora权重都在里面，但是base模型需要联网下载

如果只生效了结尾的分类器，模型本体的lora没生效，准确率会很低；lora权重生效后，准确率能80多 接近9成.  编码器+llm解码+lora+分类器

附图4 中Qwen2-Audio-7B-Instruct/这个base 模型都要，之前是自动下载到.cache里面的，单独下载的应该要代码里面重新指定路径

<img width="785" height="858" alt="32e73d7a89338a602f350225e859b67f" src="https://github.com/user-attachments/assets/26b3f4f9-2595-4567-934d-10d089785464" />

