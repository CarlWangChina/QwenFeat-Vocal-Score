# Singing-Aesthetic-Assessment项目总览

## 📜 License / 许可协议

本项目采用 **[CC BY-NC-ND 4.0](http://creativecommons.org/licenses/by-nc-nd/4.0/) (署名-非商业性使用-禁止演绎)** 国际许可协议。

- **版权所有**：魅KTV (Mei KTV)
- **商业禁止**：严禁任何未经授权的商业用途。
- **商业授权联系**：3156018231@qq.com

Assessing the Popularity of Singing Timbre with a Multimodal Large Foundation Model

[qwenaudio](./qwenaudio/README.md) Qwen评语生成+打分部分. 包含输入音频给Qwen-audio我们用Lora训练后的版本, 输出针对歌声的存在问题的评语 相当于对音频做描述性打标, 然后评语再作为输入作为深度思考部分, 最终进行音色打分, 然后用歌手音色的TTS念出来生成点评语音, 这一整套的流程:

1、qwen微调训练后的模型生成歌唱问题的评语.

2、评语+音频一起再次给qwen打分模型来辅助打分(相当于做了一次深度思考). 

3、最后再调一次大模型来对评语进行润色,  生成总结 , 然后给出唱法建议. 

4、在最后把总结用对应歌手的声音念出来. 

这一块的所有权重都保留了, 模型对应目录链接: 前往huggingface下载包含模型的完整仓库： [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/]  

[audioscore](./audioscore/README.md) MuQ打分、排序部分. 包含 加解耦 和 不加解耦 两个版本, 使用的同一套代码，只分了一个目录. 
1、使用MuQ作为encoder+后接打分器进行打分的代码,不加解耦,这一块的架构跟SongEval工作基本相同.  其中MuQ解冻用lora的权重、后面打分器的权重, 都保留了,权重对应目录链接:  [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/]  
2、加解耦 , 采用SaMoye-SVC的 spk encoder作为反向梯度训练、对说话人身份特征进行解耦合, 来提升对美学理解的准确度的实验部分. 
二者项目代码是同一个、兼容的. 加解耦部分因为机房退租时没来得及拷贝下来, 导致模型权重丢失了. 但是使用SaMoye的spk encoder或者用wespeaker, 进行反向梯度训练来解耦合, 看看效果是否会变好的实验,实验结果大致如下: 使用SaMoye的spk encoder或者 wespeaker的encoder, 分别作为反向梯度来解耦合, 然后打分, 都特别难训练、难收敛. 但是batchsize小一点也能收敛. 最后评价美学等级的准确率, 效果比原来好一点点. 说明这个解耦部分确实是有用的. 

最后注意: 本仓库不包含模型部分, 研究者需要前往huggingface下载包含模型的完整仓库： [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/]  

# qwenaudio打分工具  

[qwenaudio](./qwenaudio/README.md) Qwen评语生成+打分部分. 包含输入音频给Qwen-audio我们用Lora训练后的版本, 输出针对歌声的存在问题的评语, 然后评语再作为输入作为深度思考部分, 最终进行音色打分, 然后用歌手音色的TTS念出来, 这一整套的流程.  
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

# 人声录音打分模型  

[audioscore](./audioscore/README.md) MuQ打分、排序部分. 包含 加解耦 和 不加解耦 两个版本, 使用的同一套代码，只分了一个目录. 
1、使用MuQ作为encoder+后接打分器进行打分的代码,不加解耦,这一块的架构跟SongEval工作基本相同.  其中MuQ解冻用lora的权重、后面打分器的权重, 都保留了,权重对应目录链接:  [https://huggingface.co/karl-wang/QwenFeat-Vocal-Score/]  
2、加解耦 , 采用SaMoye-SVC的 spk encoder作为反向梯度训练、对说话人身份特征进行解耦, 来提升对美学理解的准确度的实验部分. 
二者项目代码是同一个、兼容的. 加解耦部分因为机房退租时没来得及拷贝下来, 导致模型权重丢失了. 但是使用SaMoye的spk encoder或者用wespeaker反向梯度来解耦合, 看看效果是否会变好的实验,实验结果大致如下: 使用SaMoye的spk encoder或者 wespeaker的encoder, 分别作为反向梯度来解耦合, 然后打分, 都特别难训练、难收敛. 但是batchsize小一点也能收敛. 最后评价美学等级的准确率, 效果比原来好一点点. 说明这个解耦部分确实是有用的. 

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
