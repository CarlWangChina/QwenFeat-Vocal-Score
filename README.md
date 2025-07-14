# qwenaudio打分工具  
## 调用方法

### 启用服务  

`python scripts/infer_service.py`  
调用：  
`curl -X POST http://localhost:8080/score   -F "file=@音频路径"`  

添加`get_final_text`参数会生成总结文本  
例如`http://localhost:8080/score?get_final_text=1`  

### 命令行调用
`python scripts/infer.py 音频路径.wav 输出文件.txt`

### python调用

代码参考 `tests/test_score.py`

```python
import qwenaudio.processor
import qwenaudio.prompts
import librosa

# 初始化模型和处理器
processor = qwenaudio.processor.ScoreProcessor("ckpts/lora-64-4-pf_score/model_epoch76",
                                                "ckpts/generator-lora-128-64-textonly/best_model_epoch_6/lora_weights")

# 打分
audio_path= "/home/w-4090/cutted_score_audio_separated/446892/344523004.wav"
data, sr = librosa.load(audio_path, sr=processor.processor.feature_extractor.sampling_rate, mono=True)
data = data[:processor.processor.feature_extractor.sampling_rate * 30] # 截取30秒
print(data.shape)

for i in range(4):
    print("test:", qwenaudio.prompts.prompt_mapper_reverse[i]) # 输出参数0到4的含义
    score = processor.generate_score(data, i)# 生成分数
    print(score)
    print(processor.generate_text(data, i))# 直接生成分数和评语
    print(processor.generate_text_with_score(data, i, score["score"]))# 利用分数生成评语
```

如果在加载模型时卡住，可启用huggingface镜像
`export HF_ENDPOINT=https://hf-mirror.com`
