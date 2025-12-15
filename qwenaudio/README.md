# qwenaudio打分工具  
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
