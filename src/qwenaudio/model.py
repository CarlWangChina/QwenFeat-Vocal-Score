import os
import sys
import torch
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, Qwen2AudioProcessor
from transformers.models.qwen2_audio.processing_qwen2_audio import Qwen2AudioProcessorKwargs
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
# import wespeaker

# class WespeakerScoreModel(torch.nn.Module):
#     def __init__(self, output_num=2, hidden_dim=128):
#         super().__init__()
#         self.hidden_dim = hidden_dim
        
#         model = wespeaker.load_model('chinese')
#         if model_path is not None:
#             states = torch.load(model_path)
#             states = states['model'] if 'model' in states else states
#             model.model.load_state_dict(states)

class QwenAudioScoreModel(torch.nn.Module):
    def __init__(self, output_num=2, hidden_dim=128, use_lora=True, freeze_weight=True, lora_r=16, lora_alpha=16, device="cuda", base_model=None):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        # 加载半精度主模型
        if base_model is None:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct",
                torch_dtype=torch.float16,
                device_map=device
            )
        else:
            self.model = base_model
        
        if use_lora:
            # 配置LoRA参数
            self.lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 注意力层适配
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False
            )
            
            # 应用LoRA适配
            self.model = get_peft_model(self.model, self.lora_config)
        
        self.freeze_weight = freeze_weight
        self.use_lora = use_lora
        if freeze_weight:
            for param in self.model.parameters():
                param.requires_grad = False

        # 自定义分类头
        self.output_num = output_num
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(156032, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1)
        )
        self.pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten()
        )
        self.classifier = torch.nn.Linear(hidden_dim, output_num)
 
    def forward(self, **args):
        # 主模型前向传播（自动包含LoRA更新）
        x = self.model(**args).logits.to(torch.float32)
        # print(f"Logits requires_grad: {x.requires_grad}")
        
        # 自定义处理流程
        x = self.linear(x)

        x = x.permute(0, 2, 1)

        x = self.pool(x)
        x = self.classifier(x)
        
        # 激活函数处理
        if self.output_num == 1:
            x = torch.sigmoid(x)
            
        return x
 
    def save_model(self, save_path):
        """保存完整模型（包含LoRA适配器和分类头）"""
        os.makedirs(save_path, exist_ok=True)
        # 保存LoRA适配器
        if self.use_lora:
            self.model.save_pretrained(f"{save_path}/lora_weights")
        
        # 保存分类头参数
        torch.save({
            "linear_state": self.linear.state_dict(),
            "classifier_state": self.classifier.state_dict(),
            "config": {
                "output_num": self.output_num,
                "hidden_dim": self.hidden_dim
            }
        }, f"{save_path}/head.pt")
        
        # 保存基础配置
        # self.model.base_model.save_pretrained(save_path)
        self.model.base_model.config.save_pretrained(save_path)

    def load_ckpt(self, ckpt_path):
        checkpoint = torch.load(f"{ckpt_path}/head.pt", map_location="cpu")
        self.linear.load_state_dict(checkpoint["linear_state"])
        self.classifier.load_state_dict(checkpoint["classifier_state"])

        if os.path.exists(f"{ckpt_path}/lora_weights"):
            self.model = PeftModel.from_pretrained(self.model, f"{ckpt_path}/lora_weights")

class QwenAudioScore():
    def __init__(self, half=True, device="cuda"):
        self.model = QwenAudioScoreModel()
        if half:
            self.model.half()
        self.model.to(device)
        self.device = device
        self.half = half
        self.processor = Qwen2AudioProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    
        conversation = [
            {
                "role": "user", "content": [
                    {"type": "audio", "audio_url": "input.wav"},
                    {"type": "text", "text": "请评价这段歌声音频的音色好听所以受欢迎程度，给出1到5的整数分数"},
                ]
            }
        ]
        self.text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    def preprocess_audio(self, audios: list):
        # print("input:", audios[0].shape)
        audio_inputs = self.processor(text=self.text_prompt, audios=audios, return_tensors="pt", padding=True).to(self.device)
        return audio_inputs
    
    def preprocess_audio_file(self, audio_inputs: list[str]):
        audios = [librosa.load(audio, sr=self.processor.feature_extractor.sampling_rate)[0] for audio in audio_inputs]
        return self.preprocess_audio(audios)
    
    def process_audio_file(self, audio_inputs: list[str]):
        inputs = self.preprocess_audio_file(audio_inputs)
        return self.model(**inputs)

    def process_audio(self, audios: list):
        inputs = self.preprocess_audio(audios)
        return self.model(**inputs)

