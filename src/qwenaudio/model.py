import os
import sys
import torch
from io import BytesIO
from urllib.request import urlopen
import librosa
import tempfile
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, Qwen2AudioProcessor
from transformers.models.qwen2_audio.processing_qwen2_audio import Qwen2AudioProcessorKwargs
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import numpy as np
import qwenaudio.audio_cut

import subprocess
import json
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(CURRENT_PATH))

import qwenaudio.hubert.inference as hubert
import qwenaudio.whisper_svc.inference as whisper_svc
import qwenaudio.pitch.inference as pitch

default_procesosor = "Qwen/Qwen2-Audio-7B-Instruct"
default_base_model = "Qwen/Qwen2-Audio-7B-Instruct"

if os.path.exists("ckpts/Qwen2-Audio-7B-Instruct"):
    default_procesosor = "ckpts/Qwen2-Audio-7B-Instruct"
    default_base_model = "ckpts/Qwen2-Audio-7B-Instruct"

class QwenAudioScoreModel(torch.nn.Module):
    def __init__(self, output_num=2, hidden_dim=128, use_lora=True, freeze_weight=True, lora_r=16, lora_alpha=16, device="cuda", base_model=None):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        # 加载半精度主模型
        if base_model is None:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                default_base_model,
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

class QwenAudioTowerScoreModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            default_base_model).audio_tower
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),  # 将任意长度序列压缩为1个时间步
            torch.nn.Flatten(),             # 转换为(batch_size, 1280)
            torch.nn.Linear(1280, 5)        # 二分类输出层
        )
        self.output_num = 5
        
    
    def forward(self, audio_inputs: torch.Tensor, audio_attention_mask: torch.Tensor = None):
        x = self.model(audio_inputs, attention_mask=audio_attention_mask).last_hidden_state
        x = x.permute(0, 2, 1)
        x = self.classifier(x)
        # x = torch.softmax(x, dim=1)
        return x
    
    def infer(self, audio, processor):
        ft = processor.feature_extractor(audio, return_attention_mask=True, padding="max_length")
        outputs = self(torch.tensor(ft.input_features).cuda())
        print(outputs)
        out_id = outputs[0].argmax().item()

        return {"score":out_id+1, "prompt": ""}

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), save_path+"/model.pt")
    
    def load_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path+"/model.pt", map_location="cpu"))

class QwenAudioTowerScore():
    def __init__(self):
        self.model = QwenAudioTowerScoreModel()
        self.processor = Qwen2AudioProcessor.from_pretrained(default_procesosor)
    
    def preprocess_audio(self, audios: list):
        # print("input:", audios[0].shape)
        audio_inputs = self.processor.feature_extractor(audios, return_attention_mask=True, padding="max_length")
        return audio_inputs
    
    def preprocess_audio_file(self, audio_inputs: list[str]):
        audios = [librosa.load(audio, sr=self.processor.feature_extractor.sampling_rate)[0] for audio in audio_inputs]
        return self.preprocess_audio(audios)
    
    def process_audio_file(self, audio_inputs: list[str]):
        inputs = self.preprocess_audio_file(audio_inputs)
        return self.model(torch.tensor(inputs.input_features))

    def process_audio(self, audios: list):
        inputs = self.preprocess_audio(audios)
        return self.model(torch.tensor(inputs.input_features))


class QwenAudioScore():
    def __init__(self, half=True, device="cuda"):
        self.model = QwenAudioScoreModel()
        if half:
            self.model.half()
        self.model.to(device)
        self.device = device
        self.half = half
        self.processor = Qwen2AudioProcessor.from_pretrained(default_procesosor)
    
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

class FeatExtractor():
    def __init__(self, device):
        self.whisper = whisper_svc.WhisperInference(os.path.join(ROOT_PATH, "ckpts", "whisper_pretrain", "large-v2.pt"), device)
        self.hubert = hubert.HubertInference(os.path.join(ROOT_PATH, "ckpts", "hubert_pretrain", "hubert-soft-0d54a1f4.pt"),device)
    def process_audio(self, audio):
        return self.whisper.inference(audio), self.hubert.inference(audio), pitch.compute_f0_sing_audio(audio, 16000)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding=kernel_size//2,
            bias=False
        )
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        
        self.conv2 = torch.nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size//2,
            bias=False
        )
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        
        # 捷径连接处理维度变化
        self.shortcut = torch.nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )
            self.bn_shortcut = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 处理捷径连接
        if self.shortcut is not None:
            residual = self.shortcut(residual)
            residual = self.bn_shortcut(residual)
            
        out += residual
        out = self.relu(out)
        return out

class AudioFeatClassifier_res(torch.nn.Module):
    def __init__(self, 
                 ppg_dim=1280, 
                 vec_dim=256, 
                 pit_embed_dim=32, 
                 hidden_size=512, 
                 num_layers=8, 
                 num_classes=5):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(256, pit_embed_dim)
        
        # 2. 卷积模块（包含3个残差块）
        self.conv_layers = torch.nn.Sequential(
            ResidualBlock(
                in_channels=ppg_dim + vec_dim + pit_embed_dim,
                out_channels=512,
                kernel_size=5,
                stride=4
            ),
            ResidualBlock(512, 512, 5, 2),
            ResidualBlock(512, 512, 5, 2)
        )
        
        # 3. LSTM层（输入维度改为512）
        self.lstm = torch.nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 4. 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, ppg, vec, pit):
        # 维度处理
        pit_emb = self.pit_embed(pit)  # [B, T, 32]
        x = torch.cat([ppg, vec, pit_emb], dim=-1)  # [B, T, 1568]
        
        # 维度转换：[B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)
        
        # 通过卷积模块（长度压缩约10倍）
        x = self.conv_layers(x)  # [B, 512, T'] (T' ≈ T/10)
        
        # 恢复维度：[B, C, T'] -> [B, T', C]
        x = x.permute(0, 2, 1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # [B, T', 1024]
        
        # 取最终状态
        last_output = lstm_out[:, -1, :]  # [B, 1024]
        
        # 分类
        return self.classifier(last_output)

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), save_path+"/model_weight_res.pt")
    
    def load_ckpt(self, load_path):
        self.load_state_dict(torch.load(load_path+"/model_weight_res.pt"))
        
class AudioFeatClassifier(torch.nn.Module):
    def __init__(self, 
                 ppg_dim=1280, 
                 vec_dim=256, 
                 pit_embed_dim=32, 
                 hidden_size=512, 
                 num_layers=4, 
                 num_classes=5):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(
            num_embeddings=256,  # pit值范围1-255
            embedding_dim=pit_embed_dim
        )
        
        # 2. LSTM层
        self.lstm = torch.nn.LSTM(
            input_size=ppg_dim + vec_dim + pit_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, ppg, vec, pit):
        """
        前向传播
        Args:
            ppg: [batch_size, seq_len, 1280]
            vec: [batch_size, seq_len, 256]
            pit: [batch_size, seq_len]
        Returns:
            logits: [batch_size, num_classes]
        """
        # 1. 处理pitch特征
        pit_emb = self.pit_embed(pit)  # [batch, seq_len, pit_embed_dim]
        
        # 2. 特征拼接
        x = torch.cat([ppg, vec, pit_emb], dim=-1)  # [batch, seq_len, 1280+256+pit_embed_dim]
        
        # 3. 通过LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        
        # 4. 取序列的最终状态
        last_output = lstm_out[:, -1, :]  # [batch, hidden_size]
        
        # 5. 分类
        logits = self.classifier(last_output)  # [batch, num_classes]
        
        return logits
    
    @torch.inference_mode()
    def infer(self, audio_path, processor):

        ppg, vec, pit = processor.process_audio(audio_path)
        # ppg = np.load(path_ppg.name)
        # ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
        # ppg = torch.FloatTensor(ppg)
        ppg = torch.repeat_interleave(ppg, repeats=2, dim=0) #repeat

        # ppg = torch.zeros_like(ppg)

        # vec = np.load(path_vec.name)
        # vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
        # vec = torch.FloatTensor(vec)
        vec = torch.repeat_interleave(vec, repeats=2, dim=0) #repeat
        # vec = torch.zeros_like(vec)

        pit = torch.FloatTensor(pit)

        len_pit = pit.size()[0]
        len_vec = vec.size()[0]
        len_ppg = ppg.size()[0]
        len_min = min(len_pit, len_vec)
        len_min = min(len_min, len_ppg)
        pit = pit[:len_min]
        vec = vec[:len_min, :]
        ppg = ppg[:len_min, :]
        pit = qwenaudio.audio_cut.f0_to_coarse(pit)

        # print(ppg.shape, vec.shape, pit.shape)

        ppg = ppg.view(1, -1, 1280).cuda()
        vec = vec.view(1, -1, 256).cuda()
        pit = pit.view(1, -1).cuda()
        logits = self(ppg, vec, pit)
        # return logits
        
        out_id = logits[0].argmax().item()

        return {"score":out_id+1, "prompt": ""}

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), save_path+"/model_weight.pt")
    
    def load_ckpt(self, load_path):
        self.load_state_dict(torch.load(load_path+"/model_weight.pt"))