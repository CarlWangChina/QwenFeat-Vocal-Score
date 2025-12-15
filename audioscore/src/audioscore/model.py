import torch
import os
import sys
import math
import audioread
import numpy
import librosa
import torchaudio
import safetensors
import safetensors.torch
from muq import MuQ
import peft

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audioscore.speaker.models.lstm import LSTMSpeakerEncoder
from audioscore.speaker.config import SpeakerEncoderConfig
from audioscore.speaker.utils.audio import AudioProcessor
from audioscore.speaker.infer import read_json

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

        # print(x.shape)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # [B, T', 1024]
        
        # 取最终状态
        last_output = lstm_out[:, -1, :]  # [B, 1024]
        
        # 分类
        return self.classifier(last_output)


class SiameseNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4):
        super(SiameseNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 双向RNN编码器（两个子网络共享权重）
        self.rnn = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, 128),  # 双向RNN输出拼接后维度为2*hidden_dim，两个序列拼接后为4*hidden_dim
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 5)  # 输出5个分类
        )
 
    def forward_once(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        output, (hidden, cell) = self.rnn(x)
        
        # 取最后一个时间步的双向隐藏状态
        forward_last = output[:, -1, :self.hidden_dim]
        backward_last = output[:, 0, self.hidden_dim:]
        combined = torch.cat((forward_last, backward_last), dim=1)
        return combined
 
    def forward(self, input1, input2):
        # 编码两个输入序列
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        # 计算差异特征
        diff = torch.abs(output1 - output2)
        concat = torch.cat((output1, output2, diff), dim=1)
        
        # 分类
        return self.classifier(concat)
class AudioFeatSiameseNetwork(torch.nn.Module):
    def __init__(self, 
                 ppg_dim=1280, 
                 vec_dim=256, 
                 pit_embed_dim=32, 
                 d_model=512,
                 num_classes=5,
                 seq_size=8000):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(256, pit_embed_dim)

        self.input_dim = ppg_dim + vec_dim + pit_embed_dim

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear_siam = torch.nn.Linear(self.input_dim, d_model)
        self.enc_siam = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=16,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=8)
        self.tf_out = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=16,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=8)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward_one(self, 
                    audio_feats_whisper,
                    audio_feats_hubert,
                    audio_feats_f0,
                    mask):
        pit_emb = self.pit_embed(audio_feats_f0)  # [B, T, 32]
        x = torch.cat([audio_feats_whisper, audio_feats_hubert, pit_emb], dim=-1)  # [B, T, 1568]
        
        # 维度转换：[B, T, C] -> [B, C, T]
        x = self.linear_siam(x)
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.pos_encoder(x)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.enc_siam(x, src_key_padding_mask=src_key_padding_mask)
        return x
    
    def forward(self, 
                audio_feats_0_whisper,
                audio_feats_0_hubert,
                audio_feats_0_f0,
                audio_feats_1_whisper,
                audio_feats_1_hubert,
                audio_feats_1_f0,
                mask):
        x0 = self.forward_one(audio_feats_0_whisper, audio_feats_0_hubert, audio_feats_0_f0, mask)
        x1 = self.forward_one(audio_feats_1_whisper, audio_feats_1_hubert, audio_feats_1_f0, mask)

        x = x1 - x0

        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.tf_out(x, src_key_padding_mask=src_key_padding_mask)
        x = self.classifier(x)
        return x
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path+"/weights.pt")
        with open(path+"/type.txt", "w") as f:
            f.write("AudioFeatSiameseNetwork")

        print("saved model to", path)

class AudioFeatSiameseNetwork_cross(torch.nn.Module):
    def __init__(self, 
                 ppg_dim=1280, 
                 vec_dim=256, 
                 pit_embed_dim=32, 
                 d_model=512,
                 num_classes=5):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(256, pit_embed_dim)

        self.input_dim = ppg_dim + vec_dim + pit_embed_dim

        self.pos_encoder = PositionalEncoding(d_model)
        self.linear_siam = torch.nn.Linear(self.input_dim, d_model)
        self.enc_siam = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=16,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=8)
        self.tf_out = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=16,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=8)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
        self.cross_attn = torch.nn.MultiheadAttention(d_model, 16, batch_first=True)
        self.cross_linear = torch.nn.Linear(d_model*2, d_model)

    def forward_one(self, 
                    audio_feats_whisper,
                    audio_feats_hubert,
                    audio_feats_f0,
                    mask):
        pit_emb = self.pit_embed(audio_feats_f0)  # [B, T, 32]
        x = torch.cat([audio_feats_whisper, audio_feats_hubert, pit_emb], dim=-1)  # [B, T, 1568]
        
        # 维度转换：[B, T, C] -> [B, C, T]
        x = self.linear_siam(x)
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.pos_encoder(x)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.enc_siam(x, src_key_padding_mask=src_key_padding_mask)
        return x
    
    def forward(self, 
                audio_feats_0_whisper,
                audio_feats_0_hubert,
                audio_feats_0_f0,
                audio_feats_1_whisper,
                audio_feats_1_hubert,
                audio_feats_1_f0,
                mask):
        x0 = self.forward_one(audio_feats_0_whisper, audio_feats_0_hubert, audio_feats_0_f0, mask)
        x1 = self.forward_one(audio_feats_1_whisper, audio_feats_1_hubert, audio_feats_1_f0, mask)

        # 交叉注意力机制（双向）
        # x0作为query，x1作为key和value
        attn_output_x0, _ = self.cross_attn(x0, x1, x1, key_padding_mask=mask)
        # x1作为query，x0作为key和value
        attn_output_x1, _ = self.cross_attn(x1, x0, x0, key_padding_mask=mask)
        
        # 合并双向注意力结果
        x = torch.cat([attn_output_x0, attn_output_x1], dim=-1)
        x = self.cross_linear(x)

        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.tf_out(x, src_key_padding_mask=src_key_padding_mask)
        x = self.classifier(x)
        return x
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path+"/weights.pt")
        with open(path+"/type.txt", "w") as f:
            f.write("AudioFeatSiameseNetwork")

        print("saved model to", path)
        

class AudioFeatCmpNetwork(torch.nn.Module):
    def __init__(self, 
                 ppg_dim=1280, 
                 vec_dim=256, 
                 pit_embed_dim=32, 
                 d_model=1024,
                 num_classes=5,
                 seq_size=8000):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(256, pit_embed_dim)

        self.input_dim = ppg_dim + vec_dim + pit_embed_dim

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim*2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=6)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward_one(self, 
                    audio_feats_whisper,
                    audio_feats_hubert,
                    audio_feats_f0):
        pit_emb = self.pit_embed(audio_feats_f0)  # [B, T, 32]
        x = torch.cat([audio_feats_whisper, audio_feats_hubert, pit_emb], dim=-1)  # [B, T, 1568]
        return x
    
    def forward(self, 
                audio_feats_0_whisper,
                audio_feats_0_hubert,
                audio_feats_0_f0,
                audio_feats_1_whisper,
                audio_feats_1_hubert,
                audio_feats_1_f0,
                mask):
        x0 = self.forward_one(audio_feats_0_whisper, audio_feats_0_hubert, audio_feats_0_f0)
        x1 = self.forward_one(audio_feats_1_whisper, audio_feats_1_hubert, audio_feats_1_f0)

        x = torch.cat([x0, x1], dim=-1)
        
        x = self.linear(x)
        x = self.pos_encoder(x)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.classifier(x)
        return x
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path+"/weights.pt")
        with open(path+"/type.txt", "w") as f:
            f.write("AudioFeatCmpNetwork")

        print("saved model to", path)

class AudioFeatCmpNetwork_rl(torch.nn.Module):
    def __init__(self, 
                 ppg_dim=1280, 
                 vec_dim=256, 
                 pit_embed_dim=32, 
                 d_model=1024,
                 num_classes=5,
                 seq_size=163840):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(256, pit_embed_dim)

        self.input_dim = ppg_dim + vec_dim + pit_embed_dim

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim*2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=12)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward_one(self, 
                    audio_feats_whisper,
                    audio_feats_hubert,
                    audio_feats_f0):
        pit_emb = self.pit_embed(audio_feats_f0)  # [B, T, 32]
        x = torch.cat([audio_feats_whisper, audio_feats_hubert, pit_emb], dim=-1)  # [B, T, 1568]
        return x
    
    def forward(self, 
                audio_feats_0_whisper,
                audio_feats_0_hubert,
                audio_feats_0_f0,
                audio_feats_1_whisper,
                audio_feats_1_hubert,
                audio_feats_1_f0,
                mask):
        x0 = self.forward_one(audio_feats_0_whisper, audio_feats_0_hubert, audio_feats_0_f0)
        x1 = self.forward_one(audio_feats_1_whisper, audio_feats_1_hubert, audio_feats_1_f0)

        x = torch.cat([x0, x1], dim=-1)
        
        x = self.linear(x)
        x = self.pos_encoder(x)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.classifier(x)
        return x
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path+"/weights.pt")
        with open(path+"/type.txt", "w") as f:
            f.write("AudioFeatCmpNetwork_rl")

        print("saved model to", path)

class AudioFeatCmpNetwork_music(torch.nn.Module):
    def __init__(self, 
                 d_model=1024,
                 num_classes=5,
                 seq_size=8000):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(1280, 128)

        self.input_dim = 384

        self.overtone_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(9),  # 添加输入LayerNorm
            torch.nn.Linear(9, 128),
            torch.nn.LayerNorm(128),  # 添加线性层后LayerNorm
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),  # 添加第二个线性层后LayerNorm
            torch.nn.LeakyReLU(),
        )
        self.volume_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(1),  # 添加输入LayerNorm
            torch.nn.Linear(1, 128),
            torch.nn.LayerNorm(128),  # 添加线性层后LayerNorm
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),  # 添加第二个线性层后LayerNorm
            torch.nn.LeakyReLU(),
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim*2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation='relu',
                batch_first=True
            ),
            num_layers=3)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
    
    def normalize_loud(self, tensor):
        return (tensor+40)/40

    def batch_wise_normalization(self, audio_feats):
        # 沿特征维度(dim=1)计算最大值，keepdim=True保持维度
        max_vals = audio_feats.max(dim=1, keepdim=True).values
        
        # 防止除以0（添加极小值）
        max_vals = max_vals.clamp(min=1e-8)  # 或使用 torch.where(max_vals==0, torch.ones_like(max_vals)*1e-8, max_vals)
        
        # 逐元素除法（广播机制自动对齐维度）
        normalized = audio_feats / max_vals
        return normalized

    def forward_one(self, 
                    audio_feats_overtone,
                    audio_feats_f0,
                    audio_feats_volume):
        # print(audio_feats_f0.max())
        pit_emb = self.pit_embed(audio_feats_f0)
        overtone = self.batch_wise_normalization(audio_feats_overtone)
        overtone = self.overtone_encoder(overtone)
        volume = self.normalize_loud(audio_feats_volume)
        volume = self.volume_encoder(volume.view(volume.shape[0], volume.shape[1], 1))
        x = torch.cat([pit_emb, overtone, volume], dim=-1)
        return x
    
    def forward(self, 
                audio_feats_0_overtone,
                audio_feats_0_f0,
                audio_feats_0_volume,
                audio_feats_1_overtone,
                audio_feats_1_f0,
                audio_feats_1_volume,
                mask):
        x0 = self.forward_one(audio_feats_0_overtone, audio_feats_0_f0, audio_feats_0_volume)
        x1 = self.forward_one(audio_feats_1_overtone, audio_feats_1_f0, audio_feats_1_volume)

        x = torch.cat([x0, x1], dim=-1)
        # print(x.shape)
        
        x = self.linear(x)
        x = self.pos_encoder(x)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.classifier(x)
        return x
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path+"/weights.pt")
        with open(path+"/type.txt", "w") as f:
            f.write("AudioFeatCmpNetwork_music")

        print("saved model to", path)

class MertFeatCmpNetwork(torch.nn.Module):
    def __init__(self,
                 mert_dim=1024,    # MERT hidden size
                 d_model=1024,
                 num_classes=5,
                 seq_size=8000):
        super().__init__()

        # Pitch 嵌入层 (256个离散音高值)

        # 输入维度 = mert_dim + pit_embed_dim
        self.input_dim = mert_dim

        # Transformer 编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim * 2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation="relu",
                batch_first=True
            ),
            num_layers=6
        )

        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )


    def forward(self,
                audio_feats_0_mert,
                audio_feats_1_mert,
                mask=None):

        # 对比拼接 [B, T, input_dim*2]
        x = torch.cat([audio_feats_0_mert, audio_feats_1_mert], dim=-1)

        # Transformer 编码
        x = self.linear(x)
        x = self.pos_encoder(x)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 分类
        x = self.classifier(x)
        return x

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/weights.pt")
        with open(path + "/type.txt", "w") as f:
            f.write("MertFeatCmpNetwork")
        print("saved model to", path)

class MuqFeatCmpNetwork(torch.nn.Module):
    def __init__(self,
                 mert_dim=1024,    # MERT hidden size
                 d_model=1024,
                 num_classes=5,
                 seq_size=8000):
        super().__init__()

        # Pitch 嵌入层 (256个离散音高值)

        # 输入维度 = mert_dim + pit_embed_dim
        self.input_dim = mert_dim

        # Transformer 编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim * 2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation="gelu",
                batch_first=True
            ),
            num_layers=3
        )

        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )


    def forward(self,
                audio_feats_0_mert,
                audio_feats_1_mert,
                mask=None):

        # 对比拼接 [B, T, input_dim*2]
        x = torch.cat([audio_feats_0_mert, audio_feats_1_mert], dim=-1)

        # Transformer 编码
        x = self.linear(x)
        x = self.pos_encoder(x)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 分类
        x = self.classifier(x)
        return x

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/weights.pt")
        with open(path + "/type.txt", "w") as f:
            f.write("MertFeatCmpNetwork")
        print("saved model to", path)


class MuqFeatSortNetwork(torch.nn.Module):
    def __init__(self,
                 muq_dim=1024,    # MERT hidden size
                 d_model=1024,
                 seq_size=16000):
        super().__init__()

        # Pitch 嵌入层 (256个离散音高值)
        self.input_dim = muq_dim

        # Transformer 编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation="gelu",
                batch_first=True
            ),
            num_layers=4
        )

        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1)
        )


    def forward(self,
                audio_feats_vec,
                mask=None):

        # Transformer 编码
        x = self.linear(audio_feats_vec)
        x = self.pos_encoder(x)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 打分
        x = self.classifier(x)
        
        if mask==None:
            x = x.mean(dim=1)
            return x
        else:
            # 扩展mask维度以匹配输入张量
            mask_expanded = mask.unsqueeze(-1)  # [16, 854, 1]
            
            # 应用mask（将无效位置置零）
            masked_tensor = x * mask_expanded  # [16, 854, 5]
            
            # 计算有效元素数量（避免除零）
            valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [16, 1]
            
            # 求和并计算平均值
            sum_result = masked_tensor.sum(dim=1)  # [16, 5]
            avg_result = sum_result / valid_counts  # [16, 5]
            avg_result = avg_result.view(-1)

            return avg_result

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/weights.pt")
        with open(path + "/type.txt", "w") as f:
            f.write("MuqFeatSortNetwork")
        print("saved model to", path)

class MuqFeatCmpSortNetwork(torch.nn.Module):
    def __init__(self,
                 muq_dim=1024,    # MERT hidden size
                 d_model=1024,
                 seq_size=16000):
        super().__init__()

        # Pitch 嵌入层 (256个离散音高值)
        self.input_dim = muq_dim

        # Transformer 编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=16,
                dim_feedforward=2048,
                dropout=0.3,
                activation="gelu",
                batch_first=True
            ),
            num_layers=3
        )

        # 分类头
        self.classifier_emb = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 256)
        )

        self.cmp = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1)
        )

    def forward_one(self,
                audio_feats_vec,
                mask=None):

        # Transformer 编码
        x = self.linear(audio_feats_vec)
        x = self.pos_encoder(x)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 打分
        x = self.classifier_emb(x)
        
        if mask==None:
            x = x.mean(dim=1)
            return x
        else:
            # 扩展mask维度以匹配输入张量
            mask_expanded = mask.unsqueeze(-1)  # [16, 854, 1]
            
            # 应用mask（将无效位置置零）
            masked_tensor = x * mask_expanded  # [16, 854, 5]
            
            # 计算有效元素数量（避免除零）
            valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [16, 1]
            
            # 求和并计算平均值
            sum_result = masked_tensor.sum(dim=1)  # [16, 5]
            avg_result = sum_result / valid_counts  # [16, 5]
            # print(avg_result.shape)

            return avg_result

    def forward(self,
                audio_feats_vec_a,
                mask_a,
                audio_feats_vec_b,
                mask_b):
        x1 = self.forward_one(audio_feats_vec_a, mask_a)
        x2 = self.forward_one(audio_feats_vec_b, mask_b)
        x = torch.cat((x1, x2), dim=1)
        x = self.cmp(x)
        # print(x.shape)
        return x
        

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/weights.pt")
        with open(path + "/type.txt", "w") as f:
            f.write("MuqFeatCmpSortNetwork")
        print("saved model to", path)


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# 定义模型（已修改为包含对抗训练）
class MuqFeatCmpNetwork_GRL(torch.nn.Module):
    def __init__(self,
                 mert_dim=1024,
                 d_model=1024,
                 num_classes=5,
                 seq_size=8000,
                 speaker_embed_dim=256):
        super().__init__()

        self.input_dim = mert_dim
        
        # Transformer 编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim * 2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation="gelu",
                batch_first=True
            ),
            num_layers=3
        )

        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
        
        # 说话人识别头（对抗训练用）
        self.speaker_discriminator = torch.nn.Sequential(
            torch.nn.Linear(d_model, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, speaker_embed_dim * 2)  # 预测两个说话人的embedding
        )

    def forward(self,
                audio_feats_0_mert,
                audio_feats_1_mert,
                speaker_embs_0=None,  # 可选，训练时需要
                speaker_embs_1=None,  # 可选，训练时需要
                mask=None,
                alpha=1.0):           # GRL的权重系数

        # 对比拼接 [B, T, input_dim*2]
        x = torch.cat([audio_feats_0_mert, audio_feats_1_mert], dim=-1)

        # Transformer 编码
        x = self.linear(x)
        x = self.pos_encoder(x)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 主任务分类
        main_output = self.classifier(encoded)
        
        # 如果提供了说话人embedding，则进行对抗训练
        if speaker_embs_0 is not None and speaker_embs_1 is not None:
            # 使用序列的平均表示
            if mask is not None:
                # 处理mask，计算有效长度
                # print("encoded", encoded.shape)
                # print("mask", mask.shape)
                lengths = mask.sum(dim=1)
                # print("lengths", lengths.shape)
                pooled = (encoded*mask.unsqueeze(-1).expand(-1, -1, encoded.shape[-1])).sum(dim=1)
                # print("pooled", pooled.shape)
                pooled = pooled / lengths.unsqueeze(1)
                # print("pooled", pooled.shape)
            else:
                pooled = encoded.mean(dim=1)
            
            # 应用梯度反转层
            reversed_pooled = GradientReversalLayer.apply(pooled, alpha)
            
            # 说话人识别
            speaker_pred = self.speaker_discriminator(reversed_pooled)
            speaker_pred_0 = speaker_pred[:, :256]  # 预测的第一个说话人embedding
            speaker_pred_1 = speaker_pred[:, 256:]  # 预测的第二个说话人embedding
            
            return main_output, speaker_pred_0, speaker_pred_1
        
        return main_output

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/weights.pt")
        with open(path + "/type.txt", "w") as f:
            f.write("MertFeatCmpNetwork_GRL")
        print("saved model to", path)


class MuqFeatCmpNetwork_GRL_whisper(torch.nn.Module):
    def __init__(self,
                 mert_dim=1024,
                 d_model=1024,
                 num_classes=5,
                 seq_size=8000,
                 speaker_embed_dim=1280):
        super().__init__()

        self.input_dim = mert_dim
        self.speaker_embed_dim = speaker_embed_dim
        
        # Transformer 编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim * 2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation="gelu",
                batch_first=True
            ),
            num_layers=3
        )

        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
        
        # 说话人识别头（对抗训练用）
        self.speaker_discriminator = torch.nn.Sequential(
            torch.nn.Linear(d_model, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, speaker_embed_dim * 2),  # 预测两个说话人的embedding
        )
        self.upsampler = torch.nn.ConvTranspose1d(
            speaker_embed_dim * 2, 
            speaker_embed_dim * 2,
            kernel_size=3,      # 卷积核大小
            stride=2,           # 步长=2实现上采样
            padding=1,          # 输入填充
            output_padding=1   # 输出调整（关键）
        )

    def forward(self,
                audio_feats_0_mert,
                audio_feats_1_mert,
                speaker_embs_0=None,  # 可选，训练时需要
                speaker_embs_1=None,  # 可选，训练时需要
                mask=None,
                alpha=1.0):           # GRL的权重系数

        # 对比拼接 [B, T, input_dim*2]
        x = torch.cat([audio_feats_0_mert, audio_feats_1_mert], dim=-1)

        # Transformer 编码
        x = self.linear(x)
        x = self.pos_encoder(x)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 主任务分类
        main_output = self.classifier(encoded)
        
        # 如果提供了说话人embedding，则进行对抗训练
        if speaker_embs_0 is not None and speaker_embs_1 is not None:
            
            # 应用梯度反转层
            reversed_encoded = GradientReversalLayer.apply(encoded, alpha)
            
            # 说话人识别
            speaker_pred = self.speaker_discriminator(reversed_encoded)
            # print(speaker_pred.shape)
            speaker_pred = speaker_pred.permute(0, 2, 1)
            speaker_pred = self.upsampler(speaker_pred)
            speaker_pred = speaker_pred.permute(0, 2, 1)

            speaker_pred_0 = speaker_pred[:,:, :self.speaker_embed_dim]  # 预测的第一个说话人embedding
            speaker_pred_1 = speaker_pred[:,:, self.speaker_embed_dim:]  # 预测的第二个说话人embedding
            
            return main_output, speaker_pred_0, speaker_pred_1
        
        return main_output

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/weights.pt")
        with open(path + "/type.txt", "w") as f:
            f.write("MuqFeatCmpNetwork_GRL_whisper")
        print("saved model to", path)

class MuqFeatCmpNetwork_GRL_hubert(torch.nn.Module):
    def __init__(self,
                 mert_dim=1024,
                 d_model=1024,
                 num_classes=5,
                 seq_size=8000,
                 speaker_embed_dim=256):
        super().__init__()

        self.input_dim = mert_dim
        self.speaker_embed_dim = speaker_embed_dim
        
        # Transformer 编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_size)
        self.linear = torch.nn.Linear(self.input_dim * 2, d_model)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=32,
                dim_feedforward=2048,
                dropout=0.3,
                activation="gelu",
                batch_first=True
            ),
            num_layers=3
        )

        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
        
        # 说话人识别头（对抗训练用）
        self.speaker_discriminator = torch.nn.Sequential(
            torch.nn.Linear(d_model, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, speaker_embed_dim * 2),  # 预测两个说话人的embedding
        )
        self.upsampler = torch.nn.ConvTranspose1d(
            speaker_embed_dim * 2, 
            speaker_embed_dim * 2,
            kernel_size=3,      # 卷积核大小
            stride=2,           # 步长=2实现上采样
            padding=1,          # 输入填充
            output_padding=1   # 输出调整（关键）
        )

    def forward(self,
                audio_feats_0_mert,
                audio_feats_1_mert,
                speaker_embs_0=None,  # 可选，训练时需要
                speaker_embs_1=None,  # 可选，训练时需要
                mask=None,
                alpha=1.0):           # GRL的权重系数

        # 对比拼接 [B, T, input_dim*2]
        x = torch.cat([audio_feats_0_mert, audio_feats_1_mert], dim=-1)

        # Transformer 编码
        x = self.linear(x)
        x = self.pos_encoder(x)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 主任务分类
        main_output = self.classifier(encoded)
        
        # 如果提供了说话人embedding，则进行对抗训练
        if speaker_embs_0 is not None and speaker_embs_1 is not None:
            
            # 应用梯度反转层
            reversed_encoded = GradientReversalLayer.apply(encoded, alpha)
            
            # 说话人识别
            speaker_pred = self.speaker_discriminator(reversed_encoded)
            # print(speaker_pred.shape)
            speaker_pred = speaker_pred.permute(0, 2, 1)
            speaker_pred = self.upsampler(speaker_pred)
            speaker_pred = speaker_pred.permute(0, 2, 1)

            speaker_pred_0 = speaker_pred[:,:, :self.speaker_embed_dim]  # 预测的第一个说话人embedding
            speaker_pred_1 = speaker_pred[:,:, self.speaker_embed_dim:]  # 预测的第二个说话人embedding
            
            return main_output, speaker_pred_0, speaker_pred_1
        
        return main_output

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/weights.pt")
        with open(path + "/type.txt", "w") as f:
            f.write("MuqFeatCmpNetwork_GRL_hubert")
        print("saved model to", path)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class SpkEncoderHelper(torch.nn.Module):
    def __init__(self, root_path=None):
        super(SpkEncoderHelper, self).__init__()
        # python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker
        # model
        print("ROOT_DIR", ROOT_DIR)
        self.model_path =  os.path.join(ROOT_DIR, os.path.join("pretrain", "speaker_pretrain", "best_model.pth.tar"))
        self.config_path = os.path.join(ROOT_DIR, os.path.join("pretrain", "speaker_pretrain", "config.json"))
        if root_path:
            self.model_path = os.path.join(root_path, self.model_path)
            self.config_path = os.path.join(root_path, self.config_path)
        # config
        self.config_dict = read_json(self.config_path)

        # model
        self.config = SpeakerEncoderConfig(self.config_dict)
        self.config.from_dict(self.config_dict)

        self.speaker_encoder = LSTMSpeakerEncoder(
            self.config.model_params["input_dim"],
            self.config.model_params["proj_dim"],
            self.config.model_params["lstm_dim"],
            self.config.model_params["num_lstm_layers"],
        )
        self.use_cuda = True
        self.speaker_encoder.load_checkpoint(
            self.model_path, eval=True, use_cuda=self.use_cuda
        )
        # preprocess
        self.speaker_encoder_ap = AudioProcessor(**self.config.audio)
        # normalize the input audio level and trim silences
        self.speaker_encoder_ap.do_sound_norm = True
        self.speaker_encoder_ap.do_trim_silence = True
    
    def forward(self, wav_list: list[torch.tensor], sr: int, infer: bool = True):
        """
        Args:
            wav_list: list of torch.tensor, 每个元素是一段单声道波形
            sr: int, 输入音频的采样率
            infer: bool, 是否推理模式
        
        Returns:
            torch.Tensor, shape = (len(wav_list), proj_dim)
        """
        device = next(self.speaker_encoder.parameters()).device
        embeds = torch.zeros(len(wav_list), self.speaker_encoder.proj_dim, device=device)

        for i, waveform in enumerate(wav_list):
            # 确保是float32
            waveform = waveform.to(torch.float32).numpy()
            # 自动重采样到模型所需采样率
            target_sr = self.speaker_encoder_ap.sample_rate
            if sr != target_sr:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
                # waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

            # 提取mel谱
            spec = self.speaker_encoder_ap.melspectrogram(waveform)
            spec = torch.from_numpy(spec.T).unsqueeze(0).to(device).view(1, -1, 80)

            # 计算嵌入
            embed = self.speaker_encoder.compute_embedding(spec, infer=infer)
            embeds[i] = embed

            # print("embed.shape", embed.shape)

        return embeds

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=8000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
 
    def forward(self, x):
        # print("pe.shape", self.pe.shape, x.shape)
        return x + self.pe[:, :x.size(1)].expand(x.size(0), -1, -1)
 
class AudioFeatClassifier_tf(torch.nn.Module):
    def __init__(self, 
                 ppg_dim=1280, 
                 vec_dim=256, 
                 pit_embed_dim=32, 
                 d_model=512,
                 num_classes=5):
        super().__init__()
        
        # 1. Pitch嵌入层
        self.pit_embed = torch.nn.Embedding(256, pit_embed_dim)
        
        # 2. CNN降维模块
        self.cnn = torch.nn.Sequential(
            # 输入维度: [batch, 1568, seq_len]
            torch.nn.Conv1d(ppg_dim + vec_dim + pit_embed_dim, d_model, 
                          kernel_size=5, stride=2, padding=2),  # 100 -> 20
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(d_model),
            torch.nn.Dropout(0.2),
            
            torch.nn.Conv1d(d_model, d_model, 
                          kernel_size=5, stride=2, padding=2),  # 20 -> 4
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(d_model),
            torch.nn.Dropout(0.2),
            
            torch.nn.Conv1d(d_model, d_model, 
                          kernel_size=3, stride=2, padding=1)   # 4 -> 2
        )
        
        # 3. Transformer编码器
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=16,
            dim_feedforward=2048,
            dropout=0.3,
            activation='relu',
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=16)
        
        # 4. 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
 
    def forward(self, ppg, vec, pit):
        # 1. 处理pitch特征
        pit_emb = self.pit_embed(pit)  # [batch, seq_len, 32]
        
        # 2. 特征拼接
        x = torch.cat([ppg, vec, pit_emb], dim=-1)  # [batch, 100, 1568]
        
        # 3. CNN降维
        x = x.permute(0, 2, 1)          # [batch, 1568, 100]
        x = self.cnn(x)                 # [batch, 512, 2]
        x = x.permute(0, 2, 1)          # [batch, 2, 512]
        
        # print("x.shape",x.shape)

        # 4. 添加位置编码
        x = self.pos_encoder(x)
        
        # 5. Transformer编码
        x = self.transformer(x)         # [batch, 2, 512]
        
        # 6. 分类（取最后一个时间步）
        x = x[:, -1, :]                 # [batch, 512]
        logits = self.classifier(x)     # [batch, num_classes]
        
        return logits
    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), save_path+"/model_weight_tf.pt")
    
    def load_ckpt(self, load_path):
        self.load_state_dict(torch.load(load_path+"/model_weight_tf.pt"))
    
class AudioFeatHybrid(torch.nn.Module):
    def __init__(self, 
                 pitch_dim=32,
                 whisper_dim=1280,
                 hubert_dim=256,
                 mert_dim=1024,
                 muq_dim=1024,
                 conv_channels=256,
                 transformer_d_model=512,
                 transformer_nhead=8,
                 transformer_num_layers=6,
                 seq_size=16384,
                 dropout=0.1):
        super(AudioFeatHybrid, self).__init__()
        
        # 卷积降采样网络 - 将其他特征降采样到与muq相同的序列长度
        # samoye_pitch: [batch_size, seq_pitch] -> 需要扩展维度
        self.pit_embed = torch.nn.Embedding(256, pitch_dim)
        self.pitch_conv = torch.nn.Sequential(
            torch.nn.Conv1d(pitch_dim, conv_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)  # 降采样到固定长度，后面会重复到muq长度
        )
        
        # samoye_whisper: [batch_size, seq_whisper, 1280] -> 需要转置
        self.whisper_conv = torch.nn.Sequential(
            torch.nn.Conv1d(whisper_dim, conv_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        
        # samoye_hubert: [batch_size, seq_hubert, 256] -> 需要转置
        self.hubert_conv = torch.nn.Sequential(
            torch.nn.Conv1d(hubert_dim, conv_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        
        # mert: [batch_size, seq_mert, 1024] -> 需要转置
        self.mert_conv = torch.nn.Sequential(
            torch.nn.Conv1d(mert_dim, conv_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        
        # muq投影层: [batch_size, 1, seq_muq, 1024] -> [batch_size, seq_muq, conv_channels]
        self.muq_proj = torch.nn.Linear(muq_dim, muq_dim)
        
        # 特征融合后的总维度
        total_features = conv_channels * 4 + muq_dim  # pitch + whisper + hubert + mert + muq
        
        # Transformer编码器
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        self.pos_encoder = PositionalEncoding(transformer_d_model, max_len=seq_size)
        
        # 输入投影层，将融合特征投影到transformer的维度
        self.input_proj = torch.nn.Linear(total_features, transformer_d_model)
        
        # 输出层
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(transformer_d_model, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1),
            torch.nn.ReLU()  # 输出0-1之间的分数
        )
        
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, samoye_pitch, samoye_whisper, samoye_hubert, mert, muq, muq_mask):
        """
        Args:
            - samoye_pitch: [batch_size, seq_pitch]
            - samoye_whisper: [batch_size, seq_whisper, 1280]
            - samoye_hubert: [batch_size, seq_hubert, 256]
            - mert: [batch_size, seq_mert, 1024]
            - muq: [batch_size, 1, seq_muq, 1024]
            - muq_mask: [batch_size, seq_muq]
        
        Returns:
            score: [batch_size] 0-1之间的分数
        """
        batch_size = muq.shape[0]
        seq_muq = muq.shape[2]  # muq的序列长度
        
        # 处理samoye_pitch [batch_size, seq_pitch] -> [batch_size, 1, seq_pitch]
        pitch_embedded = self.pit_embed(samoye_pitch)  # 添加通道维度
        pitch_embedded = pitch_embedded.transpose(1, 2)
        pitch_feat = self.pitch_conv(pitch_embedded)  # [batch_size, conv_channels, 1]
        pitch_feat = pitch_feat.squeeze(-1).unsqueeze(1)  # [batch_size, 1, conv_channels]
        pitch_feat = pitch_feat.repeat(1, seq_muq, 1)  # [batch_size, seq_muq, conv_channels]
        
        # 处理samoye_whisper [batch_size, seq_whisper, 1280] -> 转置为 [batch_size, 1280, seq_whisper]
        whisper = samoye_whisper.transpose(1, 2)
        whisper_feat = self.whisper_conv(whisper)  # [batch_size, conv_channels, 1]
        whisper_feat = whisper_feat.squeeze(-1).unsqueeze(1)  # [batch_size, 1, conv_channels]
        whisper_feat = whisper_feat.repeat(1, seq_muq, 1)  # [batch_size, seq_muq, conv_channels]
        
        # 处理samoye_hubert [batch_size, seq_hubert, 256] -> 转置为 [batch_size, 256, seq_hubert]
        hubert = samoye_hubert.transpose(1, 2)
        hubert_feat = self.hubert_conv(hubert)  # [batch_size, conv_channels, 1]
        hubert_feat = hubert_feat.squeeze(-1).unsqueeze(1)  # [batch_size, 1, conv_channels]
        hubert_feat = hubert_feat.repeat(1, seq_muq, 1)  # [batch_size, seq_muq, conv_channels]
        
        # 处理mert [batch_size, seq_mert, 1024] -> 转置为 [batch_size, 1024, seq_mert]
        mert = mert.transpose(1, 2)
        mert_feat = self.mert_conv(mert)  # [batch_size, conv_channels, 1]
        mert_feat = mert_feat.squeeze(-1).unsqueeze(1)  # [batch_size, 1, conv_channels]
        mert_feat = mert_feat.repeat(1, seq_muq, 1)  # [batch_size, seq_muq, conv_channels]
        
        # 处理muq [batch_size, 1, seq_muq, 1024] -> [batch_size, seq_muq, 1024]
        muq = muq.squeeze(1)  # 移除第二个维度
        muq_feat = self.muq_proj(muq)  # [batch_size, seq_muq, conv_channels]
        
        # 拼接所有特征 [batch_size, seq_muq, conv_channels * 5]
        combined_features = torch.cat([pitch_feat, whisper_feat, hubert_feat, mert_feat, muq_feat], dim=-1)
        
        # 投影到transformer维度
        x = self.input_proj(combined_features)  # [batch_size, seq_muq, transformer_d_model]
        x = self.dropout(x)
        
        # 获取muq的mask并转换为Transformer需要的格式
        # mask形状: [batch_size, seq_muq], 1表示有效位置，0表示padding
        muq_mask = muq_mask
        
        # 转换为Transformer需要的格式: 
        # src_key_padding_mask: [batch_size, seq_len] 
        
        # 添加位置编码
        x = self.pos_encoder(x)

        # print("x.shape, muq_mask.shape:", x.shape, muq_mask.shape)
        # print(muq_mask)

        # 通过Transformer
        src_key_padding_mask = (muq_mask == 0)
        transformer_output = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # transformer_output形状: [batch_size, seq_muq, transformer_d_model]
        
        # 使用全局平均池化（考虑mask）
        # 将padding位置的值设为0，然后求和，最后除以有效位置的数量
        masked_output = transformer_output * muq_mask.unsqueeze(-1)  # 应用mask
        sum_output = masked_output.sum(dim=1)  # [batch_size, transformer_d_model]
        valid_counts = muq_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
        
        # 避免除以0
        valid_counts = torch.clamp(valid_counts, min=1)
        pooled_output = sum_output / valid_counts  # [batch_size, transformer_d_model]
        
        # 最终输出层
        score = self.output_layer(pooled_output).squeeze(-1)  # [batch_size]
        
        return score

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/weights.pt")
        with open(path + "/type.txt", "w") as f:
            f.write("AudioFeatHybrid")
        print("saved model to", path)


class SongEvalGenerator(torch.nn.Module):

    def __init__(self,
                 in_features=1024,
                 ffd_hidden_size=4096,
                 num_classes=5,
                 attn_layer_num=4,
                 use_grl=False
                 ):
        super(SongEvalGenerator, self).__init__()
        
        self.attn = torch.nn.ModuleList(
            [
                torch.nn.MultiheadAttention(
                    embed_dim=in_features,
                    num_heads=8,
                    dropout=0.2,
                    batch_first=True,
                )
                for _ in range(attn_layer_num)
            ]
        )
        
        self.ffd = torch.nn.Sequential(
            torch.nn.Linear(in_features, ffd_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(ffd_hidden_size, in_features)
        )
        
        self.dropout = torch.nn.Dropout(0.2)
        
        self.fc =  torch.nn.Linear(in_features * 2, num_classes)
        
        if use_grl:
            self.speaker_discriminator = torch.nn.Sequential(
                torch.nn.Linear(in_features * 2, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, 256)  # 预测两个说话人的embedding
            )

        self.proj = torch.nn.Tanh()
        self.load_base()
        

    def forward(self, audio_feats_vec, mask=None, judge_id=None, return_grl=False, grl_alpha=1.0):
        '''
        ssl_feature: [B, T, D]   
        output: [B, num_classes]
        '''
        ssl_feature = audio_feats_vec

        B, T, D = ssl_feature.shape
        
        ssl_feature = self.ffd(ssl_feature)
        
        tmp_ssl_feature = ssl_feature
        
        for attn in self.attn:
            tmp_ssl_feature, _ = attn(tmp_ssl_feature, tmp_ssl_feature, tmp_ssl_feature)
    
        ssl_feature = self.dropout(torch.concat([torch.mean(tmp_ssl_feature, dim=1), torch.max(ssl_feature, dim=1)[0]], dim=1))  # B, 2D
        
        x = self.fc(ssl_feature)  # B, num_classes
        
        x = self.proj(x) * 2.0 + 3
        
        # 应用梯度反转层
        if return_grl:
            reversed_encoded = GradientReversalLayer.apply(ssl_feature, grl_alpha)
            
            # 说话人识别
            speaker_pred = self.speaker_discriminator(reversed_encoded)
            
            return x[:, 1], speaker_pred
        else:
            # return x
            return x[:, 1]
    
    def load_base(self):
        self.load_state_dict(
            safetensors.torch.load_file(
                os.path.join(
                    ROOT_DIR, "src", "SongEval", "ckpt", "model.safetensors"),
                      device="cpu"), strict=False)

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/weights.pt")
        with open(path + "/type.txt", "w") as f:
            f.write("SongEvalGenerator")
        print("saved model to", path)
    
class SongEvalGenerator_audio(torch.nn.Module):
    def __init__(self, use_grl=False):
        super(SongEvalGenerator_audio, self).__init__()
        self.muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        self.model = SongEvalGenerator(use_grl=use_grl)
        self.use_grl = use_grl
    def forward(self, wavs, return_grl=False, grl_alpha=1.0):
        audio_embeds = self.muq(wavs, output_hidden_states=True)["hidden_states"][6]
        return self.model(audio_feats_vec=audio_embeds, return_grl=return_grl, grl_alpha=grl_alpha)
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/weights.pt")
        with open(path + "/type.txt", "w") as f:
            f.write("SongEvalGenerator_audio")
        print("saved model to", path)

def get_lora_state_dict(model):
    """
    返回只包含 LoRA adapter 的增量权重
    兼容 peft 的 ModuleDict 结构
    """
    lora_sd = {}

    for name, module in model.named_modules():
        if isinstance(module, peft.tuners.lora.Linear):
            for adapter_name in module.lora_A.keys():  
                A = module.lora_A[adapter_name]
                B = module.lora_B[adapter_name]
                scaling = module.scaling[adapter_name]

                lora_sd[f"{name}.{adapter_name}.lora_A"] = A.weight.data.cpu()
                lora_sd[f"{name}.{adapter_name}.lora_B"] = B.weight.data.cpu()
                lora_sd[f"{name}.{adapter_name}.scaling"] = torch.tensor(scaling)

    return lora_sd


def inject_lora_into_muq(mu_q, r=128, alpha=48, target_modules=None):
    if target_modules is None:
        target_modules = ["linear_q", "linear_k", "linear_v", "linear_out"]

    for name, module in mu_q.named_modules():
        if any(t in name for t in target_modules):
            
            adapter_name = "audio_lora"

            lora_layer = peft.tuners.lora.Linear(
                module,
                adapter_name=adapter_name,
                r=r,
                lora_alpha=alpha,
                lora_dropout=0.0,
                fan_in_fan_out=False
            )

            # 替换到父模块
            parent = mu_q
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, lora_layer)

    return mu_q


def load_lora_state_dict(model, lora_sd):
    """
    将 LoRA 权重加载回模型
    """
    for name, module in model.named_modules():
        if isinstance(module, peft.tuners.lora.Linear):

            for adapter_name in module.lora_A.keys():
                A_key = f"{name}.{adapter_name}.lora_A"
                B_key = f"{name}.{adapter_name}.lora_B"
                S_key = f"{name}.{adapter_name}.scaling"

                if A_key in lora_sd:
                    module.lora_A[adapter_name].weight.data = lora_sd[A_key]

                if B_key in lora_sd:
                    module.lora_B[adapter_name].weight.data = lora_sd[B_key]

                if S_key in lora_sd:
                    module.scaling[adapter_name] = lora_sd[S_key].item()

    print("✔ Loaded LoRA-only weights.")



class SongEvalGenerator_audio_lora(torch.nn.Module):
    def __init__(self, use_grl=False, lora_r=128, lora_alpha=48):
        super().__init__()

        self.muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")

        # 注入 LoRA
        self.muq = inject_lora_into_muq(
            self.muq,
            r=lora_r,
            alpha=lora_alpha
        )

        self.model = SongEvalGenerator(use_grl=use_grl)
        self.use_grl = use_grl

    def forward(self, wavs, return_grl=False, grl_alpha=1.0):
        output = self.muq(wavs, output_hidden_states=True)
        audio_embeds = output["hidden_states"][6]
        return self.model(audio_feats_vec=audio_embeds, 
                          return_grl=return_grl, grl_alpha=grl_alpha)

    def generate_tag(self, path, inverse=True):
        target_sr = 24000
        device = next(self.parameters()).device
        dtypes = {param.dtype for param in self.parameters()} | {buf.dtype for buf in self.buffers()}

        # print(device, dtypes)

        with torch.no_grad():
            with audioread.audio_open(path) as input_file:
                audio, original_sr = librosa.load(input_file, sr=target_sr, mono=True)
                # print(audio.shape)
                tag_score = self.forward(
                    wavs = torch.tensor(audio, dtype=torch.float16 if torch.float16 in dtypes else torch.float32).view(1,-1).to(device)
                )[0].detach().cpu().item()
                if inverse:
                    tag_score = 5 - tag_score
                return tag_score

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)

        # 保存 Generator 部分
        torch.save(self.model.state_dict(), os.path.join(path, "weights.pt"))

        # 只保存 LoRA 权重 (非常小)
        lora_sd = get_lora_state_dict(self.muq)
        torch.save(lora_sd, os.path.join(path, "muq_lora.pt"))

        with open(os.path.join(path, "type.txt"), "w") as f:
            f.write("SongEvalGenerator_audio_lora")

        print(f"✔ Saved small LoRA-only model to {path}")

    def load_model(self, path):
        type_path = os.path.join(path, "type.txt")
        if not os.path.exists(type_path):
            raise FileNotFoundError("Missing "+type_path)

        # 加载 LoRA
        lora_path = os.path.join(path, "muq_lora.pt")
        if os.path.exists(lora_path):
            lora_sd = torch.load(lora_path, map_location="cpu")
            load_lora_state_dict(self.muq, lora_sd)
            print(f"✔ Loaded LoRA from {lora_path}")

        # 加载 Generator
        gen_path = os.path.join(path, "weights.pt")
        if os.path.exists(gen_path):
            self.model.load_state_dict(torch.load(gen_path))
            print(f"✔ Loaded Generator from {gen_path}")
        else:
            raise FileNotFoundError(f"Missing {gen_path}")
