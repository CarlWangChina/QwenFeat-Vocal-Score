import json
import os
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from tqdm import tqdm
from transformers import AutoProcessor
from qwenaudio.model import QwenAudioScoreModel
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, jsonl_path, processor, max_length=10):
        """
        音频数据集加载器
        Args:
            jsonl_path: 训练数据路径 (JSONL格式)
            processor: Qwen2音频处理器
            max_length: 最大音频长度(秒)
        """
        self.processor = processor
        self.conversation = [
            {
                "role": "user", "content": [
                    {"type": "audio", "audio_url": "input.wav"},
                    {"type": "text", "text": "请评价这段歌声音频的音色好听所以受欢迎程度，给出1到5的整数分数"},
                ]
            }
        ]
        self.text_prompt = self.processor.apply_chat_template(self.conversation, add_generation_prompt=True, tokenize=False)
        self.sampling_rate = processor.feature_extractor.sampling_rate
        self.max_length = max_length * self.sampling_rate  # 转换为采样点数
        print(f"Loading dataset from {jsonl_path} with max length {self.max_length} samples")
        print("sampling_rate:", self.sampling_rate)
        
        with open(jsonl_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["path"]
        score = int(item["score"])-1
        assert 0 <= score < 5, "Score must be between 0 and 4"
        
        # 加载音频并裁剪/填充
        audio, _ = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        elif len(audio) < self.max_length:
            audio = np.pad(audio, (0, max(0, self.max_length - len(audio))))
        
        # 提取特征
        inputs = self.processor(text=self.text_prompt, audios=[audio], return_tensors="pt", padding=True, sample_rate=self.sampling_rate)

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "input_features": inputs.input_features,
            "feature_attention_mask": inputs.feature_attention_mask,
            "score": torch.tensor(score, dtype=torch.float32)
        }

class QwenAudioTrainer:
    def __init__(self, model, processor, train_jsonl, val_jsonl=None, device=None):
        """
        音频评分模型训练器
        Args:
            model: QwenAudioScoreModel实例
            processor: 音频处理器
            train_jsonl: 训练集路径
            val_jsonl: 验证集路径
            device: 训练设备 (自动选择GPU如果可用)
        """
        self.model = model
        self.processor = processor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 创建数据集
        self.train_dataset = AudioDataset(train_jsonl, processor)
        self.val_dataset = AudioDataset(val_jsonl, processor) if val_jsonl else None
        
        # 根据任务类型配置训练参数
        self.output_num = model.output_num
        if self.output_num >= 2:
            self.criterion = CrossEntropyLoss()
            self.metric_name = "Accuracy"
        else:  # 回归任务
            self.criterion = MSELoss()
            self.metric_name = "MSE"
        
        # 训练参数默认值
        self.batch_size = 8
        self.lr = 1e-4
        self.epochs = 10
        self.save_dir = "./checkpoints"
        
    def create_dataloaders(self):
        """创建训练和验证数据加载器"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn
            )
            
        return train_loader, val_loader
    
    def collate_fn(self, batch):
        """批处理函数"""
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        input_features = [item["input_features"] for item in batch]
        feature_attention_mask = [item["feature_attention_mask"] for item in batch]
        
        scores = [item["score"] for item in batch]
        
        return {
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_mask": torch.cat(attention_mask, dim=0),
            "input_features": torch.cat(input_features, dim=0),
            "feature_attention_mask": torch.cat(feature_attention_mask, dim=0),
            "scores": torch.stack(scores)
        }
    
    def train(self):
        """训练主循环"""
        train_loader, val_loader = self.create_dataloaders()
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        best_val_loss = float('inf')
        os.makedirs(self.save_dir, exist_ok=True)
        
        if self.local_rank in [-1, 0]:
            print(self.model)
            self.save_model("initial_model")
            self.logger.info(f"Initial model saved to {self.save_dir}/initial_model")
        
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                # 准备输入数据
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                input_features = batch["input_features"].to(self.device)
                feature_attention_mask = batch["feature_attention_mask"].to(self.device)
                scores = batch["scores"].to(self.device)

                if self.output_num >= 2:
                    scores = scores.long()
                elif self.output_num == 1:
                    scores = scores.view(-1, 1)
                    
                # 前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, input_features=input_features, feature_attention_mask=feature_attention_mask)
                # print("scores:", scores)
                # print("outputs:", outputs)
                # print(self.criterion)
                
                # 计算损失
                loss = self.criterion(outputs, scores)
                # print(outputs, scores)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_train_loss = total_loss / len(train_loader)
            
            # 验证阶段
            val_metrics = {}
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics["loss"]
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f"best_model_epoch{epoch+1}.pt")
            
            # 打印epoch结果
            print(f"\nEpoch {epoch+1} Results:")
            print(f"Train Loss: {avg_train_loss:.4f}")
            if val_metrics:
                print(f"Val Loss: {val_metrics['loss']:.4f}")
                print(f"Val {self.metric_name}: {val_metrics['metric']:.4f}")
            
            # 定期保存模型
            if (epoch + 1) % 2 == 0:
                self.save_model(f"model_epoch{epoch+1}.pt")
    
    def evaluate(self, dataloader):
        """评估模型性能"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                input_features = batch["input_features"].to(self.device)
                feature_attention_mask = batch["feature_attention_mask"].to(self.device)
                scores = batch["scores"].to(self.device)

                if self.output_num >= 2:
                    scores_for_loss = scores.long()
                elif self.output_num == 1:
                    scores_for_loss = scores.view(-1, 1)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, input_features=input_features, feature_attention_mask=feature_attention_mask)

                loss = self.criterion(outputs, scores_for_loss)
                total_loss += loss.item()
                
                # 计算预测值
                if self.output_num >= 2:
                    # 分类任务：取argmax作为预测结果
                    preds = torch.argmax(outputs, dim=1)
                else:
                    # 回归任务：直接使用输出值
                    preds = outputs.squeeze()
                
                # 收集预测结果和真实标签
                all_preds.append(preds.view(1,-1))
                all_labels.append(scores)
        
        # 计算评估指标
        avg_loss = total_loss / len(dataloader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        metrics = {"loss": avg_loss}
        
        if self.output_num >= 2:
            # 计算分类准确率
            accuracy = (all_preds == all_labels).float().mean()
            metrics["metric"] = accuracy.item()
        else:
            # 计算均方误差
            mse = ((all_preds - all_labels) ** 2).mean()
            metrics["metric"] = mse.item()
        
        return metrics
    
    def save_model(self, filename):
        """保存模型和处理器"""
        self.model.save_model(os.path.join(self.save_dir, filename))
        self.processor.save_pretrained(os.path.join(self.save_dir, "processor"))
