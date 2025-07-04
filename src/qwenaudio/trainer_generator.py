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
import qwenaudio.prompts
import qwenaudio.audio_cut
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.amp import autocast, GradScaler

class AudioDataset(Dataset):
    def __init__(self, json_path , processor, max_length=10):
        """
        音频数据集加载器
        Args:
            jsonl_path: 训练数据路径 (JSONL格式)
            processor: Qwen2音频处理器
            max_length: 最大音频长度(秒)
        """
        self.processor = processor
        self.sampling_rate = processor.feature_extractor.sampling_rate
        self.max_length = max_length * self.sampling_rate  # 转换为采样点数
        print(f"Loading dataset from {json_path} with max length {self.max_length} samples")
        print("sampling_rate:", self.sampling_rate)

        self.data = []
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["audio"]
        text_prompt = item["text"]
        
        # 加载音频并裁剪/填充
        audio, _ = qwenaudio.audio_cut.random_cut(audio_path, self.sampling_rate, segment_duration=30)
        print(audio.shape)
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        elif len(audio) < self.max_length:
            audio = np.pad(audio, (0, max(0, self.max_length - len(audio))))
        
        
        return {
            "raw_audio": audio,
            "text": text_prompt,
            "audio_path": audio_path
        }
    
    def collate_fn(self, batch):
        """使用processor批量处理"""
        texts = [item["text"] for item in batch]
        audios = [item["raw_audio"] for item in batch]
        
        # 统一处理整个批次
        inputs = self.processor(
            text=texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            sample_rate=self.sampling_rate
        )
        
        return inputs

class QwenAudioTrainer:
    def __init__(self, model, processor, train_set, val_set, test_set=None, device=None, local_rank=-1, world_size=-1):
        """
        音频评分模型训练器（支持多卡训练）
        Args:
            model: QwenAudioScoreModel实例
            processor: 音频处理器
            train_set: (训练集路径, 标签路径) 元组
            val_set: (验证集路径, 标签路径) 元组
            test_set: (测试集路径, 标签路径) 元组 (可选)
            device: 训练设备 (自动选择GPU如果可用)
            local_rank: 分布式训练的本地进程编号
            world_size: 分布式训练的世界大小
        """
        self.model = model
        self.processor = processor
        self.local_rank = local_rank
        self.world_size = world_size
        
        # 设置设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建数据集
        self.train_dataset = AudioDataset(train_set, processor)
        self.val_dataset = AudioDataset(val_set, processor)
        self.test_dataset = AudioDataset(test_set[0], test_set[1], processor) if test_set else None
        
        # 损失函数
        self.criterion = CrossEntropyLoss()
        
        # 训练参数默认值
        self.batch_size = 8
        self.lr = 1e-4
        self.epochs = 10
        self.save_dir = "./checkpoints"
        self.use_amp = True  # 启用混合精度训练
        self.scaler = GradScaler("cuda")  # AMP梯度缩放器

        # 初始化日志记录器（只在主进程）
        if self.local_rank in [-1, 0]:
            os.makedirs(self.save_dir, exist_ok=True)
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)
            
            # 文件处理器
            file_handler = logging.FileHandler('training.log', mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            self.logger = logger
            self.logger.info("Training started")
    
    def create_dataloaders(self):
        """创建数据加载器（支持分布式采样器）"""
        train_sampler = None
        if self.local_rank != -1:
            train_sampler = DistributedSampler(
                self.train_dataset, 
                shuffle=True, 
                rank=self.local_rank, 
                num_replicas=self.world_size
            )
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=self.train_dataset.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = None
        if self.test_dataset:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.test_dataset.collate_fn,
                num_workers=2,
                pin_memory=True
            )
            
        return train_loader, val_loader, test_loader, train_sampler

    def train(self):
        """训练主循环（支持多卡训练和混合精度）"""
        self.model.to(self.device)
        
        # 分布式训练设置
        if self.local_rank != -1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        print("memory:", torch.cuda.memory_allocated() / 1024 / 1024, "MB")

        train_loader, val_loader, test_loader, train_sampler = self.create_dataloaders()
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        best_val_loss = float('inf')
        
        # 初始模型保存（只在主进程）
        if self.local_rank in [-1, 0]:
            self.save_model("initial_model")
            self.logger.info(f"Initial model saved to {self.save_dir}/initial_model")

        # 训练循环
        for epoch in range(self.epochs):
            # 设置分布式采样器的epoch
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # 训练阶段
            self.model.train()
            epoch_loss = 0.0
            
            # 进度条（只在主进程显示）
            progress_bar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{self.epochs} [Train]",
                disable=not (self.local_rank in [-1, 0])
            )
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                # 混合精度训练步骤
                with autocast("cuda"):
                    loss = self.train_step(batch)
                
                # 梯度缩放和反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                # 统计损失
                epoch_loss += loss.item()
                
                # 更新进度条
                if self.local_rank in [-1, 0]:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 计算平均训练损失
            epoch_loss /= len(train_loader)
            
            # 验证阶段（只在主进程）
            val_loss = float('inf')
            if self.local_rank in [-1, 0]:
                val_loss, val_metrics = self.evaluate(val_loader)
                self.logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {epoch_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_metrics.get('accuracy', 0.0):.4f}"
                )
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f"best_model_epoch_{epoch+1}")
                    self.logger.info(f"Best model saved (Val Loss: {val_loss:.4f})")
            
            # 定期保存模型
            # if (epoch + 1) % 2 == 0 and self.local_rank in [-1, 0]:
            self.save_model(f"model_epoch_{epoch+1}")
        
        # 训练结束后测试（只在主进程）
        if self.local_rank in [-1, 0] and test_loader:
            test_loss, test_metrics = self.evaluate(test_loader)
            self.logger.info(
                f"\nTest Results | "
                f"Loss: {test_loss:.4f} | "
                f"Acc: {test_metrics.get('accuracy', 0.0):.4f}"
            )
            return test_metrics

    def train_step(self, batch):
        batch = batch.to(self.device)
        
        outputs = self.model(**batch)
        # 计算损失 - 使用交叉熵损失
        logits = outputs.logits

        labels = batch['input_ids']
        
        # 计算损失时忽略填充部分
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
  
        # print("batch:", batch.attention_mask.shape, batch.input_ids.shape, logits.shape, shift_logits.shape, shift_labels.shape)
        
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return loss
    
    def evaluate(self, dataloader):
        """评估模型性能（返回平均损失和指标）"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", disable=not (self.local_rank in [-1, 0])):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 混合精度评估
                with autocast("cuda"):
                    outputs = self.model(**batch)
                    logits = outputs.logits
                    labels = batch['input_ids']
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    loss = self.criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                
                # 统计损失
                total_loss += loss.item() * shift_labels.size(0)
                
                # 计算准确率
                _, preds = torch.max(shift_logits, dim=-1)
                total_correct += (preds == shift_labels).sum().item()
                total_samples += shift_labels.size(0)
        
        # 计算平均指标
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            # 可添加更多指标如F1分数等
        }
        
        return avg_loss, metrics

    def save_model(self, filename):
        """保存模型和处理器"""
        save_path = os.path.join(self.save_dir, filename)
        os.makedirs(save_path, exist_ok=True)
        # 保存LoRA适配器
        self.model.module.save_pretrained(f"{save_path}/lora_weights")

    def test(self, test_set=None):
        """测试模型性能（可指定测试集）"""
        if test_set:
            self.test_dataset = AudioDataset(test_set[0], test_set[1], self.processor)
        
        if not self.test_dataset:
            raise ValueError("Test dataset not provided")
        
        _, _, test_loader, _ = self.create_dataloaders()
        test_loss, test_metrics = self.evaluate(test_loader)
        
        if self.local_rank in [-1, 0]:
            self.logger.info("\n" + "="*50)
            self.logger.info(f"Test Loss: {test_loss:.4f}")
            self.logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            self.logger.info("="*50)
        
        return test_metrics
