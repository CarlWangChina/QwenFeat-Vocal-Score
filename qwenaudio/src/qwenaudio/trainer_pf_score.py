import json
import os
import sys
import torch
import librosa
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
import logging
import math
from collections import defaultdict

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

# 注意：这里假设 qwenaudio.model 在您的路径中可用
from qwenaudio.model import QwenAudioScoreModel, QwenAudioTowerScoreModel
import qwenaudio.audio_cut

# 扩展交叉熵损失（示例权重：距离越远权重越大）
class OrdinalCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, weight_matrix):
        super().__init__()
        self.weight_matrix = weight_matrix  # 形状 [num_classes, num_classes]

    def forward(self, inputs, targets):
        log_probs = torch.nn.functional.log_softmax(inputs, dim=1)
        loss = -self.weight_matrix[targets] * log_probs
        return loss.mean()

def calculate_weights(json_path):
    """统计JSON文件中各分数类别的权重"""
    count_dict = defaultdict(int)
    
    # 统计每个分数出现的次数
    with open(json_path, 'r') as f:
        data = json.load(f)
    for line in data:
        score = line['score']
        count_dict[score] += 1
 
    # 计算权重（使用类别频率的倒数）
    total = sum(count_dict.values())
    weights = {k: total/v for k, v in count_dict.items()}
    
    # 归一化处理（使权重和为1）
    normalized_weights = {k: v/sum(weights.values()) for k, v in weights.items()}
    
    return {
        'raw_counts': dict(count_dict),
        'inverse_weights': weights,
        'normalized_weights': normalized_weights
    }

class AudioDataset(Dataset):
    def __init__(self, json_path, processor, max_length=10):
        """
        音频数据集加载器
        Args:
            jsonl_path: 训练数据路径 (JSONL格式)
            processor: Qwen2音频处理器
            max_length: 最大音频长度(秒)
        """
        self.processor = processor
        self.sampling_rate = processor.feature_extractor.sampling_rate
        self.max_length_second = max_length
        self.max_length = max_length * self.sampling_rate  # 转换为采样点数
        # if dist.get_rank() == 0:
        print(f"Loading dataset from {json_path} with max length {self.max_length} samples")
        print("sampling_rate:", self.sampling_rate)
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.weights = calculate_weights(json_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item["audio"]
        score = int(item["score"])-1
        text_prompt = item["text"]
        assert 0 <= score < 5, "Score must be between 0 and 4"
        
        # 加载音频并裁剪/填充
        # audio, _ = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
        audio, _ = qwenaudio.audio_cut.random_cut(audio_path, self.sampling_rate, segment_duration=self.max_length_second)
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        elif len(audio) < self.max_length:
            audio = np.pad(audio, (0, max(0, self.max_length - len(audio))))
        
        # 提取特征
        # inputs = self.processor(text=text_prompt, audios=[audio], return_tensors="pt", padding=True, sample_rate=self.sampling_rate)
        # inputs = self.processor.feature_extractor(
        #             audio, 
        #             return_attention_mask=True,
        #             sampling_rate=self.sampling_rate,
        #             return_tensors="pt"
        #         )

        return {
            "audio": audio,
            "audio_path": audio_path,
            # "input_ids": inputs.input_ids,
            # "attention_mask": inputs.attention_mask,
            # "input_features": inputs.input_features,
            # "feature_attention_mask": inputs.feature_attention_mask,
            "text_prompt": text_prompt,
            "score": torch.tensor(score, dtype=torch.float32)
        }

class QwenAudioTrainer:
    def __init__(self, model, processor, train_json, val_json=None, device=None, local_rank=-1, world_size=-1):
        """
        音频评分模型训练器（支持多卡训练）
        Args:
            model: QwenAudioScoreModel实例
            processor: 音频处理器
            train_json: 训练集路径
            val_json: 验证集路径
            device: 训练设备 (自动选择GPU如果可用)
            local_rank: 分布式训练的本地进程编号
        """
        self.model = model
        self.processor = processor
        self.local_rank = local_rank
        self.device = device
        self.world_size = world_size
        
        # self.model.to(self.device)

        # 创建数据集
        self.train_dataset = AudioDataset(train_json, processor)
        self.val_dataset = AudioDataset(val_json, processor) if val_json else None
        
        # 根据任务类型配置训练参数
        self.output_num = model.output_num

        self.classify_weight = []
        for i in range(self.output_num):
            self.classify_weight.append(self.train_dataset.weights["normalized_weights"][i+1])
        # self.classify_weight = torch.tensor(self.classify_weight).to(self.device)

        if self.output_num >= 2:
            self.criterion = CrossEntropyLoss()

            # # 5个类别，权重矩阵设计为距离的倒数
            # num_classes = 5
            # weight_matrix = torch.ones((num_classes, num_classes))
            # for i in range(num_classes):
            #     for j in range(num_classes):
            #         weight_matrix[i][j] = math.exp(abs(i - j))# * max(self.classify_weight[i], self.classify_weight[j])  # 距离越远权重越大
            # # 归一化weight_matrix
            # weight_matrix = weight_matrix / weight_matrix.sum(dim=1, keepdim=True)

            # self.criterion = OrdinalCrossEntropy(num_classes, weight_matrix)
            # self.criterion.weight_matrix = weight_matrix.to(self.device)
            # print(self.criterion.weight_matrix)

            self.metric_name = "Accuracy"
        else:  # 回归任务
            self.criterion = MSELoss()
            self.metric_name = "MSE"
        
        # 训练参数默认值
        self.batch_size = 8
        self.lr = 1e-4
        self.epochs = 10
        self.save_dir = "./checkpoints"

        if self.local_rank in [-1, 0]:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)  # 设置最低日志级别
            
            # 创建文件处理器（FileHandler）
            file_handler = logging.FileHandler('app.log', mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # 处理器级别
            
            # 创建格式化器（Formatter）
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 将格式化器绑定到处理器
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            self.logger = logger
            self.logger.info("start")
    
    def create_dataloaders(self):
        """创建训练和验证数据加载器（支持分布式采样器）"""
        train_sampler = DistributedSampler(self.train_dataset, shuffle=False, rank=self.local_rank, num_replicas=self.world_size)
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=4,
                pin_memory=True
            )
            
        return train_loader, val_loader, train_sampler
    
    def collate_fn(self, batch):
        """批处理函数"""
        audios = [item["audio"] for item in batch]
        # input_ids = [item["input_ids"] for item in batch]
        # attention_mask = [item["attention_mask"] for item in batch]
        # input_features = [item["input_features"] for item in batch]
        # feature_attention_mask = [item["feature_attention_mask"] for item in batch]
        
        scores = [item["score"] for item in batch]
        
        return {
            "audio": audios,
            # "input_ids": torch.cat(input_ids, dim=0),
            # "attention_mask": torch.cat(attention_mask, dim=0),
            # "input_features": torch.cat(input_features, dim=0),
            # "feature_attention_mask": torch.cat(feature_attention_mask, dim=0),
            "scores": torch.stack(scores)
        }
    
    def train(self):
        """训练主循环（支持多卡训练）"""
        self.model.to(self.device)
        if self.local_rank != -1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        train_loader, val_loader, train_sampler = self.create_dataloaders()
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        best_val_loss = float('inf')
        os.makedirs(self.save_dir, exist_ok=True)
        if self.local_rank in [-1, 0]:
            print(self.model)
            self.save_model(f"initial_model")
            self.logger.info(f"Initial model saved to {self.save_dir}/initial_model.pt")

        dist.barrier()
        for epoch in range(self.epochs):
            # 设置分布式采样器的epoch
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # 训练阶段
            self.model.train()
            total_loss = 0.0
            # 只在主进程显示进度条
            if self.local_rank in [-1, 0]:
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            else:
                progress_bar = train_loader
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                local_loss = self.train_step(batch, optimizer)

                total_loss += local_loss

                if self.local_rank in [-1, 0]:
                    progress_bar.set_postfix({"loss": f"{local_loss:.4f}"})
                
                
                if self.local_rank in [-1, 0]:
                    self.logger.info(f"Training loss: {local_loss:.4f}")
            
            # 计算平均训练损失（跨所有GPU）
            if self.local_rank != -1:
                total_loss_tensor = torch.tensor(total_loss, device=self.device)
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                total_loss = total_loss_tensor.item() / dist.get_world_size()
            
            avg_train_loss = total_loss / len(train_loader)
            
            if self.local_rank in [-1, 0]:
                self.logger.info(f"Average training loss: {avg_train_loss:.4f}")

            dist.barrier()
            # 验证阶段（只在主进程进行）
            val_metrics = {}
            if val_loader and self.local_rank in [-1, 0]:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics["loss"]
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f"best_model_epoch/{epoch+1}")
                    self.logger.info(f"Best model saved to {self.save_dir}/best_model_epoch/{epoch+1}.pt")
            
            # 打印epoch结果（只在主进程）
            if self.local_rank in [-1, 0]:
                print(f"\nEpoch {epoch+1} Results:")
                print(f"Train Loss: {avg_train_loss:.4f}")
                if val_metrics:
                    print(f"Val Loss: {val_metrics['loss']:.4f}")
                    print(f"Val {self.metric_name}: {val_metrics['metric']:.4f}")
                    self.logger.info(f"Epoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f}, "+\
                                     f"Val Loss: {val_metrics['loss']:.4f}, "+\
                                     f"Val {self.metric_name}: {val_metrics['metric']:.4f}, "+\
                                     f"Acc_dis:{val_metrics['acc_dis']:.4f} "+\
                                     f"Acc_area:{val_metrics['acc_area']:.4f} ")
            
            # 定期保存模型（只在主进程）
            if (epoch + 1) % 8 == 0 and self.local_rank in [-1, 0]:
                self.save_model(f"model_epoch{epoch+1}")
                self.logger.info(f"Model saved to {self.save_dir}/model_epoch{epoch+1}.pt")
    
    def train_step(self, batch, optimizer):
        # 准备输入数据
        # input_ids = batch["input_ids"].to(self.device)
        # attention_mask = batch["attention_mask"].to(self.device)
        # input_features = batch["input_features"].to(self.device)
        # feature_attention_mask = batch["feature_attention_mask"].to(self.device)
        ft = self.processor.feature_extractor(batch["audio"], return_attention_mask=True, padding="max_length")
        scores = batch["scores"].to(self.device)

        if self.output_num >= 2:
            scores = scores.long()
        elif self.output_num == 1:
            scores = scores.view(-1, 1)
            
        # 前向传播
        # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, input_features=input_features, feature_attention_mask=feature_attention_mask)
        outputs = self.model(torch.tensor(ft.input_features).cuda())
        
        # 计算损失
        loss = self.criterion(outputs, scores)
        
        # 反向传播
        loss.backward()
        optimizer.step()

        return loss.item()

    def evaluate(self, dataloader):
        """评估模型性能（只在主进程调用）"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        data_count = 0
        data_check_success = 0
        data_score_distance_sum = 0
        target_value_sum = 0
        score_distance_less = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # input_ids = batch["input_ids"].to(self.device)
                # feature_attention_mask = batch["feature_attention_mask"].to(self.device)
                ft = self.processor.feature_extractor(batch["audio"], return_attention_mask=True, padding="max_length")
                scores = batch["scores"].to(self.device)

                if self.output_num >= 2:
                    scores_for_loss = scores.long()
                elif self.output_num == 1:
                    scores_for_loss = scores.view(-1, 1)

                # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, input_features=input_features, feature_attention_mask=feature_attention_mask)
                outputs = self.model(torch.tensor(ft.input_features).cuda())

                loss = self.criterion(outputs, scores_for_loss)
                total_loss += loss.item()
                
                # 计算预测值
                if self.output_num >= 2:
                    
                    outputs = torch.softmax(outputs[0], dim=0)
                    out_id = outputs.argmax().item()
                    target_id = scores.item()
                    outputs = outputs.tolist()
                    target_value = outputs[int(target_id)]

                    score_distance = abs(target_id-out_id)
                    self.logger.info(f"target_id={target_id} out_id={out_id}, score_distance={score_distance} target_value={target_value} outs={outputs}")

                    if score_distance == 0:
                        data_check_success += 1
                    if score_distance < 2:
                        score_distance_less += 1
                    data_score_distance_sum += score_distance
                    target_value_sum += target_value
                    data_count += 1

                else:
                    # 回归任务：直接使用输出值
                    preds = outputs.squeeze()
                
                    # 收集预测结果和真实标签
                    all_preds.append(preds.cpu())
                    all_labels.append(scores.cpu())
        
        # 计算评估指标
        avg_loss = total_loss / len(dataloader)
        
        metrics = {"loss": avg_loss}
        
        if self.output_num >= 2:
            # 计算分类准确率
            metrics["metric"] = data_check_success/data_count
            metrics["acc_dis"] = data_score_distance_sum/data_count
            metrics["acc_area"] = score_distance_less/data_count
        else:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            # 计算均方误差
            mse = ((all_preds - all_labels) ** 2).mean()
            metrics["metric"] = mse.item()
        
        return metrics
    
    def save_model(self, filename):
        """保存模型和处理器（只在主进程调用）"""
        # 如果是DDP模型，获取原始模型
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        os.makedirs(self.save_dir, exist_ok=True)
        model_to_save.save_model(os.path.join(self.save_dir, filename))
        self.processor.save_pretrained(os.path.join(self.save_dir, filename, "processor"))
