import json
import os
import random
import sys
import torch
import librosa
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
import logging
import math
from collections import defaultdict
from scipy.stats import pearsonr

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

from audioscore.dataset import AudioDataset_pairwise
import audioscore.audio_cut

def pairwise_loss(score_i, score_j, label, margin=0.2):
    # label=1表示i应高于j，label=0表示j应高于i
    diff = score_i - score_j
    return torch.mean(torch.log(1 + torch.exp(-label * (diff - margin))))

def binary_rank_loss(score_i, score_j, label):
    # label∈{0,1}表示i是否优于j
    prob = torch.sigmoid(score_i - score_j)
    return torch.nn.functional.binary_cross_entropy(prob, label)

def masked_average(tensor, mask):
    """
    tensor: [16, 854, 5]
    mask: [16, 854] (0/1 mask)
    返回: [16, 5]
    """
    # 扩展mask维度以匹配输入张量
    mask_expanded = mask.unsqueeze(-1)  # [16, 854, 1]
    
    # 应用mask（将无效位置置零）
    masked_tensor = tensor * mask_expanded  # [16, 854, 5]
    
    # 计算有效元素数量（避免除零）
    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [16, 1]
    
    # 求和并计算平均值
    sum_result = masked_tensor.sum(dim=1)  # [16, 5]
    avg_result = sum_result / valid_counts  # [16, 5]
    
    return avg_result

class Trainer:
    def __init__(self, model, train_json, val_json=None, device=None, local_rank=-1, world_size=-1, data_type="use_audio_feat", save_dir="checkpoints"):
        """
        音频评分模型训练器（支持多卡训练）
        Args:
            train_json: 训练集路径
            val_json: 验证集路径
            device: 训练设备 (自动选择GPU如果可用)
            local_rank: 分布式训练的本地进程编号
        """
        self.model = model
        self.local_rank = local_rank
        self.device = device
        self.world_size = world_size
        self.sub_batch_size = 32
        self.contact_batch = False
        self.use_pitch_feature = False
        self.data_type = data_type
        self.grl = None
        self.gamma = 0.999
        
        # self.model.to(self.device)

        self.train_dataset = AudioDataset_pairwise(train_json)
        self.val_dataset = AudioDataset_pairwise(val_json) if val_json else None
        
        # 训练参数默认值
        self.batch_size = 8
        self.lr = 1e-4
        self.epochs = 10
        self.save_dir = save_dir
        self.eval_steps = 2000  # 每500步验证一次
        self.log_steps = 500    # 每50步记录一次日志
        
        self.criterion = CrossEntropyLoss()
        self.metric_name = "Accuracy"

        if self.local_rank in [-1, 0]:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)  # 设置最低日志级别
            
            # 创建文件处理器（FileHandler）
            os.makedirs(self.save_dir, exist_ok=True)
            file_handler = logging.FileHandler(f'{self.save_dir}/train.log', mode='a', encoding='utf-8')
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
        # 训练采样器添加随机性
        train_sampler = DistributedSampler(
            self.train_dataset, 
            shuffle=True,  # 改为True以启用随机采样
            rank=self.local_rank, 
            num_replicas=self.world_size,
            seed=42  # 设置随机种子
        )
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=False,  # 当使用sampler时，shuffle应为False
            collate_fn=AudioDataset_pairwise.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=AudioDataset_pairwise.collate_fn,
                num_workers=4,
                pin_memory=True
            )
            
        return train_loader, val_loader, train_sampler
    
    def preprocess_feat(self, ppg, vec, pit):
        
        ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
        ppg = torch.FloatTensor(ppg)
        # ppg = torch.zeros_like(ppg)

        vec = np.repeat(vec, 2, 0)  # 320 PPG -> 160 * 2
        vec = torch.FloatTensor(vec)
        # vec = torch.zeros_like(vec)

        delta = random.uniform(0.66, 1.49)
        pit = torch.FloatTensor(pit) * delta
        
        len_pit = pit.size()[0]
        len_vec = vec.size()[0]
        len_ppg = ppg.size()[0]
        len_min = min(len_pit, len_vec)
        len_min = min(len_min, len_ppg)
        pit = pit[:len_min]
        vec = vec[:len_min, :]
        ppg = ppg[:len_min, :]
        pit = audioscore.audio_cut.f0_to_coarse(pit)

        ppg = ppg.view(1, -1, 1280)
        vec = vec.view(1, -1, 256)
        pit = pit.view(1, -1)

        return ppg, vec, pit

    def train(self):
        """训练主循环（支持多卡训练）"""
        print("start training")
        self.model.to(self.device)
        print(f"create ddp model rank={self.local_rank}, {self.device}")
        self.model = DistributedDataParallel(self.model, broadcast_buffers=False)
        print("create ddp model")
        train_loader, val_loader, train_sampler = self.create_dataloaders()
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        # 添加StepLR学习率调度器
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        best_val_accuracy = float('0')
        os.makedirs(self.save_dir, exist_ok=True)
        if self.local_rank in [-1, 0]:
            print(self.model)
            self.save_model(f"initial_model")
            self.logger.info(f"Initial model saved to {self.save_dir}/initial_model.pt")
            # 记录初始学习率
            current_lr = optimizer.param_groups[0]['lr']
            self.logger.info(f"Initial learning rate: {current_lr}")

        dist.barrier()
                    
        # 只在主进程进行验证
        # if val_loader and self.local_rank in [-1, 0]:
        #     val_avg_loss, val_accuracy = self.evaluate(val_loader)
            
        #     # 记录验证结果
        #     self.logger.info(f"Initial: Validation loss: {val_avg_loss:.4f} accuracy: {val_accuracy:.4f}")
        
        dist.barrier()

        global_step = 0
        for epoch in range(self.epochs):
            # self.evaluate(val_loader)
            
            # 设置分布式采样器的epoch以确保每个epoch的随机性不同
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # 训练阶段
            self.model.train()
            total_loss = 0.0
            epoch_loss = 0.0
            # 只在主进程显示进度条
            if self.local_rank in [-1, 0]:
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            else:
                progress_bar = train_loader
            
            for step, batch in enumerate(progress_bar):
                
                local_loss = self.train_step(batch, optimizer, epoch)
                total_loss += local_loss
                epoch_loss += local_loss
                global_step += 1

                # 记录日志
                if global_step % self.log_steps == 0 and self.local_rank in [-1, 0]:
                    avg_loss = total_loss / self.log_steps
                    current_lr = optimizer.param_groups[0]['lr']
                    self.logger.info(f"Step {global_step}: Training loss: {avg_loss:.4f}, Learning rate: {current_lr:.2e}")
                    if hasattr(progress_bar, 'set_postfix'):
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}", 
                            "step": f"{global_step}",
                            "lr": f"{current_lr:.2e}"
                        })
                    total_loss = 0.0  # 重置

                # 验证阶段和学习率调整
                if global_step % self.eval_steps == 0:
                    # 所有进程同步
                    dist.barrier()
                    
                    # 只在主进程进行验证
                    if val_loader and self.local_rank in [-1, 0]:
                        val_avg_loss, val_accuracy = self.evaluate(val_loader)
                        
                        # 记录验证结果
                        self.logger.info(f"Step {global_step}: Validation loss: {val_avg_loss:.4f} accuracy: {val_accuracy:.4f}")
                        
                        # 保存最佳模型
                        if val_accuracy > best_val_accuracy:
                            best_val_accuracy = val_accuracy
                            self.save_model(f"best_model_step_{global_step}")
                            self.logger.info(f"Best model saved to {self.save_dir}/best_model_step_{global_step}")
                    
                    # 所有进程都调整学习率，确保同步
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step()
                    new_lr = optimizer.param_groups[0]['lr']
                    
                    # 只在主进程记录学习率变化
                    if self.local_rank in [-1, 0]:
                        self.logger.info(f"Step {global_step}: Learning rate updated: {old_lr:.2e} -> {new_lr:.2e}")
                
                dist.barrier()

            # 计算epoch平均训练损失（跨所有GPU）
            if self.local_rank != -1:
                epoch_loss_tensor = torch.tensor(epoch_loss, device=self.device)
                dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
                epoch_loss = epoch_loss_tensor.item() / dist.get_world_size()
            
            avg_epoch_loss = epoch_loss / len(train_loader)

            print(f"Epoch {epoch+1} avg_train_loss:{avg_epoch_loss:.4f} rank={self.local_rank}")
            
            if self.local_rank in [-1, 0]:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"Epoch {epoch+1} Average training loss: {avg_epoch_loss:.4f}, Learning rate: {current_lr:.2e}")

            # 每个epoch结束时也进行验证和学习率调整
            dist.barrier()
            
            # 只在主进程进行验证
            if val_loader and self.local_rank in [-1, 0]:
                val_avg_loss, val_accuracy = self.evaluate(val_loader)
                self.logger.info(f"Epoch {epoch+1} Validation loss: {val_avg_loss:.4f} accuracy: {val_accuracy:.4f}")
                
                # 保存最佳模型（基于epoch）
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.save_model(f"best_model_epoch_{epoch+1}")
                    self.logger.info(f"Best model saved to {self.save_dir}/best_model_epoch_{epoch+1}")
            
            # 所有进程都调整学习率，确保同步
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            
            # 只在主进程记录学习率变化
            if self.local_rank in [-1, 0]:
                self.logger.info(f"Epoch {epoch+1}: Learning rate updated: {old_lr:.2e} -> {new_lr:.2e}")
            
            # 定期保存模型（只在主进程）
            if (epoch + 1) % 8 == 0 and self.local_rank in [-1, 0]:
                self.save_model(f"model_epoch_{epoch+1}")
                self.logger.info(f"Model saved to {self.save_dir}/model_epoch_{epoch+1}")
    
    def train_step(self, batch, optimizer, epoch):
        self.model.train()
        optimizer.zero_grad()

        # print(batch["muq_a"].shape)

        train_size = 5000

        outputs_a = self.model(
            audio_feats_vec = batch["muq_a"][:,:train_size,:].to(self.device),
            mask = batch["mask_a"][:,:train_size].to(self.device),
        )
        outputs_b = self.model(
            audio_feats_vec = batch["muq_b"][:,:train_size,:].to(self.device),
            mask = batch["mask_b"][:,:train_size].to(self.device),
        )

        loss = pairwise_loss(outputs_a, outputs_b, batch["labels"].to(self.device).float())
        # loss = pairwise_loss(outputs_b, outputs_a, batch["labels"].to(self.device).float())

        # print("batch", batch)
        # print("outputs_a,outputs_b,loss:", outputs_a,outputs_b, outputs_a-outputs_b,batch["labels"],loss)

        loss.backward()
        total_loss = loss.item()
        optimizer.step()
        
        return total_loss

    def evaluate(self, val_loader):
        """验证函数"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Evaluating")
            for batch in progress_bar:
                # print("\nevaluate:",
                #     batch["muq_a"].shape,
                #     batch["mask_a"].shape,
                #     batch["muq_b"].shape,
                #     batch["mask_b"].shape
                # )
                # 前向传播
                outputs_a = self.model(
                    audio_feats_vec=batch["muq_a"].to(self.device),
                    mask=batch["mask_a"].to(self.device),
                )
                outputs_b = self.model(
                    audio_feats_vec=batch["muq_b"].to(self.device),
                    mask=batch["mask_b"].to(self.device),
                )
                
                # 计算损失
                loss = pairwise_loss(outputs_a, outputs_b, batch["labels"].to(self.device).float())
                # loss = pairwise_loss(outputs_b, outputs_a, batch["labels"].to(self.device).float())
                progress_bar.set_postfix(loss=loss.item())
                total_loss += loss.item()
                
                # 计算准确率
                # 预测：如果outputs_a < outputs_b，则预测为1，否则为0
                preds = (outputs_a > outputs_b).float()
                labels = batch["labels"].to(self.device).float()
                
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_samples += len(labels)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
    
    def save_model(self, filename):
        """保存模型和处理器（只在主进程调用）"""
        # 如果是DDP模型，获取原始模型
        model_to_save = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        os.makedirs(self.save_dir, exist_ok=True)
        model_to_save.save_model(os.path.join(self.save_dir, filename))