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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
import logging
import datetime
import math
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "src"))

from audioscore.dataset import AudioDataset_pairwise_audio, ROOT_DIR as DATASET_ROOT_DIR
import audioscore.audio_cut

def pairwise_loss(score_i, score_j, label):
    # label=1表示i应高于j，label=0表示i应低于j
    diff = score_i - score_j
    return torch.nn.functional.binary_cross_entropy_with_logits(diff, label)

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
    def __init__(
        self,
        model,
        spk_encoder,
        train_json,
        val_json=None,
        device=None,
        local_rank=-1,
        world_size=-1,
        data_type="use_audio_feat",
        save_dir="checkpoints",
        train_cache_dir=None,
        train_cache_mode="none",
        train_cache_base_dir=None,
        train_cache_target_sr=24000,
        train_cache_segment_duration=60,
        train_cache_kind="segment",
        train_cache_max_duration=None,
        train_cache_cut_mode="random",
        val_cache_dir=None,
        val_cache_mode="none",
        val_cache_base_dir=None,
        val_cache_target_sr=24000,
        val_cache_segment_duration=60,
        val_cache_kind="segment",
        val_cache_max_duration=None,
        val_cache_cut_mode="stable",
        initial_global_step=0,
    ):
        """
        音频评分模型训练器（支持多卡训练）
        Args:
            train_json: 训练集路径
            val_json: 验证集路径
            device: 训练设备 (自动选择GPU如果可用)
            local_rank: 分布式训练的本地进程编号
        """
        self.model = model
        self.spk_encoder = spk_encoder
        self.local_rank = local_rank
        self.device = device
        self.world_size = world_size
        self.sub_batch_size = 32
        self.contact_batch = False
        self.use_pitch_feature = False
        self.data_type = data_type
        self.grl = None
        self.gamma = 0.999
        self.grl_gm = 0.8
        self.InitialValidation = True
        
        # self.model.to(self.device)

        self.train_dataset = AudioDataset_pairwise_audio(
            train_json,
            cache_dir=train_cache_dir,
            cache_mode=train_cache_mode,
            cache_base_dir=train_cache_base_dir,
            cache_target_sr=train_cache_target_sr,
            cache_segment_duration=train_cache_segment_duration,
            cache_kind=train_cache_kind,
            cache_max_duration=train_cache_max_duration,
            cache_cut_mode=train_cache_cut_mode,
        )
        self.val_dataset = AudioDataset_pairwise_audio(
            val_json,
            cache_dir=val_cache_dir,
            cache_mode=val_cache_mode,
            cache_base_dir=val_cache_base_dir,
            cache_target_sr=val_cache_target_sr,
            cache_segment_duration=val_cache_segment_duration,
            cache_kind=val_cache_kind,
            cache_max_duration=val_cache_max_duration,
            cache_cut_mode=val_cache_cut_mode,
        ) if val_json else None
        
        # 训练参数默认值
        self.batch_size = 8
        self.lr = 1e-4           # head + LayerNorm 的学习率
        self.lora_lr = None      # LoRA adapter 的学习率，None 表示与 self.lr 相同
        self.epochs = 10
        self.save_dir = save_dir
        self.eval_steps = 500  # 每500步验证一次
        self.log_steps = 50    # 每50步记录一次日志
        self.lr_warmup_ratio = 0.05
        self.lr_warmup_max_steps = 2000
        self.min_lr = 1e-7
        self.min_save_pearson = 0.4
        
        self.criterion = CrossEntropyLoss()
        self.speaker_criterion = MSELoss()
        self.metric_name = "Accuracy"
        self.tb_writer = None
        self.run_id = None
        self.initial_global_step = max(0, int(initial_global_step))

        if self.local_rank in [-1, 0]:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG)  # 设置最低日志级别
            
            # 创建文件处理器（FileHandler）
            os.makedirs(self.save_dir, exist_ok=True)
            self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(f'{self.save_dir}/train_{self.run_id}.log', mode='a', encoding='utf-8')
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
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "tb", self.run_id))
    
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
            collate_fn=AudioDataset_pairwise_audio.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=AudioDataset_pairwise_audio.collate_fn,
                num_workers=4,
                pin_memory=True
            )
            
        return train_loader, val_loader, train_sampler

    def create_lr_scheduler(self, optimizer, total_training_steps):
        total_training_steps = max(1, int(total_training_steps))
        warmup_steps = max(1, int(total_training_steps * self.lr_warmup_ratio))
        if self.lr_warmup_max_steps:
            warmup_steps = min(warmup_steps, int(self.lr_warmup_max_steps))
        warmup_steps = min(warmup_steps, total_training_steps)
        decay_steps = max(1, total_training_steps - warmup_steps)

        def make_lambda(base_lr):
            min_lr_ratio = min(1.0, max(0.0, self.min_lr / base_lr)) if base_lr > 0 else 0.0
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return max(min_lr_ratio, float(current_step + 1) / float(warmup_steps))
                progress = min(1.0, float(current_step - warmup_steps) / float(decay_steps))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
            return lr_lambda

        lr_lambdas = [make_lambda(group["lr"]) for group in optimizer.param_groups]

        last_epoch = -1
        if self.initial_global_step > 0:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
            last_epoch = self.initial_global_step - 1

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas, last_epoch=last_epoch)
        return scheduler, warmup_steps, total_training_steps
    
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
        self.model = DistributedDataParallel(self.model, broadcast_buffers=False,find_unused_parameters=True)
        print("create ddp model")
        train_loader, val_loader, train_sampler = self.create_dataloaders()
        effective_lora_lr = self.lora_lr if self.lora_lr is not None else self.lr
        lora_params, other_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(param)
            else:
                other_params.append(param)
        optimizer = AdamW([
            {"params": lora_params,  "lr": effective_lora_lr},
            {"params": other_params, "lr": self.lr},
        ])
        total_training_steps = len(train_loader) * self.epochs
        scheduler, warmup_steps, total_training_steps = self.create_lr_scheduler(
            optimizer, total_training_steps
        )
        
        best_val_pearson = float('-inf')
        latest_val_pearson = None
        os.makedirs(self.save_dir, exist_ok=True)
        if self.local_rank in [-1, 0]:
            print(self.model)
            # 记录初始学习率
            current_lr = optimizer.param_groups[0]['lr']
            self.logger.info(f"Initial learning rate: {current_lr}")
            self.logger.info(
                "LR schedule: linear warmup + cosine decay, "
                f"lora_lr={effective_lora_lr:.2e}, other_lr={self.lr:.2e}, min_lr={self.min_lr:.2e}, "
                f"warmup_steps={warmup_steps}, total_steps={total_training_steps}"
            )
            self.logger.info(f"Model save Pearson threshold: {self.min_save_pearson:.4f}")
            if self.initial_global_step > 0:
                self.logger.info(f"Resume global step from: {self.initial_global_step}")

        dist.barrier()
                    
        # 只在主进程进行验证
        if self.InitialValidation:
            metric_step = self.initial_global_step
            if val_loader and self.local_rank in [-1, 0]:
                val_avg_loss, val_accuracy, val_pearson, val_spearman = self.evaluate(val_loader)
                latest_val_pearson = val_pearson
                
                # 记录验证结果
                self.logger.info(
                    f"Step {metric_step}: Validation loss: {val_avg_loss:.4f} accuracy: {val_accuracy:.4f} "
                    f"pearson: {val_pearson:.4f} spearman: {val_spearman:.4f}"
                )
                if self.tb_writer:
                    self.tb_writer.add_scalar("val/loss", val_avg_loss, metric_step)
                    self.tb_writer.add_scalar("val/acc", val_accuracy, metric_step)
                    self.tb_writer.add_scalar("val/pearson", val_pearson, metric_step)
                    self.tb_writer.add_scalar("val/spearman", val_spearman, metric_step)
                if val_pearson > best_val_pearson:
                    best_val_pearson = val_pearson
                    if val_pearson >= self.min_save_pearson:
                        self.save_model(f"initial_model")
                        self.logger.info(f"Initial model saved to {self.save_dir}/initial_model")
                    else:
                        self.logger.info(
                            f"Initial model not saved: pearson {val_pearson:.4f} "
                            f"< threshold {self.min_save_pearson:.4f}"
                        )
                if self.tb_writer:
                    self.tb_writer.add_scalar("val/best_pearson", best_val_pearson, metric_step)
        
        dist.barrier()

        global_step = self.initial_global_step
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
                compute_grad_norm = (
                    self.local_rank in [-1, 0]
                    and ((global_step + 1) % self.log_steps == 0)
                )
                local_loss, grad_norm = self.train_step(
                    batch, optimizer, epoch, compute_grad_norm=compute_grad_norm
                )
                total_loss += local_loss
                epoch_loss += local_loss
                global_step += 1
                scheduler.step()

                # 记录日志
                if global_step % self.log_steps == 0 and self.local_rank in [-1, 0]:
                    avg_loss = total_loss / self.log_steps
                    current_lr = optimizer.param_groups[0]['lr']
                    if self.model.module.use_grl:
                        self.logger.info(f"Step {global_step}: Training loss: {avg_loss:.4f}, Learning rate: {current_lr:.2e}, grl_gm: {self.grl_gm}")
                    else:
                        self.logger.info(f"Step {global_step}: Training loss: {avg_loss:.4f}, Learning rate: {current_lr:.2e}")
                    if hasattr(progress_bar, 'set_postfix'):
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}", 
                            "step": f"{global_step}",
                            "lr": f"{current_lr:.2e}"
                        })
                    if self.tb_writer:
                        self.tb_writer.add_scalar("train/loss", avg_loss, global_step)
                        self.tb_writer.add_scalar("lr", current_lr, global_step)
                        if self.model.module.use_grl:
                            self.tb_writer.add_scalar("grl_gm", self.grl_gm, global_step)
                        if grad_norm is not None:
                            self.tb_writer.add_scalar("train/grad_norm", grad_norm, global_step)
                    total_loss = 0.0  # 重置

                # 验证阶段
                if global_step % self.eval_steps == 0:
                    # 所有进程同步
                    dist.barrier()
                    
                    # 只在主进程进行验证
                    if val_loader and self.local_rank in [-1, 0]:
                        val_avg_loss, val_accuracy, val_pearson, val_spearman = self.evaluate(val_loader)
                        latest_val_pearson = val_pearson
                        
                        # 记录验证结果
                        self.logger.info(
                            f"Step {global_step}: Validation loss: {val_avg_loss:.4f} accuracy: {val_accuracy:.4f} "
                            f"pearson: {val_pearson:.4f} spearman: {val_spearman:.4f}"
                        )
                        if self.tb_writer:
                            self.tb_writer.add_scalar("val/loss", val_avg_loss, global_step)
                            self.tb_writer.add_scalar("val/acc", val_accuracy, global_step)
                            self.tb_writer.add_scalar("val/pearson", val_pearson, global_step)
                            self.tb_writer.add_scalar("val/spearman", val_spearman, global_step)
                        
                        # 保存最佳模型
                        if val_pearson > best_val_pearson:
                            best_val_pearson = val_pearson
                            if val_pearson >= self.min_save_pearson:
                                self.save_model(f"best_model_step_{global_step}")
                                self.logger.info(f"Best model saved to {self.save_dir}/best_model_step_{global_step}")
                            else:
                                self.logger.info(
                                    f"Best model at step {global_step} not saved: pearson {val_pearson:.4f} "
                                    f"< threshold {self.min_save_pearson:.4f}"
                                )
                        if self.tb_writer:
                            self.tb_writer.add_scalar("val/best_pearson", best_val_pearson, global_step)

                    # 等待主进程完成验证和日志/保存后再继续训练
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
                if self.tb_writer:
                    self.tb_writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch + 1)
                    self.tb_writer.add_scalar("lr", current_lr, global_step)

            # 每个epoch结束时也进行验证
            dist.barrier()
            
            # 只在主进程进行验证
            if val_loader and self.local_rank in [-1, 0]:
                val_avg_loss, val_accuracy, val_pearson, val_spearman = self.evaluate(val_loader)
                latest_val_pearson = val_pearson
                self.logger.info(
                    f"Epoch {epoch+1} Validation loss: {val_avg_loss:.4f} accuracy: {val_accuracy:.4f} "
                    f"pearson: {val_pearson:.4f} spearman: {val_spearman:.4f}"
                )
                if self.tb_writer:
                    self.tb_writer.add_scalar("val/loss", val_avg_loss, global_step)
                    self.tb_writer.add_scalar("val/acc", val_accuracy, global_step)
                    self.tb_writer.add_scalar("val/pearson", val_pearson, global_step)
                    self.tb_writer.add_scalar("val/spearman", val_spearman, global_step)
                
                # 保存最佳模型（基于epoch）
                if val_pearson > best_val_pearson:
                    best_val_pearson = val_pearson
                    if val_pearson >= self.min_save_pearson:
                        self.save_model(f"best_model_epoch_{epoch+1}")
                        self.logger.info(f"Best model saved to {self.save_dir}/best_model_epoch_{epoch+1}")
                    else:
                        self.logger.info(
                            f"Best model at epoch {epoch+1} not saved: pearson {val_pearson:.4f} "
                            f"< threshold {self.min_save_pearson:.4f}"
                        )
                if self.tb_writer:
                    self.tb_writer.add_scalar("val/best_pearson", best_val_pearson, global_step)

            # 定期保存模型（只在主进程）
            if (epoch + 1) % 4 == 0 and self.local_rank in [-1, 0]:
                if latest_val_pearson is not None and latest_val_pearson >= self.min_save_pearson:
                    self.save_model(f"model_epoch_{epoch+1}")
                    self.logger.info(f"Model saved to {self.save_dir}/model_epoch_{epoch+1}")
                else:
                    pearson_text = "None" if latest_val_pearson is None else f"{latest_val_pearson:.4f}"
                    self.logger.info(
                        f"Epoch {epoch+1} checkpoint not saved: latest pearson {pearson_text} "
                        f"< threshold {self.min_save_pearson:.4f}"
                    )

            # 无条件保存最新 checkpoint，供崩溃后 resume（LoRA 只有 ~50MB，代价低）
            if self.local_rank in [-1, 0]:
                self.save_model("latest_model")
                pearson_text = "None" if latest_val_pearson is None else f"{latest_val_pearson:.4f}"
                self.logger.info(f"Epoch {epoch+1} latest_model saved (pearson={pearson_text})")

            # 等待主进程完成 epoch 末验证和保存后再进入下一轮训练
            dist.barrier()
        if self.tb_writer:
            self.tb_writer.close()
    
    def train_step(self, batch, optimizer, epoch, compute_grad_norm=False):
        self.model.train()
        optimizer.zero_grad()

        if self.model.module.use_grl:
            
            outputs_a, speaker_pred_0 = self.model(
                wavs = batch["feature_a"][0].to(self.device),
                return_grl = True,
                grl_alpha=1.0
            )
            outputs_b, speaker_pred_1 = self.model(
                wavs = batch["feature_b"][0].to(self.device),
                return_grl = True,
                grl_alpha=1.0
            )
            # print(batch["feature_a"][0].shape)
            with torch.no_grad():
                outputs_grl_0 = self.spk_encoder(batch["feature_a"][0], 24000)
                outputs_grl_1 = self.spk_encoder(batch["feature_b"][0], 24000)
            
            # print(speaker_pred_0.shape, speaker_pred_1.shape, outputs_grl_0.shape, outputs_grl_1.shape)

            main_loss = pairwise_loss(outputs_a, outputs_b, batch["labels"].to(self.device).float())
            speaker_loss = self.speaker_criterion(speaker_pred_0, outputs_grl_0) + \
                            self.speaker_criterion(speaker_pred_1, outputs_grl_1)
            loss = main_loss -  speaker_loss * self.grl_gm
        else:
            outputs_a = self.model(
                wavs = batch["feature_a"][0].to(self.device),
            )
            outputs_b = self.model(
                wavs = batch["feature_b"][0].to(self.device),
            )

            # loss = binary_rank_loss(outputs_a, outputs_b, batch["labels"].to(self.device).float())
            loss = pairwise_loss(outputs_a, outputs_b, batch["labels"].to(self.device).float())

        # print("batch", batch)
        # print("outputs_a,outputs_b,loss:", outputs_a,outputs_b, outputs_a-outputs_b,batch["labels"],loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0
        )
        grad_norm = None
        if compute_grad_norm:
            grad_norm = self._compute_grad_norm()
        total_loss = loss.item()
        optimizer.step()
        
        return total_loss, grad_norm

    def _compute_grad_norm(self):
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            param_norm = param.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def evaluate(self, val_loader):
        """验证函数"""
        if val_loader is None:
            return 0.0, 0.0, 0.0, 0.0

        if isinstance(val_loader.dataset, AudioDataset_pairwise_audio):
            return self.evaluate_grouped(val_loader.dataset)

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
                    wavs = batch["feature_a"][0].to(self.device),
                )
                outputs_b = self.model(
                    wavs = batch["feature_b"][0].to(self.device),
                )
                
                # 计算损失
                # loss = binary_rank_loss(outputs_a, outputs_b, batch["labels"].to(self.device).float())
                loss = pairwise_loss(outputs_a, outputs_b, batch["labels"].to(self.device).float())
                progress_bar.set_postfix(loss=loss.item())
                total_loss += loss.item()
                
                # 计算准确率
                # 预测：如果outputs_a > outputs_b，则预测为1，否则为0
                preds = (outputs_a > outputs_b).float()
                labels = batch["labels"].to(self.device).float()
                
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_samples += len(labels)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy, 0.0, 0.0

    def evaluate_grouped(self, dataset):
        """验证函数（全局 Pearson/Spearman，保留 pairwise acc/loss）"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_pairs = 0
        all_true = []
        all_pred = []

        with torch.no_grad():
            for group in tqdm(dataset.target, desc="Evaluating groups"):
                items = []
                for item in group:
                    audio_file = item["audio_file"]
                    if dataset.file_exist(audio_file):
                        items.append(item)

                if len(items) < 2:
                    continue

                preds = []
                trues = []
                for item in items:
                    audio_path = os.path.join(DATASET_ROOT_DIR, item["audio_file"])
                    wav = dataset.load_audio_with_cache(audio_path).to(self.device)
                    score = self.model(wavs=wav).detach().float().cpu().view(-1)[0]
                    preds.append(score)
                    trues.append(float(item["score"]))

                for score in preds:
                    all_pred.append(score.item())
                all_true.extend(trues)

                scores_tensor = torch.stack(preds).view(-1)
                n = scores_tensor.numel()
                if n < 2:
                    continue

                diff = scores_tensor[:, None] - scores_tensor[None, :]
                true_scores = torch.tensor(trues, dtype=diff.dtype)
                true_diff = true_scores[:, None] - true_scores[None, :]

                base_mask = ~torch.eye(n, dtype=torch.bool)
                valid_mask = base_mask & (true_diff != 0)
                if not valid_mask.any():
                    continue

                label = (true_diff > 0).to(diff.dtype)
                total_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                    diff[valid_mask], label[valid_mask], reduction="sum"
                ).item()

                preds_matrix = diff > 0
                correct = (preds_matrix == label.bool())[valid_mask].sum().item()
                total_correct += correct
                total_pairs += int(valid_mask.sum().item())

        avg_loss = total_loss / total_pairs if total_pairs > 0 else 0.0
        accuracy = total_correct / total_pairs if total_pairs > 0 else 0.0

        pearson = 0.0
        spearman = 0.0
        if len(all_true) > 1:
            if np.std(all_true) > 0 and np.std(all_pred) > 0:
                pearson = float(pearsonr(all_true, all_pred)[0])
                spearman = float(spearmanr(all_true, all_pred)[0])

        return avg_loss, accuracy, pearson, spearman
    
    def save_model(self, filename):
        """保存模型和处理器（只在主进程调用）"""
        # 如果是DDP模型，获取原始模型
        model_to_save = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        os.makedirs(self.save_dir, exist_ok=True)
        model_to_save.save_model(os.path.join(self.save_dir, filename))
