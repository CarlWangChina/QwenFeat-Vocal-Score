import datetime
import logging
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from scipy.stats import pearsonr, spearmanr
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from audioscore.dataset_regression_audio import AudioDataset_regression_audio


class Trainer:
    def __init__(
        self,
        model,
        train_json,
        val_json=None,
        device=None,
        local_rank=-1,
        world_size=-1,
        save_dir="checkpoints",
        train_cache_dir=None,
        train_cache_mode="none",
        train_cache_base_dir=None,
        train_cache_target_sr=24000,
        train_cache_segment_duration=60,
        train_cache_kind="full",
        train_cache_max_duration=None,
        train_cache_cut_mode="random",
        val_cache_dir=None,
        val_cache_mode="none",
        val_cache_base_dir=None,
        val_cache_target_sr=24000,
        val_cache_segment_duration=60,
        val_cache_kind="full",
        val_cache_max_duration=None,
        val_cache_cut_mode="stable",
        initial_global_step=0,
    ):
        self.model = model
        self.train_json = train_json
        self.val_json = val_json
        self.local_rank = local_rank
        self.device = device
        self.world_size = world_size
        self.save_dir = save_dir
        self.initial_global_step = max(0, int(initial_global_step))
        self.distributed = dist.is_available() and dist.is_initialized() and self.local_rank != -1
        self.train_cache_config = {
            "cache_dir": train_cache_dir,
            "cache_mode": train_cache_mode,
            "cache_base_dir": train_cache_base_dir,
            "cache_target_sr": train_cache_target_sr,
            "cache_segment_duration": train_cache_segment_duration,
            "cache_kind": train_cache_kind,
            "cache_max_duration": train_cache_max_duration,
            "cache_cut_mode": train_cache_cut_mode,
        }
        self.val_cache_config = {
            "cache_dir": val_cache_dir,
            "cache_mode": val_cache_mode,
            "cache_base_dir": val_cache_base_dir,
            "cache_target_sr": val_cache_target_sr,
            "cache_segment_duration": val_cache_segment_duration,
            "cache_kind": val_cache_kind,
            "cache_max_duration": val_cache_max_duration,
            "cache_cut_mode": val_cache_cut_mode,
        }

        self.train_dataset = AudioDataset_regression_audio(
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
        self.val_dataset = AudioDataset_regression_audio(
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

        self.batch_size = 1
        self.lr = 1e-5
        self.head_lr = 1e-5
        self.lora_lr = 5e-5
        self.ln_lr = 1e-5
        self.epochs = 10
        self.eval_steps = 200
        self.log_steps = 50
        self.lr_warmup_ratio = 0.05
        self.lr_warmup_max_steps = 2000
        self.min_lr = 1e-7
        self.score_scale = 100.0
        self.huber_beta = 0.05
        self.min_save_pearson = 0.4
        self.initial_validation = True
        self.seed = None

        self.tb_writer = None
        self.logger = None
        self.run_id = None

        if self.is_main_process:
            os.makedirs(self.save_dir, exist_ok=True)
            self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            logger = logging.getLogger(f"{__name__}.{self.run_id}")
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            file_handler = logging.FileHandler(
                os.path.join(self.save_dir, f"train_{self.run_id}.log"),
                mode="a",
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)
            self.logger = logger
            self.logger.info("start regression training")
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "tb", self.run_id))

    @property
    def is_main_process(self):
        return self.local_rank in [-1, 0]

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def create_dataloaders(self):
        train_sampler = None
        if self.distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                shuffle=True,
                rank=self.local_rank,
                num_replicas=self.world_size,
                seed=42,
            )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            collate_fn=AudioDataset_regression_audio.collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=AudioDataset_regression_audio.collate_fn,
                num_workers=4,
                pin_memory=True,
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

    def build_optimizer(self):
        head_params = []
        lora_params = []
        ln_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            normalized_name = name[7:] if name.startswith("module.") else name
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(param)
            elif normalized_name.startswith("head."):
                head_params.append(param)
            elif normalized_name.startswith("muq.") and "norm" in normalized_name.lower():
                ln_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": self.head_lr, "name": "head"})
        if lora_params:
            param_groups.append({"params": lora_params, "lr": self.lora_lr, "name": "lora"})
        if ln_params:
            param_groups.append({"params": ln_params, "lr": self.ln_lr, "name": "muq_ln"})
        if other_params:
            param_groups.append({"params": other_params, "lr": self.lr, "name": "other"})
        return AdamW(param_groups)

    def train(self):
        print("start regression training")
        self.model.to(self.device)
        if self.distributed:
            print(f"create ddp regression model rank={self.local_rank}, {self.device}")
            self.model = DistributedDataParallel(
                self.model,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        train_loader, val_loader, train_sampler = self.create_dataloaders()
        optimizer = self.build_optimizer()
        total_training_steps = len(train_loader) * self.epochs
        scheduler, warmup_steps, total_training_steps = self.create_lr_scheduler(
            optimizer,
            total_training_steps,
        )

        best_val_mae = float("inf")
        best_val_pearson = float("-inf")
        latest_val_pearson = None
        global_step = self.initial_global_step

        if self.is_main_process:
            self.logger.info(
                "LR schedule: linear warmup + cosine decay, "
                f"head_lr={self.head_lr:.2e}, lora_lr={self.lora_lr:.2e}, "
                f"ln_lr={self.ln_lr:.2e}, other_lr={self.lr:.2e}, min_lr={self.min_lr:.2e}, "
                f"warmup_steps={warmup_steps}, total_steps={total_training_steps}"
            )
            self.logger.info(
                f"loss=SmoothL1Loss(beta={self.huber_beta}) on normalized score, "
                f"score_scale=0-{self.score_scale:g}, "
                f"save_pearson_threshold={self.min_save_pearson:.4f}, "
                "best_save_metric=mae_after_min_pearson"
            )
            if self.initial_global_step > 0:
                self.logger.info(f"Resume global step from: {self.initial_global_step}")

        self.barrier()

        if self.initial_validation:
            if val_loader and self.is_main_process:
                metrics = self.evaluate(val_loader)
                latest_val_pearson = metrics["pearson"]
                self.log_eval("Step", global_step, metrics)
                best_val_pearson = max(best_val_pearson, metrics["pearson"])
                best_val_mae = self.maybe_save_best(
                    best_val_mae,
                    metrics,
                    "initial_model",
                    "Initial model",
                )
                self.write_eval_tb(metrics, global_step)
            self.barrier()

        for epoch in range(self.epochs):
            if train_sampler:
                train_sampler.set_epoch(epoch)

            self.model.train()
            total_loss = 0.0
            epoch_loss = 0.0
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs} [Regression]",
            ) if self.is_main_process else train_loader

            for batch in progress_bar:
                compute_grad_norm = self.is_main_process and ((global_step + 1) % self.log_steps == 0)
                loss, grad_norm = self.train_step(batch, optimizer, compute_grad_norm)
                total_loss += loss
                epoch_loss += loss
                global_step += 1
                scheduler.step()

                if global_step % self.log_steps == 0 and self.is_main_process:
                    avg_loss = total_loss / self.log_steps
                    current_lr = optimizer.param_groups[0]["lr"]
                    self.logger.info(
                        f"Step {global_step}: train_loss_norm={avg_loss:.6f}, lr={current_lr:.2e}"
                    )
                    if hasattr(progress_bar, "set_postfix"):
                        progress_bar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "step": f"{global_step}",
                                "lr": f"{current_lr:.2e}",
                            }
                        )
                    if self.tb_writer:
                        self.tb_writer.add_scalar("train/loss", avg_loss, global_step)
                        self.tb_writer.add_scalar("lr", current_lr, global_step)
                        if grad_norm is not None:
                            self.tb_writer.add_scalar("train/grad_norm", grad_norm, global_step)
                    total_loss = 0.0

                if global_step % self.eval_steps == 0:
                    self.barrier()
                    if val_loader and self.is_main_process:
                        metrics = self.evaluate(val_loader)
                        latest_val_pearson = metrics["pearson"]
                        self.log_eval("Step", global_step, metrics)
                        best_val_pearson = max(best_val_pearson, metrics["pearson"])
                        best_val_mae = self.maybe_save_best(
                            best_val_mae,
                            metrics,
                            f"best_model_step_{global_step}",
                            f"Best model at step {global_step}",
                        )
                        self.write_eval_tb(metrics, global_step)
                        if self.tb_writer:
                            self.tb_writer.add_scalar("val/best_pearson", best_val_pearson, global_step)
                            self.tb_writer.add_scalar("val/best_mae", best_val_mae, global_step)
                    self.barrier()

            if self.distributed:
                epoch_loss_tensor = torch.tensor(epoch_loss, device=self.device)
                dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
                epoch_loss = epoch_loss_tensor.item() / dist.get_world_size()

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1} avg_train_loss:{avg_epoch_loss:.4f} rank={self.local_rank}")

            if self.is_main_process:
                current_lr = optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"Epoch {epoch + 1} Average training loss: {avg_epoch_loss:.4f}, lr={current_lr:.2e}"
                )
                if self.tb_writer:
                    self.tb_writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch + 1)
                    self.tb_writer.add_scalar("lr", current_lr, global_step)

            self.barrier()
            if val_loader and self.is_main_process:
                metrics = self.evaluate(val_loader)
                latest_val_pearson = metrics["pearson"]
                self.log_eval("Epoch", epoch + 1, metrics)
                best_val_pearson = max(best_val_pearson, metrics["pearson"])
                best_val_mae = self.maybe_save_best(
                    best_val_mae,
                    metrics,
                    f"best_model_epoch_{epoch + 1}",
                    f"Best model at epoch {epoch + 1}",
                )
                self.write_eval_tb(metrics, global_step)
                if self.tb_writer:
                    self.tb_writer.add_scalar("val/best_pearson", best_val_pearson, global_step)
                    self.tb_writer.add_scalar("val/best_mae", best_val_mae, global_step)

            if self.is_main_process:
                self.save_model("latest_model")
                pearson_text = "None" if latest_val_pearson is None else f"{latest_val_pearson:.4f}"
                self.logger.info(f"Epoch {epoch + 1} latest_model saved (pearson={pearson_text})")

            self.barrier()

        if self.tb_writer:
            self.tb_writer.close()

    def train_step(self, batch, optimizer, compute_grad_norm=False):
        self.model.train()
        optimizer.zero_grad()
        wav = batch["features"].to(self.device)
        target = batch["scores"].to(self.device)
        pred = self.model(wavs=wav).view(-1)
        pred_norm = pred / self.score_scale
        target_norm = target / self.score_scale
        loss = torch.nn.functional.smooth_l1_loss(pred_norm, target_norm, beta=self.huber_beta)
        loss.backward()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        grad_norm = float(grad_norm_tensor.detach().cpu().item()) if compute_grad_norm else None
        optimizer.step()
        return loss.item(), grad_norm

    def compute_grad_norm(self):
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            param_norm = param.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def evaluate(self, val_loader):
        self.model.eval()
        losses = []
        preds = []
        trues = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating regression"):
                wav = batch["features"].to(self.device)
                target = batch["scores"].to(self.device)
                pred = self.model(wavs=wav).view(-1)
                pred_norm = pred / self.score_scale
                target_norm = target / self.score_scale
                loss = torch.nn.functional.smooth_l1_loss(pred_norm, target_norm, beta=self.huber_beta)
                losses.append(loss.item())
                preds.extend(pred.detach().float().cpu().tolist())
                trues.extend(target.detach().float().cpu().tolist())

        pred_np = np.array(preds, dtype=np.float32)
        true_np = np.array(trues, dtype=np.float32)
        mae = float(np.mean(np.abs(pred_np - true_np))) if len(pred_np) else 0.0
        rmse = float(np.sqrt(np.mean((pred_np - true_np) ** 2))) if len(pred_np) else 0.0
        pearson = 0.0
        spearman = 0.0
        if len(true_np) > 1 and np.std(true_np) > 0 and np.std(pred_np) > 0:
            pearson = float(pearsonr(true_np, pred_np)[0])
            spearman = float(spearmanr(true_np, pred_np)[0])

        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "mae": mae,
            "rmse": rmse,
            "pearson": pearson,
            "spearman": spearman,
        }

    def log_eval(self, prefix, step, metrics):
        self.logger.info(
            f"{prefix} {step}: val_loss={metrics['loss']:.4f} "
            f"mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} "
            f"pearson={metrics['pearson']:.4f} spearman={metrics['spearman']:.4f}"
        )

    def write_eval_tb(self, metrics, step):
        if not self.tb_writer:
            return
        self.tb_writer.add_scalar("val/loss", metrics["loss"], step)
        self.tb_writer.add_scalar("val/mae", metrics["mae"], step)
        self.tb_writer.add_scalar("val/rmse", metrics["rmse"], step)
        self.tb_writer.add_scalar("val/pearson", metrics["pearson"], step)
        self.tb_writer.add_scalar("val/spearman", metrics["spearman"], step)

    def maybe_save_best(self, best_mae, metrics, filename, label):
        pearson = metrics["pearson"]
        mae = metrics["mae"]
        if pearson >= self.min_save_pearson:
            if mae < best_mae:
                best_mae = mae
                self.save_model(filename)
                self.logger.info(
                    f"{label} saved to {self.save_dir}/{filename} "
                    f"(mae={mae:.4f}, pearson={pearson:.4f})"
                )
            else:
                self.logger.info(
                    f"{label} not saved: mae {mae:.4f} >= best_mae {best_mae:.4f} "
                    f"(pearson={pearson:.4f})"
                )
        else:
            self.logger.info(
                f"{label} not saved: pearson {pearson:.4f} < threshold {self.min_save_pearson:.4f}"
            )
        return best_mae

    def build_checkpoint_config(self):
        return {
            "train_json": self.train_json,
            "val_json": self.val_json,
            "save_dir": self.save_dir,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "eval_steps": self.eval_steps,
            "log_steps": self.log_steps,
            "lr": self.lr,
            "head_lr": self.head_lr,
            "lora_lr": self.lora_lr,
            "ln_lr": self.ln_lr,
            "min_lr": self.min_lr,
            "lr_warmup_ratio": self.lr_warmup_ratio,
            "lr_warmup_max_steps": self.lr_warmup_max_steps,
            "score_scale": self.score_scale,
            "huber_beta": self.huber_beta,
            "min_save_pearson": self.min_save_pearson,
            "best_save_metric": "mae_after_min_pearson",
            "initial_validation": self.initial_validation,
            "initial_global_step": self.initial_global_step,
            "seed": self.seed,
            "device": str(self.device),
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "distributed": self.distributed,
            "train_cache": self.train_cache_config,
            "val_cache": self.val_cache_config,
        }

    def save_model(self, filename):
        model_to_save = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        os.makedirs(self.save_dir, exist_ok=True)
        model_to_save.save_model(
            os.path.join(self.save_dir, filename),
            training_config=self.build_checkpoint_config(),
        )
