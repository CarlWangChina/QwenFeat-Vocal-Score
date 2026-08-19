import json
import os

import torch
from muq import MuQ

from audioscore.model import (
    get_lora_state_dict,
    inject_lora_into_muq,
    load_lora_state_dict,
)


class MuQRegressionHead(torch.nn.Module):
    def __init__(
        self,
        in_features=1024,
        ffd_hidden_size=2048,
        attn_layer_num=2,
        dropout=0.2,
    ):
        super().__init__()
        self.config = {
            "in_features": in_features,
            "ffd_hidden_size": ffd_hidden_size,
            "attn_layer_num": attn_layer_num,
            "dropout": dropout,
        }
        self.ffd = torch.nn.Sequential(
            torch.nn.Linear(in_features, ffd_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffd_hidden_size, in_features),
        )
        self.attn = torch.nn.ModuleList(
            [
                torch.nn.MultiheadAttention(
                    embed_dim=in_features,
                    num_heads=8,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(attn_layer_num)
            ]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features * 2, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 1),
        )

    def forward(self, audio_feats_vec):
        x = self.ffd(audio_feats_vec)
        attn_x = x
        for attn in self.attn:
            attn_x, _ = attn(attn_x, attn_x, attn_x, need_weights=False)

        pooled = torch.cat(
            [
                torch.mean(attn_x, dim=1),
                torch.max(attn_x, dim=1)[0],
            ],
            dim=1,
        )
        raw_score = self.output_layer(self.dropout(pooled)).squeeze(-1)
        return torch.sigmoid(raw_score) * 100.0


class MuQRegressionAudioLora(torch.nn.Module):
    def __init__(
        self,
        lora_r=128,
        lora_alpha=48,
        hidden_state_index=6,
    ):
        super().__init__()
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.hidden_state_index = hidden_state_index
        self.muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")

        for name, param in self.muq.named_parameters():
            param.requires_grad = "norm" in name.lower()

        self.muq = inject_lora_into_muq(
            self.muq,
            r=lora_r,
            alpha=lora_alpha,
        )
        self.head = MuQRegressionHead()

    def forward(self, wavs):
        output = self.muq(wavs, output_hidden_states=True)
        audio_embeds = output["hidden_states"][self.hidden_state_index]
        return self.head(audio_embeds)

    def save_model(self, path, training_config=None):
        os.makedirs(path, exist_ok=True)
        torch.save(self.head.state_dict(), os.path.join(path, "regression_head.pt"))

        save_dict = {}
        for k, v in get_lora_state_dict(self.muq).items():
            save_dict[f"lora.{k}"] = v
        for name, param in self.muq.named_parameters():
            if "norm" in name.lower() and param.requires_grad:
                save_dict[f"ln.{name}"] = param.data.cpu()
        torch.save(save_dict, os.path.join(path, "muq_lora.pt"))

        with open(os.path.join(path, "type.txt"), "w", encoding="utf-8") as f:
            f.write("MuQRegressionAudioLora")
        config = {
            "type": "MuQRegressionAudioLora",
            "score_scale": "0-100",
            "hidden_state_index": self.hidden_state_index,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "head": self.head.config,
        }
        if training_config is not None:
            config["training"] = training_config
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"saved regression model to {path}")

    def load_model(self, path):
        type_path = os.path.join(path, "type.txt")
        if not os.path.exists(type_path):
            raise FileNotFoundError("Missing " + type_path)

        lora_path = os.path.join(path, "muq_lora.pt")
        if os.path.exists(lora_path):
            loaded = torch.load(lora_path, map_location="cpu")
            lora_sd = {k[5:]: v for k, v in loaded.items() if k.startswith("lora.")}
            ln_sd = {k[3:]: v for k, v in loaded.items() if k.startswith("ln.")}
            if lora_sd:
                load_lora_state_dict(self.muq, lora_sd)
            if ln_sd:
                for name, param in self.muq.named_parameters():
                    if name in ln_sd:
                        param.data.copy_(ln_sd[name])
            print(f"loaded MuQ LoRA + LN from {lora_path}")

        head_path = os.path.join(path, "regression_head.pt")
        if os.path.exists(head_path):
            self.head.load_state_dict(torch.load(head_path, map_location="cpu"))
            print(f"loaded regression head from {head_path}")
        else:
            raise FileNotFoundError(f"Missing {head_path}")

    def trainable_parameter_names(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name
