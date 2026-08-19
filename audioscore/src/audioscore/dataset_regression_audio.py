import json
import os

import torch
from torch.utils.data import Dataset

from audioscore.dataset import AudioDataset_pairwise_audio, ROOT_DIR


class AudioDataset_regression_audio(Dataset):
    def __init__(
        self,
        target_file,
        cache_dir=None,
        cache_mode="none",
        cache_base_dir=None,
        cache_target_sr=24000,
        cache_segment_duration=60,
        cache_kind="full",
        cache_max_duration=None,
        cache_cut_mode="random",
    ):
        with open(target_file, "r", encoding="utf-8") as f:
            target = json.load(f)

        if target and isinstance(target[0], list):
            target = [item for group in target for item in group]

        self.items = []
        self.file_exist_cache = {}
        self.cache_dir = cache_dir
        self.cache_mode = cache_mode
        self.cache_base_dir = cache_base_dir or (os.path.dirname(cache_dir) if cache_dir else None)
        self.cache_target_sr = cache_target_sr
        self.cache_segment_duration = cache_segment_duration
        self.cache_kind = cache_kind
        self.cache_max_duration = cache_max_duration
        self.cache_cut_mode = cache_cut_mode

        for item in target:
            audio_file = item["audio_file"]
            if self.file_exist(audio_file):
                self.items.append(item)

        print("load regression items:", len(self.items))

    def __len__(self):
        return len(self.items)

    def resolve_audio_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(ROOT_DIR, path)

    def file_exist(self, path):
        if path in self.file_exist_cache:
            return self.file_exist_cache[path]
        audio_path = self.resolve_audio_path(path)
        res = os.path.exists(audio_path)
        cache_mode = (self.cache_mode or "none").lower()
        if not res and cache_mode in {"read", "read_write"}:
            cache_path = AudioDataset_pairwise_audio.build_cache_path(
                audio_path,
                self.cache_dir,
                self.cache_base_dir,
            )
            res = bool(cache_path and os.path.exists(cache_path))
        self.file_exist_cache[path] = res
        return res

    def load_audio_with_cache(self, audio_path):
        cache_path = AudioDataset_pairwise_audio.build_cache_path(
            audio_path,
            self.cache_dir,
            self.cache_base_dir,
        )
        cache_mode = (self.cache_mode or "none").lower()
        if cache_mode not in {"none", "read", "write", "read_write"}:
            cache_mode = "none"

        cache_kind = (self.cache_kind or "full").lower()
        if cache_kind not in {"segment", "full"}:
            cache_kind = "full"

        if cache_mode == "none" or not cache_path:
            if cache_kind == "full":
                audio_tensor = AudioDataset_pairwise_audio.load_m4a_full_as_tensor(
                    audio_path,
                    target_sr=self.cache_target_sr,
                    max_duration=self.cache_max_duration,
                )[0]
                return self.cut_cached_full_audio(audio_tensor, audio_path)
            return AudioDataset_pairwise_audio.load_m4a_as_tensor(
                audio_path,
                target_sr=self.cache_target_sr,
                segment_duration=self.cache_segment_duration,
            )[0]

        if cache_mode != "write":
            cached = AudioDataset_pairwise_audio.load_cached_tensor(cache_path)
            if cached is not None:
                if cache_kind == "full":
                    return self.cut_cached_full_audio(cached, audio_path)
                return cached
            if cache_mode == "read":
                raise FileNotFoundError(f"Missing cached audio: {cache_path}")

        if cache_kind == "full":
            audio_tensor = AudioDataset_pairwise_audio.load_m4a_full_as_tensor(
                audio_path,
                target_sr=self.cache_target_sr,
                max_duration=self.cache_max_duration,
            )[0]
        else:
            audio_tensor = AudioDataset_pairwise_audio.load_m4a_as_tensor(
                audio_path,
                target_sr=self.cache_target_sr,
                segment_duration=self.cache_segment_duration,
            )[0]

        if cache_mode in {"write", "read_write"}:
            AudioDataset_pairwise_audio.save_cached_tensor(
                cache_path,
                audio_tensor,
                self.cache_target_sr,
            )

        if cache_kind == "full":
            return self.cut_cached_full_audio(audio_tensor, audio_path)
        return audio_tensor

    def cut_cached_full_audio(self, audio_tensor, audio_path):
        mode = (self.cache_cut_mode or "random").lower()
        if mode == "center":
            return AudioDataset_pairwise_audio.center_cut_tensor(
                audio_tensor,
                self.cache_target_sr,
                self.cache_segment_duration,
            )
        if mode == "stable":
            return AudioDataset_pairwise_audio.stable_cut_tensor(
                audio_tensor,
                self.cache_target_sr,
                self.cache_segment_duration,
                audio_path,
            )
        return AudioDataset_pairwise_audio.random_cut_tensor(
            audio_tensor,
            self.cache_target_sr,
            self.cache_segment_duration,
        )

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_path = self.resolve_audio_path(item["audio_file"])
        feature = self.load_audio_with_cache(audio_path)
        score = float(item["score"])
        return feature, score, item["audio_file"]

    @staticmethod
    def collate_fn(batch):
        audio_tensors = []
        for item in batch:
            audio = item[0]
            if audio.dim() == 2 and audio.size(0) == 1:
                audio = audio.squeeze(0)
            audio_tensors.append(audio)

        max_len = max(audio.size(-1) for audio in audio_tensors)
        features = torch.zeros(
            len(audio_tensors),
            max_len,
            dtype=audio_tensors[0].dtype,
        )
        for idx, audio in enumerate(audio_tensors):
            features[idx, : audio.size(-1)] = audio

        return {
            "features": features,
            "scores": torch.tensor([item[1] for item in batch], dtype=torch.float32),
            "audio_files": [item[2] for item in batch],
        }
