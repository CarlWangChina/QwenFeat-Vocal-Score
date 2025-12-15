import os
import audioscore.hubert.inference as hubert
import audioscore.whisper_svc.inference as whisper_svc
import audioscore.pitch.inference as pitch

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(CURRENT_PATH))

class FeatExtractor():
    def __init__(self, device):
        self.whisper = whisper_svc.WhisperInference(os.path.join(ROOT_PATH, "pretrain", "whisper_pretrain", "large-v2.pt"), device)
        self.hubert = hubert.HubertInference(os.path.join(ROOT_PATH, "pretrain", "hubert_pretrain", "hubert-soft-0d54a1f4.pt"),device)
    def process_audio(self, audio):
        return self.whisper.inference(audio), self.hubert.inference(audio), pitch.compute_f0_sing_audio(audio, 16000)