import torch, librosa
from muq import MuQMuLan

# This will automatically fetch checkpoints from huggingface
device = 'cuda'
mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
mulan = mulan.to(device).eval()

# Extract music embeddings
wav, sr = librosa.load("/home/xiaoyu/audioscore/audioscore/data/sort/data/audio/203887_250822105518005501.m4a", sr = 24000)
wavs = torch.tensor(wav).unsqueeze(0).to(device) 
print(wavs.shape)
with torch.no_grad():
    audio_embeds = mulan(wavs = wavs) 

# Extract text embeddings (texts can be in English or Chinese)
texts = ["classical genres, hopeful mood, piano.", "一首适合海边风景的小提琴曲，节奏欢快"]
with torch.no_grad():
    text_embeds = mulan(texts = texts)

# Calculate dot product similarity
sim = mulan.calc_similarity(audio_embeds, text_embeds)
print(sim)
print(audio_embeds.shape, text_embeds.shape, sim.shape)