import torch
import os
import sys
import csv
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class Mulan_Audio2Text(torch.nn.Module):
    def __init__(self):
        super(Mulan_Audio2Text, self).__init__()
        self.mlp_1 = torch.nn.Linear(512, 512)
        self.mlp_2 = torch.nn.Linear(512, 512)
    
    def forward(self, x):
        x = self.mlp_1(x)
        x = torch.relu(x)
        x = self.mlp_2(x)
        return x

class MulanDataset(Dataset):
    def __init__(self, dir_path):
        self.data = []
        self.load_data(dir_path)
    
    def load_data(self, dir_path):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".pkl"):
                    data = torch.load(os.path.join(root, file), map_location=torch.device('cpu'), weights_only=True)
                    text_emb = data["text_emb"]
                    user_emb = data["user"]
                    num_seg = len(data["user_seg"])
                    for i in range(num_seg):
                        # 每个segment作为一个样本
                        self.data.append({
                            'user_emb': user_emb,
                            'user_emb_seg': data["user_seg"][i],
                            'text_emb': text_emb
                        })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'user_emb': torch.tensor(sample['user_emb'], dtype=torch.float32),
            'user_emb_seg': torch.tensor(sample['user_emb_seg'], dtype=torch.float32),
            'text_emb': torch.tensor(sample['text_emb'], dtype=torch.float32)
        }

def train_mulan_dist(dir_path="data/processed_mulan", batch_size=32, epochs=10000, lr=0.001):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集和数据加载器
    dataset = MulanDataset(dir_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型、损失函数和优化器
    model = Mulan_Audio2Text().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch in dataloader:
            # 获取当前批次的数据
            user_emb = batch['user_emb'].to(device)
            user_emb_seg = batch['user_emb_seg'].to(device)
            text_emb = batch['text_emb'].to(device)
            
            # 前向传播
            # 使用user_emb_seg作为输入，因为它是分段的
            outputs = model(user_emb_seg)
            
            # 计算损失
            loss = criterion(outputs, text_emb)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计损失
            running_loss += loss.item()
        
        # 打印每个epoch的统计信息
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}')
    
    # 训练完成后保存模型
    torch.save(model.state_dict(), 'ckpts/mulan_text/mulan_audio2text_model.pth')
    print('Training complete. Model saved.')

if __name__ == "__main__":
    train_mulan_dist()