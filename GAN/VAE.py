import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision import transforms

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon

# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# 损失函数
def loss_function(x_recon, x, mu, logvar):
    x = x.view(-1, 784)  # 将目标也展平为 [batch_size, 784]
    x = torch.clamp(x, 0, 1)  # 确保目标值在[0, 1]之间
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence

# 训练 VAE
def train_vae():
    batch_size = 128
    num_epochs = 50  # 增加训练周期
    lr = 1e-4  # 减小学习率
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    mnist = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs('generated_images', exist_ok=True)  # 创建保存图像的目录
    loss_history = []
    os.makedirs('VAE-generated_images', exist_ok=True) 
    for epoch in range(num_epochs):
        total_loss = 0
        for batch, _ in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch)
            loss = loss_function(x_recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss = total_loss/len(dataloader.dataset) 
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader.dataset):.4f}")
        loss_history.append({
            'epoch':epoch+1,
            'Loss': loss
        })
        # 每个epoch保存生成图像
        model.eval()
        with torch.no_grad():
            z = torch.randn(16, 20).to(device)  # 随机采样潜在变量
            generated_images = model.decoder(z).cpu().view(-1, 1, 28, 28)  # 生成图像

        # 保存图像
        plt.figure(figsize=(8, 8))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated_images[i].squeeze(), cmap='gray')
            plt.axis('off')
        plt.savefig(f'VAE-generated_images/epoch_{epoch+1}.png')  # 保存为图像文件
        plt.close()
    df = pd.DataFrame(loss_history)
    df.to_csv('./VAE_training-loss.csv',index=False)
    
# 执行训练
train_vae()
