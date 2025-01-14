import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

# 生成器
class Generator(nn.Module):
    def __init__(self, input_size=100, hidden_size=128, output_size=28*28):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(True),
            nn.Linear(hidden_size * 4, output_size),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
    
    def forward(self, x):
        return self.main(x)

# 判别器
class Discriminator(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=32):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)

# 显示生成的图像并保存
def save_image(img, epoch, output_dir='./generated_images'):
    os.makedirs(output_dir, exist_ok=True)
    img = img / 2 + 0.5  # 逆归一化，将像素值范围从 [-1, 1] 转换到 [0, 1]
    npimg = img.numpy()
    plt.imshow(npimg, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(f"{output_dir}/epoch_{epoch}.png")  # 保存图像
    plt.close()

# 训练过程
if __name__ == '__main__':
    G = Generator(input_size=100, hidden_size=128, output_size=28*28)
    D = Discriminator(input_size=28*28, hidden_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    G.to(device)
    D.to(device)
    
    # 数据预处理
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    mnist = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=32, shuffle=True)
    
    criterion = nn.BCELoss()
    learning_rate = 0.0002

    # 优化器
    optimizer_g = optim.Adam(G.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(D.parameters(), lr=learning_rate)
    num_epochs = 50

    # 保存训练损失
    loss_history = []
# 每隔一定训练周期进行图像显示
sample_interval = 5  # 每5个周期显示一次生成的图像

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        # 准备真实数据和生成器输入
        images = images.view(images.size(0), -1).to(device)  # 展平图像
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)
        
        # 训练判别器
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # 生成假图像
        z = torch.randn(images.size(0), 100).to(device)
        fake_images = G(z)
        outputs = D(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # 总判别器损失并反向传播
        d_loss = d_loss_real + d_loss_fake
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()
        
        # 训练生成器
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)
        
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    # 每隔一定周期打印损失并保存生成的图像
    if (epoch + 1) % sample_interval == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'd_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, '
              f'D(x): {real_score.mean().item():.4f}, D(G(z)): {fake_score.mean().item():.4f}')
        
        # 保存生成的图像
        
        with torch.no_grad():
            z = torch.randn(16, 100).to(device)  # 使用单个随机噪声生成一张图像
        
            fake_images = G(z)  # 不进行 reshape，直接得到假图像
            fake_images = fake_images.view(-1, 28, 28)  # 重塑为 (batch_size, 28, 28)
              

        # 生成并显示16张图像
        plt.figure(figsize=(8,8))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            fake_image = fake_images[i].cpu().detach()  # 获取图像并移除梯度
            plt.imshow(fake_image.numpy(), cmap='gray')
            plt.axis('off')  # 不显示坐标轴
        plt.tight_layout()
        plt.savefig(f'./generated_images/epoch_{epoch+1}_grid.png')
        plt.close()

    # 保存每个epoch的损失数据
    loss_history.append({
        'epoch': epoch + 1,
        'd_loss': d_loss.item(),
        'g_loss': g_loss.item(),
        'D(x)': real_score.mean().item(),
        'D(G(z))': fake_score.mean().item()
    })

# 将损失数据保存为CSV文件
df = pd.DataFrame(loss_history)
df.to_csv('./training_loss.csv', index=False)
print("训练损失已保存至 'training_loss.csv'.")
