import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from thymoma import ThymomaDataset, get_unetplusplus
from simsiam import SimSiam
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm  # 导入tqdm库
import csv
def save_model(model, filename):
    torch.save(model.encoder.state_dict(),  filename)

def load_model(model, filename):
    state_dict = torch.load(filename, map_location=device)
    model.encoder.load_state_dict(state_dict, strict=False)  # 使用 strict=False 来忽略不匹配的层
    return model
    
def save_loss_to_csv(loss, epoch, model_name, filename='loss.csv'):
    # 检查文件是否已经存在，如果不存在则写入标题
    file_exists = os.path.exists(filename)
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'Model', 'Train Loss'])  # 写入标题
        
        writer.writerow([epoch, model_name, loss])  # 写入当前周期的损失值和模型名称
# 数据增强转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 定义TwoCropsTransform
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

# Unlabeled dataset class
class UnlabeledDataset(ThymomaDataset):
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')
        if self.transform:
            images = self.transform(image)
        else:
            images = [transforms.ToTensor()(image), transforms.ToTensor()(image)]
        return images[0], images[1]  # 返回两个增广版本

# 创建数据集和DataLoader
data_dir_thymoma = r'/home/xn/DL大作业/D/data/thymoma'
unlabeled_data_dir = r'/home/xn/DL大作业/D/data/subset_2'

dataset_thymoma = ThymomaDataset(data_dir_thymoma, transform=transform)
train_size = int(0.8 * len(dataset_thymoma))
test_size = len(dataset_thymoma) - train_size
train_dataset_thymoma, test_dataset_thymoma = torch.utils.data.random_split(dataset_thymoma, [train_size, test_size])
train_loader_thymoma = DataLoader(train_dataset_thymoma, batch_size=4, shuffle=True)
test_loader_thymoma = DataLoader(test_dataset_thymoma, batch_size=4, shuffle=False)

unlabeled_dataset = UnlabeledDataset(unlabeled_data_dir, transform=TwoCropsTransform(transform))
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=4, shuffle=True)

# 初始化SimSiam模型
base_encoder = get_unetplusplus()
model_simsiam = SimSiam(base_encoder)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_simsiam = model_simsiam.to(device)

# 损失函数和优化器
criterion_simsiam = nn.CosineSimilarity(dim=1).to(device)
optimizer_simsiam = optim.Adam(model_simsiam.parameters(), lr=0.0003)

# 训练SimSiam模型

def train_simsiam_model(model, dataloader, criterion, optimizer, num_epochs=10, filename='loss.csv'):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        # 使用 tqdm 显示训练进度条
        for batch_idx, (inputs1, inputs2) in enumerate(tqdm(dataloader, desc="Training", unit="batch", ncols=100)):
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)

            # 前向传播
            p1, p2, z1, z2 = model(inputs1, inputs2)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            # 反向传播 + 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs1.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Train Loss: {epoch_loss:.4f}')
        
        # 保存损失到CSV，模型名为 'SimSiam'
        save_loss_to_csv(epoch_loss, epoch, 'SimSiam', filename)

    return model

trained_simsiam = train_simsiam_model(model_simsiam, unlabeled_loader, criterion_simsiam, optimizer_simsiam, num_epochs=5)

# # 保存 SimSiam 编码器权重
torch.save(trained_simsiam.encoder.state_dict(), 'simsiam_encoder-1.pth')

# 加载 UNet++ 模型
unetplusplus = get_unetplusplus()

# 加载 SimSiam 编码器的权重
unetplusplus = load_model(unetplusplus, 'simsiam_encoder-1.pth')

unetplusplus = unetplusplus.to(device)

# 设置损失函数和优化器
criterion_unet = nn.BCEWithLogitsLoss()
optimizer_unet = optim.Adam(unetplusplus.parameters(), lr=0.001)




# 损失函数和优化器为UNet++
criterion_unet = nn.BCEWithLogitsLoss()
optimizer_unet = optim.Adam(unetplusplus.parameters(), lr=0.001)

# 训练UNet++模型（如果需要）

def train_unet_model(model, dataloader, criterion, optimizer, num_epochs=10, filename='unet++.csv'):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        # 使用 tqdm 显示训练进度条
        for batch_idx, (inputs, masks) in enumerate(tqdm(dataloader, desc="Training", unit="batch", ncols=100)):
            inputs = inputs.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, masks)

            # 反向传播 + 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Train Loss: {epoch_loss:.4f}')
        
        # 保存损失到CSV，模型名为 'UNet++'
        save_loss_to_csv(epoch_loss, epoch, 'UNet++', filename)

    return model
# 这里可以调用train_unet_model来进一步训练UNet++模型
trained_unet = train_unet_model(unetplusplus, train_loader_thymoma, criterion_unet, optimizer_unet, num_epochs=100)

# 测试函数
# def test_model(model, dataloader, device):
#     model.eval()
#     all_preds = []
#     all_masks = []

#     with torch.no_grad():
#         for inputs, masks in dataloader:
#             inputs = inputs.to(device)
#             masks = masks.to(device)

#             outputs = model(inputs)
#             preds = torch.sigmoid(outputs) > 0.5

#             all_preds.append(preds.cpu().numpy())
#             all_masks.append(masks.cpu().numpy())

#     return np.concatenate(all_preds), np.concatenate(all_masks)
def collect_images(model,dataloader, num_images=100):
    model.eval()
    images_list = []
    masks_list = []
    preds_list = []
    
    count = 0
    with torch.no_grad():
        for inputs, masks in dataloader:
            print(f"Input shape: {inputs.shape}")  # 输出输入形状
            inputs = inputs.to(device).float()
            masks = masks.to(device).float()
            
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            
            images_list.append(inputs.cpu().numpy())
            masks_list.append(masks.cpu().numpy())
            preds_list.append(preds)
            
            count += inputs.shape[0]
            if count >= num_images:
                break

    images = np.concatenate(images_list)[:num_images]
    masks = np.concatenate(masks_list)[:num_images]
    preds = np.concatenate(preds_list)[:num_images]
    
    return images, masks, preds

def save_results(images, masks, preds, output_dir='checkpoints-1', num_images=100):
    # 如果输出目录不存在，创建该目录
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        # 创建一个3列的图，分别存放原图、掩码和预测
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # 显示原图
        ax[0].imshow(images[i].squeeze(), cmap='gray')
        ax[0].set_title('Image')
        ax[0].axis('off')

        # 显示掩码
        ax[1].imshow(masks[i].squeeze(), cmap='gray')
        ax[1].set_title('Mask')
        ax[1].axis('off')

        # 显示预测
        ax[2].imshow(preds[i].squeeze(), cmap='gray')
        ax[2].set_title('Prediction')
        ax[2].axis('off')

        # 保存拼接后的图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'result_{i+1}.png'))
        plt.close(fig)

    print(f"Results saved to {output_dir}")
# 测试UNet++模型并获取预测结果
# preds, masks = test_model(trained_unet, test_loader_thymoma, device)


images, masks, preds = collect_images(trained_unet,test_loader_thymoma, num_images=100)

# 保存结果
save_results(images, masks, preds, num_images=100)

