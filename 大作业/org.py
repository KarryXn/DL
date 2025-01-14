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
data_dir_thymoma = r'data\thymoma\thymoma'
unlabeled_data_dir = r'data\subset_2'
print("数据路径是否存在:", os.path.exists(data_dir_thymoma))
print("数据路径下的文件:", os.listdir(data_dir_thymoma) if os.path.exists(data_dir_thymoma) else "路径不存在")
dataset_thymoma = ThymomaDataset(data_dir_thymoma, transform=transform)
train_size = int(0.8 * len(dataset_thymoma))
test_size = len(dataset_thymoma) - train_size
train_dataset_thymoma, test_dataset_thymoma = torch.utils.data.random_split(dataset_thymoma, [train_size, test_size])
train_loader_thymoma = DataLoader(train_dataset_thymoma, batch_size=32, shuffle=True)
test_loader_thymoma = DataLoader(test_dataset_thymoma, batch_size=32, shuffle=False)

unlabeled_dataset = UnlabeledDataset(unlabeled_data_dir, transform=TwoCropsTransform(transform))
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True)

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
def train_simsiam_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        for (inputs1, inputs2) in dataloader:
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

    return model

trained_simsiam = train_simsiam_model(model_simsiam, unlabeled_loader, criterion_simsiam, optimizer_simsiam, num_epochs=5)

# 使用SimSiam预训练编码器初始化UNet++
unetplusplus = get_unetplusplus()
unetplusplus.encoder.load_state_dict(trained_simsiam.encoder.state_dict())
unetplusplus = unetplusplus.to(device)

# 损失函数和优化器为UNet++
criterion_unet = nn.BCEWithLogitsLoss()
optimizer_unet = optim.Adam(unetplusplus.parameters(), lr=0.001)

# 训练UNet++模型（如果需要）
def train_unet_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        for inputs, masks in dataloader:
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

    return model

# 这里可以调用train_unet_model来进一步训练UNet++模型
trained_unet = train_unet_model(unetplusplus, train_loader_thymoma, criterion_unet, optimizer_unet, num_epochs=5)

# 测试函数
def test_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5

            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_masks)

# 可视化结果
def visualize_results(images, masks, preds, num_images=4):
    fig, ax = plt.subplots(num_images, 3, figsize=(12, 12))

    for i in range(num_images):
        ax[i, 0].imshow(images[i].squeeze(), cmap='gray')
        ax[i, 0].set_title('Image')
        ax[i, 1].imshow(masks[i].squeeze(), cmap='gray')
        ax[i, 1].set_title('Mask')
        ax[i, 2].imshow(preds[i].squeeze(), cmap='gray')
        ax[i, 2].set_title('Prediction')

    for a in ax.flat:
        a.axis('off')

    plt.tight_layout()
    plt.show()

# 测试UNet++模型并获取预测结果
preds, masks = test_model(trained_unet, test_loader_thymoma, device)

# 获取一些测试图像用于可视化
images, _ = next(iter(test_loader_thymoma))
images = images.numpy()

# 可视化结果
visualize_results(images, masks, preds)