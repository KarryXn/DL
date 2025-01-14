import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, base_num_filters=16):
        super(UNetEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, base_num_filters, kernel_size=3, padding=1)  # (N, 3, 128, 128) -> (N, 16, 128, 128)
        self.conv2 = nn.Conv2d(base_num_filters, base_num_filters*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_num_filters*2, base_num_filters*4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(base_num_filters*4, base_num_filters*8, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(base_num_filters*8, base_num_filters*16, kernel_size=3, padding=1)

    def forward(self, x):
       
        # Encoding path
        c1 = F.relu(self.conv1(x))  # (N, 16, 128, 128)
        p1 = F.max_pool2d(c1, kernel_size=2, stride=2)  # (N, 16, 64, 64)
        
        c2 = F.relu(self.conv2(p1))  # (N, 32, 64, 64)
        p2 = F.max_pool2d(c2, kernel_size=2, stride=2)  # (N, 32, 32, 32)
        
        c3 = F.relu(self.conv3(p2))  # (N, 64, 32, 32)
        p3 = F.max_pool2d(c3, kernel_size=2, stride=2)  # (N, 64, 16, 16)
        
        c4 = F.relu(self.conv4(p3))  # (N, 128, 16, 16)
        p4 = F.max_pool2d(c4, kernel_size=2, stride=2)  # (N, 128, 8, 8)
        
        c5 = F.relu(self.conv5(p4))  # (N, 256, 8, 8)
       
        return  c5
class Projector(nn.Module):
    def __init__(self, inchannels=512, z_dims=256):  # 设置inchannels为512，匹配UNetEncoder的输出
        super(Projector, self).__init__()
        self.avgpool = nn.AdaptiveMaxPool2d((8, 8))
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(8*8*inchannels, z_dims),  # 输入形状应该是8*8*512
            nn.ReLU(),
            nn.Linear(z_dims, z_dims)
        )

    def forward(self, x):
       
        x = self.avgpool(x)  # 将输入大小调整为(8, 8)
        
        x = torch.flatten(x, start_dim=1, end_dim=-1)  # 展平为 (batch_size, 512*8*8)
       
        x = self.projection(x)
     
        return x


class SimCLR(nn.Module):
    def __init__(self, inchannels=3, base_num_filters=32, z_dim=512):
        super(SimCLR, self).__init__()
        self.encoder = UNetEncoder(in_channels=inchannels, base_num_filters=base_num_filters)
        self.projector = Projector(inchannels=512, z_dims=z_dim)  # 最后一层通道数是base_num_filters*16
        self.tau = nn.Parameter(torch.ones([])*np.log(1/0.07))

    def forward(self, x):
        c5 = self.encoder(x)  # 只返回最后一层
        
        if isinstance(c5, tuple):  # 如果是 tuple，选择第一个输出
            c5 = c5[0]
        
        z = self.projector(c5)

        return z
class MAEDecoder(nn.Module):
    def __init__(self, z_dim=256, base_num_filters=16):
        super(MAEDecoder, self).__init__()

        # 解码器层：反卷积（转置卷积）逐步恢复空间分辨率
        self.deconv1 = nn.ConvTranspose2d(z_dim, base_num_filters*8, kernel_size=4, stride=2, padding=1)  # (N, 256, 8, 8) -> (N, 128, 16, 16)
        self.deconv2 = nn.ConvTranspose2d(base_num_filters*8, base_num_filters*4, kernel_size=4, stride=2, padding=1)  # (N, 128, 16, 16) -> (N, 64, 32, 32)
        self.deconv3 = nn.ConvTranspose2d(base_num_filters*4, base_num_filters*2, kernel_size=4, stride=2, padding=1)  # (N, 64, 32, 32) -> (N, 32, 64, 64)
        self.deconv4 = nn.ConvTranspose2d(base_num_filters*2, 3, kernel_size=4, stride=2, padding=1)  # (N, 32, 64, 64) -> (N, 3, 128, 128)

    def forward(self, x):
        # 解码路径
        x = F.relu(self.deconv1(x))  # (N, 128, 16, 16)
        x = F.relu(self.deconv2(x))  # (N, 64, 32, 32)
        x = F.relu(self.deconv3(x))  # (N, 32, 64, 64)
        x = self.deconv4(x)  # (N, 3, 128, 128) 这里假设重建到原始图像大小
        return x

class MAE(nn.Module):
    def __init__(self, in_channels=3, base_num_filters=32, z_dim=256):
        super(MAE, self).__init__()
        self.encoder = UNetEncoder(in_channels=in_channels, base_num_filters=base_num_filters)  # 使用UNet编码器
        self.decoder = MAEDecoder(z_dim=base_num_filters*16, base_num_filters=base_num_filters)  # 解码器
        self.mask_ratio = 0.75  # 设置mask比例，通常为0.75

    def forward(self, x):
        # 随机遮挡输入图像的部分区域
        masked_x, mask = self.mask_input(x)

        # 编码器提取特征
        c5 = self.encoder(masked_x)

        # 解码器重建
        reconstruction = self.decoder(c5)

        return reconstruction, mask

    def mask_input(self, x):
        # 随机遮挡输入图像的部分区域，生成一个mask
        B, C, H, W = x.size()
        mask = torch.ones_like(x)  # 创建一个全1的mask
        num_masked_pixels = int(self.mask_ratio * H * W)  # 计算遮挡的像素数

        # 随机生成mask，遮挡部分像素
        mask_flat = mask.view(B, -1)
        mask_flat[:, torch.randperm(mask_flat.size(1))[:num_masked_pixels]] = 0  # 随机遮挡像素

        mask = mask.view(B, C, H, W)
        masked_x = x * mask  # 将遮挡的区域置为0

        return masked_x, mask

# 测试
if __name__ == '__main__':
    mae = MAE(in_channels=3, base_num_filters=32, z_dim=256)  # MAE模型
    batch = torch.randn((10, 3, 128, 128))  # 输入一个形状为(N, 3, 128, 128)的批次

    reconstruction, mask = mae(batch)  # 得到重建的图像和遮挡的mask
    print(reconstruction.shape, mask.shape)  # 期望输出：(10, 3, 256, 256) (10, 3, 128, 128)