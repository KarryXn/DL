# import torch
# import torch.nn as nn
# from thymoma import get_unetplusplus

# class SimSiam(nn.Module):
#     def __init__(self, base_encoder, dim=2048, pred_dim=512):
#         super(SimSiam, self).__init__()

#         # 创建编码器（backbone）
#         self.encoder = base_encoder
#         # 为encoder创建投影头（projection head）
#         prev_dim = self.encoder.segmentation_head[0].in_channels
#         self.encoder.segmentation_head = nn.Identity()  # 移除原有的分割头
#         self.projector = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
#             nn.Flatten(),                  # 展平操作
#             nn.Linear(prev_dim, prev_dim, bias=False),
#             nn.BatchNorm1d(prev_dim),
#             nn.ReLU(),
#             nn.Linear(prev_dim, dim, bias=False),
#             nn.BatchNorm1d(dim),
#             nn.ReLU(),
#             nn.Linear(dim, dim),
#             nn.BatchNorm1d(dim)
#         )
#         # 预测头（prediction head）
#         self.predictor = nn.Sequential(
#             nn.Linear(dim, pred_dim, bias=False),
#             nn.BatchNorm1d(pred_dim),
#             nn.ReLU(),
#             nn.Linear(pred_dim, dim)
#         )

#     def forward(self, x1, x2):
#         # 假设编码器输出是一个四维张量 (batch_size, channels, height, width)
#         z1 = self.projector(self.encoder(x1))  # NxD
#         z2 = self.projector(self.encoder(x2))  # NxD

#         p1 = self.predictor(z1)  # NxD
#         p2 = self.predictor(z2)  # NxD
        
#         return p1, p2, z1.detach(), z2.detach()

import torch
import torch.nn as nn
from thymoma import get_unetplusplus

class SimSiam(nn.Module):
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        super(SimSiam, self).__init__()
        # 创建编码器（backbone）
        self.encoder = base_encoder
        prev_dim = self.encoder.segmentation_head[0].in_channels
        self.encoder.segmentation_head = nn.Identity()  # 移除原有的分割头
        # 投影头（projection head）
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),                  # 展平操作
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(),
            nn.Linear(prev_dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        # 预测头（prediction head）
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))  # NxD
        z2 = self.projector(self.encoder(x2))  # NxD
        p1 = self.predictor(z1)  # NxD
        p2 = self.predictor(z2)  # NxD
        return p1, p2, z1.detach(), z2.detach()