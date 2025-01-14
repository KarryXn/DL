import torch
import albumentations as A
import numpy as np
from skin_datamask import SkinDataset
from torch.utils.data import DataLoader
from model import UNetEncoder, SimCLR  
from loss import MultiClassDiceCoeff, MultiClassDiceLoss
import pandas as pd
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import sys



def calc_loss(z1, z2, tau, device):
    z1 = z1 / z1.norm(dim=1, keepdim=True)
    z2 = z2 / z2.norm(dim=1, keepdim=True)
    similar_matrix = tau * (z1 @ z2.t())
    N = similar_matrix.shape[1]
    labels = torch.arange(N).to(device)

    l1 = F.cross_entropy(similar_matrix, labels)
    l2 = F.cross_entropy(similar_matrix.t(), labels)

    return 0.5 * (l1 + l2)

def draw_progress_bar(cur, total, bar_len=50):
    cur_len = int(cur / total * bar_len)
    sys.stdout.write('\r')
    sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
    sys.stdout.flush()

def plot_preds(ims, preds, masks, save_path=None):
    preds = torch.softmax(preds, dim=1)

    ims = ims.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(np.uint8(ims[0, ...]).transpose((1, 2, 0)))
    plt.title("Input Image")

    plt.subplot(1, 3, 2)
    plt.imshow(preds[0, 1, ...], cmap='gray')
    plt.title("Predicted Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(masks[0, ...], cmap='gray')
    plt.title("Ground Truth")

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    plt.close()

class SimCLRWithUNet(torch.nn.Module):
    def __init__(self, simclr_model, unet_encoder):
        super(SimCLRWithUNet, self).__init__()
        self.simclr = simclr_model
        self.unet_encoder = unet_encoder  # UNet encoder part
        
        # 添加用于分割的卷积层
        self.conv = torch.nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)  # 假设是二分类

    def forward(self, x):
        features = self.simclr(x)  # 使用SimCLR提取特征
        seg_map = self.unet_encoder(features)  # 使用UNet编码器进行处理
        seg_map = self.conv(seg_map)  # 最终分割预测
        return seg_map

if __name__ == '__main__':
    print('start main')

    transforms = A.Compose([A.HorizontalFlip(p=0.5),
                            A.RandomBrightnessContrast(p=0.2),
                            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=180, p=0.5),
                            A.GridDistortion(p=0.1),
                            A.OpticalDistortion(p=0.1),
                            A.Resize(128, 128)])

    batch_size = 10
    train_dataset = SkinDataset('ISBI2016_ISIC_Part1_Training_Data\ISBI2016_ISIC_Part1_Training_Data', transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    test_dataset = SkinDataset('ISBI2016_ISIC_Part1_Test_Data\ISBI2016_ISIC_Part1_Test_Data')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=True)

    # 初始化模型，假设SimCLR和UNetEncoder已经定义
    simclr_model = SimCLR(in_channels=3, base_num_filters=32, z_dim=256)
    unet_encoder = UNetEncoder(in_channels=256, out_channels=2)  # 假设UNet的输入是256通道
    model = SimCLRWithUNet(simclr_model, unet_encoder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(device)

    ce_func = torch.nn.CrossEntropyLoss()
    dl_func = MultiClassDiceLoss(num_classes=2, skip_bg=True)
    dsc_func = MultiClassDiceCoeff(num_classes=2, skip_bg=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    save_path = 'checkpoints/simclr_unet'
    os.makedirs(save_path, exist_ok=True)

    df_summary = pd.DataFrame(columns=['time', 'step', 'train loss'])
    df_summary.to_csv(os.path.join(save_path, "training_summary.csv"), index=False)

    num_epochs = 30
    max_dsc = 0
    steps_per_epoch = len(train_dataset) // batch_size
    for epoch in range(num_epochs):
        print('Epoch: #', epoch + 1)
        t1 = time.time()

        train_loss = 0.0
        model.train()
        for step, (im1, im2, masks) in enumerate(train_dataloader):
            im1 = im1.to(device).float()
            im2 = im2.to(device).float()
            masks = masks.to(device).long()  # 假设masks是长整型标签

            # 获取SimCLR输出的嵌入，并用UNet进行分割
            pred_masks = model(im1)

            # 计算损失，使用DiceLoss或交叉熵损失
            loss = dl_func(pred_masks, masks)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            draw_progress_bar(step + 1, steps_per_epoch)

        t2 = time.time()

        train_loss = train_loss / len(train_dataset)
        print('Train loss:', train_loss, ' Train time:', t2 - t1)

        # 保存最佳模型
        checkpoint_file = os.path.join(save_path, "best_weights.pth")
        torch.save(model.state_dict(), checkpoint_file)

        # 保存训练日志
        current_time = "%s" % datetime.now()
        step = "Step[%d]" % epoch
        str_train_loss = "%f" % train_loss

        list = [current_time, step, str_train_loss]
        df_summary = pd.DataFrame([list])
        df_summary.to_csv(os.path.join(save_path, "training_summary.csv"), mode='a', header=False, index=False)
