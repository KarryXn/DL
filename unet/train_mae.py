import torch
import albumentations as A
import numpy as np
from torch.utils.data import DataLoader
from model import MAE  # MAE模型
from skin_dataset import SkinDataset  # 数据集
import os
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import sys
def draw_progress_bar(cur, total, bar_len=50):
    '''
    print progress bar during training
    '''
    cur_len = int(cur / total * bar_len)
    sys.stdout.write('\r')
    sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
    sys.stdout.flush()

def plot_preds(ims, preds, masks, save_path=None):
    '''
    用于可视化训练中间结果，并选择性保存到文件
    '''
    preds = torch.sigmoid(preds)

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

def calc_loss(reconstruction, input_image):
    # 确保输入和重建的值范围一致
    if input_image.max() > 1:  # 假设输入在 [0, 255]
        input_image = input_image / 255.0
    if reconstruction.max() > 1:  # 假设输出在 [0, 255]
        reconstruction = reconstruction / 255.0

    # 使用均方误差损失
    mae_loss = F.mse_loss(reconstruction, input_image)
    return mae_loss

if __name__ == '__main__':
    print('start main')

    # 数据增强
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=180, p=0.5),
        A.GridDistortion(p=0.1),
        A.OpticalDistortion(p=0.1),
        A.Resize(128, 128)
    ])

    batch_size = 10
    train_dataset = SkinDataset('ISBI2016_ISIC_Part1_Training_Data\\ISBI2016_ISIC_Part1_Training_Data', transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    test_dataset = SkinDataset('ISBI2016_ISIC_Part1_Test_Data\\ISBI2016_ISIC_Part1_Test_Data')
    test_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
   
    # 初始化 MAE 模型
    mae_model = MAE(in_channels=3, base_num_filters=32, z_dim=256)  # MAE的模型定义
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mae_model.to(device)
    print(f"Using device: {device}")

    # 初始化优化器和学习率调度器
    optimizer = torch.optim.Adam(mae_model.parameters(), lr=0.0001)
    lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.95)

    # 保存目录
    save_path = 'checkpoints-11/MAE'
    os.makedirs(save_path, exist_ok=True)

    # 训练参数保存
    df_summary = pd.DataFrame(columns=['time', 'step', 'Train loss','Eval loss'])
    df_summary.to_csv(os.path.join(save_path, "training_summary.csv"), index=False)

    num_epochs = 30
    steps_per_epoch = len(train_dataset) // batch_size
    for epoch in range(num_epochs):
        print(f'Epoch: #{epoch + 1}')
        t1 = time.time()

        train_loss = 0.0
        mae_model.train()
        # 修复 MAE 模型调用和输出形状处理
        for step, (im, _) in enumerate(train_dataloader):
            im = im.to(device).float()

            # 前向传播
            reconstruction, mask = mae_model(im)

            # 打印形状以验证
            # print(f"Reconstruction output shape: {reconstruction.shape}")

            # 计算损失
            loss = calc_loss(reconstruction, im)
            train_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            draw_progress_bar(step + 1, steps_per_epoch)


        t2 = time.time()

        # 评估阶段
        print('\nStarting evaluation...')
        mae_model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for im, _ in test_dataloader:
                im = im.to(device).float()
                reconstruction, _ = mae_model(im)  # 解构元组
                eval_loss += calc_loss(reconstruction, im).item()


        eval_loss /= len(test_dataset)

        # 保存最佳模型
        checkpoint_file = os.path.join(save_path, "best_weights.pth")
        torch.save(mae_model.state_dict(), checkpoint_file)

        # 保存训练记录
        current_time = "%s" % datetime.now()
        step = f"Step[{epoch}]"
        str_train_loss = f"{train_loss / len(train_dataset):.6f}"

        df_summary = pd.DataFrame([[current_time, step, str_train_loss,eval_loss]])
        df_summary.to_csv(os.path.join(save_path, "training_summary.csv"), mode='a', header=False, index=False)

        print(f'Train loss: {train_loss / len(train_dataset)}  Eval loss: {eval_loss}')
        print(f'Epoch {epoch + 1} completed in {t2 - t1:.2f}s')
