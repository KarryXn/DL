import torch
import albumentations as A
import numpy as np
from skin_dataset import SkinDataset  # Assuming this dataset class is defined elsewhere
from torch.utils.data import DataLoader
from model import UNetEncoder, SimCLR  # Assuming SimCLR and UNetEncoder are defined elsewhere
from loss import MultiClassDiceCoeff, MultiClassDiceLoss  # Assuming loss functions are defined elsewhere
import pandas as pd
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F


# 对比学习损失函数（NT-Xent Loss）
def calc_loss(z1, z2, tau, device):
    z1 = z1 / z1.norm(dim=1, keepdim=True)  # Normalize the embeddings
    z2 = z2 / z2.norm(dim=1, keepdim=True)  # Normalize the embeddings
    
    # Compute similarity matrix
    similarity_matrix = tau * (z1 @ z2.t())  # Cosine similarity matrix
    N = similarity_matrix.shape[1]  # Batch size
    labels = torch.arange(N).to(device)  # Labels for contrastive learning

    # Compute the contrastive loss
    l1 = F.cross_entropy(similarity_matrix, labels)
    l2 = F.cross_entropy(similarity_matrix.t(), labels)
    
    return 0.5 * (l1 + l2)


# 打印训练进度条
def draw_progress_bar(cur, total, bar_len=50):
    '''
    print progress bar during training
    '''
    cur_len = int(cur / total * bar_len)
    sys.stdout.write('\r')
    sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
    sys.stdout.flush()


# 可视化训练中间结果
def plot_preds(ims, preds, masks, save_path=None):
    '''
    用于可视化训练中间结果，并选择性保存到文件
    '''
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


# 数据增强配置
transforms = A.Compose([A.HorizontalFlip(p=0.5),
                        A.RandomBrightnessContrast(p=0.2),
                        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=180, p=0.5),
                        A.GridDistortion(p=0.1),
                        A.OpticalDistortion(p=0.1),
                        A.Resize(128, 128)])


# 主程序入口
if __name__ == '__main__':
    print('start main')

    # 构建数据读取器
    batch_size = 10
    train_dataset = SkinDataset('ISBI2016_ISIC_Part1_Training_Data\\ISBI2016_ISIC_Part1_Training_Data', transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    test_dataset = SkinDataset('ISBI2016_ISIC_Part1_Test_Data\\ISBI2016_ISIC_Part1_Test_Data')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)

    # 初始化神经网络模型实例
    model = SimCLR(inchannels=3, base_num_filters=32, z_dim=256)  # Assuming SimCLR is defined elsewhere
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(device)
    
    # 初始化损失函数和优化器
    ce_func = torch.nn.CrossEntropyLoss()  # 用于计算模型预测值和标签值之间的距离
    dl_func = MultiClassDiceLoss(num_classes=2, skip_bg=True)
    dsc_func = MultiClassDiceCoeff(num_classes=2, skip_bg=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
  
    # 开辟保存目录，存储模型参数和训练参数
    save_path = 'checkpoints/simclr'
    os.makedirs(save_path, exist_ok=True)

    df_summary = pd.DataFrame(columns=['time', 'step', 'train loss'])
    df_summary.to_csv(os.path.join(save_path, "training_summary.csv"), index=False)

    # 循环优化模型参数
    num_epochs = 30
    steps_per_epoch = len(train_dataset) // batch_size
    max_dsc = 0

    for epoch in range(num_epochs):
        print('Epoch: #', epoch + 1)
        t1 = time.time()

        train_loss = 0.0
        model.train()
        for step, (im1, im2) in enumerate(train_dataloader):
            # 注意：数据tensor和model必须在同一个设备里面
            im1 = im1.to(device).float()
            im2 = im2.to(device).float()  

            # 获取特征表示
            z1 = model(im1)
            z2 = model(im2)     

            # 计算对比损失
            loss = calc_loss(z1, z2, model.tau, device)
            train_loss += loss.item()

            # 反向传播并优化模型参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印训练进度
            draw_progress_bar(step + 1, steps_per_epoch)  

        t2 = time.time()
       
        # 评估阶段（每个epoch结束后）
        print('\nStarting evaluation...')
        eval_dsc = 0.0
        model.eval()

        # 计算Dice系数（如有需要）
        # eval_dsc = ...

        # 打印训练损失信息
        train_loss = train_loss / len(train_dataset)
        print('Train loss:', train_loss, ' Train time:', t2 - t1)

        # 保存最佳模型参数
        checkpoint_file = os.path.join(save_path, "best_weights.pth")  
        torch.save(model.state_dict(), checkpoint_file)

        # 保存训练参数
        current_time = "%s" % datetime.now()
        step = "Step[%d]" % epoch
        str_train_loss = "%f" % train_loss
        summary = [current_time, step, str_train_loss]
        df_summary = pd.DataFrame([summary])
        df_summary.to_csv(os.path.join(save_path, "training_summary.csv"), mode='a', header=False, index=False)

        # 可视化训练结果（可选）
        if epoch % 5 == 0:  # 每5个epoch保存一次可视化结果
            ims, preds, masks = im1, z1, im2  # 替换为实际的预测结果
            plot_preds(ims, preds, masks, save_path=os.path.join(save_path, f"epoch_{epoch}_visualization.png"))

