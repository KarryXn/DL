import torch
import albumentations as A
import numpy as np
from skin_datamask import SkinDataset
from torch.utils.data import DataLoader
from model import UNetEncoder
from model import SimCLR
from loss import MultiClassDiceCoeff, MultiClassDiceLoss
import pandas as pd
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
from unet import UNet
import cv2
def calc_loss(z1,z2,tau,device):
    z1  =z1/z1.norm(dim = 1,keepdim=True)
    z2 = z2/z2.norm(dim=1,keepdim=True)
    similar_matrix = tau*(z1@z2.t())
    N = similar_matrix.shape[1]
    labels = torch.arange(N).to(device)

    l1 = F.cross_entropy(similar_matrix,labels)
    l2 = F.cross_entropy(similar_matrix.t(),labels)

    return 0.5*(l1+l2)


def draw_progress_bar(cur, total, bar_len=50):
    '''
    print progress bar during training
    '''
    cur_len = int(cur/total*bar_len)
    sys.stdout.write('\r')
    sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
    sys.stdout.flush()



def plot_preds(ims, preds, masks, save_path=None):
    '''
    用于可视化训练中间结果，并选择性保存到文件
    - 绿色：金标签掩码的疾病轮廓
    - 红色：预测掩码的疾病轮廓
    '''
    preds = torch.softmax(preds, dim=1)  # 如果是多类，softmax处理
    preds = (preds[0, 1, :, :] > 0.5).cpu().numpy()  # 选择第1类疾病，二值化为0和1
    masks = masks.cpu().numpy()  # 真实掩码

    ims = ims.detach().cpu().numpy().transpose((0, 2, 3, 1))  # 转换为 HxWxC 格式
    image = ims[0]  # 假设处理的是第一个图像

    # 获取预测和真实掩码的轮廓
    pred_mask = preds.astype(np.uint8)  # 预测掩码
    true_mask = masks[0, :, :].astype(np.uint8)  # 真实掩码（第一个样本）

    # 使用cv2.findContours获取掩码轮廓
    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    true_contours, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始图像上绘制轮廓
    image_true_contour = image.copy()
    image_pred_contour = image.copy()

    # 确保图像是三通道（RGB）图像
    if image_true_contour.ndim == 2:
        image_true_contour = cv2.cvtColor(image_true_contour, cv2.COLOR_GRAY2BGR)
    if image_pred_contour.ndim == 2:
        image_pred_contour = cv2.cvtColor(image_pred_contour, cv2.COLOR_GRAY2BGR)

    # 绘制真实掩码轮廓（绿色）到原始图像
    cv2.drawContours(image_true_contour, true_contours, -1, (0, 255, 0), 2)  # 绿色表示真实标签

    # 绘制预测掩码轮廓（红色）到原始图像
    cv2.drawContours(image_pred_contour, pred_contours, -1, (0, 0, 255), 2)  # 红色表示预测标签

    # 如果提供了保存路径，则保存图像
    if save_path:
        # 保存预测掩码轮廓图像
        pred_contour_path = save_path.replace(".png", "_pred_contour.png")
        cv2.imwrite(pred_contour_path, image_pred_contour)  # 保存预测掩码轮廓
        print(f"Predicted contour saved to {pred_contour_path}")

        # 保存真实掩码轮廓图像
        true_contour_path = save_path.replace(".png", "_true_contour.png")
        cv2.imwrite(true_contour_path, image_true_contour)  # 保存真实掩码轮廓
        print(f"True contour saved to {true_contour_path}")

        # 保存叠加轮廓的图像
        image_with_contours = image.copy()
        if image_with_contours.ndim == 2:
            image_with_contours = cv2.cvtColor(image_with_contours, cv2.COLOR_GRAY2BGR)
        # 绘制绿色真实掩码轮廓
        cv2.drawContours(image_with_contours, true_contours, -1, (0, 255, 0), 2)
        # 绘制红色预测掩码轮廓
        cv2.drawContours(image_with_contours, pred_contours, -1, (0, 0, 255), 2)
        
        save_overlay_path = save_path.replace(".png", "_contours_overlay.png")
        cv2.imwrite(save_overlay_path, image_with_contours)
        print(f"Overlay with contours saved to {save_overlay_path}")

    # 绘制原始图像、预测掩码和金标签掩码
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(np.uint8(image * 255))  # 转换为 [0, 255] 范围
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap='gray')  # 预测掩码
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(true_mask, cmap='gray')  # 金标签掩码
    plt.title("Ground Truth")
    plt.axis('off')

    # 绘制叠加轮廓的图像
    plt.figure(figsize=(6, 6))
    plt.imshow(image_with_contours)
    plt.title("Predicted and Ground Truth Contours")
    plt.axis('off')

    # 显示所有图像
    plt.show()

if __name__=='__main__':
 

    print('start main')

    #构建数据读取器
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
    test_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
   


    model = UNet(in_channels=3,num_classes=2,base_num_filters=32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(device)
    
    pretrain_file = r'checkpoints-11\MAE\best_weights.pth'
    assert os.path.exists(pretrain_file)
    print('loading pretrained from',pretrain_file)
    model.load_state_dict(torch.load(pretrain_file,map_location=device),strict=False)
    #初始化损失函数和优化器
    ce_func = torch.nn.CrossEntropyLoss()#用于计算模型预测值和标签值之间的距离
    dl_func =  MultiClassDiceLoss(num_classes=2, skip_bg=True)
    dsc_func = MultiClassDiceCoeff(num_classes=2, skip_bg=True)

    #optimizer负责更新model.parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
  
    #开辟保存目录，存储模型参数和训练参数
    save_path = 'checkpoints/base_mae_unet'
    os.makedirs(save_path, exist_ok=True)

    df_summary = pd.DataFrame(columns = ['time', 'step', 'train loss'])
    df_summary.to_csv(os.path.join(save_path, "training_summary.csv"), index=False)

    #循环优化模型参数
     #循环优化模型参数
    num_epochs = 30

    max_dsc = 0

    steps_per_epoch = len(train_dataset)//batch_size
    for epoch in range(num_epochs):
        print('Epoch: #', epoch+1)
        t1= time.time()  

        train_loss = 0.0
        model.train()
        for step, (ims, masks) in enumerate(train_dataloader):
            #注意：数据tensor和model必须在同一个设备里面
            ims = ims.to(device).float()        
            masks = masks.to(device).long()

            preds = model(ims)
            # 修改目标张量的形状
            
            # print("masks.shape",masks.shape)

               
            #torch.nn.CrossEntropyLoss()自动计算了softmax概率和labels的one hot编码
            loss = ce_func(preds, masks) + dl_func(preds, masks)
            train_loss+=loss.item()

            optimizer.zero_grad()
            loss.backward()#方向传播算loss对模型参数的梯度
            optimizer.step()#基于梯度更新模型参数
            draw_progress_bar(step+1, steps_per_epoch)  

        t2 = time.time()
       
        #一个epoch优化完成，执行验证，打印损失和准确率信息  
        print('\n starting evaluating...')
        eval_dsc = 0.0
        model.eval()

        show_yet = False
        with torch.no_grad():             
            for ims, masks in test_dataloader:
                ims = ims.to(device).float()        
                masks = masks.to(device).long()

                preds = model(ims)
                # print("masks shape:", masks.shape)
                # print("masks dtype:", masks.dtype)
                # print("unique values in masks:", masks.unique())
                # print("preds shape:", preds.shape)
                # print("preds dtype:", preds.dtype)
                dsc = dsc_func(preds, masks)
                eval_dsc += dsc.item()
                #如果epoch是5的整数倍，且没有显示过分割结果，则可视化展示
                if show_yet==False and epoch%5==0:
                  
                    save_visual_path = os.path.join(save_path, f"epoch_{epoch}_step_{step}_viz.png")
                    plot_preds(ims, preds, masks, save_path=save_visual_path)
                    show_yet = True

        #打印训练进度信息
        train_loss = train_loss/len(train_dataset)
        eval_dsc = eval_dsc/len(test_dataset)        
        print('Train dice loss:', train_loss, ' Eval DSC:', eval_dsc, ' Train time:', t2-t1)

        #保存最佳模型参数
        checkpoint_file = os.path.join(save_path, "best_weights.pth")  
        if max_dsc<=eval_dsc:
            max_dsc = eval_dsc
            print ('Saving weights to %s' % (checkpoint_file))    
            torch.save(model.state_dict(), checkpoint_file)

        #保存训练参数
        current_time = "%s"%datetime.now()#获取当前时间
        step = "Step[%d]"%epoch
        str_train_loss = "%f"%train_loss
        str_eval_dsc = "%f"%eval_dsc
    
        list = [current_time, step, str_train_loss, str_eval_dsc]
        df_summary = pd.DataFrame([list])
        df_summary.to_csv(os.path.join(save_path, "training_summary.csv"), mode='a', header=False, index=False)



    

