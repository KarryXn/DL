import torch
import albumentations as A
import numpy as np
from skin_dataset import SkinDataset
from torch.utils.data import DataLoader
from model import UNetEncoder
from model import SimCLR
from unet import UNet
from loss import MultiClassDiceCoeff, MultiClassDiceLoss
import pandas as pd
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import sys


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


if __name__=='__main__':
 

    print('start main')

    #构建数据读取器
    transforms = A.Compose([A.HorizontalFlip(p=0.5),
                            A.RandomBrightnessContrast(p=0.2),
                            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=180, p=0.5),
                            A.GridDistortion(p=0.1),
                            A.OpticalDistortion(p=0.1),
                            A.Resize(128, 128)])

    batch_size = 1
    train_dataset = SkinDataset('ISBI2016_ISIC_Part1_Training_Data\ISBI2016_ISIC_Part1_Training_Data', transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

   

    #初始化神经网络模型实例
    model = UNet(in_channels = 3,num_classes =3 ,base_num_filters=32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(device)
    # pretrain_file = ''
    # assert os.path.exists(pretrain_file)
    # print('loading pretrained from',pretrain_file)
    
    #初始化损失函数和优化器
    ce_func = torch.nn.CrossEntropyLoss()#用于计算模型预测值和标签值之间的距离
    dl_func =  MultiClassDiceLoss(num_classes=2, skip_bg=True)
    dsc_func = MultiClassDiceCoeff(num_classes=2, skip_bg=True)

    #optimizer负责更新model.parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
  
    #开辟保存目录，存储模型参数和训练参数
    save_path = 'checkpoints/ae'
    os.makedirs(save_path, exist_ok=True)

    df_summary = pd.DataFrame(columns = ['time', 'step', 'train loss'])
    df_summary.to_csv(os.path.join(save_path, "training_summary.csv"), index=False)

    #循环优化模型参数
    num_epochs = 30

    max_dsc = 0

    steps_per_epoch = len(train_dataset)//batch_size
    for epoch in range(num_epochs):
        print('Epoch: #', epoch+1)
        t1= time.time()  

        train_loss = 0.0
        model.train()
        loss_func = torch.nn.MSELoss()
        for step, (im1, im2) in enumerate(train_dataloader):
            #注意：数据tensor和model必须在同一个设备里面
            im = torch.concat((im1,im2),axis=0)#(2N,C,H,W)
            im = im.to(device).float()
            im = im/255.0
            noised_im = add_noise(im)
            recon_im= model(noised_im)#(2N,C,H,W)
            recon_im = torch.nn.Sigmoid()(recon_im)



            loss = loss_func(recon_im,im)
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
       
        #打印训练进度信息
        train_loss = train_loss/len(train_dataset)
          
        print('Train dice loss:', train_loss, ' Train time:', t2-t1)

        #保存最佳模型参数
        checkpoint_file = os.path.join(save_path, "best_weights.pth")  
          
        torch.save(model.state_dict(), checkpoint_file)

        #保存训练参数
        current_time = "%s"%datetime.now()#获取当前时间
        step = "Step[%d]"%epoch
        str_train_loss = "%f"%train_loss
        
    
        list = [current_time, step, str_train_loss,]
        df_summary = pd.DataFrame([list])
        df_summary.to_csv(os.path.join(save_path, "training_summary.csv"), mode='a', header=False, index=False)



    
