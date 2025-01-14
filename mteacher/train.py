import torch.nn.functional
from data_loader import MTDataset
import albumentations as A
from torch.utils.data import DataLoader
from unet import UNet
import torch
from dice_loss import MultiClassDiceCoeff, MultiClassDiceLoss
import sys
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from datetime import datetime
import numpy as np
def draw_progress_bar(cur, total, bar_len=50):
    '''
    print progress bar during training
    '''
    cur_len = int(cur/total*bar_len)
    sys.stdout.write('\r')
    sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
    sys.stdout.flush()

class EMA():
    def __init__(self, model, decay = 0.98):
        self.model = model
        self.decay = decay
        self.shadow = {}#记录教师模型的参数
        self.backup = {}#备份学生模型的参数

    def register(self):
        #开始训练前就要调用，将学生模型的参数都记录到shadow里面去
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name] = param.data.clone()

    def update(self):
        #把学生的参数更新到teacher模型里面去
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        #把shadow的参数copy到model中
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        #模型参数恢复为学生模型的参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
class UNetWithDropout(UNet):
    def __init__(self, *args, dropout_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = super().forward(x)
        x = self.dropout(x)
        return x



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
if __name__ == '__main__':
    #准备好数据
    batch_size= 10

    transforms = A.Compose([A.HorizontalFlip(p=0.5),
                            A.RandomBrightnessContrast(p=0.2),
                            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=180, p=0.5),
                            A.GridDistortion(p=0.1),
                            A.OpticalDistortion(p=0.1),
                            A.Resize(128, 128)])
   
    train_roots = ['ISBI2016_ISIC_Part1_Training_Data\ISBI2016_ISIC_Part1_Training_Data']
    
    train_dataset = MTDataset(train_roots,transforms=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, persistent_workers=True)

    eval_dataset = MTDataset(['ISBI2016_ISIC_Part1_Test_Data\ISBI2016_ISIC_Part1_Test_Data'], transforms)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=2, persistent_workers=True)

    #准备好模型
  
    model = UNet(in_channels=3,num_classes=2,base_num_filters=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    ema = EMA(model, 0.95)
    ema.register()
   
    #准备训练
    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    save_path = 'checkpoints/Meanteacher_unet'
    os.makedirs(save_path, exist_ok=True)
    df_summary = pd.DataFrame(columns = ['time', 'step', 'train loss','dice'])
    df_summary.to_csv(os.path.join(save_path, "training_summary.csv"), index=False)


    num_epochs = 30
    ce_func = torch.nn.CrossEntropyLoss()#用于计算模型预测值和标签值之间的距离
    dl_func =  MultiClassDiceLoss(num_classes=2, skip_bg=True)#有标签的分割dice损失
    dsc_func = MultiClassDiceCoeff(num_classes=2, skip_bg=True)
    mse_func = torch.nn.MSELoss()#一致性损失

    steps_per_epoch = len(train_dataset)//batch_size
    max_dsc = 0

    for epoch in range(num_epochs):
        print('Epoch: #', epoch+1)
        t1= time.time() 

        model.train()
        for step, (images, masks, ys) in enumerate(train_dataloader):
            #print(images.shape, masks.shape, ys)
            #将数据转化为tensor
            images = torch.tensor(images, dtype=torch.float32).to(device)#（N,3,H,W）
            masks = torch.tensor(masks, dtype=torch.int64).to(device)#(N,H,W)
            ys = torch.tensor(ys, dtype=torch.int32).to(device)#(N)
            #将图像输入student模型中，执行inference
            student_outputs = model(images)
            #计算损失函数
            if torch.count_nonzero(ys)>0:#minibatch是否存在有标签数据
                keep = torch.where(ys>0)#只有有标签的图像才需要算ce loss
                keep_outputs = student_outputs[keep]#(N_labeled, H, W)
                keep_masks = masks[keep]
                ce_loss = ce_func(keep_outputs, keep_masks)
                dice_loss = dl_func(keep_outputs, keep_masks)
            else:
                ce_loss = 0.0
                dice_loss = 0.0

            #将图像输入teacher模型中，执行inference,不管是否有标签都要计算mse损失
            ema.apply_shadow()
            teacher_outputs = model(images)
            ema.restore()
            consistent_loss = mse_func(teacher_outputs, student_outputs)
            
            total_loss = ce_loss+dice_loss+consistent_loss

            #更新模型参数
            optimizer.zero_grad()
            total_loss.backward()
            lr_scheduler.step()

            #EMA更新教师的参数
            ema.update()
            draw_progress_bar(step+1, steps_per_epoch) 

        t2 = time.time()
        #一个epoch优化完成，执行验证，打印损失和准确率信息  
        print('\n starting evaluating...')
        eval_dsc = 0.0
        model.eval()
        show_yet = False
        with torch.no_grad():             
            for ims, masks, ys in eval_dataloader:
                ims = ims.to(device).float()        
                masks = masks.to(device).long()

                preds = model(ims)
                dsc = dsc_func(preds, masks)
                eval_dsc += dsc.item()
                if show_yet==False and epoch%5==0:
                  
                    save_visual_path = os.path.join(save_path, f"epoch_{epoch}_step_{step}_viz.png")
                    plot_preds(ims, preds, masks, save_path=save_visual_path)
                    show_yet = True
               
        train_loss = total_loss/len(train_dataset)
        #打印训练进度信息
       
        eval_dsc = eval_dsc/len(eval_dataset)        
        print('Train dice loss:', train_loss, ' Eval DSC:', eval_dsc, ' Train time:', t2-t1)
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


          



    
