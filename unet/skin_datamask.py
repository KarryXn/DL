from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

class SkinDataset(Dataset):
    def __init__(self, data_root, transforms=None):
        super().__init__()
        self.transforms = transforms
        im_files = glob.glob(osp.join(data_root, '*.jpg'))  # 图像文件
        self.im_mask_file = []
        for im_f in  im_files:
            mask_f = im_f.replace('_Data','_GroundTruth').replace('.jpg','_Segmentation.png')
            if osp.exists(mask_f):
                self.im_mask_file.append([im_f,mask_f])
       
        
        print("Found images and masks", len(self.im_mask_file))

    def __getitem__(self, index):
        im_f,mask_f = self.im_mask_file[index]
        
        im = cv2.imread(im_f)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        
        
        mask = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)  # 以灰度方式读取掩码

        trans = self.transforms(image=im,mask=mask)
       
        im = trans['image']
        mask = trans['mask']
        mask[mask>0] = 1
        im = im.transpose((2,0,1))  # 将图像转换为 (C, H, W)
      

        return im, mask

    def __len__(self):
        return len(self.im_mask_file)

if __name__ == '__main__':
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=180, p=0.5),
        A.GridDistortion(p=0.8),
        A.OpticalDistortion(p=0.8),
        A.Resize(128, 128)  # 确保图像和掩码的尺寸一致
    ])
    dataset = SkinDataset(data_root=r'ISBI2016_ISIC_Part1_Training_Data\ISBI2016_ISIC_Part1_Training_Data', transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    
    for im, mask in dataloader:
        print(im.shape)  
        print(mask.shape) 
        im = im[0, ...].numpy().transpose((1,2,0))
        mask = mask[0, ...].numpy()

        # 可视化图像和掩码
        plt.subplot(1, 2, 1)
        plt.imshow(im)
        plt.title('Image')
        plt.subplot(1, 2, 2)
        plt.imshow(mask.squeeze(), cmap='gray')  # 用灰度色显示掩码
        plt.title('Mask')
        
        plt.show()
