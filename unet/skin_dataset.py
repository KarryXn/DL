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
        self.im_file = glob.glob(osp.join(data_root, '*.jpg'))
        
        
            
        print("Found images and masks", len(self.im_file))

    def __getitem__(self, index):
        im_f = self.im_file[index]
        im = cv2.imread(im_f)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        trans1 = self.transforms(image = im)
        trans2 = self.transforms(image = im)
        im1 = trans1['image']
        im2 = trans2['image']
        
        im1 = im1.transpose((2,0,1))
        im2 = im2.transpose((2,0,1))

        return im1,im2

    def __len__(self):
        return len(self.im_file)

if __name__ == '__main__':
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=180, p=0.5),
        A.GridDistortion(p=0.1),
        A.OpticalDistortion(p=0.1),
        A.Resize(128, 128)
    ])
    dataset = SkinDataset(data_root=r"ISBI2016_ISIC_Part1_Training_Data\ISBI2016_ISIC_Part1_Training_Data", transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    
    for im1, im2 in dataloader:
        print(im1.shape)  
        print(im2.shape) 
        im1 = im1[0, ...].numpy().transpose((1,2,0))
        im2 = im2[0, ...].numpy().transpose((1,2,0))
       
        plt.subplot(1, 2, 1)
        plt.imshow(im1)
        plt.subplot(1, 2, 2)
        plt.imshow(im2)
        
        plt.show()
