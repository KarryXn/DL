from typing import Any
from torch.utils.data import Dataset
import glob 
import os.path as osp
import numpy as np
import json
import cv2
from PIL import Image
import PIL

class MTDataset(Dataset):
    def __init__(self, data_roots, transforms=None):
        super(MTDataset, self).__init__()
        #data_roots指定多个路径
        self.data_roots = data_roots
        self.transforms = transforms

        #记录所有图像文件名称
        self.samples = []
      
        for root in data_roots:         
            #搜索root下面的所有的jpg图形
            im_files = glob.glob(osp.join(root, '**', '*.jpg'), recursive=True)
            for im_file in im_files:
                anno_file = im_file.replace('_Data', '_GroundTruth').replace('.jpg', '_Segmentation.png')
                if osp.exists(anno_file):
                    self.samples.append([im_file, anno_file])
                else:#无标签的数据，用None填空
                    self.samples.append([im_file, None])

        np.random.shuffle(self.samples)
       
        print('Total samples:', len(self.samples))
      
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        im_file, anno_file = self.samples[index]
        #用cv2读取BGR
        image = cv2.imread(im_file, 1)
        #转为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if anno_file is None:
            mask = np.zeros(image.shape[0:2], dtype=np.int32)#创建0矩阵, 后面也不会用到
            y = 0.0#指示minibatch里面哪张图像无标签
        else:
            mask = cv2.imread(anno_file, 0)#灰度图方式读取
            mask[mask>0] = 1
            y = 1.0#指示minibatch里面哪张图像有标签

        if self.transforms:
            transformed = self.transforms(image = image, mask = mask)
            image = transformed['image']
            mask = transformed['mask']   

        image = image.transpose((2, 0, 1))#(H,W,3)->(3,H,W)   
        return image, mask, y


    
if __name__ == '__main__':
    import albumentations as A
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader    

    transforms = A.Compose([A.Resize(height=128, width=128, p=1)], p=1)

    # 测试目录   ISBI2016_ISIC_Part1_Training_Data\ISBI2016_ISIC_Part1_Training_Data../../datasets/ISIC2016/without label'
    data_roots = ['ISBI2016_ISIC_Part1_Training_Data\ISBI2016_ISIC_Part1_Training_Data']
    dataset = MTDataset(data_roots,transforms=transforms)
    
    for k in range(len(dataset)):
        image, mask, y = dataset[k]
        print(image.shape, mask.shape)
        print(np.unique(mask))

        image = image.transpose((1,2,0))
        plt.subplot(1,2,1) 
        plt.imshow(image)
        plt.subplot(1,2,2) 
        plt.imshow(mask,cmap='gray')
        
        plt.show()

