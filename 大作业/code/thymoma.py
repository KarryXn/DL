# import os
# import torch
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms
# from PIL import Image
# import segmentation_models_pytorch as smp

# class ThymomaDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.image_files = sorted([f for f in os.listdir(data_dir) if not f.endswith('_mask.jpg')])
#         self.mask_files = [f.replace('.jpg', '_mask.jpg') for f in self.image_files]

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.data_dir, self.image_files[idx])
#         mask_path = os.path.join(self.data_dir, self.mask_files[idx])
#         image = Image.open(img_path).convert('L')
#         mask = Image.open(mask_path).convert('L')

#         if self.transform:
#             image = self.transform(image)
#             mask = self.transform(mask)

#         return image, mask

# def get_unetplusplus():
#     model = smp.UnetPlusPlus(
#         encoder_name="resnet34",
#         encoder_weights="imagenet",
#         in_channels=1,
#         classes=1,
#     )
#     return model

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

class ThymomaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(data_dir) if not f.endswith('_mask.jpg')])
        self.mask_files = [f.replace('.jpg', '_mask.jpg') for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        mask_path = os.path.join(self.data_dir, self.mask_files[idx])
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def get_unetplusplus():
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )
    return model