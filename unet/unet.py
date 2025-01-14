import torch.nn as nn
import torch
import torch.nn.functional as F
class UNet(nn.Module):
    def __init__(self,in_channels = 3,num_classes =2,base_num_filters =16):
        super(UNet,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,base_num_filters,kernel_size=3,padding=1) #（N，3，128，128）-》（N，16，128，128）
        #max pooling
        self.conv2 = nn.Conv2d(base_num_filters,base_num_filters*2,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(base_num_filters*2,base_num_filters*4,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(base_num_filters*4,base_num_filters*8,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(base_num_filters*8,base_num_filters*16,kernel_size=3,padding=1)

        self.upconv4 = nn.ConvTranspose2d(base_num_filters*16,base_num_filters*8,kernel_size=2,stride=2)
        self.conv6 = nn.Conv2d(base_num_filters*16,base_num_filters*8,kernel_size=3,padding=1)

        self.upconv3 = nn.ConvTranspose2d(base_num_filters*8,base_num_filters*4,kernel_size=2,stride=2)
        self.conv7 = nn.Conv2d(base_num_filters*8,base_num_filters*4,kernel_size=3,padding=1)

        self.upconv2 = nn.ConvTranspose2d(base_num_filters*4,base_num_filters*2,kernel_size=2,stride=2)
        self.conv8 = nn.Conv2d(base_num_filters*4,base_num_filters*2,kernel_size=3,padding=1)

        self.upconv1 = nn.ConvTranspose2d(base_num_filters*2,base_num_filters,kernel_size=2,stride=2)
        self.conv9 = nn.Conv2d(base_num_filters*2,base_num_filters,kernel_size=3,padding=1)
        
        self.conv10 = nn.Conv2d(base_num_filters,num_classes,kernel_size=1)

    def forward(self, x):
        # Encoding path
        c1 = F.relu(self.conv1(x))  # (N, 16, 128, 128)
        p1 = F.max_pool2d(c1, kernel_size=2, stride=2)  # (N, 16, 64, 64)
        
        c2 = F.relu(self.conv2(p1))  # (N, 32, 64, 64)
        p2 = F.max_pool2d(c2, kernel_size=2, stride=2)  # (N, 32, 32, 32)
        
        c3 = F.relu(self.conv3(p2))  # (N, 64, 32, 32)
        p3 = F.max_pool2d(c3, kernel_size=2, stride=2)  # (N, 64, 16, 16)
        
        c4 = F.relu(self.conv4(p3))  # (N, 128, 16, 16)
        p4 = F.max_pool2d(c4, kernel_size=2, stride=2)  # (N, 128, 8, 8)
        
        c5 = F.relu(self.conv5(p4))  # (N, 256, 8, 8)

        # Decoding path
        up4 = self.upconv4(c5)  # (N, 128, 16, 16)
        up4 = torch.cat([up4, c4], dim=1)  # (N, 256, 16, 16)
        c6 = F.relu(self.conv6(up4))  # (N, 128, 16, 16)

        up3 = self.upconv3(c6)  # (N, 64, 32, 32)
        up3 = torch.cat([up3, c3], dim=1)  # (N, 128, 32, 32)
        c7 = F.relu(self.conv7(up3))  # (N, 64, 32, 32)

        up2 = self.upconv2(c7)  # (N, 32, 64, 64)
        up2 = torch.cat([up2, c2], dim=1)  # (N, 64, 64, 64)
        c8 = F.relu(self.conv8(up2))  # (N, 32, 64, 64)

        up1 = self.upconv1(c8)  # (N, 16, 128, 128)
        up1 = torch.cat([up1, c1], dim=1)  # (N, 32, 128, 128)
        c9 = F.relu(self.conv9(up1))  # (N, 16, 128, 128)

        out = self.conv10(c9)  # (N, num_classes, 128, 128)
        return out


if __name__=='__main__':
    model =UNet(in_channels=3,num_classes=2,base_num_filters=16)
    batch =torch.randn((1,3,128,128))
    out = model(batch)
    print(out.shape)          
