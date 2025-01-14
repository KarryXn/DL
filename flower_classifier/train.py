import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=102): 
        super(FlowerClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=224*224*3, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=102)
        )
    
    def forward(self, xs):
        xs = xs.view(xs.size(0), -1) 
        ys_logit = self.network(xs)
        return ys_logit
    
    if __name__ == '__main__':
        model = FlowerClassifier(num_classes=32)
        xs= torch.randn(128,3,224,224)
        ys_logit = model(xs)
        print(ys_logit.shape)

    if __name__ == '__main__':
        transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        #准备数据集 train, test, val
        train_set = torchvision.datasets.Flowers102(
            root=r'D:\dataset',
            split = 'train',
            transform = transform,
            download = True
        )
        test_set = torchvision.datasets.Flowers102(
            root=r'D:\dataset',
            split = 'test',
            transform = transform,
            download = True
        )

        val_set = torchvision.datasets.Flowers102(
            root=r'D:\dataset',
            split = 'val',
            transform = transform,
            download = True
        )
        print(len(train_set))
        print(len(val_set))

        train_dataloader = torch.utils.data.DataLoader(dataset = train_set, 
                                                    batch_size = 102, 
                                                    shuffle =True, #是否打乱样本
                                                    num_workers =2)#子进程个数，加快进程读取
        test_dataloader = torch.utils.data.DataLoader(dataset =test_set, batch_size = 102, shuffle =True, num_workers =2)
        val_dataloader =torch.utils.data.DataLoader(dataset =val_set, batch_size = 102, shuffle =True, num_workers =2)
        model = FlowerClassifier(num_classes=32)
        device  ='cpu'
        model.to(device)
        #初始化损失函数
        loss_ce = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.001,momentum=0.9)
        #循环优化模型
        num_epochs =10


        for epoch in range(num_epochs):
            print('Epoch: #',epoch+1)
            running_loss = 0.0
            model.train()

            for xs, labels in train_dataloader:
                #注意数据tensor和model必须在同一个设备
                xs = xs.to(device)
                xs=xs.reshape(-1,224*224*3)
                labels = labels.to(device)

                ys_logit = model(xs)

                ce_loss = loss_ce(ys_logit,labels)
                running_loss += ce_loss.item()
                optimizer.zero_grad()

                ce_loss.backward()
                optimizer.step()
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}') 
            # print('Start validation...')
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xs,labels in val_dataloader:
                    xs = xs.to(device)
                    xs=xs.reshape(-1,224*224*3)
                    labels = labels.to(device)

                    ys_logit = model(xs)
                    pred_labels = torch.argmax(ys_logit,dim=-1)
                    total +=labels.size(0)
                    correct += (pred_labels==labels).sum().item()
                

                # torch.save(model.state_dict(), 'flower_classifier.pth')
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f},Validation Accuracy: {accuracy:.2f}%') 

