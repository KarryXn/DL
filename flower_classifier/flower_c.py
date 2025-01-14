import torch
import torch.nn as nn


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