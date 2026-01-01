from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn
import torch

# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class DerivedResNet(ResNet):
    def __init__(self):
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

class LateFusionModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        weights = ResNet18_Weights.DEFAULT
        standard_resnet = resnet18(weights=weights)
        state_dict = standard_resnet.state_dict()

        self.core_model = DerivedResNet()
        self.core_model.load_state_dict(state_dict, strict=False)
        
        # layer4 has 512 features * 2 streams
        self.fc = nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        
        feat1 = self.core_model(x1)
        feat2 = self.core_model(x2)

        # fusion
        combined = torch.cat((feat1, feat2), dim=1)
        return self.fc(combined)