
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import pretrainedmodels.models as premodels
from pretrainedmodels.models.resnext_features import (
    resnext101_32x4d_features
)



# 1. VGG16
class VGG16(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16, self).__init__()
        model = models.vgg16(pretrained=True)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 2. ResNet family
class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet101(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 3. DenseNet
class DenseNet169(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = models.densenet169(pretrained=True)
        self.features = model.features
        self.fc = nn.Linear(1664, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DenseNet201(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = models.densenet201(pretrained=True)
        self.features = model.features
        self.fc = nn.Linear(1920, num_classes)
        self.feature_map = None  # for CAM

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        self.feature_map = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def cam_out(self):
        return self.feature_map



# 4. ResNeXt101
class ResNeXt101(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = resnext101_32x4d_features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



# 5. SENet
class SENet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = premodels.se_resnet50()
        self.features = nn.Sequential(
            model.layer0,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        self.avgpool = model.avg_pool
        self.dropout = model.dropout
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    model = ResNet50(num_classes=2)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
