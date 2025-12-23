

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

# 1. VGG16
class VGG16(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
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
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 2. ResNet Family
class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet101(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



# 3. DenseNet Family
class DenseNet169(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        self.features = model.features
        self.fc = nn.Linear(model.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DenseNet201(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        self.features = model.features
        self.fc = nn.Linear(model.classifier.in_features, num_classes)
        self.feature_map = None  # for CAM

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        self.feature_map = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def cam_out(self):
        return self.feature_map


# 4. ResNeXt101
class ResNeXt101(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = timm.create_model(
            "resnext101_32x4d",
            pretrained=True
        )
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 5. SENet50 
class SENet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        model = timm.create_model(
            "seresnet50",
            pretrained=True
        )
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

