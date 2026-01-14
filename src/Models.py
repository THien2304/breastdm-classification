import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

# ---------------- VGG16 ----------------
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
        return self.classifier(x)


# ---------------- ResNet ----------------
class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.fc = nn.Linear(m.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(torch.flatten(x, 1))


class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.fc = nn.Linear(m.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(torch.flatten(x, 1))


class ResNet101(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        m = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.fc = nn.Linear(m.fc.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(torch.flatten(x, 1))


# ---------------- DenseNet ----------------
class DenseNet169(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        m = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        self.features = m.features
        self.fc = nn.Linear(m.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.fc(torch.flatten(x, 1))


class DenseNet201(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        m = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        self.features = m.features
        self.fc = nn.Linear(m.classifier.in_features, num_classes)
        self.feature_map = None

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        self.feature_map = x
        x = F.adaptive_avg_pool2d(x, 1)
        return self.fc(torch.flatten(x, 1))

    def cam_out(self):
        return self.feature_map


# ---------------- ResNeXt ----------------
class ResNeXt101(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        m = timm.create_model("resnext101_32x4d", pretrained=True)
        self.backbone = m.forward_features
        self.fc = nn.Linear(m.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)


# ---------------- SENet ----------------
class SENet50(nn.Module):
    def __init__(self, num_classes=2, dropout_p=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            "seresnet50",
            pretrained=True,
            num_classes=0,
            global_pool="avg"   
        )
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)
    def forward(self, x):
        x = self.backbone(x)     
        x = self.dropout(x)
        x = self.fc(x)
        return x
