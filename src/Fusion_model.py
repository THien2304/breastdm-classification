import torch
import torch.nn as nn
import torch.nn.functional as F

from Models import SENet50
from VIT_model import ViT_BreastDM


# FCUUp: ViT tokens -> CNN feature map
class FCUUp(nn.Module):
    def __init__(self, in_dim=768, out_dim=512, up_stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 1)
        self.bn = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.up_stride = up_stride

    def forward(self, x, H, W):
        """
        x: [B, N+1, C] (ViT output)
        """
        B, N, C = x.shape
        x = x[:, 1:]                       # remove CLS
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.act(self.bn(self.conv(x)))
        return F.interpolate(
            x, scale_factor=self.up_stride,
            mode="bilinear", align_corners=False
        )


# Cross Non-Local Block (2D)
class CrossNonLocal2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        inter = in_channels // 2

        self.theta = nn.Conv2d(in_channels, inter, 1)
        self.phi   = nn.Conv2d(in_channels, inter, 1)
        self.g     = nn.Conv2d(in_channels, inter, 1)

        self.out = nn.Sequential(
            nn.Conv2d(inter, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )

        # identity initialization (Non-Local paper)
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, x_this, x_other):
        B, C, H, W = x_this.shape

        theta = self.theta(x_this).view(B, -1, H * W)
        phi   = self.phi(x_other).view(B, -1, H * W)
        g     = self.g(x_other).view(B, -1, H * W)

        attn = torch.matmul(theta.transpose(1, 2), phi)
        attn = F.softmax(attn, dim=-1)

        y = torch.matmul(attn, g.transpose(1, 2))
        y = y.transpose(1, 2).view(B, -1, H, W)

        return x_this + self.out(y)


# Fusion Model
class FusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # -------- CNN branch --------
        self.cnn = SENet50(num_classes=num_classes)
        self.cnn_backbone = self.cnn.backbone   # forward_features
        cnn_channels = 512

        # -------- ViT branch --------
        self.vit = ViT_BreastDM(num_classes=num_classes)

        # -------- ViT -> CNN --------
        self.fcu_up = FCUUp(
            in_dim=768,
            out_dim=cnn_channels,
            up_stride=2
        )

        # -------- Cross Fusion --------
        self.nl_cnn = CrossNonLocal2D(cnn_channels)
        self.nl_vit = CrossNonLocal2D(cnn_channels)

        # -------- Head --------
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cnn_channels * 2, num_classes)

    def forward(self, x):
        # ===== CNN =====
        cnn_feat = self.cnn_backbone(x)          # [B, 512, 28, 28]

        # ===== ViT =====
        vit_tokens = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.size(0), -1, -1)
        vit_tokens = torch.cat((cls_token, vit_tokens), dim=1)
        vit_tokens = self.vit.pos_drop(vit_tokens + self.vit.pos_embed)
        vit_tokens = self.vit.blocks(vit_tokens)
        vit_tokens = self.vit.norm(vit_tokens)

        vit_feat = self.fcu_up(vit_tokens, 14, 14)

        # ===== Cross Non-Local =====
        cnn_att = self.nl_cnn(cnn_feat, vit_feat)
        vit_att = self.nl_vit(vit_feat, cnn_feat)

        # ===== Fusion =====
        fusion = torch.cat([cnn_att, vit_att], dim=1)
        fusion = self.avgpool(fusion).flatten(1)
        out = self.fc(fusion)

        return out


