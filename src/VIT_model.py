"""
ViT-7 BreastDM
Refactor A+ + Pretrained support

Original ViT code from:
https://github.com/rwightman/pytorch-image-models
"""

from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

# =========================================================
# DropPath
# =========================================================
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# =========================================================
# Patch Embedding
# =========================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_c, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)               # [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)
        return x


# =========================================================
# Attention
# =========================================================
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# =========================================================
# MLP
# =========================================================
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# =========================================================
# Transformer Block
# =========================================================
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =========================================================
# Vision Transformer – ViT-7
# =========================================================
class ViT7_BreastDM(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=7,
        num_heads=12,
        num_classes=2
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size, patch_size, 3, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        dpr = torch.linspace(0, 0.1, depth)
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, drop_path=dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        feat = self.forward_features(x)
        return self.head(feat)


# =========================================================
# Pretrained loader (ViT-B/16 → ViT-7)
# =========================================================
def load_pretrained_vit7(model, pretrained_path):
    """
    Load ViT-B/16 pretrained weights
    Keep only first 7 transformer blocks
    """
    checkpoint = torch.load(pretrained_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    model_dict = model.state_dict()
    load_dict = {}

    for k, v in state_dict.items():
        if "head" in k:
            continue

        if k.startswith("blocks."):
            block_id = int(k.split(".")[1])
            if block_id >= 7:
                continue

        if k in model_dict and model_dict[k].shape == v.shape:
            load_dict[k] = v

    model_dict.update(load_dict)
    model.load_state_dict(model_dict)

    print(f"✅ Pretrained ViT-B/16 loaded (first 7 blocks)")
