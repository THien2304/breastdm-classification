import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict

# ==============================================================================
# 1. Các Hàm Hỗ trợ và Lớp Chính Quy hóa (Regularization)
# ==============================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Kỹ thuật Stochastic Depth (DropPath)
    Giúp huấn luyện các mô hình rất sâu bằng cách bỏ qua ngẫu nhiên các Block.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # Tạo mask ngẫu nhiên (chỉ giữ lại 1 nếu giá trị ngẫu nhiên < keep_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (N, 1, 1, 1) or (N, 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask = random_tensor.floor_()
    # Nhân với mask và scale lại
    output = x.div(keep_prob) * mask
    return output

class DropPath(nn.Module):
    """Lớp bọc cho hàm drop_path"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# ==============================================================================
# 2. Khối Nhúng Miếng Vá (Patch Embedding)
# ==============================================================================

class PatchEmbed(nn.Module):
    """
    Lớp chuyển đổi ảnh 2D thành chuỗi tokens (miếng vá)
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        # Lớp tích chập đóng vai trò như phép chiếu tuyến tính (linear projection)
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # Chuyển đổi từ (B, C, H, W) -> (B, embed_dim, num_patches_h, num_patches_w)
        x = self.proj(x)
        # Làm phẳng: (B, embed_dim, N_h, N_w) -> (B, embed_dim, N_patches)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)
        x = self.norm(x)
        return x

# ==============================================================================
# 3. Khối MLP (Feed-Forward)
# ==============================================================================

class Mlp(nn.Module):
    """
    Khối mạng nơ-ron truyền thẳng (Feed-Forward Network)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop_ratio=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ==============================================================================
# 4. Cơ chế Attention (Tự Chú ý)
# ==============================================================================

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Lớp tuyến tính để tạo Q, K, V (Query, Key, Value)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        # 1. Tính Q, K, V: (B, N, C) -> (B, N, 3 * C) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # q, k, v có shape (B, num_heads, N, head_dim)

        # 2. Tính Ma trận Tương đồng (Attention Score)
        # q @ k.transpose(-2, -1): (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 3. Tính Đầu ra (Attention Output)
        # attn @ v: (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 4. Phép Chiếu Tuyến tính Cuối cùng
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ==============================================================================
# 5. Khối Transformer Encoder (Block)
# ==============================================================================

class Block(nn.Module):
    """
    Khối Encoder cơ bản của Transformer (bao gồm MSA và MLP)
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0.,
                 drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio
        )
        # Stochastic depth
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_ratio=drop_ratio)

    def forward(self, x):
        # 1. Multi-Head Self-Attention (MSA) với Residual Connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # 2. Feed-Forward Network (MLP) với Residual Connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x