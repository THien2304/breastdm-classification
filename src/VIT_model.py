import torch
import torch.nn as nn
import timm

# ---------------- DropPath (Stochastic Depth) ----------------
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

# ---------------- Patch Embedding ----------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

# ---------------- Multi-head Attention ----------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ---------------- MLP ----------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---------------- Transformer Block ----------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ---------------- Full ViT-B/16 for BreastDM ----------------
class ViT_BreastDM(nn.Module):
    def __init__(self, num_classes=2, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        dpr = torch.linspace(0, drop_path_rate, depth)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token,std=0.02)
        nn.init.trunc_normal_(self.pos_embed,std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight,std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:,0]  # CLS token

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)

# ---------------- Load pretrained ViT-B/16 ----------------
def load_pretrained_vit_full(model):
    print("🔹 Loading full pretrained ViT-B/16 from timm ...")
    pretrained = timm.create_model("vit_base_patch16_224", pretrained=True)
    pretrained_state = pretrained.state_dict()
    model_state = model.state_dict()
    load_dict = {}

    for k,v in pretrained_state.items():
        # skip head only
        if "head" in k:
            continue
        if k in model_state and model_state[k].shape == v.shape:
            load_dict[k] = v

    model_state.update(load_dict)
    model.load_state_dict(model_state)
    print(f"✅ Loaded {len(load_dict)} layers from full pretrained ViT-B/16")
