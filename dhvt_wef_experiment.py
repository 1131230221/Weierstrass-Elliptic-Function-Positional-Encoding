import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Dict
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import random
import os
import time
from functools import partial
from timm.layers import trunc_normal_, to_2tuple
from timm.models import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# ========== 从 W_Ti.py 导入 WEF 相关类 ==========
class WeierstrassEllipticFunction(nn.Module):
    """魏尔斯特拉斯椭圆函数计算器"""
    def __init__(self, 
                 g2: float = 1.0, 
                 g3: float = 0.0,
                 eps: float = 1e-8,
                 alpha_scale: float = 0.15):
        super().__init__()
        self.g2 = g2
        self.g3 = g3
        self.eps = eps
        self.alpha_scale = alpha_scale
        
        discriminant = g2**3 - 27*g3**2
        assert abs(discriminant) > eps, f"判别式太接近零: {discriminant}"
        
        if abs(g3) < eps:
            omega1_r = torch.tensor(2.62205755429212)
            omega1_i = torch.tensor(0.0)
            omega3_r = torch.tensor(0.0)
            omega3_i = torch.tensor(2.62205755429212)
        else:
            omega1_r = torch.tensor(1.0)
            omega1_i = torch.tensor(0.0)
            omega3_r = torch.tensor(0.0)
            omega3_i = torch.tensor(1.0)

        self.register_buffer('omega1_r', omega1_r)
        self.register_buffer('omega1_i', omega1_i)
        self.register_buffer('omega3_r', omega3_r)
        self.register_buffer('omega3_i', omega3_i)
    
    def _improved_series_sum(self, z: torch.Tensor, max_m: int = 8, max_n: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """改进的级数求和，增加数值稳定性"""
        z = z.to(torch.complex64)  # 使用float32以减少内存使用
        wp_sum = torch.zeros_like(z, dtype=torch.complex64)
        wp_prime_sum = torch.zeros_like(z, dtype=torch.complex64)
        
        omega1 = torch.complex(self.omega1_r, self.omega1_i)
        omega3 = torch.complex(self.omega3_r, self.omega3_i)
        
        lattice_points = []
        for m in range(-max_m, max_m + 1):
            for n in range(-max_n, max_n + 1):
                if m == 0 and n == 0:
                    continue
                w = 2 * m * omega1 + 2 * n * omega3
                lattice_points.append((abs(w), w, m, n))
        
        lattice_points.sort(key=lambda x: x[0].real if isinstance(x[0], torch.Tensor) else x[0])
        
        # 限制处理的格点数量避免内存问题
        lattice_points = lattice_points[:min(len(lattice_points), 64)]
        
        for _, w, m, n in lattice_points:
            try:
                if isinstance(w, torch.Tensor):
                    w = w.to(z.device)
                diff = z - w
                
                mask = torch.abs(diff) > self.eps * 20
                if mask.any():
                    diff_masked = diff[mask]
                    w_term = 1.0/w**2 if abs(w) > self.eps else 0.0
                    
                    # 添加数值稳定性检查
                    with torch.no_grad():
                        diff_abs = torch.abs(diff_masked)
                        valid_mask = (diff_abs > self.eps * 10) & (diff_abs < 1e3)
                        
                    if valid_mask.any():
                        diff_valid = diff_masked[valid_mask]
                        wp_term = 1.0/diff_valid**2 - w_term
                        wp_prime_term = -2.0/diff_valid**3
                        
                        # 更保守的裁剪
                        wp_term = torch.clamp(wp_term.real, -1e3, 1e3) + 1j * torch.clamp(wp_term.imag, -1e3, 1e3)
                        wp_prime_term = torch.clamp(wp_prime_term.real, -1e3, 1e3) + 1j * torch.clamp(wp_prime_term.imag, -1e3, 1e3)
                        
                        # 只更新有效位置
                        full_mask = torch.zeros_like(mask)
                        full_mask[mask] = valid_mask
                        wp_sum[full_mask] += wp_term
                        wp_prime_sum[full_mask] += wp_prime_term
                        
            except Exception as e:
                # 忽略数值计算错误，继续处理
                continue
        
        return wp_sum, wp_prime_sum
    
    def wp_and_wp_prime(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算Weierstrass椭圆函数及其导数，增强数值稳定性"""
        z = z.to(torch.complex64)  # 使用float32减少内存
        near_origin = torch.abs(z) < self.eps * 20
        
        wp = torch.zeros_like(z, dtype=torch.complex64)
        wp_prime = torch.zeros_like(z, dtype=torch.complex64)
        
        valid_mask = ~near_origin
        if valid_mask.any():
            try:
                z_valid = z[valid_mask]
                # 添加数值稳定性检查
                z_abs = torch.abs(z_valid)
                stable_mask = (z_abs > self.eps * 10) & (z_abs < 100)
                
                if stable_mask.any():
                    z_stable = z_valid[stable_mask]
                    wp_main = 1.0 / z_stable**2
                    wp_prime_main = -2.0 / z_stable**3
                    
                    # 只对稳定的值计算级数
                    wp_series, wp_prime_series = self._improved_series_sum(z_stable)
                    
                    # 更新对应位置
                    full_stable_mask = torch.zeros_like(valid_mask)
                    full_stable_mask[valid_mask] = stable_mask
                    wp[full_stable_mask] = wp_main + wp_series
                    wp_prime[full_stable_mask] = wp_prime_main + wp_prime_series
                    
            except Exception as e:
                # 如果计算失败，使用默认值
                pass
        
        # 对原点附近和不稳定区域使用较小的默认值
        large_value = 1e2  # 减小默认值
        invalid_mask = near_origin | ~torch.isfinite(wp.real) | ~torch.isfinite(wp.imag)
        wp[invalid_mask] = large_value
        wp_prime[invalid_mask] = large_value
        
        # 更保守的裁剪
        wp = torch.clamp(wp.real, -5e3, 5e3) + 1j * torch.clamp(wp.imag, -5e3, 5e3)
        wp_prime = torch.clamp(wp_prime.real, -5e3, 5e3) + 1j * torch.clamp(wp_prime.imag, -5e3, 5e3)
        
        return wp, wp_prime


class WEFPositionalEncoding(nn.Module):
    """基于WEF的位置编码模块，适配DHVT架构"""
    def __init__(self, 
                 d_model: int,
                 max_h: int = 14,
                 max_w: int = 14,
                 g2: float = 1.0,
                 g3: float = 0.0,
                 projection_type: str = 'linear',
                 alpha_scale: float = 0.15,
                 learnable_alpha: bool = True,
                 use_4d: bool = True):
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        self.use_4d = use_4d
        
        if learnable_alpha:
            self.log_alpha_scale = nn.Parameter(torch.log(torch.tensor(alpha_scale)))
        else:
            self.register_buffer('log_alpha_scale', torch.log(torch.tensor(alpha_scale)))
        
        self.alpha_learn = nn.Parameter(torch.tensor(0.12))
        
        self.wef = WeierstrassEllipticFunction(
            g2=g2, 
            g3=g3, 
            alpha_scale=alpha_scale
        )
        
        input_dim = 4 if use_4d else 2
        if projection_type == 'mlp':
            self.projection = nn.Sequential(
                nn.Linear(input_dim, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(0.08),  # 优化: 轻微增加dropout
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model)
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model)
            )
        
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.015)
        self.pos_scale = nn.Parameter(torch.ones(1) * 0.8)
    
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """为DHVT生成位置编码"""
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        omega_3_prime = F.softplus(self.alpha_learn).clamp(0.02, 8.0)
        omega_3 = 1j * omega_3_prime
        
        position_encodings = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # CLS token编码
        position_encodings[:, 0:1, :] = self.cls_pos_embedding
        
        # 生成patch坐标
        row_idx = torch.arange(h, device=device).unsqueeze(1).repeat(1, w).reshape(-1)
        col_idx = torch.arange(w, device=device).repeat(h)
        
        u = (col_idx.float() + 0.5) / w
        v = (row_idx.float() + 0.5) / h
        
        z_real = u * (2 * self.wef.omega1_r) * 0.4
        z_imag = v * (2 * omega_3_prime) * 0.4
        z = torch.complex(z_real, z_imag)
        
        wp, wp_prime = self.wef.wp_and_wp_prime(z)
        
        alpha_scale = torch.exp(self.log_alpha_scale).clamp(0.002, 0.8)
        wp_real_compressed = torch.tanh(alpha_scale * wp.real)
        wp_imag_compressed = torch.tanh(alpha_scale * wp.imag)
        
        if self.use_4d:
            wp_prime_real_compressed = torch.tanh(alpha_scale * wp_prime.real)
            wp_prime_imag_compressed = torch.tanh(alpha_scale * wp_prime.imag)
            features = torch.stack([
                wp_real_compressed,
                wp_imag_compressed,
                wp_prime_real_compressed,
                wp_prime_imag_compressed
            ], dim=1).float()
        else:
            features = torch.stack([
                wp_real_compressed,
                wp_imag_compressed
            ], dim=1).float()
        
        patch_encodings = self.projection(features)
        position_encodings[:, 1:1+h*w, :] = patch_encodings.unsqueeze(0).expand(batch_size, -1, -1)
        
        position_encodings = position_encodings * self.pos_scale
        
        return position_encodings


# ========== 从 vision_transformer.py 导入 DHVT 相关类 ==========
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class DAFF(nn.Module):
    """Dynamic Attention-based Feed-Forward"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 kernel_size=3, with_bn=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=hidden_features)
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.bn2 = nn.BatchNorm2d(hidden_features)
        self.bn3 = nn.BatchNorm2d(out_features)
        
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Linear(in_features, in_features//4)
        self.excitation = nn.Linear(in_features//4, in_features)
                
    def forward(self, x):
        B, N, C = x.size()
        cls_token, tokens = torch.split(x, [1, N - 1], dim=1)
        x = tokens.reshape(B, int(math.sqrt(N - 1)), int(math.sqrt(N - 1)), C).permute(0, 3, 1, 2).contiguous()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        shortcut = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = shortcut + x

        x = self.conv3(x)
        x = self.bn3(x)

        weight = self.squeeze(x).flatten(1).reshape(B, 1, C)
        weight = self.excitation(self.act(self.compress(weight)))
        cls_token = cls_token * weight
        
        tokens = x.flatten(2).permute(0, 2, 1)
        out = torch.cat((cls_token, tokens), dim=1)
        
        return out


class HI_Attention(nn.Module):
    """Hierarchical Interactive Multi-Head Self-Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.act = nn.GELU()
        self.ht_proj = nn.Linear(dim//self.num_heads, dim, bias=True)
        self.ht_norm = nn.LayerNorm(dim//self.num_heads)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_heads, dim))
        trunc_normal_(self.pos_embed, std=.02)
    
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N-1))

        head_pos = self.pos_embed.expand(x.shape[0], -1, -1)
        x_ = x.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        x_ = x_.mean(dim=2)
        x_ = self.ht_proj(x_).reshape(B, -1, self.num_heads, C // self.num_heads)
        x_ = self.act(self.ht_norm(x_)).flatten(2)
        x_ = x_ + head_pos
        x = torch.cat([x, x_], dim=1)
        
        qkv = self.qkv(x).reshape(B, N+self.num_heads, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N+self.num_heads, C)
        x = self.proj(x)
        
        cls, patch, ht = torch.split(x, [1, N-1, self.num_heads], dim=1)
        cls = cls + torch.mean(ht, dim=1, keepdim=True)
        x = torch.cat([cls, patch], dim=1)

        x = self.proj_drop(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
        nn.BatchNorm2d(out_planes)
    )


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1, 1]), requires_grad=True)

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x


class ConvPatchEmbed(nn.Module):
    """Stacked Overlapping Patch Embedding (SOPE)"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, init_values=1e-2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        if patch_size[0] == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 8, 2),
                nn.GELU(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 4:  
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 2:  
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim, 2),
                nn.GELU(),
            )
        else:
            raise ValueError("For convolutional projection, patch size has to be in [2, 4, 16]")
        self.pre_affine = Affine(3)
        self.post_affine = Affine(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape 
        
        x = self.pre_affine(x)
        x = self.proj(x)
        x = self.post_affine(x)

        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        return x


class DHVT_WEF_Block(nn.Module):
    """DHVT Block with WEF position encoding support"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = HI_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DAFF(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, kernel_size=3)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DHVT_WEF_Ti(nn.Module):
    """DHVT-Tiny with WEF Positional Encoding for CIFAR-100"""
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 100,
                 embed_dim: int = 192,
                 depth: int = 12,
                 num_heads: int = 3,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.05,
                 drop_path: float = 0.05,
                 # WEF相关参数
                 wef_g2: float = 1.0,
                 wef_g3: float = 0.0,
                 wef_alpha_scale: float = 0.15,
                 wef_projection_type: str = 'linear',
                 wef_use_4d: bool = True):
        super().__init__()
        
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 使用DHVT的SOPE
        self.patch_embed = ConvPatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_channels, 
            embed_dim=embed_dim
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # WEF位置编码
        self.pos_encoding = WEFPositionalEncoding(
            d_model=embed_dim,
            max_h=img_size // patch_size,
            max_w=img_size // patch_size,
            g2=wef_g2,
            g3=wef_g3,
            projection_type=wef_projection_type,
            alpha_scale=wef_alpha_scale,
            learnable_alpha=True,
            use_4d=wef_use_4d
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        # DHVT blocks
        self.blocks = nn.ModuleList([
            DHVT_WEF_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=dropout,
                attn_drop=0.0,
                drop_path=dpr[i],
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm
            ) for i in range(depth)
        ])
        
        # Output
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding using SOPE
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, 1 + num_patches, embed_dim]
        
        # Add WEF position encoding
        h = w = int(math.sqrt(self.num_patches))
        pos_encoding = self.pos_encoding(x, h, w)
        x = x + pos_encoding
        
        x = self.dropout(x)
        
        # DHVT blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.norm(x)
        x = x[:, 0]  # CLS token
        x = self.head(x)
        
        return x


# ========== 数据增强和训练相关函数 ==========
class Cutout:
    """Cutout augmentation - 优化版"""
    def __init__(self, n_holes=1, length=16):  # 优化: 增加cutout强度
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        
        for _ in range(self.n_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            
            mask[y1:y2, x1:x2] = 0.0
        
        mask = mask.expand_as(img)
        img = img * mask
        
        return img


def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_cifar100_dataloaders(batch_size=96, num_workers=4):  # 减少batch size和workers
    """创建CIFAR-100数据加载器"""
    # 数据增强 - 针对DHVT优化
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),  # DHVT使用224x224输入
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                           std=[0.2673, 0.2564, 0.2762]),
        Cutout(n_holes=1, length=16)  # 优化: 与类定义保持一致
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                           std=[0.2673, 0.2564, 0.2762])
    ])
    
    # 数据集
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # 数据加载器 - 简化设置避免段错误
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
        # 移除persistent_workers避免内存问题
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
        # 移除persistent_workers避免内存问题
    )
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, num_epochs=300,
                initial_lr=0.001, warmup_epochs=10, use_amp=True, use_mixup=True):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化多GPU使用 - 添加错误处理
    if torch.cuda.device_count() > 1:
        try:
            model = nn.DataParallel(model)
            print(f"Using {torch.cuda.device_count()} GPUs for training")
        except Exception as e:
            print(f"DataParallel failed: {e}, using single GPU")
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 优化器 - 使用AdamW，优化权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.03)  # 优化: 降低权重衰减
    
    # 学习率调度器 - Cosine with warmup
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = warmup_epochs * len(train_loader)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 损失函数 - 使用标签平滑，优化平滑程度
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  # 优化: 增加标签平滑
    scaler = GradScaler() if use_amp else None
    
    best_acc = 0.0
    start_epoch = 0
    checkpoint_path = 'checkpoint_dhvt_wef_ti_v2.pth'  # 优化版文件名
    log_path = 'train_log_dhvt_wef_ti_v2.txt'          # 优化版日志
    
    # 加载检查点
    if os.path.exists(checkpoint_path):
        try:
            print(f"=> Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if use_amp and 'scaler' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('best_acc', 0.0)
            print(f"=> Resumed from epoch {checkpoint['epoch']}, best_acc={best_acc:.2f}%")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}, starting from scratch")
            start_epoch = 0
            best_acc = 0.0
    
    # 日志文件头
    if start_epoch == 0 and (not os.path.exists(log_path) or os.path.getsize(log_path) == 0):
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write('epoch,batch,phase,loss,acc,lr,time\n')
    
    # 梯度裁剪值
    max_grad_norm = 1.0
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        try:
            for batch_idx, (images, labels) in enumerate(train_loader):
                batch_start = time.time()
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                # Mixup - 优化混合强度
                if use_mixup and random.random() > 0.3:
                    mixup_alpha = 0.6 if epoch >= warmup_epochs else 0.3  # 优化: 增强mixup强度
                    images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                    
                    optimizer.zero_grad()
                    if use_amp:
                        with autocast('cuda'):
                            outputs = model(images)
                            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(images)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                else:
                    optimizer.zero_grad()
                    if use_amp:
                        with autocast('cuda'):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                batch_acc = 100.0 * predicted.eq(labels).sum().item() / labels.size(0)
                batch_time = time.time() - batch_start
                lr = optimizer.param_groups[0]["lr"]
                
                # 每50个batch打印一次
                if batch_idx % 50 == 0:
                    log_str = (f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                               f'Loss: {loss.item():.4f} Acc: {batch_acc:.2f}% LR: {lr:.6f} Time: {batch_time:.2f}s')
                    print(log_str)
                    
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f'{epoch+1},{batch_idx},train,{loss.item():.4f},{batch_acc:.2f},{lr:.6f},{batch_time:.2f}\n')
                
                # 定期清理GPU缓存
                if batch_idx % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"Training error at epoch {epoch+1}, batch {batch_idx}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        
        # 验证
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        val_start = time.time()
        
        try:
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
        except Exception as e:
            print(f"Validation error at epoch {epoch+1}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        
        # 统计
        train_acc = 100.0 * train_correct / max(train_total, 1)
        test_acc = 100.0 * test_correct / max(test_total, 1)
        epoch_time = time.time() - epoch_start
        
        summary_str = (f'Epoch [{epoch+1}/{num_epochs}] '
                       f'Train Loss: {train_loss/len(train_loader):.4f} Train Acc: {train_acc:.2f}% '
                       f'Test Loss: {test_loss/len(test_loader):.4f} Test Acc: {test_acc:.2f}% '
                       f'Epoch Time: {epoch_time:.2f}s')
        print(summary_str)
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'{epoch+1},-1,summary,{train_loss/len(train_loader):.4f},{train_acc:.2f},{lr:.6f},{epoch_time:.2f}\n')
            f.write(f'{epoch+1},-1,summary_val,{test_loss/len(test_loader):.4f},{test_acc:.2f},-,{epoch_time:.2f}\n')
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            try:
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), 'best_dhvt_wef_ti_v2.pth')  # 优化版模型
                else:
                    torch.save(model.state_dict(), 'best_dhvt_wef_ti_v2.pth')
                print(f'Best model saved with accuracy: {best_acc:.2f}%')
            except Exception as e:
                print(f"Failed to save best model: {e}")
        
        # 保存断点
        try:
            checkpoint = {
                'model': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }
            if use_amp and scaler is not None:
                checkpoint['scaler'] = scaler.state_dict()
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return model, best_acc


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    
    # 设置cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 创建数据加载器 - 使用更安全的设置
    train_loader, test_loader = create_cifar100_dataloaders(batch_size=96, num_workers=4)
    
    # 创建优化版DHVT-WEF-Ti模型
    model = DHVT_WEF_Ti(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=100,
        embed_dim=224,      # 优化: 增加模型容量 (192→224)
        depth=12,           # 12层
        num_heads=4,        # 优化: 对应调整注意力头数 (3→4)
        mlp_ratio=4.0,
        dropout=0.08,       # 优化: 适度增加dropout (0.05→0.08)
        drop_path=0.15,     # 优化: 增加随机深度 (0.1→0.15)
        # WEF参数优化
        wef_g2=0.8,         # 优化: 调整椭圆函数参数 (1.0→0.8)
        wef_g3=0.2,         # 优化: 引入g3参数 (0.0→0.2)
        wef_alpha_scale=0.10, # 优化: 降低压缩强度 (0.15→0.10)
        wef_projection_type='mlp', # 优化: 使用更强的MLP投影
        wef_use_4d=True
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 训练模型 - 优化版参数
    model, best_acc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=300,
        initial_lr=0.0015,   # 优化: 提高初始学习率 (0.001→0.0015)
        warmup_epochs=15,    # 优化: 延长预热期 (10→15)
        use_amp=True,
        use_mixup=True
    )
    
    print(f"DHVT-WEF-Ti v2 (Optimized) Training completed! Best accuracy: {best_acc:.2f}%")
    print(f"Target baseline DHVT-T: 83.54%")
    print(f"Improvement over baseline: {best_acc - 83.54:.2f}%")
    print(f"Improvement over v1 (81.22%): {best_acc - 81.22:.2f}%")
    
    # 优化效果分析
    if best_acc > 83.54:
        print(f"🎉 SUCCESS! Surpassed DHVT-T baseline by {best_acc - 83.54:.2f}%")
    elif best_acc > 82.5:
        print(f"📈 GOOD PROGRESS! Close to baseline, gap: {83.54 - best_acc:.2f}%")
    else:
        print(f"📊 NEEDS MORE TUNING! Gap to baseline: {83.54 - best_acc:.2f}%")


if __name__ == "__main__":
    main()
