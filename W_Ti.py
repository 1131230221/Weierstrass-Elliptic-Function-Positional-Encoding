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


class WeierstrassEllipticFunction:
    """
    改进的魏尔斯特拉斯椭圆函数计算器，增强数值稳定性
    """
    def __init__(self, 
                 g2: float = 1.0, 
                 g3: float = 0.0,
                 eps: float = 1e-8,
                 alpha_scale: float = 0.15,  # 为ViT-Ti调整缩放因子
                 device: torch.device = None):
        """
        参数:
            g2: 椭圆不变量g₂
            g3: 椭圆不变量g₃
            eps: 数值稳定性小量
            alpha_scale: tanh压缩的缩放因子
            device: 计算设备
        """
        self.g2 = g2
        self.g3 = g3
        self.eps = eps
        self.alpha_scale = alpha_scale
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 计算判别式
        discriminant = g2**3 - 27*g3**2
        assert abs(discriminant) > eps, f"判别式太接近零: {discriminant}"
        
        # 对于g3=0的双纽线情况，使用精确的半周期值
        if abs(g3) < eps:
            # 精确值：omega1 = Gamma(1/4)^2 / sqrt(2*pi)
            self.omega1 = torch.tensor(2.62205755429212, device=self.device, dtype=torch.complex128)
            self.omega3 = torch.tensor(complex(0, 2.62205755429212), device=self.device, dtype=torch.complex128)
        else:
            # 一般情况需要数值计算周期
            self.omega1 = torch.tensor(complex(1.0, 0.0), device=self.device, dtype=torch.complex128)
            self.omega3 = torch.tensor(complex(0.0, 1.0), device=self.device, dtype=torch.complex128)
    
    def _improved_series_sum(self, z: torch.Tensor, max_m: int = 12, max_n: int = 12) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        改进的级数求和，为ViT-Ti优化计算复杂度
        """
        z = z.to(torch.complex128)
        wp_sum = torch.zeros_like(z, dtype=torch.complex128)
        wp_prime_sum = torch.zeros_like(z, dtype=torch.complex128)
        
        # 使用排序的格点以改进收敛
        lattice_points = []
        for m in range(-max_m, max_m + 1):
            for n in range(-max_n, max_n + 1):
                if m == 0 and n == 0:
                    continue
                w = 2 * m * self.omega1 + 2 * n * self.omega3
                lattice_points.append((abs(w), w, m, n))
        
        # 按模长排序
        lattice_points.sort(key=lambda x: x[0].real if isinstance(x[0], torch.Tensor) else x[0])
        
        # 计算级数项，加入更强的稳定性控制
        for _, w, m, n in lattice_points:
            if isinstance(w, torch.Tensor):
                w = w.to(z.device)
            diff = z - w
            
            # 增加阈值避免除零和极大值
            mask = torch.abs(diff) > self.eps * 15
            if mask.any():
                # 使用更稳定的计算方式
                diff_masked = diff[mask]
                w_term = 1.0/w**2 if abs(w) > self.eps else 0.0
                
                # 限制最大值避免数值爆炸
                wp_term = 1.0/diff_masked**2 - w_term
                wp_prime_term = -2.0/diff_masked**3
                
                # 裁剪极值，为ViT-Ti使用更保守的限制
                wp_term = torch.clamp(wp_term.real, -5e3, 5e3) + 1j * torch.clamp(wp_term.imag, -5e3, 5e3)
                wp_prime_term = torch.clamp(wp_prime_term.real, -5e3, 5e3) + 1j * torch.clamp(wp_prime_term.imag, -5e3, 5e3)
                
                wp_sum[mask] += wp_term
                wp_prime_sum[mask] += wp_prime_term
        
        return wp_sum, wp_prime_sum
    
    def wp_and_wp_prime(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同时计算℘(z)和℘'(z)，使用改进的数值方法
        """
        z = z.to(torch.complex128)
        
        # 处理接近原点的情况
        near_origin = torch.abs(z) < self.eps * 15
        
        # 初始化结果
        wp = torch.zeros_like(z, dtype=torch.complex128)
        wp_prime = torch.zeros_like(z, dtype=torch.complex128)
        
        # 对于不接近原点的点
        valid_mask = ~near_origin
        if valid_mask.any():
            z_valid = z[valid_mask]
            
            # 主部
            wp_main = 1.0 / z_valid**2
            wp_prime_main = -2.0 / z_valid**3
            
            # 级数部分
            wp_series, wp_prime_series = self._improved_series_sum(z_valid)
            
            wp[valid_mask] = wp_main + wp_series
            wp_prime[valid_mask] = wp_prime_main + wp_prime_series
        
        # 对于接近原点的点，使用较大值但避免inf
        large_value = 5e2  # 为ViT-Ti降低大值
        wp[near_origin] = large_value
        wp_prime[near_origin] = large_value
        
        # 最终裁剪确保数值稳定
        wp = torch.clamp(wp.real, -1e4, 1e4) + 1j * torch.clamp(wp.imag, -1e4, 1e4)
        wp_prime = torch.clamp(wp_prime.real, -1e4, 1e4) + 1j * torch.clamp(wp_prime.imag, -1e4, 1e4)
        
        return wp, wp_prime


class WEFPositionalEncoding(nn.Module):
    """
    改进的基于WEF的位置编码模块，为ViT-Ti优化
    """
    def __init__(self, 
                 d_model: int,
                 max_h: int = 14,
                 max_w: int = 14,
                 g2: float = 1.0,
                 g3: float = 0.0,
                 projection_type: str = 'linear',  # ViT-Ti使用简单投影
                 alpha_scale: float = 0.15,
                 learnable_alpha: bool = True,
                 use_4d: bool = True,
                 device: torch.device = None):
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        self.use_4d = use_4d
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 可学习的缩放参数
        if learnable_alpha:
            self.log_alpha_scale = nn.Parameter(torch.log(torch.tensor(alpha_scale)))
        else:
            self.register_buffer('log_alpha_scale', torch.log(torch.tensor(alpha_scale)))
        
        # 可学习的格形状参数，初始化为小的正值
        self.alpha_learn = nn.Parameter(torch.tensor(0.12))  # 为ViT-Ti调整
        
        # WEF计算器
        self.wef = WeierstrassEllipticFunction(
            g2=g2, 
            g3=g3, 
            alpha_scale=alpha_scale,
            device=self.device
        )
        
        # 投影层，为ViT-Ti简化结构
        input_dim = 4 if use_4d else 2
        if projection_type == 'mlp':
            self.projection = nn.Sequential(
                nn.Linear(input_dim, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(0.05),  # 降低dropout
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model)
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model)
            )
        
        # CLS token的特殊位置编码
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.015)  # 减小初始化
        
        # 添加一个可学习的缩放因子来控制位置编码的强度
        self.pos_scale = nn.Parameter(torch.ones(1) * 0.8)  # 为ViT-Ti降低初始缩放
    
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        生成并添加位置编码，增加稳定性措施
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 计算omega_3，限制范围
        omega_3_prime = F.softplus(self.alpha_learn).clamp(0.02, 8.0)  # 为ViT-Ti调整范围
        omega_3 = 1j * omega_3_prime
        
        # 生成位置编码
        position_encodings = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # CLS token编码
        position_encodings[:, 0:1, :] = self.cls_pos_embedding
        
        # 生成patch坐标
        row_idx = torch.arange(h, device=device).unsqueeze(1).repeat(1, w).reshape(-1)
        col_idx = torch.arange(w, device=device).repeat(h)
        
        # 归一化坐标
        u = (col_idx.float() + 0.5) / w
        v = (row_idx.float() + 0.5) / h
        
        # 映射到复平面，缩小范围避免极值
        z_real = u * (2 * self.wef.omega1.real) * 0.4  # 为ViT-Ti进一步缩小范围
        z_imag = v * (2 * omega_3_prime) * 0.4
        z = torch.complex(z_real, z_imag)
        
        # 计算WEF值
        wp, wp_prime = self.wef.wp_and_wp_prime(z)
        
        # 应用tanh压缩，使用动态缩放
        alpha_scale = torch.exp(self.log_alpha_scale).clamp(0.002, 0.8)  # 为ViT-Ti调整范围
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
        
        # 投影到d_model维度
        patch_encodings = self.projection(features)
        position_encodings[:, 1:1+h*w, :] = patch_encodings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 应用可学习的缩放因子
        position_encodings = position_encodings * self.pos_scale
        
        return x + position_encodings


class ImprovedViT_Ti(nn.Module):
    """
    ViT-Ti实现，使用WEF位置编码
    ViT-Ti配置: embed_dim=192, depth=12, num_heads=3
    """
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 100,
                 embed_dim: int = 192,  # ViT-Ti配置
                 depth: int = 12,       # ViT-Ti配置
                 num_heads: int = 3,    # ViT-Ti配置
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.05,  # 降低dropout
                 drop_path: float = 0.05,  # 降低drop_path
                 use_wef: bool = True):
        super().__init__()
        
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding with layer norm
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position encoding
        if use_wef:
            self.pos_encoding = WEFPositionalEncoding(
                d_model=embed_dim,
                max_h=img_size // patch_size,
                max_w=img_size // patch_size,
                g2=1.0,
                g3=0.0,
                projection_type='linear',  # ViT-Ti使用简单投影
                alpha_scale=0.15,
                learnable_alpha=True,
                use_4d=True
            )
        else:
            # 标准可学习位置编码
            self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            self.pos_encoding = None
        
        self.use_wef = use_wef
        self.dropout = nn.Dropout(dropout)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i]
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
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.patch_norm(x)
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, 1 + num_patches, embed_dim]
        
        # Add position encoding
        if self.use_wef:
            h = w = int(math.sqrt(self.num_patches))
            x = self.pos_encoding(x, h, w)
        else:
            x = x + self.pos_embedding
        
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.norm(x)
        x = x[:, 0]  # CLS token
        x = self.head(x)
        
        return x


class TransformerBlock(nn.Module):
    """标准Transformer块"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_out)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# 数据增强
class Cutout:
    """Cutout augmentation"""
    def __init__(self, n_holes=1, length=12):  # 为ViT-Ti调整cutout大小
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


def create_cifar100_dataloaders(batch_size=160, num_workers=8):  # 增大batch_size和num_workers
    """创建CIFAR-100数据加载器"""
    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                           std=[0.2673, 0.2564, 0.2762]),
        Cutout(n_holes=1, length=12)  # 为ViT-Ti调整
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
    
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, num_epochs=120,  # 增加epoch数
                initial_lr=0.0015, warmup_epochs=15, use_amp=True, use_mixup=True):  # 调整学习率和warmup
    """训练模型，使用改进的训练策略"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 使用AdamW优化器，为ViT-Ti调整weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.03)
    
    # 改进的学习率调度器
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
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)  # 为ViT-Ti降低标签平滑
    scaler = GradScaler() if use_amp else None
    
    best_acc = 0.0
    start_epoch = 0
    checkpoint_path = 'checkpoint_vit_ti.pth'
    log_path = 'train_log_vit_ti.txt'
    
    # 加载检查点
    if os.path.exists(checkpoint_path):
        print(f"=> Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if use_amp and 'scaler' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"=> Resumed from epoch {checkpoint['epoch']}, best_acc={best_acc:.2f}%")
    
    # 日志文件头
    if start_epoch == 0 and (not os.path.exists(log_path) or os.path.getsize(log_path) == 0):
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write('epoch,batch,phase,loss,acc,lr,time\n')
    
    # 梯度裁剪值，为ViT-Ti调整
    max_grad_norm = 0.8
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.time()
            images, labels = images.to(device), labels.to(device)
            
            # Mixup - 为ViT-Ti调整参数
            if use_mixup and random.random() > 0.4:  # 提高mixup概率
                mixup_alpha = 0.3 if epoch >= warmup_epochs else 0.15  # 增大mixup强度
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                
                optimizer.zero_grad()
                if use_amp:
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    scaler.scale(loss).backward()
                    # 梯度裁剪
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
                    # 梯度裁剪
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
            
            scheduler.step()  # 每个batch更新学习率
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            batch_acc = 100.0 * predicted.eq(labels).sum().item() / labels.size(0)
            batch_time = time.time() - batch_start
            lr = optimizer.param_groups[0]["lr"]
            
            # 每80个batch打印一次
            if batch_idx % 80 == 0:
                log_str = (f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                           f'Loss: {loss.item():.4f} Acc: {batch_acc:.2f}% LR: {lr:.6f} Time: {batch_time:.2f}s')
                print(log_str)
                
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f'{epoch+1},{batch_idx},train,{loss.item():.4f},{batch_acc:.2f},{lr:.6f},{batch_time:.2f}\n')
        
        # 验证
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        val_start = time.time()
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        # 统计
        train_acc = 100.0 * train_correct / train_total
        test_acc = 100.0 * test_correct / test_total
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
            torch.save(model.state_dict(), 'best_wef_vit_ti.pth')
            print(f'Best model saved with accuracy: {best_acc:.2f}%')
        
        # 保存断点
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc
        }
        if use_amp and scaler is not None:
            checkpoint['scaler'] = scaler.state_dict()
        torch.save(checkpoint, checkpoint_path)
    
    return model, best_acc


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 创建数据加载器，利用强大的硬件配置
    train_loader, test_loader = create_cifar100_dataloaders(batch_size=160, num_workers=8)
    
    # 创建ViT-Ti模型
    model = ImprovedViT_Ti(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=100,
        embed_dim=192,      # ViT-Ti配置
        depth=12,           # 12层Transformer
        num_heads=3,        # 3个注意力头
        mlp_ratio=4.0,      # MLP隐藏层维度 = 192 * 4 = 768
        dropout=0.05,       # 降低dropout
        drop_path=0.05,     # 降低drop_path
        use_wef=True        # 使用WEF位置编码
    )
    
    print(f"ViT-Ti Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练模型
    model, best_acc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=120,     # 增加训练轮数
        initial_lr=0.0015,  # 调整学习率
        warmup_epochs=15,   # 增加warmup
        use_amp=True,
        use_mixup=True
    )
    
    print(f"ViT-Ti Training completed! Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()