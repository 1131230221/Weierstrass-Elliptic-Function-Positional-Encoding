#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, time, json, pickle, argparse, random
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision
from tqdm import tqdm

# =========================
#  数据集：从 .pkl 子集读取
# =========================

class CIFAR100PKLSubset(Dataset):
    """
    读取由 make_cifar100_subset.py 生成的 pkl 或 CIFAR-100 官方数据文件：
      {"data": (N,3072) uint8/np.int, "fine_labels": list[int]} 或
      {"data": (N,3072) uint8/np.int, "labels": list[int]}
    """
    def __init__(self, pkl_path: str, train: bool = True):
        super().__init__()
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f, encoding="latin1")
        self.data = obj["data"]  # (N,3072)
        # 兼容不同的标签键名
        if "fine_labels" in obj:
            self.labels = obj["fine_labels"]
        elif "labels" in obj:
            self.labels = obj["labels"]
        else:
            raise KeyError("未找到标签字段，期望 'fine_labels' 或 'labels'")
        assert len(self.data) == len(self.labels)

        # —— 变换：为 ViT 从 32→224，并强化稀缺场景的泛化 ——
        normalize = T.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                std=[0.2673, 0.2564, 0.2762])

        # RandAugment 可能因 torchvision 版本不存在，做降级处理
        try:
            randaug = T.RandAugment(num_ops=2, magnitude=9)
        except Exception:
            randaug = T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)

        if train:
            self.transform = T.Compose([
                ToPILFromFlat(),
                T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(),
                randaug,
                T.ToTensor(),
                normalize,
                Cutout(n_holes=1, length=12),
            ])
        else:
            self.transform = T.Compose([
                ToPILFromFlat(),
                T.Resize((224, 224)),
                T.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        # 转为 CHW 的 32x32x3
        img = x.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
        y = int(self.labels[idx])
        return self.transform(img), y


class ToPILFromFlat:
    """numpy(H,W,C) uint8 → PIL.Image"""
    def __call__(self, img_ndarray: np.ndarray):
        from PIL import Image
        # Pillow 将在未来移除 mode 参数的支持，这里交由其自动推断
        return Image.fromarray(img_ndarray)


class Cutout:
    """简易 Cutout"""
    def __init__(self, n_holes=1, length=12):
        self.n_holes = n_holes
        self.length = length
    def __call__(self, img: torch.Tensor):
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
        return img * mask


# =========================
#    模型（WEF 版本 ViT-Ti）
#    —— 来自你之前代码并小幅调参/润色 —— 
# =========================

# ---- 省略：这里完整粘贴你之前的 WEF 与 ViT 实现（略去注释以节省篇幅）----
# 说明：以下实现保持与您先前提供版本一致，仅少量缺省改动：
# 1) dropout=0.10, drop_path=0.10；
# 2) WEF 内部数值范围与初始化维持稳定；
# 3) 其余结构不变。

class WeierstrassEllipticFunction:
    def __init__(self, g2=1.0, g3=0.0, eps=1e-8, alpha_scale=0.15, device=None):
        self.g2, self.g3, self.eps, self.alpha_scale = g2, g3, eps, alpha_scale
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        disc = g2**3 - 27*g3**2
        assert abs(disc) > eps, f"判别式太接近零: {disc}"
        if abs(g3) < eps:
            self.omega1 = torch.tensor(2.62205755429212, device=self.device, dtype=torch.complex128)
            self.omega3 = torch.tensor(complex(0, 2.62205755429212), device=self.device, dtype=torch.complex128)
        else:
            self.omega1 = torch.tensor(complex(1.0, 0.0), device=self.device, dtype=torch.complex128)
            self.omega3 = torch.tensor(complex(0.0, 1.0), device=self.device, dtype=torch.complex128)

    def _improved_series_sum(self, z, max_m=12, max_n=12):
        z = z.to(torch.complex128)
        wp_sum = torch.zeros_like(z)
        wp_p_sum = torch.zeros_like(z)
        lattice = []
        for m in range(-max_m, max_m + 1):
            for n in range(-max_n, max_n + 1):
                if m == 0 and n == 0:
                    continue
                w = 2*m*self.omega1 + 2*n*self.omega3
                lattice.append((abs(w), w))
        lattice.sort(key=lambda x: x[0].real if isinstance(x[0], torch.Tensor) else x[0])
        for _, w in lattice:
            if isinstance(w, torch.Tensor):
                w = w.to(z.device)
            diff = z - w
            mask = torch.abs(diff) > 1.5e-7
            if mask.any():
                d = diff[mask]
                w_term = 1.0/w**2 if abs(w) > 1e-8 else 0.0
                wp_t = 1.0/d**2 - w_term
                wp_p = -2.0/d**3
                wp_t = torch.clamp(wp_t.real, -5e3, 5e3) + 1j*torch.clamp(wp_t.imag, -5e3, 5e3)
                wp_p = torch.clamp(wp_p.real, -5e3, 5e3) + 1j*torch.clamp(wp_p.imag, -5e3, 5e3)
                wp_sum[mask] += wp_t
                wp_p_sum[mask] += wp_p
        return wp_sum, wp_p_sum

    def wp_and_wp_prime(self, z):
        z = z.to(torch.complex128)
        near = torch.abs(z) < 1.5e-7
        wp = torch.zeros_like(z)
        wp_p = torch.zeros_like(z)
        valid = ~near
        if valid.any():
            zv = z[valid]
            wp_main = 1.0/zv**2
            wp_p_main = -2.0/zv**3
            wp_s, wp_p_s = self._improved_series_sum(zv)
            wp[valid] = wp_main + wp_s
            wp_p[valid] = wp_p_main + wp_p_s
        large = 5e2
        wp[near] = large
        wp_p[near] = large
        wp = torch.clamp(wp.real, -1e4, 1e4) + 1j*torch.clamp(wp.imag, -1e4, 1e4)
        wp_p = torch.clamp(wp_p.real, -1e4, 1e4) + 1j*torch.clamp(wp_p.imag, -1e4, 1e4)
        return wp, wp_p


class WEFPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_h=14, max_w=14, g2=1.0, g3=0.0,
                 projection_type='linear', alpha_scale=0.15,
                 learnable_alpha=True, use_4d=True, device=None):
        super().__init__()
        self.d_model, self.max_h, self.max_w = d_model, max_h, max_w
        self.use_4d = use_4d
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if learnable_alpha:
            self.log_alpha_scale = nn.Parameter(torch.log(torch.tensor(alpha_scale)))
        else:
            self.register_buffer('log_alpha_scale', torch.log(torch.tensor(alpha_scale)))
        self.alpha_learn = nn.Parameter(torch.tensor(0.12))
        self.wef = WeierstrassEllipticFunction(g2=g2, g3=g3, alpha_scale=alpha_scale, device=self.device)
        in_dim = 4 if use_4d else 2
        if projection_type == 'mlp':
            self.proj = nn.Sequential(
                nn.Linear(in_dim, d_model//2),
                nn.LayerNorm(d_model//2),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(d_model//2, d_model),
                nn.LayerNorm(d_model)
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model)
            )
        self.cls_pos = nn.Parameter(torch.randn(1,1,d_model) * 0.015)
        self.pos_scale = nn.Parameter(torch.ones(1) * 0.8)

    def forward(self, x, h, w):
        B, S, D = x.shape
        device = x.device
        omega_3p = F.softplus(self.alpha_learn).clamp(0.02, 8.0)
        row = torch.arange(h, device=device).unsqueeze(1).repeat(1, w).reshape(-1)
        col = torch.arange(w, device=device).repeat(h)
        u = (col.float()+0.5)/w
        v = (row.float()+0.5)/h
        z_real = u * (2 * self.wef.omega1.real) * 0.4
        z_imag = v * (2 * omega_3p) * 0.4
        z = torch.complex(z_real, z_imag)
        wp, wp_p = self.wef.wp_and_wp_prime(z)
        alpha = torch.exp(self.log_alpha_scale).clamp(0.002, 0.8)
        f1 = torch.tanh(alpha * wp.real)
        f2 = torch.tanh(alpha * wp.imag)
        if self.use_4d:
            f3 = torch.tanh(alpha * wp_p.real)
            f4 = torch.tanh(alpha * wp_p.imag)
            feats = torch.stack([f1, f2, f3, f4], dim=1).float()
        else:
            feats = torch.stack([f1, f2], dim=1).float()
        patch_enc = self.proj(feats)
        pos = torch.zeros(B, S, D, device=device)
        pos[:, :1, :] = self.cls_pos
        pos[:, 1:1+h*w, :] = patch_enc.unsqueeze(0).expand(B, -1, -1)
        pos = pos * self.pos_scale
        return x + pos


class DropPath(nn.Module):
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
        return x.div(keep_prob) * random_tensor


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim*mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        xn = self.norm1(x)
        a, _ = self.attn(xn, xn, xn)
        x = x + self.drop_path(a)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ImprovedViT_Ti(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=100,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0,
                 dropout=0.10, drop_path=0.10, use_wef=True):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_ln = nn.LayerNorm(embed_dim)
        self.cls = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.use_wef = use_wef
        if use_wef:
            self.pos = WEFPositionalEncoding(embed_dim, img_size//patch_size, img_size//patch_size)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, 1+self.num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop = nn.Dropout(dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight); 
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)                  # [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)   # [B, N, C]
        x = self.patch_ln(x)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if self.use_wef:
            h = w = int(math.sqrt(self.num_patches))
            x = self.pos(x, h, w)
        else:
            x = x + self.pos_embed
        x = self.drop(x)
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)[:, 0]
        return self.head(x)


# =========================
#        训练实用函数
# =========================

def linear_scaled_lr(base_lr: float, batch_size: int, base_bs: int = 128) -> float:
    return base_lr * (batch_size / base_bs)

def mixup_data(x, y, alpha: float):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def mixup_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)

def cosine_with_warmup(optimizer, warmup_steps, total_steps, cycles=0.5):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * cycles * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)


def train_one_setting(train_loader, test_loader, args, out_dir) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedViT_Ti(dropout=0.10, drop_path=0.10, use_wef=True).to(device)

    # 动态 lr（线性缩放）
    lr = linear_scaled_lr(args.base_lr, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup * len(train_loader)
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(enabled=args.amp)
    best_acc = 0.0
    max_grad_norm = 1.0

    log_file = Path(out_dir) / "train_log.csv"
    with open(log_file, "w") as f:
        f.write("epoch,phase,loss,acc,lr,secs\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        # -------- train ----------
        model.train()
        train_loss, train_corr, train_cnt = 0.0, 0, 0
        
        # 训练进度条
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f"Epoch {epoch+1}/{args.epochs} [Train]", 
                         leave=False, ncols=100)
        
        for batch_idx, (imgs, labels) in train_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            # 稀缺数据 → 更强的 mixup
            mixed = imgs; ya = labels; yb = labels; lam = 1.0
            if args.mixup_alpha > 1e-6:
                mixed, ya, yb, lam = mixup_data(imgs, labels, alpha=args.mixup_alpha)

            if args.amp:
                with autocast('cuda'):
                    logits = model(mixed)
                    loss = mixup_criterion(criterion, logits, ya, yb, lam) if lam < 0.999 else criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(mixed)
                loss = mixup_criterion(criterion, logits, ya, yb, lam) if lam < 0.999 else criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            # 修复学习率调度器警告：确保在 optimizer.step() 之后调用
            scheduler.step()

            train_loss += loss.item()
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                train_corr += (preds == labels).sum().item()
                train_cnt  += labels.size(0)
            
            # 更新进度条显示
            current_acc = 100.0 * train_corr / max(1, train_cnt)
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        # -------- eval ----------
        model.eval()
        val_loss, val_corr, val_cnt = 0.0, 0, 0
        
        # 验证进度条
        val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", 
                       leave=False, ncols=100)
        
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_corr += (preds == labels).sum().item()
                val_cnt  += labels.size(0)
                
                # 更新验证进度条显示
                current_val_acc = 100.0 * val_corr / max(1, val_cnt)
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_val_acc:.2f}%'
                })

        tr_acc = 100.0 * train_corr / max(1, train_cnt)
        va_acc = 100.0 * val_corr / max(1, val_cnt)
        secs = time.time() - t0
        
        # 显示epoch总结
        print(f"Epoch {epoch+1}/{args.epochs} - Train Acc: {tr_acc:.2f}%, Val Acc: {va_acc:.2f}%, Time: {secs:.1f}s")

        with open(log_file, "a") as f:
            f.write(f"{epoch+1},train,{train_loss/len(train_loader):.4f},{tr_acc:.2f},{optimizer.param_groups[0]['lr']:.6f},{secs:.2f}\n")
            f.write(f"{epoch+1},val,{val_loss/len(test_loader):.4f},{va_acc:.2f},-,{secs:.2f}\n")

        # 保存最好
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), str(Path(out_dir) / "best_wef_vitti.pth"))
        if (epoch+1) % 10 == 0:
            print(f"[{epoch+1}/{args.epochs}] TrainAcc={tr_acc:.2f}  ValAcc={va_acc:.2f}  lr={optimizer.param_groups[0]['lr']:.6f}")

    return best_acc


# =========================
#        主控脚本
# =========================

def main():
    parser = argparse.ArgumentParser("WEF on CIFAR-100 scarce-data subsets")
    parser.add_argument("--root", type=str, default="/root/shared-nvme", help="根目录（保存曲线/日志/权重）")
    parser.add_argument("--subsets", type=str, nargs="+",
                        default=["/root/shared-nvme/cifar100_train_20pct.pkl",
                                 "/root/shared-nvme/cifar100_train_40pct.pkl",
                                 "/root/shared-nvme/cifar100_train_60pct.pkl",
                                 "/root/shared-nvme/cifar100_train_90pct.pkl"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=5e-4, help="以 bs=128 为基准的 base lr，按批量线性缩放")
    parser.add_argument("--amp", action="store_true", default=True)
    # 根据稀缺程度动态设定 mixup α（也可手动指定一个固定值）
    parser.add_argument("--mixup_alpha_small", type=float, default=0.6)  # 20%
    parser.add_argument("--mixup_alpha_mid1", type=float, default=0.5)   # 40%
    parser.add_argument("--mixup_alpha_mid2", type=float, default=0.4)   # 60%
    parser.add_argument("--mixup_alpha_big",  type=float, default=0.3)   # 90%
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_root = Path(args.root) / "wef_cifar100_subsets"
    out_root.mkdir(parents=True, exist_ok=True)

    # 测试集（从指定路径加载 CIFAR-100 test）
    print("Loading test dataset...")
    normalize = T.Normalize(mean=[0.5071, 0.4865, 0.4409],
                            std=[0.2673, 0.2564, 0.2762])
    test_tf = T.Compose([T.Resize((224,224)), T.ToTensor(), normalize])
    try:
        test_set = CIFAR100PKLSubset("/root/shared-nvme/cifar-100-python/test", train=False)
        print(f"Test dataset loaded successfully, size: {len(test_set)}")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        raise
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    print("Test dataloader created successfully")

    frac_to_alpha = {}
    best_results: Dict[str, float] = {}
    rel_sizes, accs = [], []

    for pkl_path in args.subsets:
        frac = Path(pkl_path).stem.split("_")[-1].replace("pct","")  # "20" / "40" / ...
        rel = float(frac)/100.0
        # 动态 mixup α
        if rel <= 0.2: mix_alpha = args.mixup_alpha_small
        elif rel <= 0.4: mix_alpha = args.mixup_alpha_mid1
        elif rel <= 0.6: mix_alpha = args.mixup_alpha_mid2
        else: mix_alpha = args.mixup_alpha_big
        frac_to_alpha[rel] = mix_alpha

        subset = CIFAR100PKLSubset(pkl_path, train=True)
        train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)

        exp_dir = out_root / f"subset_{frac}pct"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 把 alpha 作为 args 传入
        class _Args: pass
        run_args = _Args()
        run_args.__dict__.update(vars(args))
        run_args.mixup_alpha = mix_alpha

        print(f"\n=== Training on {frac}% subset  (N={len(subset)})  mixup_alpha={mix_alpha}  ===")
        best_acc = train_one_setting(train_loader, test_loader, run_args, exp_dir)
        best_results[str(rel)] = float(best_acc)
        rel_sizes.append(rel); accs.append(best_acc)
        print(f"--> BEST ACC ({frac}%): {best_acc:.2f}%")

    # 保存结果并绘图
    with open(out_root / "best_results.json", "w") as f:
        json.dump(best_results, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys = zip(*sorted(zip(rel_sizes, accs)))
    plt.figure(figsize=(7,5))
    plt.plot(xs, ys, marker="o", label="WEF (ViT-Tiny)")
    plt.xlabel("Relative Dataset Size")
    plt.ylabel("Max Accuracy")
    plt.title("WEF on CIFAR-100 (scarce data)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    fig_path = out_root / "wef_cifar100_subset_curve.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")

    print("\nAll done.")
    print(f"- Results JSON: {out_root/'best_results.json'}")
    print(f"- Curve PNG   : {fig_path}")

if __name__ == "__main__":
    main()
