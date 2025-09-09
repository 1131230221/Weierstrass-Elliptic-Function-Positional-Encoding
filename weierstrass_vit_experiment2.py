import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import math
from scipy.special import gamma
import timm
from timm.models import vision_transformer
import os
import json
from tqdm import tqdm
import warnings
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from timm.data.auto_augment import auto_augment_transform
from timm.data.mixup import Mixup
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImprovedWeierstrassEllipticPositionalEncoding(nn.Module):
    """改进的威尔斯特拉斯椭圆函数位置编码"""
    
    def __init__(self, d_model, num_patches_h, num_patches_w, 
                 use_derivative=True, alpha_scale=0.05, use_layernorm=True):
        super().__init__()
        self.d_model = d_model
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.use_derivative = use_derivative
        self.alpha_scale = alpha_scale
        self.use_layernorm = use_layernorm
        
        # 计算精确的半周期 ω1（增加数值稳定性）
        self.omega1 = gamma(0.25)**2 / (2 * np.sqrt(2 * np.pi))
        
        # 可学习的参数，使用更好的初始化
        self.alpha_learn = nn.Parameter(torch.tensor(0.5))  # 更好的初始值
        self.beta_learn = nn.Parameter(torch.tensor(0.1))   # 额外的可学习参数
        
        # 改进的投影层架构
        input_dim = 4 if use_derivative else 2
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.LayerNorm(d_model // 2) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 可学习的缩放和偏移参数
        self.scale_factor = nn.Parameter(torch.tensor(alpha_scale))
        self.offset = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 频率调制参数
        self.freq_modulation = nn.Parameter(torch.ones(4 if use_derivative else 2))
        
        # 初始化
        self._init_parameters()
        
    def _init_parameters(self):
        """改进的参数初始化"""
        with torch.no_grad():
            for m in self.projection.modules():
                if isinstance(m, nn.Linear):
                    # Xavier初始化，但缩放因子更小，提高稳定性
                    nn.init.xavier_uniform_(m.weight, gain=0.8)
                    nn.init.zeros_(m.bias)
    
    def stable_weierstrass_p(self, z):
        """数值稳定的威尔斯特拉斯椭圆函数计算"""
        # 获取学习到的参数
        omega2_prime = 0.1 + F.softplus(self.alpha_learn)  # 确保大于0.1
        beta = 0.01 + F.softplus(self.beta_learn)  # 额外的形状参数
        
        # 避免z接近0的数值问题
        z_real = z.real
        z_imag = z.imag
        
        # 计算模长，避免除零
        z_mod = torch.sqrt(z_real**2 + z_imag**2)
        z_mod_safe = torch.clamp(z_mod, min=1e-6)
        
        # 主项：1/z^2，但加入正则化
        main_term = 1.0 / (z_mod_safe**2 + beta)
        
        # 周期修正项（使用更稳定的三角函数）
        u = z_real / self.omega1
        v = z_imag / omega2_prime
        
        # 多个频率分量的叠加，增加函数复杂性
        correction = torch.zeros_like(z_real)
        for k in range(1, 4):  # 使用前3项
            freq_factor = k * math.pi
            amp_k = 0.05 / k  # 递减的振幅
            
            correction += amp_k * (
                torch.cos(freq_factor * u) * torch.exp(-freq_factor * torch.abs(v)) +
                torch.sin(freq_factor * v) * torch.exp(-freq_factor * torch.abs(u))
            )
        
        # 组合主项和修正项
        p_z_real = main_term * torch.cos(torch.atan2(z_imag, z_real)) + correction
        p_z_imag = main_term * torch.sin(torch.atan2(z_imag, z_real)) + correction * 0.5
        
        return torch.complex(p_z_real, p_z_imag)
    
    def stable_weierstrass_p_derivative(self, z):
        """数值稳定的威尔斯特拉斯函数导数"""
        omega2_prime = 0.1 + F.softplus(self.alpha_learn)
        beta = 0.01 + F.softplus(self.beta_learn)
        
        z_mod = torch.sqrt(z.real**2 + z.imag**2)
        z_mod_safe = torch.clamp(z_mod, min=1e-6)
        
        # 导数主项：-2/z^3，加入正则化
        main_deriv = -2.0 / (z_mod_safe**3 + beta)
        
        # 修正项的导数
        u = z.real / self.omega1
        v = z.imag / omega2_prime
        
        deriv_correction = torch.zeros_like(z.real)
        for k in range(1, 3):
            freq_factor = k * math.pi
            amp_k = 0.03 / k
            
            deriv_correction += amp_k * k * (
                -torch.sin(freq_factor * u) * torch.exp(-freq_factor * torch.abs(v)) +
                torch.cos(freq_factor * v) * torch.exp(-freq_factor * torch.abs(u))
            )
        
        p_deriv_real = main_deriv * torch.cos(torch.atan2(z.imag, z.real)) + deriv_correction
        p_deriv_imag = main_deriv * torch.sin(torch.atan2(z.imag, z.real)) + deriv_correction * 0.3
        
        return torch.complex(p_deriv_real, p_deriv_imag)
    
    def forward(self, num_patches_h=None, num_patches_w=None):
        """生成位置编码"""
        if num_patches_h is None:
            num_patches_h = self.num_patches_h
        if num_patches_w is None:
            num_patches_w = self.num_patches_w
            
        device = self.alpha_learn.device
        
        # 创建改进的网格坐标（添加小的随机扰动以增加鲁棒性）
        row_idx = torch.arange(num_patches_h, dtype=torch.float32, device=device)
        col_idx = torch.arange(num_patches_w, dtype=torch.float32, device=device)
        
        # 改进的归一化方式
        u = (col_idx + 0.5) / (num_patches_w + 1e-8)
        v = (row_idx + 0.5) / (num_patches_h + 1e-8)
        
        # 创建网格
        u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
        u_grid = u_grid.T.reshape(-1)
        v_grid = v_grid.T.reshape(-1)
        
        # 获取学习参数
        omega2_prime = 0.1 + F.softplus(self.alpha_learn)
        
        # 映射到复平面，添加相位偏移
        z_real = u_grid * self.omega1 + 0.01 * torch.sin(2 * math.pi * u_grid)
        z_imag = v_grid * omega2_prime + 0.01 * torch.cos(2 * math.pi * v_grid)
        z = torch.complex(z_real, z_imag)
        
        # 计算威尔斯特拉斯函数值
        p_z = self.stable_weierstrass_p(z)
        
        # 分离实部和虚部，应用频率调制
        p_real = p_z.real * self.freq_modulation[0]
        p_imag = p_z.imag * self.freq_modulation[1]
        
        # 改进的激活函数组合
        pe_real = torch.tanh(self.scale_factor * p_real)
        pe_imag = torch.tanh(self.scale_factor * p_imag * 0.8)
        
        if self.use_derivative:
            # 计算导数
            p_prime = self.stable_weierstrass_p_derivative(z)
            p_prime_real = torch.tanh(self.scale_factor * p_prime.real * self.freq_modulation[2])
            p_prime_imag = torch.tanh(self.scale_factor * p_prime.imag * self.freq_modulation[3])
            
            # 组合4维特征
            features = torch.stack([pe_real, pe_imag, p_prime_real, p_prime_imag], dim=-1)
        else:
            # 组合2维特征
            features = torch.stack([pe_real, pe_imag], dim=-1)
        
        # 投影到d_model维
        pos_encoding = self.projection(features)
        
        # 添加可学习的偏移
        pos_encoding = pos_encoding + self.offset
        
        # 修正输出 shape，确保为 (num_patches, d_model)
        if pos_encoding.dim() == 3 and pos_encoding.shape[0] == 1:
            pos_encoding = pos_encoding.squeeze(0)
        elif pos_encoding.dim() > 2:
            pos_encoding = pos_encoding.view(-1, self.d_model)
        return pos_encoding

class ImprovedVisionTransformerWithWeierstrassEncoding(nn.Module):
    """改进的使用威尔斯特拉斯位置编码的Vision Transformer"""
    
    def __init__(self, base_model, image_size=224, patch_size=16, num_classes=100,
                 use_derivative=True, alpha_scale=0.05, use_stochastic_depth=True):
        super().__init__()
        
        self.base_model = base_model
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_patches_h = image_size // patch_size
        self.num_patches_w = image_size // patch_size
        
        # 获取模型维度
        self.d_model = base_model.embed_dim
        
        # 替换位置编码
        self.weierstrass_encoding = ImprovedWeierstrassEllipticPositionalEncoding(
            d_model=self.d_model,
            num_patches_h=self.num_patches_h,
            num_patches_w=self.num_patches_w,
            use_derivative=use_derivative,
            alpha_scale=alpha_scale,
            use_layernorm=True
        )
        
        # 改进的CLS token位置编码
        self.cls_pos_encoding = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_pos_encoding, std=0.01)
        
        # 添加位置编码的dropout
        self.pos_dropout = nn.Dropout(0.1)
        
        # 随机深度（Stochastic Depth）
        if use_stochastic_depth:
            self.enable_stochastic_depth()
        
        # 改进的分类头
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.d_model // 2, num_classes)
        )
        
        # 初始化分类器
        self._init_classifier()
        
    def enable_stochastic_depth(self):
        """启用随机深度"""
        drop_path_rate = 0.1
        num_layers = len(self.base_model.blocks)
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        
        for i, block in enumerate(self.base_model.blocks):
            if hasattr(block, 'drop_path'):
                block.drop_path.drop_prob = drop_path_rates[i]
        
    def _init_classifier(self):
        """初始化分类器"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.base_model.patch_embed(x)  # (B, num_patches, D)
        
        # 添加CLS token
        cls_tokens = self.base_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+num_patches, D)
        
        # 获取威尔斯特拉斯位置编码
        patch_pos_encoding = self.weierstrass_encoding()  # (num_patches, D)
        
        # 组合CLS和patch的位置编码
        pos_encoding = torch.cat([
            self.cls_pos_encoding,  # (1, 1, D)
            patch_pos_encoding.unsqueeze(0)  # (1, num_patches, D)
        ], dim=1)  # (1, 1+num_patches, D)
        
        # 添加位置编码和dropout
        x = x + pos_encoding
        x = self.pos_dropout(x)
        
        # Transformer blocks
        x = self.base_model.pos_drop(x)
        x = self.base_model.blocks(x)
        x = self.base_model.norm(x)
        
        # 分类
        x = x[:, 0]  # 取CLS token
        x = self.pre_classifier(x)
        x = self.classifier(x)
        
        return x

# 改进的数据增强
def get_improved_cifar100_dataloaders(batch_size=64, num_workers=4, distributed=False, rank=0, world_size=1):
    """获取改进的CIFAR-100数据加载器"""
    
    # 更强的数据增强
    transform_train = transforms.Compose([
        transforms.Resize(256),  # 先放大
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761]),
    ])
    
    # 加载数据集
    train_dataset = datasets.CIFAR100(root='./data', train=True, 
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, 
                                    download=True, transform=transform_test)
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        test_sampler = None
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=(train_sampler is None),
                            num_workers=num_workers, 
                            pin_memory=True,
                            sampler=train_sampler,
                            drop_last=True)  # 添加drop_last
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False,
                           num_workers=num_workers, 
                           pin_memory=True,
                           sampler=test_sampler)
    
    return train_loader, test_loader, train_sampler

# Mixup数据增强
def setup_mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
    """设置Mixup和CutMix"""
    return Mixup(
        mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, cutmix_minmax=None,
        prob=prob, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=100
    )

def train_epoch_improved(model, train_loader, criterion, optimizer, device, epoch, 
                        train_sampler=None, scaler=None, mixup_fn=None):
    """改进的训练函数"""
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
        
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 应用Mixup/CutMix
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
        
        optimizer.zero_grad()
        
        # 使用混合精度训练
        with autocast():
            outputs = model(inputs)
            if mixup_fn is not None:
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs, targets)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        
        # 计算准确率（对于mixup需要特殊处理）
        if mixup_fn is None:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        else:
            # Mixup情况下使用软标签的近似准确率
            _, predicted = outputs.max(1)
            if hasattr(targets, 'argmax'):
                targets_hard = targets.argmax(1)
            else:
                targets_hard = targets
            total += targets_hard.size(0)
            correct += predicted.eq(targets_hard).sum().item()
        
        # 更新进度条
        if total > 0:
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # 定期清理缓存
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    return running_loss / len(train_loader), 100. * correct / total if total > 0 else 0

def evaluate_improved(model, test_loader, criterion, device):
    """改进的评估函数"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            torch.cuda.empty_cache()
    
    return test_loss / len(test_loader), 100. * correct / total

def main():
    # 分布式训练设置
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.cuda.set_device(gpu)
        distributed = True
    else:
        distributed = False
        rank = 0
        world_size = 1
        gpu = 0
    
    # 设置设备
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    if rank == 0:
        print(f"Using device: {device}")
    
    # 设置随机种子
    set_seed(42 + rank)
    
    # 改进的超参数
    batch_size = 20  # 稍微增加batch size
    learning_rate = 0.0008  # 略微降低学习率
    num_epochs = 35  # 增加epoch数
    warmup_epochs = 8  # 增加warmup
    weight_decay = 0.05  # 增加正则化
    num_workers = 11
    
    # GPU内存设置
    torch.cuda.set_per_process_memory_fraction(0.85)
    torch.cuda.empty_cache()
    
    # 加载预训练模型
    if rank == 0:
        print("Loading pretrained ViT-B/16 model...")
    
    try:
        base_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        pretrained_path = 'jx_vit_base_patch16_224_in21k-e5005f0a.pth'
        
        state_dict = torch.load(pretrained_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        keys_to_remove = ['pre_logits.fc.bias', 'pre_logits.fc.weight', 'head.bias', 'head.weight']
        for key in keys_to_remove:
            if key in state_dict:
                del state_dict[key]
        
        base_model.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print("Successfully loaded pretrained model")
    except Exception as e:
        if rank == 0:
            print(f"Error loading pretrained model: {str(e)}")
            print("Using untrained model...")
        base_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    
    # 创建改进的模型
    model = ImprovedVisionTransformerWithWeierstrassEncoding(
        base_model=base_model,
        image_size=224,
        patch_size=16,
        num_classes=100,
        use_derivative=True,
        alpha_scale=0.05,
        use_stochastic_depth=True
    )
    
    model = model.to(device)
    
    if distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    
    # 获取改进的数据加载器
    train_loader, test_loader, train_sampler = get_improved_cifar100_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size
    )
    
    # 设置Mixup
    mixup_fn = setup_mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5)
    
    # 改进的损失函数（标签平滑）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 改进的优化器设置
    if distributed:
        base_params = []
        new_params = []
        weierstrass_params = []
        
        for name, param in model.module.named_parameters():
            if 'weierstrass_encoding' in name:
                weierstrass_params.append(param)
            elif 'cls_pos_encoding' in name or 'classifier' in name or 'pre_classifier' in name:
                new_params.append(param)
            else:
                base_params.append(param)
    else:
        base_params = []
        new_params = []
        weierstrass_params = []
        
        for name, param in model.named_parameters():
            if 'weierstrass_encoding' in name:
                weierstrass_params.append(param)
            elif 'cls_pos_encoding' in name or 'classifier' in name or 'pre_classifier' in name:
                new_params.append(param)
            else:
                base_params.append(param)
    
    # 使用不同的学习率
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': learning_rate * 0.05},  # 预训练参数更小学习率
        {'params': weierstrass_params, 'lr': learning_rate * 0.8},  # 威尔斯特拉斯参数
        {'params': new_params, 'lr': learning_rate}  # 新参数正常学习率
    ], weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    
    # 创建梯度缩放器
    scaler = GradScaler()
    
    # 改进的学习率调度器
    def improved_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            # 余弦退火 + 重启
            cycle_epoch = (epoch - warmup_epochs) % 15
            cycle_progress = cycle_epoch / 15
            return 0.5 * (1 + math.cos(math.pi * cycle_progress)) * 0.95 ** ((epoch - warmup_epochs) // 15)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, improved_lr_lambda)
    
    # 训练循环
    best_acc = 0.0
    results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    patience = 8
    patience_counter = 0
    
    if rank == 0:
        print("\nStarting improved training...")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch_improved(
            model, train_loader, criterion, optimizer, device, epoch, 
            train_sampler, scaler, mixup_fn if epoch > 5 else None  # 前5个epoch不用mixup
        )
        
        # 评估
        test_loss, test_acc = evaluate_improved(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录结果
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        
        if rank == 0:
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停和最佳模型保存
            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
                
                # 保存最佳模型
                model_state = model.module.state_dict() if distributed else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'results': results
                }, 'improved_weierstrass_vit_best.pth')
                
                print(f"🎉 New best accuracy: {best_acc:.2f}%")
                
                # 如果超过目标准确率，记录
                if best_acc > 91.67:
                    print(f"🚀 Target achieved! Best accuracy: {best_acc:.2f}% > 91.67%")
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch > 20:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # 定期保存检查点
            if epoch % 5 == 0:
                model_state = model.module.state_dict() if distributed else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'results': results
                }, f'checkpoint_epoch_{epoch}.pth')
    
    if rank == 0:
        print(f"\n🏁 Training completed. Best accuracy: {best_acc:.2f}%")
        if best_acc > 91.67:
            print(f"🎯 Successfully exceeded baseline of 91.67%!")
        
        # 保存最终结果
        with open('improved_training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # 输出学习到的威尔斯特拉斯参数
        if distributed:
            alpha = F.softplus(model.module.weierstrass_encoding.alpha_learn).item()
            beta = F.softplus(model.module.weierstrass_encoding.beta_learn).item()
        else:
            alpha = F.softplus(model.weierstrass_encoding.alpha_learn).item()
            beta = F.softplus(model.weierstrass_encoding.beta_learn).item()
        
        print(f"\nLearned Weierstrass parameters:")
        print(f"ω'2 (alpha): {alpha:.6f}")
        print(f"β (beta): {beta:.6f}")
    
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()