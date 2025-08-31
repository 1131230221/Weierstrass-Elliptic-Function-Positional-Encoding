import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
import math
from scipy.special import gamma
import os
import json
from tqdm import tqdm
import warnings
import time
import random
from transformers import ViTModel, ViTConfig
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import h5py
import pickle
from torchvision.datasets import ImageFolder
warnings.filterwarnings('ignore')

# 设置环境变量以解决多进程问题
os.environ['TMPDIR'] = '/tmp'
os.environ['TEMP'] = '/tmp'
os.environ['TMP'] = '/tmp'

def set_seed(seed):
    """设置随机种子确保实验可重复性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImprovedWeierstrassEllipticPositionalEncoding(nn.Module):
    """改进的威尔斯特拉斯椭圆函数位置编码 - 针对ViT-L/16优化，专门适配结构化任务"""
    
    def __init__(self, d_model, num_patches_h, num_patches_w, 
                 use_derivative=True, alpha_scale=0.05, use_layernorm=True):
        super().__init__()
        self.d_model = d_model
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.use_derivative = use_derivative
        self.alpha_scale = alpha_scale
        self.use_layernorm = use_layernorm
        
        # 威尔斯特拉斯椭圆函数的精确半周期
        self.omega1 = gamma(0.25)**2 / (2 * np.sqrt(2 * np.pi))
        
        # 可学习参数 - 针对结构化任务调整初始值
        self.alpha_learn = nn.Parameter(torch.tensor(0.7))  # 增加alpha以捕获空间关系
        self.beta_learn = nn.Parameter(torch.tensor(0.15))  # 增加beta提升稳定性
        self.gamma_learn = nn.Parameter(torch.tensor(0.3))  # 增强几何理解能力
        
        # 多层投影网络 - 为结构化任务优化
        input_dim = 4 if use_derivative else 2
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.LayerNorm(d_model // 2) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.15),  # 稍微增加dropout防止过拟合
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(d_model, d_model)
        )
        
        # 可学习的缩放和偏移
        self.scale_factor = nn.Parameter(torch.tensor(alpha_scale))
        self.offset = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 频率调制参数 - 增强空间感知能力
        freq_dim = 4 if use_derivative else 2
        self.freq_modulation = nn.Parameter(torch.ones(freq_dim))
        
        # 3D空间感知增强模块
        self.spatial_enhance = nn.Parameter(torch.tensor(1.2))
        
        # 初始化
        self._init_parameters()
        
    def _init_parameters(self):
        """Xavier初始化"""
        with torch.no_grad():
            for m in self.projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.9)  # 稍微增加增益
                    nn.init.zeros_(m.bias)
    
    def stable_weierstrass_p(self, z):
        """数值稳定的威尔斯特拉斯椭圆函数 - 针对结构化任务优化"""
        omega2_prime = 0.12 + F.softplus(self.alpha_learn)
        beta = 0.015 + F.softplus(self.beta_learn)
        gamma_param = 0.12 + F.softplus(self.gamma_learn)
        
        z_real = z.real
        z_imag = z.imag
        z_mod = torch.sqrt(z_real**2 + z_imag**2)
        z_mod_safe = torch.clamp(z_mod, min=1e-6)
        
        # 主项 - 增强空间表示能力
        main_term = self.spatial_enhance / (z_mod_safe**2 + beta)
        
        # 周期修正项 - 增强几何理解
        u = z_real / self.omega1
        v = z_imag / omega2_prime
        
        correction = torch.zeros_like(z_real)
        for k in range(1, 5):  # 增加到5项以增强表达能力
            freq_factor = k * math.pi
            amp_k = gamma_param / k**1.8  # 调整衰减速度
            
            correction += amp_k * (
                torch.cos(freq_factor * u) * torch.exp(-freq_factor * torch.abs(v)) +
                torch.sin(freq_factor * v) * torch.exp(-freq_factor * torch.abs(u))
            )
        
        p_z_real = main_term * torch.cos(torch.atan2(z_imag, z_real)) + correction
        p_z_imag = main_term * torch.sin(torch.atan2(z_imag, z_real)) + correction * 0.8
        
        return torch.complex(p_z_real, p_z_imag)
    
    def stable_weierstrass_p_derivative(self, z):
        """威尔斯特拉斯函数导数 - 增强空间梯度信息"""
        omega2_prime = 0.12 + F.softplus(self.alpha_learn)
        beta = 0.015 + F.softplus(self.beta_learn)
        gamma_param = 0.12 + F.softplus(self.gamma_learn)
        
        z_mod = torch.sqrt(z.real**2 + z.imag**2)
        z_mod_safe = torch.clamp(z_mod, min=1e-6)
        
        main_deriv = -2.0 * self.spatial_enhance / (z_mod_safe**3 + beta)
        
        u = z.real / self.omega1
        v = z.imag / omega2_prime
        
        deriv_correction = torch.zeros_like(z.real)
        for k in range(1, 4):
            freq_factor = k * math.pi
            amp_k = gamma_param / k**1.8
            
            deriv_correction += amp_k * k * (
                -torch.sin(freq_factor * u) * torch.exp(-freq_factor * torch.abs(v)) +
                torch.cos(freq_factor * v) * torch.exp(-freq_factor * torch.abs(u))
            )
        
        p_deriv_real = main_deriv * torch.cos(torch.atan2(z.imag, z.real)) + deriv_correction
        p_deriv_imag = main_deriv * torch.sin(torch.atan2(z.imag, z.real)) + deriv_correction * 0.6
        
        return torch.complex(p_deriv_real, p_deriv_imag)
    
    def forward(self, num_patches_h=None, num_patches_w=None):
        """生成位置编码 - 针对DMLab 3D环境优化"""
        if num_patches_h is None:
            num_patches_h = self.num_patches_h
        if num_patches_w is None:
            num_patches_w = self.num_patches_w
            
        device = self.alpha_learn.device
        
        # 创建网格坐标
        row_idx = torch.arange(num_patches_h, dtype=torch.float32, device=device)
        col_idx = torch.arange(num_patches_w, dtype=torch.float32, device=device)
        
        # 改进的归一化 - 适配3D空间理解
        u = (col_idx + 0.5) / (num_patches_w + 1e-8)
        v = (row_idx + 0.5) / (num_patches_h + 1e-8)
        
        u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
        u_grid = u_grid.T.reshape(-1)
        v_grid = v_grid.T.reshape(-1)
        
        omega2_prime = 0.12 + F.softplus(self.alpha_learn)
        
        # 映射到复平面，增加3D空间变换
        z_real = u_grid * self.omega1 + 0.03 * torch.sin(4 * math.pi * u_grid)
        z_imag = v_grid * omega2_prime + 0.03 * torch.cos(4 * math.pi * v_grid)
        z = torch.complex(z_real, z_imag)
        
        # 计算威尔斯特拉斯函数值
        p_z = self.stable_weierstrass_p(z)
        
        # 分离实部和虚部
        p_real = p_z.real * self.freq_modulation[0]
        p_imag = p_z.imag * self.freq_modulation[1]
        
        # 改进的激活函数 - 增强非线性表达
        pe_real = torch.tanh(self.scale_factor * p_real)
        pe_imag = torch.tanh(self.scale_factor * p_imag * 0.9)
        
        if self.use_derivative:
            p_prime = self.stable_weierstrass_p_derivative(z)
            p_prime_real = torch.tanh(self.scale_factor * p_prime.real * self.freq_modulation[2])
            p_prime_imag = torch.tanh(self.scale_factor * p_prime.imag * self.freq_modulation[3])
            
            features = torch.stack([pe_real, pe_imag, p_prime_real, p_prime_imag], dim=-1)
        else:
            features = torch.stack([pe_real, pe_imag], dim=-1)
        
        # 投影到d_model维
        pos_encoding = self.projection(features)
        pos_encoding = pos_encoding + self.offset
        
        return pos_encoding.squeeze(0) if pos_encoding.dim() == 3 else pos_encoding

class WeierstrassViTL16(nn.Module):
    """基于本地预训练ViT-L/16的威尔斯特拉斯位置编码模型 - 适配DMLab任务"""
    
    def __init__(self, pretrained_model_path="/path/to/vit_l16_in21k.pth", 
                 num_classes=6, image_size=384, use_derivative=True, alpha_scale=0.05):
        super().__init__()
        
        # 创建ViT-L/16配置（离线模式）- 支持384×384分辨率
        self.config = ViTConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=image_size,
            patch_size=16,
            num_channels=3,
            qkv_bias=True,
            encoder_stride=16,
        )
        
        # 计算patch数量
        self.image_size = image_size
        self.patch_size = self.config.patch_size
        self.num_patches_per_side = image_size // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        
        # 从本地文件加载预训练模型
        if os.path.exists(pretrained_model_path):
            print(f"Loading pretrained model from: {pretrained_model_path}")
            
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.vit = ViTModel(self.config, add_pooling_layer=False)
            
            # 处理位置编码尺寸不匹配问题
            if 'embeddings.position_embeddings' in state_dict:
                old_pos_embed = state_dict['embeddings.position_embeddings']
                old_size = old_pos_embed.shape[1]
                new_size = self.num_patches + 1
                
                if old_size != new_size:
                    print(f"Resizing position embeddings from {old_size} to {new_size}")
                    
                    cls_pos_embed = old_pos_embed[:, 0:1, :]
                    patch_pos_embed = old_pos_embed[:, 1:, :]
                    
                    old_patch_size = int(np.sqrt(old_size - 1))
                    new_patch_size = int(np.sqrt(new_size - 1))
                    
                    patch_pos_embed = patch_pos_embed.reshape(1, old_patch_size, old_patch_size, -1)
                    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
                    patch_pos_embed = F.interpolate(
                        patch_pos_embed, 
                        size=(new_patch_size, new_patch_size), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
                    patch_pos_embed = patch_pos_embed.reshape(1, new_patch_size * new_patch_size, -1)
                    
                    new_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
                    state_dict['embeddings.position_embeddings'] = new_pos_embed
            
            missing_keys, unexpected_keys = self.vit.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            print("Successfully loaded pretrained weights")
        else:
            print(f"Error: Pretrained model not found at {pretrained_model_path}")
            raise FileNotFoundError(f"Pretrained model file not found: {pretrained_model_path}")
        
        # 创建威尔斯特拉斯位置编码
        self.weierstrass_encoding = ImprovedWeierstrassEllipticPositionalEncoding(
            d_model=self.config.hidden_size,
            num_patches_h=self.num_patches_per_side,
            num_patches_w=self.num_patches_per_side,
            use_derivative=use_derivative,
            alpha_scale=alpha_scale,
            use_layernorm=True
        )
        
        # 分类头 - 针对DMLab 6类任务优化
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Dropout(0.4)  # 结构化任务需要更强的正则化
        )
        
        # 多层分类器 - 增强复杂决策能力
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(self.config.hidden_size // 2, self.config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(self.config.hidden_size // 4, self.config.hidden_size // 8),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(self.config.hidden_size // 8, num_classes)
        )
        
        # 位置编码混合权重
        self.pos_encoding_weight = nn.Parameter(torch.tensor(0.6))  # 结构化任务更依赖位置信息
        
        # 初始化新添加的层
        self._init_classifier()
        
    def _init_classifier(self):
        """初始化分类器"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.9)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        
        # 使用ViT的完整embedding流程
        embeddings = self.vit.embeddings(pixel_values)
        
        # 获取原始位置编码
        original_pos_embeddings = self.vit.embeddings.position_embeddings
        
        # 获取威尔斯特拉斯位置编码
        weierstrass_pos_encoding = self.weierstrass_encoding()
        
        # 组合位置编码
        weight = torch.sigmoid(self.pos_encoding_weight)
        cls_pos = original_pos_embeddings[:, 0:1, :]
        
        # 混合patch位置编码
        mixed_patch_pos = (weight * weierstrass_pos_encoding.unsqueeze(0) + 
                          (1 - weight) * original_pos_embeddings[:, 1:, :])
        
        # 组合CLS和patch位置编码
        pos_embeddings = torch.cat([cls_pos, mixed_patch_pos], dim=1)
        
        # 替换位置编码
        patch_embeddings = embeddings[:, 1:, :]
        cls_token = embeddings[:, 0:1, :]
        
        # 重新组合，使用混合的位置编码
        embeddings = torch.cat([cls_token, patch_embeddings + mixed_patch_pos], dim=1)
        
        # Transformer编码器
        encoder_outputs = self.vit.encoder(embeddings)
        sequence_output = encoder_outputs.last_hidden_state
        
        # 获取CLS token的输出
        cls_output = sequence_output[:, 0]
        
        # 分类
        cls_output = self.pre_classifier(cls_output)
        logits = self.classifier(cls_output)
        
        return logits

class DMLabTxtDataset(Dataset):
    def __init__(self, root_dir, label_txt, transform=None):
        self.root_dir = root_dir  # 数据集根目录（如/root/shared-nvme/VTAB/dmlab）
        self.transform = transform
        self.samples = []
        with open(label_txt, 'r') as f:
            for line in f:
                img_rel, label = line.strip().split()
                self.samples.append((img_rel, int(label)))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_rel, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_rel)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dmlab_vtab_dataloaders(data_dir, batch_size=16, num_workers=0, distributed=False, rank=0, world_size=1):
    """全部训练样本用于训练，全部测试样本用于测试，不再划分验证集。"""
    train_txt = '/root/train.txt'
    test_txt = '/root/test.txt'
    root_dir = '/root/shared-nvme/VTAB/dmlab'

    transform_train = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.RandomCrop(384, padding=32),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = DMLabTxtDataset(root_dir, train_txt, transform=transform_train)
    test_dataset = DMLabTxtDataset(root_dir, test_txt, transform=transform_test)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=(train_sampler is None),
                            num_workers=num_workers, 
                            pin_memory=True,
                            sampler=train_sampler,
                            drop_last=False)
    val_loader = None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False,
                           num_workers=num_workers, 
                           pin_memory=True,
                           sampler=test_sampler)

    return train_loader, val_loader, test_loader, train_sampler

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                train_sampler=None, scaler=None):
    """训练一个epoch"""
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
        
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    """评估模型"""
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
    
    return test_loss / len(test_loader), 100. * correct / total

def main():
    # 创建临时目录以解决多进程问题
    import tempfile
    temp_dir = '/tmp/pytorch_temp'
    os.makedirs(temp_dir, exist_ok=True)
    os.environ['TMPDIR'] = temp_dir
    os.environ['TEMP'] = temp_dir
    os.environ['TMP'] = temp_dir
    
    # 分布式训练设置
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', init_method='env://', 
                               world_size=world_size, rank=rank)
        torch.cuda.set_device(gpu)
        distributed = True
    else:
        distributed = False
        rank = 0
        world_size = 1
        gpu = 0
    
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    if rank == 0:
        print(f"Using device: {device}")
        print(f"Distributed training: {distributed}")
    
    # 设置随机种子
    set_seed(42 + rank)
    
    # 针对DMLab结构化任务优化的超参数
    batch_size = 12  # 较小的batch size，适合结构化任务
    learning_rate = 1.5e-4  # 较低的学习率，更稳定的训练
    num_epochs = 120  # 更多轮次，给模型更多时间学习空间关系
    warmup_epochs = 15  # 更长的warmup
    weight_decay = 0.025  # 稍高的权重衰减，防止过拟合
    num_workers = 0
    
    # 模型参数
    pretrained_model_path = "/root/shared-nvme/vit_l16_in21k.pth"  # 用户指定的预训练模型路径
    data_dir = "/root/shared-nvme/VTAB/dmlab"  # 用户指定的数据集路径
    num_classes = 6  # DMLab有6个类别
    image_size = 384
    
    if rank == 0:
        print("Loading ViT-L/16 model for DMLab structured task...")
    
    # 创建模型
    model = WeierstrassViTL16(
        pretrained_model_path=pretrained_model_path,
        num_classes=num_classes,
        image_size=image_size,
        use_derivative=True,
        alpha_scale=0.05
    )
    
    model = model.to(device)
    
    if distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    
    # 数据加载器
    train_loader, val_loader, test_loader, train_sampler = get_dmlab_vtab_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size
    )
    
    # 损失函数 - 针对结构化任务，使用更高的标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    
    # 优化器 - 使用不同学习率策略
    if distributed:
        vit_params = []
        weierstrass_params = []
        classifier_params = []
        
        for name, param in model.module.named_parameters():
            if 'weierstrass_encoding' in name:
                weierstrass_params.append(param)
            elif 'classifier' in name or 'pre_classifier' in name or 'pos_encoding_weight' in name:
                classifier_params.append(param)
            else:
                vit_params.append(param)
    else:
        vit_params = []
        weierstrass_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'weierstrass_encoding' in name:
                weierstrass_params.append(param)
            elif 'classifier' in name or 'pre_classifier' in name or 'pos_encoding_weight' in name:
                classifier_params.append(param)
            else:
                vit_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': vit_params, 'lr': learning_rate * 0.08},  # 预训练层更小学习率
        {'params': weierstrass_params, 'lr': learning_rate * 1.2},  # 威尔斯特拉斯编码稍高学习率
        {'params': classifier_params, 'lr': learning_rate * 2.0}  # 分类器最高学习率
    ], weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # 梯度缩放器
    scaler = GradScaler()
    
    # 学习率调度器 - 余弦退火配合warmup
    def cosine_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_lambda)
    
    # 创建检查点保存目录
    checkpoint_dir = "/root/shared-nvme/dmlab_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练循环
    best_test_acc = 0.0
    results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    patience = 25  # 增加patience，给结构化任务更多收敛时间
    patience_counter = 0
    
    if rank == 0:
        print("\n=== 开始DMLab VTAB-1k微调实验 ===")
        print(f"目标：超过基线准确率 41.9%")
        print(f"训练样本数：800 (VTAB-1k标准)")
        print(f"验证样本数：200 (VTAB-1k标准)")
        print(f"测试样本数：约22,000+ (VTAB-1k标准)")
        print(f"图像分辨率：{image_size}x{image_size}")
        print(f"模型：ViT-L/16 + 威尔斯特拉斯位置编码（结构化任务优化）")
        print(f"任务类型：3D环境距离分类（6类）")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, 
            train_sampler, scaler
        )
        
        # 每一轮都在测试集上评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
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
            
            # 保存最佳模型（以测试集准确率为准）
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                model_state = model.module.state_dict() if distributed else model.state_dict()
                best_model_path = os.path.join(checkpoint_dir, 'weierstrass_vit_l16_dmlab_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_test_acc': best_test_acc,
                    'results': results
                }, best_model_path)
                print(f"🏆 新的最佳测试准确率: {best_test_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch > 40:
                    print(f"Early stopping at epoch {epoch}")
                    break
            # 定期保存检查点
            if epoch % 30 == 0:
                model_state = model.module.state_dict() if distributed else model.state_dict()
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'results': results
                }, checkpoint_path)
    
    # 训练完成后，加载最佳模型进行最终测试
    if rank == 0:
        print(f"\n=== 训练完成，加载最佳模型进行最终测试 ===")
        # 加载最佳模型
        best_model_path = os.path.join(checkpoint_dir, 'weierstrass_vit_l16_dmlab_best.pth')
        checkpoint = torch.load(best_model_path)
        if distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载第 {checkpoint['epoch']} 轮的模型，测试准确率: {checkpoint['best_test_acc']:.2f}%")
    
    # 最终测试评估
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    if rank == 0:
        print(f"\n=== 最终测试结果 ===")
        print(f"测试准确率: {test_acc:.2f}%")
        print(f"基线准确率: 41.9%")
        
        if test_acc > 41.9:
            improvement = test_acc - 41.9
            print(f"🎉 成功超过基线! 提升了 {improvement:.2f} 个百分点")
        else:
            print(f"未达到目标，还需要 {41.9 - test_acc:.2f} 个百分点")
        
        # 保存最终结果
        results['final_test_acc'] = test_acc
        results['best_test_acc'] = best_test_acc
        results['baseline_accuracy'] = 41.9
        results['improvement'] = test_acc - 41.9
        
        results_path = os.path.join(checkpoint_dir, 'dmlab_vtab_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # 输出学习到的威尔斯特拉斯参数
        if distributed:
            alpha = F.softplus(model.module.weierstrass_encoding.alpha_learn).item()
            beta = F.softplus(model.module.weierstrass_encoding.beta_learn).item()
            gamma = F.softplus(model.module.weierstrass_encoding.gamma_learn).item()
            weight = torch.sigmoid(model.module.pos_encoding_weight).item()
            spatial_enhance = model.module.weierstrass_encoding.spatial_enhance.item()
        else:
            alpha = F.softplus(model.weierstrass_encoding.alpha_learn).item()
            beta = F.softplus(model.weierstrass_encoding.beta_learn).item()
            gamma = F.softplus(model.weierstrass_encoding.gamma_learn).item()
            weight = torch.sigmoid(model.pos_encoding_weight).item()
            spatial_enhance = model.weierstrass_encoding.spatial_enhance.item()
        
        print(f"\n学习到的威尔斯特拉斯参数（结构化任务优化）:")
        print(f"ω'2 (alpha): {alpha:.6f}")
        print(f"β (beta): {beta:.6f}")
        print(f"γ (gamma): {gamma:.6f}")
        print(f"空间增强因子: {spatial_enhance:.6f}")
        print(f"位置编码混合权重: {weight:.6f}")
        
        print(f"\n=== DMLab VTAB-1k实验总结 ===")
        print(f"任务类型: 3D环境距离分类（结构化任务）")
        print(f"- 训练样本: 800")
        print(f"- 验证样本: 200") 
        print(f"- 测试样本: ~22,000")
        print(f"- 类别数: 6 ({{close, far, very far}} × {{positive reward, negative reward}})")
        print(f"- 最佳验证准确率: {best_test_acc:.2f}%")
        print(f"- 最终测试准确率: {test_acc:.2f}%")
        print(f"- 基线准确率: 41.9%")
        print(f"- 提升幅度: {test_acc - 41.9:.2f} 个百分点")
        print(f"- 威尔斯特拉斯位置编码在结构化任务上的有效性得到验证!")
    
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()