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
import scipy.io as sio
from PIL import Image
import os
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
    """改进的威尔斯特拉斯椭圆函数位置编码 - 针对ViT-L/16优化"""
    
    def __init__(self, d_model, num_patches_h, num_patches_w, 
                 use_derivative=True, alpha_scale=0.03, use_layernorm=True):
        super().__init__()
        self.d_model = d_model
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.use_derivative = use_derivative
        self.alpha_scale = alpha_scale
        self.use_layernorm = use_layernorm
        
        # 威尔斯特拉斯椭圆函数的精确半周期
        self.omega1 = gamma(0.25)**2 / (2 * np.sqrt(2 * np.pi))
        
        # 可学习参数
        self.alpha_learn = nn.Parameter(torch.tensor(0.5))
        self.beta_learn = nn.Parameter(torch.tensor(0.1))
        self.gamma_learn = nn.Parameter(torch.tensor(0.2))  # 新增参数
        
        # 多层投影网络
        input_dim = 6 if use_derivative else 3  # 增加维度
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.LayerNorm(d_model // 2) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(d_model, d_model)
        )
        
        # 可学习的缩放和偏移
        self.scale_factor = nn.Parameter(torch.tensor(alpha_scale))
        self.offset = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 频率调制参数
        freq_dim = 6 if use_derivative else 3
        self.freq_modulation = nn.Parameter(torch.ones(freq_dim))
        
        # 初始化
        self._init_parameters()
        
    def _init_parameters(self):
        """Xavier初始化"""
        with torch.no_grad():
            for m in self.projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.8)
                    nn.init.zeros_(m.bias)
    
    def stable_weierstrass_p(self, z):
        """数值稳定的威尔斯特拉斯椭圆函数"""
        omega2_prime = 0.1 + F.softplus(self.alpha_learn)
        beta = 0.01 + F.softplus(self.beta_learn)
        gamma_param = 0.1 + F.softplus(self.gamma_learn)
        
        z_real = z.real
        z_imag = z.imag
        z_mod = torch.sqrt(z_real**2 + z_imag**2)
        z_mod_safe = torch.clamp(z_mod, min=1e-6)
        
        # 主项
        main_term = 1.0 / (z_mod_safe**2 + beta)
        
        # 周期修正项
        u = z_real / self.omega1
        v = z_imag / omega2_prime
        
        correction = torch.zeros_like(z_real)
        for k in range(1, 4):
            freq_factor = k * math.pi
            amp_k = gamma_param / k**2
            
            correction += amp_k * (
                torch.cos(freq_factor * u) * torch.exp(-freq_factor * torch.abs(v)) +
                torch.sin(freq_factor * v) * torch.exp(-freq_factor * torch.abs(u))
            )
        
        p_z_real = main_term * torch.cos(torch.atan2(z_imag, z_real)) + correction
        p_z_imag = main_term * torch.sin(torch.atan2(z_imag, z_real)) + correction * 0.7
        
        return torch.complex(p_z_real, p_z_imag)
    
    def stable_weierstrass_p_derivative(self, z):
        """威尔斯特拉斯函数导数"""
        omega2_prime = 0.1 + F.softplus(self.alpha_learn)
        beta = 0.01 + F.softplus(self.beta_learn)
        gamma_param = 0.1 + F.softplus(self.gamma_learn)
        
        z_mod = torch.sqrt(z.real**2 + z.imag**2)
        z_mod_safe = torch.clamp(z_mod, min=1e-6)
        
        main_deriv = -2.0 / (z_mod_safe**3 + beta)
        
        u = z.real / self.omega1
        v = z.imag / omega2_prime
        
        deriv_correction = torch.zeros_like(z.real)
        for k in range(1, 3):
            freq_factor = k * math.pi
            amp_k = gamma_param / k**2
            
            deriv_correction += amp_k * k * (
                -torch.sin(freq_factor * u) * torch.exp(-freq_factor * torch.abs(v)) +
                torch.cos(freq_factor * v) * torch.exp(-freq_factor * torch.abs(u))
            )
        
        p_deriv_real = main_deriv * torch.cos(torch.atan2(z.imag, z.real)) + deriv_correction
        p_deriv_imag = main_deriv * torch.sin(torch.atan2(z.imag, z.real)) + deriv_correction * 0.5
        
        return torch.complex(p_deriv_real, p_deriv_imag)
    
    def forward(self, num_patches_h=None, num_patches_w=None):
        """生成位置编码"""
        if num_patches_h is None:
            num_patches_h = self.num_patches_h
        if num_patches_w is None:
            num_patches_w = self.num_patches_w
            
        device = self.alpha_learn.device
        
        # 创建网格坐标
        row_idx = torch.arange(num_patches_h, dtype=torch.float32, device=device)
        col_idx = torch.arange(num_patches_w, dtype=torch.float32, device=device)
        
        # 改进的归一化
        u = (col_idx + 0.5) / (num_patches_w + 1e-8)
        v = (row_idx + 0.5) / (num_patches_h + 1e-8)
        
        u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
        u_grid = u_grid.T.reshape(-1)
        v_grid = v_grid.T.reshape(-1)
        
        omega2_prime = 0.1 + F.softplus(self.alpha_learn)
        
        # 映射到复平面，增加非线性变换
        z_real = u_grid * self.omega1 + 0.02 * torch.sin(3 * math.pi * u_grid)
        z_imag = v_grid * omega2_prime + 0.02 * torch.cos(3 * math.pi * v_grid)
        z = torch.complex(z_real, z_imag)
        
        # 计算威尔斯特拉斯函数值
        p_z = self.stable_weierstrass_p(z)
        
        # 分离实部和虚部
        p_real = p_z.real * self.freq_modulation[0]
        p_imag = p_z.imag * self.freq_modulation[1]
        
        # 改进的激活函数
        pe_real = torch.tanh(self.scale_factor * p_real)
        pe_imag = torch.tanh(self.scale_factor * p_imag * 0.8)
        
        # 添加径向分量
        radius = torch.sqrt(u_grid**2 + v_grid**2)
        pe_radius = torch.tanh(self.scale_factor * radius * self.freq_modulation[2])
        
        if self.use_derivative:
            p_prime = self.stable_weierstrass_p_derivative(z)
            p_prime_real = torch.tanh(self.scale_factor * p_prime.real * self.freq_modulation[3])
            p_prime_imag = torch.tanh(self.scale_factor * p_prime.imag * self.freq_modulation[4])
            p_prime_radius = torch.tanh(self.scale_factor * p_prime.real.abs() * self.freq_modulation[5])
            
            features = torch.stack([pe_real, pe_imag, pe_radius, 
                                  p_prime_real, p_prime_imag, p_prime_radius], dim=-1)
        else:
            features = torch.stack([pe_real, pe_imag, pe_radius], dim=-1)
        
        # 投影到d_model维
        pos_encoding = self.projection(features)
        pos_encoding = pos_encoding + self.offset
        
        return pos_encoding.squeeze(0) if pos_encoding.dim() == 3 else pos_encoding

class WeierstrassViTL16(nn.Module):
    """基于本地预训练ViT-L/16的威尔斯特拉斯位置编码模型"""
    
    def __init__(self, pretrained_model_path="/root/shared-nvme/vit_l16_in21k.pth", 
                 num_classes=10, image_size=384, use_derivative=True, alpha_scale=0.03):
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
            image_size=image_size,  # 使用传入的image_size
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
            
            # 加载本地权重
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                # 如果是完整的checkpoint格式
                state_dict = checkpoint['model_state_dict']
            else:
                # 如果是直接的state_dict格式
                state_dict = checkpoint
            
            # 创建ViT模型实例
            self.vit = ViTModel(self.config, add_pooling_layer=False)
            
            # 处理位置编码尺寸不匹配问题
            if 'embeddings.position_embeddings' in state_dict:
                old_pos_embed = state_dict['embeddings.position_embeddings']
                old_size = old_pos_embed.shape[1]  # 应该是197 (224x224)
                new_size = self.num_patches + 1    # 应该是577 (384x384)
                
                if old_size != new_size:
                    print(f"Resizing position embeddings from {old_size} to {new_size}")
                    
                    # 获取CLS token的位置编码
                    cls_pos_embed = old_pos_embed[:, 0:1, :]
                    
                    # 获取patch位置编码
                    patch_pos_embed = old_pos_embed[:, 1:, :]
                    
                    # 重塑patch位置编码以匹配新的尺寸
                    old_patch_size = int(np.sqrt(old_size - 1))  # 14 (224/16)
                    new_patch_size = int(np.sqrt(new_size - 1))  # 24 (384/16)
                    
                    # 重塑为2D网格
                    patch_pos_embed = patch_pos_embed.reshape(1, old_patch_size, old_patch_size, -1)
                    
                    # 使用双线性插值调整尺寸
                    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # (1, D, H, W)
                    patch_pos_embed = F.interpolate(
                        patch_pos_embed, 
                        size=(new_patch_size, new_patch_size), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # (1, H, W, D)
                    patch_pos_embed = patch_pos_embed.reshape(1, new_patch_size * new_patch_size, -1)
                    
                    # 组合CLS token和新的patch位置编码
                    new_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
                    
                    # 更新state_dict
                    state_dict['embeddings.position_embeddings'] = new_pos_embed
            
            # 加载权重
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
        
        # 更新embedding层以处理384x384分辨率
        # 对于384×384分辨率，patch数量为 (384/16)^2 = 576
        # 位置编码的调整在加载权重时处理
        
        # 分类头
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Dropout(0.5)  # 增加dropout防止过拟合
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.4),  # 增加dropout
            nn.Linear(self.config.hidden_size // 2, self.config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.3),  # 增加dropout
            nn.Linear(self.config.hidden_size // 4, num_classes)
        )
        
        # 位置编码混合权重
        self.pos_encoding_weight = nn.Parameter(torch.tensor(0.5))
        
        # 初始化新添加的层
        self._init_classifier()
        
    def _init_classifier(self):
        """初始化分类器"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        
        # 使用ViT的完整embedding流程
        embeddings = self.vit.embeddings(pixel_values)
        
        # 获取原始位置编码
        original_pos_embeddings = self.vit.embeddings.position_embeddings
        
        # 获取威尔斯特拉斯位置编码
        weierstrass_pos_encoding = self.weierstrass_encoding()  # (num_patches, D)
        
        # 组合位置编码
        weight = torch.sigmoid(self.pos_encoding_weight)
        cls_pos = original_pos_embeddings[:, 0:1, :]  # CLS token位置编码
        
        # 混合patch位置编码
        mixed_patch_pos = (weight * weierstrass_pos_encoding.unsqueeze(0) + 
                          (1 - weight) * original_pos_embeddings[:, 1:, :])
        
        # 组合CLS和patch位置编码
        pos_embeddings = torch.cat([cls_pos, mixed_patch_pos], dim=1)
        
        # 替换位置编码
        # 首先移除原始的位置编码
        patch_embeddings = embeddings[:, 1:, :]  # 移除CLS token
        cls_token = embeddings[:, 0:1, :]  # 保留CLS token
        
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

class SVHNVTabDataset(Dataset):
    """SVHN VTAB-1k数据集，使用固定的数据划分"""
    
    def __init__(self, train_mat_path, test_mat_path, splits_json_path, 
                 split='train', transform=None, train_ratio=0.8):
        """
        Args:
            train_mat_path: 训练数据.mat文件路径
            test_mat_path: 测试数据.mat文件路径  
            splits_json_path: VTAB划分JSON文件路径
            split: 'train', 'val', 'test'
            transform: 数据变换
            train_ratio: 训练集在1000个样本中的比例（默认0.8，即800个训练，200个验证）
        """
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        
        # 加载VTAB划分
        with open(splits_json_path, 'r') as f:
            splits_data = json.load(f)
        
        # 获取SVHN的1000个样本文件名
        svhn_splits = splits_data['svhn']
        self.trainval_filenames = svhn_splits['train'] + svhn_splits['val']
        
        # 将文件名转换为索引（去掉.png后缀）
        self.trainval_indices = []
        for filename in self.trainval_filenames:
            # 从文件名提取索引，例如 "1400.png" -> 1400
            index = int(filename.replace('.png', ''))
            self.trainval_indices.append(index)
        
        # 加载.mat文件数据
        print(f"Loading SVHN data from {train_mat_path} and {test_mat_path}")
        train_data = sio.loadmat(train_mat_path)
        test_data = sio.loadmat(test_mat_path)
        
        # 合并训练和测试数据
        # SVHN数据格式: (32, 32, 3, N) -> 转换为 (N, 3, 32, 32)
        train_images = train_data['X'].transpose(3, 2, 0, 1)  # (32, 32, 3, N) -> (N, 3, 32, 32)
        test_images = test_data['X'].transpose(3, 2, 0, 1)   # (32, 32, 3, N) -> (N, 3, 32, 32)
        
        self.images = np.concatenate([train_images, test_images], axis=0)
        
        self.labels = np.concatenate([
            train_data['y'].flatten(),
            test_data['y'].flatten()
        ], axis=0)
        
        # SVHN标签从1开始，需要转换为0-9
        self.labels = (self.labels - 1) % 10
        
        print(f"Total images: {len(self.images)}")
        print(f"Total labels: {len(self.labels)}")
        print(f"VTAB trainval indices: {len(self.trainval_indices)}")
        
        # 根据split选择数据
        if split in ['train', 'val']:
            # 使用VTAB的1000个样本
            num_train = int(len(self.trainval_indices) * train_ratio)
            
            if split == 'train':
                # 前800个样本用于训练
                self.indices = self.trainval_indices[:num_train]
            else:  # val
                # 后200个样本用于验证
                self.indices = self.trainval_indices[num_train:]
        else:  # test
            # 使用所有不在VTAB 1000个样本中的数据作为测试集
            all_indices = set(range(len(self.images)))
            trainval_set = set(self.trainval_indices)
            self.indices = list(all_indices - trainval_set)
        
        print(f"{split} split: {len(self.indices)} samples")
        
        # 验证索引的有效性
        max_index = max(self.indices)
        if max_index >= len(self.images):
            raise ValueError(f"Index {max_index} exceeds dataset size {len(self.images)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取实际的数据索引
        data_idx = self.indices[idx]
        
        # 获取图像和标签
        image = self.images[data_idx]  # (3, 32, 32)
        label = self.labels[data_idx]
        
        # 确保图像数据类型正确并转换为PIL图像
        image = image.transpose(1, 2, 0)  # (32, 32, 3)
        image = image.astype(np.uint8)  # 确保是uint8类型
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_svhn_vtab_dataloaders(batch_size=32, num_workers=0, distributed=False, rank=0, world_size=1):
    """获取SVHN VTAB-1k数据加载器，使用固定的数据划分"""
    
    # 数据路径
    train_mat_path = "/root/shared-nvme/VTAB/SVHN/train_32x32.mat"
    test_mat_path = "/root/shared-nvme/VTAB/SVHN/test_32x32.mat"
    splits_json_path = "/root/svhn_trainval_splits.json"
    
    # 检查文件是否存在
    for path in [train_mat_path, test_mat_path, splits_json_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
    
    # SVHN数据预处理
    transform_train = transforms.Compose([
        transforms.Resize(384),  # 目标分辨率
        transforms.RandomCrop(384, padding=32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # 增加旋转角度
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),  # 增强颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.3), ratio=(0.3, 3.3)),  # 增强随机擦除
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 创建数据集
    train_dataset = SVHNVTabDataset(
        train_mat_path=train_mat_path,
        test_mat_path=test_mat_path,
        splits_json_path=splits_json_path,
        split='train',
        transform=transform_train,
        train_ratio=0.8
    )
    
    val_dataset = SVHNVTabDataset(
        train_mat_path=train_mat_path,
        test_mat_path=test_mat_path,
        splits_json_path=splits_json_path,
        split='val',
        transform=transform_val,
        train_ratio=0.8
    )
    
    test_dataset = SVHNVTabDataset(
        train_mat_path=train_mat_path,
        test_mat_path=test_mat_path,
        splits_json_path=splits_json_path,
        split='test',
        transform=transform_test,
        train_ratio=0.8
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=(train_sampler is None),
                            num_workers=num_workers, 
                            pin_memory=True,
                            sampler=train_sampler,
                            drop_last=False)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False,
                          num_workers=num_workers, 
                          pin_memory=True,
                          sampler=val_sampler)
    
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
    
    # 针对VTAB-1k优化的超参数 - 平衡学习率和正则化
    batch_size = 16  # 较小的batch size适合小数据集
    learning_rate = 2e-4  # 适中的学习率
    num_epochs = 100  # VTAB-1k设置
    warmup_epochs = 10  # 适中的warmup轮数
    weight_decay = 0.02  # 适中的权重衰减
    num_workers = 0  # 禁用多进程以避免临时目录问题
    
    # 模型参数
    pretrained_model_path = "/root/shared-nvme/vit_l16_in21k.pth"
    num_classes = 10  # SVHN有10个类别
    image_size = 384
    
    if rank == 0:
        print("Loading ViT-L/16 model from local pretrained weights...")
    
    # 创建模型
    model = WeierstrassViTL16(
        pretrained_model_path=pretrained_model_path,
        num_classes=num_classes,
        image_size=image_size,
        use_derivative=True,
        alpha_scale=0.03
    )
    
    model = model.to(device)
    
    if distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    
    # 数据加载器
    train_loader, val_loader, test_loader, train_sampler = get_svhn_vtab_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size
    )
    
    # 损失函数 - 适中的标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 优化器 - 使用不同学习率
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
        {'params': vit_params, 'lr': learning_rate * 0.1},  # 预训练层较小学习率
        {'params': weierstrass_params, 'lr': learning_rate},  # 威尔斯特拉斯编码正常学习率
        {'params': classifier_params, 'lr': learning_rate * 1.5}  # 分类器较大学习率
    ], weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # 梯度缩放器
    scaler = GradScaler()
    
    # 学习率调度器 - 余弦退火
    def cosine_lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_lambda)
    
    # 创建检查点保存目录
    checkpoint_dir = "/root/shared-nvme/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练循环
    best_val_acc = 0.0
    results = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    patience = 20  # 增加patience防止过早停止
    patience_counter = 0
    
    if rank == 0:
        print("\n=== 开始SVHN VTAB-1k微调实验 ===")
        print(f"目标：超过基线准确率 80.9%")
        print(f"训练样本数：800 (VTAB-1k标准)")
        print(f"验证样本数：200 (VTAB-1k标准)")
        print(f"测试样本数：26,032 (VTAB-1k标准)")
        print(f"图像分辨率：{image_size}x{image_size}")
        print(f"模型：ViT-L/16 + 威尔斯特拉斯位置编码")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, 
            train_sampler, scaler
        )
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录结果
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        
        if rank == 0:
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型（基于验证集）
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                model_state = model.module.state_dict() if distributed else model.state_dict()
                best_model_path = os.path.join(checkpoint_dir, 'weierstrass_vit_l16_svhn_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'results': results
                }, best_model_path)
                
                print(f"🏆 新的最佳验证准确率: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch > 30:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # 定期保存检查点
            if epoch % 20 == 0:
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
        best_model_path = os.path.join(checkpoint_dir, 'weierstrass_vit_l16_svhn_best.pth')
        checkpoint = torch.load(best_model_path)
        if distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"加载第 {checkpoint['epoch']} 轮的模型，验证准确率: {checkpoint['best_val_acc']:.2f}%")
    
    # 最终测试评估
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    if rank == 0:
        print(f"\n=== 最终测试结果 ===")
        print(f"测试准确率: {test_acc:.2f}%")
        print(f"基线准确率: 80.9%")
        
        if test_acc > 80.9:
            improvement = test_acc - 80.9
            print(f"🎉 成功超过基线! 提升了 {improvement:.2f} 个百分点")
        else:
            print(f"未达到目标，还需要 {80.9 - test_acc:.2f} 个百分点")
        
        # 保存最终结果
        results['final_test_acc'] = test_acc
        results['best_val_acc'] = best_val_acc
        results['baseline_accuracy'] = 80.9
        results['improvement'] = test_acc - 80.9
        
        results_path = os.path.join(checkpoint_dir, 'svhn_vtab_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # 输出学习到的威尔斯特拉斯参数
        if distributed:
            alpha = F.softplus(model.module.weierstrass_encoding.alpha_learn).item()
            beta = F.softplus(model.module.weierstrass_encoding.beta_learn).item()
            gamma = F.softplus(model.module.weierstrass_encoding.gamma_learn).item()
            weight = torch.sigmoid(model.module.pos_encoding_weight).item()
        else:
            alpha = F.softplus(model.weierstrass_encoding.alpha_learn).item()
            beta = F.softplus(model.weierstrass_encoding.beta_learn).item()
            gamma = F.softplus(model.weierstrass_encoding.gamma_learn).item()
            weight = torch.sigmoid(model.pos_encoding_weight).item()
        
        print(f"\n学习到的威尔斯特拉斯参数:")
        print(f"ω'2 (alpha): {alpha:.6f}")
        print(f"β (beta): {beta:.6f}")
        print(f"γ (gamma): {gamma:.6f}")
        print(f"位置编码混合权重: {weight:.6f}")
        
        print(f"\n=== 实验总结 ===")
        print(f"VTAB-1k标准配置:")
        print(f"- 训练样本: 800")
        print(f"- 验证样本: 200") 
        print(f"- 测试样本: 26,032")
        print(f"- 最佳验证准确率: {best_val_acc:.2f}%")
        print(f"- 最终测试准确率: {test_acc:.2f}%")
        print(f"- 基线准确率: 80.9%")
        print(f"- 提升幅度: {test_acc - 80.9:.2f} 个百分点")
    
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()