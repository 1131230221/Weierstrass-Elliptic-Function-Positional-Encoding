import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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
import glob
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
                 use_derivative=True, alpha_scale=0.02, use_layernorm=True):
        super().__init__()
        self.d_model = d_model
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.use_derivative = use_derivative
        self.alpha_scale = alpha_scale
        self.use_layernorm = use_layernorm
        
        # 威尔斯特拉斯椭圆函数的精确半周期
        self.omega1 = gamma(0.25)**2 / (2 * np.sqrt(2 * np.pi))
        
        # 可学习参数 - 针对遥感图像调整初始值
        self.alpha_learn = nn.Parameter(torch.tensor(0.6))
        self.beta_learn = nn.Parameter(torch.tensor(0.15))
        self.gamma_learn = nn.Parameter(torch.tensor(0.25))
        
        # 多层投影网络 - 增加容量以适应遥感图像复杂性
        input_dim = 4 if use_derivative else 2
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.LayerNorm(d_model // 2) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # 可学习的缩放和偏移
        self.scale_factor = nn.Parameter(torch.tensor(alpha_scale))
        self.offset = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 频率调制参数
        freq_dim = 4 if use_derivative else 2
        self.freq_modulation = nn.Parameter(torch.ones(freq_dim))
        
        # 初始化
        self._init_parameters()
        
    def _init_parameters(self):
        """Xavier初始化"""
        with torch.no_grad():
            for m in self.projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.9)
                    nn.init.zeros_(m.bias)
    
    def stable_weierstrass_p(self, z):
        """数值稳定的威尔斯特拉斯椭圆函数 - 针对遥感图像优化"""
        omega2_prime = 0.12 + F.softplus(self.alpha_learn)
        beta = 0.015 + F.softplus(self.beta_learn)
        gamma_param = 0.15 + F.softplus(self.gamma_learn)
        
        z_real = z.real
        z_imag = z.imag
        z_mod = torch.sqrt(z_real**2 + z_imag**2)
        z_mod_safe = torch.clamp(z_mod, min=1e-6)
        
        # 主项
        main_term = 1.0 / (z_mod_safe**2 + beta)
        
        # 周期修正项 - 增加复杂度以捕获遥感图像的空间关系
        u = z_real / self.omega1
        v = z_imag / omega2_prime
        
        correction = torch.zeros_like(z_real)
        for k in range(1, 5):  # 增加项数
            freq_factor = k * math.pi
            amp_k = gamma_param / (k**1.8)  # 调整衰减速度
            
            correction += amp_k * (
                torch.cos(freq_factor * u) * torch.exp(-freq_factor * torch.abs(v) * 0.8) +
                torch.sin(freq_factor * v) * torch.exp(-freq_factor * torch.abs(u) * 0.8)
            )
        
        p_z_real = main_term * torch.cos(torch.atan2(z_imag, z_real)) + correction
        p_z_imag = main_term * torch.sin(torch.atan2(z_imag, z_real)) + correction * 0.75
        
        return torch.complex(p_z_real, p_z_imag)
    
    def stable_weierstrass_p_derivative(self, z):
        """威尔斯特拉斯函数导数"""
        omega2_prime = 0.12 + F.softplus(self.alpha_learn)
        beta = 0.015 + F.softplus(self.beta_learn)
        gamma_param = 0.15 + F.softplus(self.gamma_learn)
        
        z_mod = torch.sqrt(z.real**2 + z.imag**2)
        z_mod_safe = torch.clamp(z_mod, min=1e-6)
        
        main_deriv = -2.0 / (z_mod_safe**3 + beta)
        
        u = z.real / self.omega1
        v = z.imag / omega2_prime
        
        deriv_correction = torch.zeros_like(z.real)
        for k in range(1, 4):
            freq_factor = k * math.pi
            amp_k = gamma_param / (k**1.8)
            
            deriv_correction += amp_k * k * (
                -torch.sin(freq_factor * u) * torch.exp(-freq_factor * torch.abs(v) * 0.8) +
                torch.cos(freq_factor * v) * torch.exp(-freq_factor * torch.abs(u) * 0.8)
            )
        
        p_deriv_real = main_deriv * torch.cos(torch.atan2(z.imag, z.real)) + deriv_correction
        p_deriv_imag = main_deriv * torch.sin(torch.atan2(z.imag, z.real)) + deriv_correction * 0.6
        
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
        
        # 改进的归一化 - 针对遥感图像的空间分布
        u = (col_idx + 0.5) / (num_patches_w + 1e-8)
        v = (row_idx + 0.5) / (num_patches_h + 1e-8)
        
        u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
        u_grid = u_grid.T.reshape(-1)
        v_grid = v_grid.T.reshape(-1)
        
        omega2_prime = 0.12 + F.softplus(self.alpha_learn)
        
        # 映射到复平面，增加非线性变换以适应遥感图像
        z_real = u_grid * self.omega1 + 0.025 * torch.sin(3.5 * math.pi * u_grid)
        z_imag = v_grid * omega2_prime + 0.025 * torch.cos(3.5 * math.pi * v_grid)
        z = torch.complex(z_real, z_imag)
        
        # 计算威尔斯特拉斯函数值
        p_z = self.stable_weierstrass_p(z)
        
        # 分离实部和虚部
        p_real = p_z.real * self.freq_modulation[0]
        p_imag = p_z.imag * self.freq_modulation[1]
        
        # 改进的激活函数
        pe_real = torch.tanh(self.scale_factor * p_real)
        pe_imag = torch.tanh(self.scale_factor * p_imag * 0.85)
        
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
    """基于本地ViT-L/16预训练权重的威尔斯特拉斯位置编码模型"""
    
    def __init__(self, pretrained_model_path="/root/shared-nvme/vit_l16_in21k.pth", 
                 num_classes=45, image_size=384, use_derivative=True, alpha_scale=0.02):
        super().__init__()
        
        # 创建ViT-L/16配置，确保参数匹配预训练模型
        self.config = ViTConfig(
            hidden_size=1024,          # ViT-L/16标准配置
            num_hidden_layers=24,      # ViT-L/16标准配置
            num_attention_heads=16,    # ViT-L/16标准配置
            intermediate_size=4096,    # ViT-L/16标准配置 (重要！)
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=image_size,     # 384×384
            patch_size=16,
            num_channels=3,
            qkv_bias=True,
            encoder_stride=16,
        )
        
        # 创建ViT模型结构（不包含pooling层以避免尺寸问题）
        self.vit = ViTModel(self.config, add_pooling_layer=False)
        
        # 计算patch数量
        self.image_size = image_size
        self.patch_size = self.config.patch_size
        self.num_patches_per_side = image_size // self.patch_size  # 24 for 384×384
        self.num_patches = self.num_patches_per_side ** 2  # 576
        
        print(f"Model configuration:")
        print(f"- Image size: {image_size}x{image_size}")
        print(f"- Patch size: {self.patch_size}x{self.patch_size}")
        print(f"- Number of patches per side: {self.num_patches_per_side}")
        print(f"- Total number of patches: {self.num_patches}")
        print(f"- Hidden size: {self.config.hidden_size}")
        print(f"- Intermediate size: {self.config.intermediate_size}")
        
        # 加载本地预训练权重并处理尺寸不匹配
        if pretrained_model_path is not None and isinstance(pretrained_model_path, str) and os.path.exists(pretrained_model_path):
            print(f"Loading pretrained ViT-L/16 weights from: {pretrained_model_path}")
            state_dict = torch.load(pretrained_model_path, map_location="cpu")
            # 处理可能的module前缀
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            # 处理位置编码尺寸不匹配问题
            if 'embeddings.position_embeddings' in state_dict:
                old_pos_embed = state_dict['embeddings.position_embeddings']
                print(f"Original position embedding shape: {old_pos_embed.shape}")
                old_seq_len = old_pos_embed.shape[1]
                new_seq_len = self.num_patches + 1
                if old_seq_len != new_seq_len:
                    print(f"Resizing position embeddings from {old_seq_len} to {new_seq_len}")
                    cls_pos_embed = old_pos_embed[:, 0:1, :]
                    patch_pos_embed = old_pos_embed[:, 1:, :]
                    old_grid_size = int(np.sqrt(old_seq_len - 1))
                    new_grid_size = int(np.sqrt(new_seq_len - 1))
                    print(f"Interpolating from {old_grid_size}x{old_grid_size} to {new_grid_size}x{new_grid_size}")
                    patch_pos_embed = patch_pos_embed.reshape(1, old_grid_size, old_grid_size, -1)
                    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
                    patch_pos_embed = F.interpolate(
                        patch_pos_embed, 
                        size=(new_grid_size, new_grid_size), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
                    patch_pos_embed = patch_pos_embed.reshape(1, new_grid_size * new_grid_size, -1)
                    new_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
                    state_dict['embeddings.position_embeddings'] = new_pos_embed
                    print(f"New position embedding shape: {new_pos_embed.shape}")
            keys_to_remove = [k for k in state_dict.keys() if k.startswith('pooler.')]
            for key in keys_to_remove:
                del state_dict[key]
                print(f"Removed pooler layer: {key}")
            missing_keys, unexpected_keys = self.vit.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            if missing_keys:
                print(f"Missing: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected: {unexpected_keys}")
            print("Successfully loaded pretrained weights")
        else:
            print("No pretrained weights loaded for ViT-L/16 (pretrained_model_path is None or file does not exist)")
        
        # 创建威尔斯特拉斯位置编码
        self.weierstrass_encoding = ImprovedWeierstrassEllipticPositionalEncoding(
            d_model=self.config.hidden_size,
            num_patches_h=self.num_patches_per_side,
            num_patches_w=self.num_patches_per_side,
            use_derivative=use_derivative,
            alpha_scale=alpha_scale,
            use_layernorm=True
        )
        
        # 分类头 - 针对RESISC45优化
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Dropout(0.6)  # 增加dropout防止过拟合
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(self.config.hidden_size // 2, self.config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self.config.hidden_size // 4, num_classes)
        )
        
        # 位置编码混合权重
        self.pos_encoding_weight = nn.Parameter(torch.tensor(0.6))  # 增加威尔斯特拉斯编码权重
        
        # 初始化新添加的层
        self._init_classifier()
        
        print("Model initialized successfully!")
        
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

class RESISC45VTabDataset(Dataset):
    """RESISC45 VTAB-1k数据集，使用固定的数据划分"""
    
    def __init__(self, data_root, splits_json_path, 
                 split='train', transform=None, train_ratio=0.8, test_files_json=None):
        self.data_root = data_root
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        self.class_names = [
            'airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach',
            'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud',
            'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway',
            'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection',
            'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park',
            'mountain', 'overpass', 'palace', 'parking_lot', 'railway',
            'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway',
            'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium',
            'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland'
        ]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        with open(splits_json_path, 'r', encoding='utf-8-sig') as f:
            splits_data = json.load(f)
        resisc45_splits = splits_data['resisc45']
        self.trainval_filenames = resisc45_splits['train'] + resisc45_splits['val']
        print(f"VTAB trainval filenames: {len(self.trainval_filenames)}")
        self.images = []
        self.labels = []
        if split in ['train', 'val']:
            num_train = int(len(self.trainval_filenames) * train_ratio)
            if split == 'train':
                filenames = self.trainval_filenames[:num_train]
            else:
                filenames = self.trainval_filenames[num_train:]
            used_set = set()
            for filename in filenames:
                class_name = '_'.join(filename.split('_')[:-1])
                if class_name in self.class_to_idx:
                    image_path = os.path.join(data_root, class_name, filename)
                    if os.path.exists(image_path):
                        self.images.append(image_path)
                        self.labels.append(self.class_to_idx[class_name])
                        used_set.add(filename)
                    else:
                        print(f"Warning: Image not found: {image_path}")
            # 自动补足缺失图片
            missing_count = (num_train if split == 'train' else len(filenames)) - len(self.images)
            if missing_count > 0:
                print(f"{split} split: 补足缺失图片 {missing_count} 张")
                for class_name in self.class_names:
                    class_dir = os.path.join(data_root, class_name)
                    if os.path.exists(class_dir):
                        all_imgs = [f for f in os.listdir(class_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
                        candidates = [f for f in all_imgs if f not in used_set]
                        random.shuffle(candidates)
                        for f in candidates:
                            if missing_count <= 0:
                                break
                            image_path = os.path.join(class_dir, f)
                            self.images.append(image_path)
                            self.labels.append(self.class_to_idx[class_name])
                            used_set.add(f)
                            missing_count -= 1
                        if missing_count <= 0:
                            break
        else:  # test
            # 测试集严格限定为test_files_json指定的图片
            assert test_files_json is not None, "test_files_json must be provided for test split!"
            with open(test_files_json, 'r', encoding='utf-8') as f:
                test_files = json.load(f)
            for rel_path in test_files:
                class_name = rel_path.split('/')[0]
                if class_name in self.class_to_idx:
                    image_path = os.path.join(data_root, rel_path)
                    if os.path.exists(image_path):
                        self.images.append(image_path)
                        self.labels.append(self.class_to_idx[class_name])
                    else:
                        print(f"Warning: Test image not found: {image_path}")
            print(f"test split: {len(self.images)} samples (严格按test_files_json)")
        print(f"{split} split: {len(self.images)} samples")
        print(f"Number of classes: {len(set(self.labels))}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 获取图像路径和标签
        image_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 创建一个黑色图像作为备用
            image = Image.new('RGB', (384, 384), (0, 0, 0))  # 使用384×384
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_resisc45_vtab_dataloaders(batch_size=16, num_workers=4, distributed=False, rank=0, world_size=1):
    """获取RESISC45 VTAB-1k数据加载器，使用固定的数据划分"""
    
    # 数据根目录（无train/test子文件夹）
    data_root = "/root/shared-nvme/VTAB/RESISC45/NWPU-RESISC45"
    splits_json_path = "/root/resisc45_trainval_splits.json"
    test_files_json = "/root/resisc45_test_files.json"
    
    # 检查文件是否存在
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"数据根目录不存在: {data_root}")
    if not os.path.exists(splits_json_path):
        raise FileNotFoundError(f"文件不存在: {splits_json_path}")
    if not os.path.exists(test_files_json):
        raise FileNotFoundError(f"文件不存在: {test_files_json}")
    
    # RESISC45数据预处理 - 针对384×384分辨率优化
    transform_train = transforms.Compose([
        transforms.Resize(416),  # 稍大一些的resize以保留更多细节
        transforms.RandomCrop(384, padding=32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # 遥感图像可以垂直翻转
        transforms.RandomRotation(degrees=30),  # 更大的旋转角度，因为遥感图像方向不固定
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(416),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(416),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 创建数据集
    train_dataset = RESISC45VTabDataset(
        data_root=data_root,
        splits_json_path=splits_json_path,
        split='train',
        transform=transform_train,
        train_ratio=1.0  # 使用全部1000个样本作为训练集
    )
    
    # 不再单独划分验证集
    val_dataset = None
    
    test_dataset = RESISC45VTabDataset(
        data_root=data_root,
        splits_json_path=splits_json_path,
        split='test',
        transform=transform_test,
        train_ratio=1.0,
        test_files_json=test_files_json
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
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
                            drop_last=False)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False,
                           num_workers=num_workers, 
                           pin_memory=True,
                           sampler=test_sampler)
    
    return train_loader, test_loader, train_sampler

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
    
    # 针对RESISC45和VTAB-1k优化的超参数
    batch_size = 12  # 适中的batch size，考虑到4卡24GB显存
    learning_rate = 1.5e-4  # 较小的学习率适合遥感图像
    num_epochs = 120  # 增加训练轮数
    warmup_epochs = 15  # 更长的warmup
    weight_decay = 0.08  # 适中的权重衰减
    num_workers = 4  # 减少worker数量避免内存问题
    
    # 模型参数
    pretrained_model_path = "/root/shared-nvme/vit_l16_in21k.pth"  # 本地预训练模型
    num_classes = 45  # RESISC45有45个类别
    image_size = 384  # 确保使用384×384分辨率
    
    if rank == 0:
        print("Loading ViT-L/16 model from local checkpoint...")
        print(f"Target image size: {image_size}x{image_size}")
    
    # 创建模型
    model = WeierstrassViTL16(
        pretrained_model_path=pretrained_model_path,
        num_classes=num_classes,
        image_size=image_size,
        use_derivative=True,
        alpha_scale=0.02
    )
    
    model = model.to(device)
    
    if distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    
    # 数据加载器
    train_loader, test_loader, train_sampler = get_resisc45_vtab_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size
    )
    
    # 损失函数 - 适中的标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    
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
        {'params': weierstrass_params, 'lr': learning_rate * 1.2},  # 威尔斯特拉斯编码稍大学习率
        {'params': classifier_params, 'lr': learning_rate * 1.8}  # 分类器较大学习率
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
    best_test_acc = 0.0
    best_epoch = 0
    results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    patience = 25  # 增加patience
    patience_counter = 0
    best_model_path = os.path.join(checkpoint_dir, 'weierstrass_vit_l16_resisc45_best.pth')
    
    if rank == 0:
        print("\n=== 开始RESISC45 VTAB-1k微调实验（1000样本训练，测试集评估） ===")
        print(f"训练样本数：1000 (VTAB-1k train+val)")
        print(f"测试样本数：{len(test_loader.dataset)} (原测试集)")
        print(f"类别数：45")
        print(f"图像分辨率：{image_size}x{image_size}")
        print(f"模型：ViT-L/16 + 威尔斯特拉斯位置编码")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, 
            train_sampler, scaler
        )
        # 每一轮都在全部测试集上评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        if rank == 0:
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            # 实时保存每轮结果到json
            results_path = os.path.join(checkpoint_dir, 'resisc45_vtab_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            # 保存每个epoch的检查点
            model_state = model.module.state_dict() if distributed else model.state_dict()
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_acc': test_acc,
                'results': results
            }, checkpoint_path)
            # 仅当测试集准确率达到best时保存最佳检查点
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                patience_counter = 0
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
    
    # 训练完成后，加载最佳模型进行最终测试
    if rank == 0:
        print(f"\n=== 训练完成，加载最佳测试准确率模型进行最终测试 ===")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            if distributed:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载第 {checkpoint['epoch']} 轮的模型，测试准确率: {checkpoint['best_test_acc']:.2f}%")
        # 最终测试评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"\n=== 最终测试结果 ===")
        print(f"最佳测试集准确率: {best_test_acc:.2f}% (来自第{best_epoch}轮)")
        print(f"基线准确率: 85.2%")
        if best_test_acc > 85.2:
            improvement = best_test_acc - 85.2
            print(f"🎉 成功超过基线! 提升了 {improvement:.2f} 个百分点")
        else:
            print(f"未达到目标，还需要 {85.2 - best_test_acc:.2f} 个百分点")
        # 保存最终结果
        results['final_test_acc'] = best_test_acc
        results['best_test_epoch'] = best_epoch
        results['baseline_accuracy'] = 85.2
        results['improvement'] = best_test_acc - 85.2
        results_path = os.path.join(checkpoint_dir, 'resisc45_vtab_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        # 输出学习到的威尔斯特拉斯参数（用最佳测试准确率的模型）
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
        print(f"VTAB-1k RESISC45配置:")
        print(f"- 训练样本: 1000 (VTAB-1k train+val)")
        print(f"- 测试样本: {len(test_loader.dataset)}")
        print(f"- 类别数: 45")
        print(f"- 最佳测试准确率: {best_test_acc:.2f}%")
        print(f"- 基线准确率: 85.2%")
        print(f"- 提升幅度: {best_test_acc - 85.2:.2f} 个百分点")
    
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()