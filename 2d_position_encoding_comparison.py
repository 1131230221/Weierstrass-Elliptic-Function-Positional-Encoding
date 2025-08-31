import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import os
import json
from tqdm import tqdm
import random
import math
import time
from W_Ti import ImprovedViT_Ti, TransformerBlock, DropPath, Cutout
import matplotlib
from matplotlib import font_manager


def _apply_publication_style():
    """应用学术发表风格的matplotlib设置"""
    matplotlib.rcParams.update({
        'font.family': 'DejaVu Sans',
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#222222',
        'axes.grid': True,
        'grid.color': '#AAAAAA',
        'grid.alpha': 0.25,
        'grid.linestyle': '-',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.framealpha': 0.9,
        'legend.borderpad': 0.6,
    })


class TwoD_SinusoidalPositionalEncoding(nn.Module):
    """2D正弦位置编码 - Transformer正弦编码的直接2D扩展"""
    
    def __init__(self, d_model: int, max_h: int = 14, max_w: int = 14):
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        
        # 确保d_model是4的倍数，以便平均分配给x, y坐标的sin, cos
        assert d_model % 4 == 0, "d_model必须是4的倍数"
        
        # 为每个坐标轴分配d_model/2的维度
        coord_dim = d_model // 2
        
        # 生成2D位置编码
        self.register_buffer('pos_encoding', self._generate_2d_encoding(max_h, max_w, coord_dim))
        
        # CLS token的特殊位置编码
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    
    def _generate_2d_encoding(self, max_h: int, max_w: int, coord_dim: int):
        """生成2D正弦位置编码"""
        # 为每个坐标轴生成位置编码
        pe = torch.zeros(max_h, max_w, self.d_model)
        
        # X坐标编码 (占用前coord_dim维度)
        div_term_x = torch.exp(torch.arange(0, coord_dim, 2).float() * 
                              -(math.log(10000.0) / coord_dim))
        
        # Y坐标编码 (占用后coord_dim维度) 
        div_term_y = torch.exp(torch.arange(0, coord_dim, 2).float() * 
                              -(math.log(10000.0) / coord_dim))
        
        for h in range(max_h):
            for w in range(max_w):
                # X坐标的正弦余弦编码
                pe[h, w, 0::4] = torch.sin(w * div_term_x)
                pe[h, w, 1::4] = torch.cos(w * div_term_x)
                
                # Y坐标的正弦余弦编码
                pe[h, w, 2::4] = torch.sin(h * div_term_y)
                pe[h, w, 3::4] = torch.cos(h * div_term_y)
        
        return pe.view(max_h * max_w, self.d_model)
    
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        应用2D正弦位置编码
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            h, w: patch网格尺寸
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 生成位置编码
        position_encodings = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # CLS token编码
        position_encodings[:, 0:1, :] = self.cls_pos_embedding
        
        # Patch编码
        if h * w <= self.pos_encoding.size(0):
            patch_encodings = self.pos_encoding[:h*w].to(device)
            position_encodings[:, 1:1+h*w, :] = patch_encodings.unsqueeze(0).expand(batch_size, -1, -1)
        
        return x + position_encodings


class TwoD_LearnableGridPositionalEncoding(nn.Module):
    """2D可学习网格位置编码 - 直接根据2D坐标查找参数矩阵"""
    
    def __init__(self, d_model: int, max_h: int = 14, max_w: int = 14):
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        
        # 创建可学习的2D网格参数矩阵
        self.grid_embeddings = nn.Parameter(torch.randn(max_h, max_w, d_model) * 0.02)
        
        # CLS token的特殊位置编码
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        应用2D可学习网格位置编码
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            h, w: patch网格尺寸
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 生成位置编码
        position_encodings = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # CLS token编码
        position_encodings[:, 0:1, :] = self.cls_pos_embedding
        
        # Patch编码 - 直接从2D网格中提取
        if h <= self.max_h and w <= self.max_w:
            # 重新排列网格编码为序列形式
            patch_encodings = self.grid_embeddings[:h, :w].reshape(h*w, self.d_model)
            position_encodings[:, 1:1+h*w, :] = patch_encodings.unsqueeze(0).expand(batch_size, -1, -1)
        
        return x + position_encodings


class MultiMethodViT_Ti(nn.Module):
    """支持多种位置编码方法的ViT-Ti模型"""
    
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
                 pos_encoding_type: str = 'wef'):  # 'wef', 'ape', '2d_sin', '2d_grid'
        super().__init__()
        
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.pos_encoding_type = pos_encoding_type
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 根据类型选择位置编码
        grid_size = img_size // patch_size
        if pos_encoding_type == 'wef':
            from W_Ti import WEFPositionalEncoding
            self.pos_encoding = WEFPositionalEncoding(
                d_model=embed_dim,
                max_h=grid_size,
                max_w=grid_size,
                g2=1.0,
                g3=0.0,
                projection_type='linear',
                alpha_scale=0.15,
                learnable_alpha=True,
                use_4d=True
            )
        elif pos_encoding_type == 'ape':
            # 标准1D可学习位置编码
            self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            self.pos_encoding = None
        elif pos_encoding_type == '2d_sin':
            self.pos_encoding = TwoD_SinusoidalPositionalEncoding(
                d_model=embed_dim,
                max_h=grid_size,
                max_w=grid_size
            )
        elif pos_encoding_type == '2d_grid':
            self.pos_encoding = TwoD_LearnableGridPositionalEncoding(
                d_model=embed_dim,
                max_h=grid_size,
                max_w=grid_size
            )
        else:
            raise ValueError(f"不支持的位置编码类型: {pos_encoding_type}")
        
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
        if self.pos_encoding_type in ['wef', '2d_sin', '2d_grid']:
            h = w = int(math.sqrt(self.num_patches))
            x = self.pos_encoding(x, h, w)
        else:  # APE
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


def create_cifar100_dataloaders_for_comparison(batch_size=160, num_workers=8):
    """创建用于对比实验的CIFAR-100数据加载器"""
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                           std=[0.2673, 0.2564, 0.2762]),
        Cutout(n_holes=1, length=12)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                           std=[0.2673, 0.2564, 0.2762])
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='/root/shared-nvme', train=True, download=False, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='/root/shared-nvme', train=False, download=False, transform=test_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def train_model_with_pos_encoding(model, train_loader, test_loader, pos_type, 
                                  num_epochs=60, initial_lr=0.0015, device='cuda'):
    """训练使用特定位置编码的模型"""
    
    model = model.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.03)
    
    # 学习率调度器
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    
    num_warmup_steps = 10 * len(train_loader)  # 10 epoch warmup
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    
    # 训练记录
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"\n开始训练 {pos_type.upper()} 模型...")
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'{pos_type.upper()} Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                acc = 100.0 * train_correct / train_total
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc:.2f}%', 'LR': f'{lr:.6f}'})
        
        # 计算训练指标
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / train_total
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        epoch_val_acc = 100.0 * val_correct / val_total
        
        # 记录
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)
        
        print(f'{pos_type.upper()} Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, '
              f'Train Acc={epoch_train_acc:.2f}%, Val Acc={epoch_val_acc:.2f}%')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'final_val_acc': val_accuracies[-1],
        'best_val_acc': max(val_accuracies)
    }


def plot_comparison_results(results_dict, save_path='2d_position_encoding_comparison.png'):
    """绘制多种位置编码方法的对比结果"""
    _apply_publication_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 定义颜色和标签
    colors = {
        'wef': '#2E86AB',
        'ape': '#A23B72', 
        '2d_sin': '#F18F01',
        '2d_grid': '#C73E1D'
    }
    
    labels = {
        'wef': 'WEF-PE (Ours)',
        'ape': 'APE (1D Flattened)',
        '2d_sin': '2D Sinusoidal PE',
        '2d_grid': '2D Learnable Grid PE'
    }
    
    methods = list(results_dict.keys())
    epochs = range(1, len(results_dict[methods[0]]['train_losses']) + 1)
    
    # 1. 训练损失对比
    for method in methods:
        ax1.plot(epochs, results_dict[method]['train_losses'], 
                color=colors[method], label=labels[method], linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 验证准确率对比
    for method in methods:
        ax2.plot(epochs, results_dict[method]['val_accuracies'], 
                color=colors[method], label=labels[method], linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 最终准确率柱状图
    final_accs = [results_dict[method]['final_val_acc'] for method in methods]
    method_labels = [labels[method] for method in methods]
    method_colors = [colors[method] for method in methods]
    
    bars = ax3.bar(method_labels, final_accs, color=method_colors, alpha=0.8)
    ax3.set_ylabel('Final Validation Accuracy (%)')
    ax3.set_title('Final Performance Comparison')
    ax3.tick_params(axis='x', rotation=15)
    
    # 添加数值标签
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        ax3.annotate(f'{acc:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # 4. 最佳准确率对比
    best_accs = [results_dict[method]['best_val_acc'] for method in methods]
    
    bars2 = ax4.bar(method_labels, best_accs, color=method_colors, alpha=0.8)
    ax4.set_ylabel('Best Validation Accuracy (%)')
    ax4.set_title('Best Performance Comparison')
    ax4.tick_params(axis='x', rotation=15)
    
    # 添加数值标签
    for bar, acc in zip(bars2, best_accs):
        height = bar.get_height()
        ax4.annotate(f'{acc:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # 添加面板标签
    panels = ['A', 'B', 'C', 'D']
    axes = [ax1, ax2, ax3, ax4]
    for panel, ax in zip(panels, axes):
        ax.text(-0.1, 1.05, panel, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"对比结果图已保存到: {save_path}")
    plt.show()


def run_2d_position_encoding_comparison():
    """运行2D位置编码对比实验"""
    print("=" * 60)
    print("2D位置编码方法公平对比实验")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, test_loader = create_cifar100_dataloaders_for_comparison(batch_size=160, num_workers=8)
    
    # 定义要对比的位置编码方法
    encoding_methods = {
        'wef': 'Weierstrass Elliptic Function PE',
        'ape': 'Absolute Positional Embedding (1D)',
        '2d_sin': '2D Sinusoidal PE',
        '2d_grid': '2D Learnable Grid PE'
    }
    
    results = {}
    
    for method, description in encoding_methods.items():
        print(f"\n{'='*50}")
        print(f"训练 {description}")
        print(f"{'='*50}")
        
        # 创建模型
        model = MultiMethodViT_Ti(
            img_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=100,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            dropout=0.05,
            drop_path=0.05,
            pos_encoding_type=method
        )
        
        # 训练模型
        result = train_model_with_pos_encoding(
            model, train_loader, test_loader, method,
            num_epochs=60, initial_lr=0.0015, device=device
        )
        
        results[method] = result
        
        # 保存模型
        torch.save(model.state_dict(), f'best_{method}_vit_ti_comparison.pth')
        print(f"{description} 训练完成!")
        print(f"最终验证准确率: {result['final_val_acc']:.2f}%")
        print(f"最佳验证准确率: {result['best_val_acc']:.2f}%")
    
    # 打印对比结果
    print("\n" + "=" * 70)
    print("实验结果总结")
    print("=" * 70)
    print(f"{'方法':<25} {'最终准确率':<12} {'最佳准确率':<12} {'相对WEF改善':<12}")
    print("-" * 70)
    
    wef_final = results['wef']['final_val_acc']
    wef_best = results['wef']['best_val_acc']
    
    for method, description in encoding_methods.items():
        final_acc = results[method]['final_val_acc']
        best_acc = results[method]['best_val_acc']
        
        if method == 'wef':
            improvement_final = 0.0
            improvement_best = 0.0
        else:
            improvement_final = wef_final - final_acc
            improvement_best = wef_best - best_acc
        
        print(f"{description:<25} {final_acc:<12.2f} {best_acc:<12.2f} "
              f"{improvement_final:<12.1f}")
    
    # 绘制对比图
    print("\n绘制对比结果...")
    plot_comparison_results(results)
    
    # 保存详细结果
    # 为了JSON序列化，移除复杂数据类型
    results_for_json = {}
    for method, result in results.items():
        results_for_json[method] = {
            'final_val_acc': float(result['final_val_acc']),
            'best_val_acc': float(result['best_val_acc']),
            'train_losses': [float(x) for x in result['train_losses']],
            'train_accuracies': [float(x) for x in result['train_accuracies']],
            'val_accuracies': [float(x) for x in result['val_accuracies']]
        }
    
    with open('2d_position_encoding_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_for_json, f, indent=2, ensure_ascii=False)
    
    print("实验结果已保存到: 2d_position_encoding_comparison_results.json")
    print("\n2D位置编码对比实验完成!")
    
    return results


if __name__ == "__main__":
    results = run_2d_position_encoding_comparison()