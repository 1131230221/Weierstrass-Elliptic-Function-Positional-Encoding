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
from W_Ti import ImprovedViT_Ti
import matplotlib
from matplotlib import font_manager
import math


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


class RelativePositionPredictor(nn.Module):
    """相对位置预测MLP头"""
    
    def __init__(self, embed_dim: int = 192, hidden_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 输入是两个patch的完整表示：ei+pi 和 ej+pj
        # 所以输入维度是2*embed_dim
        self.predictor = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, 2)  # 输出(Δx, Δy)
        )
    
    def forward(self, patch_i: torch.Tensor, patch_j: torch.Tensor) -> torch.Tensor:
        """
        预测两个patch之间的相对位置
        Args:
            patch_i: 第i个patch的表示 (ei + pi) [batch_size, embed_dim]
            patch_j: 第j个patch的表示 (ej + pj) [batch_size, embed_dim]
        Returns:
            relative_pos: 相对位置 (Δx, Δy) [batch_size, 2]
        """
        # 拼接两个patch的表示
        combined = torch.cat([patch_i, patch_j], dim=-1)  # [batch_size, 2*embed_dim]
        return self.predictor(combined)


class RelativePositionDataset(torch.utils.data.Dataset):
    """相对位置预测数据集"""
    
    def __init__(self, model, dataloader, device, num_samples=10000, use_wef=True):
        """
        Args:
            model: 预训练的ViT模型
            dataloader: 数据加载器
            device: 计算设备
            num_samples: 生成的样本数量
            use_wef: 是否使用WEF位置编码
        """
        self.device = device
        self.embed_dim = model.embed_dim
        self.patch_size = model.patch_size
        self.use_wef = use_wef
        
        # 计算patch网格尺寸
        self.grid_h = self.grid_w = 224 // self.patch_size  # 14x14
        
        print(f"正在生成{'WEF-PE' if use_wef else 'APE'}相对位置数据集...")
        self.samples = self._generate_samples(model, dataloader, num_samples)
        print(f"数据集生成完成，共{len(self.samples)}个样本")
    
    def _extract_patch_embeddings(self, model, images):
        """提取patch embeddings（不包含位置编码）"""
        model.eval()
        with torch.no_grad():
            B = images.shape[0]
            
            # Patch embedding
            x = model.patch_embed(images)  # [B, embed_dim, H, W]
            x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
            x = model.patch_norm(x)
            
            return x
    
    def _get_position_encodings(self, model, batch_size):
        """获取位置编码"""
        if self.use_wef:
            # 获取WEF位置编码
            dummy_input = torch.zeros(batch_size, 1 + self.grid_h * self.grid_w, model.embed_dim, device=self.device)
            pos_encodings = model.pos_encoding(dummy_input, self.grid_h, self.grid_w)
            # 只返回patch的位置编码，去掉CLS token
            return pos_encodings[:, 1:, :]  # [B, num_patches, embed_dim]
        else:
            # 获取APE位置编码
            pos_encodings = model.pos_embedding.expand(batch_size, -1, -1)
            # 只返回patch的位置编码，去掉CLS token
            return pos_encodings[:, 1:, :]  # [B, num_patches, embed_dim]
    
    def _generate_samples(self, model, dataloader, num_samples):
        """生成相对位置预测样本"""
        samples = []
        model.eval()
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="生成样本")):
                images = images.to(self.device)
                batch_size = images.shape[0]
                
                # 提取patch embeddings（不含位置编码）
                patch_embeddings = self._extract_patch_embeddings(model, images)  # [B, num_patches, embed_dim]
                
                # 获取位置编码
                pos_encodings = self._get_position_encodings(model, batch_size)  # [B, num_patches, embed_dim]
                
                # 为每个图像生成多对patch样本
                samples_per_image = min(20, num_samples // len(dataloader.dataset) + 1)
                
                for b in range(batch_size):
                    for _ in range(samples_per_image):
                        # 随机选择两个不同的patch
                        i, j = random.sample(range(self.grid_h * self.grid_w), 2)
                        
                        # 计算patch的网格坐标
                        i_y, i_x = i // self.grid_w, i % self.grid_w
                        j_y, j_x = j // self.grid_w, j % self.grid_w
                        
                        # 计算相对位置
                        delta_x = j_x - i_x
                        delta_y = j_y - i_y
                        
                        # 获取patch表示 (embedding + position encoding)
                        patch_i_repr = patch_embeddings[b, i] + pos_encodings[b, i]
                        patch_j_repr = patch_embeddings[b, j] + pos_encodings[b, j]
                        
                        samples.append({
                            'patch_i': patch_i_repr.cpu(),
                            'patch_j': patch_j_repr.cpu(),
                            'delta_x': float(delta_x),
                            'delta_y': float(delta_y)
                        })
                        
                        if len(samples) >= num_samples:
                            break
                    
                    if len(samples) >= num_samples:
                        break
                
                if len(samples) >= num_samples:
                    break
        
        return samples[:num_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            sample['patch_i'],
            sample['patch_j'],
            torch.tensor([sample['delta_x'], sample['delta_y']], dtype=torch.float32)
        )


def train_relative_position_predictor(predictor, dataset, device, num_epochs=50, batch_size=256, lr=0.001):
    """训练相对位置预测器"""
    
    # 数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    predictor.train()
    
    training_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for patch_i, patch_j, target_delta in pbar:
            patch_i = patch_i.to(device)
            patch_j = patch_j.to(device)
            target_delta = target_delta.to(device)
            
            optimizer.zero_grad()
            
            # 预测相对位置
            pred_delta = predictor(patch_i, patch_j)
            
            # 计算损失
            loss = criterion(pred_delta, target_delta)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        training_losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')
    
    return training_losses


def evaluate_relative_position_predictor(predictor, dataset, device, batch_size=256):
    """评估相对位置预测器"""
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    predictor.eval()
    
    total_mse = 0.0
    total_mae = 0.0
    all_errors = []
    num_samples = 0
    
    with torch.no_grad():
        for patch_i, patch_j, target_delta in tqdm(dataloader, desc='Evaluating'):
            patch_i = patch_i.to(device)
            patch_j = patch_j.to(device)
            target_delta = target_delta.to(device)
            
            # 预测相对位置
            pred_delta = predictor(patch_i, patch_j)
            
            # 计算误差
            mse = F.mse_loss(pred_delta, target_delta, reduction='sum')
            mae = F.l1_loss(pred_delta, target_delta, reduction='sum')
            
            total_mse += mse.item()
            total_mae += mae.item()
            num_samples += target_delta.size(0)
            
            # 保存每个样本的误差用于分析
            errors = torch.sqrt(torch.sum((pred_delta - target_delta)**2, dim=1))
            all_errors.extend(errors.cpu().numpy())
    
    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples
    avg_rmse = math.sqrt(avg_mse)
    
    return {
        'mse': avg_mse,
        'mae': avg_mae,
        'rmse': avg_rmse,
        'all_errors': all_errors
    }


def plot_relative_position_results(wef_results, ape_results, wef_losses, ape_losses, save_path='relative_position_results.png'):
    """绘制相对位置预测结果对比图"""
    _apply_publication_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 训练损失对比
    epochs = range(1, len(wef_losses) + 1)
    ax1.plot(epochs, wef_losses, 'o-', color='#2E86AB', label='WEF-PE', linewidth=2, markersize=4)
    ax1.plot(epochs, ape_losses, 's--', color='#A23B72', label='APE', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss (MSE)')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 评估指标对比
    metrics = ['MSE', 'MAE', 'RMSE']
    wef_scores = [wef_results['mse'], wef_results['mae'], wef_results['rmse']]
    ape_scores = [ape_results['mse'], ape_results['mae'], ape_results['rmse']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, wef_scores, width, label='WEF-PE', color='#2E86AB', alpha=0.8)
    bars2 = ax2.bar(x + width/2, ape_scores, width, label='APE', color='#A23B72', alpha=0.8)
    
    ax2.set_xlabel('Evaluation Metrics')
    ax2.set_ylabel('Error Value')
    ax2.set_title('Relative Position Prediction Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # 3. 误差分布直方图
    ax3.hist(wef_results['all_errors'], bins=50, alpha=0.7, color='#2E86AB', label='WEF-PE', density=True)
    ax3.hist(ape_results['all_errors'], bins=50, alpha=0.7, color='#A23B72', label='APE', density=True)
    ax3.set_xlabel('Prediction Error (L2 Distance)')
    ax3.set_ylabel('Density')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 累积误差分析
    wef_errors_sorted = np.sort(wef_results['all_errors'])
    ape_errors_sorted = np.sort(ape_results['all_errors'])
    
    wef_cdf = np.arange(1, len(wef_errors_sorted) + 1) / len(wef_errors_sorted)
    ape_cdf = np.arange(1, len(ape_errors_sorted) + 1) / len(ape_errors_sorted)
    
    ax4.plot(wef_errors_sorted, wef_cdf, color='#2E86AB', label='WEF-PE', linewidth=2)
    ax4.plot(ape_errors_sorted, ape_cdf, color='#A23B72', label='APE', linewidth=2)
    ax4.set_xlabel('Prediction Error (L2 Distance)')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加面板标签
    panels = ['A', 'B', 'C', 'D']
    axes = [ax1, ax2, ax3, ax4]
    for panel, ax in zip(panels, axes):
        ax.text(-0.1, 1.05, panel, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"结果图表已保存到: {save_path}")
    plt.show()


def create_cifar100_dataloader(batch_size=64, num_workers=4):
    """创建CIFAR-100数据加载器（用于提取patch embeddings）"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                           std=[0.2673, 0.2564, 0.2762])
    ])
    
    dataset = torchvision.datasets.CIFAR100(
        root='/root/shared-nvme', train=False, download=False, transform=transform
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    return dataloader


def load_model_for_relative_position_test(model_path, use_wef=True, device='cuda'):
    """加载模型用于相对位置测试"""
    model = ImprovedViT_Ti(
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
        use_wef=use_wef
    )
    
    if os.path.exists(model_path):
        print(f"加载模型权重: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"警告: 未找到模型权重文件 {model_path}")
    
    model = model.to(device)
    return model


def run_relative_position_awareness_experiment():
    """运行相对位置感知实验"""
    print("=" * 60)
    print("相对位置感知能力验证实验")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    dataloader = create_cifar100_dataloader(batch_size=32, num_workers=4)
    
    # 加载WEF和APE模型
    print("加载预训练模型...")
    wef_model = load_model_for_relative_position_test('best_wef_vit_ti.pth', use_wef=True, device=device)
    ape_model = load_model_for_relative_position_test('best_ape_vit_ti.pth', use_wef=False, device=device)
    
    # 生成WEF相对位置数据集
    print("\n生成WEF-PE相对位置数据集...")
    wef_dataset = RelativePositionDataset(wef_model, dataloader, device, num_samples=8000, use_wef=True)
    
    # 生成APE相对位置数据集
    print("\n生成APE相对位置数据集...")
    ape_dataset = RelativePositionDataset(ape_model, dataloader, device, num_samples=8000, use_wef=False)
    
    # 创建和训练WEF预测器
    print("\n训练WEF-PE相对位置预测器...")
    wef_predictor = RelativePositionPredictor(embed_dim=192, hidden_dim=128).to(device)
    wef_losses = train_relative_position_predictor(
        wef_predictor, wef_dataset, device, num_epochs=30, batch_size=256, lr=0.001
    )
    
    # 创建和训练APE预测器
    print("\n训练APE相对位置预测器...")
    ape_predictor = RelativePositionPredictor(embed_dim=192, hidden_dim=128).to(device)
    ape_losses = train_relative_position_predictor(
        ape_predictor, ape_dataset, device, num_epochs=30, batch_size=256, lr=0.001
    )
    
    # 评估WEF预测器
    print("\n评估WEF-PE预测器...")
    wef_results = evaluate_relative_position_predictor(wef_predictor, wef_dataset, device)
    
    # 评估APE预测器
    print("\n评估APE预测器...")
    ape_results = evaluate_relative_position_predictor(ape_predictor, ape_dataset, device)
    
    # 打印结果
    print("\n" + "=" * 50)
    print("实验结果对比")
    print("=" * 50)
    print(f"{'指标':<12} {'WEF-PE':<12} {'APE':<12} {'改善比例':<12}")
    print("-" * 50)
    
    metrics = [('MSE', 'mse'), ('MAE', 'mae'), ('RMSE', 'rmse')]
    for name, key in metrics:
        wef_val = wef_results[key]
        ape_val = ape_results[key]
        improvement = (ape_val - wef_val) / ape_val * 100
        print(f"{name:<12} {wef_val:<12.4f} {ape_val:<12.4f} {improvement:<12.1f}%")
    
    # 绘制结果
    print("\n绘制结果图表...")
    plot_relative_position_results(wef_results, ape_results, wef_losses, ape_losses)
    
    # 保存结果
    results_dict = {
        'wef_results': wef_results,
        'ape_results': ape_results,
        'wef_training_losses': wef_losses,
        'ape_training_losses': ape_losses,
        'improvement_percentages': {
            'mse': (ape_results['mse'] - wef_results['mse']) / ape_results['mse'] * 100,
            'mae': (ape_results['mae'] - wef_results['mae']) / ape_results['mae'] * 100,
            'rmse': (ape_results['rmse'] - wef_results['rmse']) / ape_results['rmse'] * 100
        }
    }
    
    # 移除numpy数组以便JSON序列化
    results_for_json = {k: v for k, v in results_dict.items() if k not in ['wef_results', 'ape_results']}
    results_for_json['wef_metrics'] = {k: v for k, v in wef_results.items() if k != 'all_errors'}
    results_for_json['ape_metrics'] = {k: v for k, v in ape_results.items() if k != 'all_errors'}
    
    with open('relative_position_awareness_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_for_json, f, indent=2, ensure_ascii=False)
    
    print("实验结果已保存到: relative_position_awareness_results.json")
    print("\n相对位置感知实验完成!")
    
    return results_dict


if __name__ == "__main__":
    results = run_relative_position_awareness_experiment()