import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import os
import json
from tqdm import tqdm
import random
from W_Ti import ImprovedViT_Ti
import matplotlib
from matplotlib import font_manager


def _setup_chinese_font():
    """
    为Matplotlib设置中文字体，优先按候选列表自动选择已安装字体。
    若均不可用，则保持默认字体并尽量避免负号显示问题。
    """
    candidate_families = [
        'Noto Sans CJK SC', 'Noto Sans CJK', 'Noto Sans SC',
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
        'Source Han Sans SC', 'Source Han Sans CN', 'PingFang SC'
    ]
    chosen = None
    for fam in candidate_families:
        try:
            font_manager.FontProperties(family=fam)
            # findfont会在找不到时抛出异常（fallback_to_default=False）
            font_path = font_manager.findfont(
                font_manager.FontProperties(family=fam), fallback_to_default=False
            )
            if os.path.exists(font_path):
                chosen = fam
                break
        except Exception:
            continue
    if chosen is not None:
        matplotlib.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    else:
        # 无可用中文字体时保留默认，但仍设置备选以减少告警
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    # 避免负号显示为方块
    matplotlib.rcParams['axes.unicode_minus'] = False


def _apply_publication_style():
    """Apply clean publication-style Matplotlib rcParams (English labels)."""
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

# 假设原始代码已经导入，这里只添加缺失的部分

class OcclusionTransform:
    """
    随机遮挡变换，支持不同的遮挡强度
    """
    def __init__(self, occlusion_ratio: float = 0.1, fill_value: float = 0.0):
        """
        Args:
            occlusion_ratio: 遮挡面积比例 (0.0-1.0)
            fill_value: 遮挡区域填充值
        """
        self.occlusion_ratio = occlusion_ratio
        self.fill_value = fill_value
    
    def __call__(self, img):
        """
        对输入图像应用随机遮挡
        Args:
            img: PIL Image 或 tensor
        Returns:
            遮挡后的图像
        """
        if isinstance(img, torch.Tensor):
            c, h, w = img.shape
        else:
            # PIL Image
            w, h = img.size
            c = 3
        
        # 计算遮挡块大小
        total_area = h * w
        occlusion_area = int(total_area * self.occlusion_ratio)
        
        # 生成多个小的遮挡块而不是单个大块，更接近实际场景
        num_blocks = random.randint(1, max(1, int(self.occlusion_ratio * 20)))
        
        if isinstance(img, torch.Tensor):
            occluded_img = img.clone()
        else:
            occluded_img = transforms.ToTensor()(img)
        
        # 应用多个遮挡块
        total_occluded = 0
        for _ in range(num_blocks):
            if total_occluded >= occlusion_area:
                break
                
            # 随机生成遮挡块大小
            remaining_area = occlusion_area - total_occluded
            block_area = min(remaining_area, random.randint(1, remaining_area + 1))
            
            # 计算遮挡块的长宽比
            aspect_ratio = random.uniform(0.5, 2.0)  # 长宽比在0.5-2.0之间
            block_h = max(1, int(np.sqrt(block_area / aspect_ratio)))
            block_w = max(1, int(block_area / block_h))
            
            # 确保不超出图像边界
            block_h = max(1, min(block_h, h))
            block_w = max(1, min(block_w, w))
            
            # 随机选择遮挡位置
            start_h = random.randint(0, max(0, h - block_h))
            start_w = random.randint(0, max(0, w - block_w))
            
            # 应用遮挡
            occluded_img[:, start_h:start_h + block_h, start_w:start_w + block_w] = self.fill_value
            total_occluded += block_h * block_w
        
        return occluded_img


def create_baseline_vit_ti():
    """创建使用APE的基线ViT-Ti模型"""
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
        use_wef=False  # 使用标准APE
    )
    return model


def evaluate_model_with_occlusion(model, dataloader, device, occlusion_ratios):
    """
    评估模型在不同遮挡率下的性能
    
    Args:
        model: 待评估的模型
        dataloader: 数据加载器
        device: 计算设备
        occlusion_ratios: 遮挡率列表
    
    Returns:
        results: 包含不同遮挡率下准确率的字典
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        for occlusion_ratio in tqdm(occlusion_ratios, desc="评估不同遮挡率"):
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, leave=False, desc=f"遮挡率 {occlusion_ratio:.1%}")):
                # 对每个batch应用遮挡
                if occlusion_ratio > 0:
                    occluded_images = []
                    for img in images:
                        # 将tensor转换为PIL进行处理，然后再转回tensor
                        # 先反归一化
                        mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)
                        std = torch.tensor([0.2673, 0.2564, 0.2762]).view(3, 1, 1)
                        unnorm_img = img * std + mean
                        unnorm_img = torch.clamp(unnorm_img, 0, 1)
                        
                        # 应用遮挡
                        occlusion_transform = OcclusionTransform(occlusion_ratio, fill_value=0.0)
                        occluded_img = occlusion_transform(unnorm_img)
                        
                        # 重新归一化
                        occluded_img = (occluded_img - mean) / std
                        occluded_images.append(occluded_img)
                    
                    images = torch.stack(occluded_images)
                
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 为了节省时间，可以限制评估的batch数量
                if batch_idx >= 50:  # 只评估前50个batch
                    break
            
            accuracy = 100.0 * correct / total
            results[occlusion_ratio] = accuracy
            print(f"遮挡率 {occlusion_ratio:.1%}: 准确率 = {accuracy:.2f}%")
    
    return results


def plot_occlusion_robustness_comparison(wef_results, ape_results, save_path='occlusion_robustness.png'):
    """
    Plot occlusion robustness comparison (English labels)
    """
    _apply_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    
    occlusion_ratios = sorted(wef_results.keys())
    wef_accuracies = [wef_results[ratio] for ratio in occlusion_ratios]
    ape_accuracies = [ape_results[ratio] for ratio in occlusion_ratios]
    
    # Accuracy curves
    x_vals = [r*100 for r in occlusion_ratios]
    ax1.plot(x_vals, wef_accuracies, marker='o', linestyle='-', linewidth=2.2, markersize=6,
             color='#2E86AB', label='WEF-PE (Ours)')
    ax1.plot(x_vals, ape_accuracies, marker='s', linestyle='--', linewidth=2.0, markersize=6,
             color='#A23B72', label='APE (Baseline)')
    
    ax1.set_xlabel('Occlusion (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Occlusion Ratio', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, ncol=1)
    ax1.grid(True, alpha=0.25)
    y_min = min(min(wef_accuracies), min(ape_accuracies))
    y_max = max(max(wef_accuracies), max(ape_accuracies))
    ax1.set_ylim([max(0, y_min - 3), y_max + 3])
    ax1.set_xticks(x_vals)
    
    # Value annotations
    for i, (ratio, wef_acc, ape_acc) in enumerate(zip(occlusion_ratios, wef_accuracies, ape_accuracies)):
        ax1.annotate(f'{wef_acc:.1f}', (ratio*100, wef_acc),
                    textcoords="offset points", xytext=(0,8), ha='center', fontsize=9, color='#2E86AB')
        ax1.annotate(f'{ape_acc:.1f}', (ratio*100, ape_acc),
                    textcoords="offset points", xytext=(0,-12), ha='center', fontsize=9, color='#A23B72')
    
    # Degradation curves
    wef_baseline = wef_accuracies[0]
    ape_baseline = ape_accuracies[0]
    
    wef_degradation = [(wef_baseline - acc) for acc in wef_accuracies]
    ape_degradation = [(ape_baseline - acc) for acc in ape_accuracies]
    
    ax2.plot(x_vals, wef_degradation, marker='o', linestyle='-', linewidth=2.2, markersize=6,
             color='#2E86AB', label='WEF-PE (Ours)')
    ax2.plot(x_vals, ape_degradation, marker='s', linestyle='--', linewidth=2.0, markersize=6,
             color='#A23B72', label='APE (Baseline)')
    
    ax2.set_xlabel('Occlusion (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Degradation (pp)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Degradation', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10, ncol=1)
    ax2.grid(True, alpha=0.25)
    ax2.set_xticks(x_vals)
    
    # Value annotations for B panel (match A panel style/colors)
    for ratio, w_deg, a_deg in zip(occlusion_ratios, wef_degradation, ape_degradation):
        ax2.annotate(f'{w_deg:.1f}', (ratio*100, w_deg),
                     textcoords="offset points", xytext=(0,8), ha='center', fontsize=9, color='#2E86AB')
        ax2.annotate(f'{a_deg:.1f}', (ratio*100, a_deg),
                     textcoords="offset points", xytext=(0,-12), ha='center', fontsize=9, color='#A23B72')

    # Improvement annotations
    for i, (ratio, wef_deg, ape_deg) in enumerate(zip(occlusion_ratios[1:], wef_degradation[1:], ape_degradation[1:])):
        if ape_deg > wef_deg:
            improvement = ape_deg - wef_deg
            ax2.annotate(f'+{improvement:.1f} pp', 
                        (ratio*100, (wef_deg + ape_deg) / 2), 
                        textcoords="offset points", xytext=(10,0), ha='left', 
                        fontsize=9, color='green', fontweight='bold')

    # Panel labels
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
    
    plt.tight_layout()
    # Save high-quality images
    base, ext = os.path.splitext(save_path)
    plt.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
    try:
        plt.savefig(f'{base}.pdf', bbox_inches='tight')
    except Exception:
        pass
    print(f"对比图已保存到: {save_path}")
    plt.show()


def create_test_dataloader_cifar100(batch_size=64, num_workers=4):
    """创建CIFAR-100测试数据加载器"""
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                           std=[0.2673, 0.2564, 0.2762])
    ])
    
    test_dataset = torchvision.datasets.CIFAR100(
        root='/root/shared-nvme', train=False, download=False, transform=test_transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return test_loader


def load_trained_models(wef_model_path='best_wef_vit_ti.pth', device='cuda'):
    """
    加载训练好的模型
    
    Args:
        wef_model_path: WEF模型权重路径
        device: 计算设备
    
    Returns:
        wef_model, ape_model: 加载好的模型
    """
    # 创建WEF模型并加载权重
    wef_model = ImprovedViT_Ti(
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
        use_wef=True
    )
    
    if os.path.exists(wef_model_path):
        print(f"加载WEF模型权重: {wef_model_path}")
        wef_model.load_state_dict(torch.load(wef_model_path, map_location=device))
    else:
        print(f"警告: 未找到WEF模型权重文件 {wef_model_path}")
        print("将使用随机初始化的模型进行演示")
    
    wef_model = wef_model.to(device)
    
    # 为了对比，我们需要训练一个APE基线模型
    # 这里我们假设你也有一个APE模型的权重文件
    ape_model_path = 'best_ape_vit_ti.pth'  # 假设的APE模型路径
    
    ape_model = ImprovedViT_Ti(
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
        use_wef=False  # 使用APE
    )
    
    if os.path.exists(ape_model_path):
        print(f"加载APE模型权重: {ape_model_path}")
        ape_model.load_state_dict(torch.load(ape_model_path, map_location=device))
    else:
        print(f"警告: 未找到APE模型权重文件 {ape_model_path}")
        print("将训练一个快速的APE基线模型进行对比")
        ape_model = train_ape_baseline_quickly(ape_model, device)
    
    ape_model = ape_model.to(device)
    
    return wef_model, ape_model


def train_ape_baseline_quickly(model, device, epochs=20):
    """
    快速训练一个APE基线模型用于对比
    注意：这是一个简化的训练过程，主要用于演示
    """
    print("开始快速训练APE基线模型...")
    
    # 创建训练数据加载器
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                           std=[0.2673, 0.2564, 0.2762])
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root='/root/shared-nvme', train=True, download=False, transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                acc = 100.0 * correct / total
                pbar.set_postfix({'Loss': f'{running_loss/(batch_idx+1):.3f}', 'Acc': f'{acc:.2f}%'})
            
            # 为了节省时间，限制每个epoch的batch数量
            if batch_idx >= 100:
                break
    
    # 保存快速训练的模型
    torch.save(model.state_dict(), 'best_ape_vit_ti.pth')
    print("APE基线模型训练完成并已保存")
    
    return model


def run_occlusion_robustness_experiment():
    """
    运行完整的遮挡鲁棒性对比实验
    """
    print("=" * 60)
    print("WEF-PE vs APE 遮挡鲁棒性对比实验")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据加载器
    print("创建测试数据加载器...")
    test_loader = create_test_dataloader_cifar100(batch_size=32, num_workers=4)
    
    # 加载训练好的模型
    print("加载训练好的模型...")
    wef_model, ape_model = load_trained_models(device=device)
    
    # 定义遮挡率
    occlusion_ratios = [0.0, 0.1, 0.2, 0.3]
    print(f"测试遮挡率: {[f'{r:.1%}' for r in occlusion_ratios]}")
    
    # 评估WEF模型
    print("\n评估WEF-PE模型...")
    wef_results = evaluate_model_with_occlusion(wef_model, test_loader, device, occlusion_ratios)
    
    # 评估APE基线模型
    print("\n评估APE基线模型...")
    ape_results = evaluate_model_with_occlusion(ape_model, test_loader, device, occlusion_ratios)
    
    # 打印详细结果
    print("\n" + "=" * 50)
    print("实验结果总结")
    print("=" * 50)
    print(f"{'遮挡率':<10} {'WEF-PE':<10} {'APE':<10} {'改善':<10}")
    print("-" * 40)
    
    for ratio in occlusion_ratios:
        wef_acc = wef_results[ratio]
        ape_acc = ape_results[ratio]
        improvement = wef_acc - ape_acc
        print(f"{ratio:>6.1%}   {wef_acc:>7.2f}%   {ape_acc:>6.2f}%   {improvement:>+6.2f}pp")
    
    # 计算平均改善
    improvements = [wef_results[r] - ape_results[r] for r in occlusion_ratios[1:]]  # 排除无遮挡情况
    avg_improvement = np.mean(improvements)
    print(f"\n平均改善: {avg_improvement:+.2f} 个百分点")
    
    # 绘制对比图
    print("\n绘制对比图...")
    plot_occlusion_robustness_comparison(wef_results, ape_results)
    
    # 保存结果到JSON文件
    results_dict = {
        'wef_results': wef_results,
        'ape_results': ape_results,
        'occlusion_ratios': occlusion_ratios,
        'average_improvement': float(avg_improvement)
    }
    
    with open('occlusion_robustness_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print("实验结果已保存到: occlusion_robustness_results.json")
    print("\n实验完成! 🎉")
    
    return results_dict


if __name__ == "__main__":
    results = run_occlusion_robustness_experiment()