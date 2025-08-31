import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import affine, rotate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Union
import os
import json
from tqdm import tqdm
import random
from W_Ti import ImprovedViT_Ti
import matplotlib
from matplotlib import font_manager
import cv2
from PIL import Image


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


def _format_affine_label(params: Dict) -> str:
    """将仿射参数字典格式化为简洁的人类可读标签。

    例如：
        {} -> Baseline
        {"scale": 1.1, "angle": 0, "translate": [0, 0], "shear": [0, 0]} -> Scale 1.1x
        {"scale": 0.9, ...} -> Scale 0.9x
        {"translate": [5, 5]} 或 translate 不为 0 -> Translate (5,5)
        {"shear": [5, 0]} -> Shear (5,0)
        其余组合将拼接非零项
    """
    if not params:
        return 'Baseline'

    pieces: List[str] = []

    scale = params.get('scale')
    if isinstance(scale, (int, float)) and abs(scale - 1.0) > 1e-6:
        pieces.append(f"Scale {scale:.2f}x")

    angle = params.get('angle')
    if isinstance(angle, (int, float)) and abs(angle) > 1e-6:
        pieces.append(f"Rotate {angle}°")

    translate = params.get('translate')
    if isinstance(translate, (list, tuple)) and any(abs(float(t)) > 1e-6 for t in translate):
        tx, ty = translate[0], translate[1]
        pieces.append(f"Translate ({tx},{ty})")

    shear = params.get('shear')
    if isinstance(shear, (list, tuple)) and any(abs(float(s)) > 1e-6 for s in shear):
        sx, sy = shear[0], shear[1]
        pieces.append(f"Shear ({sx},{sy})")

    if not pieces:
        return 'Baseline'
    return ' + '.join(pieces)


class GeometricTransform:
    """几何变换类，支持旋转和仿射变换"""
    
    def __init__(self, transform_type: str = 'rotation'):
        """
        Args:
            transform_type: 变换类型 ('rotation' 或 'affine')
        """
        self.transform_type = transform_type
    
    def apply_rotation(self, image: torch.Tensor, angle: float) -> torch.Tensor:
        """
        应用旋转变换
        Args:
            image: 输入图像tensor [C, H, W]
            angle: 旋转角度（度）
        Returns:
            旋转后的图像tensor
        """
        if isinstance(image, torch.Tensor):
            # 转换为PIL图像进行旋转
            image_pil = transforms.ToPILImage()(image)
        else:
            image_pil = image
        
        # 应用旋转，使用白色填充
        rotated = rotate(image_pil, angle, fill=0)
        
        # 转换回tensor
        if isinstance(image, torch.Tensor):
            return transforms.ToTensor()(rotated)
        else:
            return rotated
    
    def apply_affine(self, image: torch.Tensor, params: Dict) -> torch.Tensor:
        """
        应用仿射变换
        Args:
            image: 输入图像tensor [C, H, W]
            params: 仿射变换参数字典
                   包含: translate, scale, shear, angle
        Returns:
            变换后的图像tensor
        """
        if isinstance(image, torch.Tensor):
            image_pil = transforms.ToPILImage()(image)
        else:
            image_pil = image
        
        # 应用仿射变换
        transformed = affine(
            image_pil,
            angle=params.get('angle', 0),
            translate=params.get('translate', [0, 0]),
            scale=params.get('scale', 1.0),
            shear=params.get('shear', [0, 0]),
            fill=0
        )
        
        if isinstance(image, torch.Tensor):
            return transforms.ToTensor()(transformed)
        else:
            return transformed
    
    def __call__(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """应用变换"""
        if self.transform_type == 'rotation':
            angle = kwargs.get('angle', 0)
            return self.apply_rotation(image, angle)
        elif self.transform_type == 'affine':
            params = kwargs.get('params', {})
            return self.apply_affine(image, params)
        else:
            raise ValueError(f"不支持的变换类型: {self.transform_type}")


def evaluate_model_with_transforms(model, dataloader, device, transform_configs, transform_type='rotation'):
    """
    评估模型在几何变换下的性能
    
    Args:
        model: 待评估的模型
        dataloader: 数据加载器
        device: 计算设备
        transform_configs: 变换配置列表
        transform_type: 变换类型 ('rotation' 或 'affine')
    
    Returns:
        results: 包含不同变换下准确率的字典
    """
    model.eval()
    results = {}
    
    # 创建几何变换对象
    geometric_transform = GeometricTransform(transform_type)
    
    with torch.no_grad():
        for config in tqdm(transform_configs, desc=f"Evaluating {transform_type}"):
            correct = 0
            total = 0
            
            # 根据变换类型设置描述
            if transform_type == 'rotation':
                angle = config
                desc = f"Rotation angle {angle}°"
                config_key = f"rotation_{angle}"
            else:
                desc = f"Affine transform {config}"
                config_key = f"affine_{hash(str(config))}"
            
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, leave=False, desc=desc)):
                # 对每个batch应用几何变换
                if (transform_type == 'rotation' and config != 0) or \
                   (transform_type == 'affine' and config != {}):
                    
                    transformed_images = []
                    for img in images:
                        # 先反归一化
                        mean = torch.tensor([0.5071, 0.4865, 0.4409]).view(3, 1, 1)
                        std = torch.tensor([0.2673, 0.2564, 0.2762]).view(3, 1, 1)
                        unnorm_img = img * std + mean
                        unnorm_img = torch.clamp(unnorm_img, 0, 1)
                        
                        # 应用变换
                        if transform_type == 'rotation':
                            transformed_img = geometric_transform(unnorm_img, angle=config)
                        else:
                            transformed_img = geometric_transform(unnorm_img, params=config)
                        
                        # 重新归一化
                        transformed_img = (transformed_img - mean) / std
                        transformed_images.append(transformed_img)
                    
                    images = torch.stack(transformed_images)
                
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 限制评估的batch数量以节省时间
                if batch_idx >= 50:
                    break
            
            accuracy = 100.0 * correct / total
            results[config_key] = {
                'accuracy': accuracy,
                'config': config,
                'description': desc
            }
            print(f"{desc}: accuracy = {accuracy:.2f}%")
    
    return results


def plot_geometric_invariance_comparison(wef_results, ape_results, transform_type='rotation', 
                                       save_path='geometric_invariance.png'):
    """
    绘制几何不变性对比图
    """
    _apply_publication_style()
    
    if transform_type == 'rotation':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
        
        # 提取旋转角度和准确率
        angles = []
        wef_accuracies = []
        ape_accuracies = []
        
        for key in sorted(wef_results.keys()):
            if 'rotation' in key:
                angle = wef_results[key]['config']
                angles.append(angle)
                wef_accuracies.append(wef_results[key]['accuracy'])
                ape_accuracies.append(ape_results[key]['accuracy'])
        
        # 绘制准确率对比
        ax1.plot(angles, wef_accuracies, marker='o', linestyle='-', linewidth=2.2, markersize=6,
                 color='#2E86AB', label='WEF-PE (Ours)')
        ax1.plot(angles, ape_accuracies, marker='s', linestyle='--', linewidth=2.0, markersize=6,
                 color='#A23B72', label='APE (Baseline)')
        
        ax1.set_xlabel('Rotation Angle (degrees)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy vs Rotation Angle', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.25)
        ax1.set_xticks(angles)
        
        # 添加数值标注
        for i, (angle, wef_acc, ape_acc) in enumerate(zip(angles, wef_accuracies, ape_accuracies)):
            ax1.annotate(f'{wef_acc:.1f}', (angle, wef_acc),
                        textcoords="offset points", xytext=(0,8), ha='center', 
                        fontsize=9, color='#2E86AB')
            ax1.annotate(f'{ape_acc:.1f}', (angle, ape_acc),
                        textcoords="offset points", xytext=(0,-12), ha='center', 
                        fontsize=9, color='#A23B72')
        
        # 绘制性能下降对比
        wef_baseline = wef_accuracies[0]
        ape_baseline = ape_accuracies[0]
        
        wef_degradation = [(wef_baseline - acc) for acc in wef_accuracies]
        ape_degradation = [(ape_baseline - acc) for acc in ape_accuracies]
        
        ax2.plot(angles, wef_degradation, marker='o', linestyle='-', linewidth=2.2, markersize=6,
                 color='#2E86AB', label='WEF-PE (Ours)')
        ax2.plot(angles, ape_degradation, marker='s', linestyle='--', linewidth=2.0, markersize=6,
                 color='#A23B72', label='APE (Baseline)')
        
        ax2.set_xlabel('Rotation Angle (degrees)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Degradation (pp)', fontsize=12, fontweight='bold')
        ax2.set_title('Performance Degradation', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.25)
        ax2.set_xticks(angles)
        
        # 添加改善幅度标注
        for i, (angle, wef_deg, ape_deg) in enumerate(zip(angles[1:], wef_degradation[1:], ape_degradation[1:])):
            if ape_deg > wef_deg:
                improvement = ape_deg - wef_deg
                ax2.annotate(f'+{improvement:.1f} pp', 
                            (angle, (wef_deg + ape_deg) / 2), 
                            textcoords="offset points", xytext=(10,0), ha='left', 
                            fontsize=9, color='green', fontweight='bold')
        
        # 面板标签
        ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')
        ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
    
    else:
        # 仿射变换的可视化（简化版）
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 提取仿射变换结果
        transform_names = []
        wef_accuracies = []
        ape_accuracies = []
        
        for key in sorted(wef_results.keys()):
            if 'affine' in key:
                # 使用更简洁的标签替代原始字典字符串，避免标签过长与难以阅读
                params = wef_results[key]['config']
                transform_names.append(_format_affine_label(params))
                wef_accuracies.append(wef_results[key]['accuracy'])
                ape_accuracies.append(ape_results[key]['accuracy'])
        
        x = np.arange(len(transform_names))
        width = 0.35
        
        ax.bar(x - width/2, wef_accuracies, width, label='WEF-PE (Ours)', color='#2E86AB', alpha=0.8)
        ax.bar(x + width/2, ape_accuracies, width, label='APE (Baseline)', color='#A23B72', alpha=0.8)
        
        ax.set_xlabel('Affine Transformation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy under Affine Transformations', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        # 更友好的标签排版：旋转角度、右对齐并增加底部留白
        ax.set_xticklabels(transform_names, rotation=30, ha='right')
        ax.margins(x=0.02)
        ax.set_ylim(bottom=max(0, min(wef_accuracies + ape_accuracies) - 5))
        ax.legend()
        ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    
    # 保存图片
    base, ext = os.path.splitext(save_path)
    plt.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
    try:
        plt.savefig(f'{base}.pdf', bbox_inches='tight')
    except Exception:
        pass
    
    print(f"Figure saved to: {save_path}")
    plt.show()


def create_test_dataloader_cifar100_geometric(batch_size=64, num_workers=4):
    """创建用于几何变换测试的CIFAR-100数据加载器"""
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


def load_trained_models_for_geometric_test(wef_model_path='best_wef_vit_ti.pth', 
                                          ape_model_path='best_ape_vit_ti.pth', 
                                          device='cuda'):
    """加载训练好的模型用于几何不变性测试"""
    
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
    
    wef_model = wef_model.to(device)
    
    # 创建APE模型并加载权重
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
        use_wef=False
    )
    
    if os.path.exists(ape_model_path):
        print(f"加载APE模型权重: {ape_model_path}")
        ape_model.load_state_dict(torch.load(ape_model_path, map_location=device))
    else:
        print(f"警告: 未找到APE模型权重文件 {ape_model_path}")
        print("将使用快速训练的APE基线模型")
        from occlusion_experiment import train_ape_baseline_quickly
        ape_model = train_ape_baseline_quickly(ape_model, device)
    
    ape_model = ape_model.to(device)
    
    return wef_model, ape_model


def run_rotation_invariance_experiment():
    """运行旋转不变性实验"""
    print("=" * 60)
    print("WEF-PE vs APE 旋转不变性对比实验")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据加载器
    print("创建测试数据加载器...")
    test_loader = create_test_dataloader_cifar100_geometric(batch_size=32, num_workers=4)
    
    # 加载训练好的模型
    print("加载训练好的模型...")
    wef_model, ape_model = load_trained_models_for_geometric_test(device=device)
    
    # 定义旋转角度
    rotation_angles = [0, 5, 10, 15, 30]  # 度
    print(f"测试旋转角度: {rotation_angles}°")
    
    # 评估WEF模型
    print("\n评估WEF-PE模型旋转不变性...")
    wef_results = evaluate_model_with_transforms(
        wef_model, test_loader, device, rotation_angles, 'rotation'
    )
    
    # 评估APE基线模型
    print("\n评估APE基线模型旋转不变性...")
    ape_results = evaluate_model_with_transforms(
        ape_model, test_loader, device, rotation_angles, 'rotation'
    )
    
    return wef_results, ape_results


def run_affine_invariance_experiment():
    """运行仿射变换不变性实验"""
    print("=" * 60)
    print("WEF-PE vs APE 仿射变换不变性对比实验")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据加载器
    print("创建测试数据加载器...")
    test_loader = create_test_dataloader_cifar100_geometric(batch_size=32, num_workers=4)
    
    # 加载训练好的模型
    print("加载训练好的模型...")
    wef_model, ape_model = load_trained_models_for_geometric_test(device=device)
    
    # 定义仿射变换参数
    affine_transforms = [
        {},  # 无变换（基线）
        {'scale': 1.1, 'angle': 0, 'translate': [0, 0], 'shear': [0, 0]},  # 轻微缩放
        {'scale': 0.9, 'angle': 0, 'translate': [0, 0], 'shear': [0, 0]},  # 轻微缩小
        {'scale': 1.0, 'angle': 0, 'translate': [5, 5], 'shear': [0, 0]},  # 平移
        {'scale': 1.0, 'angle': 0, 'translate': [0, 0], 'shear': [5, 0]},  # 轻微剪切
    ]
    
    print(f"测试仿射变换配置数量: {len(affine_transforms)}")
    
    # 评估WEF模型
    print("\n评估WEF-PE模型仿射不变性...")
    wef_results = evaluate_model_with_transforms(
        wef_model, test_loader, device, affine_transforms, 'affine'
    )
    
    # 评估APE基线模型
    print("\n评估APE基线模型仿射不变性...")
    ape_results = evaluate_model_with_transforms(
        ape_model, test_loader, device, affine_transforms, 'affine'
    )
    
    return wef_results, ape_results


def print_detailed_results(wef_results, ape_results, experiment_type='rotation'):
    """打印详细的实验结果"""
    print("\n" + "=" * 60)
    print("实验结果详细分析")
    print("=" * 60)
    
    if experiment_type == 'rotation':
        print(f"{'旋转角度':<12} {'WEF-PE':<12} {'APE':<12} {'改善':<12}")
        print("-" * 50)
        
        angles = []
        improvements = []
        
        for key in sorted(wef_results.keys()):
            if 'rotation' in key:
                angle = wef_results[key]['config']
                wef_acc = wef_results[key]['accuracy']
                ape_acc = ape_results[key]['accuracy']
                improvement = wef_acc - ape_acc
                
                angles.append(f"{angle}°")
                improvements.append(improvement)
                
                print(f"{angle:>8}°   {wef_acc:>8.2f}%   {ape_acc:>8.2f}%   {improvement:>+8.2f}pp")
    
    else:  # affine
        print(f"{'变换类型':<20} {'WEF-PE':<12} {'APE':<12} {'改善':<12}")
        print("-" * 58)
        
        improvements = []
        
        for key in sorted(wef_results.keys()):
            if 'affine' in key:
                desc = wef_results[key]['description']
                wef_acc = wef_results[key]['accuracy']
                ape_acc = ape_results[key]['accuracy']
                improvement = wef_acc - ape_acc
                
                improvements.append(improvement)
                
                print(f"{desc:<18} {wef_acc:>8.2f}%   {ape_acc:>8.2f}%   {improvement:>+8.2f}pp")
    
    # 计算平均改善
    if len(improvements) > 1:  # 排除基线情况
        avg_improvement = np.mean(improvements[1:])
        print(f"\n平均改善: {avg_improvement:+.2f} 个百分点")
    
    return improvements


def run_complete_geometric_invariance_experiment():
    """运行完整的几何不变性实验"""
    print("🔄 开始几何不变性综合实验...")
    
    # 1. 旋转不变性实验
    print("\n📐 第一阶段：旋转不变性测试")
    wef_rotation_results, ape_rotation_results = run_rotation_invariance_experiment()
    
    # 打印旋转实验详细结果
    rotation_improvements = print_detailed_results(
        wef_rotation_results, ape_rotation_results, 'rotation'
    )
    
    # 绘制旋转不变性对比图
    print("\n绘制旋转不变性对比图...")
    plot_geometric_invariance_comparison(
        wef_rotation_results, ape_rotation_results, 
        'rotation', 'rotation_invariance.png'
    )
    
    # 2. 仿射变换不变性实验
    print("\n🔧 第二阶段：仿射变换不变性测试")
    wef_affine_results, ape_affine_results = run_affine_invariance_experiment()
    
    # 打印仿射变换实验详细结果
    affine_improvements = print_detailed_results(
        wef_affine_results, ape_affine_results, 'affine'
    )
    
    # 绘制仿射变换不变性对比图
    print("\n绘制仿射变换不变性对比图...")
    plot_geometric_invariance_comparison(
        wef_affine_results, ape_affine_results, 
        'affine', 'affine_invariance.png'
    )
    
    # 保存完整结果
    complete_results = {
        'rotation_results': {
            'wef': wef_rotation_results,
            'ape': ape_rotation_results,
            'average_improvement': float(np.mean(rotation_improvements[1:]))
        },
        'affine_results': {
            'wef': wef_affine_results,
            'ape': ape_affine_results,
            'average_improvement': float(np.mean(affine_improvements[1:]))
        }
    }
    
    with open('geometric_invariance_results.json', 'w', encoding='utf-8') as f:
        json.dump(complete_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("🎯 几何不变性实验总结")
    print("=" * 60)
    print(f"旋转不变性平均改善: {np.mean(rotation_improvements[1:]):+.2f} 个百分点")
    print(f"仿射变换不变性平均改善: {np.mean(affine_improvements[1:]):+.2f} 个百分点")
    print("\n实验结果已保存到: geometric_invariance_results.json")
    print("几何不变性实验完成! ✅")
    
    return complete_results


if __name__ == "__main__":
    # 运行完整的几何不变性实验
    results = run_complete_geometric_invariance_experiment()