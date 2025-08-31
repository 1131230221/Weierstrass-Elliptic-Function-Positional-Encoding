# visualize_final_attention.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2
from typing import List, Dict, Any
import os

# --- 修复加载错误的补丁 ---
# 这是一个虚拟模块，用于解决在不同环境中加载模型时可能出现的依赖缺失问题。
import sys
class DummyExperimentConfig: pass
class DummyModelConfig: pass
class DummyTrainingConfig: pass
class DummyConfigModule:
    ExperimentConfig = DummyExperimentConfig
    ModelConfig = DummyModelConfig
    TrainingConfig = DummyTrainingConfig
sys.modules['config'] = DummyConfigModule()
# --- 补丁结束 ---


# 从您的 W_Ti.py 文件中导入必要的模型类
try:
    from W_Ti import ImprovedViT_Ti
except ImportError:
    print("Error: Could not find 'W_Ti.py'. Please ensure it's in the same directory.")
    exit()

class MultiLayerAttentionExtractor:
    """一个可以从模型所有Transformer层提取注意力权重的辅助类。"""
    def __init__(self, model: torch.nn.Module, num_layers: int):
        self.attention_maps = []
        self.hooks = []
        for i in range(num_layers):
            target_layer_name = f'blocks.{i}.attn'
            module_dict: Dict[str, torch.nn.Module] = dict(model.named_modules())
            if target_layer_name not in module_dict:
                raise ValueError(f"Layer '{target_layer_name}' not found.")
            target_layer = module_dict[target_layer_name]
            handle = target_layer.register_forward_hook(self.hook_fn)
            self.hooks.append(handle)

    def hook_fn(self, module: torch.nn.Module, inputs: Any, outputs: Any):
        self.attention_maps.append(outputs[1].detach().cpu()[0])

    def release(self):
        for handle in self.hooks:
            handle.remove()

def compute_attention_rollout(attention_maps: List[torch.Tensor]) -> torch.Tensor:
    """计算Attention Rollout。"""
    if not attention_maps: return torch.empty(0)
    rollout = torch.eye(attention_maps[0].shape[-1])
    for attn_map in attention_maps:
        attn_map_reweighted = 0.5 * attn_map + 0.5 * torch.eye(attn_map.shape[-1])
        rollout = torch.matmul(attn_map_reweighted, rollout)
    return rollout

def load_trained_model(use_wef: bool, model_path: str, device: torch.device):
    """加载一个训练好的、12层深度的ViT-Ti模型。"""
    
    # 两个模型都使用完全相同的架构
    model = ImprovedViT_Ti(
        img_size=224, patch_size=16, num_classes=100,
        embed_dim=192, depth=12, num_heads=3, use_wef=use_wef
    )

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 智能提取state_dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)

    except FileNotFoundError:
        print(f"致命错误: 在路径 '{model_path}' 未找到模型权重文件。")
        exit()
    except Exception as e:
        print(f"致命错误: 从 '{model_path}' 加载模型权重时发生错误。")
        print(f"错误详情: {e}")
        exit()

    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wef_model_path = '/root/best_wef_vit_ti.pth'
    ape_model_path = '/root/best_ape_vit_ti.pth'
    
    # --- 使用您指定的五张高清图片 ---
    image_paths = [
        '/root/cat.jpeg',
        '/root/Airplane.jpg',
        '/root/Sunflower.jpg',
        '/root/Golden_Gate_Bridge.jpg',
        '/root/Red_Sports_Car.jpg'
    ]
    
    print("Loading trained models...")
    wef_model = load_trained_model(use_wef=True, model_path=wef_model_path, device=device)
    ape_model = load_trained_model(use_wef=False, model_path=ape_model_path, device=device)
    print("Models loaded successfully.\n")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 两个模型现在都是12层
    wef_extractor = MultiLayerAttentionExtractor(wef_model, num_layers=12)
    ape_extractor = MultiLayerAttentionExtractor(ape_model, num_layers=12)

    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found at {image_path}, skipping.")
            continue
            
        original_pil_img = Image.open(image_path).convert('RGB')
        img_tensor = transform(original_pil_img).unsqueeze(0).to(device)
        
        wef_extractor.attention_maps = []
        with torch.no_grad(): _ = wef_model(img_tensor)
        attn_rollout_wef = compute_attention_rollout(wef_extractor.attention_maps)
        
        ape_extractor.attention_maps = []
        with torch.no_grad(): _ = ape_model(img_tensor)
        attn_rollout_ape = compute_attention_rollout(ape_extractor.attention_maps)
        
        if attn_rollout_wef.numel() == 0 or attn_rollout_ape.numel() == 0:
            print(f"Skipping image {i+1} due to empty attention map.")
            continue

        attn_map_wef = attn_rollout_wef[0, 1:].view(14, 14).numpy()
        attn_map_ape = attn_rollout_ape[0, 1:].view(14, 14).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        image_name = os.path.basename(image_path).split('.')[0]
        fig.suptitle(f'Attention Rollout on Hi-Res Image: {image_name}', fontsize=16)

        axes[0].imshow(original_pil_img.resize((224, 224)))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        def plot_attention_overlay(ax, original_img, attn_map, title):
            img_cv2 = cv2.cvtColor(np.array(original_img.resize((224, 224))), cv2.COLOR_RGB2BGR)
            attn_map_resized = cv2.resize(attn_map, (224, 224))
            attn_map_norm = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min())
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_norm), cv2.COLORMAP_JET)
            overlaid_img = cv2.addWeighted(img_cv2, 0.5, heatmap, 0.5, 0)
            ax.imshow(cv2.cvtColor(overlaid_img, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis('off')

        plot_attention_overlay(axes[1], original_pil_img, attn_map_wef, 'WEF-ViT Attention Rollout')
        plot_attention_overlay(axes[2], original_pil_img, attn_map_ape, 'APE-ViT Attention Rollout')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        output_filename = f'final_hires_attention_{i+1}_{image_name}.png'
        plt.savefig(output_filename, dpi=150)
        print(f"Saved visualization to {output_filename}")
        plt.close(fig)

    wef_extractor.release()
    ape_extractor.release()

if __name__ == "__main__":
    main()
