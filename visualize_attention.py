# visualize_attention.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import Dict, Any
import matplotlib
import numpy as np
import random
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 从您的 W_Ti.py 文件中导入必要的模型类
# 请确保 W_Ti.py 与此脚本在同一目录下
try:
    from W_Ti import ImprovedViT_Ti
except ImportError:
    print("Error: Could not find 'W_Ti.py' or the 'ImprovedViT_Ti' class within it.")
    print("Please ensure 'W_Ti.py' is in the same directory as this script.")
    exit()

class AttentionExtractor:
    """
    A helper class to extract attention weights from a specified nn.Module.
    This is achieved by registering a forward hook without modifying the original model code.
    """
    def __init__(self, model: torch.nn.Module, target_layer_name: str = 'blocks.0.attn'):
        """
        Initializes the hook.
        
        Args:
            model (torch.nn.Module): The model to attach the hook to.
            target_layer_name (str): The name of the target MultiheadAttention layer.
                                     'blocks.0.attn' refers to the attention layer in the first Transformer block.
        """
        self.attention_weights = None
        
        module_dict: Dict[str, torch.nn.Module] = dict(model.named_modules())
        if target_layer_name not in module_dict:
            raise ValueError(f"Error: Layer named '{target_layer_name}' not found in the model.")
        
        target_layer = module_dict[target_layer_name]
        
        self.handle = target_layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module: torch.nn.Module, inputs: Any, outputs: Any):
        """
        The hook function called during each forward pass.
        For nn.MultiheadAttention, the output is a tuple (attn_output, attn_output_weights).
        We are interested in the second element: the attention weights.
        """
        self.attention_weights = outputs[1].detach().cpu()

    def release(self):
        """
        Removes the hook to free up resources.
        """
        self.handle.remove()


def visualize_initial_attention(img_size: int = 224, patch_size: int = 16):
    """
    Main function to instantiate models, extract, and visualize initial attention maps.
    """
    # --- 固定随机种子 ---
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=====================================================")
    print(f"Experiment Environment:")
    print(f"  - PyTorch Version: {torch.__version__}")
    print(f"  - Using Device: {device}")
    print(f"=====================================================\n")
    
    print("Step 1/7: Initializing two ViT-Ti models (WEF vs APE)...")
    wef_model = ImprovedViT_Ti(img_size=img_size, patch_size=patch_size, use_wef=True).to(device).eval()
    ape_model = ImprovedViT_Ti(img_size=img_size, patch_size=patch_size, use_wef=False).to(device).eval()
    print("Model initialization complete.\n")

    print("Step 2/7: Registering hooks on the first attention layer...")
    wef_extractor = AttentionExtractor(wef_model, 'blocks.0.attn')
    ape_extractor = AttentionExtractor(ape_model, 'blocks.0.attn')
    print("Hook registration complete.\n")

    print("Step 3/7: Creating a dummy input tensor...")
    dummy_input = torch.zeros(1, 3, img_size, img_size, device=device)
    print("Dummy input created.\n")

    print("Step 4/7: Performing forward pass to capture attention weights...")
    with torch.no_grad():
        wef_model(dummy_input)
        ape_model(dummy_input)
    print("Forward pass complete, attention weights captured.\n")

    print("Step 5/7: Extracting attention weights...")
    wef_attn = wef_extractor.attention_weights.squeeze(0)
    ape_attn = ape_extractor.attention_weights.squeeze(0)
    print("Weight extraction complete.\n")

    wef_extractor.release()
    ape_extractor.release()

    print("Step 6/7: Processing data for visualization...")
    num_patches_side = img_size // patch_size
    center_patch_linear_idx = (num_patches_side * num_patches_side // 2) + (num_patches_side // 2)
    center_token_idx = center_patch_linear_idx + 1
    
    wef_attn_map = wef_attn[center_token_idx, 1:] 
    ape_attn_map = ape_attn[center_token_idx, 1:]
    
    wef_attn_map_2d = wef_attn_map.view(num_patches_side, num_patches_side)
    ape_attn_map_2d = ape_attn_map.view(num_patches_side, num_patches_side)
    print("Data processing complete.\n")

    print("Step 7/7: Generating visualization plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # 去掉总标题，不再使用 fig.suptitle

    vmin = min(wef_attn_map_2d.min(), ape_attn_map_2d.min())
    vmax = max(wef_attn_map_2d.max(), ape_attn_map_2d.max())

    # 左图
    sns.heatmap(wef_attn_map_2d.numpy(), ax=axes[0], cmap='viridis', cbar=True,
                square=True, vmin=vmin, vmax=vmax, cbar_kws={'label': 'Attention Weight', 'shrink': 0.85, 'aspect': 20, 'pad': 0.02})
    axes[0].set_title('WEF-ViT (Shows Locality Bias)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Patch X Coordinate', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Patch Y Coordinate', fontsize=14, fontweight='bold')
    axes[0].text(-0.15, 1.05, '(a)', transform=axes[0].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # 右图
    sns.heatmap(ape_attn_map_2d.numpy(), ax=axes[1], cmap='viridis', cbar=True,
                square=True, vmin=vmin, vmax=vmax, cbar_kws={'label': 'Attention Weight', 'shrink': 0.85, 'aspect': 20, 'pad': 0.02})
    axes[1].set_title('Standard APE-ViT (Shows Random Distribution)', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Patch X Coordinate', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Patch Y Coordinate', fontsize=14, fontweight='bold')
    axes[1].text(-0.15, 1.05, '(b)', transform=axes[1].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 1], pad=2.0, w_pad=2.5)
    
    # --- 关键修改 ---
    # 保存为 png 和 pdf
    output_filename_png = 'attention_map_comparison.png'
    output_filename_pdf = 'attention_map_comparison.pdf'
    plt.savefig(output_filename_png, dpi=150, bbox_inches='tight')
    plt.savefig(output_filename_pdf, dpi=300, bbox_inches='tight')
    
    print(f"Plot generation complete! Images saved as '{output_filename_png}' and '{output_filename_pdf}'.")
    print("Please check the file explorer on the left to view the images.")


if __name__ == "__main__":
    visualize_initial_attention()
