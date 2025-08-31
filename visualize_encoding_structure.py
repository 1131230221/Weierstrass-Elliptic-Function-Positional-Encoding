# visualize_encoding_structure.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import math
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

# 从您的 W_Ti.py 文件中导入必要的模型类
# 请确保 W_Ti.py 与此脚本在同一目录下
try:
    from W_Ti import WEFPositionalEncoding
except ImportError:
    print("Error: Could not find 'W_Ti.py' or the 'WEFPositionalEncoding' class within it.")
    print("Please ensure 'W_Ti.py' is in the same directory as this script.")
    exit()

def get_wef_patch_encodings(d_model, h, w, device):
    """
    实例化WEFPositionalEncoding并提取其生成的图块编码。
    """
    # 实例化编码模块
    wef_encoder = WEFPositionalEncoding(
        d_model=d_model,
        max_h=h,
        max_w=w
    ).to(device).eval()

    # --- 复现 forward 方法中的编码生成逻辑 ---
    # 生成patch坐标
    row_idx = torch.arange(h, device=device).unsqueeze(1).repeat(1, w).reshape(-1)
    col_idx = torch.arange(w, device=device).repeat(h)
    
    # 归一化坐标
    u = (col_idx.float() + 0.5) / w
    v = (row_idx.float() + 0.5) / h
    
    # 映射到复平面
    omega_3_prime = torch.nn.functional.softplus(wef_encoder.alpha_learn).clamp(0.02, 8.0)
    z_real = u * (2 * wef_encoder.wef.omega1.real) * 0.4
    z_imag = v * (2 * omega_3_prime) * 0.4
    z = torch.complex(z_real, z_imag)
    
    # 计算WEF值
    wp, wp_prime = wef_encoder.wef.wp_and_wp_prime(z)
    
    # 应用tanh压缩
    alpha_scale = torch.exp(wef_encoder.log_alpha_scale).clamp(0.002, 0.8)
    wp_real_compressed = torch.tanh(alpha_scale * wp.real)
    wp_imag_compressed = torch.tanh(alpha_scale * wp.imag)
    wp_prime_real_compressed = torch.tanh(alpha_scale * wp_prime.real)
    wp_prime_imag_compressed = torch.tanh(alpha_scale * wp_prime.imag)
    
    features = torch.stack([
        wp_real_compressed,
        wp_imag_compressed,
        wp_prime_real_compressed,
        wp_prime_imag_compressed
    ], dim=1).float()
    
    # 投影到d_model维度
    patch_encodings = wef_encoder.projection(features)
    
    return patch_encodings.detach()

def get_ape_patch_encodings(num_patches, embed_dim):
    """
    生成标准的可学习绝对位置编码 (APE)。
    """
    # 创建一个与ViT中 pos_embedding 类似的可学习参数
    pos_embedding = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
    # 使用截断正态分布进行初始化，与ViT标准做法保持一致
    nn.init.trunc_normal_(pos_embedding, std=0.02)
    # 提取图块部分的编码 (忽略CLS token)
    patch_encodings = pos_embedding[0, 1:, :]
    return patch_encodings.detach()

def main():
    """
    主函数，用于生成编码、降维、计算相似度并进行可视化。
    """
    # --- 配置参数 ---
    img_size = 224
    patch_size = 16
    embed_dim = 192 # ViT-Ti 配置
    num_patches_side = img_size // patch_size
    num_patches = num_patches_side ** 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=====================================================")
    print("Executing Plan A: Visualizing Positional Encoding Structure")
    print(f"  - Using Device: {device}")
    print("=====================================================\n")

    # --- 1. 生成两种位置编码 ---
    print("Step 1/4: Generating WEF and APE positional encodings...")
    wef_encodings = get_wef_patch_encodings(embed_dim, num_patches_side, num_patches_side, device).cpu().numpy()
    ape_encodings = get_ape_patch_encodings(num_patches, embed_dim).cpu().numpy()
    print("Encodings generated.\n")

    # --- 2. PCA 降维 ---
    print("Step 2/4: Performing PCA for dimensionality reduction...")
    pca = PCA(n_components=2)
    wef_pca = pca.fit_transform(wef_encodings)
    ape_pca = pca.fit_transform(ape_encodings)
    print("PCA complete.\n")

    # --- 3. 计算相似度矩阵 ---
    print("Step 3/4: Calculating cosine similarity matrices...")
    wef_sim = torch.nn.functional.cosine_similarity(torch.from_numpy(wef_encodings).unsqueeze(1), torch.from_numpy(wef_encodings).unsqueeze(0), dim=-1)
    ape_sim = torch.nn.functional.cosine_similarity(torch.from_numpy(ape_encodings).unsqueeze(1), torch.from_numpy(ape_encodings).unsqueeze(0), dim=-1)
    print("Similarity matrices calculated.\n")

    # --- 4. 绘图 ---
    print("Step 4/4: Generating visualization plot...")
    # ====== 美化设置 ======
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    # 去掉总标题

    # 为PCA图创建颜色映射，根据原始空间位置着色
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, num_patches_side), np.linspace(0, 1, num_patches_side))
    colors = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(num_patches)], axis=1)

    # PCA - WEF
    axes[0, 0].scatter(wef_pca[:, 0], wef_pca[:, 1], c=colors, s=32, edgecolor='k', linewidth=0.5, alpha=0.85)
    axes[0, 0].set_title('PCA of WEF Encodings (Structured Manifold)', fontsize=16, fontweight='bold')
    axes[0, 0].set_xlabel('Principal Component 1', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Principal Component 2', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    axes[0, 0].set_xlim(-13, 13)
    axes[0, 0].set_ylim(-13, 13)
    axes[0, 0].set_aspect('equal', 'box')
    axes[0, 0].set_box_aspect(1)
    axes[0, 0].text(-0.13, 1.08, '(a)', transform=axes[0, 0].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)

    # PCA - APE
    axes[0, 1].scatter(ape_pca[:, 0], ape_pca[:, 1], c=colors, s=32, edgecolor='k', linewidth=0.5, alpha=0.85)
    axes[0, 1].set_title('PCA of APE Encodings (Unstructured Cloud)', fontsize=16, fontweight='bold')
    axes[0, 1].set_xlabel('Principal Component 1', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Principal Component 2', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    axes[0, 1].set_xlim(-0.13, 0.13)
    axes[0, 1].set_ylim(-0.13, 0.13)
    axes[0, 1].set_aspect('equal', 'box')
    axes[0, 1].set_box_aspect(1)
    axes[0, 1].text(-0.13, 1.08, '(b)', transform=axes[0, 1].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)

    # Similarity Matrix - WEF
    cbar_kws = {'label': 'Cosine Similarity', 'shrink': 0.85, 'aspect': 20, 'pad': 0.02, 'format': '%.1f'}
    sns.heatmap(wef_sim.numpy(), ax=axes[1, 0], cmap='viridis', square=True, cbar=True, cbar_kws=cbar_kws)
    axes[1, 0].set_title('Similarity Matrix of WEF (Periodic Structure)', fontsize=16, fontweight='bold')
    axes[1, 0].set_xlabel('Patch Index', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Patch Index', fontsize=14, fontweight='bold')
    axes[1, 0].text(-0.13, 1.08, '(c)', transform=axes[1, 0].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
    axes[1, 0].spines['top'].set_visible(False)
    axes[1, 0].spines['right'].set_visible(False)

    # Similarity Matrix - APE
    sns.heatmap(ape_sim.numpy(), ax=axes[1, 1], cmap='viridis', square=True, cbar=True, cbar_kws=cbar_kws)
    axes[1, 1].set_title('Similarity Matrix of APE (Random Noise)', fontsize=16, fontweight='bold')
    axes[1, 1].set_xlabel('Patch Index', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Patch Index', fontsize=14, fontweight='bold')
    axes[1, 1].text(-0.13, 1.08, '(d)', transform=axes[1, 1].transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 1], pad=2.0, h_pad=2.5, w_pad=2.5)
    
    # 保存图片到文件
    output_filename_png = 'encoding_structure_comparison.png'
    output_filename_pdf = 'encoding_structure_comparison.pdf'
    plt.savefig(output_filename_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_filename_pdf, dpi=600, bbox_inches='tight')
    
    print(f"Plot generation complete! Images saved as '{output_filename_png}' and '{output_filename_pdf}'.")
    print("Please check the file explorer on the left to view the images.")

if __name__ == "__main__":
    main()
