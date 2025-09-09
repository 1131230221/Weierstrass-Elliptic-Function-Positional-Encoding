import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from W_Ti import WEFPositionalEncoding
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class PatchPlusPosCurveAnalyzer:
    """
    分析patch特征+位置编码后的相似度随距离的变化
    """
    def __init__(self, 
                 d_model: int = 192,
                 img_size: int = 224,
                 patch_size: int = 16,
                 g2: float = 1.0,
                 g3: float = 0.0,
                 device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_per_side = img_size // patch_size
        self.num_patches = self.patches_per_side ** 2
        self.g2 = g2
        self.g3 = g3
        self.pos_encoding = WEFPositionalEncoding(
            d_model=d_model,
            max_h=self.patches_per_side,
            max_w=self.patches_per_side,
            g2=g2,
            g3=g3,
            device=self.device
        ).to(self.device)

    def get_patch_coordinates(self):
        coordinates = []
        for i in range(self.patches_per_side):
            for j in range(self.patches_per_side):
                coordinates.append((i, j))
        return coordinates

    def calculate_euclidean_distance(self, coord1, coord2):
        return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

    def generate_patch_plus_pos_embeddings(self):
        batch_size = 1
        h = w = self.patches_per_side
        # 随机patch特征
        patch_features = torch.randn(batch_size, 1 + h*w, self.d_model).to(self.device)
        with torch.no_grad():
            patch_plus_pos = self.pos_encoding(patch_features, h, w)
            patch_plus_pos = patch_plus_pos.squeeze(0)  # [seq_len, d_model]
            patch_plus_pos = patch_plus_pos[1:]  # 去除CLS
        return patch_plus_pos

    def analyze_patch_plus_pos_curve(self, num_distance_bins: int = 100):
        print("Generating patch+pos_encoding relative distance vs similarity curve...")
        patch_plus_pos = self.generate_patch_plus_pos_embeddings()
        coordinates = self.get_patch_coordinates()
        patch_plus_pos_norm = F.normalize(patch_plus_pos, p=2, dim=1)
        distances = []
        similarities = []
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                coord1 = coordinates[i]
                coord2 = coordinates[j]
                distance = self.calculate_euclidean_distance(coord1, coord2)
                cosine_sim = torch.dot(patch_plus_pos_norm[i], patch_plus_pos_norm[j]).item()
                distances.append(distance)
                similarities.append(cosine_sim)
        distances = np.array(distances)
        similarities = np.array(similarities)
        max_distance = distances.max()
        relative_distances = distances / max_distance * 100
        max_similarity = similarities.max()
        min_similarity = similarities.min()
        normalized_similarities = (similarities - min_similarity) / (max_similarity - min_similarity)
        relative_upper_bounds = 6 + normalized_similarities * 14
        distance_bins = np.linspace(0, 100, num_distance_bins)
        bin_centers = []
        bin_upper_bounds = []
        for i in range(len(distance_bins) - 1):
            bin_start = distance_bins[i]
            bin_end = distance_bins[i + 1]
            mask = (relative_distances >= bin_start) & (relative_distances < bin_end)
            if mask.sum() > 0:
                bin_center = (bin_start + bin_end) / 2
                bin_upper_bound = relative_upper_bounds[mask].mean()
                bin_centers.append(bin_center)
                bin_upper_bounds.append(bin_upper_bound)
        return {
            'relative_distances': np.array(bin_centers),
            'relative_upper_bounds': np.array(bin_upper_bounds),
            'raw_distances': relative_distances,
            'raw_upper_bounds': relative_upper_bounds,
            'num_points': len(distances)
        }

    def plot_distance_curve(self, results, save_dir='patch_plus_pos_curve_results'):
        os.makedirs(save_dir, exist_ok=True)
        mpl.rcParams['font.family'] = 'Times New Roman'
        mpl.rcParams['axes.labelweight'] = 'bold'
        mpl.rcParams['axes.titlesize'] = 16
        mpl.rcParams['axes.labelsize'] = 14
        mpl.rcParams['xtick.labelsize'] = 13
        mpl.rcParams['ytick.labelsize'] = 13
        mpl.rcParams['legend.fontsize'] = 13
        mpl.rcParams['axes.linewidth'] = 1.2
        mpl.rcParams['lines.linewidth'] = 2.5
        mpl.rcParams['figure.dpi'] = 300
        mpl.rcParams['savefig.dpi'] = 600
        plt.figure(figsize=(9, 6))
        plt.plot(results['relative_distances'], results['relative_upper_bounds'], 
                 linewidth=2.5, color='crimson', alpha=0.85)
        plt.xlabel('Relative Distance', fontsize=15, fontweight='bold')
        plt.ylabel('Relative Upper Bound', fontsize=15, fontweight='bold')
        plt.title('Patch+Pos Encoding: Relative Distance vs Similarity', fontsize=17, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        plt.ylim(6, 20)
        correlation = np.corrcoef(results['relative_distances'], results['relative_upper_bounds'])[0, 1]
        plt.text(0.02, 0.98, f'Correlation: {correlation:.4f}\nData points: {results["num_points"]}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/patch_plus_pos_curve.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'{save_dir}/patch_plus_pos_curve.pdf', bbox_inches='tight')
        plt.show()
        print(f"Curve plot saved to: {save_dir}/patch_plus_pos_curve.png and .pdf")

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    analyzer = PatchPlusPosCurveAnalyzer(
        d_model=192,
        img_size=224,
        patch_size=16,
        g2=1.0,
        g3=0.0,
        device=device
    )
    results = analyzer.analyze_patch_plus_pos_curve(num_distance_bins=80)
    analyzer.plot_distance_curve(results)
    print("\n🎉 Successfully generated patch+pos_encoding distance-similarity curve!")

if __name__ == "__main__":
    main() 