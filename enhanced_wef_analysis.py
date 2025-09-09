import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import os
from W_Ti import WEFPositionalEncoding, WeierstrassEllipticFunction
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedWEFAnalyzer:
    """
    增强版威尔斯特拉斯椭圆函数位置编码分析器
    专门研究椭圆函数的数学特性如何影响位置编码的相关性
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
        
        # 创建威尔斯特拉斯椭圆函数计算器
        self.wef = WeierstrassEllipticFunction(g2=g2, g3=g3, device=self.device)
        
        # 创建位置编码模块
        self.pos_encoding = WEFPositionalEncoding(
            d_model=d_model,
            max_h=self.patches_per_side,
            max_w=self.patches_per_side,
            g2=g2,
            g3=g3,
            device=self.device
        ).to(self.device)
        
    def get_all_patch_coordinates(self) -> List[Tuple[int, int]]:
        """获取所有patch的坐标"""
        coordinates = []
        for i in range(self.patches_per_side):
            for j in range(self.patches_per_side):
                coordinates.append((i, j))
        return coordinates
    
    def calculate_distance_metrics(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> Dict[str, float]:
        """计算多种距离度量"""
        x1, y1 = coord1
        x2, y2 = coord2
        
        return {
            'euclidean': np.sqrt((x1 - x2)**2 + (y1 - y2)**2),
            'manhattan': abs(x1 - x2) + abs(y1 - y2),
            'chebyshev': max(abs(x1 - x2), abs(y1 - y2)),
            'dx': abs(x1 - x2),
            'dy': abs(y1 - y2)
        }
    
    def generate_position_embeddings(self) -> torch.Tensor:
        """生成所有patch的位置编码"""
        batch_size = 1
        h = w = self.patches_per_side
        # 创建包含CLS token的dummy patches，seq_len = 1 + h*w
        dummy_patches = torch.randn(batch_size, 1 + h*w, self.d_model).to(self.device)
        
        with torch.no_grad():
            pos_embeddings = self.pos_encoding(dummy_patches, h, w)
            pos_embeddings = pos_embeddings.squeeze(0)  # [seq_len, embed_dim]
            
            # 去除CLS token，只保留patch位置编码
            pos_embeddings = pos_embeddings[1:]  # 现在是[h*w, embed_dim]
                
        return pos_embeddings
    
    def analyze_wef_properties(self) -> Dict:
        """分析威尔斯特拉斯椭圆函数的数学性质"""
        print("分析威尔斯特拉斯椭圆函数的数学性质...")
        
        # 生成复平面上的网格点
        x_range = np.linspace(-2, 2, 50)
        y_range = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x_range, y_range)
        Z = X + 1j * Y
        
        # 计算椭圆函数值
        z_tensor = torch.tensor(Z, dtype=torch.complex128, device=self.device)
        
        with torch.no_grad():
            wp_values, wp_prime_values = self.wef.wp_and_wp_prime(z_tensor)
            
        wp_real = wp_values.real.cpu().numpy()
        wp_imag = wp_values.imag.cpu().numpy()
        wp_magnitude = np.abs(wp_values.cpu().numpy())
        
        return {
            'X': X, 'Y': Y, 'Z': Z,
            'wp_real': wp_real,
            'wp_imag': wp_imag,
            'wp_magnitude': wp_magnitude
        }
    
    def comprehensive_distance_analysis(self) -> Dict:
        """全面的距离-相关性分析"""
        print("进行全面的距离-相关性分析...")
        
        # 生成位置编码
        pos_embeddings = self.generate_position_embeddings()
        coordinates = self.get_all_patch_coordinates()
        
        # 存储分析结果
        results = {
            'distances': {
                'euclidean': [],
                'manhattan': [],
                'chebyshev': [],
                'dx': [],
                'dy': []
            },
            'correlations': {
                'dot_product': [],
                'cosine_similarity': [],
                'pearson_correlation': []
            },
            'position_info': {
                'coord1': [],
                'coord2': []
            }
        }
        
        # 归一化位置编码用于计算余弦相似度
        pos_embeddings_norm = F.normalize(pos_embeddings, p=2, dim=1)
        
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):  # 只计算上三角，避免重复
                coord1 = coordinates[i]
                coord2 = coordinates[j]
                
                # 计算距离
                distances = self.calculate_distance_metrics(coord1, coord2)
                
                # 计算相关性度量
                embed1 = pos_embeddings[i]
                embed2 = pos_embeddings[j]
                embed1_norm = pos_embeddings_norm[i]
                embed2_norm = pos_embeddings_norm[j]
                
                dot_product = torch.dot(embed1, embed2).item()
                cosine_sim = torch.dot(embed1_norm, embed2_norm).item()
                
                # 计算皮尔逊相关系数（将embedding视为两个变量的观测值）
                embed1_np = embed1.cpu().numpy()
                embed2_np = embed2.cpu().numpy()
                pearson_corr = np.corrcoef(embed1_np, embed2_np)[0, 1]
                
                # 存储结果
                for dist_type, dist_value in distances.items():
                    results['distances'][dist_type].append(dist_value)
                
                results['correlations']['dot_product'].append(dot_product)
                results['correlations']['cosine_similarity'].append(cosine_sim)
                results['correlations']['pearson_correlation'].append(pearson_corr)
                
                results['position_info']['coord1'].append(coord1)
                results['position_info']['coord2'].append(coord2)
        
        return results
    
    def analyze_symmetry_properties(self) -> Dict:
        """分析位置编码的对称性质"""
        print("分析位置编码的对称性质...")
        
        pos_embeddings = self.generate_position_embeddings()
        coordinates = self.get_all_patch_coordinates()
        
        symmetry_results = {
            'horizontal_symmetry': [],
            'vertical_symmetry': [],
            'diagonal_symmetry': [],
            'rotational_symmetry': []
        }
        
        center = (self.patches_per_side - 1) / 2
        
        for i, coord in enumerate(coordinates):
            x, y = coord
            
            # 水平对称点
            x_sym = int(2 * center - x) if (2 * center - x) >= 0 and (2 * center - x) < self.patches_per_side else None
            if x_sym is not None and (x_sym, y) in coordinates:
                j = coordinates.index((x_sym, y))
                cosine_sim = F.cosine_similarity(pos_embeddings[i:i+1], pos_embeddings[j:j+1]).item()
                symmetry_results['horizontal_symmetry'].append(cosine_sim)
            
            # 垂直对称点
            y_sym = int(2 * center - y) if (2 * center - y) >= 0 and (2 * center - y) < self.patches_per_side else None
            if y_sym is not None and (x, y_sym) in coordinates:
                j = coordinates.index((x, y_sym))
                cosine_sim = F.cosine_similarity(pos_embeddings[i:i+1], pos_embeddings[j:j+1]).item()
                symmetry_results['vertical_symmetry'].append(cosine_sim)
            
            # 对角对称点
            if (y, x) in coordinates:
                j = coordinates.index((y, x))
                cosine_sim = F.cosine_similarity(pos_embeddings[i:i+1], pos_embeddings[j:j+1]).item()
                symmetry_results['diagonal_symmetry'].append(cosine_sim)
        
        return symmetry_results
    
    def create_heatmaps(self, results: Dict, save_dir: str = 'wef_analysis_results'):
        """创建相关性热力图"""
        os.makedirs(save_dir, exist_ok=True)
        
        pos_embeddings = self.generate_position_embeddings()
        
        # 计算所有patch对的余弦相似度矩阵
        similarity_matrix = torch.zeros(self.num_patches, self.num_patches)
        
        pos_embeddings_norm = F.normalize(pos_embeddings, p=2, dim=1)
        
        for i in range(self.num_patches):
            for j in range(self.num_patches):
                if i != j:
                    similarity = F.cosine_similarity(pos_embeddings_norm[i:i+1], pos_embeddings_norm[j:j+1]).item()
                    similarity_matrix[i, j] = similarity
                else:
                    similarity_matrix[i, j] = 1.0
        
        # 创建2D热力图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 相似度矩阵热力图
        im1 = axes[0].imshow(similarity_matrix.numpy(), cmap='RdYlBu_r', vmin=-1, vmax=1)
        axes[0].set_title('位置编码余弦相似度矩阵')
        axes[0].set_xlabel('Patch索引')
        axes[0].set_ylabel('Patch索引')
        plt.colorbar(im1, ax=axes[0])
        
        # 将相似度矩阵重塑为2D图像格式
        spatial_similarity = similarity_matrix.view(self.patches_per_side, self.patches_per_side, 
                                                   self.patches_per_side, self.patches_per_side)
        
        # 选择中心patch作为参考点
        center_idx = self.patches_per_side // 2
        center_similarity = spatial_similarity[center_idx, center_idx, :, :].numpy()
        
        im2 = axes[1].imshow(center_similarity, cmap='RdYlBu_r')
        axes[1].set_title(f'中心patch ({center_idx}, {center_idx}) 与其他patch的相似度')
        axes[1].set_xlabel('X坐标')
        axes[1].set_ylabel('Y坐标')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/similarity_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_comprehensive_analysis(self, results: Dict, save_dir: str = 'wef_analysis_results'):
        """绘制全面的分析结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass
        
        # 创建大型图形展示所有结果
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 距离vs余弦相似度 (2x2子图)
        distance_types = ['euclidean', 'manhattan', 'chebyshev']
        for i, dist_type in enumerate(distance_types):
            ax = plt.subplot(4, 3, i+1)
            plt.scatter(results['distances'][dist_type], 
                       results['correlations']['cosine_similarity'],
                       alpha=0.6, s=15)
            plt.xlabel(f'{dist_type.capitalize()} 距离')
            plt.ylabel('余弦相似度')
            plt.title(f'{dist_type.capitalize()} 距离 vs 余弦相似度')
            
            # 添加相关系数
            corr = np.corrcoef(results['distances'][dist_type], 
                              results['correlations']['cosine_similarity'])[0,1]
            plt.text(0.05, 0.95, f'相关系数: {corr:.3f}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. 距离vs内积
        for i, dist_type in enumerate(distance_types):
            ax = plt.subplot(4, 3, i+4)
            plt.scatter(results['distances'][dist_type], 
                       results['correlations']['dot_product'],
                       alpha=0.6, s=15, c='red')
            plt.xlabel(f'{dist_type.capitalize()} 距离')
            plt.ylabel('内积')
            plt.title(f'{dist_type.capitalize()} 距离 vs 内积')
            
            corr = np.corrcoef(results['distances'][dist_type], 
                              results['correlations']['dot_product'])[0,1]
            plt.text(0.05, 0.95, f'相关系数: {corr:.3f}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. X距离和Y距离的分别分析
        ax = plt.subplot(4, 3, 7)
        plt.scatter(results['distances']['dx'], 
                   results['correlations']['cosine_similarity'],
                   alpha=0.6, s=15, c='green', label='X方向距离')
        plt.scatter(results['distances']['dy'], 
                   results['correlations']['cosine_similarity'],
                   alpha=0.6, s=15, c='orange', label='Y方向距离')
        plt.xlabel('距离')
        plt.ylabel('余弦相似度')
        plt.title('X/Y方向距离 vs 余弦相似度')
        plt.legend()
        
        # 4. 距离分布直方图
        ax = plt.subplot(4, 3, 8)
        plt.hist(results['distances']['euclidean'], bins=20, alpha=0.7, 
                label='欧几里得距离', density=True)
        plt.hist(results['distances']['manhattan'], bins=20, alpha=0.7, 
                label='曼哈顿距离', density=True)
        plt.xlabel('距离')
        plt.ylabel('密度')
        plt.title('距离分布')
        plt.legend()
        
        # 5. 相似度分布直方图
        ax = plt.subplot(4, 3, 9)
        plt.hist(results['correlations']['cosine_similarity'], bins=30, alpha=0.7,
                label='余弦相似度', density=True)
        plt.hist(results['correlations']['dot_product'], bins=30, alpha=0.7,
                label='内积', density=True)
        plt.xlabel('值')
        plt.ylabel('密度')
        plt.title('相关性度量分布')
        plt.legend()
        
        # 6-9. 距离区间分析
        self.plot_distance_bins_analysis(results, fig, start_subplot=10)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 创建热力图
        self.create_heatmaps(results, save_dir)
    
    def plot_distance_bins_analysis(self, results: Dict, fig, start_subplot: int = 10):
        """绘制距离区间分析"""
        euclidean_dist = np.array(results['distances']['euclidean'])
        cosine_sim = np.array(results['correlations']['cosine_similarity'])
        dot_products = np.array(results['correlations']['dot_product'])
        
        # 创建距离区间
        max_dist = euclidean_dist.max()
        bins = np.linspace(0, max_dist, 8)
        
        bin_centers = []
        avg_cosine = []
        avg_dot = []
        counts = []
        
        for i in range(len(bins) - 1):
            mask = (euclidean_dist >= bins[i]) & (euclidean_dist < bins[i+1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                avg_cosine.append(cosine_sim[mask].mean())
                avg_dot.append(dot_products[mask].mean())
                counts.append(mask.sum())
        
        # 绘制区间平均值曲线
        ax = plt.subplot(4, 3, start_subplot)
        plt.plot(bin_centers, avg_cosine, 'o-', label='平均余弦相似度', linewidth=2)
        plt.xlabel('欧几里得距离')
        plt.ylabel('平均余弦相似度')
        plt.title('距离区间内平均相似度变化')
        plt.grid(True, alpha=0.3)
        
        ax = plt.subplot(4, 3, start_subplot+1)
        plt.plot(bin_centers, avg_dot, 'o-', color='red', label='平均内积', linewidth=2)
        plt.xlabel('欧几里得距离')
        plt.ylabel('平均内积')
        plt.title('距离区间内平均内积变化')
        plt.grid(True, alpha=0.3)
        
        ax = plt.subplot(4, 3, start_subplot+2)
        plt.bar(range(len(bin_centers)), counts, alpha=0.7)
        plt.xlabel('距离区间')
        plt.ylabel('样本数量')
        plt.title('各距离区间样本分布')
        plt.xticks(range(len(bin_centers)), [f'{b:.1f}' for b in bin_centers], rotation=45)
    
    def print_detailed_statistics(self, results: Dict):
        """打印详细的统计信息"""
        print("\n" + "="*60)
        print("威尔斯特拉斯椭圆函数位置编码详细分析报告")
        print("="*60)
        
        # 基本统计
        print(f"\n数据集规模:")
        print(f"  - 图像大小: {self.img_size}x{self.img_size}")
        print(f"  - 补丁大小: {self.patch_size}x{self.patch_size}")
        print(f"  - 补丁数量: {self.num_patches}")
        print(f"  - 分析的补丁对数量: {len(results['distances']['euclidean'])}")
        
        # 距离统计
        print(f"\n距离统计:")
        for dist_type in ['euclidean', 'manhattan', 'chebyshev']:
            dist_values = np.array(results['distances'][dist_type])
            print(f"  {dist_type.capitalize()} 距离:")
            print(f"    平均值: {dist_values.mean():.3f}")
            print(f"    标准差: {dist_values.std():.3f}")
            print(f"    范围: [{dist_values.min():.3f}, {dist_values.max():.3f}]")
        
        # 相关性统计
        print(f"\n相关性度量统计:")
        for corr_type in ['cosine_similarity', 'dot_product', 'pearson_correlation']:
            corr_values = np.array(results['correlations'][corr_type])
            # 过滤NaN值
            corr_values = corr_values[~np.isnan(corr_values)]
            print(f"  {corr_type.replace('_', ' ').title()}:")
            print(f"    平均值: {corr_values.mean():.4f}")
            print(f"    标准差: {corr_values.std():.4f}")
            print(f"    范围: [{corr_values.min():.4f}, {corr_values.max():.4f}]")
        
        # 距离-相关性分析
        print(f"\n距离与相关性的相关系数:")
        for dist_type in ['euclidean', 'manhattan', 'chebyshev']:
            for corr_type in ['cosine_similarity', 'dot_product']:
                dist_arr = np.array(results['distances'][dist_type])
                corr_arr = np.array(results['correlations'][corr_type])
                corr_arr = corr_arr[~np.isnan(corr_arr)]
                
                if len(dist_arr) == len(corr_arr):
                    correlation = np.corrcoef(dist_arr, corr_arr)[0,1]
                    print(f"  {dist_type.capitalize()} 距离 vs {corr_type.replace('_', ' ')}: {correlation:.4f}")
    
    def run_enhanced_analysis(self):
        """运行增强版分析"""
        print(f"开始增强版威尔斯特拉斯椭圆函数位置编码分析")
        print(f"椭圆函数参数: g2={self.g2}, g3={self.g3}")
        print(f"使用设备: {self.device}")
        print("-" * 60)
        
        try:
            # 1. 全面的距离-相关性分析
            results = self.comprehensive_distance_analysis()
            
            # 2. 对称性分析
            symmetry_results = self.analyze_symmetry_properties()
            
            # 3. 打印统计信息
            self.print_detailed_statistics(results)
            
            # 4. 绘制结果
            self.plot_comprehensive_analysis(results)
            
            # 5. 分析对称性结果
            print(f"\n对称性分析结果:")
            for sym_type, values in symmetry_results.items():
                if values:
                    print(f"  {sym_type.replace('_', ' ').title()}: "
                          f"平均相似度 = {np.mean(values):.4f} ± {np.std(values):.4f}")
            
            print(f"\n分析完成！结果已保存到 'wef_analysis_results' 目录")
            
        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if device == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建分析器并运行分析
    analyzer = EnhancedWEFAnalyzer(
        d_model=192,
        img_size=224,
        patch_size=16,
        g2=1.0,
        g3=0.0,
        device=device
    )
    
    analyzer.run_enhanced_analysis()


if __name__ == "__main__":
    main() 