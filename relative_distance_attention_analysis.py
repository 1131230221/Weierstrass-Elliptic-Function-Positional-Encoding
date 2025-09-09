import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import os
from W_Ti import WEFPositionalEncoding, ImprovedViT_Ti
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AttentionAnalyzer:
    """
    分析威尔斯特拉斯椭圆函数位置编码中相对距离与自注意力交互强度的关系
    """
    def __init__(self, 
                 model_path: str = None,
                 img_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 192,
                 num_heads: int = 3,
                 device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 创建模型
        self.model = ImprovedViT_Ti(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_wef=True
        ).to(self.device)
        
        # 如果提供了模型路径，加载预训练权重
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"已加载预训练模型: {model_path}")
        else:
            print("使用随机初始化的模型进行分析")
        
        self.model.eval()
        
    def get_patch_coordinates(self) -> List[Tuple[int, int]]:
        """获取所有patch的坐标"""
        patches_per_side = self.img_size // self.patch_size
        coordinates = []
        for i in range(patches_per_side):
            for j in range(patches_per_side):
                coordinates.append((i, j))
        return coordinates
    
    def calculate_euclidean_distance(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> float:
        """计算两个patch之间的欧几里得距离"""
        return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
    
    def calculate_manhattan_distance(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> int:
        """计算两个patch之间的曼哈顿距离"""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])
    
    def extract_attention_weights(self, batch_size: int = 1) -> Dict[int, torch.Tensor]:
        """
        提取模型各层的注意力权重
        """
        # 创建随机输入
        x = torch.randn(batch_size, 3, self.img_size, self.img_size).to(self.device)
        
        attention_weights = {}
        
        def get_attention_hook(layer_idx):
            def hook(module, input, output):
                # output: (batch_size, seq_len, seq_len)
                if isinstance(output, tuple):
                    attn_weights = output[1]  # 注意力权重通常是第二个输出
                else:
                    attn_weights = output
                
                if attn_weights is not None:
                    # 去除CLS token (假设CLS token在第0位)
                    patch_attn = attn_weights[:, :, 1:, 1:]  # [batch, heads, patches, patches]
                    attention_weights[layer_idx] = patch_attn.detach().cpu()
            return hook
        
        # 注册钩子到每个Transformer块的注意力层
        hooks = []
        for i, block in enumerate(self.model.transformer_blocks):
            if hasattr(block, 'attn'):
                hook = block.attn.register_forward_hook(get_attention_hook(i))
                hooks.append(hook)
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(x)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def analyze_distance_attention_relationship(self, num_samples: int = 10) -> Dict:
        """
        分析相对距离与注意力强度的关系
        """
        print("开始分析相对距离与注意力强度的关系...")
        
        # 获取patch坐标
        coordinates = self.get_patch_coordinates()
        
        # 收集多次采样的结果
        all_results = {
            'euclidean_distances': [],
            'manhattan_distances': [],
            'attention_strengths': [],
            'layer_wise_results': {}
        }
        
        for sample in range(num_samples):
            print(f"处理样本 {sample + 1}/{num_samples}")
            
            # 提取注意力权重
            attention_weights = self.extract_attention_weights()
            
            if not attention_weights:
                print("警告：未能提取到注意力权重，尝试修改hook机制")
                continue
            
            # 对每一层进行分析
            for layer_idx, attn_weight in attention_weights.items():
                if layer_idx not in all_results['layer_wise_results']:
                    all_results['layer_wise_results'][layer_idx] = {
                        'euclidean_distances': [],
                        'manhattan_distances': [],
                        'attention_strengths': []
                    }
                
                # attn_weight: [batch, heads, patches, patches]
                batch_size, num_heads, num_patches, _ = attn_weight.shape
                
                # 对所有头求平均
                avg_attn = attn_weight.mean(dim=1).squeeze(0)  # [patches, patches]
                
                # 计算所有patch对的距离和注意力强度
                for i in range(num_patches):
                    for j in range(num_patches):
                        if i != j:  # 排除自注意力
                            coord_i = coordinates[i]
                            coord_j = coordinates[j]
                            
                            euclidean_dist = self.calculate_euclidean_distance(coord_i, coord_j)
                            manhattan_dist = self.calculate_manhattan_distance(coord_i, coord_j)
                            attention_strength = avg_attn[i, j].item()
                            
                            all_results['layer_wise_results'][layer_idx]['euclidean_distances'].append(euclidean_dist)
                            all_results['layer_wise_results'][layer_idx]['manhattan_distances'].append(manhattan_dist)
                            all_results['layer_wise_results'][layer_idx]['attention_strengths'].append(attention_strength)
                            
                            if sample == 0:  # 只在第一个样本时收集全局结果
                                all_results['euclidean_distances'].append(euclidean_dist)
                                all_results['manhattan_distances'].append(manhattan_dist)
                                all_results['attention_strengths'].append(attention_strength)
        
        return all_results
    
    def create_alternative_analysis(self) -> Dict:
        """
        创建替代分析方法：直接分析位置编码的内积
        """
        print("使用位置编码内积进行分析...")
        
        # 创建位置编码
        pos_encoding = WEFPositionalEncoding(
            d_model=self.embed_dim,
            max_h=self.img_size // self.patch_size,
            max_w=self.img_size // self.patch_size,
            device=self.device
        ).to(self.device)
        
        # 生成位置编码
        batch_size = 1
        h = w = self.img_size // self.patch_size
        # 创建包含CLS token的dummy patches，seq_len = 1 + h*w
        dummy_patches = torch.randn(batch_size, 1 + h*w, self.embed_dim).to(self.device)
        
        with torch.no_grad():
            pos_embeddings = pos_encoding(dummy_patches, h, w)  # [batch, seq_len, embed_dim]
            pos_embeddings = pos_embeddings.squeeze(0)  # [seq_len, embed_dim]
            
            # 去除CLS token，只保留patch位置编码
            pos_embeddings = pos_embeddings[1:]  # 去除CLS token，现在是[h*w, embed_dim]
        
        # 计算所有位置编码对的内积
        coordinates = self.get_patch_coordinates()
        
        results = {
            'euclidean_distances': [],
            'manhattan_distances': [],
            'dot_products': [],
            'cosine_similarities': []
        }
        
        # L2归一化位置编码用于计算余弦相似度
        pos_embeddings_norm = F.normalize(pos_embeddings, p=2, dim=1)
        
        for i in range(len(coordinates)):
            for j in range(len(coordinates)):
                if i != j:
                    coord_i = coordinates[i]
                    coord_j = coordinates[j]
                    
                    euclidean_dist = self.calculate_euclidean_distance(coord_i, coord_j)
                    manhattan_dist = self.calculate_manhattan_distance(coord_i, coord_j)
                    
                    # 计算位置编码的内积
                    dot_product = torch.dot(pos_embeddings[i], pos_embeddings[j]).item()
                    
                    # 计算余弦相似度
                    cosine_sim = torch.dot(pos_embeddings_norm[i], pos_embeddings_norm[j]).item()
                    
                    results['euclidean_distances'].append(euclidean_dist)
                    results['manhattan_distances'].append(manhattan_dist)
                    results['dot_products'].append(dot_product)
                    results['cosine_similarities'].append(cosine_sim)
        
        return results
    
    def plot_results(self, results: Dict, save_dir: str = 'attention_analysis_results'):
        """
        绘制分析结果
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass  # 使用默认样式
        
        # 1. 位置编码内积分析
        if 'dot_products' in results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 欧几里得距离 vs 内积
            axes[0, 0].scatter(results['euclidean_distances'], results['dot_products'], 
                             alpha=0.6, s=20, c='blue')
            axes[0, 0].set_xlabel('欧几里得距离')
            axes[0, 0].set_ylabel('位置编码内积')
            axes[0, 0].set_title('欧几里得距离 vs 位置编码内积')
            
            # 曼哈顿距离 vs 内积
            axes[0, 1].scatter(results['manhattan_distances'], results['dot_products'], 
                             alpha=0.6, s=20, c='red')
            axes[0, 1].set_xlabel('曼哈顿距离')
            axes[0, 1].set_ylabel('位置编码内积')
            axes[0, 1].set_title('曼哈顿距离 vs 位置编码内积')
            
            # 欧几里得距离 vs 余弦相似度
            axes[1, 0].scatter(results['euclidean_distances'], results['cosine_similarities'], 
                             alpha=0.6, s=20, c='green')
            axes[1, 0].set_xlabel('欧几里得距离')
            axes[1, 0].set_ylabel('位置编码余弦相似度')
            axes[1, 0].set_title('欧几里得距离 vs 位置编码余弦相似度')
            
            # 曼哈顿距离 vs 余弦相似度
            axes[1, 1].scatter(results['manhattan_distances'], results['cosine_similarities'], 
                             alpha=0.6, s=20, c='purple')
            axes[1, 1].set_xlabel('曼哈顿距离')
            axes[1, 1].set_ylabel('位置编码余弦相似度')
            axes[1, 1].set_title('曼哈顿距离 vs 位置编码余弦相似度')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/position_encoding_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 统计分析
            self.print_correlation_stats(results)
    
    def print_correlation_stats(self, results: Dict):
        """
        打印相关性统计信息
        """
        if 'dot_products' in results:
            euclidean_dist = np.array(results['euclidean_distances'])
            manhattan_dist = np.array(results['manhattan_distances'])
            dot_products = np.array(results['dot_products'])
            cosine_sim = np.array(results['cosine_similarities'])
            
            print("\n=== 相关性分析结果 ===")
            print(f"欧几里得距离 vs 位置编码内积的相关系数: {np.corrcoef(euclidean_dist, dot_products)[0,1]:.4f}")
            print(f"曼哈顿距离 vs 位置编码内积的相关系数: {np.corrcoef(manhattan_dist, dot_products)[0,1]:.4f}")
            print(f"欧几里得距离 vs 位置编码余弦相似度的相关系数: {np.corrcoef(euclidean_dist, cosine_sim)[0,1]:.4f}")
            print(f"曼哈顿距离 vs 位置编码余弦相似度的相关系数: {np.corrcoef(manhattan_dist, cosine_sim)[0,1]:.4f}")
            
            # 分析距离范围内的变化趋势
            self.analyze_distance_bins(euclidean_dist, dot_products, cosine_sim)
    
    def analyze_distance_bins(self, distances: np.ndarray, dot_products: np.ndarray, cosine_sim: np.ndarray):
        """
        分析不同距离区间内的注意力变化趋势
        """
        print("\n=== 距离区间分析 ===")
        
        # 定义距离区间
        max_dist = distances.max()
        bins = np.linspace(0, max_dist, 8)
        
        print("距离区间\t平均内积\t\t平均余弦相似度\t样本数")
        print("-" * 60)
        
        for i in range(len(bins) - 1):
            mask = (distances >= bins[i]) & (distances < bins[i+1])
            if mask.sum() > 0:
                avg_dot = dot_products[mask].mean()
                avg_cosine = cosine_sim[mask].mean()
                count = mask.sum()
                print(f"[{bins[i]:.1f}, {bins[i+1]:.1f})\t{avg_dot:.4f}\t\t{avg_cosine:.4f}\t\t{count}")
    
    def run_comprehensive_analysis(self, model_path: str = None):
        """
        运行完整的分析流程
        """
        print("开始威尔斯特拉斯椭圆函数位置编码的相对距离-注意力强度分析")
        print(f"设备: {self.device}")
        print(f"图像大小: {self.img_size}x{self.img_size}")
        print(f"补丁大小: {self.patch_size}x{self.patch_size}")
        print(f"补丁数量: {self.num_patches}")
        print("-" * 50)
        
        try:
            # 方法1：直接分析位置编码的内积（更可靠）
            print("\n方法1：分析位置编码的内积和余弦相似度")
            pos_encoding_results = self.create_alternative_analysis()
            
            # 绘制结果
            self.plot_results(pos_encoding_results)
            
            # 尝试方法2：分析实际注意力权重（如果可能）
            print("\n方法2：尝试分析实际的注意力权重")
            try:
                attention_results = self.analyze_distance_attention_relationship(num_samples=3)
                if attention_results['layer_wise_results']:
                    print("成功提取注意力权重！")
                    # 这里可以添加注意力权重的进一步分析
                else:
                    print("未能成功提取注意力权重")
            except Exception as e:
                print(f"注意力权重分析失败: {e}")
                print("继续使用位置编码分析结果")
            
        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    主函数
    """
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查GPU可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if device == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建分析器
    analyzer = AttentionAnalyzer(
        model_path=None,  # 如果有预训练模型，可以在这里指定路径
        img_size=224,
        patch_size=16,
        embed_dim=192,
        num_heads=3,
        device=device
    )
    
    # 运行分析
    analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main() 