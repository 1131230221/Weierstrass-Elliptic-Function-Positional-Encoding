#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
威尔斯特拉斯椭圆函数位置编码(WEF-PE) 高分辨率查找表预计算系统
基于论文实现的完整离线预计算框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
from typing import Dict, Tuple, Optional, Union
import argparse
from pathlib import Path

class WeierstrassEllipticFunctionLUT:
    """
    威尔斯特拉斯椭圆函数计算器 - 用于生成高分辨率查找表
    """
    def __init__(self, 
                 g2: float = 1.0,
                 g3: float = 0.0,
                 eps: float = 1e-8,
                 alpha_scale: float = 0.15,
                 device: torch.device = None,
                 computation_mode: str = "pretraining"):
        """
        参数:
            g2: 椭圆不变量 g2
            g3: 椭圆不变量 g3  
            eps: 数值稳定性小量
            alpha_scale: tanh压缩缩放因子
            device: 计算设备
            computation_mode: 计算模式 ("pretraining" 或 "finetuning")
        """
        self.g2 = g2
        self.g3 = g3
        self.eps = eps
        self.alpha_scale = alpha_scale
        self.computation_mode = computation_mode
        self.device = device if device is not None else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        print(f"计算模式: {computation_mode}")
        
        # 计算判别式
        discriminant = g2**3 - 27*g3**2
        assert abs(discriminant) > eps, f"判别式过于接近零: {discriminant}"
        
        # 对于双纽线情况 (g3=0), 使用精确的半周期值
        if abs(g3) < eps:
            # 精确值: omega1 = Gamma(1/4)^2 / sqrt(2*pi)
            self.omega1 = torch.tensor(2.62205755429212,
                                     device=self.device,
                                     dtype=torch.complex128)
            self.omega3 = torch.tensor(complex(0, 2.62205755429212),
                                     device=self.device,
                                     dtype=torch.complex128)
        else:
            # 一般情况需要数值计算周期
            self.omega1 = torch.tensor(complex(1.0, 0.0),
                                     device=self.device,
                                     dtype=torch.complex128)
            self.omega3 = torch.tensor(complex(0.0, 1.0),
                                     device=self.device,
                                     dtype=torch.complex128)
    
    def _lattice_summation(self, z: torch.Tensor, max_m: int = 12, max_n: int = 12) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        改进的格点求和法 - 用于预训练模式
        
        返回:
            Tuple[torch.Tensor, torch.Tensor]: wp_sum, wp_prime_sum
        """
        z = z.to(torch.complex128)
        wp_sum = torch.zeros_like(z, dtype=torch.complex128)
        wp_prime_sum = torch.zeros_like(z, dtype=torch.complex128)
        
        # 生成按模长排序的格点
        lattice_points = []
        for m in range(-max_m, max_m + 1):
            for n in range(-max_n, max_n + 1):
                if m == 0 and n == 0:
                    continue
                w = 2 * m * self.omega1 + 2 * n * self.omega3
                lattice_points.append((torch.abs(w), w, m, n))
        
        # 按模长排序以改善收敛性
        lattice_points.sort(key=lambda x: float(x[0].real) if isinstance(x[0], torch.Tensor) else x[0])
        
        # 计算级数项
        for _, w, m, n in lattice_points:
            if isinstance(w, torch.Tensor):
                w = w.to(z.device)
            diff = z - w
            
            # 避免除零
            mask = torch.abs(diff) > self.eps * 15
            if mask.any():
                diff_masked = diff[mask]
                w_term = 1.0/w**2 if abs(w) > self.eps else 0.0
                
                # 更稳定的计算方式
                wp_term = 1.0/diff_masked**2 - w_term
                wp_prime_term = -2.0/diff_masked**3
                
                # 裁剪极值防止数值爆炸
                wp_term = torch.clamp(wp_term.real, -5e3, 5e3) + \
                         1j * torch.clamp(wp_term.imag, -5e3, 5e3)
                wp_prime_term = torch.clamp(wp_prime_term.real, -5e3, 5e3) + \
                               1j * torch.clamp(wp_prime_term.imag, -5e3, 5e3)
                
                wp_sum[mask] += wp_term
                wp_prime_sum[mask] += wp_prime_term
        
        return wp_sum, wp_prime_sum
    
    def _fourier_approximation(self, z: torch.Tensor, K: int = 10, 
                              beta: float = 0.1, gamma: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        快速收敛傅里叶近似法 - 用于微调模式
        
        参数:
            z: 复数输入
            K: 截断项数
            beta: 稳定性参数
            gamma: 幅度参数
            
        返回:
            Tuple[torch.Tensor, torch.Tensor]: wp, wp_prime
        """
        z = z.to(torch.complex128)
        
        # 归一化坐标
        u_prime = z.real / self.omega1.real
        v_prime = z.imag / self.omega3.imag
        
        # 主项处理奇点
        wp_main = 1.0 / (torch.abs(z)**2 + beta)
        
        # 傅里叶级数项
        wp_correction = torch.zeros_like(z, dtype=torch.complex128)
        for k in range(1, K + 1):
            term1 = torch.cos(k * np.pi * u_prime) * torch.exp(-k * np.pi * torch.abs(v_prime))
            term2 = torch.sin(k * np.pi * v_prime) * torch.exp(-k * np.pi * torch.abs(u_prime))
            wp_correction += (gamma / k**2) * (term1 + 1j * term2)
        
        wp = wp_main + wp_correction.real  # 取实部
        
        # 近似计算导数
        wp_prime = -2.0 * z / (torch.abs(z)**2 + beta)**2
        
        return wp, wp_prime
    
    def compute_wef_features(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算WEF 4维特征向量
        
        参数:
            z: 复数坐标张量
            
        返回:
            torch.Tensor: 4维特征向量 [Re(℘), Im(℘), Re(℘'), Im(℘')]
        """
        z = z.to(torch.complex128)
        
        # 处理接近原点的点
        near_origin = torch.abs(z) < self.eps * 15
        
        # 初始化结果
        wp = torch.zeros_like(z, dtype=torch.complex128)
        wp_prime = torch.zeros_like(z, dtype=torch.complex128)
        
        # 对于不接近原点的点进行计算
        valid_mask = ~near_origin
        if valid_mask.any():
            z_valid = z[valid_mask]
            
            if self.computation_mode == "pretraining":
                # 主项
                wp_main = 1.0 / z_valid**2
                wp_prime_main = -2.0 / z_valid**3
                
                # 级数项
                wp_series, wp_prime_series = self._lattice_summation(z_valid)
                
                wp[valid_mask] = wp_main + wp_series
                wp_prime[valid_mask] = wp_prime_main + wp_prime_series
                
            else:  # finetuning mode
                wp_result, wp_prime_result = self._fourier_approximation(z_valid)
                wp[valid_mask] = wp_result
                wp_prime[valid_mask] = wp_prime_result
        
        # 对于接近原点的点，使用大值但避免inf
        large_value = 5e2
        wp[near_origin] = large_value
        wp_prime[near_origin] = large_value
        
        # 最终裁剪确保数值稳定性
        wp = torch.clamp(wp.real, -1e4, 1e4) + 1j * torch.clamp(wp.imag, -1e4, 1e4)
        wp_prime = torch.clamp(wp_prime.real, -1e4, 1e4) + 1j * torch.clamp(wp_prime.imag, -1e4, 1e4)
        
        # 提取4维特征
        features = torch.stack([
            wp.real,
            wp.imag,
            wp_prime.real,
            wp_prime.imag
        ], dim=-1)
        
        # 应用tanh压缩
        features = torch.tanh(self.alpha_scale * features)
        
        return features.float()


class WEFPELUTGenerator:
    """
    WEF-PE高分辨率查找表生成器
    """
    def __init__(self, 
                 resolution: int = 512,
                 alpha_u: float = 1.0,
                 alpha_v: float = 1.0,
                 computation_mode: str = "pretraining",
                 device: Optional[torch.device] = None):
        """
        参数:
            resolution: 查找表分辨率
            alpha_u: 水平方向缩放因子
            alpha_v: 垂直方向缩放因子
            computation_mode: 计算模式
            device: 计算设备
        """
        self.resolution = resolution
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.computation_mode = computation_mode
        self.device = device if device is not None else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化椭圆函数计算器
        self.wef_calculator = WeierstrassEllipticFunctionLUT(
            device=self.device,
            computation_mode=computation_mode
        )
        
        print(f"初始化WEF-PE查找表生成器:")
        print(f"  - 分辨率: {resolution}x{resolution}")
        print(f"  - 缩放因子: α_u={alpha_u}, α_v={alpha_v}")
        print(f"  - 计算模式: {computation_mode}")
        print(f"  - 设备: {device}")
    
    def generate_coordinate_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成归一化坐标网格
        
        返回:
            Tuple[torch.Tensor, torch.Tensor]: (u_grid, v_grid)
        """
        # 生成[0, 1]范围内的均匀网格
        u = torch.linspace(0, 1, self.resolution, device=self.device)
        v = torch.linspace(0, 1, self.resolution, device=self.device)
        
        # 创建网格
        u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
        
        return u_grid, v_grid
    
    def coordinates_to_complex(self, u_grid: torch.Tensor, v_grid: torch.Tensor) -> torch.Tensor:
        """
        将坐标映射到复平面
        
        参数:
            u_grid: 水平坐标网格
            v_grid: 垂直坐标网格
            
        返回:
            torch.Tensor: 复数坐标网格
        """
        # 映射到复平面: z = αu * u * 2Re(ω1) + i * αv * v * 2Im(ω3)
        real_part = self.alpha_u * u_grid * 2 * self.wef_calculator.omega1.real
        imag_part = self.alpha_v * v_grid * 2 * self.wef_calculator.omega3.imag
        
        z_grid = real_part + 1j * imag_part
        
        return z_grid
    
    def generate_lut(self, batch_size: int = 1024, save_progress: bool = True) -> torch.Tensor:
        """
        生成高分辨率查找表
        
        参数:
            batch_size: 批处理大小
            save_progress: 是否保存进度
            
        返回:
            torch.Tensor: 形状为[resolution, resolution, 4]的查找表
        """
        print(f"开始生成 {self.resolution}x{self.resolution} 分辨率的WEF-PE查找表...")
        
        # 生成坐标网格
        u_grid, v_grid = self.generate_coordinate_grid()
        z_grid = self.coordinates_to_complex(u_grid, v_grid)
        
        # 展平为一维以便批处理
        total_points = self.resolution * self.resolution
        z_flat = z_grid.reshape(-1)
        
        # 初始化结果张量
        lut = torch.zeros(total_points, 4, device=self.device, dtype=torch.float32)
        
        # 分批计算
        num_batches = (total_points + batch_size - 1) // batch_size
        
        print(f"总共 {total_points} 个点，分 {num_batches} 批处理")
        
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_points)
            
            # 提取当前批次
            z_batch = z_flat[start_idx:end_idx]
            
            # 计算WEF特征
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                features_batch = self.wef_calculator.compute_wef_features(z_batch)
            
            # 存储结果
            lut[start_idx:end_idx] = features_batch
            
            # 进度显示
            if (batch_idx + 1) % max(1, num_batches // 20) == 0:
                elapsed = time.time() - start_time
                progress = (batch_idx + 1) / num_batches
                eta = elapsed / progress * (1 - progress)
                print(f"进度: {progress*100:.1f}% ({batch_idx+1}/{num_batches}) | "
                      f"已用时间: {elapsed:.1f}s | 预计剩余: {eta:.1f}s")
        
        # 重新整形为原始网格形状
        lut = lut.reshape(self.resolution, self.resolution, 4)
        
        elapsed_time = time.time() - start_time
        print(f"查找表生成完成! 总用时: {elapsed_time:.2f}s")
        
        return lut
    
    def save_lut(self, lut: torch.Tensor, save_dir: str, metadata: Optional[Dict] = None):
        """
        保存查找表到文件
        
        参数:
            lut: 查找表张量
            save_dir: 保存目录
            metadata: 元数据字典
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存查找表
        lut_file = save_path / "wef_pe_lut.pt"
        torch.save(lut.cpu(), lut_file)
        print(f"查找表已保存至: {lut_file}")
        
        # 保存元数据
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "resolution": self.resolution,
            "alpha_u": self.alpha_u,
            "alpha_v": self.alpha_v,
            "computation_mode": self.computation_mode,
            "g2": self.wef_calculator.g2,
            "g3": self.wef_calculator.g3,
            "alpha_scale": self.wef_calculator.alpha_scale,
            "omega1": [self.wef_calculator.omega1.real.item(), self.wef_calculator.omega1.imag.item()],
            "omega3": [self.wef_calculator.omega3.real.item(), self.wef_calculator.omega3.imag.item()],
            "shape": list(lut.shape),
            "dtype": str(lut.dtype)
        })
        
        metadata_file = save_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"元数据已保存至: {metadata_file}")


class WEFPEFastInference:
    """
    基于查找表的快速推理类
    """
    def __init__(self, lut_path: str, metadata_path: str = None):
        """
        参数:
            lut_path: 查找表文件路径
            metadata_path: 元数据文件路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载查找表
        self.lut = torch.load(lut_path, map_location=self.device)
        print(f"查找表加载完成，形状: {self.lut.shape}")
        
        # 加载元数据
        if metadata_path is None:
            metadata_path = Path(lut_path).parent / "metadata.json"
        
        if Path(metadata_path).exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print("元数据加载完成")
        else:
            self.metadata = {}
            print("警告: 未找到元数据文件")
    
    def interpolate_features(self, u_coords: torch.Tensor, v_coords: torch.Tensor) -> torch.Tensor:
        """
        基于双线性插值获取位置编码
        
        参数:
            u_coords: 水平归一化坐标 [0, 1]
            v_coords: 垂直归一化坐标 [0, 1]
            
        返回:
            torch.Tensor: 插值后的4维特征向量
        """
        # 确保坐标在有效范围内
        u_coords = torch.clamp(u_coords, 0, 1)
        v_coords = torch.clamp(v_coords, 0, 1)
        
        # 准备用于grid_sample的坐标
        # grid_sample需要[-1, 1]范围的坐标
        grid_u = 2 * u_coords - 1
        grid_v = 2 * v_coords - 1
        
        # 重新整理维度用于grid_sample
        # 输入形状: [batch_size, height, width, 2]
        batch_size = u_coords.shape[0]
        height, width = u_coords.shape[1], u_coords.shape[2]
        
        grid = torch.stack([grid_u, grid_v], dim=-1)  # [B, H, W, 2]
        
        # 重新整理LUT用于grid_sample: [1, 4, resolution, resolution]
        lut_input = self.lut.permute(2, 0, 1).unsqueeze(0)  # [1, 4, res, res]
        
        # 使用双线性插值
        interpolated = F.grid_sample(
            lut_input, 
            grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=False
        )
        
        # 重新整理输出形状: [batch_size, height, width, 4]
        interpolated = interpolated.squeeze(0).permute(1, 2, 0)  # [H, W, 4]
        
        return interpolated
    
    def get_positional_encoding(self, height: int, width: int) -> torch.Tensor:
        """
        获取指定尺寸的位置编码
        
        参数:
            height: patch网格高度
            width: patch网格宽度
            
        返回:
            torch.Tensor: 位置编码张量 [height, width, 4]
        """
        # 生成归一化坐标
        u = torch.linspace(0.5/width, (width-0.5)/width, width, device=self.device)
        v = torch.linspace(0.5/height, (height-0.5)/height, height, device=self.device)
        
        u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
        u_grid = u_grid.unsqueeze(0)  # [1, H, W]
        v_grid = v_grid.unsqueeze(0)  # [1, H, W]
        
        # 插值获取特征
        features = self.interpolate_features(v_grid, u_grid)  # 注意这里交换了u和v
        
        return features.squeeze(0)  # [H, W, 4]


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='WEF-PE查找表预计算系统')
    parser.add_argument('--resolution', type=int, default=512, help='查找表分辨率')
    parser.add_argument('--batch_size', type=int, default=1024, help='批处理大小')
    parser.add_argument('--save_dir', type=str, default='./wef_pe_luts', help='保存目录')
    parser.add_argument('--mode', choices=['pretraining', 'finetuning'], 
                       default='pretraining', help='计算模式')
    parser.add_argument('--alpha_u', type=float, default=1.0, help='水平缩放因子')
    parser.add_argument('--alpha_v', type=float, default=1.0, help='垂直缩放因子')
    parser.add_argument('--test_inference', action='store_true', help='测试快速推理')
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = WEFPELUTGenerator(
        resolution=args.resolution,
        alpha_u=args.alpha_u,
        alpha_v=args.alpha_v,
        computation_mode=args.mode
    )
    
    # 生成查找表
    print(f"\n{'='*50}")
    print("开始预计算WEF-PE查找表")
    print(f"{'='*50}")
    
    lut = generator.generate_lut(batch_size=args.batch_size)
    
    # 保存查找表
    save_dir = f"{args.save_dir}/{args.mode}_res{args.resolution}"
    generator.save_lut(lut, save_dir)
    
    # 测试快速推理
    if args.test_inference:
        print(f"\n{'='*50}")
        print("测试快速推理")
        print(f"{'='*50}")
        
        fast_inference = WEFPEFastInference(
            f"{save_dir}/wef_pe_lut.pt",
            f"{save_dir}/metadata.json"
        )
        
        # 测试不同尺寸的位置编码获取
        test_sizes = [(14, 14), (24, 24), (32, 32)]
        
        for h, w in test_sizes:
            start_time = time.time()
            pe = fast_inference.get_positional_encoding(h, w)
            elapsed = time.time() - start_time
            
            print(f"尺寸 {h}x{w}: 形状={pe.shape}, 用时={elapsed*1000:.2f}ms")
            print(f"  特征范围: [{pe.min():.3f}, {pe.max():.3f}]")
    
    print(f"\n{'='*50}")
    print("预计算完成!")
    print(f"查找表保存在: {save_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()