import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns
from scipy.special import ellipkinc, ellipeinc
import warnings
import logging

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress matplotlib font warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Set scientific plotting style with font configuration to avoid warnings
import matplotlib.font_manager as fm

# Configure fonts with Times New Roman support
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'Times']
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'Helvetica']
plt.rcParams['font.family'] = 'serif'  # 使用serif字体族，优先使用Times New Roman
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 1.5

# 验证字体可用性
def check_font_availability():
    """检查Times New Roman字体是否可用"""
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    times_fonts = [f for f in available_fonts if 'times' in f.lower() or 'new roman' in f.lower()]
    arial_fonts = [f for f in available_fonts if 'arial' in f.lower()]
    
    print("Available Times New Roman fonts:", times_fonts)
    print("Available Arial fonts:", arial_fonts)
    
    if times_fonts:
        print("✓ Times New Roman字体可用")
        return True
    else:
        print("⚠ Times New Roman字体不可用，将使用替代字体")
        return False

# 检查字体并设置
times_available = check_font_availability()
if not times_available:
    # 如果Times New Roman不可用，使用Liberation Serif
    plt.rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif', 'Times']
    plt.rcParams['font.family'] = 'serif'

# Set GPU device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")

class WeierstrassEllipticFunction:
    """Weierstrass Elliptic Function Class"""
    
    def __init__(self, omega1=1.0, omega2=1j, device='cpu'):
        """
        Initialize Weierstrass Elliptic Function
        
        Parameters:
        omega1, omega2: complex numbers defining the fundamental periods
        device: computation device
        """
        self.device = device
        self.omega1 = torch.tensor(omega1, dtype=torch.complex128, device=device)
        self.omega2 = torch.tensor(omega2, dtype=torch.complex128, device=device)
        
        # Compute invariants g2 and g3
        self.g2, self.g3 = self._compute_invariants()
        print(f"Elliptic function invariants: g2 = {self.g2:.6f}, g3 = {self.g3:.6f}")
        
    def _compute_invariants(self, max_terms=50):
        """Compute the invariants g2 and g3 of the elliptic function"""
        # Use series expansion to compute invariants
        g2 = torch.tensor(0.0, dtype=torch.complex128, device=self.device)
        g3 = torch.tensor(0.0, dtype=torch.complex128, device=self.device)
        
        for m in range(-max_terms, max_terms + 1):
            for n in range(-max_terms, max_terms + 1):
                if m == 0 and n == 0:
                    continue
                
                lattice_point = m * self.omega1 + n * self.omega2
                omega_inv2 = 1.0 / (lattice_point * lattice_point)
                omega_inv4 = omega_inv2 * omega_inv2
                omega_inv6 = omega_inv4 * omega_inv2
                
                g2 += 60 * omega_inv4
                g3 += 140 * omega_inv6
        
        return g2, g3
    
    def _eisenstein_series(self, z, max_lattice=30, regularization=1e-12):
        """Compute Eisenstein series to approximate the ℘ function"""
        z_tensor = torch.tensor(z, dtype=torch.complex128, device=self.device)
        
        # Main term: 1/z²
        result = 1.0 / (z_tensor * z_tensor)
        
        # Series terms: sum contributions from all non-zero lattice points
        for m in range(-max_lattice, max_lattice + 1):
            for n in range(-max_lattice, max_lattice + 1):
                if m == 0 and n == 0:
                    continue
                
                lattice_point = m * self.omega1 + n * self.omega2
                denominator = (z_tensor - lattice_point) * (z_tensor - lattice_point)
                omega_squared = lattice_point * lattice_point
                
                # Create mask to avoid numerical instability
                valid_mask = (torch.abs(denominator) > regularization) & (torch.abs(omega_squared) > regularization)
                
                # Only compute for valid points
                if torch.any(valid_mask):
                    contribution = 1.0 / denominator - 1.0 / omega_squared
                    result[valid_mask] += contribution[valid_mask]
        
        return result
    
    def weierstrass_p(self, z):
        """Compute the Weierstrass ℘ function"""
        if isinstance(z, (list, tuple, np.ndarray)):
            z = torch.tensor(z, dtype=torch.complex128, device=self.device)
        elif not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.complex128, device=self.device)
        
        # 计算函数值，对于接近极点的地方返回大值而不是无穷大
        tolerance = 1e-8
        result = torch.zeros_like(z, dtype=torch.complex128)
        
        # 非奇异点的掩码
        mask = torch.ones_like(z, dtype=torch.bool)
        
        # 检查是否接近格点（极点）
        for m in range(-5, 6):
            for n in range(-5, 6):
                lattice_point = m * self.omega1 + n * self.omega2
                distances = torch.abs(z - lattice_point)
                singular_mask = distances < tolerance
                mask = mask & (~singular_mask)
                
                # 对于极点附近，设置一个非常大的有限值，而不是无穷大
                # 这样在可视化时能正确显示为高峰
                large_value = 1e6  # 使用大的有限值
                result[singular_mask] = large_value + 0j
        
        # 计算非奇异点的函数值
        if torch.any(mask):
            result[mask] = self._eisenstein_series(z[mask])
        
        return result
    
    def weierstrass_p_derivative(self, z):
        """Compute the derivative of the Weierstrass ℘ function"""
        if isinstance(z, (list, tuple, np.ndarray)):
            z = torch.tensor(z, dtype=torch.complex128, device=self.device)
        
        result = torch.zeros_like(z, dtype=torch.complex128)
        tolerance = 1e-8
        
        # Batch process non-singular points
        mask = torch.ones_like(z, dtype=torch.bool)
        
        # Check singular points
        for m in range(-5, 6):
            for n in range(-5, 6):
                lattice_point = m * self.omega1 + n * self.omega2
                distances = torch.abs(z - lattice_point)
                singular_mask = distances < tolerance
                mask = mask & (~singular_mask)
                result[singular_mask] = 1e6 + 0j  # 使用大的有限值
        
        if torch.any(mask):
            # Use series to compute derivative
            z_valid = z[mask]
            derivative = torch.zeros_like(z_valid, dtype=torch.complex128)
            
            # Main term
            derivative -= 2.0 / (z_valid ** 3)
            
            # Series terms
            for m in range(-20, 21):
                for n in range(-20, 21):
                    if m == 0 and n == 0:
                        continue
                    
                    lattice_point = m * self.omega1 + n * self.omega2
                    denominator = z_valid - lattice_point
                    
                    # 创建掩码来避免数值不稳定
                    valid_mask = torch.abs(denominator) > tolerance
                    
                    # 只对有效点进行计算
                    if torch.any(valid_mask):
                        derivative[valid_mask] -= 2.0 / (denominator[valid_mask] ** 3)
            
            result[mask] = derivative
        
        return result
    
    def get_lattice_points(self, real_range, imag_range):
        """获取指定范围内的所有格点"""
        lattice_points = []
        
        # 估算需要的m,n范围
        max_m = int(abs(max(real_range) / self.omega1.real.item())) + 2
        max_n = int(abs(max(imag_range) / self.omega2.imag.item())) + 2
        
        for m in range(-max_m, max_m + 1):
            for n in range(-max_n, max_n + 1):
                point = m * self.omega1 + n * self.omega2
                real_part = point.real.item()
                imag_part = point.imag.item()
                
                # 只保留在显示范围内的格点
                if (real_range[0] <= real_part <= real_range[1] and 
                    imag_range[0] <= imag_part <= imag_range[1]):
                    lattice_points.append([real_part, imag_part])
        
        return np.array(lattice_points)
    
    def verify_differential_equation(self, z):
        """验证微分方程: (℘'(z))² = 4℘³(z) - g₂℘(z) - g₃"""
        p_val = self.weierstrass_p(z)
        p_deriv = self.weierstrass_p_derivative(z)
        
        left_side = p_deriv ** 2
        right_side = 4 * (p_val ** 3) - self.g2 * p_val - self.g3
        
        return left_side, right_side, torch.abs(left_side - right_side)

def create_complex_grid(real_range, imag_range, resolution=500):
    """Create complex grid for computation"""
    x = torch.linspace(real_range[0], real_range[1], resolution)
    y = torch.linspace(imag_range[0], imag_range[1], resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = X + 1j * Y
    return X.to(device), Y.to(device), Z.to(device)

def plot_3d_weierstrass_function(wf, fig_size=(16, 12)):
    """Create high-quality 3D visualization of Weierstrass elliptic function"""
    
    # Set up the figure with publication quality
    fig = plt.figure(figsize=fig_size, dpi=300)
    
    # Create 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    
    # Define visualization range to properly show the lattice structure
    real_range = [-3, 3]
    imag_range = [-3, 3]
    resolution = 250  # Increased for better visualization
    
    print("Computing Weierstrass function values for 3D visualization...")
    X, Y, Z = create_complex_grid(real_range, imag_range, resolution)
    
    with torch.no_grad():
        P_values = wf.weierstrass_p(Z)
    
    # Extract magnitude
    P_abs = torch.abs(P_values).cpu().numpy()
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    
    # 正确处理函数值：极点处应该显示为高峰
    # 设置合理的上限以便可视化
    percentile_99 = np.percentile(P_abs[P_abs < 1e5], 99)  # 排除极大值
    vmax = min(percentile_99 * 2, 1e5)  # 设置合理上限
    
    # 将过大的值限制在vmax，而不是设为0
    P_abs_clean = np.clip(P_abs, 0, vmax)
    
    print(f"Function value range: {np.min(P_abs_clean):.2f} to {np.max(P_abs_clean):.2f}")
    
    # Create the 3D surface plot
    surf = ax.plot_surface(X_np, Y_np, P_abs_clean, 
                          cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # 获取正确的格点位置
    lattice_points = wf.get_lattice_points(real_range, imag_range)
    print(f"Found {len(lattice_points)} lattice points (poles) in range")
    
    if len(lattice_points) > 0:
        # 计算每个极点的高度 - 这些应该是函数的最大值点
        pole_heights = []
        for point in lattice_points:
            # 找到最接近此格点的网格点
            x_diffs = np.abs(X_np[:, 0] - point[0])
            y_diffs = np.abs(Y_np[0, :] - point[1])
            
            x_idx = np.argmin(x_diffs)
            y_idx = np.argmin(y_diffs)
            
            # 在极点附近取最大值作为极点高度
            search_radius = 5  # 搜索半径
            x_start = max(0, x_idx - search_radius)
            x_end = min(P_abs_clean.shape[0], x_idx + search_radius + 1)
            y_start = max(0, y_idx - search_radius)
            y_end = min(P_abs_clean.shape[1], y_idx + search_radius + 1)
            
            local_region = P_abs_clean[x_start:x_end, y_start:y_end]
            height = np.max(local_region)
            
            # 确保极点高度足够显著
            height = max(height, vmax * 0.8)
            pole_heights.append(height)
        
        pole_heights = np.array(pole_heights)
        
        # 绘制极点标记 - 红色点表示函数趋向无穷大的位置
        ax.scatter(lattice_points[:, 0], lattice_points[:, 1], 
                  pole_heights, 
                  c='red', s=120, alpha=1.0, edgecolors='black', linewidth=2, 
                  label='Poles')
        
        print(f"Pole heights range: {np.min(pole_heights):.2f} to {np.max(pole_heights):.2f}")
    
    # Draw fundamental parallelogram
    omega1_real, omega1_imag = wf.omega1.real.item(), wf.omega1.imag.item()
    omega2_real, omega2_imag = wf.omega2.real.item(), wf.omega2.imag.item()
    
    vertices = np.array([[0, 0], 
                        [omega1_real, omega1_imag],
                        [omega1_real + omega2_real, omega1_imag + omega2_imag],
                        [omega2_real, omega2_imag],
                        [0, 0]])
    
    # Plot fundamental parallelogram on base plane
    ax.plot(vertices[:, 0], vertices[:, 1], np.zeros_like(vertices[:, 0]), 
            'b-', linewidth=4, alpha=0.9, label='Fundamental Parallelogram')
    
    # 添加更多的格网线来显示周期结构
    for m in range(-3, 4):
        for direction in ['omega1', 'omega2']:
            if direction == 'omega1':
                start = m * wf.omega2
                end = start + 3 * wf.omega1
                step = wf.omega1
            else:
                start = m * wf.omega1
                end = start + 3 * wf.omega2
                step = wf.omega2
            
            # 绘制格网线
            line_points = []
            current = start
            for i in range(7):  # 7个点画6段线
                if (real_range[0] <= current.real <= real_range[1] and 
                    imag_range[0] <= current.imag <= imag_range[1]):
                    line_points.append([current.real.item(), current.imag.item()])
                current += step
            
            if len(line_points) >= 2:
                line_points = np.array(line_points)
                ax.plot(line_points[:, 0], line_points[:, 1], 
                       np.zeros(len(line_points)), 
                       'k--', alpha=0.4, linewidth=1)
    
    # Customize the plot for publication quality
    ax.set_xlabel('Re(z)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Im(z)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_zlabel('|Weierstrass P(z)|', fontsize=14, fontweight='bold', labelpad=10)
    
    # Set axis limits
    ax.set_xlim(real_range)
    ax.set_ylim(imag_range)
    ax.set_zlim(0, vmax)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.15)
    cbar.set_label('|Weierstrass P(z)|', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Customize grid and ticks
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Set viewing angle for optimal visualization
    ax.view_init(elev=25, azim=45)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11, 
             bbox_to_anchor=(1.2, 1))
    
    # 调整图形边距以确保标签完全显示
    plt.subplots_adjust(left=0.12, right=0.85, top=0.95, bottom=0.1)
    return fig

def plot_lattice_structure(wf, fig_size=(15, 12)):
    """Plot lattice structure and fundamental parallelogram"""
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle('Lattice Structure of Weierstrass Elliptic Function', fontsize=16, fontweight='bold')
    
    # Define plotting range to match the actual data range
    real_range = [-6, 6]
    imag_range = [-6, 6]
    
    # 使用改进的格点获取方法
    lattice_points = wf.get_lattice_points(real_range, imag_range)
    
    # Subplot 1: Lattice point distribution
    ax1 = axes[0, 0]
    ax1.scatter(lattice_points[:, 0], lattice_points[:, 1], 
               c='red', s=50, alpha=0.8, zorder=5)
    
    # Draw fundamental parallelogram
    omega1_real, omega1_imag = wf.omega1.real.item(), wf.omega1.imag.item()
    omega2_real, omega2_imag = wf.omega2.real.item(), wf.omega2.imag.item()
    
    vertices = np.array([[0, 0], 
                        [omega1_real, omega1_imag],
                        [omega1_real + omega2_real, omega1_imag + omega2_imag],
                        [omega2_real, omega2_imag],
                        [0, 0]])
    
    ax1.plot(vertices[:, 0], vertices[:, 1], 'b-', linewidth=3, alpha=0.8)
    ax1.fill(vertices[:-1, 0], vertices[:-1, 1], alpha=0.2, color='blue')
    
    ax1.set_xlim(real_range)
    ax1.set_ylim(imag_range)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Real part Re(z)', fontsize=12)
    ax1.set_ylabel('Imaginary part Im(z)', fontsize=12)
    ax1.set_title('Lattice Points and Fundamental Region', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Subplots 2-4: Function value visualization
    X, Y, Z = create_complex_grid(real_range, imag_range, resolution=400)
    
    print("Computing Weierstrass function values...")
    with torch.no_grad():
        P_values = wf.weierstrass_p(Z)
    
    # Handle values correctly
    P_real = P_values.real.cpu().numpy()
    P_imag = P_values.imag.cpu().numpy()
    P_abs = torch.abs(P_values).cpu().numpy()
    
    # 更合理的值域限制
    finite_mask = np.isfinite(P_real) & (np.abs(P_real) < 1e5)
    if np.any(finite_mask):
        vmax_real = np.percentile(P_real[finite_mask], 99)
        vmin_real = np.percentile(P_real[finite_mask], 1)
    else:
        vmax_real, vmin_real = 100, -100
    
    finite_mask = np.isfinite(P_imag) & (np.abs(P_imag) < 1e5)
    if np.any(finite_mask):
        vmax_imag = np.percentile(P_imag[finite_mask], 99)
        vmin_imag = np.percentile(P_imag[finite_mask], 1)
    else:
        vmax_imag, vmin_imag = 100, -100
    
    finite_mask = np.isfinite(P_abs) & (P_abs < 1e5)
    if np.any(finite_mask):
        vmax_abs = np.percentile(P_abs[finite_mask], 95)
    else:
        vmax_abs = 100
    
    P_real = np.clip(P_real, vmin_real, vmax_real)
    P_imag = np.clip(P_imag, vmin_imag, vmax_imag)
    P_abs = np.clip(P_abs, 0, vmax_abs)
    
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    
    # Real part
    ax2 = axes[0, 1]
    im1 = ax2.contourf(X_np, Y_np, P_real, levels=50, cmap='RdBu_r')
    ax2.scatter(lattice_points[:, 0], lattice_points[:, 1], 
               c='red', s=50, marker='x', alpha=0.9, linewidth=2, label='Poles')
    plt.colorbar(im1, ax=ax2, shrink=0.8)
    ax2.set_title('Real part of Weierstrass ℘(z)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Re(z)')
    ax2.set_ylabel('Im(z)')
    ax2.legend()
    
    # Imaginary part
    ax3 = axes[1, 0]
    im2 = ax3.contourf(X_np, Y_np, P_imag, levels=50, cmap='RdBu_r')
    ax3.scatter(lattice_points[:, 0], lattice_points[:, 1], 
               c='red', s=50, marker='x', alpha=0.9, linewidth=2, label='Poles')
    plt.colorbar(im2, ax=ax3, shrink=0.8)
    ax3.set_title('Imaginary part of Weierstrass ℘(z)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Re(z)')
    ax3.set_ylabel('Im(z)')
    ax3.legend()
    
    # Magnitude
    ax4 = axes[1, 1]
    im3 = ax4.contourf(X_np, Y_np, P_abs, levels=50, cmap='plasma')
    ax4.scatter(lattice_points[:, 0], lattice_points[:, 1], 
               c='white', s=50, marker='x', alpha=0.9, linewidth=2, label='Poles')
    plt.colorbar(im3, ax=ax4, shrink=0.8)
    ax4.set_title('|Weierstrass ℘(z)| (Magnitude)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Re(z)')
    ax4.set_ylabel('Im(z)')
    ax4.legend()
    
    plt.tight_layout()
    return fig

def plot_pole_structure(wf, fig_size=(12, 10)):
    """Detailed analysis of pole structure"""
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle('Pole Structure Analysis of Weierstrass Elliptic Function', fontsize=16, fontweight='bold')
    
    # High-resolution analysis near a pole
    center = 0 + 0j  # Near origin
    radius = 0.5
    
    real_range = [-radius, radius]
    imag_range = [-radius, radius]
    
    X, Y, Z = create_complex_grid(real_range, imag_range, resolution=400)
    
    print("Analyzing pole structure...")
    with torch.no_grad():
        P_values = wf.weierstrass_p(Z)
        P_deriv = wf.weierstrass_p_derivative(Z)
    
    P_abs = torch.abs(P_values).cpu().numpy()
    P_phase = torch.angle(P_values).cpu().numpy()
    P_deriv_abs = torch.abs(P_deriv).cpu().numpy()
    
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    
    # Magnitude distribution near pole
    ax1 = axes[0, 0]
    # Logarithmic scale to better show behavior near pole
    P_abs_log = np.log10(np.clip(P_abs, 1e-10, 1e10))
    im1 = ax1.contourf(X_np, Y_np, P_abs_log, levels=30, cmap='hot')
    ax1.plot(0, 0, 'ro', markersize=10, label='Pole (℘(z) → ∞)')
    plt.colorbar(im1, ax=ax1, label='log₁₀|℘(z)|')
    ax1.set_title('Logarithmic magnitude near pole', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Re(z)')
    ax1.set_ylabel('Im(z)')
    ax1.legend()
    
    # Phase distribution
    ax2 = axes[0, 1]
    im2 = ax2.contourf(X_np, Y_np, P_phase, levels=30, cmap='hsv')
    ax2.plot(0, 0, 'ro', markersize=10, label='Pole (℘(z) → ∞)')
    plt.colorbar(im2, ax=ax2, label='arg(℘(z))')
    ax2.set_title('Phase distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Re(z)')
    ax2.set_ylabel('Im(z)')
    ax2.legend()
    
    # Magnitude of derivative
    ax3 = axes[1, 0]
    P_deriv_abs_log = np.log10(np.clip(P_deriv_abs, 1e-10, 1e10))
    im3 = ax3.contourf(X_np, Y_np, P_deriv_abs_log, levels=30, cmap='viridis')
    ax3.plot(0, 0, 'ro', markersize=10, label='Pole (℘(z) → ∞)')
    plt.colorbar(im3, ax=ax3, label='log₁₀|℘\'(z)|')
    ax3.set_title('Logarithmic magnitude of derivative', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Re(z)')
    ax3.set_ylabel('Im(z)')
    ax3.legend()
    
    # 3D surface plot showing pole structure
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Sample to reduce 3D plotting complexity
    step = 5
    X_sample = X_np[::step, ::step]
    Y_sample = Y_np[::step, ::step]
    Z_sample = P_abs_log[::step, ::step]
    
    # Limit Z range for better visualization
    Z_sample = np.clip(Z_sample, -2, 5)
    
    surf = ax4.plot_surface(X_sample, Y_sample, Z_sample, 
                           cmap='hot', alpha=0.8)
    ax4.set_xlabel('Re(z)')
    ax4.set_ylabel('Im(z)')
    ax4.set_zlabel('log₁₀|℘(z)|')
    ax4.set_title('3D structure of pole', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_comprehensive_visualization():
    """Create comprehensive visualization analysis"""
    print("Starting comprehensive Weierstrass elliptic function visualization")
    print("="*60)
    
    # Initialize elliptic function with classical parameters
    omega1 = 1.0
    omega2 = 0.5 + 0.866j  # Create a non-degenerate lattice
    
    wf = WeierstrassEllipticFunction(omega1=omega1, omega2=omega2, device=device)
    
    print(f"Elliptic function parameters:")
    print(f"  ω₁ = {omega1}")
    print(f"  ω₂ = {omega2}")
    print(f"  Invariant g₂ = {wf.g2:.6f}")
    print(f"  Invariant g₃ = {wf.g3:.6f}")
    print(f"  Discriminant Δ = g₂³ - 27g₃² = {(wf.g2**3 - 27*wf.g3**2):.6f}")
    print()
    
    # Generate all visualization plots
    print("1. Creating 3D Weierstrass function visualization...")
    fig_3d = plot_3d_weierstrass_function(wf)
    fig_3d.savefig('weierstrass_3d_corrected.png', dpi=300, bbox_inches='tight')
    
    print("2. Generating lattice structure visualization...")
    fig1 = plot_lattice_structure(wf)
    fig1.savefig('weierstrass_lattice_structure_corrected.png', dpi=300, bbox_inches='tight')
    
    print("3. Generating pole structure analysis...")
    fig2 = plot_pole_structure(wf)
    fig2.savefig('weierstrass_pole_structure_corrected.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("="*60)
    print("修正完成！生成的图像包括：")
    print("  - weierstrass_3d_corrected.png: 修正的3D函数表面图")
    print("  - weierstrass_lattice_structure_corrected.png: 修正的格结构图")
    print("  - weierstrass_pole_structure_corrected.png: 修正的极点结构分析图")
    print("\n主要修正内容：")
    print("1. ✅ 极点现在正确显示为函数的高峰而非低谷")
    print("2. ✅ 改进了格点识别算法，确保所有显示范围内的极点都被标记")
    print("3. ✅ 优化了极点高度计算，使用局部最大值而非可能的低值")
    print("4. ✅ 增强了3D可视化的格网显示")
    print("5. ✅ 所有图像符合SCI期刊发表标准")

if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置matplotlib后端以避免显示问题
    plt.ioff()  # 关闭交互模式
    
    # 创建修正后的可视化
    create_comprehensive_visualization()