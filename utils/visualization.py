"""
图像融合可视化工具
包含图像对比、中间结果、训练曲线等可视化功能

作者: Christopher32527
邮箱: 2546507517@qq.com
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os


# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def tensor_to_numpy(img):
    """
    将PyTorch张量转换为numpy数组用于可视化
    
    Args:
        img: 图像张量, shape [B, C, H, W] 或 [C, H, W]
    
    Returns:
        img_np: numpy数组, shape [H, W] 或 [H, W, C]
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    # 移除batch维度
    if img.ndim == 4:
        img = img[0]
    
    # 移除通道维度（如果是单通道）
    if img.shape[0] == 1:
        img = img[0]
    # 转换通道顺序（如果是多通道）
    elif img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    # 归一化到[0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    return img


def visualize_fusion_comparison(img1, img2, fused, save_path=None, titles=None):
    """
    可视化融合对比图
    
    Args:
        img1: 源图像1 (IR或CT)
        img2: 源图像2 (VIS或MRI)
        fused: 融合图像
        save_path: 保存路径 (可选)
        titles: 标题列表 (可选)
    """
    if titles is None:
        titles = ['Source Image 1 (IR/CT)', 'Source Image 2 (VIS/MRI)', 'Fused Image']
    
    # 转换为numpy
    img1_np = tensor_to_numpy(img1)
    img2_np = tensor_to_numpy(img2)
    fused_np = tensor_to_numpy(fused)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示图像
    axes[0].imshow(img1_np, cmap='gray')
    axes[0].set_title(titles[0], fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(img2_np, cmap='gray')
    axes[1].set_title(titles[1], fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(fused_np, cmap='gray')
    axes[2].set_title(titles[2], fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 对比图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


# 别名，方便训练脚本导入
save_fusion_comparison = visualize_fusion_comparison


def visualize_intermediate_results(img1, img2, fused_low, fused_high, fused_final, save_path=None):
    """
    可视化中间结果（FAM阶段的低频图和高频图）
    
    Args:
        img1: 源图像1
        img2: 源图像2
        fused_low: FAM融合的低频图
        fused_high: FAM融合的高频图
        fused_final: 最终融合图像
        save_path: 保存路径 (可选)
    """
    # 转换为numpy
    img1_np = tensor_to_numpy(img1)
    img2_np = tensor_to_numpy(img2)
    fused_low_np = tensor_to_numpy(fused_low)
    fused_high_np = tensor_to_numpy(fused_high)
    fused_final_np = tensor_to_numpy(fused_final)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：源图像和最终融合图像
    axes[0, 0].imshow(img1_np, cmap='gray')
    axes[0, 0].set_title('Source Image 1', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_np, cmap='gray')
    axes[0, 1].set_title('Source Image 2', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(fused_final_np, cmap='gray')
    axes[0, 2].set_title('Final Fused Image', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 第二行：中间结果
    axes[1, 0].imshow(fused_low_np, cmap='gray')
    axes[1, 0].set_title('FAM Low Frequency', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(fused_high_np, cmap='gray')
    axes[1, 1].set_title('FAM High Frequency', fontsize=12)
    axes[1, 1].axis('off')
    
    # 第三个位置显示低频+高频的组合
    combined = (fused_low_np + fused_high_np) / 2
    axes[1, 2].imshow(combined, cmap='gray')
    axes[1, 2].set_title('Low + High Combined', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 中间结果图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(train_losses, val_losses=None, save_path=None):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表 (可选)
        save_path: 保存路径 (可选)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 训练曲线已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    绘制指标对比图
    
    Args:
        metrics_dict: 指标字典，格式 {'method1': {'EN': 7.5, 'SF': 12.3, ...}, ...}
        save_path: 保存路径 (可选)
    """
    methods = list(metrics_dict.keys())
    metric_names = list(metrics_dict[methods[0]].keys())
    
    x = np.arange(len(metric_names))
    width = 0.8 / len(methods)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, method in enumerate(methods):
        values = [metrics_dict[method][metric] for metric in metric_names]
        ax.bar(x + i * width, values, width, label=method)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Fusion Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 指标对比图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_loss_components(loss_components, save_path=None):
    """
    绘制损失分量图
    
    Args:
        loss_components: 损失分量列表，每个元素是字典 {'total': x, 'ssim': y, ...}
        save_path: 保存路径 (可选)
    """
    epochs = range(1, len(loss_components) + 1)
    
    # 提取各个分量
    total_losses = [comp['total'] for comp in loss_components]
    ssim_losses = [comp['ssim'] for comp in loss_components]
    gradient_losses = [comp['gradient'] for comp in loss_components]
    intensity_losses = [comp['intensity'] for comp in loss_components]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 总损失
    axes[0, 0].plot(epochs, total_losses, 'k-', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # SSIM损失
    axes[0, 1].plot(epochs, ssim_losses, 'b-', linewidth=2)
    axes[0, 1].set_title('SSIM Loss', fontsize=12)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 梯度损失
    axes[1, 0].plot(epochs, gradient_losses, 'r-', linewidth=2)
    axes[1, 0].set_title('Gradient Loss', fontsize=12)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 强度损失
    axes[1, 1].plot(epochs, intensity_losses, 'g-', linewidth=2)
    axes[1, 1].set_title('Intensity Loss', fontsize=12)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 损失分量图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("Visualization Tools Test")
    print("=" * 70)
    
    # 创建测试数据
    img_size = 256
    
    # 创建有意义的测试图像
    img1 = torch.linspace(0, 1, img_size).view(1, img_size).repeat(img_size, 1)
    img1 = img1.unsqueeze(0).unsqueeze(0)
    
    img2 = torch.linspace(0, 1, img_size).view(img_size, 1).repeat(1, img_size)
    img2 = img2.unsqueeze(0).unsqueeze(0)
    
    fused = (img1 + img2) / 2
    
    print(f"\n测试数据:")
    print(f"  - 图像1: 水平渐变")
    print(f"  - 图像2: 垂直渐变")
    print(f"  - 融合图像: 简单平均")
    
    # 创建输出目录
    output_dir = "test_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试融合对比图
    print(f"\n测试融合对比图...")
    visualize_fusion_comparison(
        img1, img2, fused,
        save_path=os.path.join(output_dir, "fusion_comparison.png")
    )
    
    # 测试中间结果图
    print(f"\n测试中间结果图...")
    fused_low = img1 * 0.7 + img2 * 0.3
    fused_high = img1 * 0.3 + img2 * 0.7
    visualize_intermediate_results(
        img1, img2, fused_low, fused_high, fused,
        save_path=os.path.join(output_dir, "intermediate_results.png")
    )
    
    # 测试训练曲线
    print(f"\n测试训练曲线...")
    train_losses = [10.0 - i * 0.5 + np.random.rand() * 0.5 for i in range(20)]
    val_losses = [10.5 - i * 0.45 + np.random.rand() * 0.6 for i in range(20)]
    plot_training_curves(
        train_losses, val_losses,
        save_path=os.path.join(output_dir, "training_curves.png")
    )
    
    # 测试指标对比图
    print(f"\n测试指标对比图...")
    metrics_dict = {
        'Method 1': {'EN': 7.2, 'SF': 12.5, 'SSIM': 0.85, 'MI': 3.2},
        'Method 2': {'EN': 7.5, 'SF': 13.2, 'SSIM': 0.88, 'MI': 3.5},
        'Ours': {'EN': 7.8, 'SF': 14.1, 'SSIM': 0.92, 'MI': 3.8}
    }
    plot_metrics_comparison(
        metrics_dict,
        save_path=os.path.join(output_dir, "metrics_comparison.png")
    )
    
    # 测试损失分量图
    print(f"\n测试损失分量图...")
    loss_components = [
        {
            'total': 10.0 - i * 0.4 + np.random.rand() * 0.3,
            'ssim': 1.0 - i * 0.04 + np.random.rand() * 0.05,
            'gradient': 5.0 - i * 0.2 + np.random.rand() * 0.15,
            'intensity': 1.5 - i * 0.06 + np.random.rand() * 0.08
        }
        for i in range(20)
    ]
    plot_loss_components(
        loss_components,
        save_path=os.path.join(output_dir, "loss_components.png")
    )
    
    print("\n" + "=" * 70)
    print("✓ 可视化工具测试通过!")
    print("=" * 70)
    print(f"\n所有测试图像已保存到: {output_dir}/")
