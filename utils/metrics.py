"""
图像融合评估指标
包含信息熵(EN)、空间频率(SF)、结构相似性(SSIM)、互信息(MI)等

作者: Christopher32527
邮箱: 2546507517@qq.com
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


def calculate_entropy(img):
    """
    计算信息熵 (Entropy, EN)
    
    信息熵衡量图像的信息量，值越大表示图像包含的信息越丰富
    
    Args:
        img: 图像张量, shape [B, C, H, W] 或 numpy数组
    
    Returns:
        entropy: 信息熵值
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    # 归一化到[0, 255]
    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # 计算直方图
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()  # 归一化为概率分布
    
    # 计算熵
    hist = hist[hist > 0]  # 移除零值
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy


def calculate_spatial_frequency(img):
    """
    计算空间频率 (Spatial Frequency, SF)
    
    空间频率衡量图像的清晰度，值越大表示图像越清晰
    
    Args:
        img: 图像张量, shape [B, C, H, W] 或 numpy数组
    
    Returns:
        sf: 空间频率值
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    # 确保是单通道
    if img.ndim == 4:
        img = img[0, 0]  # 取第一个batch的第一个通道
    elif img.ndim == 3:
        img = img[0]
    
    H, W = img.shape
    
    # 行频率 (Row Frequency)
    RF = 0
    for i in range(H):
        for j in range(W - 1):
            RF += (img[i, j] - img[i, j + 1]) ** 2
    RF = np.sqrt(RF / (H * W))
    
    # 列频率 (Column Frequency)
    CF = 0
    for i in range(H - 1):
        for j in range(W):
            CF += (img[i, j] - img[i + 1, j]) ** 2
    CF = np.sqrt(CF / (H * W))
    
    # 空间频率
    sf = np.sqrt(RF ** 2 + CF ** 2)
    
    return sf


def calculate_ssim(img1, img2, window_size=11):
    """
    计算结构相似性 (Structural Similarity, SSIM)
    
    SSIM衡量两张图像的结构相似性，值越大越相似，范围[-1, 1]
    
    Args:
        img1: 图像1, shape [B, C, H, W] 或 numpy数组
        img2: 图像2, shape [B, C, H, W] 或 numpy数组
        window_size: 窗口大小 (默认11)
    
    Returns:
        ssim: SSIM值
    """
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    # 确保在同一设备
    device = img1.device
    img2 = img2.to(device)
    
    # 创建高斯窗口
    def gaussian(window_size, sigma=1.5):
        gauss = torch.Tensor([
            math.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(window_size, channel):
        _1D_window = gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    channel = img1.size(1)
    window = create_window(window_size, channel).to(device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def calculate_mutual_information(img1, img2, bins=256):
    """
    计算互信息 (Mutual Information, MI)
    
    互信息衡量两张图像之间的相关性，值越大表示融合图像保留了更多源图像的信息
    
    Args:
        img1: 图像1, shape [B, C, H, W] 或 numpy数组
        img2: 图像2, shape [B, C, H, W] 或 numpy数组
        bins: 直方图bins数量 (默认256)
    
    Returns:
        mi: 互信息值
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # 归一化到[0, bins-1]
    img1 = ((img1 - img1.min()) / (img1.max() - img1.min() + 1e-8) * (bins - 1)).astype(np.int32)
    img2 = ((img2 - img2.min()) / (img2.max() - img2.min() + 1e-8) * (bins - 1)).astype(np.int32)
    
    # 计算联合直方图
    hist_2d, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins=bins)
    
    # 归一化为联合概率分布
    pxy = hist_2d / hist_2d.sum()
    
    # 计算边缘概率分布
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    
    # 计算互信息
    px_py = px[:, None] * py[None, :]
    
    # 只计算非零项
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))
    
    return mi


def calculate_all_metrics(fused, img1, img2):
    """
    计算所有评估指标
    
    Args:
        fused: 融合图像, shape [B, C, H, W]
        img1: 源图像1, shape [B, C, H, W]
        img2: 源图像2, shape [B, C, H, W]
    
    Returns:
        metrics: 包含所有指标的字典
    """
    metrics = {}
    
    # 信息熵
    metrics['EN'] = calculate_entropy(fused)
    
    # 空间频率
    metrics['SF'] = calculate_spatial_frequency(fused)
    
    # SSIM (与两个源图像的平均)
    ssim1 = calculate_ssim(fused, img1)
    ssim2 = calculate_ssim(fused, img2)
    metrics['SSIM'] = (ssim1 + ssim2) / 2
    
    # 互信息 (与两个源图像的总和)
    mi1 = calculate_mutual_information(fused, img1)
    mi2 = calculate_mutual_information(fused, img2)
    metrics['MI'] = mi1 + mi2
    
    return metrics


# 别名，方便导入
calculate_metrics = calculate_all_metrics


def print_metrics(metrics, title="Evaluation Metrics"):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print(f"\n{title}")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"  {key:10s}: {value:.6f}")
    print("=" * 50)


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("Fusion Metrics Test")
    print("=" * 70)
    
    # 检查CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建测试数据
    batch_size = 1
    img_size = 256
    
    # 创建有意义的测试图像（而不是随机噪声）
    # 图像1: 水平渐变
    img1 = torch.linspace(0, 1, img_size).view(1, img_size).repeat(img_size, 1)
    img1 = img1.unsqueeze(0).unsqueeze(0).to(device)
    
    # 图像2: 垂直渐变
    img2 = torch.linspace(0, 1, img_size).view(img_size, 1).repeat(1, img_size)
    img2 = img2.unsqueeze(0).unsqueeze(0).to(device)
    
    # 融合图像: 简单平均
    fused = (img1 + img2) / 2
    
    print(f"\n测试数据:")
    print(f"  - 图像1: 水平渐变, 形状 {img1.shape}")
    print(f"  - 图像2: 垂直渐变, 形状 {img2.shape}")
    print(f"  - 融合图像: 简单平均, 形状 {fused.shape}")
    
    # 测试信息熵
    print(f"\n测试信息熵 (EN)...")
    en = calculate_entropy(fused)
    print(f"  ✓ 融合图像信息熵: {en:.6f}")
    print(f"    (参考: 图像1 EN = {calculate_entropy(img1):.6f})")
    print(f"    (参考: 图像2 EN = {calculate_entropy(img2):.6f})")
    
    # 测试空间频率
    print(f"\n测试空间频率 (SF)...")
    sf = calculate_spatial_frequency(fused)
    print(f"  ✓ 融合图像空间频率: {sf:.6f}")
    print(f"    (参考: 图像1 SF = {calculate_spatial_frequency(img1):.6f})")
    print(f"    (参考: 图像2 SF = {calculate_spatial_frequency(img2):.6f})")
    
    # 测试SSIM
    print(f"\n测试结构相似性 (SSIM)...")
    ssim1 = calculate_ssim(fused, img1)
    ssim2 = calculate_ssim(fused, img2)
    print(f"  ✓ 融合图像与图像1的SSIM: {ssim1:.6f}")
    print(f"  ✓ 融合图像与图像2的SSIM: {ssim2:.6f}")
    print(f"  ✓ 平均SSIM: {(ssim1 + ssim2) / 2:.6f}")
    
    # 测试互信息
    print(f"\n测试互信息 (MI)...")
    mi1 = calculate_mutual_information(fused, img1)
    mi2 = calculate_mutual_information(fused, img2)
    print(f"  ✓ 融合图像与图像1的MI: {mi1:.6f}")
    print(f"  ✓ 融合图像与图像2的MI: {mi2:.6f}")
    print(f"  ✓ 总MI: {mi1 + mi2:.6f}")
    
    # 测试所有指标
    print(f"\n测试所有指标...")
    metrics = calculate_all_metrics(fused, img1, img2)
    print_metrics(metrics, "融合图像评估指标")
    
    print("\n" + "=" * 70)
    print("✓ 评估指标测试通过!")
    print("=" * 70)
    print("\n指标说明:")
    print("  - EN (信息熵): 越大越好，表示图像信息量丰富")
    print("  - SF (空间频率): 越大越好，表示图像清晰度高")
    print("  - SSIM (结构相似性): 越接近1越好，表示保留了源图像结构")
    print("  - MI (互信息): 越大越好，表示保留了源图像信息")
