"""
图像融合损失函数
包含SSIM损失、梯度损失、强度损失和组合损失

作者: Christopher32527
邮箱: 2546507517@qq.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SSIMLoss(nn.Module):
    """
    结构相似性损失 (Structural Similarity Loss)
    
    SSIM衡量两张图像的结构相似性，值越大越相似
    SSIM Loss = 1 - SSIM，用于优化
    
    Args:
        window_size: 窗口大小 (默认11)
        size_average: 是否对batch求平均 (默认True)
    """
    
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        """创建高斯窗口"""
        gauss = torch.Tensor([
            math.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        """创建2D高斯窗口"""
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """计算SSIM"""
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
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        """
        前向传播
        
        Args:
            img1: 图像1, shape [B, C, H, W]
            img2: 图像2, shape [B, C, H, W]
        
        Returns:
            loss: SSIM损失值
        """
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        ssim_value = self.ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return 1 - ssim_value


class GradientLoss(nn.Module):
    """
    梯度损失 (Gradient Loss)
    
    保持融合图像的梯度信息，使融合图像包含源图像的边缘细节
    
    Args:
        loss_type: 损失类型，'l1' 或 'l2' (默认'l1')
    """
    
    def __init__(self, loss_type='l1'):
        super(GradientLoss, self).__init__()
        self.loss_type = loss_type
        
        # Sobel算子
        self.sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3)
        self.sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3)
    
    def gradient(self, img):
        """计算图像梯度"""
        if img.is_cuda:
            self.sobel_x = self.sobel_x.cuda(img.get_device())
            self.sobel_y = self.sobel_y.cuda(img.get_device())
        
        self.sobel_x = self.sobel_x.type_as(img)
        self.sobel_y = self.sobel_y.type_as(img)
        
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    
    def forward(self, fused, img1, img2):
        """
        前向传播
        
        Args:
            fused: 融合图像, shape [B, C, H, W]
            img1: 源图像1, shape [B, C, H, W]
            img2: 源图像2, shape [B, C, H, W]
        
        Returns:
            loss: 梯度损失值
        """
        grad_fused = self.gradient(fused)
        grad_img1 = self.gradient(img1)
        grad_img2 = self.gradient(img2)
        
        # 取两个源图像梯度的最大值作为目标
        grad_target = torch.max(grad_img1, grad_img2)
        
        if self.loss_type == 'l1':
            loss = F.l1_loss(grad_fused, grad_target)
        else:
            loss = F.mse_loss(grad_fused, grad_target)
        
        return loss


class IntensityLoss(nn.Module):
    """
    强度损失 (Intensity Loss)
    
    保持融合图像的强度信息，使融合图像的亮度接近源图像
    
    Args:
        loss_type: 损失类型，'l1' 或 'l2' (默认'l1')
    """
    
    def __init__(self, loss_type='l1'):
        super(IntensityLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, fused, img1, img2):
        """
        前向传播
        
        Args:
            fused: 融合图像, shape [B, C, H, W]
            img1: 源图像1, shape [B, C, H, W]
            img2: 源图像2, shape [B, C, H, W]
        
        Returns:
            loss: 强度损失值
        """
        # 取两个源图像强度的平均值作为目标
        intensity_target = (img1 + img2) / 2
        
        if self.loss_type == 'l1':
            loss = F.l1_loss(fused, intensity_target)
        else:
            loss = F.mse_loss(fused, intensity_target)
        
        return loss


class FusionLoss(nn.Module):
    """
    图像融合组合损失
    
    组合SSIM损失、梯度损失和强度损失
    
    Args:
        ssim_weight: SSIM损失权重 (默认1.0)
        gradient_weight: 梯度损失权重 (默认10.0)
        intensity_weight: 强度损失权重 (默认1.0)
        window_size: SSIM窗口大小 (默认11)
    """
    
    def __init__(self, 
                 ssim_weight=1.0,
                 gradient_weight=10.0,
                 intensity_weight=1.0,
                 window_size=11):
        super(FusionLoss, self).__init__()
        
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        self.intensity_weight = intensity_weight
        
        self.ssim_loss = SSIMLoss(window_size=window_size)
        self.gradient_loss = GradientLoss(loss_type='l1')
        self.intensity_loss = IntensityLoss(loss_type='l1')
    
    def forward(self, fused, img1, img2, return_components=False):
        """
        前向传播
        
        Args:
            fused: 融合图像, shape [B, C, H, W]
            img1: 源图像1 (IR或CT), shape [B, C, H, W]
            img2: 源图像2 (VIS或MRI), shape [B, C, H, W]
            return_components: 是否返回各个损失分量 (默认False)
        
        Returns:
            loss: 总损失值
            components (可选): 各个损失分量的字典
        """
        # 计算各个损失
        ssim_loss_val = self.ssim_loss(fused, img1) + self.ssim_loss(fused, img2)
        gradient_loss_val = self.gradient_loss(fused, img1, img2)
        intensity_loss_val = self.intensity_loss(fused, img1, img2)
        
        # 加权求和
        total_loss = (self.ssim_weight * ssim_loss_val +
                     self.gradient_weight * gradient_loss_val +
                     self.intensity_weight * intensity_loss_val)
        
        if return_components:
            components = {
                'total': total_loss.item(),
                'ssim': ssim_loss_val.item(),
                'gradient': gradient_loss_val.item(),
                'intensity': intensity_loss_val.item()
            }
            return total_loss, components
        
        return total_loss


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("Fusion Loss Test")
    print("=" * 70)
    
    # 检查CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建测试数据
    batch_size = 2
    img_size = 256
    img1 = torch.randn(batch_size, 1, img_size, img_size).to(device)
    img2 = torch.randn(batch_size, 1, img_size, img_size).to(device)
    fused = (img1 + img2) / 2  # 简单平均作为融合结果
    
    print(f"\n测试数据:")
    print(f"  - 图像1形状: {img1.shape}")
    print(f"  - 图像2形状: {img2.shape}")
    print(f"  - 融合图像形状: {fused.shape}")
    
    # 测试SSIM损失
    print(f"\n测试SSIM损失...")
    ssim_loss = SSIMLoss().to(device)
    loss_val = ssim_loss(fused, img1)
    print(f"  ✓ SSIM损失: {loss_val.item():.6f}")
    
    # 测试梯度损失
    print(f"\n测试梯度损失...")
    gradient_loss = GradientLoss().to(device)
    loss_val = gradient_loss(fused, img1, img2)
    print(f"  ✓ 梯度损失: {loss_val.item():.6f}")
    
    # 测试强度损失
    print(f"\n测试强度损失...")
    intensity_loss = IntensityLoss().to(device)
    loss_val = intensity_loss(fused, img1, img2)
    print(f"  ✓ 强度损失: {loss_val.item():.6f}")
    
    # 测试组合损失
    print(f"\n测试组合损失...")
    fusion_loss = FusionLoss(
        ssim_weight=1.0,
        gradient_weight=10.0,
        intensity_weight=1.0
    ).to(device)
    
    total_loss, components = fusion_loss(fused, img1, img2, return_components=True)
    
    print(f"  ✓ 总损失: {components['total']:.6f}")
    print(f"    - SSIM损失: {components['ssim']:.6f}")
    print(f"    - 梯度损失: {components['gradient']:.6f}")
    print(f"    - 强度损失: {components['intensity']:.6f}")
    
    # 测试梯度反向传播
    print(f"\n测试梯度反向传播...")
    fused.requires_grad = True
    loss = fusion_loss(fused, img1, img2)
    loss.backward()
    print(f"  ✓ 梯度反向传播成功")
    print(f"  - 融合图像梯度范围: [{fused.grad.min():.6f}, {fused.grad.max():.6f}]")
    
    print("\n" + "=" * 70)
    print("✓ 损失函数测试通过!")
    print("=" * 70)
