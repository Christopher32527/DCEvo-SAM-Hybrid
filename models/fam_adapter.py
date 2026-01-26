"""
FAM (Frequency-Aware Matching) Adapter Module
适配自FAMNet (AAAI 2025)，用于DCEvo-FAM融合架构

主要修改:
1. 输入从特征向量改为图像张量
2. 输出从三频段(低/中/高)改为双频段(低/高)
3. 添加图像-特征转换层
4. 简化接口以适配DCEvo

原始代码来源: https://github.com/primebo1/FAMNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionMatching(nn.Module):
    """
    注意力匹配模块
    计算两个模态特征之间的相似度并进行加权融合
    
    Args:
        feature_dim: 特征维度 (默认512)
        seq_len: 序列长度 (默认900)
    """
    
    def __init__(self, feature_dim=512, seq_len=900):
        super(AttentionMatching, self).__init__()
        
        # Support特征投影
        self.fc_spt = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        
        # Query特征投影
        self.fc_qry = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        
        # 融合层
        self.fc_fusion = nn.Sequential(
            nn.Linear(seq_len * 2, seq_len // 5),
            nn.ReLU(),
            nn.Linear(seq_len // 5, seq_len),
        )
        
        self.sigmoid = nn.Sigmoid()

    def correlation_matrix(self, feat_a, feat_b):
        """
        计算两个特征之间的余弦相似度
        
        Args:
            feat_a: 特征A, shape [B, C, N]
            feat_b: 特征B, shape [B, C, N]
        
        Returns:
            similarity: 相似度矩阵, shape [B, 1, N]
        """
        # L2归一化
        feat_a_norm = F.normalize(feat_a, p=2, dim=1)
        feat_b_norm = F.normalize(feat_b, p=2, dim=1)
        
        # 计算余弦相似度
        cosine_similarity = torch.sum(feat_a_norm * feat_b_norm, dim=1, keepdim=True)
        
        return cosine_similarity

    def forward(self, feat_a, feat_b, band='low'):
        """
        前向传播
        
        Args:
            feat_a: 模态A特征, shape [B, C, N]
            feat_b: 模态B特征, shape [B, C, N]
            band: 频段类型 ('low' 或 'high')
        
        Returns:
            fused: 融合后的特征, shape [B, C, N]
        """
        # 特征投影
        feat_a_proj = F.relu(self.fc_spt(feat_a))
        feat_b_proj = F.relu(self.fc_qry(feat_b))
        
        # 计算相似度
        similarity = self.sigmoid(self.correlation_matrix(feat_a, feat_b))
        
        # 根据频段类型选择加权策略
        if band in ['low', 'high']:
            # 低频和高频: 使用互补权重 (1-similarity)
            weighted_a = (1 - similarity) * feat_a_proj
            weighted_b = (1 - similarity) * feat_b_proj
        else:
            # 其他: 使用相似度权重
            weighted_a = similarity * feat_a_proj
            weighted_b = similarity * feat_b_proj
        
        # 拼接并融合
        combined = torch.cat((weighted_a, weighted_b), dim=2)
        fused = F.relu(self.fc_fusion(combined))
        
        return fused


class FAMAdapter(nn.Module):
    """
    FAM适配器模块
    将FAMNet的FAM模块适配为处理图像输入的双频段融合模块
    
    Args:
        in_channels: 输入图像通道数 (默认1, 灰度图)
        feature_dim: 特征维度 (默认512)
        cutoff: 频率截止阈值 (默认0.3)
    """
    
    def __init__(self, in_channels=1, feature_dim=512, cutoff=0.3):
        super(FAMAdapter, self).__init__()
        
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.cutoff = cutoff
        
        # 图像到特征的转换 (简单的卷积编码器)
        self.img_to_feat = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 注意力匹配模块 (低频和高频各一个)
        self.attention_low = AttentionMatching(feature_dim, seq_len=900)
        self.attention_high = AttentionMatching(feature_dim, seq_len=900)
        
        # 自适应池化 (将特征序列长度统一为900)
        self.adapt_pooling = nn.AdaptiveAvgPool1d(900)
        
        # 特征到图像的转换 (简单的卷积解码器)
        self.feat_to_img = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def reshape_to_square(self, tensor):
        """
        将1D特征序列重塑为2D方形张量 (用于FFT)
        
        Args:
            tensor: 输入张量, shape [B, C, N]
        
        Returns:
            square_tensor: 方形张量, shape [B, C, H, W]
            H, W: 方形尺寸
            N: 原始序列长度
        """
        B, C, N = tensor.shape
        side_length = int(np.ceil(np.sqrt(N)))
        padded_length = side_length ** 2
        
        # 填充到完全平方数
        padded_tensor = torch.zeros((B, C, padded_length), device=tensor.device)
        padded_tensor[:, :, :N] = tensor
        
        # 重塑为方形
        square_tensor = padded_tensor.view(B, C, side_length, side_length)
        
        return square_tensor, side_length, side_length, N

    def filter_frequency_bands(self, tensor, cutoff=0.3):
        """
        频率域分解: 将特征分解为低频和高频两个频段
        
        Args:
            tensor: 输入特征, shape [B, C, N]
            cutoff: 频率截止阈值 (0-1之间)
        
        Returns:
            low_freq_tensor: 低频特征, shape [B, C, N]
            high_freq_tensor: 高频特征, shape [B, C, N]
        """
        tensor = tensor.float()
        device = tensor.device  # 使用tensor的device而不是self.device
        
        # 重塑为方形以便进行2D FFT
        tensor_2d, H, W, N = self.reshape_to_square(tensor)
        B, C, _, _ = tensor_2d.shape
        
        # 计算频率半径
        max_radius = np.sqrt((H // 2)**2 + (W // 2)**2)
        low_cutoff = max_radius * cutoff
        high_cutoff = max_radius * (1 - cutoff)
        
        # 2D FFT变换
        fft_tensor = torch.fft.fftshift(torch.fft.fft2(tensor_2d, dim=(-2, -1)), dim=(-2, -1))
        
        # 创建低频和高频滤波器
        def create_filter(shape, cutoff_val, mode='low'):
            rows, cols = shape
            center_row, center_col = rows // 2, cols // 2
            
            y, x = torch.meshgrid(
                torch.arange(rows, device=device),
                torch.arange(cols, device=device),
                indexing='ij'
            )
            distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)
            
            mask = torch.zeros((rows, cols), dtype=torch.float32, device=device)
            
            if mode == 'low':
                mask[distance <= cutoff_val] = 1
            elif mode == 'high':
                mask[distance >= cutoff_val] = 1
            
            return mask
        
        # 创建滤波器
        low_pass_filter = create_filter((H, W), low_cutoff, mode='low')[None, None, :, :]
        high_pass_filter = create_filter((H, W), high_cutoff, mode='high')[None, None, :, :]
        
        # 应用滤波器
        low_freq_fft = fft_tensor * low_pass_filter
        high_freq_fft = fft_tensor * high_pass_filter
        
        # 逆FFT变换
        low_freq_2d = torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        high_freq_2d = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        
        # 重塑回1D序列
        low_freq_tensor = low_freq_2d.view(B, C, H * W)[:, :, :N]
        high_freq_tensor = high_freq_2d.view(B, C, H * W)[:, :, :N]
        
        return low_freq_tensor, high_freq_tensor

    def forward(self, img_a, img_b, return_intermediate=False):
        """
        前向传播: 双频段融合
        
        Args:
            img_a: 模态A图像, shape [B, C, H, W]
            img_b: 模态B图像, shape [B, C, H, W]
            return_intermediate: 是否返回中间结果
        
        Returns:
            fused_low: 融合低频图, shape [B, C, H, W]
            fused_high: 融合高频图, shape [B, C, H, W]
            intermediate (可选): 中间结果字典
        """
        B, C, H, W = img_a.shape
        
        # 1. 图像转特征
        feat_a = self.img_to_feat(img_a)  # [B, feature_dim, H, W]
        feat_b = self.img_to_feat(img_b)
        
        # 2. 特征展平为序列
        feat_a_flat = feat_a.view(B, self.feature_dim, -1)  # [B, feature_dim, H*W]
        feat_b_flat = feat_b.view(B, self.feature_dim, -1)
        
        # 3. 自适应池化到固定长度
        feat_a_pooled = self.adapt_pooling(feat_a_flat)  # [B, feature_dim, 900]
        feat_b_pooled = self.adapt_pooling(feat_b_flat)
        
        # 4. 频率域分解
        feat_a_low, feat_a_high = self.filter_frequency_bands(feat_a_pooled, self.cutoff)
        feat_b_low, feat_b_high = self.filter_frequency_bands(feat_b_pooled, self.cutoff)
        
        # 5. 注意力匹配融合
        fused_feat_low = self.attention_low(feat_a_low, feat_b_low, band='low')
        fused_feat_high = self.attention_high(feat_a_high, feat_b_high, band='high')
        
        # 6. 特征序列重塑回2D
        # 先上采样回原始空间分辨率
        fused_feat_low_2d = F.interpolate(
            fused_feat_low.view(B, self.feature_dim, 30, 30),  # sqrt(900) = 30
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        fused_feat_high_2d = F.interpolate(
            fused_feat_high.view(B, self.feature_dim, 30, 30),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        # 7. 特征转图像
        fused_low = self.feat_to_img(fused_feat_low_2d)
        fused_high = self.feat_to_img(fused_feat_high_2d)
        
        if return_intermediate:
            intermediate = {
                'feat_a': feat_a,
                'feat_b': feat_b,
                'feat_a_low': feat_a_low,
                'feat_a_high': feat_a_high,
                'feat_b_low': feat_b_low,
                'feat_b_high': feat_b_high,
                'fused_feat_low': fused_feat_low,
                'fused_feat_high': fused_feat_high
            }
            return fused_low, fused_high, intermediate
        
        return fused_low, fused_high


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("FAM Adapter Module Test")
    print("=" * 60)
    
    # 创建模型
    model = FAMAdapter(in_channels=1, feature_dim=512, cutoff=0.3)
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    img_size = 256
    img_a = torch.randn(batch_size, 1, img_size, img_size)
    img_b = torch.randn(batch_size, 1, img_size, img_size)
    
    print(f"\n输入图像A形状: {img_a.shape}")
    print(f"输入图像B形状: {img_b.shape}")
    
    # 前向传播
    with torch.no_grad():
        fused_low, fused_high, intermediate = model(img_a, img_b, return_intermediate=True)
    
    print(f"\n输出融合低频图形状: {fused_low.shape}")
    print(f"输出融合高频图形状: {fused_high.shape}")
    
    print(f"\n中间特征:")
    print(f"  - feat_a: {intermediate['feat_a'].shape}")
    print(f"  - feat_a_low: {intermediate['feat_a_low'].shape}")
    print(f"  - feat_a_high: {intermediate['feat_a_high'].shape}")
    print(f"  - fused_feat_low: {intermediate['fused_feat_low'].shape}")
    print(f"  - fused_feat_high: {intermediate['fused_feat_high'].shape}")
    
    # 检查输出范围
    print(f"\n输出值范围:")
    print(f"  - 低频图: [{fused_low.min():.3f}, {fused_low.max():.3f}]")
    print(f"  - 高频图: [{fused_high.min():.3f}, {fused_high.max():.3f}]")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("✓ FAM Adapter测试通过!")
    print("=" * 60)
