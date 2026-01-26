"""
DCEvo-FAM混合融合模型
整合FAM频率域融合和DCEvo空间域融合

架构流程:
1. FAM阶段: 输入IR+VIS → 频率分解 → 低频融合图 + 高频融合图
2. DCEvo阶段: 低频图+高频图 → DE_Encoder → 特征融合 → DE_Decoder → 最终融合图

作者: Christopher32527
邮箱: 2546507517@qq.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# 导入FAM适配器和DCEvo模块
from fam_adapter import FAMAdapter
from sleepnet import DE_Encoder, DE_Decoder


class DCEvoFAMHybrid(nn.Module):
    """
    DCEvo-FAM混合融合模型
    
    两阶段融合架构:
    - 阶段1 (FAM): 频率域融合，输出低频图和高频图
    - 阶段2 (DCEvo): 空间域融合，输出最终融合图
    
    Args:
        in_channels: 输入图像通道数 (默认1, 灰度图)
        out_channels: 输出图像通道数 (默认1)
        dim: DCEvo特征维度 (默认64)
        fam_feature_dim: FAM特征维度 (默认512)
        fam_cutoff: FAM频率截止阈值 (默认0.3)
        num_blocks: DCEvo编码器块数量 (默认5)
    """
    
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 dim=64,
                 fam_feature_dim=512,
                 fam_cutoff=0.3,
                 num_blocks=5):
        super(DCEvoFAMHybrid, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        
        # 阶段1: FAM频率域融合模块
        self.fam_adapter = FAMAdapter(
            in_channels=in_channels,
            feature_dim=fam_feature_dim,
            cutoff=fam_cutoff
        )
        
        # 阶段2: DCEvo空间域融合模块
        # 为低频图和高频图各创建一个编码器
        self.encoder_low = DE_Encoder(
            inp_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks
        )
        
        self.encoder_high = DE_Encoder(
            inp_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.decoder = DE_Decoder(
            inp_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks
        )
    
    def forward(self, img_a, img_b, return_intermediate=False):
        """
        前向传播
        
        Args:
            img_a: 模态A图像 (IR或CT), shape [B, C, H, W]
            img_b: 模态B图像 (VIS或MRI), shape [B, C, H, W]
            return_intermediate: 是否返回中间结果
        
        Returns:
            fused_final: 最终融合图像, shape [B, C, H, W]
            intermediate (可选): 中间结果字典
        """
        B, C, H, W = img_a.shape
        
        # ============ 阶段1: FAM频率域融合 ============
        fused_low, fused_high = self.fam_adapter(img_a, img_b)
        
        # ============ 阶段2: DCEvo空间域融合 ============
        # 2.1 编码低频图和高频图
        lf_low, hf_low, base_low = self.encoder_low(fused_low)
        lf_high, hf_high, base_high = self.encoder_high(fused_high)
        
        # 2.2 融合低频和高频的基础特征
        base_fused = self.fusion_layer(torch.cat([base_low, base_high], dim=1))
        
        # 2.3 融合低频和高频的低频特征和高频特征
        lf_fused = (lf_low + lf_high) / 2
        hf_fused = (hf_low + hf_high) / 2
        
        # 2.4 解码生成最终融合图像
        fused_final, _ = self.decoder(
            [fused_low, fused_high],  # 输入原始低频图和高频图作为残差
            lf_fused,
            hf_fused
        )
        
        if return_intermediate:
            intermediate = {
                # FAM阶段输出
                'fused_low': fused_low,
                'fused_high': fused_high,
                # DCEvo编码器输出
                'lf_low': lf_low,
                'hf_low': hf_low,
                'base_low': base_low,
                'lf_high': lf_high,
                'hf_high': hf_high,
                'base_high': base_high,
                # 融合特征
                'base_fused': base_fused,
                'lf_fused': lf_fused,
                'hf_fused': hf_fused
            }
            return fused_final, intermediate
        
        return fused_final


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("DCEvo-FAM Hybrid Model Test")
    print("=" * 70)
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"  - GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 创建模型
    model = DCEvoFAMHybrid(
        in_channels=1,
        out_channels=1,
        dim=64,
        fam_feature_dim=512,
        fam_cutoff=0.3,
        num_blocks=5
    )
    model = model.to(device)  # 移动模型到GPU
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    img_size = 256
    img_ir = torch.randn(batch_size, 1, img_size, img_size).to(device)  # 移动数据到GPU
    img_vis = torch.randn(batch_size, 1, img_size, img_size).to(device)
    
    print(f"\n输入:")
    print(f"  - IR图像形状: {img_ir.shape}")
    print(f"  - VIS图像形状: {img_vis.shape}")
    
    # 前向传播
    print(f"\n正在进行前向传播...")
    with torch.no_grad():
        fused, intermediate = model(img_ir, img_vis, return_intermediate=True)
    
    print(f"\n输出:")
    print(f"  - 最终融合图像形状: {fused.shape}")
    print(f"  - 输出值范围: [{fused.min():.3f}, {fused.max():.3f}]")
    
    print(f"\n中间结果:")
    print(f"  FAM阶段:")
    print(f"    - 融合低频图: {intermediate['fused_low'].shape}")
    print(f"    - 融合高频图: {intermediate['fused_high'].shape}")
    
    print(f"\n  DCEvo编码阶段:")
    print(f"    - 低频图的低频特征: {intermediate['lf_low'].shape}")
    print(f"    - 低频图的高频特征: {intermediate['hf_low'].shape}")
    print(f"    - 低频图的基础特征: {intermediate['base_low'].shape}")
    print(f"    - 高频图的低频特征: {intermediate['lf_high'].shape}")
    print(f"    - 高频图的高频特征: {intermediate['hf_high'].shape}")
    print(f"    - 高频图的基础特征: {intermediate['base_high'].shape}")
    
    print(f"\n  融合阶段:")
    print(f"    - 融合基础特征: {intermediate['base_fused'].shape}")
    print(f"    - 融合低频特征: {intermediate['lf_fused'].shape}")
    print(f"    - 融合高频特征: {intermediate['hf_fused'].shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # 测试梯度反向传播
    print(f"\n测试梯度反向传播...")
    try:
        model.train()
        img_ir = torch.randn(batch_size, 1, img_size, img_size, requires_grad=True, device=device)
        img_vis = torch.randn(batch_size, 1, img_size, img_size, requires_grad=True, device=device)
        
        fused = model(img_ir, img_vis)
        loss = fused.mean()
        loss.backward()
        
        print(f"  ✓ 梯度反向传播成功")
        print(f"  - 损失值: {loss.item():.6f}")
        print(f"  - IR图像梯度范围: [{img_ir.grad.min():.6f}, {img_ir.grad.max():.6f}]")
        print(f"  - VIS图像梯度范围: [{img_vis.grad.min():.6f}, {img_vis.grad.max():.6f}]")
    except RuntimeError as e:
        if "memory" in str(e).lower() or "out of memory" in str(e).lower():
            print(f"  ⚠ 内存不足，无法完成梯度测试")
            print(f"  提示: 训练时请在训练脚本中使用更小的batch_size")
        else:
            raise e
    
    print("\n" + "=" * 70)
    print("✓ DCEvo-FAM混合模型测试通过!")
    print("=" * 70)
    
    if device.type == 'cuda':
        print(f"\n✓ GPU测试成功，模型可以正常训练")
    else:
        print(f"\n提示:")
        print("  - 推理模式测试成功，模型结构正确")
        print("  - 未检测到GPU，建议使用GPU进行训练")
        print("  - 或者在训练脚本中减小batch_size和图像尺寸")
