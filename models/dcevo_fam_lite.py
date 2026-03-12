"""
DCEvo-FAM轻量版融合模型
针对显存受限环境优化

主要优化:
1. 共享编码器 (而不是两个独立编码器)
2. 减少FAM特征维度
3. 简化融合层
4. 支持梯度检查点

作者: Christopher32527
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


class DCEvoFAMLite(nn.Module):
    """
    DCEvo-FAM轻量版融合模型
    
    优化策略:
    - 共享编码器减少参数量
    - 降低特征维度
    - 简化融合层
    
    Args:
        in_channels: 输入图像通道数 (默认1)
        out_channels: 输出图像通道数 (默认1)
        dim: DCEvo特征维度 (默认64)
        fam_feature_dim: FAM特征维度 (默认128, 大幅减少)
        fam_cutoff: FAM频率截止阈值 (默认0.3)
        num_blocks: DCEvo编码器块数量 (默认3, 减少)
        use_checkpoint: 是否使用梯度检查点 (默认True)
    """
    
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 dim=64,
                 fam_feature_dim=128,  # 大幅减少
                 fam_cutoff=0.3,
                 num_blocks=3,         # 减少块数
                 use_checkpoint=True):
        super(DCEvoFAMLite, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        
        # 阶段1: FAM频率域融合模块 (轻量版)
        self.fam_adapter = FAMAdapter(
            in_channels=in_channels,
            feature_dim=fam_feature_dim,  # 减少特征维度
            cutoff=fam_cutoff
        )
        
        # 阶段2: 共享DCEvo编码器 (节省显存)
        self.shared_encoder = DE_Encoder(
            inp_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks
        )
        
        # 简化的特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),  # 简化融合
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
        前向传播 (轻量版)
        
        Args:
            img_a: 模态A图像, shape [B, C, H, W]
            img_b: 模态B图像, shape [B, C, H, W]
            return_intermediate: 是否返回中间结果
        
        Returns:
            fused_final: 最终融合图像, shape [B, C, H, W]
            intermediate (可选): 中间结果字典
        """
        B, C, H, W = img_a.shape
        
        # ============ 阶段1: FAM频率域融合 ============
        if self.use_checkpoint and self.training:
            fused_low, fused_high = torch.utils.checkpoint.checkpoint(
                self.fam_adapter, img_a, img_b
            )
        else:
            fused_low, fused_high = self.fam_adapter(img_a, img_b)
        
        # ============ 阶段2: 共享编码器 ============
        # 使用共享编码器分别编码低频图和高频图
        if self.use_checkpoint and self.training:
            lf_low, hf_low, base_low = torch.utils.checkpoint.checkpoint(
                self.shared_encoder, fused_low
            )
            lf_high, hf_high, base_high = torch.utils.checkpoint.checkpoint(
                self.shared_encoder, fused_high
            )
        else:
            lf_low, hf_low, base_low = self.shared_encoder(fused_low)
            lf_high, hf_high, base_high = self.shared_encoder(fused_high)
        
        # 简化的特征融合
        base_fused = self.fusion_layer(torch.cat([base_low, base_high], dim=1))
        
        # 简单平均融合低频和高频特征
        lf_fused = (lf_low + lf_high) * 0.5
        hf_fused = (hf_low + hf_high) * 0.5
        
        # 解码生成最终融合图像
        if self.use_checkpoint and self.training:
            fused_final, _ = torch.utils.checkpoint.checkpoint(
                self.decoder, [fused_low, fused_high], lf_fused, hf_fused
            )
        else:
            fused_final, _ = self.decoder(
                [fused_low, fused_high], lf_fused, hf_fused
            )
        
        if return_intermediate:
            intermediate = {
                'fused_low': fused_low,
                'fused_high': fused_high,
                'base_low': base_low,
                'base_high': base_high,
                'base_fused': base_fused,
                'lf_fused': lf_fused,
                'hf_fused': hf_fused
            }
            return fused_final, intermediate
        
        return fused_final


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("DCEvo-FAM Lite Model Test")
    print("=" * 70)
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"  - GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 创建轻量版模型
    model = DCEvoFAMLite(
        in_channels=1,
        out_channels=1,
        dim=64,
        fam_feature_dim=128,  # 轻量版
        fam_cutoff=0.3,
        num_blocks=3,         # 轻量版
        use_checkpoint=True   # 开启梯度检查点
    )
    model = model.to(device)
    model.eval()
    
    # 创建测试输入 (更小的batch和图像)
    batch_size = 1
    img_size = 128  # 更小的图像
    img_ir = torch.randn(batch_size, 1, img_size, img_size).to(device)
    img_vis = torch.randn(batch_size, 1, img_size, img_size).to(device)
    
    print(f"\n输入:")
    print(f"  - IR图像形状: {img_ir.shape}")
    print(f"  - VIS图像形状: {img_vis.shape}")
    
    # 前向传播
    print(f"\n正在进行前向传播...")
    with torch.no_grad():
        fused = model(img_ir, img_vis)
    
    print(f"\n输出:")
    print(f"  - 最终融合图像形状: {fused.shape}")
    print(f"  - 输出值范围: [{fused.min():.3f}, {fused.max():.3f}]")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # 测试梯度反向传播 (轻量版)
    print(f"\n测试梯度反向传播...")
    try:
        model.train()
        img_ir = torch.randn(1, 1, 128, 128, requires_grad=True, device=device)
        img_vis = torch.randn(1, 1, 128, 128, requires_grad=True, device=device)
        
        fused = model(img_ir, img_vis)
        loss = fused.mean()
        loss.backward()
        
        print(f"  ✓ 梯度反向传播成功")
        print(f"  - 损失值: {loss.item():.6f}")
    except RuntimeError as e:
        if "memory" in str(e).lower() or "out of memory" in str(e).lower():
            print(f"  ⚠ 内存不足，请进一步减小模型或图像尺寸")
        else:
            print(f"  ❌ 其他错误: {e}")
    
    print("\n" + "=" * 70)
    print("✓ DCEvo-FAM轻量版模型测试完成!")
    print("=" * 70)
    
    print(f"\n优化说明:")
    print("  - 使用共享编码器 (减少50%参数)")
    print("  - FAM特征维度: 512 → 128 (减少75%)")
    print("  - 编码器块数: 5 → 3 (减少40%)")
    print("  - 开启梯度检查点 (节省显存)")
    print("  - 简化融合层 (减少计算)")