"""
DCEvo + SAM 混合架构
保留DCEvo的图像融合能力,使用SAM进行分割
"""

import torch
import torch.nn as nn
import sys
import os

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sleepnet import DE_Encoder, DE_Decoder, LowFreqExtractor, HighFreqExtractor
from segment_anything import sam_model_registry, SamPredictor


class DCEvo_SAM_Hybrid(nn.Module):
    """
    混合架构:
    1. 使用DCEvo的DE_Encoder提取红外和可见光特征
    2. 使用LowFreqExtractor和HighFreqExtractor进行特征融合
    3. 使用DE_Decoder生成融合图像
    4. 使用SAM的Decoder进行分割
    """
    def __init__(self, 
                 sam_checkpoint='ckpt/sam_vit_b_01ec64.pth',
                 sam_model_type='vit_b',
                 dcevo_fusion_checkpoint='ckpt/DCEvo_fusion.pth',
                 device='cuda'):
        super(DCEvo_SAM_Hybrid, self).__init__()
        
        self.device = device
        
        # ===== DCEvo 融合模块 =====
        self.encoder = DE_Encoder().to(device)
        self.decoder = DE_Decoder().to(device)
        self.lf_extractor = LowFreqExtractor(dim=64).to(device)
        self.hf_extractor = HighFreqExtractor(num_layers=3).to(device)
        
        # 加载DCEvo融合权重
        if dcevo_fusion_checkpoint:
            self.load_dcevo_weights(dcevo_fusion_checkpoint)
        
        # ===== SAM 分割模块 =====
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam.to(device)
        
        # 特征适配层: 将DCEvo的64通道特征转换为SAM需要的256通道
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        ).to(device)
        
    def load_dcevo_weights(self, checkpoint_path):
        """加载DCEvo融合模型的预训练权重"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 处理DataParallel保存的权重(移除'module.'前缀)
        def remove_module_prefix(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # 移除'module.'前缀
                else:
                    new_state_dict[k] = v
            return new_state_dict
        
        self.encoder.load_state_dict(remove_module_prefix(checkpoint['DE_Encoder']))
        self.decoder.load_state_dict(remove_module_prefix(checkpoint['DE_Decoder']))
        self.lf_extractor.load_state_dict(remove_module_prefix(checkpoint['LowFreqExtractor']))
        self.hf_extractor.load_state_dict(remove_module_prefix(checkpoint['HighFreqExtractor']))
        print(f"✓ DCEvo融合权重加载成功: {checkpoint_path}")
    
    def freeze_dcevo(self):
        """冻结DCEvo参数,只训练SAM部分"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.lf_extractor.parameters():
            param.requires_grad = False
        for param in self.hf_extractor.parameters():
            param.requires_grad = False
        print("✓ DCEvo参数已冻结")
    
    def unfreeze_all(self):
        """解冻所有参数,进行端到端训练"""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ 所有参数已解冻")
    
    def forward_fusion(self, ir_img, vis_img):
        """
        DCEvo融合前向传播
        输入: 
            ir_img: [B, 1, H, W] 红外图像
            vis_img: [B, 1, H, W] 可见光图像
        输出:
            fused_img: [B, 1, H, W] 融合图像
            fusion_feature: [B, 64, H, W] 融合特征
        """
        # 提取特征
        lf_ir, hf_ir, base_ir = self.encoder(ir_img)
        lf_vis, hf_vis, base_vis = self.encoder(vis_img)
        
        # 特征融合
        lf_fused = self.lf_extractor(lf_ir + lf_vis)
        hf_fused = self.hf_extractor(hf_ir + hf_vis)
        
        # 解码生成融合图像
        fused_img, fusion_feature = self.decoder(
            ir_img * 0.5 + vis_img * 0.5, 
            lf_fused, 
            hf_fused
        )
        
        return fused_img, fusion_feature
    
    def forward_segmentation(self, fused_img, fusion_feature, point_coords=None, point_labels=None, boxes=None):
        """
        SAM分割前向传播
        输入:
            fused_img: [B, 1, H, W] 融合图像
            fusion_feature: [B, 64, H, W] DCEvo融合特征
            point_coords: [B, N, 2] 提示点坐标 (可选)
            point_labels: [B, N] 提示点标签 (可选)
            boxes: [B, 4] 边界框 (可选)
        输出:
            masks: [B, 1, H, W] 分割掩码
            iou_predictions: [B, 1] IoU预测
        """
        # 将单通道融合图像转换为3通道 (SAM需要RGB输入)
        fused_img_rgb = fused_img.repeat(1, 3, 1, 1)  # [B, 3, H, W]
        
        # 使用SAM的图像编码器
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(fused_img_rgb)  # [B, 256, 64, 64]
        
        # 适配DCEvo特征到SAM特征空间
        adapted_feature = self.feature_adapter(fusion_feature)  # [B, 256, H, W]
        
        # 调整特征尺寸以匹配SAM的嵌入尺寸
        adapted_feature = nn.functional.interpolate(
            adapted_feature, 
            size=image_embeddings.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # 融合DCEvo特征和SAM图像嵌入
        combined_embeddings = image_embeddings + adapted_feature * 0.5
        
        # 准备提示
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(point_coords, point_labels) if point_coords is not None else None,
            boxes=boxes,
            masks=None,
        )
        
        # SAM掩码解码器
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=combined_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # 上采样到原始分辨率
        masks = nn.functional.interpolate(
            low_res_masks,
            size=(fused_img.shape[-2], fused_img.shape[-1]),
            mode='bilinear',
            align_corners=False
        )
        
        return masks, iou_predictions
    
    def forward(self, ir_img, vis_img, point_coords=None, point_labels=None, boxes=None):
        """
        完整的前向传播: 融合 + 分割
        输入:
            ir_img: [B, 1, H, W] 红外图像
            vis_img: [B, 1, H, W] 可见光图像
            point_coords: [B, N, 2] 提示点坐标 (可选)
            point_labels: [B, N] 提示点标签 (可选)
            boxes: [B, 4] 边界框 (可选)
        输出:
            fused_img: [B, 1, H, W] 融合图像
            masks: [B, 1, H, W] 分割掩码
            iou_predictions: [B, 1] IoU预测
        """
        # 步骤1: DCEvo融合
        fused_img, fusion_feature = self.forward_fusion(ir_img, vis_img)
        
        # 步骤2: SAM分割
        masks, iou_predictions = self.forward_segmentation(
            fused_img, fusion_feature, point_coords, point_labels, boxes
        )
        
        return fused_img, masks, iou_predictions


if __name__ == '__main__':
    # 测试代码
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 获取正确的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sam_ckpt = os.path.join(project_root, 'ckpt', 'sam_vit_b_01ec64.pth')
    dcevo_ckpt = os.path.join(project_root, 'ckpt', 'DCEvo_fusion.pth')
    
    # 创建模型
    model = DCEvo_SAM_Hybrid(
        sam_checkpoint=sam_ckpt,
        sam_model_type='vit_b',
        dcevo_fusion_checkpoint=dcevo_ckpt,
        device=device
    )
    
    # 测试输入 (SAM需要1024x1024)
    ir_img = torch.randn(1, 1, 1024, 1024).to(device)
    vis_img = torch.randn(1, 1, 1024, 1024).to(device)
    point_coords = torch.tensor([[[512, 512]]]).float().to(device)
    point_labels = torch.tensor([[1]]).to(device)
    
    # 前向传播
    with torch.no_grad():
        fused_img, masks, iou_pred = model(ir_img, vis_img, point_coords, point_labels)
    
    print(f"融合图像形状: {fused_img.shape}")
    print(f"分割掩码形状: {masks.shape}")
    print(f"IoU预测: {iou_pred}")
