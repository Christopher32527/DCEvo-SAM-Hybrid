"""
DCEvo + SAM 混合模型推理脚本
用于红外+可见光图像融合和分割
"""

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.dcevo_sam_hybrid import DCEvo_SAM_Hybrid
from utils.img_read_save import image_read_cv2, img_save
import argparse


def show_mask(mask, ax, random_color=False):
    """可视化分割掩码"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """可视化提示点"""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    """可视化边界框"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                               facecolor=(0,0,0,0), lw=2))


def inference_single_image(model, ir_path, vis_path, output_dir, 
                          point_coords=None, point_labels=None, boxes=None,
                          visualize=True):
    """
    对单张图像���行推理
    """
    device = next(model.parameters()).device
    
    # 读取图像
    ir_img = image_read_cv2(ir_path, mode='GRAY')
    vis_img = image_read_cv2(vis_path, mode='GRAY')
    
    # 保存原始尺寸
    original_h, original_w = ir_img.shape
    
    # SAM需要1024x1024输入,先resize
    target_size = 1024
    ir_img_resized = cv2.resize(ir_img, (target_size, target_size))
    vis_img_resized = cv2.resize(vis_img, (target_size, target_size))
    
    # 转换为tensor
    ir_tensor = torch.FloatTensor(ir_img_resized[np.newaxis, np.newaxis, ...] / 255.0).to(device)
    vis_tensor = torch.FloatTensor(vis_img_resized[np.newaxis, np.newaxis, ...] / 255.0).to(device)
    
    # 调整提示点坐标到resize后的尺寸
    if point_coords is not None:
        scale_x = target_size / original_w
        scale_y = target_size / original_h
        point_coords_scaled = [[int(point_coords[0][0] * scale_x), int(point_coords[0][1] * scale_y)]]
        point_coords = torch.FloatTensor(point_coords_scaled).unsqueeze(0).to(device)
        point_labels = torch.LongTensor(point_labels).unsqueeze(0).to(device)
    if boxes is not None:
        boxes = torch.FloatTensor(boxes).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        fused_img, masks, iou_pred = model(
            ir_tensor, vis_tensor, 
            point_coords, point_labels, boxes
        )
    
    # 转换为numpy并resize回原始尺寸
    fused_np = (fused_img.squeeze().cpu().numpy() * 255).astype(np.uint8)
    fused_np = cv2.resize(fused_np, (original_w, original_h))
    
    mask_np = (masks.squeeze().cpu().numpy() > 0).astype(np.uint8)
    mask_np = cv2.resize(mask_np, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    # 保存结果
    img_name = os.path.basename(ir_path).split('.')[0]
    
    # 保存融合图像
    fusion_path = os.path.join(output_dir, f'{img_name}_fused.png')
    cv2.imwrite(fusion_path, fused_np)
    
    # 保存分割掩码
    mask_path = os.path.join(output_dir, f'{img_name}_mask.png')
    cv2.imwrite(mask_path, mask_np * 255)
    
    # 可视化
    if visualize:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 原始红外图像
        axes[0].imshow(ir_img, cmap='gray')
        axes[0].set_title('Infrared')
        axes[0].axis('off')
        
        # 原始可见光图像
        axes[1].imshow(vis_img, cmap='gray')
        axes[1].set_title('Visible')
        axes[1].axis('off')
        
        # 融合图像
        axes[2].imshow(fused_np, cmap='gray')
        axes[2].set_title('Fused Image')
        axes[2].axis('off')
        
        # 分割结果
        axes[3].imshow(fused_np, cmap='gray')
        show_mask(mask_np, axes[3])
        if point_coords is not None:
            show_points(point_coords.squeeze().cpu().numpy(), 
                       point_labels.squeeze().cpu().numpy(), axes[3])
        if boxes is not None:
            show_box(boxes.squeeze().cpu().numpy(), axes[3])
        axes[3].set_title(f'Segmentation (IoU: {iou_pred.item():.3f})')
        axes[3].axis('off')
        
        plt.tight_layout()
        vis_path = os.path.join(output_dir, f'{img_name}_result.png')
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 结果已保存: {vis_path}")
    
    return fused_np, mask_np, iou_pred.item()


def inference_dataset(model, dataset_name='M3FD', output_dir='results', 
                     use_center_point=True):
    """
    对整个数据集进行推理
    """
    dataset_path = os.path.join('datasets', dataset_name)
    ir_dir = os.path.join(dataset_path, 'ir')
    vis_dir = os.path.join(dataset_path, 'vi')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像列表
    img_names = os.listdir(ir_dir)
    
    print(f"开始处理 {dataset_name} 数据集, 共 {len(img_names)} 张图像...")
    
    for idx, img_name in enumerate(img_names):
        ir_path = os.path.join(ir_dir, img_name)
        vis_path = os.path.join(vis_dir, img_name)
        
        # 如果使用中心点作为提示
        if use_center_point:
            # 读取图像尺寸
            img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            point_coords = [[w//2, h//2]]  # 中心点
            point_labels = [1]  # 前景点
        else:
            point_coords = None
            point_labels = None
        
        # 推理
        fused, mask, iou = inference_single_image(
            model, ir_path, vis_path, output_dir,
            point_coords=point_coords,
            point_labels=point_labels,
            visualize=(idx < 5)  # 只可视化前5张
        )
        
        print(f"[{idx+1}/{len(img_names)}] {img_name} - IoU: {iou:.3f}")
    
    print(f"\n✓ 所有结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='DCEvo+SAM混合模型推理')
    parser.add_argument('--sam_checkpoint', type=str, 
                       default='ckpt/sam_vit_b_01ec64.pth',
                       help='SAM权重路径')
    parser.add_argument('--dcevo_checkpoint', type=str,
                       default='ckpt/DCEvo_fusion.pth',
                       help='DCEvo融合权重路径')
    parser.add_argument('--dataset', type=str, default='M3FD',
                       choices=['M3FD', 'FMB', 'TNO', 'RoadScene'],
                       help='数据集名称')
    parser.add_argument('--output_dir', type=str, default='results/hybrid',
                       help='输出目录')
    parser.add_argument('--use_center_point', action='store_true',
                       help='使用图像中心点作为提示')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 检查设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载模型...")
    model = DCEvo_SAM_Hybrid(
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type='vit_b',
        dcevo_fusion_checkpoint=args.dcevo_checkpoint,
        device=device
    )
    model.eval()
    print("✓ 模型加载完成")
    
    # 推理
    inference_dataset(
        model, 
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        use_center_point=args.use_center_point
    )


if __name__ == '__main__':
    main()
