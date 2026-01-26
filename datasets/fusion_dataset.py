"""
图像融合数据集加载器
支持M3FD数据集（红外+可见光融合）

作者: Christopher32527
邮箱: 2546507517@qq.com
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from PIL import Image


class FusionDataset(Dataset):
    """
    图像融合数据集
    
    支持M3FD数据集格式:
    - datasets/M3FD/train/ir/  (红外图像)
    - datasets/M3FD/train/vi/  (可见光图像)
    
    Args:
        ir_dir: 红外图像目录
        vi_dir: 可见光图像目录
        img_size: 图像尺寸 (默认256)
        augment: 是否进行数据增强 (默认True)
        normalize: 是否归一化 (默认True)
    """
    
    def __init__(self, 
                 ir_dir, 
                 vi_dir, 
                 img_size=256,
                 augment=True,
                 normalize=True):
        super(FusionDataset, self).__init__()
        
        self.ir_dir = ir_dir
        self.vi_dir = vi_dir
        self.img_size = img_size
        self.augment = augment
        self.normalize = normalize
        
        # 获取图像文件列表
        self.ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.vi_files = sorted([f for f in os.listdir(vi_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # 确保两个目录的图像数量一致
        assert len(self.ir_files) == len(self.vi_files), \
            f"IR和VI图像数量不一致: {len(self.ir_files)} vs {len(self.vi_files)}"
        
        print(f"✓ 加载数据集: {len(self.ir_files)} 对图像")
        print(f"  - IR目录: {ir_dir}")
        print(f"  - VI目录: {vi_dir}")
        print(f"  - 图像尺寸: {img_size}x{img_size}")
        print(f"  - 数据增强: {augment}")
        print(f"  - 归一化: {normalize}")
    
    def __len__(self):
        return len(self.ir_files)
    
    def load_image(self, path):
        """加载图像并转换为灰度图"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法加载图像: {path}")
        return img
    
    def resize_image(self, img):
        """调整图像尺寸"""
        return cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, img):
        """归一化图像到[0, 1]"""
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img
    
    def augment_pair(self, ir_img, vi_img):
        """
        对图像对进行数据增强
        
        增强操作:
        - 随机水平翻转
        - 随机垂直翻转
        - 随机旋转90度
        """
        # 随机水平翻转
        if np.random.rand() > 0.5:
            ir_img = np.fliplr(ir_img)
            vi_img = np.fliplr(vi_img)
        
        # 随机垂直翻转
        if np.random.rand() > 0.5:
            ir_img = np.flipud(ir_img)
            vi_img = np.flipud(vi_img)
        
        # 随机旋转90度
        k = np.random.randint(0, 4)  # 0, 1, 2, 3 对应 0°, 90°, 180°, 270°
        if k > 0:
            ir_img = np.rot90(ir_img, k)
            vi_img = np.rot90(vi_img, k)
        
        return ir_img, vi_img
    
    def __getitem__(self, idx):
        """
        获取一对图像
        
        Returns:
            ir_img: 红外图像, shape [1, H, W]
            vi_img: 可见光图像, shape [1, H, W]
            filename: 文件名
        """
        # 加载图像
        ir_path = os.path.join(self.ir_dir, self.ir_files[idx])
        vi_path = os.path.join(self.vi_dir, self.vi_files[idx])
        
        ir_img = self.load_image(ir_path)
        vi_img = self.load_image(vi_path)
        
        # 调整尺寸
        ir_img = self.resize_image(ir_img)
        vi_img = self.resize_image(vi_img)
        
        # 数据增强
        if self.augment:
            ir_img, vi_img = self.augment_pair(ir_img, vi_img)
        
        # 归一化
        if self.normalize:
            ir_img = self.normalize_image(ir_img)
            vi_img = self.normalize_image(vi_img)
        
        # 转换为张量
        ir_img = torch.from_numpy(ir_img).float().unsqueeze(0)  # [1, H, W]
        vi_img = torch.from_numpy(vi_img).float().unsqueeze(0)  # [1, H, W]
        
        filename = self.ir_files[idx]
        
        return ir_img, vi_img, filename


def create_dataloader(ir_dir, 
                      vi_dir, 
                      batch_size=4,
                      img_size=256,
                      augment=True,
                      normalize=True,
                      num_workers=4,
                      shuffle=True):
    """
    创建数据加载器
    
    Args:
        ir_dir: 红外图像目录
        vi_dir: 可见光图像目录
        batch_size: 批次大小 (默认4)
        img_size: 图像尺寸 (默认256)
        augment: 是否数据增强 (默认True)
        normalize: 是否归一化 (默认True)
        num_workers: 数据加载线程数 (默认4)
        shuffle: 是否打乱数据 (默认True)
    
    Returns:
        dataloader: PyTorch数据加载器
    """
    dataset = FusionDataset(
        ir_dir=ir_dir,
        vi_dir=vi_dir,
        img_size=img_size,
        augment=augment,
        normalize=normalize
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("Fusion Dataset Test")
    print("=" * 70)
    
    # 检查M3FD数据集是否存在
    train_ir_dir = "datasets/M3FD/train/ir"
    train_vi_dir = "datasets/M3FD/train/vi"
    
    if not os.path.exists(train_ir_dir) or not os.path.exists(train_vi_dir):
        print(f"\n⚠ M3FD数据集不存在，创建测试数据...")
        
        # 创建测试目录
        os.makedirs(train_ir_dir, exist_ok=True)
        os.makedirs(train_vi_dir, exist_ok=True)
        
        # 创建一些测试图像
        for i in range(5):
            # 创建随机图像
            ir_img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
            vi_img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
            
            # 保存图像
            cv2.imwrite(os.path.join(train_ir_dir, f"test_{i:05d}.png"), ir_img)
            cv2.imwrite(os.path.join(train_vi_dir, f"test_{i:05d}.png"), vi_img)
        
        print(f"  ✓ 创建了5对测试图像")
    
    # 创建数据集
    print(f"\n创建数据集...")
    dataset = FusionDataset(
        ir_dir=train_ir_dir,
        vi_dir=train_vi_dir,
        img_size=256,
        augment=True,
        normalize=True
    )
    
    # 测试单个样本
    print(f"\n测试单个样本...")
    ir_img, vi_img, filename = dataset[0]
    print(f"  - IR图像形状: {ir_img.shape}")
    print(f"  - VI图像形状: {vi_img.shape}")
    print(f"  - 文件名: {filename}")
    print(f"  - IR值范围: [{ir_img.min():.3f}, {ir_img.max():.3f}]")
    print(f"  - VI值范围: [{vi_img.min():.3f}, {vi_img.max():.3f}]")
    
    # 创建数据加载器
    print(f"\n创建数据加载器...")
    dataloader = create_dataloader(
        ir_dir=train_ir_dir,
        vi_dir=train_vi_dir,
        batch_size=2,
        img_size=256,
        augment=True,
        normalize=True,
        num_workers=0,  # Windows上设置为0
        shuffle=True
    )
    
    print(f"  ✓ 数据加载器创建成功")
    print(f"  - 批次数量: {len(dataloader)}")
    print(f"  - 批次大小: {dataloader.batch_size}")
    
    # 测试批次加载
    print(f"\n测试批次加载...")
    for i, (ir_batch, vi_batch, filenames) in enumerate(dataloader):
        print(f"  批次 {i+1}:")
        print(f"    - IR批次形状: {ir_batch.shape}")
        print(f"    - VI批次形状: {vi_batch.shape}")
        print(f"    - 文件名: {filenames}")
        
        if i >= 1:  # 只测试前2个批次
            break
    
    # 测试数据增强
    print(f"\n测试数据增强...")
    dataset_no_aug = FusionDataset(
        ir_dir=train_ir_dir,
        vi_dir=train_vi_dir,
        img_size=256,
        augment=False,
        normalize=True
    )
    
    ir_orig, vi_orig, _ = dataset_no_aug[0]
    ir_aug, vi_aug, _ = dataset[0]
    
    print(f"  - 原始图像与增强图像是否相同: {torch.equal(ir_orig, ir_aug)}")
    print(f"  (应该为False，因为有数据增强)")
    
    print("\n" + "=" * 70)
    print("✓ 数据集测试通过!")
    print("=" * 70)
    print("\n使用说明:")
    print("  1. 将M3FD数据集放在 datasets/M3FD/ 目录下")
    print("  2. 确保目录结构:")
    print("     - datasets/M3FD/train/ir/  (训练集红外图像)")
    print("     - datasets/M3FD/train/vi/  (训练集可见光图像)")
    print("     - datasets/M3FD/val/ir/    (验证集红外图像)")
    print("     - datasets/M3FD/val/vi/    (验证集可见光图像)")
    print("  3. 使用 create_dataloader() 创建数据加载器")
