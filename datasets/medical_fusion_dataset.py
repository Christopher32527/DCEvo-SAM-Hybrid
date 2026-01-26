"""
医学图像融合数据集加载器
支持Paired MRI and CT数据集（CT + T1-MRI / CT + T2-MRI / T1-MRI + T2-MRI）

作者: Christopher32527
邮箱: 2546507517@qq.com
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import glob
from pathlib import Path


class MedicalFusionDataset(Dataset):
    """
    医学图像融合数据集
    
    支持Paired MRI and CT数据集格式:
    - CT/PNG/Patient_XX/CT_PNG (N).png
    - T1-MRI/PNG/Patient_XX/T1_PNG (N).png
    - T2-MRI/PNG/Patient_XX/T2_PNG (N).png
    
    Args:
        data_root: 数据集根目录
        modality_a: 模态A ('CT', 'T1-MRI', 'T2-MRI')
        modality_b: 模态B ('CT', 'T1-MRI', 'T2-MRI')
        patient_ids: 病人ID列表，例如 ['Patient_01', 'Patient_02', ...]
        img_size: 图像尺寸 (默认256)
        augment: 是否进行数据增强 (默认True)
        normalize: 是否归一化 (默认True)
    """
    
    def __init__(self, 
                 data_root,
                 modality_a='CT',
                 modality_b='T1-MRI',
                 patient_ids=None,
                 img_size=256,
                 augment=True,
                 normalize=True):
        super(MedicalFusionDataset, self).__init__()
        
        self.data_root = data_root
        self.modality_a = modality_a
        self.modality_b = modality_b
        self.img_size = img_size
        self.augment = augment
        self.normalize = normalize
        
        # 构建路径
        self.path_a = os.path.join(data_root, modality_a, 'PNG')
        self.path_b = os.path.join(data_root, modality_b, 'PNG')
        
        # 获取所有病人ID
        if patient_ids is None:
            patient_dirs = sorted(glob.glob(os.path.join(self.path_a, 'Patient_*')))
            self.patient_ids = [os.path.basename(p) for p in patient_dirs]
        else:
            self.patient_ids = patient_ids
        
        # 收集所有配对的图像路径
        self.image_pairs = []
        for patient_id in self.patient_ids:
            patient_dir_a = os.path.join(self.path_a, patient_id)
            patient_dir_b = os.path.join(self.path_b, patient_id)
            
            # 获取该病人的所有切片
            if modality_a == 'CT':
                files_a = sorted(glob.glob(os.path.join(patient_dir_a, 'CT_PNG*.png')),
                                key=lambda x: self._extract_slice_number(x))
            elif modality_a == 'T1-MRI':
                files_a = sorted(glob.glob(os.path.join(patient_dir_a, 'T1_PNG*.png')),
                                key=lambda x: self._extract_slice_number(x))
            else:  # T2-MRI
                files_a = sorted(glob.glob(os.path.join(patient_dir_a, 'T2_PNG*.png')),
                                key=lambda x: self._extract_slice_number(x))
            
            if modality_b == 'CT':
                files_b = sorted(glob.glob(os.path.join(patient_dir_b, 'CT_PNG*.png')),
                                key=lambda x: self._extract_slice_number(x))
            elif modality_b == 'T1-MRI':
                files_b = sorted(glob.glob(os.path.join(patient_dir_b, 'T1_PNG*.png')),
                                key=lambda x: self._extract_slice_number(x))
            else:  # T2-MRI
                files_b = sorted(glob.glob(os.path.join(patient_dir_b, 'T2_PNG*.png')),
                                key=lambda x: self._extract_slice_number(x))
            
            # 确保两个模态的切片数量一致
            assert len(files_a) == len(files_b), \
                f"病人 {patient_id} 的 {modality_a} 和 {modality_b} 切片数量不一致: {len(files_a)} vs {len(files_b)}"
            
            # 添加配对
            for file_a, file_b in zip(files_a, files_b):
                self.image_pairs.append((file_a, file_b, patient_id))
        
        print(f"✓ 加载医学图像融合数据集:")
        print(f"  - 数据根目录: {data_root}")
        print(f"  - 模态A: {modality_a}")
        print(f"  - 模态B: {modality_b}")
        print(f"  - 病人数量: {len(self.patient_ids)}")
        print(f"  - 图像对数量: {len(self.image_pairs)}")
        print(f"  - 图像尺寸: {img_size}x{img_size}")
        print(f"  - 数据增强: {augment}")
        print(f"  - 归一化: {normalize}")
    
    def _extract_slice_number(self, filepath):
        """从文件名中提取切片编号"""
        # 例如: "CT_PNG (15).png" -> 15
        filename = os.path.basename(filepath)
        try:
            num = int(filename.split('(')[1].split(')')[0])
            return num
        except:
            return 0
    
    def __len__(self):
        return len(self.image_pairs)
    
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
        # 使用min-max归一化
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img
    
    def augment_pair(self, img_a, img_b):
        """
        对图像对进行数据增强
        
        增强操作:
        - 随机水平翻转
        - 随机垂直翻转
        - 随机旋转90度
        """
        # 随机水平翻转
        if np.random.rand() > 0.5:
            img_a = np.fliplr(img_a)
            img_b = np.fliplr(img_b)
        
        # 随机垂直翻转
        if np.random.rand() > 0.5:
            img_a = np.flipud(img_a)
            img_b = np.flipud(img_b)
        
        # 随机旋转90度
        k = np.random.randint(0, 4)  # 0, 1, 2, 3 对应 0°, 90°, 180°, 270°
        if k > 0:
            img_a = np.rot90(img_a, k)
            img_b = np.rot90(img_b, k)
        
        return img_a, img_b
    
    def __getitem__(self, idx):
        """
        获取一对图像
        
        Returns:
            img_a: 模态A图像, shape [1, H, W]
            img_b: 模态B图像, shape [1, H, W]
            info: 信息字典 {'patient_id': ..., 'slice_idx': ...}
        """
        path_a, path_b, patient_id = self.image_pairs[idx]
        
        # 加载图像
        img_a = self.load_image(path_a)
        img_b = self.load_image(path_b)
        
        # 调整尺寸
        img_a = self.resize_image(img_a)
        img_b = self.resize_image(img_b)
        
        # 数据增强
        if self.augment:
            img_a, img_b = self.augment_pair(img_a, img_b)
        
        # 归一化
        if self.normalize:
            img_a = self.normalize_image(img_a)
            img_b = self.normalize_image(img_b)
        
        # 转换为张量
        img_a = torch.from_numpy(img_a).float().unsqueeze(0)  # [1, H, W]
        img_b = torch.from_numpy(img_b).float().unsqueeze(0)  # [1, H, W]
        
        # 提取切片编号
        slice_idx = self._extract_slice_number(path_a)
        
        info = {
            'patient_id': patient_id,
            'slice_idx': slice_idx,
            'path_a': path_a,
            'path_b': path_b
        }
        
        return img_a, img_b, info


def create_medical_dataloader(data_root,
                              modality_a='CT',
                              modality_b='T1-MRI',
                              patient_ids=None,
                              batch_size=4,
                              img_size=256,
                              augment=True,
                              normalize=True,
                              num_workers=4,
                              shuffle=True):
    """
    创建医学图像融合数据加载器
    
    Args:
        data_root: 数据集根目录
        modality_a: 模态A ('CT', 'T1-MRI', 'T2-MRI')
        modality_b: 模态B ('CT', 'T1-MRI', 'T2-MRI')
        patient_ids: 病人ID列表（None表示使用所有病人）
        batch_size: 批次大小 (默认4)
        img_size: 图像尺寸 (默认256)
        augment: 是否数据增强 (默认True)
        normalize: 是否归一化 (默认True)
        num_workers: 数据加载线程数 (默认4)
        shuffle: 是否打乱数据 (默认True)
    
    Returns:
        dataloader: PyTorch数据加载器
    """
    dataset = MedicalFusionDataset(
        data_root=data_root,
        modality_a=modality_a,
        modality_b=modality_b,
        patient_ids=patient_ids,
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


def split_patients_train_val_test(data_root, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=2025):
    """
    将病人划分为训练集、验证集和测试集
    
    Args:
        data_root: 数据集根目录
        train_ratio: 训练集比例 (默认0.7)
        val_ratio: 验证集比例 (默认0.15)
        test_ratio: 测试集比例 (默认0.15)
        seed: 随机种子
    
    Returns:
        train_ids: 训练集病人ID列表
        val_ids: 验证集病人ID列表
        test_ids: 测试集病人ID列表
    """
    # 获取所有病人ID
    ct_path = os.path.join(data_root, 'CT', 'PNG')
    patient_dirs = sorted(glob.glob(os.path.join(ct_path, 'Patient_*')))
    all_patient_ids = [os.path.basename(p) for p in patient_dirs]
    
    # 设置随机种子
    np.random.seed(seed)
    np.random.shuffle(all_patient_ids)
    
    # 计算划分点
    n_total = len(all_patient_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 划分
    train_ids = all_patient_ids[:n_train]
    val_ids = all_patient_ids[n_train:n_train+n_val]
    test_ids = all_patient_ids[n_train+n_val:]
    
    print(f"✓ 数据集划分:")
    print(f"  - 总病人数: {n_total}")
    print(f"  - 训练集: {len(train_ids)} 病人 ({train_ratio*100:.0f}%)")
    print(f"  - 验证集: {len(val_ids)} 病人 ({val_ratio*100:.0f}%)")
    print(f"  - 测试集: {len(test_ids)} 病人 ({test_ratio*100:.0f}%)")
    
    return train_ids, val_ids, test_ids


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("Medical Fusion Dataset Test")
    print("=" * 70)
    
    # 数据集路径
    data_root = "datasets/Paired MRI (T1, T2) and CT Scans Dataset"
    
    if not os.path.exists(data_root):
        print(f"\n⚠ 数据集不存在: {data_root}")
        print("请确保数据集路径正确")
        exit(1)
    
    # 划分数据集
    print(f"\n划分数据集...")
    train_ids, val_ids, test_ids = split_patients_train_val_test(
        data_root, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15
    )
    
    # 创建训练集
    print(f"\n创建训练集...")
    train_dataset = MedicalFusionDataset(
        data_root=data_root,
        modality_a='CT',
        modality_b='T1-MRI',
        patient_ids=train_ids,
        img_size=256,
        augment=True,
        normalize=True
    )
    
    # 测试单个样本
    print(f"\n测试单个样本...")
    img_a, img_b, info = train_dataset[0]
    print(f"  - 模态A图像形状: {img_a.shape}")
    print(f"  - 模态B图像形状: {img_b.shape}")
    print(f"  - 病人ID: {info['patient_id']}")
    print(f"  - 切片编号: {info['slice_idx']}")
    print(f"  - 模态A值范围: [{img_a.min():.3f}, {img_a.max():.3f}]")
    print(f"  - 模态B值范围: [{img_b.min():.3f}, {img_b.max():.3f}]")
    
    # 创建数据加载器
    print(f"\n创建数据加载器...")
    train_loader = create_medical_dataloader(
        data_root=data_root,
        modality_a='CT',
        modality_b='T1-MRI',
        patient_ids=train_ids,
        batch_size=2,
        img_size=256,
        augment=True,
        normalize=True,
        num_workers=0,  # Windows上设置为0
        shuffle=True
    )
    
    print(f"  ✓ 数据加载器创建成功")
    print(f"  - 批次数量: {len(train_loader)}")
    print(f"  - 批次大小: {train_loader.batch_size}")
    
    # 测试批次加载
    print(f"\n测试批次加载...")
    for i, (img_a_batch, img_b_batch, info_batch) in enumerate(train_loader):
        print(f"  批次 {i+1}:")
        print(f"    - 模态A批次形状: {img_a_batch.shape}")
        print(f"    - 模态B批次形状: {img_b_batch.shape}")
        print(f"    - 病人ID: {info_batch['patient_id']}")
        print(f"    - 切片编号: {info_batch['slice_idx']}")
        
        if i >= 1:  # 只测试前2个批次
            break
    
    print("\n" + "=" * 70)
    print("✓ 医学图像融合数据集测试通过!")
    print("=" * 70)
    print("\n使用说明:")
    print("  1. 数据集已自动划分为训练集/验证集/测试集")
    print("  2. 支持三种融合任务:")
    print("     - CT + T1-MRI 融合")
    print("     - CT + T2-MRI 融合")
    print("     - T1-MRI + T2-MRI 融合")
    print("  3. 使用 create_medical_dataloader() 创建数据加载器")
    print("  4. 使用 split_patients_train_val_test() 划分数据集")
