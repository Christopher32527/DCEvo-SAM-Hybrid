"""
DCEvo-FAM医学图像融合训练脚本
支持CT+MRI融合任务

作者: Christopher32527
邮箱: 2546507517@qq.com
"""

import os
import sys
import yaml
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入模型和工具
from models.dcevo_fam_hybrid import DCEvoFAMHybrid
from models.losses import FusionLoss
from datasets.medical_fusion_dataset import create_medical_dataloader, split_patients_train_val_test
from utils.metrics import calculate_metrics
from utils.visualization import save_fusion_comparison


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=20, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config):
    """设置设备"""
    device_type = config['device']['type']
    
    if device_type == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_type == 'cuda':
        if not torch.cuda.is_available():
            print("⚠ CUDA不可用，使用CPU")
            device = torch.device('cpu')
        else:
            gpu_id = config['device']['gpu_id']
            device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    
    return device


def create_save_dir(config):
    """创建保存目录"""
    save_dir = config['save']['save_dir']
    exp_name = config['save']['exp_name']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    exp_dir = os.path.join(save_dir, f"{exp_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    
    return exp_dir


def train_epoch(model, dataloader, criterion, optimizer, device, config, epoch, writer=None):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    total_ssim = 0.0
    total_gradient = 0.0
    total_intensity = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (img_a, img_b, info) in enumerate(pbar):
        # 移动数据到设备
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        
        # 前向传播
        fused = model(img_a, img_b)
        
        # 计算损失
        loss, components = criterion(fused, img_a, img_b, return_components=True)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if config['training']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        
        optimizer.step()
        
        # 累计损失
        total_loss += components['total']
        total_ssim += components['ssim']
        total_gradient += components['gradient']
        total_intensity += components['intensity']
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{components['total']:.4f}",
            'ssim': f"{components['ssim']:.4f}",
            'grad': f"{components['gradient']:.4f}"
        })
        
        # 记录到TensorBoard
        if writer and batch_idx % config['logging']['print_interval'] == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', components['total'], global_step)
            writer.add_scalar('Train/SSIM_Loss', components['ssim'], global_step)
            writer.add_scalar('Train/Gradient_Loss', components['gradient'], global_step)
            writer.add_scalar('Train/Intensity_Loss', components['intensity'], global_step)
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    avg_gradient = total_gradient / len(dataloader)
    avg_intensity = total_intensity / len(dataloader)
    
    return {
        'loss': avg_loss,
        'ssim': avg_ssim,
        'gradient': avg_gradient,
        'intensity': avg_intensity
    }


def validate_epoch(model, dataloader, criterion, device, config, epoch, exp_dir, writer=None):
    """验证一个epoch"""
    model.eval()
    
    total_loss = 0.0
    total_ssim = 0.0
    total_gradient = 0.0
    total_intensity = 0.0
    
    # 评估指标
    all_metrics = {
        'EN': [],
        'SF': [],
        'SSIM': [],
        'MI': []
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]')
    
    with torch.no_grad():
        for batch_idx, (img_a, img_b, info) in enumerate(pbar):
            # 移动数据到设备
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            
            # 前向传播
            fused = model(img_a, img_b)
            
            # 计算损失
            loss, components = criterion(fused, img_a, img_b, return_components=True)
            
            # 累计损失
            total_loss += components['total']
            total_ssim += components['ssim']
            total_gradient += components['gradient']
            total_intensity += components['intensity']
            
            # 计算评估指标
            metrics = calculate_metrics(fused, img_a, img_b)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
            
            # 保存验证图像
            if config['validation']['save_images'] and batch_idx < config['validation']['num_save_images']:
                save_path = os.path.join(exp_dir, 'images', f'epoch{epoch+1:03d}_batch{batch_idx:03d}.png')
                save_fusion_comparison(img_a[0], img_b[0], fused[0], save_path)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{components['total']:.4f}",
                'EN': f"{metrics['EN']:.4f}",
                'SF': f"{metrics['SF']:.4f}"
            })
    
    # 计算平均损失和指标
    avg_loss = total_loss / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    avg_gradient = total_gradient / len(dataloader)
    avg_intensity = total_intensity / len(dataloader)
    
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    # 记录到TensorBoard
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/SSIM_Loss', avg_ssim, epoch)
        writer.add_scalar('Val/Gradient_Loss', avg_gradient, epoch)
        writer.add_scalar('Val/Intensity_Loss', avg_intensity, epoch)
        
        for key, value in avg_metrics.items():
            writer.add_scalar(f'Val/Metrics/{key}', value, epoch)
    
    return {
        'loss': avg_loss,
        'ssim': avg_ssim,
        'gradient': avg_gradient,
        'intensity': avg_intensity,
        'metrics': avg_metrics
    }


def save_checkpoint(model, optimizer, epoch, best_loss, exp_dir, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }
    
    if is_best:
        save_path = os.path.join(exp_dir, 'checkpoints', 'best_model.pth')
        torch.save(checkpoint, save_path)
        print(f"  ✓ 保存最佳模型: {save_path}")
    else:
        save_path = os.path.join(exp_dir, 'checkpoints', f'epoch_{epoch+1:03d}.pth')
        torch.save(checkpoint, save_path)
        print(f"  ✓ 保存检查点: {save_path}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DCEvo-FAM医学图像融合训练')
    parser.add_argument('--config', type=str, default='configs/medical_fusion_config.yaml',
                       help='配置文件路径')
    args = parser.parse_args()
    
    # 清除GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU缓存已清除")
    
    # 加载配置
    print("=" * 70)
    print("DCEvo-FAM Medical Image Fusion Training")
    print("=" * 70)
    print(f"\n加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 设置设备
    device = setup_device(config)
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"  - GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 创建保存目录
    exp_dir = create_save_dir(config)
    print(f"实验目录: {exp_dir}")
    
    # 保存配置文件
    config_save_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    # 划分数据集
    print(f"\n划分数据集...")
    train_ids, val_ids, test_ids = split_patients_train_val_test(
        data_root=config['data']['data_root'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        seed=config['data']['seed']
    )
    
    # 创建数据加载器
    print(f"\n创建数据加载器...")
    train_loader = create_medical_dataloader(
        data_root=config['data']['data_root'],
        modality_a=config['data']['modality_a'],
        modality_b=config['data']['modality_b'],
        patient_ids=train_ids,
        batch_size=config['training']['batch_size'],
        img_size=config['data']['img_size'],
        augment=config['data']['augment'],
        normalize=config['data']['normalize'],
        num_workers=config['training']['num_workers'],
        shuffle=True
    )
    
    val_loader = create_medical_dataloader(
        data_root=config['data']['data_root'],
        modality_a=config['data']['modality_a'],
        modality_b=config['data']['modality_b'],
        patient_ids=val_ids,
        batch_size=config['training']['batch_size'],
        img_size=config['data']['img_size'],
        augment=False,
        normalize=config['data']['normalize'],
        num_workers=config['training']['num_workers'],
        shuffle=False
    )
    
    print(f"  ✓ 训练集批次数: {len(train_loader)}")
    print(f"  ✓ 验证集批次数: {len(val_loader)}")
    
    # 创建模型
    print(f"\n创建模型...")
    model = DCEvoFAMHybrid(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        dim=config['model']['dim'],
        fam_feature_dim=config['model']['fam_feature_dim'],
        fam_cutoff=config['model']['fam_cutoff'],
        num_blocks=config['model']['num_blocks']
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ 模型参数量: {total_params:,}")
    print(f"  ✓ 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 创建损失函数
    criterion = FusionLoss(
        ssim_weight=config['loss']['ssim_weight'],
        gradient_weight=config['loss']['gradient_weight'],
        intensity_weight=config['loss']['intensity_weight'],
        window_size=config['loss']['window_size']
    ).to(device)
    
    # 创建优化器
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    
    # 创建学习率调度器
    if config['training']['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['step_size'],
            gamma=config['training']['gamma']
        )
    elif config['training']['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config['training']['patience'],
            factor=config['training']['factor']
        )
    
    # 创建TensorBoard
    writer = None
    if config['logging']['use_tensorboard']:
        log_dir = os.path.join(exp_dir, 'logs')
        writer = SummaryWriter(log_dir)
        print(f"  ✓ TensorBoard日志: {log_dir}")
    
    # 早停机制
    early_stopping = None
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        )
    
    # 训练循环
    print(f"\n开始训练...")
    print("=" * 70)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config['training']['epochs']):
        # 训练
        train_results = train_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch, writer
        )
        
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print(f"  Train Loss: {train_results['loss']:.4f} "
              f"(SSIM: {train_results['ssim']:.4f}, "
              f"Grad: {train_results['gradient']:.4f}, "
              f"Intensity: {train_results['intensity']:.4f})")
        
        # 验证
        if (epoch + 1) % config['validation']['val_interval'] == 0:
            val_results = validate_epoch(
                model, val_loader, criterion, device, config, epoch, exp_dir, writer
            )
            
            print(f"  Val Loss: {val_results['loss']:.4f} "
                  f"(SSIM: {val_results['ssim']:.4f}, "
                  f"Grad: {val_results['gradient']:.4f}, "
                  f"Intensity: {val_results['intensity']:.4f})")
            print(f"  Metrics: EN={val_results['metrics']['EN']:.4f}, "
                  f"SF={val_results['metrics']['SF']:.4f}, "
                  f"SSIM={val_results['metrics']['SSIM']:.4f}, "
                  f"MI={val_results['metrics']['MI']:.4f}")
            
            # 保存最佳模型
            if val_results['loss'] < best_val_loss:
                best_val_loss = val_results['loss']
                if config['save']['save_best']:
                    save_checkpoint(model, optimizer, epoch, best_val_loss, exp_dir, is_best=True)
            
            # 早停检查
            if early_stopping:
                early_stopping(val_results['loss'])
                if early_stopping.early_stop:
                    print(f"\n早停触发，停止训练")
                    break
        
        # 保存检查点
        if (epoch + 1) % config['save']['save_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, best_val_loss, exp_dir, is_best=False)
        
        # 更新学习率
        if config['training']['lr_scheduler'] == 'plateau':
            scheduler.step(val_results['loss'])
        else:
            scheduler.step()
        
        # 打印学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")
    
    # 保存最后模型
    if config['save']['save_last']:
        save_checkpoint(model, optimizer, epoch, best_val_loss, exp_dir, is_best=False)
    
    # 训练完成
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✓ 训练完成!")
    print(f"  - 总时间: {total_time/3600:.2f} 小时")
    print(f"  - 最佳验证损失: {best_val_loss:.4f}")
    print(f"  - 实验目录: {exp_dir}")
    print("=" * 70)
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
