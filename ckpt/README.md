# 模型权重文件

本目录需要放置以下权重文件:

## 1. DCEvo融合权重

### DCEvo官方权重下载
所有DCEvo预训练权重可从官方仓库下载:
- **GitHub仓库**: https://github.com/Beate-Suy-Zhang/DCEvo/tree/main/ckpt

### 可用的权重文件:
- `DCEvo_fusion.pth` - DCEvo图像融合模型权重
- `DCEvo_fusion_branch.pth` - DCEvo融合分支权重
- `DCEvo_detect_branch.pt` - DCEvo检测分支权重
- `pretrained_yolov8s.pt` - 预训练YOLOv8s权重

**下载方式**:
1. 访问: https://github.com/Beate-Suy-Zhang/DCEvo/tree/main/ckpt
2. 点击需要的权重文件
3. 点击 "Download" 按钮下载
4. 将下载的文件放到本目录 (`DCEvo-SAM-Hybrid/ckpt/`)

## 2. SAM分割权重 (选择一个)

### ViT-B (推荐)
- 文件名: `sam_vit_b_01ec64.pth`
- 大小: 91 MB
- 下载: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

### ViT-L (高精度)
- 文件名: `sam_vit_l_0b3195.pth`
- 大小: 375 MB
- 下载: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

### ViT-H (最高精度)
- 文件名: `sam_vit_h_4b8939.pth`
- 大小: 2.6 GB
- 下载: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

## 下载后放置位置
```
DCEvo-SAM-Hybrid/
└── ckpt/
    ├── DCEvo_fusion.pth          ← 必需 (医学图像融合训练)
    ├── DCEvo_fusion_branch.pth   ← 可选
    ├── DCEvo_detect_branch.pt    ← 可选
    ├── pretrained_yolov8s.pt     ← 可选
    └── sam_vit_b_01ec64.pth      ← 必需 (如果使用SAM分割)
```

## 注意事项
- ⚠️ 权重文件较大，不包含在Git仓库中
- ✅ 下载后请确保文件名正确
- ✅ 医学图像融合训练至少需要 `DCEvo_fusion.pth`
- ✅ 如果使用SAM分割功能，还需要下载SAM权重
