# 模型权重文件

本目录需要放置以下权重文件:

## 1. DCEvo融合权重
- 文件名: `DCEvo_fusion.pth`
- 大小: ~XX MB
- 说明: DCEvo图像融合模型权重
- 下载: [提供下载链接或联系方式]

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
DCEvo-main/
└── ckpt/
    ├── DCEvo_fusion.pth          ← 必需
    └── sam_vit_b_01ec64.pth      ← 必需 (选一个)
```
