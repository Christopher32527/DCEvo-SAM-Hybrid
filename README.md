# DCEvo-SAM Hybrid: 多模态图像融合与分割

[![License](https://img.shields.io/badge/License-Mixed-blue.svg)](LICENSE.md)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> 基于DCEvo和SAM的混合架构,用于红外+可见光图像融合与零样本分割

---

## 📋 项目简介

本项目将**DCEvo图像融合模型**与**SAM (Segment Anything Model)** 分割模型相结合,实现:
- ✅ 红外+可见光图像融合
- ✅ 零样本交互式分割
- ✅ 高质量分割掩码生成

### 系统架构

```
输入: 红外图像 + 可见光图像
  ↓
DCEvo融合 (DE_Encoder → 特征融合 → DE_Decoder)
  ↓
融合图像 + 融合特征
  ↓
SAM分割 (Image Encoder → Mask Decoder)
  ↓
输出: 融合图像 + 分割掩码
```

---

## 🎯 核心功能

- **多模态融合**: 融合红外和可见光图像,保留两者优势
- **零样本分割**: 无需训练即可分割任意物体
- **交互式提示**: 支持点击点和边界框提示
- **高质量输出**: 精细的分割边缘和融合效果

---

## 📦 代码来源声明

### 本项目基于以下开源项目:

#### 1. DCEvo (图像融合)
- **项目**: DCEvo - Discriminative Cross-Dimensional Evolutionary Learning for Infrared and Visible Image Fusion
- **作者**: Jinyuan Liu et al.
- **许可证**: [查看许可证](https://github.com/Beate-Suy-Zhang/DCEvo/blob/main/LICENSE)
- **项目地址**: https://github.com/Beate-Suy-Zhang/DCEvo
- **使用模块**:
  - `sleepnet.py` - 完整保留
  - `models/common.py` - 完整保留
  - `utils/` - 部分保留
  - 预训练权重

#### 2. SAM (分割模型)
- **项目**: Segment Anything (Meta AI)
- **许可证**: Apache License 2.0
- **项目地址**: https://github.com/facebookresearch/segment-anything
- **使用模块**: SAM完整模型架构和权重

### 本项目原创部分:

- ✨ `models/dcevo_sam_hybrid.py` - 混合架构设计
- ✨ `inference_hybrid.py` - 推理脚本
- ✨ 特征适配层和融合策略
- ✨ 完整文档和使用指南

**详细许可信息请查看**: [LICENSE.md](LICENSE.md)

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装PyTorch (CUDA 11.7)
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu117

# 安装依赖
pip install timm einops opencv-python scikit-image seaborn kornia pygad ultralytics matplotlib

# 安装SAM
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

### 2. 下载权重

- **DCEvo权重**: `ckpt/DCEvo_fusion.pth` (项目自带)
- **SAM权重**: [下载ViT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) → 放到 `ckpt/`

### 3. 运行测试

```bash
# 测试模型
python models/dcevo_sam_hybrid.py

# 对M3FD数据集推理
python inference_hybrid.py --dataset M3FD --use_center_point --output_dir results/m3fd
```

---

## 📖 详细文档

- [完整使用指南](DCEvo-SAM使用指南.md) - 详细的安装、使用和API文档
- [架构修改记录](DCEvo-SAM混合架构修改记录.md) - 技术细节和设计思路
- [许可证声明](LICENSE.md) - 代码来源和引用要求

---

## 📊 性能指标

### 推理速度 (RTX 3090)
| 图像尺寸 | 融合时间 | 分割时间 | 总时间 |
|----------|----------|----------|--------|
| 1024x1024 | ~80ms | ~300ms | ~380ms |

### 显存占用
| SAM模型 | 显存占用 |
|---------|----------|
| ViT-B | ~4GB |
| ViT-L | ~6GB |

---

## 🎓 引用

如果使用本项目,请引用:

```bibtex
@article{liu2024dcevo,
  title={DCEvo: Discriminative Cross-Dimensional Evolutionary Learning for Infrared and Visible Image Fusion},
  author={Liu, Jinyuan and others},
  journal={[期刊]},
  year={2024},
  note={GitHub: https://github.com/Beate-Suy-Zhang/DCEvo}
}

@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and others},
  journal={arXiv:2304.02643},
  year={2023}
}
```

---

## 📁 项目结构

```
DCEvo-main/
├── models/
│   ├── dcevo_sam_hybrid.py      # 混合模型 (原创)
│   ├── common.py                # DCEvo模块 (来自DCEvo)
│   └── yolo.py                  # YOLO模块 (来自DCEvo)
├── ckpt/
│   ├── DCEvo_fusion.pth         # DCEvo权重
│   └── sam_vit_b_01ec64.pth     # SAM权重
├── sleepnet.py                  # DCEvo核心 (来自DCEvo)
├── inference_hybrid.py          # 推理脚本 (原创)
├── utils/                       # 工具函数 (来自DCEvo)
├── datasets/                    # 数据集目录
├── results/                     # 结果输出
├── README.md                    # 本文件
├── LICENSE.md                   # 许可证声明
├── DCEvo-SAM使用指南.md         # 使用文档
└── DCEvo-SAM混合架构修改记录.md # 技术文档
```

---

## 🤝 贡献指南

欢迎贡献!请遵循以下步骤:

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## ⚠️ 注意事项

1. **学术诚信**: 使用本项目请正确引用DCEvo和SAM的原始论文
2. **许可证**: 遵守DCEvo和SAM的开源许可证
3. **仅供研究**: 本项目仅供学术研究和教育用途
4. **权重文件**: 大文件不包含在仓库中,需单独下载

---

## 📧 联系方式

- **Issues**: [GitHub Issues](https://github.com/Christopher32527/DCEvo-SAM-Hybrid/issues)
- **Email**: 2546507517@qq.com
- **原项目**:
  - DCEvo: https://github.com/Beate-Suy-Zhang/DCEvo
  - SAM: https://github.com/facebookresearch/segment-anything

---

## 🙏 致谢

感谢以下项目的开源贡献:
- **DCEvo团队** (Jinyuan Liu et al.): 提供优秀的图像融合模型
- **Meta AI**: 开源SAM分割模型
- **PyTorch团队**: 提供深度学习框架

---

## 📝 更新日志

### v1.0.0 (2025-01-25)
- ✅ 初始版本发布
- ✅ 集成DCEvo融合模块
- ✅ 集成SAM分割模块
- ✅ 实现混合架构
- ✅ 提供完整文档

---

**⭐ 如果觉得有用,请给个Star!**

---

<p align="center">
  Made with ❤️ for Multi-Modal Image Processing
</p>
