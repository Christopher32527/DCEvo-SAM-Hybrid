# 许可证声明

## 项目说明

本项目是基于以下开源项目的二次开发和集成:

### 1. DCEvo (图像融合部分)
- **原项目**: DCEvo - Discriminative Cross-Dimensional Evolutionary Learning for Infrared and Visible Image Fusion
- **作者**: Jinyuan Liu et al.
- **许可证**: [查看许可证](https://github.com/Beate-Suy-Zhang/DCEvo/blob/main/LICENSE)
- **项目地址**: https://github.com/Beate-Suy-Zhang/DCEvo
- **使用部分**: 
  - `sleepnet.py` - DE_Encoder, DE_Decoder, LowFreqExtractor, HighFreqExtractor
  - `models/common.py` - C2f, SPPF等模块
  - `utils/` - 部分工具函数
  - 预训练权重: `DCEvo_fusion.pth`

### 2. SAM (Segment Anything Model)
- **原项目**: Segment Anything
- **作者**: Meta AI Research
- **许可证**: Apache License 2.0
- **项目地址**: https://github.com/facebookresearch/segment-anything
- **论文**: Kirillov et al., "Segment Anything", ICCV 2023
- **使用部分**:
  - SAM模型架构和预训练权重
  - 图像编码器、提示编码器、掩码解码器

## 本项目的贡献

### 原创部分
- `models/dcevo_sam_hybrid.py` - DCEvo与SAM的混合架构设计
- `inference_hybrid.py` - 混合模型推理脚本
- 特征适配层设计
- 多模态特征融合策略
- 文档和使用指南

### 修改部分
- 对DCEvo模块的适配和封装
- 对SAM模块的集成和调用
- 图像预处理和后处理流程

## 使用声明

本项目遵循以下原则:

1. **学术诚信**: 明确标注所有引用的代码来源
2. **开源协议**: 遵守原项目的开源许可证
3. **引用要求**: 使用本项目请同时引用DCEvo和SAM的原始论文

## 引用格式

如果使用本项目,请引用:

```bibtex
% DCEvo原始论文
@article{liu2024dcevo,
  title={DCEvo: Discriminative Cross-Dimensional Evolutionary Learning for Infrared and Visible Image Fusion},
  author={Liu, Jinyuan and others},
  journal={[期刊名]},
  year={2024},
  note={GitHub: https://github.com/Beate-Suy-Zhang/DCEvo}
}

% SAM原始论文
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

% 本项目 (如果发表)
@misc{dcevo_sam_hybrid2025,
  title={DCEvo-SAM: A Hybrid Architecture for Multi-Modal Image Fusion and Segmentation},
  author={[你的名字]},
  year={2025},
  howpublished={\url{https://github.com/你的用户名/DCEvo-SAM-Hybrid}}
}
```

## 免责声明

本项目仅供学术研究和教育用途。使用者需自行承担使用本项目的风险。

## 联系方式

如有任何问题或建议,请通过以下方式联系:
- GitHub Issues: https://github.com/Christopher32527/DCEvo-SAM-Hybrid/issues
- Email: 2546507517@qq.com

---

**最后更新**: 2025-01-25
