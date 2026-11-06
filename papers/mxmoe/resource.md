---
id: mxmoe
title: "MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design"
type: paper
url: https://openreview.net/forum?id=pXoZLGMNDm
status: read
summary: 联合精度分配、专家激活与硬件资源，生成混合精度 GroupGEMM 内核。
tags:
  - topic/quantization
  - architecture/moe
  - scheme/mixed-precision
  - scheme/weight-activation
  - topic/inference
links:
  code: https://github.com/cat538/MxMoE
attachments:
  - url: https://openreview.net/pdf?id=pXoZLGMNDm
    path: resources/mxmoe.pdf
    source: zotero
  - url: https://arxiv.org/pdf/2505.05799
    path: resources/mxmoe-akffll3p.pdf
    source: zotero
    source_zotero_attachment_key: AKFFLL3P
    source_zotero_item_key: S2MCVR2T
authors:
  - Haojie Duanmu
  - Xiuhong Li
  - Zhihang Yuan
  - Size Zheng
  - Jiangfei Duan
  - Xingcheng Zhang
  - Dahua Lin
year: 2025
venue: Forty-second International Conference on Machine Learning
code_review:
  checked_on: 2026-11-06
  status: source-available
  evidence:
    - https://arxiv.org/html/2505.05799
    - https://github.com/cat538/MxMoE
    - https://github.com/cat538/MxMoE/tree/main/mxmoe
    - https://raw.githubusercontent.com/cat538/MxMoE/main/run_mxmoe_gg.py
    - https://raw.githubusercontent.com/cat538/MxMoE/main/mxmoe/kernels/compose_kernel.py
  note: 论文 HTML 直接指向该仓库，README 题名及作者一致；已核对混合精度 GroupGEMM 调用入口和实际内核生成逻辑。本次在根目录、README 及所查源码中未找到适用于项目的明确开源许可，不以第三方子模块许可替代，不添加开源标签。未运行代码。
---

# MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design
