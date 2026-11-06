---
id: owq
title: "OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models"
type: paper
url: https://ojs.aaai.org/index.php/AAAI/article/view/29237
status: read
summary: 敏感权重保留高精度，比较混合精度的存储和计算代价。
tags:
  - topic/quantization
  - target/weight
  - scheme/weight-only
  - scheme/mixed-precision
links:
  code: https://github.com/xvyaward/owq
code_review:
  checked_on: 2026-11-06
  status: source-available
  evidence:
    - https://raw.githubusercontent.com/xvyaward/owq/main/README.md
    - https://github.com/xvyaward/owq/tree/main/owq
    - https://raw.githubusercontent.com/xvyaward/owq/main/owq/recon.py
  note: 官方 README 对应本论文且重建量化实现可读；本次在根目录、README 和核心实现中未找到适用的开源许可正文，不以依赖 GPTQ 的许可代替本项目授权，故不加开源标签。GitHub 许可 API 限流，已改查仓库页面与源码。
attachments:
  - url: https://ojs.aaai.org/index.php/AAAI/article/download/29237/30335
    path: resources/owq.pdf
    source: zotero
authors:
  - Changhun Lee
  - Jungyu Jin
  - Taesu Kim
  - Hyungjun Kim
  - Eunhyeok Park
year: 2024
doi: 10.1609/aaai.v38i12.29237
venue: Proceedings of the AAAI Conference on Artificial Intelligence
---

# OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models
