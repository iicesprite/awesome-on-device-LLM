---
id: milo
title: 'MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators'
type: paper
url: https://openreview.net/forum?id=NXVXiJmhe1
status: unread
summary: 低秩补偿恢复低比特 MoE 精度，并设计 Tensor Core 友好的 3-bit 内核。
tags:
- topic/quantization
- architecture/moe
- target/weight
- scheme/weight-only
- topic/inference
- availability/open-source
links:
  code: https://github.com/Supercomputing-System-AI-Lab/MiLo
attachments:
- url: https://openreview.net/pdf?id=NXVXiJmhe1
  path: resources/milo.pdf
  source: zotero
authors:
- Beichen Huang
- Yueming Yuan
- Zelei Shao
- Minjia Zhang
year: 2025
venue: Eighth Conference on Machine Learning and Systems
code_review:
  checked_on: '2026-11-06'
  status: open-source
  license: MIT
  license_url: https://raw.githubusercontent.com/Supercomputing-System-AI-Lab/MiLo/main/LICENSE
  evidence:
  - https://github.com/Supercomputing-System-AI-Lab/MiLo
  - https://raw.githubusercontent.com/Supercomputing-System-AI-Lab/MiLo/main/LICENSE
  - https://raw.githubusercontent.com/Supercomputing-System-AI-Lab/MiLo/main/MiLo_compress_Mixtral.py
  - https://raw.githubusercontent.com/Supercomputing-System-AI-Lab/MiLo/main/MiLo/core/quantize.py
  note: 官方 README 的完整题名及四位作者与本论文一致，区别于其他同名量化方法；已核对 INT3 压缩入口、量化及打包核心实现和根目录 MIT 许可正文。未运行代码。
---

# MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators
