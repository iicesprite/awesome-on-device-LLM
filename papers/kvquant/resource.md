---
id: kvquant
title: "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization"
type: paper
url: https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html
status: read
summary: 非均匀及离群值感知 KV 量化与定制 CUDA 内核。
tags:
  - topic/quantization
  - target/kv-cache
  - granularity/per-channel
  - backend/cuda
  - method/kvquant
links:
  code: https://github.com/SqueezeAILab/KVQuant
code_review:
  checked_on: 2026-11-06
  status: source-available
  evidence:
    - https://raw.githubusercontent.com/SqueezeAILab/KVQuant/main/README.md
    - https://github.com/SqueezeAILab/KVQuant/tree/main/quant
    - https://raw.githubusercontent.com/SqueezeAILab/KVQuant/main/quant/kvquant/simquant_module_quantizer.py
    - https://raw.githubusercontent.com/SqueezeAILab/KVQuant/main/quant/pyproject.toml
  note: 官方 README 对应本论文且量化实现可读；本次未找到适用于该实现的许可正文，包元数据中的 Apache 分类不足以核实授权，故不加开源标签。GitHub 许可 API 限流，已改查仓库页面与源码。
attachments:
  - url: https://proceedings.neurips.cc/paper_files/paper/2024/file/028fcbcf85435d39a40c4d61b42c99a4-Paper-Conference.pdf
    path: resources/kvquant.pdf
    source: zotero
authors:
  - Coleman Hooper
  - Sehoon Kim
  - Hiva Mohammadzadeh
  - Michael W. Mahoney
  - Yakun S. Shao
  - Kurt Keutzer
  - Amir Gholami
year: 2024
doi: 10.52202/079017-0040
venue: Advances in Neural Information Processing Systems
---

# KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization
