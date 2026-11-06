---
id: guidedquant
title: "GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance"
type: paper
url: https://proceedings.mlr.press/v267/kim25d.html
status: unread
summary: 利用端到端损失的梯度信息引导训练后量化，在输出通道内保留权重间依赖，覆盖仅权重标量、向量及权重激活联合量化，并提出非均匀标量量化算法 LNQ。
tags:
  - topic/quantization
  - target/weight
  - target/activation
  - scheme/weight-only
  - scheme/weight-activation
  - scheme/ptq
  - scheme/nonuniform
  - availability/open-source
links:
  arxiv: https://arxiv.org/abs/2505.07004
  openreview: https://openreview.net/forum?id=ZawsPjlIGu
  code: https://github.com/snu-mllab/GuidedQuant
code_review:
  checked_on: '2026-11-06'
  status: open-source
  license: MIT
  license_url: https://raw.githubusercontent.com/snu-mllab/GuidedQuant/main/LICENSE
  evidence:
  - https://github.com/snu-mllab/GuidedQuant/blob/main/README.md
  - https://raw.githubusercontent.com/snu-mllab/GuidedQuant/main/layerwise_nuq.py
  - https://raw.githubusercontent.com/snu-mllab/GuidedQuant/main/any_precision/quantization/layerwise_quantize.py
  - https://raw.githubusercontent.com/snu-mllab/GuidedQuant/main/LICENSE
  note: 官方 README 对应本论文，已核实逐层非均匀量化目标与优化实现及根目录 MIT 许可；未运行代码。
attachments:
  - url: https://raw.githubusercontent.com/mlresearch/v267/main/assets/kim25d/kim25d.pdf
    path: resources/guidedquant-icml2025.pdf
authors:
  - Jinuk Kim
  - Marwa El Halabi
  - Wonpyo Park
  - Clemens JS Schaefer
  - Deokjae Lee
  - Yeonhong Park
  - Jae W. Lee
  - Hyun Oh Song
year: 2025
venue: ICML
---

# GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
