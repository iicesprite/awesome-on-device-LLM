---
id: hcinfer
title: "HCInfer: An Efficient Inference System via Error Compensation for Resource-Constrained Devices"
type: paper
url: https://arxiv.org/abs/2605.05819
status: read
summary: 将量化主干放在 GPU、低秩残差补偿放在 CPU，通过共享补偿窗口和敏感度秩分配恢复质量；需留意未公开代码、分配证明与实验表格的口径问题。
tags:
  - topic/quantization
  - topic/inference
  - topic/memory
  - target/weight
  - scheme/weight-only
  - scheme/ptq
  - method/gptq
  - architecture/moe
  - system/error-compensation
  - system/heterogeneous
  - system/offloading
  - system/pipelining
  - system/dynamic-scheduling
  - memory/discrete
  - hardware/nvidia-dgpu
  - hardware/x86
  - backend/cpu
  - backend/cuda
  - stage/prefill
  - stage/decode
links:
  html: https://arxiv.org/html/2605.05819v1
attachments:
  - url: https://arxiv.org/pdf/2605.05819v1
    path: resources/hcinfer-2605.05819v1.pdf
    version: arXiv:2605.05819v1
authors:
  - Shen Xu
  - Xiangwen Zhuge
  - Zhe Xu
  - Yingkun Hu
  - Zheng Yang
  - Yunhao Liu
year: 2026
doi: 10.48550/arXiv.2605.05819
first_published: 2026-05-07
source_checked_on: 2026-10-30
tag_note: 标签依据论文正文及附录；未确认公开源码，故未添加 packed-weights/kernel 等实现核验标签。主干为 GPTQ INT4/INT3，论文未完整交代激活与补偿矩阵 dtype，不强填 W4A16。与 RTSS 2025 同名 HCInfer 是不同论文。
code_review:
  checked_on: 2026-11-06
  status: not-found
  evidence:
    - https://arxiv.org/abs/2605.05819
    - https://arxiv.org/html/2605.05819v1
  note: 论文代码开放检查项回答 No，说明代码仍受实验室知识产权审查、计划录用后发布。本次检索未找到可确认的后续公开实现；不与 RTSS 2025 的同名方法混用，不标记开源。
---

# HCInfer: An Efficient Inference System via Error Compensation for Resource-Constrained Devices

[阅读笔记](notes.md)
