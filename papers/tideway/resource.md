---
id: tideway
title: "Tideway: Phase-Aware Tiered Memory Management for MoE Inference on Consumer Devices"
type: paper
url: https://arxiv.org/abs/2611.14848
status: unread
summary: 面向显存和主存受限的消费级 PC，Tideway 将 GPU 显存和主机内存作为一个逻辑池，以 NVMe SSD 存放放不下的 MoE 专家权重；预填充用双缓冲让权重传输与计算重叠，解码用专家缓存与分段 CUDA Graph 降低换页开销。
tags:
  - topic/inference
  - topic/memory
  - architecture/moe
  - precision/fp4
  - system/offloading
  - system/memory-layout
  - system/pipelining
  - system/expert-caching
  - workload/agent-serving
  - workload/long-context
  - stage/prefill
  - stage/decode
  - memory/discrete
  - hardware/nvidia-dgpu
  - backend/cuda
  - availability/open-source
links:
  html: https://arxiv.org/html/2611.14848v1
  code: https://github.com/Afterglow231/Tideway
attachments:
  - url: https://arxiv.org/pdf/2611.14848v1
    path: resources/tideway-2611.14848v1.pdf
    version: arXiv:2611.14848v1
authors:
  - Yifei Wang
  - Tsun-yu Yang
  - Yechen Xu
  - Hancheng Ye
  - Qinsi Wang
  - Baijun Zhou
  - Tingjun Chen
  - 'Hai "Helen" Li'
  - Danyang Zhuo
  - Yiran Chen
year: 2026
doi: 10.48550/arXiv.2611.14848
source_checked_on: '2026-11-20'
publication_note: 目前仅见 arXiv v1，提交于 2026-11-10，正文标注为审稿中的预印本；所查来源未见撤稿或更正。
tag_note: 论文在 RTX 5090 独立显存平台上测量最长 128K token 输入并回放 TraceLab 多轮编码智能体请求。FP4 对应评测使用的原生 FP4 DeepSeek-V4-Flash，不代表论文提出量化方法；逻辑内存池不等于统一物理内存。
code_review:
  checked_on: '2026-11-20'
  status: open-source
  license: Apache-2.0
  license_url: https://raw.githubusercontent.com/Afterglow231/Tideway/main/LICENSE
  evidence:
    - https://github.com/Afterglow231/Tideway
    - https://raw.githubusercontent.com/Afterglow231/Tideway/main/NOTICE
    - https://raw.githubusercontent.com/Afterglow231/Tideway/main/src/tideway/offload/phase_policy.py
    - https://raw.githubusercontent.com/Afterglow231/Tideway/main/src/tideway/offload/paged_decode_graph.py
  note: 论文与仓库互链；已核对阶段策略源码、面向 BF16 的分段解码图源码及根许可证正文。后端与 FreeToken 0.1.2 组合使用，未运行或复核性能。
---

# Tideway: Phase-Aware Tiered Memory Management for MoE Inference on Consumer Devices

[阅读笔记](notes.md)
