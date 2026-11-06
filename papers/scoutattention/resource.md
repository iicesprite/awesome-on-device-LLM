---
id: scoutattention
title: "ScoutAttention: Efficient KV Cache Offloading via Layer-Ahead CPU Pre-computation for LLM Inference"
type: paper
url: https://arxiv.org/abs/2603.27138
status: unread
summary: 通过 CPU–GPU 协同的分块稀疏注意力、跨层提前 CPU 预计算和异步周期性召回，降低长上下文解码中 KV 缓存卸载的通信与 CPU 计算瓶颈。
tags:
  - topic/inference
  - topic/memory
  - system/offloading
  - system/heterogeneous
  - system/sparse-inference
  - stage/decode
  - workload/long-context
links:
  arxiv_version: https://arxiv.org/abs/2603.27138v1
attachments:
  - url: https://arxiv.org/pdf/2603.27138v1
    path: resources/scoutattention-2603.27138v1.pdf
    version: arXiv v1
authors:
  - Qiuyang Zhang
  - Kai Zhou
  - Ding Tang
  - Kai Lu
  - Cheng Li
  - Zhenyu Yang
  - Peng Xu
  - Jiguang Wan
year: 2026
venue: DAC
doi: 10.48550/arXiv.2603.27138
code_review:
  checked_on: '2026-11-06'
  status: not-found
  evidence:
    - https://arxiv.org/abs/2603.27138
    - https://arxiv.org/html/2603.27138
    - https://emperorlu.github.io/
  note: 已查论文 HTML、作者主页及题名代码检索；论文说明基于 SGLang 实现，但作者主页的本论文项未给代码入口，本次未找到可核对的公开实现与代码许可。论文自身许可不等于源码许可，不能仅因基于 SGLang 而加开源标签。
---

# ScoutAttention: Efficient KV Cache Offloading via Layer-Ahead CPU Pre-computation for LLM Inference
