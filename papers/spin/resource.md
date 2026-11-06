---
id: spin
title: "Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving"
type: paper
url: https://arxiv.org/abs/2604.26837
status: unread
summary: SPIN 在 vLLM 上以统一分区抽象、局部性感知 KV 缓存管理和两级元数据布局，协同设计稀疏注意力执行与 CPU–GPU 分层 KV 存储，减少细粒度 KV 访问的跨设备传输开销。
tags:
  - topic/inference
  - topic/memory
  - system/offloading
  - system/sparse-inference
  - stage/decode
  - workload/long-context
links:
  arxiv_version: https://arxiv.org/abs/2604.26837v1
attachments:
  - url: https://arxiv.org/pdf/2604.26837v1
    path: resources/spin-2604.26837v1.pdf
    version: arXiv v1
authors:
  - Zihan Zhao
  - Baotong Lu
  - Shengjie Lin
  - Yizou Chen
  - Jing Liu
  - Yanqi Zhang
  - Ziming Miao
  - Ming-Chang Yang
  - Haiying Shen
  - Qi Chen
  - Fan Yang
year: 2026
doi: 10.48550/arXiv.2604.26837
code_review:
  checked_on: '2026-11-06'
  status: not-found
  evidence:
    - https://arxiv.org/abs/2604.26837
    - https://arxiv.org/html/2604.26837
    - https://www.microsoft.com/en-us/research/publication/unifying-sparse-attention-with-hierarchical-memory-for-scalable-long-context-llm-serving/
  note: 已查论文 HTML、微软研究院官方发表页及题名代码检索；本次未找到可核对的 SPIN 公开实现与代码许可。基于 vLLM 及集成 RetroInfer 的描述不能代替本框架源码发布证据，也不与其他同名 SPIN 项目混用。
---

# Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving
