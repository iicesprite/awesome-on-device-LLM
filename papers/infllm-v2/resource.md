---
id: infllm-v2
title: "InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation"
type: paper
url: https://proceedings.iclr.cc/paper_files/paper/2026/hash/716e60181e1782f5bd0a629033f85310-Abstract-Conference.html
status: unread
summary: 短序列用原有的稠密注意力，长序列复用同一套投影参数做块稀疏注意力。论文从同一个 8B 稠密检查点出发比较长上下文微调，也把块选择开销与整模型推理耗时分开测量。
tags:
  - topic/inference
  - target/kv-cache
  - system/sparse-inference
  - system/native-sparse-attention
  - system/operator-fusion
  - workload/long-context
  - stage/prefill
  - stage/decode
  - hardware/nvidia-dgpu
  - backend/cuda
  - availability/open-source
links:
  arxiv: https://arxiv.org/abs/2509.24663
  arxiv-doi: https://doi.org/10.48550/arXiv.2509.24663
  openreview: https://openreview.net/forum?id=ZzF9V0H6Vi
  code: https://github.com/OpenBMB/infllmv2_cuda_impl
  long-sparse-base: https://huggingface.co/openbmb/InfLLM-V2-Long-Sparse-Base
  short-dense-base: https://huggingface.co/openbmb/InfLLM-V2-Short-Dense-Base
  minicpm4-1: https://huggingface.co/openbmb/MiniCPM4.1-8B
attachments:
  - url: https://proceedings.iclr.cc/paper_files/paper/2026/file/716e60181e1782f5bd0a629033f85310-Paper-Conference.pdf
    path: resources/infllm-v2-iclr2026.pdf
    version: ICLR 2026 正式版
authors:
  - Weilin Zhao
  - Zihan Zhou
  - Zhou Su
  - Chaojun Xiao
  - Yuxuan Li
  - Yanghao Li
  - Yudi Zhang
  - Weilun Zhao
  - Zhen Li
  - Yuxiang Huang
  - Ao Sun
  - Xu Han
  - Zhiyuan Liu
year: 2026
venue: ICLR
first_published: 2025-09-29
source_checked_on: 2026-11-15
version_note: 主链接与附件是 ICLR 2026 正式版。arXiv 只列出 2025-09-29 的 v1。已核对会议官网、正式版 PDF 与 arXiv 版本记录，未见撤稿或更正标记；OpenReview 论坛页受浏览器验证限制。
code_review:
  checked_on: 2026-11-15
  status: open-source
  license: Apache-2.0
  license_url: https://github.com/OpenBMB/infllmv2_cuda_impl#license
  note: 正式版第 1 节链接 OpenBMB 的实现。仓库可见两阶段 CUDA 实现的源码目录，README 明示该仓库采用 Apache-2.0。未运行代码或审计完整推理链路。
---

# InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation

[阅读笔记](notes.md)
