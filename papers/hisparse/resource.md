---
id: hisparse
title: "HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management"
type: paper
url: https://arxiv.org/abs/2608.07009
status: unread
summary: 将完整 KV 历史保存在主存，以固定容量 GPU 缓存和融合 CUDA 内核处理稀疏解码的命中、LRU 替换与数据取回，并用跨层预取降低传输开销；仅改变 KV 放置而不改变模型输出。
tags:
  - topic/inference
  - topic/memory
  - system/offloading
  - system/sparse-inference
  - system/native-sparse-attention
  - system/operator-fusion
  - stage/decode
  - workload/long-context
  - availability/open-source
links:
  arxiv_version: https://arxiv.org/abs/2608.07009v1
  code: https://github.com/sgl-project/sglang
attachments:
  - url: https://arxiv.org/pdf/2608.07009v1
    path: resources/hisparse-2608.07009v1.pdf
    version: arXiv v1
authors:
  - Zhiqiang Xie
  - Zhangheng Huang
  - Tingwei Huang
  - Ziyi Xu
  - Ruiyang Ma
  - Christos Kozyrakis
year: 2026
doi: 10.48550/arXiv.2608.07009
code_review:
  checked_on: '2026-11-06'
  status: open-source
  license: Apache-2.0
  license_url: https://raw.githubusercontent.com/sgl-project/sglang/6a79021d7ec225eecbc64637f537379d6d148c1a/LICENSE
  evidence:
    - https://arxiv.org/html/2608.07009
    - https://docs.sglang.io/advanced_features/hisparse_guide.html
    - https://github.com/sgl-project/sglang/pull/20343
    - https://raw.githubusercontent.com/sgl-project/sglang/6a79021d7ec225eecbc64637f537379d6d148c1a/python/sglang/jit_kernel/csrc/hisparse.cuh
    - https://raw.githubusercontent.com/sgl-project/sglang/main/LICENSE
    - https://raw.githubusercontent.com/sgl-project/sglang/6a79021d7ec225eecbc64637f537379d6d148c1a/LICENSE
  note: 论文明确指向上游 SGLang，官方指南与作者合并 PR 相符；已核对公开 CUDA 缓存取回内核及仓库 Apache-2.0 许可。当前旧内核路径已变动，源码证据固定为 PR 提交；未运行实现。
---

# HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management
