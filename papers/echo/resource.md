---
id: echo
title: "ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs"
type: paper
url: https://www.usenix.org/conference/osdi26/presentation/liu-guangda
status: unread
summary: ECHO 把原生稀疏注意力模型的主要 KV 缓存放到主存，用 CUDA Graph 内的缓存管理和融合预取减少取回开销。预取只改变搬运时机，最终仍按模型的精确 top-k 取齐数据。
tags:
  - topic/inference
  - topic/memory
  - target/kv-cache
  - system/heterogeneous
  - system/offloading
  - system/sparse-inference
  - system/native-sparse-attention
  - system/operator-fusion
  - system/pipelining
  - workload/online-serving
  - workload/long-context
  - stage/prefill
  - stage/decode
  - hardware/nvidia-dgpu
  - backend/cuda
links:
  code: https://github.com/sjtu-zhao-lab/ECHO
  artifact: https://github.com/MachineLearningSystem/26OSDI-ECHO
attachments:
  - url: https://www.usenix.org/system/files/osdi26-liu-guangda.pdf
    path: resources/echo-osdi2026.pdf
    version: OSDI 2026 正式版
authors:
  - Guangda Liu
  - Wenhao Chen
  - Chengwei Li
  - Zhenyu Ning
  - Jing Lin
  - Yiwu Yao
  - Quan Chen
  - Shixuan Sun
  - Jieru Zhao
  - Minyi Guo
year: 2026
venue: OSDI
version_note: 采用 USENIX 发布的 OSDI 2026 正式版，论文页码 17–37；发表页未见撤稿或更正标记。
---

# ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs

[阅读笔记](notes.md)
