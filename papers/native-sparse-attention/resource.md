---
id: native-sparse-attention
title: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
type: paper
url: https://aclanthology.org/2025.acl-long.1126/
status: unread
summary: 把块级稀疏注意力直接用于预训练。压缩分支用块摘要扫过历史，并为选择分支提供打分；选择分支细读命中的原始 KV，滑窗照顾近邻。组内查询头共用块索引，可复用连续读入的 KV。
tags:
  - topic/inference
  - target/kv-cache
  - system/sparse-inference
  - system/native-sparse-attention
  - system/memory-layout
  - workload/long-context
  - stage/prefill
  - stage/decode
  - architecture/moe
  - hardware/nvidia-dgpu
  - backend/cuda
links:
  arxiv: https://arxiv.org/abs/2502.11089v2
  arxiv-doi: https://doi.org/10.48550/arXiv.2502.11089
attachments:
  - url: https://aclanthology.org/2025.acl-long.1126.pdf
    path: resources/native-sparse-attention-acl2025.pdf
    version: ACL 2025 正式版
authors:
  - Jingyang Yuan
  - Huazuo Gao
  - Damai Dai
  - Junyu Luo
  - Liang Zhao
  - Zhengyan Zhang
  - Zhenda Xie
  - Yuxing Wei
  - Lean Wang
  - Zhiping Xiao
  - Yuqing Wang
  - Chong Ruan
  - Ming Zhang
  - Wenfeng Liang
  - Wangding Zeng
year: 2025
venue: ACL
doi: 10.18653/v1/2025.acl-long.1126
first_published: 2025-02-16
last_revised: 2025-02-27
version_note: 主链接与阅读笔记采用 ACL 2025 年 7 月正式版；两个日期字段分别指 arXiv 首次提交和 v2 修订。ACL Anthology 与 arXiv 记录未见撤稿或更正标记。
---

# Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

[阅读笔记](notes.md)
