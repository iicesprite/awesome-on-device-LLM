---
id: pariskv
title: "ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs"
type: paper
url: https://proceedings.mlr.press/v306/qi26b.html
status: unread
summary: ParisKV 针对长生成中检索质心逐渐失配的问题，用固定方向质心筛选候选，再用 4-bit key 摘要估计内积进行重排。只把选中的原精度 KV 从主存取回 GPU。
tags:
  - topic/inference
  - topic/memory
  - target/kv-cache
  - transform/rotation
  - system/sparse-inference
  - system/offloading
  - system/operator-fusion
  - workload/long-context
  - stage/decode
  - backend/cuda
links:
  arxiv: https://arxiv.org/abs/2602.07721
  arxiv-doi: https://doi.org/10.48550/arXiv.2602.07721
  code: https://github.com/amy-77/ParisKV
attachments:
  - url: https://raw.githubusercontent.com/mlresearch/v306/main/assets/qi26b/qi26b.pdf
    path: resources/pariskv-icml2026.pdf
    version: ICML 2026 正式版
authors:
  - Yanlin Qi
  - Xinhang Chen
  - Huiqiang Jiang
  - Qitong Wang
  - Botao Peng
  - Themis Palpanas
year: 2026
venue: ICML
first_published: 2026-02-07
last_revised: 2026-05-28
version_note: 主链接采用 ICML 2026 正式发表版本。上述首发和修订日期指 arXiv v1、v3；arXiv 版本页与正式发表页均未见撤稿或更正标记。
---

# ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs

[阅读笔记](notes.md)
