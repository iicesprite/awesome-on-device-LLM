---
id: moe-lightning
title: 'MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs'
type: paper
url: https://doi.org/10.1145/3669940.3707267
status: unread
summary: 面向离线批量 MoE 推理，用分页权重、CPU–GPU–I/O 流水线 CGOPipe 和分层 roofline 模型选择策略，提高有限显存下的吞吐与内存利用率。
tags:
- topic/inference
- topic/memory
- architecture/moe
- target/weight
- target/kv-cache
- system/heterogeneous
- system/offloading
- system/pipelining
- system/memory-layout
- workload/offline-batch
- stage/decode
- memory/discrete
- hardware/nvidia-dgpu
- hardware/x86
- backend/cpu
links:
  arxiv: https://arxiv.org/abs/2411.11217
  project: https://pschafhalter.com/moe-lightning/
attachments:
- url: https://pschafhalter.com/papers/2025-asplos-moe-lightning.pdf
  path: resources/moe-lightning-asplos2025.pdf
  version: ASPLOS 2025 author-hosted paper
authors:
- Shiyi Cao
- Shu Liu
- Tyler Griggs
- Peter Schafhalter
- Xiaoxuan Liu
- Ying Sheng
- Joseph E. Gonzalez
- Matei Zaharia
- Ion Stoica
year: 2025
venue: ASPLOS 2025, Volume 1
doi: 10.1145/3669940.3707267
first_published: '2024-11-18'
source_checked_on: '2026-10-30'
version_note: year 为正式发表年份；首次 arXiv 预印本发布于 2024 年，本地保存作者网站提供的 ASPLOS 2025 版本。
code_availability:
  url: https://github.com/caoshiyi/FastMoE
  linked_from: https://pschafhalter.com/
  checked_on: '2026-10-30'
  status: http-404
  detail: 作者主页提供的代码链接本次返回 404；未以第三方 fork 替代官方实现。
code_review:
  checked_on: '2026-11-06'
  status: unverified
  evidence:
  - https://pschafhalter.com/moe-lightning/
  - https://github.com/caoshiyi/FastMoE
  - https://api.github.com/repos/caoshiyi/FastMoE
  - https://shiyicao.com/
  - https://github.com/caoshiyi/artifacts
  - https://github.com/caoshiyi/artifacts/branches
  - https://raw.githubusercontent.com/caoshiyi/artifacts/main/README.md
  note: 项目页的 FastMoE 链接返回 404，GitHub API 重试受限；第一作者现主页改指 artifacts，但所查唯一 main 分支仅有一句 README，未见论文实现或许可。本次未能核实公开实现，不添加代码链接或开源标签；保留历史失效链接记录。
---

# MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
