---
id: retrievalattention
title: "RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval"
type: paper
url: https://proceedings.neurips.cc/paper_files/paper/2025/hash/4e36d4049fb0fea195a8267c8dcd0824-Abstract-Conference.html
status: read
summary: 面向可离线预建索引的固定上下文，以适应 Q/K 分布差异的注意力感知 ANNS 索引将多数 KV 缓存保留在 CPU，并通过 CPU–GPU 协同稀疏检索加速长上下文解码、降低显存占用。
tags:
  - topic/inference
  - topic/memory
  - system/heterogeneous
  - system/offloading
  - system/sparse-inference
  - stage/decode
  - workload/long-context
links:
  arxiv: https://arxiv.org/abs/2409.10516
  arxiv-doi: https://doi.org/10.48550/arXiv.2409.10516
attachments:
  - url: https://papers.nips.cc/paper_files/paper/2025/file/4e36d4049fb0fea195a8267c8dcd0824-Paper-Conference.pdf
    path: resources/retrievalattention-neurips2025.pdf
authors:
  - Di Liu
  - Meng Chen
  - Baotong Lu
  - Huiqiang Jiang
  - Zhenhua Han
  - Qianxi Zhang
  - Qi Chen
  - Chengruidong Zhang
  - Bailu Ding
  - Kai Zhang
  - Chen Chen
  - Fan Yang
  - Yuqing Yang
  - Lili Qiu
year: 2025
venue: NeurIPS
doi: 10.52202/085713-1816
first_published: 2024-09-16
last_revised: 2024-12-31
source_checked_on: 2026-11-03
version_note: 主链接及附件为 NeurIPS 2025 正式版，year 采用官方 BibTeX 的会议年份；发表页 citation_publication_date 为 2026-04-23。first_published 和 last_revised 分别记录 arXiv 首次提交与最新 v3 修订日期，预印本不等同于正式版。
tag_note: 标签依据正式版摘要及 CPU–GPU 协同机制；固定上下文需离线预建索引，KV 卸载与稀疏检索并非 KV 量化。未核验对应论文实现，不添加 evidence 标签。
code_review:
  checked_on: '2026-11-06'
  status: unverified
  evidence:
    - https://raw.githubusercontent.com/microsoft/RetrievalAttention/main/README.md
    - https://github.com/microsoft/RetrievalAttention/tree/main/attn_hub
    - https://github.com/microsoft/RetrievalAttention/tree/main/cache_hub
    - https://github.com/microsoft/RetrievalAttention/branches
    - https://github.com/microsoft/RetrievalAttention/commits/main/
    - https://raw.githubusercontent.com/microsoft/RetrievalAttention/26611f0a726f4bd3cd2fc456c33b4530dcae781b/README.md
    - https://raw.githubusercontent.com/microsoft/RetrievalAttention/main/LICENSE
  note: 同名官方仓库虽引用本论文且有 MIT 许可，但当前 README、注意力与缓存模块实际为 RetroInfer 的聚类及三分区方案；早期 README 为模板，仅见 main 分支。本次未能核实本论文注意力感知 ANNS 的独立实现，不能将 RetroInfer 实现视为本论文已开源，故不新增代码链接或标签。
---

# RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval

[阅读笔记](notes.md)
