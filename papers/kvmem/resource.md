---
id: kvmem
title: "KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU"
type: paper
url: https://arxiv.org/abs/2609.04852
status: unread
summary: 将长时运行智能体的历史 KV 按页保存在显存、主机内存和 NVMe 中，每步按当前查询召回相关块，避免把已处理的历史文本重新预填充。作者在 24 GB RTX 5090 Laptop GPU 上演示了百万 token 工作区，单步仍受模型原生上下文窗口限制。
tags:
  - topic/inference
  - topic/memory
  - target/kv-cache
  - system/offloading
  - workload/agent-serving
  - workload/long-context
  - memory/discrete
  - hardware/nvidia-dgpu
  - backend/cuda
  - availability/open-source
links:
  html: https://arxiv.org/html/2609.04852v1
  code: https://github.com/kvmem/kvmem-qw3
authors:
  - Di Chai
  - Leye Wang
  - Zeshen Su
  - Zhiguo Xia
  - Zhihang Yu
year: 2026
doi: 10.48550/arXiv.2609.04852
code_review:
  checked_on: 2026-11-20
  status: open-source
  license: Apache-2.0
  license_url: https://raw.githubusercontent.com/kvmem/kvmem-qw3/main/LICENSE
  evidence:
    - https://arxiv.org/html/2609.04852v1
    - https://raw.githubusercontent.com/kvmem/kvmem-qw3/main/src/kvmem_archive.cpp
    - https://raw.githubusercontent.com/kvmem/kvmem-qw3/main/THIRD_PARTY_NOTICES.md
  note: 论文指向该仓库；已核对公开的 KV 归档源码和仓库级许可。第三方声明另列 tokenizer 与 MTP 部分代码的来源待确认，不将许可结论外推至这些部分，未运行代码。
---

# KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU

[阅读笔记](notes.md)
