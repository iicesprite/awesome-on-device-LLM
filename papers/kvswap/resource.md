---
id: kvswap
title: "KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference"
type: paper
url: https://doi.org/10.1145/3745756.3809234
status: read
summary: 面向 Jetson 统一内存设备的磁盘 KV 卸载：低秩索引预测、分组预取与缓存复用协同，正式版含 AGX Orin 和 Orin Nano 实测。
tags:
  - topic/inference
  - topic/memory
  - target/kv-cache
  - system/offloading
  - memory/unified
  - system/sparse-inference
  - system/pipelining
  - system/memory-layout
  - workload/long-context
  - workload/offline-batch
  - stage/decode
  - hardware/jetson-orin
  - hardware/arm
  - backend/cuda
  - evidence/end-to-end
links:
  arxiv: https://arxiv.org/abs/2511.11907v2
  published: https://eprints.whiterose.ac.uk/id/eprint/240121/
  code: https://github.com/hwhwz23/KVSWAP-CODE
  project: https://issl-uk.com/publication/mobisys-26/
attachments:
  - url: https://arxiv.org/pdf/2511.11907
    path: resources/kvswap.pdf
    source: zotero
    version: arXiv:2511.11907v2
  - url: https://eprints.whiterose.ac.uk/id/eprint/240121/8/3745756.3809234.pdf
    path: resources/kvswap-mobisys2026.pdf
    version: MobiSys 2026 published version
authors:
  - Huawei Zhang
  - Chunwei Xia
  - Zheng Wang
year: 2026
venue: MobiSys 2026
doi: 10.1145/3745756.3809234
first_published: 2025-11-14
published_on: 2026-06-20
source_checked_on: 2026-10-30
tag_note: 低秩 K 仅用于索引预测，未将 KV 量化作为实现贡献；硬件标签对应 Jetson AGX Orin/Orin Nano 实测，不扩展为高通或 UFS 支持。evidence/end-to-end 根据作者端到端评测及官方执行路径只读检查，未本地复现。
code_review:
  checked_on: 2026-11-06
  status: source-available
  evidence:
    - https://raw.githubusercontent.com/hwhwz23/KVSWAP-CODE/main/README.md
    - https://github.com/hwhwz23/KVSWAP-CODE/tree/main/engine
    - https://github.com/hwhwz23/KVSWAP-CODE/tree/main/quality
    - https://raw.githubusercontent.com/hwhwz23/KVSWAP-CODE/main/engine/src/cache_manager.py
  note: README 明确对应本论文，已查看 engine 缓存管理实现；根目录、engine 与 quality 入口本次未找到适用于作者实现的明确开源许可，不能以第三方代码许可代替，故保留代码链接但不加开源标签。未运行实现。
---

# KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference
