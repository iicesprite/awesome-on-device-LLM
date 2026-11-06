---
id: headinfer
title: 'HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading'
type: paper
url: https://arxiv.org/abs/2502.12574
status: unread
summary: 将 KV cache 按 attention head 细粒度卸载到 CPU 内存，结合分块与计算传输重叠，降低消费级 GPU 上长上下文推理的显存需求。
tags:
- topic/inference
- topic/memory
- target/kv-cache
- system/offloading
- system/pipelining
- workload/long-context
- stage/prefill
- stage/decode
- memory/discrete
- hardware/nvidia-dgpu
- backend/cuda
links:
  code: https://github.com/wdlctc/headinfer
attachments:
- url: https://arxiv.org/pdf/2502.12574v1
  path: resources/headinfer-2502.12574v1.pdf
  version: arXiv:2502.12574v1
authors:
- Cheng Luo
- Zefan Cai
- Hanshi Sun
- Jinqi Xiao
- Bo Yuan
- Wen Xiao
- Junjie Hu
- Jiawei Zhao
- Beidi Chen
- Anima Anandkumar
year: 2025
doi: 10.48550/arXiv.2502.12574
first_published: '2025-02-18'
source_checked_on: '2026-10-30'
code_review:
  checked_on: '2026-11-06'
  status: source-available
  evidence:
  - https://github.com/wdlctc/headinfer
  - https://github.com/wdlctc/headinfer/tree/main/headinfer
  - https://raw.githubusercontent.com/wdlctc/headinfer/main/main.py
  - https://raw.githubusercontent.com/wdlctc/headinfer/main/setup.py
  - https://raw.githubusercontent.com/wdlctc/headinfer/main/pyproject.toml
  note: README 引用 arXiv 2502.12574，已确认推理入口及按头卸载代码。README 徽章与 setup.py 声明 MIT，但根目录及实现目录未见许可证文件，本次未取得完整许可正文，暂不添加开源标签；未运行代码。
---

# HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading
