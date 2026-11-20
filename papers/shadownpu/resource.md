---
id: shadownpu
title: 'ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference'
type: paper
url: http://arxiv.org/abs/2508.16703
status: unread
summary: 针对 Attention 回退问题，协同设计稀疏 Attention 与 NPU 异构流水线。
tags:
- topic/inference
- backend/npu
- system/heterogeneous
- system/sparse-inference
- workload/agent-serving
- availability/open-source
links:
  code: https://github.com/yinwangsong/mllm/tree/feat/shadownpu-sparse-attention
attachments:
- url: https://arxiv.org/pdf/2508.16703v4
  path: resources/shadownpu.pdf
  source: zotero
authors:
- Wangsong Yin
- Daliang Xu
- Mengwei Xu
- Gang Huang
- Xuanzhe Liu
year: 2026
doi: 10.48550/arXiv.2508.16703
code_review:
  checked_on: '2026-11-06'
  status: open-source
  license: MIT
  license_url: https://raw.githubusercontent.com/yinwangsong/mllm/feat/shadownpu-sparse-attention/LICENSE
  evidence:
  - https://arxiv.org/html/2508.16703
  - https://yinwangsong.github.io/
  - https://github.com/yinwangsong/mllm/branches
  - https://raw.githubusercontent.com/yinwangsong/mllm/feat/shadownpu-sparse-attention/docs/shadownpu_sparse_attention.md
  - https://raw.githubusercontent.com/yinwangsong/mllm/feat/shadownpu-sparse-attention/tools/head_sparsity_profiler/head_sparsity_profiler.py
  - https://raw.githubusercontent.com/yinwangsong/mllm/feat/shadownpu-sparse-attention/tools/hmx_int8_matmul/src/dsp/hmx_i8_kernel.c
  - https://raw.githubusercontent.com/yinwangsong/mllm/feat/shadownpu-sparse-attention/LICENSE
  - https://zenodo.org/records/19555734
  note: 从第一作者主页确认 GitHub 身份并找到其 mllm 的 ShadowNPU 专用分支，核对稀疏注意力复现文档、逐头保留率分析及 HMX INT8 QK 内核和根目录 MIT 许可；分析工具关联同题名作者的 Zenodo 软件工件。该分支文档说明主要打包 Qwen2.5-1.5B prefill，decode 仍为 dense，其他模型需另备工件；未运行或宣称全部论文实验可复现。
---

# ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference

[阅读笔记](notes.md)
