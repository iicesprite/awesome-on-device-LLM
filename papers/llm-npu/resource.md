---
id: llm-npu
title: Fast On-device LLM Inference with NPUs
type: paper
url: http://arxiv.org/abs/2407.05858
status: read
summary: 手机 NPU 的 prefill 卸载、离群值拆分与异构调度。
tags:
  - topic/inference
  - backend/npu
  - system/heterogeneous
  - system/offloading
  - stage/prefill
  - availability/open-source
links:
  code: https://github.com/UbiquitousLearning/mllm
attachments:
  - url: https://arxiv.org/pdf/2407.05858
    path: resources/llm-npu-aggggkdz.pdf
    source: zotero
  - url: https://arxiv.org/pdf/2407.05858v1
    path: resources/llm-npu.pdf
    version: arXiv:2407.05858v1
authors:
  - Daliang Xu
  - Hao Zhang
  - Liming Yang
  - Ruiqi Liu
  - Gang Huang
  - Mengwei Xu
  - Xuanzhe Liu
year: 2024
doi: 10.48550/arXiv.2407.05858
code_review:
  checked_on: 2026-11-06
  status: open-source
  license: MIT
  license_url: https://raw.githubusercontent.com/UbiquitousLearning/mllm/v1/LICENSE
  evidence:
    - https://arxiv.org/html/2407.05858
    - https://github.com/UbiquitousLearning/mllm/tree/v1/mllm/backends/qnn
    - https://raw.githubusercontent.com/UbiquitousLearning/mllm/v1/mllm/backends/qnn/QNNBackend.cpp
    - https://raw.githubusercontent.com/UbiquitousLearning/mllm/v1/mllm/backends/qnn/QNNModel.cpp
    - https://raw.githubusercontent.com/UbiquitousLearning/mllm/v1/LICENSE
  note: 论文明确给出 mllm 代码入口；核对 v1 的 QNN 图构建、算子注册及执行集成源码和根目录 MIT 许可。QNN/Hexagon SDK 仍需另行获取并遵守厂商条款，开源标签仅指项目代码，未运行验证。
---

# Fast On-device LLM Inference with NPUs
