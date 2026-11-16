---
id: sparda
title: "SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference"
type: paper
url: https://arxiv.org/abs/2606.04511v1
status: unread
summary: 在原生块稀疏模型中加入可训练的 Forecast 投影，每个 GQA 组用一个头提前为下一层选块；解码时把选中 KV 的主存回读与当前层计算重叠。
tags:
  - topic/inference
  - topic/memory
  - target/kv-cache
  - system/sparse-inference
  - system/native-sparse-attention
  - system/offloading
  - system/pipelining
  - stage/prefill
  - stage/decode
  - workload/long-context
  - hardware/nvidia-dgpu
  - backend/cuda
  - availability/open-source
links:
  arxiv-html: https://arxiv.org/html/2606.04511v1
  code: https://github.com/NVlabs/SparDA
attachments:
  - url: https://arxiv.org/pdf/2606.04511v1
    path: resources/sparda-arxiv-v1.pdf
    version: arXiv v1
authors:
  - Yaosheng Fu
  - Guangxuan Xiao
  - Xin Dong
  - Song Han
  - Oreste Villa
year: 2026
doi: 10.48550/arXiv.2606.04511
first_published: 2026-06-03
source_checked_on: 2026-11-15
version_note: arXiv 记录仅有 v1，未标记撤稿或更正。PDF 封面署 2026-06-02，首发日期取 arXiv 的提交记录。未找到可核对的正式会议或期刊版本。
code_review:
  checked_on: 2026-11-15
  status: open-source
  license: Apache-2.0
  license_url: https://raw.githubusercontent.com/NVlabs/SparDA/main/LICENSE
  evidence:
    - https://arxiv.org/abs/2606.04511v1
    - https://github.com/NVlabs/SparDA
    - https://raw.githubusercontent.com/NVlabs/SparDA/main/training/train_indexer.py
    - https://raw.githubusercontent.com/NVlabs/SparDA/main/LICENSE
  note: 论文直接链接 NVlabs/SparDA。核对了训练入口只把 Forecast 投影加入优化器的代码和根目录 Apache-2.0 许可，第三方组件另列许可。未运行训练或评测。
---

# SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference

[阅读笔记](notes.md)
