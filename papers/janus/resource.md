---
id: janus
title: Efficient Agentic LLM Serving over SSD-based Sparse KV Storage
type: paper
url: https://arxiv.org/abs/2609.36938
status: unread
summary: Janus 针对多轮 agent 的追加预填充，提前用模型自带的索引器预测本层要从 SSD 读取的历史 KV，等精确结果出来再补读。论文比较了提前预取、等选完再读和顺序读取整段历史三种路径。
tags:
  - topic/inference
  - target/kv-cache
  - system/offloading
  - system/native-sparse-attention
  - system/pipelining
  - workload/long-context
  - workload/online-serving
  - workload/agent-serving
  - stage/prefill
  - hardware/nvidia-dgpu
  - backend/cuda
  - availability/open-source
links:
  code: https://github.com/WenhaoHe02/Janus
  html: https://arxiv.org/html/2609.36938v1
attachments:
  - url: https://arxiv.org/pdf/2609.36938v1
    path: resources/janus-arxiv-v1.pdf
    version: arXiv v1（2026-09-29）
authors: [Wenhao He, Ping Zhang, Xiaohe Hu, Chutian Wang, Jinlong Hou, Yuan Cheng, Peng Sun, Fangcheng Fu]
year: 2026
venue: arXiv
---

# Efficient Agentic LLM Serving over SSD-based Sparse KV Storage

[阅读笔记](notes.md)
