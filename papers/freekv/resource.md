---
id: freekv
title: "FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference"
type: paper
url: https://proceedings.iclr.cc/paper_files/paper/2026/hash/5137e40ee586c4f1fc3615cf8cee4f87-Abstract-Conference.html
status: reading
summary: 长上下文解码时，用上一步的 query 预取 KV 页，遇到明显偏离再按头纠错；CPU 与 GPU 采用不同布局，让连续搬运和布局转换可以重叠。
tags:
  - topic/inference
  - topic/memory
  - target/kv-cache
  - system/heterogeneous
  - system/offloading
  - system/sparse-inference
  - system/memory-layout
  - system/pipelining
  - stage/decode
  - workload/long-context
  - hardware/nvidia-dgpu
  - backend/cuda
  - availability/open-source
links:
  arxiv: https://arxiv.org/abs/2505.13109
  arxiv-doi: https://doi.org/10.48550/arXiv.2505.13109
  openreview: https://openreview.net/forum?id=wXAn7orB1H
  code: https://github.com/sjtu-zhao-lab/FreeKV
attachments:
  - url: https://proceedings.iclr.cc/paper_files/paper/2026/file/5137e40ee586c4f1fc3615cf8cee4f87-Paper-Conference.pdf
    path: resources/freekv-iclr2026.pdf
    version: ICLR 2026 正式版
authors:
  - Guangda Liu
  - Chengwei Li
  - Zhenyu Ning
  - Jing Lin
  - Yiwu Yao
  - Danning Ke
  - Minyi Guo
  - Jieru Zhao
year: 2026
venue: ICLR
first_published: 2025-05-19
last_revised: 2026-03-09
source_checked_on: 2026-11-10
version_note: 主链接及附件来源采用 ICLR 2026 正式发表版本，官方发表日期为 2026-04-20；arXiv 首版发表于 2025-05-19，最近修订为 2026-03-09 的 v5。已核对官方发表页、arXiv 版本记录与作者代码仓库，所查页面未见撤稿或更正标记；初次收录时 OpenReview 页面受浏览器验证限制。阅读笔记以会议正式版正文和所引附录为准，仓库未保留 PDF。
tag_note: 主题及系统标签依据官方摘要中的推测检索、细粒度纠错、CPU–GPU 混合布局及双缓冲流式召回；稀疏推理指按需检索部分 KV，不表示永久淘汰。
code_review:
  checked_on: 2026-11-10
  status: open-source
  license: MIT
  license_url: https://raw.githubusercontent.com/sjtu-zhao-lab/FreeKV/HEAD/LICENSE
  evidence:
    - https://arxiv.org/abs/2505.13109v5
    - https://github.com/sjtu-zhao-lab/FreeKV
    - https://raw.githubusercontent.com/sjtu-zhao-lab/FreeKV/HEAD/LICENSE
  note: arXiv 摘要直接链接此作者仓库；已核对 README 的论文题名、推测检索与纠错运行入口、源码目录及根目录 MIT 许可。未审计实现或运行代码，不将开源许可核验视为性能复现或端到端实现审计。
---

# FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference

[阅读笔记](notes.md)
