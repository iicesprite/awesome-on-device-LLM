---
id: awq
title: "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration"
type: paper
url: https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html
status: read
summary: 激活感知权重量化与 TinyChat 的打包、融合和端侧推理实现。
tags:
  - topic/quantization
  - target/weight
  - scheme/weight-only
  - precision/w4a16
  - method/awq
  - hardware/jetson-orin
  - availability/open-source
links:
  code: https://github.com/mit-han-lab/llm-awq
code_review:
  checked_on: 2026-11-06
  status: open-source
  license: MIT
  license_url: https://raw.githubusercontent.com/mit-han-lab/llm-awq/main/LICENSE
  evidence:
    - https://github.com/mit-han-lab/llm-awq/blob/main/README.md
    - https://raw.githubusercontent.com/mit-han-lab/llm-awq/main/awq/entry.py
    - https://raw.githubusercontent.com/mit-han-lab/llm-awq/main/awq/quantize/pre_quant.py
    - https://raw.githubusercontent.com/mit-han-lab/llm-awq/main/LICENSE
  note: 官方 README 对应 AWQ 论文，已核实量化入口、预量化实现及根目录 MIT 许可；未运行代码。
attachments:
  - url: https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf
    path: resources/awq.pdf
    source: zotero
authors:
  - Ji Lin
  - Jiaming Tang
  - Haotian Tang
  - Shang Yang
  - Wei-Ming Chen
  - Wei-Chen Wang
  - Guangxuan Xiao
  - Xingyu Dang
  - Chuang Gan
  - Song Han
year: 2024
venue: Proceedings of Machine Learning and Systems
---

# AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
