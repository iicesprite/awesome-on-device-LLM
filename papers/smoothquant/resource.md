---
id: smoothquant
title: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
type: paper
url: https://proceedings.mlr.press/v202/xiao23c.html
status: read
summary: 等价变换处理激活离群值，连接校准与 W8A8 推理。
tags:
  - topic/quantization
  - target/weight
  - target/activation
  - precision/w8a8
  - method/smoothquant
  - availability/open-source
links:
  code: https://github.com/mit-han-lab/smoothquant
code_review:
  checked_on: 2026-11-06
  status: open-source
  license: MIT
  license_url: https://raw.githubusercontent.com/mit-han-lab/smoothquant/main/LICENSE
  evidence:
    - https://github.com/mit-han-lab/smoothquant
    - https://raw.githubusercontent.com/mit-han-lab/smoothquant/main/smoothquant/opt.py
    - https://raw.githubusercontent.com/mit-han-lab/smoothquant/main/LICENSE
  note: 官方 README 对应 ICML 2023 论文题名及作者；已核对 INT8 OPT 实现及根目录 MIT 许可，实际 INT8 算子依赖 torch-int，未运行实现。
attachments:
  - url: https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf
    path: resources/smoothquant-icml2023.pdf
    version: ICML 2023 proceedings
authors:
  - Guangxuan Xiao
  - Ji Lin
  - Mickael Seznec
  - Hao Wu
  - Julien Demouth
  - Song Han
year: 2023
venue: Proceedings of the 40th International Conference on Machine Learning
attachment_note: 已补充 ICML 2023 官方 PDF，并交由 Zotero 管理副本。
---

# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
