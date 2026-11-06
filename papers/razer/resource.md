---
id: razer
title: 'RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping'
type: paper
url: http://arxiv.org/abs/2501.04052
status: unread
summary: 围绕 NVFP4 冗余编码设计数值格式、GPU 内核和硬件支持。
tags:
- topic/quantization
- precision/fp4
- target/weight
- target/activation
- topic/inference
- availability/open-source
links:
  code: https://github.com/abdelfattah-lab/NVFP4-RaZeR
code_review:
  checked_on: '2026-11-06'
  status: open-source
  license: MIT
  license_url: https://raw.githubusercontent.com/abdelfattah-lab/NVFP4-RaZeR/main/LICENSE
  evidence:
  - https://github.com/abdelfattah-lab/NVFP4-RaZeR
  - https://raw.githubusercontent.com/abdelfattah-lab/NVFP4-RaZeR/main/quantize/quantizer.py
  - https://raw.githubusercontent.com/abdelfattah-lab/NVFP4-RaZeR/main/LICENSE
  note: 官方 README 对应当前 NVFP4 论文题名、作者和 arXiv；已核对 RaZeR 特殊值搜索量化实现及根目录 MIT 许可。README 另声明康奈尔大学专利申请，未运行实现或核验专利许可。
attachments:
- url: http://arxiv.org/pdf/2501.04052v3
  path: resources/razer.pdf
  source: zotero
- url: http://arxiv.org/pdf/2501.04052v3
  path: resources/razer-657x6tne.pdf
  source: zotero
  source_zotero_attachment_key: 657X6TNE
  source_zotero_item_key: A7QP7PQU
authors:
- Yuzong Chen
- Xilai Dai
- Jake Hyun
- Chi-Chih Chang
- Wonsuk Jang
- Yuheng Wu
- Thierry Tambe
- Jae-sun Seo
- Mohamed S. Abdelfattah
year: 2026
doi: 10.48550/arXiv.2501.04052
---

# RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping
