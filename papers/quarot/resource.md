---
id: quarot
title: "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs"
type: paper
url: http://arxiv.org/abs/2404.00456
status: read
summary: 旋转去除离群值，联合量化权重、激活和 KV cache。
tags:
  - topic/quantization
  - target/weight
  - target/activation
  - target/kv-cache
  - precision/w4a4
  - transform/rotation
  - availability/open-source
links:
  code: https://github.com/spcl/QuaRot
code_review:
  checked_on: 2026-11-06
  status: open-source
  license: Apache-2.0
  license_url: https://raw.githubusercontent.com/spcl/QuaRot/main/LICENSE
  evidence:
    - https://github.com/spcl/QuaRot
    - https://raw.githubusercontent.com/spcl/QuaRot/main/quarot/kernels/gemm.cu
    - https://raw.githubusercontent.com/spcl/QuaRot/main/LICENSE
  note: 官方 README 对应论文题名及 arXiv；已核对 INT4 GEMM 实现及根目录 Apache-2.0 许可，未运行实现。
attachments:
  - url: https://arxiv.org/pdf/2404.00456
    path: resources/quarot.pdf
    source: zotero
  - url: https://proceedings.neurips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf
    path: resources/quarot-xjathrv4.pdf
    source: zotero
    source_zotero_attachment_key: XJATHRV4
    source_zotero_item_key: K6BJYZTY
  - url: https://proceedings.neurips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf
    path: resources/quarot-x4pnmt6u.pdf
    source: zotero
    source_zotero_attachment_key: X4PNMT6U
    source_zotero_item_key: FTKB2B3N
authors:
  - Saleh Ashkboos
  - Amirkeivan Mohtashami
  - Maximilian L. Croci
  - Bo Li
  - Pashmina Cameron
  - Martin Jaggi
  - Dan Alistarh
  - Torsten Hoefler
  - James Hensman
year: 2024
doi: 10.48550/arXiv.2404.00456
---

# QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
