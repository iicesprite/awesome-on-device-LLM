---
id: qmoe
title: "QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models"
type: paper
url: http://arxiv.org/abs/2310.16795
status: read
summary: 亚 1-bit MoE 压缩格式与 GPU 解码内核协同设计。
tags:
  - topic/quantization
  - architecture/moe
  - target/weight
  - topic/memory
  - backend/cuda
  - availability/open-source
links:
  code: https://github.com/IST-DASLab/qmoe
code_review:
  checked_on: 2026-11-06
  status: open-source
  license: Apache-2.0
  license_url: https://raw.githubusercontent.com/IST-DASLab/qmoe/main/LICENSE
  evidence:
    - https://github.com/IST-DASLab/qmoe
    - https://raw.githubusercontent.com/IST-DASLab/qmoe/main/sub1_cuda_kernel.cu
    - https://raw.githubusercontent.com/IST-DASLab/qmoe/main/LICENSE
  note: 官方 README 对应论文题名及 arXiv；已核对压缩推理 CUDA 源码和根目录 Apache-2.0 许可，未运行实现。
attachments:
  - url: http://arxiv.org/pdf/2310.16795v1
    path: resources/qmoe.pdf
    source: zotero
authors:
  - Elias Frantar
  - Dan Alistarh
year: 2023
doi: 10.48550/arXiv.2310.16795
---

# QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models
