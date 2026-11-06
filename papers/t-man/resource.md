---
id: t-man
title: "T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup"
type: paper
url: http://arxiv.org/abs/2511.11248
status: read
summary: 统一查表布局、分块和流水线降低 NPU 低比特推理开销。
tags:
  - topic/inference
  - topic/quantization
  - backend/npu
  - system/lut
  - system/memory-layout
  - availability/open-source
links:
  code: https://github.com/microsoft/T-MAC/tree/main/t-man
attachments:
  - url: http://arxiv.org/pdf/2511.11248v1
    path: resources/t-man-w8fbwgvf.pdf
    source: zotero
  - url: https://arxiv.org/pdf/2511.11248v1
    path: resources/t-man.pdf
    version: arXiv:2511.11248v1
authors:
  - Jianyu Wei
  - Qingtao Li
  - Shijie Cao
  - Lingxiao Ma
  - Zixu Hao
  - Yanyong Zhang
  - Xiaoyan Hu
  - Ting Cao
year: 2025
doi: 10.48550/arXiv.2511.11248
code_review:
  checked_on: 2026-11-06
  status: open-source
  license: MIT / BSD-3-Clause
  license_url: https://raw.githubusercontent.com/kaleid-liner/executorch/817bd29dfbea4979a27f2f539c753bd46757ea7a/LICENSE
  evidence:
    - https://arxiv.org/html/2511.11248
    - https://github.com/microsoft/T-MAC/tree/main/t-man
    - https://raw.githubusercontent.com/microsoft/T-MAC/main/t-man/docs/build.md
    - https://raw.githubusercontent.com/microsoft/T-MAC/main/LICENSE
    - https://raw.githubusercontent.com/kaleid-liner/executorch/817bd29dfbea4979a27f2f539c753bd46757ea7a/LICENSE
    - https://raw.githubusercontent.com/kaleid-liner/executorch/817bd29dfbea4979a27f2f539c753bd46757ea7a/backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANLinear.cpp
  note: 论文链接指向 T-MAC 的 t-man；实际 TMANLinear 查表算子在固定提交的 ExecuTorch 子模块中，已核对算子实现及其根目录 BSD 三条款许可，外层 T-MAC 为 MIT。QNN/Hexagon SDK 另受厂商条款约束，未执行或核验论文性能。
---

# T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup
