---
id: mobile-npu-test-time-compute
title: Scaling LLM Test-Time Compute with Mobile NPU on Smartphones
type: paper
url: https://dl.acm.org/doi/10.1145/3767295.3769382
status: unread
summary: 高通手机 NPU 的测试时计算扩展、分块量化与查表运算。
tags:
- topic/inference
- hardware/qualcomm
- backend/npu
- topic/quantization
- system/lut
links:
  code: https://github.com/haozixu/llama.cpp-npu
attachments:
- url: https://dl.acm.org/doi/pdf/10.1145/3767295.3769382?download=true
  path: resources/mobile-npu-test-time-compute-xnsvnidb.pdf
  source: zotero
- url: https://arxiv.org/pdf/2509.23324v1
  path: resources/mobile-npu-test-time-compute-arxiv-v1.pdf
  version: arXiv:2509.23324v1; preprint, not the missing EuroSys version
authors:
- Zixu Hao
- Jianyu Wei
- Tuowei Wang
- Minxing Huang
- Huiqiang Jiang
- Shiqi Jiang
- Ting Cao
- Ju Ren
year: 2026
doi: 10.1145/3767295.3769382
venue: Proceedings of the 21st European Conference on Computer Systems
code_review:
  checked_on: '2026-11-06'
  status: source-available
  license: MIT
  license_url: https://raw.githubusercontent.com/haozixu/llama.cpp-npu/htp-backend/LICENSE
  evidence:
  - https://arxiv.org/html/2509.23324
  - https://github.com/haozixu/llama.cpp-npu
  - https://raw.githubusercontent.com/haozixu/llama.cpp-npu/htp-backend/ggml/src/ggml-htp/ggml-htp.cc
  - https://raw.githubusercontent.com/haozixu/llama.cpp-npu/htp-backend/LICENSE
  - https://github.com/haozixu/htp-ops-lib
  - https://raw.githubusercontent.com/haozixu/htp-ops-lib/main/src/dsp/ops/flash_attn.c
  note: 论文与两个仓库 README 相互确认对应关系，主仓库 HTP 后端及独立算子库均有实际源码；MIT 仅核实于主仓库，必需的独立 htp-ops-lib 根目录及所查算子源码未找到明确许可，因此暂不将整套论文实现标为开源；非因 Hexagon SDK 限制否定主项目许可。
---

# Scaling LLM Test-Time Compute with Mobile NPU on Smartphones
