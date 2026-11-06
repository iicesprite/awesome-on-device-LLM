---
id: floe
title: 'FloE: On-the-Fly MoE Inference on Memory-constrained GPU'
type: paper
url: https://proceedings.mlr.press/v267/zhou25j.html
status: unread
summary: 将专家内部稀疏化、up projection 低比特权重量化与稀疏预测结合，流水化专家传输和计算，在有限显存上运行 MoE。
tags:
- topic/inference
- topic/memory
- topic/quantization
- architecture/moe
- target/weight
- scheme/weight-only
- scheme/mixed-precision
- system/offloading
- system/sparse-inference
- system/pipelining
- system/operator-fusion
- system/memory-layout
- stage/decode
- memory/discrete
- hardware/nvidia-dgpu
- hardware/x86
- backend/cuda
- evidence/end-to-end
links:
  arxiv: https://arxiv.org/abs/2505.05950
  code: https://github.com/zju-stu-lizheng/FloE
attachments:
- url: https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhou25j/zhou25j.pdf
  path: resources/floe-icml2025.pdf
  version: ICML 2025 proceedings
authors:
- Yuxin Zhou
- Zheng Li
- Jun Zhang
- Jue Wang
- Yiping Wang
- Zhongle Xie
- Ke Chen
- Lidan Shou
year: 2025
venue: Proceedings of the 42nd International Conference on Machine Learning
source_checked_on: '2026-10-30'
version_note: '使用 PMLR 267 的发表版题名；arXiv 预印本题名为 FloE: On-the-Fly MoE Inference。'
implementation_evidence:
- tag: evidence/end-to-end
  url: https://github.com/zju-stu-lizheng/FloE/blob/main/start.py
  detail: 已检查 HQQ 量化模型加载、PipelineLLM 转换和 llm.generate 评测路径；脚本中的模型、阈值及数据集路径需配置。未运行复现。
code_review:
  checked_on: '2026-11-06'
  status: source-available
  evidence:
  - https://github.com/zju-stu-lizheng/FloE
  - https://raw.githubusercontent.com/zju-stu-lizheng/FloE/main/start.py
  - https://raw.githubusercontent.com/zju-stu-lizheng/FloE/main/kernel/kernels.py
  note: 仓库明确对应论文题名及 arXiv 2505.05950，已核对模型加载、生成入口和 Triton 稀疏 GEMV 实现；本次在根目录、README 及所查源码中未找到适用于本项目的明确开源许可，不以依赖库许可替代。未运行代码。
---

# FloE: On-the-Fly MoE Inference on Memory-constrained GPU
