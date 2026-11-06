---
id: zeroquant
title: 'ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers'
type: paper
url: https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html
status: unread
summary: 结合权重激活量化、逐层蒸馏与后端优化的推理流水线。
tags:
- topic/quantization
- scheme/weight-activation
- scheme/ptq
- precision/w8a8
- topic/inference
- availability/open-source
links:
  code: https://github.com/deepspeedai/DeepSpeedExamples/tree/37bbe1364cc17fd2940038bafa45cb1dd958d9a2/model_compression/bert
code_review:
  checked_on: '2026-11-06'
  status: open-source
  license: MIT AND Apache-2.0
  license_url: https://raw.githubusercontent.com/deepspeedai/DeepSpeedExamples/37bbe1364cc17fd2940038bafa45cb1dd958d9a2/LICENSE
  evidence:
  - https://github.com/deepspeedai/DeepSpeed/issues/2207
  - https://github.com/deepspeedai/DeepSpeedExamples/pull/214
  - https://github.com/deepspeedai/DeepSpeedExamples/pull/214/files
  - https://github.com/deepspeedai/DeepSpeedExamples/tree/master/compression
  - https://raw.githubusercontent.com/deepspeedai/DeepSpeedExamples/37bbe1364cc17fd2940038bafa45cb1dd958d9a2/model_compression/bert/run_glue_lkd.py
  - https://raw.githubusercontent.com/deepspeedai/DeepSpeedExamples/37bbe1364cc17fd2940038bafa45cb1dd958d9a2/LICENSE
  note: 论文作者在官方 issue 指向已合并的 ZeroQuant LKD 示例；已核对实际脚本，固定历史提交根目录为 MIT，run_glue_lkd.py 自带 Apache-2.0 许可。当前主分支已删除旧压缩模块及示例，故代码链接固定历史版本；仅确认量化精度验证及 LKD 代码，本次未能核实论文完整加速推理引擎公开，不能以整个 DeepSpeed 框架替代该证据；未运行实现。
attachments:
- url: https://proceedings.neurips.cc/paper_files/paper/2022/file/adf7fa39d65e2983d724ff7da57f00ac-Paper-Conference.pdf
  path: resources/zeroquant.pdf
  source: zotero
- url: https://arxiv.org/pdf/2206.01861
  path: resources/zeroquant-zathrbd8.pdf
  source: zotero
  source_zotero_attachment_key: ZATHRBD8
  source_zotero_item_key: 9KJY9PPD
- url: https://arxiv.org/pdf/2206.01861
  path: resources/zeroquant-yicka85g.pdf
  source: zotero
  source_zotero_attachment_key: YICKA85G
  source_zotero_item_key: 3WUXDXS9
authors:
- Zhewei Yao
- Reza Yazdani Aminabadi
- Minjia Zhang
- Xiaoxia Wu
- Conglong Li
- Yuxiong He
year: 2022
venue: Advances in Neural Information Processing Systems
---

# ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers
