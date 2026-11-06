---
id: helm
title: 'HELM: Compiler-Guided Heterogeneous Execution for Large Language Models on Memory-Constrained Systems'
type: paper
url: https://doi.org/10.1109/TCAD.2026.3729371
status: unread
summary: 面向独显容量不足的稠密 LLM，以实测 roofline 模型选择静态 CPU→GPU 层划分，结合 x86 GEMV 与分页 KV；重点区分 decode 收益、prefill 代价及真正分页时的带宽限制。
tags:
- topic/inference
- topic/memory
- topic/compilation
- target/weight
- target/kv-cache
- system/heterogeneous
- system/offloading
- system/pipelining
- workload/long-context
- stage/prefill
- stage/decode
- memory/discrete
- hardware/nvidia-dgpu
- hardware/x86
- backend/cpu
- backend/cuda
- evidence/end-to-end
- availability/open-source
links:
  publisher: https://ieeexplore.ieee.org/document/11673279/
  author-publications: https://mpslab-asu.github.io/publications/
  code: https://github.com/MPSLab-ASU/HELM
  artifact: https://github.com/MPSLab-ASU/HELM/blob/main/ARTIFACT.md
attachments:
- url: https://mpslab-asu.github.io/publications/papers/Khedkar2026CASES.pdf
  path: resources/helm-tcad2026.pdf
  version: Author-hosted IEEE TCAD 2026 article, 14 pages
authors:
- Atharva Khedkar
- Shashwat Pandey
- Aviral Shrivastava
year: 2026
venue: IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (Early Access; ESWEEK CASES 2026)
doi: 10.1109/TCAD.2026.3729371
first_published: '2026-08-31'
source_checked_on: '2026-10-30'
tag_note: system/pipelining 指 KV 页面预取与 attention 重叠，不表示同一 token 的 CPU/GPU 层并行；本文未实现低比特量化，因此不标量化内核。实现证据来自论文实验与官方源码只读核对，未本机运行。
code_review:
  checked_on: '2026-11-06'
  status: open-source
  license: Apache-2.0
  license_url: https://raw.githubusercontent.com/MPSLab-ASU/HELM/main/LICENSE
  evidence:
  - https://github.com/MPSLab-ASU/HELM
  - https://raw.githubusercontent.com/MPSLab-ASU/HELM/main/CITATION.cff
  - https://raw.githubusercontent.com/MPSLab-ASU/HELM/main/helm/_pipeline.py
  - https://raw.githubusercontent.com/MPSLab-ASU/HELM/main/LICENSE
  note: CITATION.cff 的题名、作者及论文 DOI 与资源一致；已核对实际编译和执行代码及根目录 Apache-2.0 许可正文。仅核实公开实现和许可，未运行复现。
---

# HELM: Compiler-Guided Heterogeneous Execution for Large Language Models on Memory-Constrained Systems
