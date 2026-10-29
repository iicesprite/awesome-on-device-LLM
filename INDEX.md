# 资源索引

<!-- 自动生成；请修改各条目的 meta.yaml，然后运行 uv run python scripts/catalog.py build。 -->

共 **22** 条资源。按类型浏览，或在下方按标签交叉查阅。

[项目说明](README.md) · [CSV 索引](catalog.csv) · [标签词表](tags.yaml)

## papers（22）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [Extreme Compression of Large Language Models via Additive Quantization](https://proceedings.mlr.press/v235/egiazarian24a.html) | 论文 | 多码本加性权重量化与 CPU/GPU token generation 实现。 | backend/cpu、scheme/weight-only、target/weight、topic/inference、topic/quantization | 未读 | [aqlm](papers/aqlm/meta.yaml) | — |
| [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) | 论文 | 激活感知权重量化与 TinyChat 的打包、融合和端侧推理实现。 | hardware/jetson-orin、method/awq、precision/w4a16、scheme/weight-only、target/weight、topic/quantization | 未读 | [awq](papers/awq/meta.yaml) | — |
| [FlattenQuant: Breaking through the Inference Compute-bound for Large Language Models with Per-tensor Quantization](https://aclanthology.org/2024.lrec-main.648/) | 论文 | 通道展开实现逐张量量化，面向大 batch 或长序列计算受限场景。 | granularity/per-tensor、scheme/mixed-precision、scheme/weight-activation、topic/quantization | 未读 | [flattenquant](papers/flattenquant/meta.yaml) | — |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](http://arxiv.org/abs/2210.17323) | 论文 | 近似二阶信息驱动的一次性权重量化，作为算法与推理基线。 | method/gptq、scheme/ptq、scheme/weight-only、target/weight、topic/quantization | 未读 | [gptq](papers/gptq/meta.yaml) | — |
| [Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference](https://dl.acm.org/doi/10.1145/3731569.3764808) | 论文 | 移动 SoC 的 GPU–NPU 算力、带宽、同步与异构推理。 | backend/npu、memory/unified、system/heterogeneous、topic/inference | 未读 | [heteroinfer](papers/heteroinfer/meta.yaml) | — |
| [HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs platform with Heterogeneous AI Accelerators](http://arxiv.org/abs/2501.14794) | 论文 | 统一内存下的层级与张量级异构执行。 | memory/unified、stage/decode、stage/prefill、system/heterogeneous、topic/inference | 未读 | [heterollm](papers/heterollm/meta.yaml) | — |
| [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html) | 论文 | 非均匀及离群值感知 KV 量化与定制 CUDA 内核。 | backend/cuda、granularity/per-channel、method/kvquant、target/kv-cache、topic/quantization | 未读 | [kvquant](papers/kvquant/meta.yaml) | — |
| [KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference](https://arxiv.org/abs/2511.11907v2) | 论文 | 统一内存端侧设备的磁盘感知 KV cache 卸载与 I/O–计算重叠。 | memory/unified、system/offloading、target/kv-cache、topic/inference、topic/memory | 未读 | [kvswap](papers/kvswap/meta.yaml) | — |
| [Fast On-device LLM Inference with NPUs](http://arxiv.org/abs/2407.05858) | 论文 | 手机 NPU 的 prefill 卸载、离群值拆分与异构调度。 | backend/npu、stage/prefill、system/heterogeneous、system/offloading、topic/inference | 未读 | [llm-npu](papers/llm-npu/meta.yaml) | — |
| [MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators](https://openreview.net/forum?id=NXVXiJmhe1) | 论文 | 低秩补偿恢复低比特 MoE 精度，并设计 Tensor Core 友好的 3-bit 内核。 | architecture/moe、scheme/weight-only、target/weight、topic/inference、topic/quantization | 未读 | [milo](papers/milo/meta.yaml) | — |
| [MoBiE: Efficient Inference of Mixture of Binary Experts under Post-Training Quantization](http://arxiv.org/abs/2604.06798) | 论文 | 【已撤稿】二值 MoE 量化论文；作者撤稿说明涉及推导错误，不作为有效结论或性能依据。 | architecture/moe、publication/withdrawn、scheme/ptq、target/weight、topic/inference、topic/quantization | 未读 | [mobie](papers/mobie/meta.yaml) | — |
| [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://dl.acm.org/doi/10.1145/3767295.3769382) | 论文 | 高通手机 NPU 的测试时计算扩展、分块量化与查表运算。 | backend/npu、hardware/qualcomm、system/lut、topic/inference、topic/quantization | 未读 | [mobile-npu-test-time-compute](papers/mobile-npu-test-time-compute/meta.yaml) | — |
| [MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design](https://openreview.net/forum?id=pXoZLGMNDm) | 论文 | 联合精度分配、专家激活与硬件资源，生成混合精度 GroupGEMM 内核。 | architecture/moe、scheme/mixed-precision、scheme/weight-activation、topic/inference、topic/quantization | 未读 | [mxmoe](papers/mxmoe/meta.yaml) | — |
| [OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/29237) | 论文 | 敏感权重保留高精度，比较混合精度的存储和计算代价。 | scheme/mixed-precision、scheme/weight-only、target/weight、topic/quantization | 未读 | [owq](papers/owq/meta.yaml) | — |
| [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](http://arxiv.org/abs/2310.16795) | 论文 | 亚 1-bit MoE 压缩格式与 GPU 解码内核协同设计。 | architecture/moe、backend/cuda、target/weight、topic/memory、topic/quantization | 未读 | [qmoe](papers/qmoe/meta.yaml) | — |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](http://arxiv.org/abs/2404.00456) | 论文 | 旋转去除离群值，联合量化权重、激活和 KV cache。 | precision/w4a4、target/activation、target/kv-cache、target/weight、topic/quantization、transform/rotation | 未读 | [quarot](papers/quarot/meta.yaml) | — |
| [RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping](http://arxiv.org/abs/2501.04052) | 论文 | 围绕 NVFP4 冗余编码设计数值格式、GPU 内核和硬件支持。 | precision/fp4、target/activation、target/weight、topic/inference、topic/quantization | 未读 | [razer](papers/razer/meta.yaml) | — |
| [ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference](http://arxiv.org/abs/2508.16703) | 论文 | 针对 Attention 回退问题，协同设计稀疏 Attention 与 NPU 异构流水线。 | backend/npu、system/heterogeneous、system/sparse-inference、topic/inference | 未读 | [shadownpu](papers/shadownpu/meta.yaml) | — |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html) | 论文 | 等价变换处理激活离群值，连接校准与 W8A8 推理。 | method/smoothquant、precision/w8a8、target/activation、target/weight、topic/quantization | 未读 | [smoothquant](papers/smoothquant/meta.yaml) | — |
| [SqueezeLLM: Dense-and-Sparse Quantization](http://arxiv.org/abs/2306.07629) | 论文 | 非均匀权重量化与 dense/sparse 分解，面向带宽受限推理。 | scheme/ptq、scheme/weight-only、target/weight、topic/memory、topic/quantization | 未读 | [squeezellm](papers/squeezellm/meta.yaml) | — |
| [T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup](http://arxiv.org/abs/2511.11248) | 论文 | 统一查表布局、分块和流水线降低 NPU 低比特推理开销。 | backend/npu、system/lut、system/memory-layout、topic/inference、topic/quantization | 未读 | [t-man](papers/t-man/meta.yaml) | — |
| [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html) | 论文 | 结合权重激活量化、逐层蒸馏与后端优化的推理流水线。 | precision/w8a8、scheme/ptq、scheme/weight-activation、topic/inference、topic/quantization | 未读 | [zeroquant](papers/zeroquant/meta.yaml) | — |

## blogs（0）

暂无条目。

## docs（0）

暂无条目。

## 按标签浏览

### architecture/moe · 混合专家架构（4）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators](https://openreview.net/forum?id=NXVXiJmhe1) | 论文 | 低秩补偿恢复低比特 MoE 精度，并设计 Tensor Core 友好的 3-bit 内核。 | architecture/moe、scheme/weight-only、target/weight、topic/inference、topic/quantization | 未读 | [milo](papers/milo/meta.yaml) | — |
| [MoBiE: Efficient Inference of Mixture of Binary Experts under Post-Training Quantization](http://arxiv.org/abs/2604.06798) | 论文 | 【已撤稿】二值 MoE 量化论文；作者撤稿说明涉及推导错误，不作为有效结论或性能依据。 | architecture/moe、publication/withdrawn、scheme/ptq、target/weight、topic/inference、topic/quantization | 未读 | [mobie](papers/mobie/meta.yaml) | — |
| [MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design](https://openreview.net/forum?id=pXoZLGMNDm) | 论文 | 联合精度分配、专家激活与硬件资源，生成混合精度 GroupGEMM 内核。 | architecture/moe、scheme/mixed-precision、scheme/weight-activation、topic/inference、topic/quantization | 未读 | [mxmoe](papers/mxmoe/meta.yaml) | — |
| [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](http://arxiv.org/abs/2310.16795) | 论文 | 亚 1-bit MoE 压缩格式与 GPU 解码内核协同设计。 | architecture/moe、backend/cuda、target/weight、topic/memory、topic/quantization | 未读 | [qmoe](papers/qmoe/meta.yaml) | — |

### backend/cpu · CPU后端（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [Extreme Compression of Large Language Models via Additive Quantization](https://proceedings.mlr.press/v235/egiazarian24a.html) | 论文 | 多码本加性权重量化与 CPU/GPU token generation 实现。 | backend/cpu、scheme/weight-only、target/weight、topic/inference、topic/quantization | 未读 | [aqlm](papers/aqlm/meta.yaml) | — |

### backend/cuda · CUDA后端（2）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html) | 论文 | 非均匀及离群值感知 KV 量化与定制 CUDA 内核。 | backend/cuda、granularity/per-channel、method/kvquant、target/kv-cache、topic/quantization | 未读 | [kvquant](papers/kvquant/meta.yaml) | — |
| [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](http://arxiv.org/abs/2310.16795) | 论文 | 亚 1-bit MoE 压缩格式与 GPU 解码内核协同设计。 | architecture/moe、backend/cuda、target/weight、topic/memory、topic/quantization | 未读 | [qmoe](papers/qmoe/meta.yaml) | — |

### backend/npu · NPU后端（5）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference](https://dl.acm.org/doi/10.1145/3731569.3764808) | 论文 | 移动 SoC 的 GPU–NPU 算力、带宽、同步与异构推理。 | backend/npu、memory/unified、system/heterogeneous、topic/inference | 未读 | [heteroinfer](papers/heteroinfer/meta.yaml) | — |
| [Fast On-device LLM Inference with NPUs](http://arxiv.org/abs/2407.05858) | 论文 | 手机 NPU 的 prefill 卸载、离群值拆分与异构调度。 | backend/npu、stage/prefill、system/heterogeneous、system/offloading、topic/inference | 未读 | [llm-npu](papers/llm-npu/meta.yaml) | — |
| [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://dl.acm.org/doi/10.1145/3767295.3769382) | 论文 | 高通手机 NPU 的测试时计算扩展、分块量化与查表运算。 | backend/npu、hardware/qualcomm、system/lut、topic/inference、topic/quantization | 未读 | [mobile-npu-test-time-compute](papers/mobile-npu-test-time-compute/meta.yaml) | — |
| [ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference](http://arxiv.org/abs/2508.16703) | 论文 | 针对 Attention 回退问题，协同设计稀疏 Attention 与 NPU 异构流水线。 | backend/npu、system/heterogeneous、system/sparse-inference、topic/inference | 未读 | [shadownpu](papers/shadownpu/meta.yaml) | — |
| [T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup](http://arxiv.org/abs/2511.11248) | 论文 | 统一查表布局、分块和流水线降低 NPU 低比特推理开销。 | backend/npu、system/lut、system/memory-layout、topic/inference、topic/quantization | 未读 | [t-man](papers/t-man/meta.yaml) | — |

### granularity/per-channel · 逐通道量化（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html) | 论文 | 非均匀及离群值感知 KV 量化与定制 CUDA 内核。 | backend/cuda、granularity/per-channel、method/kvquant、target/kv-cache、topic/quantization | 未读 | [kvquant](papers/kvquant/meta.yaml) | — |

### granularity/per-tensor · 逐张量量化（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [FlattenQuant: Breaking through the Inference Compute-bound for Large Language Models with Per-tensor Quantization](https://aclanthology.org/2024.lrec-main.648/) | 论文 | 通道展开实现逐张量量化，面向大 batch 或长序列计算受限场景。 | granularity/per-tensor、scheme/mixed-precision、scheme/weight-activation、topic/quantization | 未读 | [flattenquant](papers/flattenquant/meta.yaml) | — |

### hardware/jetson-orin · Jetson Orin（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) | 论文 | 激活感知权重量化与 TinyChat 的打包、融合和端侧推理实现。 | hardware/jetson-orin、method/awq、precision/w4a16、scheme/weight-only、target/weight、topic/quantization | 未读 | [awq](papers/awq/meta.yaml) | — |

### hardware/qualcomm · 高通（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://dl.acm.org/doi/10.1145/3767295.3769382) | 论文 | 高通手机 NPU 的测试时计算扩展、分块量化与查表运算。 | backend/npu、hardware/qualcomm、system/lut、topic/inference、topic/quantization | 未读 | [mobile-npu-test-time-compute](papers/mobile-npu-test-time-compute/meta.yaml) | — |

### memory/unified · 统一内存（3）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference](https://dl.acm.org/doi/10.1145/3731569.3764808) | 论文 | 移动 SoC 的 GPU–NPU 算力、带宽、同步与异构推理。 | backend/npu、memory/unified、system/heterogeneous、topic/inference | 未读 | [heteroinfer](papers/heteroinfer/meta.yaml) | — |
| [HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs platform with Heterogeneous AI Accelerators](http://arxiv.org/abs/2501.14794) | 论文 | 统一内存下的层级与张量级异构执行。 | memory/unified、stage/decode、stage/prefill、system/heterogeneous、topic/inference | 未读 | [heterollm](papers/heterollm/meta.yaml) | — |
| [KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference](https://arxiv.org/abs/2511.11907v2) | 论文 | 统一内存端侧设备的磁盘感知 KV cache 卸载与 I/O–计算重叠。 | memory/unified、system/offloading、target/kv-cache、topic/inference、topic/memory | 未读 | [kvswap](papers/kvswap/meta.yaml) | — |

### method/awq · AWQ（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) | 论文 | 激活感知权重量化与 TinyChat 的打包、融合和端侧推理实现。 | hardware/jetson-orin、method/awq、precision/w4a16、scheme/weight-only、target/weight、topic/quantization | 未读 | [awq](papers/awq/meta.yaml) | — |

### method/gptq · GPTQ（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](http://arxiv.org/abs/2210.17323) | 论文 | 近似二阶信息驱动的一次性权重量化，作为算法与推理基线。 | method/gptq、scheme/ptq、scheme/weight-only、target/weight、topic/quantization | 未读 | [gptq](papers/gptq/meta.yaml) | — |

### method/kvquant · KVQuant（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html) | 论文 | 非均匀及离群值感知 KV 量化与定制 CUDA 内核。 | backend/cuda、granularity/per-channel、method/kvquant、target/kv-cache、topic/quantization | 未读 | [kvquant](papers/kvquant/meta.yaml) | — |

### method/smoothquant · SmoothQuant（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html) | 论文 | 等价变换处理激活离群值，连接校准与 W8A8 推理。 | method/smoothquant、precision/w8a8、target/activation、target/weight、topic/quantization | 未读 | [smoothquant](papers/smoothquant/meta.yaml) | — |

### precision/fp4 · FP4（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping](http://arxiv.org/abs/2501.04052) | 论文 | 围绕 NVFP4 冗余编码设计数值格式、GPU 内核和硬件支持。 | precision/fp4、target/activation、target/weight、topic/inference、topic/quantization | 未读 | [razer](papers/razer/meta.yaml) | — |

### precision/w4a16 · W4A16（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) | 论文 | 激活感知权重量化与 TinyChat 的打包、融合和端侧推理实现。 | hardware/jetson-orin、method/awq、precision/w4a16、scheme/weight-only、target/weight、topic/quantization | 未读 | [awq](papers/awq/meta.yaml) | — |

### precision/w4a4 · W4A4（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](http://arxiv.org/abs/2404.00456) | 论文 | 旋转去除离群值，联合量化权重、激活和 KV cache。 | precision/w4a4、target/activation、target/kv-cache、target/weight、topic/quantization、transform/rotation | 未读 | [quarot](papers/quarot/meta.yaml) | — |

### precision/w8a8 · W8A8（2）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html) | 论文 | 等价变换处理激活离群值，连接校准与 W8A8 推理。 | method/smoothquant、precision/w8a8、target/activation、target/weight、topic/quantization | 未读 | [smoothquant](papers/smoothquant/meta.yaml) | — |
| [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html) | 论文 | 结合权重激活量化、逐层蒸馏与后端优化的推理流水线。 | precision/w8a8、scheme/ptq、scheme/weight-activation、topic/inference、topic/quantization | 未读 | [zeroquant](papers/zeroquant/meta.yaml) | — |

### publication/withdrawn · 已撤稿（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [MoBiE: Efficient Inference of Mixture of Binary Experts under Post-Training Quantization](http://arxiv.org/abs/2604.06798) | 论文 | 【已撤稿】二值 MoE 量化论文；作者撤稿说明涉及推导错误，不作为有效结论或性能依据。 | architecture/moe、publication/withdrawn、scheme/ptq、target/weight、topic/inference、topic/quantization | 未读 | [mobie](papers/mobie/meta.yaml) | — |

### scheme/mixed-precision · 混合精度（3）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [FlattenQuant: Breaking through the Inference Compute-bound for Large Language Models with Per-tensor Quantization](https://aclanthology.org/2024.lrec-main.648/) | 论文 | 通道展开实现逐张量量化，面向大 batch 或长序列计算受限场景。 | granularity/per-tensor、scheme/mixed-precision、scheme/weight-activation、topic/quantization | 未读 | [flattenquant](papers/flattenquant/meta.yaml) | — |
| [MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design](https://openreview.net/forum?id=pXoZLGMNDm) | 论文 | 联合精度分配、专家激活与硬件资源，生成混合精度 GroupGEMM 内核。 | architecture/moe、scheme/mixed-precision、scheme/weight-activation、topic/inference、topic/quantization | 未读 | [mxmoe](papers/mxmoe/meta.yaml) | — |
| [OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/29237) | 论文 | 敏感权重保留高精度，比较混合精度的存储和计算代价。 | scheme/mixed-precision、scheme/weight-only、target/weight、topic/quantization | 未读 | [owq](papers/owq/meta.yaml) | — |

### scheme/ptq · 训练后量化（4）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](http://arxiv.org/abs/2210.17323) | 论文 | 近似二阶信息驱动的一次性权重量化，作为算法与推理基线。 | method/gptq、scheme/ptq、scheme/weight-only、target/weight、topic/quantization | 未读 | [gptq](papers/gptq/meta.yaml) | — |
| [MoBiE: Efficient Inference of Mixture of Binary Experts under Post-Training Quantization](http://arxiv.org/abs/2604.06798) | 论文 | 【已撤稿】二值 MoE 量化论文；作者撤稿说明涉及推导错误，不作为有效结论或性能依据。 | architecture/moe、publication/withdrawn、scheme/ptq、target/weight、topic/inference、topic/quantization | 未读 | [mobie](papers/mobie/meta.yaml) | — |
| [SqueezeLLM: Dense-and-Sparse Quantization](http://arxiv.org/abs/2306.07629) | 论文 | 非均匀权重量化与 dense/sparse 分解，面向带宽受限推理。 | scheme/ptq、scheme/weight-only、target/weight、topic/memory、topic/quantization | 未读 | [squeezellm](papers/squeezellm/meta.yaml) | — |
| [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html) | 论文 | 结合权重激活量化、逐层蒸馏与后端优化的推理流水线。 | precision/w8a8、scheme/ptq、scheme/weight-activation、topic/inference、topic/quantization | 未读 | [zeroquant](papers/zeroquant/meta.yaml) | — |

### scheme/weight-activation · 权重激活联合量化（3）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [FlattenQuant: Breaking through the Inference Compute-bound for Large Language Models with Per-tensor Quantization](https://aclanthology.org/2024.lrec-main.648/) | 论文 | 通道展开实现逐张量量化，面向大 batch 或长序列计算受限场景。 | granularity/per-tensor、scheme/mixed-precision、scheme/weight-activation、topic/quantization | 未读 | [flattenquant](papers/flattenquant/meta.yaml) | — |
| [MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design](https://openreview.net/forum?id=pXoZLGMNDm) | 论文 | 联合精度分配、专家激活与硬件资源，生成混合精度 GroupGEMM 内核。 | architecture/moe、scheme/mixed-precision、scheme/weight-activation、topic/inference、topic/quantization | 未读 | [mxmoe](papers/mxmoe/meta.yaml) | — |
| [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html) | 论文 | 结合权重激活量化、逐层蒸馏与后端优化的推理流水线。 | precision/w8a8、scheme/ptq、scheme/weight-activation、topic/inference、topic/quantization | 未读 | [zeroquant](papers/zeroquant/meta.yaml) | — |

### scheme/weight-only · 仅权重量化（6）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [Extreme Compression of Large Language Models via Additive Quantization](https://proceedings.mlr.press/v235/egiazarian24a.html) | 论文 | 多码本加性权重量化与 CPU/GPU token generation 实现。 | backend/cpu、scheme/weight-only、target/weight、topic/inference、topic/quantization | 未读 | [aqlm](papers/aqlm/meta.yaml) | — |
| [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) | 论文 | 激活感知权重量化与 TinyChat 的打包、融合和端侧推理实现。 | hardware/jetson-orin、method/awq、precision/w4a16、scheme/weight-only、target/weight、topic/quantization | 未读 | [awq](papers/awq/meta.yaml) | — |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](http://arxiv.org/abs/2210.17323) | 论文 | 近似二阶信息驱动的一次性权重量化，作为算法与推理基线。 | method/gptq、scheme/ptq、scheme/weight-only、target/weight、topic/quantization | 未读 | [gptq](papers/gptq/meta.yaml) | — |
| [MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators](https://openreview.net/forum?id=NXVXiJmhe1) | 论文 | 低秩补偿恢复低比特 MoE 精度，并设计 Tensor Core 友好的 3-bit 内核。 | architecture/moe、scheme/weight-only、target/weight、topic/inference、topic/quantization | 未读 | [milo](papers/milo/meta.yaml) | — |
| [OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/29237) | 论文 | 敏感权重保留高精度，比较混合精度的存储和计算代价。 | scheme/mixed-precision、scheme/weight-only、target/weight、topic/quantization | 未读 | [owq](papers/owq/meta.yaml) | — |
| [SqueezeLLM: Dense-and-Sparse Quantization](http://arxiv.org/abs/2306.07629) | 论文 | 非均匀权重量化与 dense/sparse 分解，面向带宽受限推理。 | scheme/ptq、scheme/weight-only、target/weight、topic/memory、topic/quantization | 未读 | [squeezellm](papers/squeezellm/meta.yaml) | — |

### stage/decode · 解码（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs platform with Heterogeneous AI Accelerators](http://arxiv.org/abs/2501.14794) | 论文 | 统一内存下的层级与张量级异构执行。 | memory/unified、stage/decode、stage/prefill、system/heterogeneous、topic/inference | 未读 | [heterollm](papers/heterollm/meta.yaml) | — |

### stage/prefill · 预填充（2）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs platform with Heterogeneous AI Accelerators](http://arxiv.org/abs/2501.14794) | 论文 | 统一内存下的层级与张量级异构执行。 | memory/unified、stage/decode、stage/prefill、system/heterogeneous、topic/inference | 未读 | [heterollm](papers/heterollm/meta.yaml) | — |
| [Fast On-device LLM Inference with NPUs](http://arxiv.org/abs/2407.05858) | 论文 | 手机 NPU 的 prefill 卸载、离群值拆分与异构调度。 | backend/npu、stage/prefill、system/heterogeneous、system/offloading、topic/inference | 未读 | [llm-npu](papers/llm-npu/meta.yaml) | — |

### system/heterogeneous · 异构计算（4）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference](https://dl.acm.org/doi/10.1145/3731569.3764808) | 论文 | 移动 SoC 的 GPU–NPU 算力、带宽、同步与异构推理。 | backend/npu、memory/unified、system/heterogeneous、topic/inference | 未读 | [heteroinfer](papers/heteroinfer/meta.yaml) | — |
| [HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs platform with Heterogeneous AI Accelerators](http://arxiv.org/abs/2501.14794) | 论文 | 统一内存下的层级与张量级异构执行。 | memory/unified、stage/decode、stage/prefill、system/heterogeneous、topic/inference | 未读 | [heterollm](papers/heterollm/meta.yaml) | — |
| [Fast On-device LLM Inference with NPUs](http://arxiv.org/abs/2407.05858) | 论文 | 手机 NPU 的 prefill 卸载、离群值拆分与异构调度。 | backend/npu、stage/prefill、system/heterogeneous、system/offloading、topic/inference | 未读 | [llm-npu](papers/llm-npu/meta.yaml) | — |
| [ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference](http://arxiv.org/abs/2508.16703) | 论文 | 针对 Attention 回退问题，协同设计稀疏 Attention 与 NPU 异构流水线。 | backend/npu、system/heterogeneous、system/sparse-inference、topic/inference | 未读 | [shadownpu](papers/shadownpu/meta.yaml) | — |

### system/lut · 查表计算（2）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://dl.acm.org/doi/10.1145/3767295.3769382) | 论文 | 高通手机 NPU 的测试时计算扩展、分块量化与查表运算。 | backend/npu、hardware/qualcomm、system/lut、topic/inference、topic/quantization | 未读 | [mobile-npu-test-time-compute](papers/mobile-npu-test-time-compute/meta.yaml) | — |
| [T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup](http://arxiv.org/abs/2511.11248) | 论文 | 统一查表布局、分块和流水线降低 NPU 低比特推理开销。 | backend/npu、system/lut、system/memory-layout、topic/inference、topic/quantization | 未读 | [t-man](papers/t-man/meta.yaml) | — |

### system/memory-layout · 内存布局（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup](http://arxiv.org/abs/2511.11248) | 论文 | 统一查表布局、分块和流水线降低 NPU 低比特推理开销。 | backend/npu、system/lut、system/memory-layout、topic/inference、topic/quantization | 未读 | [t-man](papers/t-man/meta.yaml) | — |

### system/offloading · 计算与存储卸载（2）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference](https://arxiv.org/abs/2511.11907v2) | 论文 | 统一内存端侧设备的磁盘感知 KV cache 卸载与 I/O–计算重叠。 | memory/unified、system/offloading、target/kv-cache、topic/inference、topic/memory | 未读 | [kvswap](papers/kvswap/meta.yaml) | — |
| [Fast On-device LLM Inference with NPUs](http://arxiv.org/abs/2407.05858) | 论文 | 手机 NPU 的 prefill 卸载、离群值拆分与异构调度。 | backend/npu、stage/prefill、system/heterogeneous、system/offloading、topic/inference | 未读 | [llm-npu](papers/llm-npu/meta.yaml) | — |

### system/sparse-inference · 稀疏推理（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference](http://arxiv.org/abs/2508.16703) | 论文 | 针对 Attention 回退问题，协同设计稀疏 Attention 与 NPU 异构流水线。 | backend/npu、system/heterogeneous、system/sparse-inference、topic/inference | 未读 | [shadownpu](papers/shadownpu/meta.yaml) | — |

### target/activation · 激活（3）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](http://arxiv.org/abs/2404.00456) | 论文 | 旋转去除离群值，联合量化权重、激活和 KV cache。 | precision/w4a4、target/activation、target/kv-cache、target/weight、topic/quantization、transform/rotation | 未读 | [quarot](papers/quarot/meta.yaml) | — |
| [RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping](http://arxiv.org/abs/2501.04052) | 论文 | 围绕 NVFP4 冗余编码设计数值格式、GPU 内核和硬件支持。 | precision/fp4、target/activation、target/weight、topic/inference、topic/quantization | 未读 | [razer](papers/razer/meta.yaml) | — |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html) | 论文 | 等价变换处理激活离群值，连接校准与 W8A8 推理。 | method/smoothquant、precision/w8a8、target/activation、target/weight、topic/quantization | 未读 | [smoothquant](papers/smoothquant/meta.yaml) | — |

### target/kv-cache · KV缓存（3）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html) | 论文 | 非均匀及离群值感知 KV 量化与定制 CUDA 内核。 | backend/cuda、granularity/per-channel、method/kvquant、target/kv-cache、topic/quantization | 未读 | [kvquant](papers/kvquant/meta.yaml) | — |
| [KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference](https://arxiv.org/abs/2511.11907v2) | 论文 | 统一内存端侧设备的磁盘感知 KV cache 卸载与 I/O–计算重叠。 | memory/unified、system/offloading、target/kv-cache、topic/inference、topic/memory | 未读 | [kvswap](papers/kvswap/meta.yaml) | — |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](http://arxiv.org/abs/2404.00456) | 论文 | 旋转去除离群值，联合量化权重、激活和 KV cache。 | precision/w4a4、target/activation、target/kv-cache、target/weight、topic/quantization、transform/rotation | 未读 | [quarot](papers/quarot/meta.yaml) | — |

### target/weight · 权重（11）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [Extreme Compression of Large Language Models via Additive Quantization](https://proceedings.mlr.press/v235/egiazarian24a.html) | 论文 | 多码本加性权重量化与 CPU/GPU token generation 实现。 | backend/cpu、scheme/weight-only、target/weight、topic/inference、topic/quantization | 未读 | [aqlm](papers/aqlm/meta.yaml) | — |
| [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) | 论文 | 激活感知权重量化与 TinyChat 的打包、融合和端侧推理实现。 | hardware/jetson-orin、method/awq、precision/w4a16、scheme/weight-only、target/weight、topic/quantization | 未读 | [awq](papers/awq/meta.yaml) | — |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](http://arxiv.org/abs/2210.17323) | 论文 | 近似二阶信息驱动的一次性权重量化，作为算法与推理基线。 | method/gptq、scheme/ptq、scheme/weight-only、target/weight、topic/quantization | 未读 | [gptq](papers/gptq/meta.yaml) | — |
| [MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators](https://openreview.net/forum?id=NXVXiJmhe1) | 论文 | 低秩补偿恢复低比特 MoE 精度，并设计 Tensor Core 友好的 3-bit 内核。 | architecture/moe、scheme/weight-only、target/weight、topic/inference、topic/quantization | 未读 | [milo](papers/milo/meta.yaml) | — |
| [MoBiE: Efficient Inference of Mixture of Binary Experts under Post-Training Quantization](http://arxiv.org/abs/2604.06798) | 论文 | 【已撤稿】二值 MoE 量化论文；作者撤稿说明涉及推导错误，不作为有效结论或性能依据。 | architecture/moe、publication/withdrawn、scheme/ptq、target/weight、topic/inference、topic/quantization | 未读 | [mobie](papers/mobie/meta.yaml) | — |
| [OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/29237) | 论文 | 敏感权重保留高精度，比较混合精度的存储和计算代价。 | scheme/mixed-precision、scheme/weight-only、target/weight、topic/quantization | 未读 | [owq](papers/owq/meta.yaml) | — |
| [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](http://arxiv.org/abs/2310.16795) | 论文 | 亚 1-bit MoE 压缩格式与 GPU 解码内核协同设计。 | architecture/moe、backend/cuda、target/weight、topic/memory、topic/quantization | 未读 | [qmoe](papers/qmoe/meta.yaml) | — |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](http://arxiv.org/abs/2404.00456) | 论文 | 旋转去除离群值，联合量化权重、激活和 KV cache。 | precision/w4a4、target/activation、target/kv-cache、target/weight、topic/quantization、transform/rotation | 未读 | [quarot](papers/quarot/meta.yaml) | — |
| [RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping](http://arxiv.org/abs/2501.04052) | 论文 | 围绕 NVFP4 冗余编码设计数值格式、GPU 内核和硬件支持。 | precision/fp4、target/activation、target/weight、topic/inference、topic/quantization | 未读 | [razer](papers/razer/meta.yaml) | — |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html) | 论文 | 等价变换处理激活离群值，连接校准与 W8A8 推理。 | method/smoothquant、precision/w8a8、target/activation、target/weight、topic/quantization | 未读 | [smoothquant](papers/smoothquant/meta.yaml) | — |
| [SqueezeLLM: Dense-and-Sparse Quantization](http://arxiv.org/abs/2306.07629) | 论文 | 非均匀权重量化与 dense/sparse 分解，面向带宽受限推理。 | scheme/ptq、scheme/weight-only、target/weight、topic/memory、topic/quantization | 未读 | [squeezellm](papers/squeezellm/meta.yaml) | — |

### topic/inference · 推理加速（13）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [Extreme Compression of Large Language Models via Additive Quantization](https://proceedings.mlr.press/v235/egiazarian24a.html) | 论文 | 多码本加性权重量化与 CPU/GPU token generation 实现。 | backend/cpu、scheme/weight-only、target/weight、topic/inference、topic/quantization | 未读 | [aqlm](papers/aqlm/meta.yaml) | — |
| [Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference](https://dl.acm.org/doi/10.1145/3731569.3764808) | 论文 | 移动 SoC 的 GPU–NPU 算力、带宽、同步与异构推理。 | backend/npu、memory/unified、system/heterogeneous、topic/inference | 未读 | [heteroinfer](papers/heteroinfer/meta.yaml) | — |
| [HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs platform with Heterogeneous AI Accelerators](http://arxiv.org/abs/2501.14794) | 论文 | 统一内存下的层级与张量级异构执行。 | memory/unified、stage/decode、stage/prefill、system/heterogeneous、topic/inference | 未读 | [heterollm](papers/heterollm/meta.yaml) | — |
| [KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference](https://arxiv.org/abs/2511.11907v2) | 论文 | 统一内存端侧设备的磁盘感知 KV cache 卸载与 I/O–计算重叠。 | memory/unified、system/offloading、target/kv-cache、topic/inference、topic/memory | 未读 | [kvswap](papers/kvswap/meta.yaml) | — |
| [Fast On-device LLM Inference with NPUs](http://arxiv.org/abs/2407.05858) | 论文 | 手机 NPU 的 prefill 卸载、离群值拆分与异构调度。 | backend/npu、stage/prefill、system/heterogeneous、system/offloading、topic/inference | 未读 | [llm-npu](papers/llm-npu/meta.yaml) | — |
| [MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators](https://openreview.net/forum?id=NXVXiJmhe1) | 论文 | 低秩补偿恢复低比特 MoE 精度，并设计 Tensor Core 友好的 3-bit 内核。 | architecture/moe、scheme/weight-only、target/weight、topic/inference、topic/quantization | 未读 | [milo](papers/milo/meta.yaml) | — |
| [MoBiE: Efficient Inference of Mixture of Binary Experts under Post-Training Quantization](http://arxiv.org/abs/2604.06798) | 论文 | 【已撤稿】二值 MoE 量化论文；作者撤稿说明涉及推导错误，不作为有效结论或性能依据。 | architecture/moe、publication/withdrawn、scheme/ptq、target/weight、topic/inference、topic/quantization | 未读 | [mobie](papers/mobie/meta.yaml) | — |
| [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://dl.acm.org/doi/10.1145/3767295.3769382) | 论文 | 高通手机 NPU 的测试时计算扩展、分块量化与查表运算。 | backend/npu、hardware/qualcomm、system/lut、topic/inference、topic/quantization | 未读 | [mobile-npu-test-time-compute](papers/mobile-npu-test-time-compute/meta.yaml) | — |
| [MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design](https://openreview.net/forum?id=pXoZLGMNDm) | 论文 | 联合精度分配、专家激活与硬件资源，生成混合精度 GroupGEMM 内核。 | architecture/moe、scheme/mixed-precision、scheme/weight-activation、topic/inference、topic/quantization | 未读 | [mxmoe](papers/mxmoe/meta.yaml) | — |
| [RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping](http://arxiv.org/abs/2501.04052) | 论文 | 围绕 NVFP4 冗余编码设计数值格式、GPU 内核和硬件支持。 | precision/fp4、target/activation、target/weight、topic/inference、topic/quantization | 未读 | [razer](papers/razer/meta.yaml) | — |
| [ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference](http://arxiv.org/abs/2508.16703) | 论文 | 针对 Attention 回退问题，协同设计稀疏 Attention 与 NPU 异构流水线。 | backend/npu、system/heterogeneous、system/sparse-inference、topic/inference | 未读 | [shadownpu](papers/shadownpu/meta.yaml) | — |
| [T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup](http://arxiv.org/abs/2511.11248) | 论文 | 统一查表布局、分块和流水线降低 NPU 低比特推理开销。 | backend/npu、system/lut、system/memory-layout、topic/inference、topic/quantization | 未读 | [t-man](papers/t-man/meta.yaml) | — |
| [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html) | 论文 | 结合权重激活量化、逐层蒸馏与后端优化的推理流水线。 | precision/w8a8、scheme/ptq、scheme/weight-activation、topic/inference、topic/quantization | 未读 | [zeroquant](papers/zeroquant/meta.yaml) | — |

### topic/memory · 内存优化（3）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference](https://arxiv.org/abs/2511.11907v2) | 论文 | 统一内存端侧设备的磁盘感知 KV cache 卸载与 I/O–计算重叠。 | memory/unified、system/offloading、target/kv-cache、topic/inference、topic/memory | 未读 | [kvswap](papers/kvswap/meta.yaml) | — |
| [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](http://arxiv.org/abs/2310.16795) | 论文 | 亚 1-bit MoE 压缩格式与 GPU 解码内核协同设计。 | architecture/moe、backend/cuda、target/weight、topic/memory、topic/quantization | 未读 | [qmoe](papers/qmoe/meta.yaml) | — |
| [SqueezeLLM: Dense-and-Sparse Quantization](http://arxiv.org/abs/2306.07629) | 论文 | 非均匀权重量化与 dense/sparse 分解，面向带宽受限推理。 | scheme/ptq、scheme/weight-only、target/weight、topic/memory、topic/quantization | 未读 | [squeezellm](papers/squeezellm/meta.yaml) | — |

### topic/quantization · 量化（17）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [Extreme Compression of Large Language Models via Additive Quantization](https://proceedings.mlr.press/v235/egiazarian24a.html) | 论文 | 多码本加性权重量化与 CPU/GPU token generation 实现。 | backend/cpu、scheme/weight-only、target/weight、topic/inference、topic/quantization | 未读 | [aqlm](papers/aqlm/meta.yaml) | — |
| [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) | 论文 | 激活感知权重量化与 TinyChat 的打包、融合和端侧推理实现。 | hardware/jetson-orin、method/awq、precision/w4a16、scheme/weight-only、target/weight、topic/quantization | 未读 | [awq](papers/awq/meta.yaml) | — |
| [FlattenQuant: Breaking through the Inference Compute-bound for Large Language Models with Per-tensor Quantization](https://aclanthology.org/2024.lrec-main.648/) | 论文 | 通道展开实现逐张量量化，面向大 batch 或长序列计算受限场景。 | granularity/per-tensor、scheme/mixed-precision、scheme/weight-activation、topic/quantization | 未读 | [flattenquant](papers/flattenquant/meta.yaml) | — |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](http://arxiv.org/abs/2210.17323) | 论文 | 近似二阶信息驱动的一次性权重量化，作为算法与推理基线。 | method/gptq、scheme/ptq、scheme/weight-only、target/weight、topic/quantization | 未读 | [gptq](papers/gptq/meta.yaml) | — |
| [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html) | 论文 | 非均匀及离群值感知 KV 量化与定制 CUDA 内核。 | backend/cuda、granularity/per-channel、method/kvquant、target/kv-cache、topic/quantization | 未读 | [kvquant](papers/kvquant/meta.yaml) | — |
| [MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators](https://openreview.net/forum?id=NXVXiJmhe1) | 论文 | 低秩补偿恢复低比特 MoE 精度，并设计 Tensor Core 友好的 3-bit 内核。 | architecture/moe、scheme/weight-only、target/weight、topic/inference、topic/quantization | 未读 | [milo](papers/milo/meta.yaml) | — |
| [MoBiE: Efficient Inference of Mixture of Binary Experts under Post-Training Quantization](http://arxiv.org/abs/2604.06798) | 论文 | 【已撤稿】二值 MoE 量化论文；作者撤稿说明涉及推导错误，不作为有效结论或性能依据。 | architecture/moe、publication/withdrawn、scheme/ptq、target/weight、topic/inference、topic/quantization | 未读 | [mobie](papers/mobie/meta.yaml) | — |
| [Scaling LLM Test-Time Compute with Mobile NPU on Smartphones](https://dl.acm.org/doi/10.1145/3767295.3769382) | 论文 | 高通手机 NPU 的测试时计算扩展、分块量化与查表运算。 | backend/npu、hardware/qualcomm、system/lut、topic/inference、topic/quantization | 未读 | [mobile-npu-test-time-compute](papers/mobile-npu-test-time-compute/meta.yaml) | — |
| [MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design](https://openreview.net/forum?id=pXoZLGMNDm) | 论文 | 联合精度分配、专家激活与硬件资源，生成混合精度 GroupGEMM 内核。 | architecture/moe、scheme/mixed-precision、scheme/weight-activation、topic/inference、topic/quantization | 未读 | [mxmoe](papers/mxmoe/meta.yaml) | — |
| [OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/29237) | 论文 | 敏感权重保留高精度，比较混合精度的存储和计算代价。 | scheme/mixed-precision、scheme/weight-only、target/weight、topic/quantization | 未读 | [owq](papers/owq/meta.yaml) | — |
| [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](http://arxiv.org/abs/2310.16795) | 论文 | 亚 1-bit MoE 压缩格式与 GPU 解码内核协同设计。 | architecture/moe、backend/cuda、target/weight、topic/memory、topic/quantization | 未读 | [qmoe](papers/qmoe/meta.yaml) | — |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](http://arxiv.org/abs/2404.00456) | 论文 | 旋转去除离群值，联合量化权重、激活和 KV cache。 | precision/w4a4、target/activation、target/kv-cache、target/weight、topic/quantization、transform/rotation | 未读 | [quarot](papers/quarot/meta.yaml) | — |
| [RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping](http://arxiv.org/abs/2501.04052) | 论文 | 围绕 NVFP4 冗余编码设计数值格式、GPU 内核和硬件支持。 | precision/fp4、target/activation、target/weight、topic/inference、topic/quantization | 未读 | [razer](papers/razer/meta.yaml) | — |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html) | 论文 | 等价变换处理激活离群值，连接校准与 W8A8 推理。 | method/smoothquant、precision/w8a8、target/activation、target/weight、topic/quantization | 未读 | [smoothquant](papers/smoothquant/meta.yaml) | — |
| [SqueezeLLM: Dense-and-Sparse Quantization](http://arxiv.org/abs/2306.07629) | 论文 | 非均匀权重量化与 dense/sparse 分解，面向带宽受限推理。 | scheme/ptq、scheme/weight-only、target/weight、topic/memory、topic/quantization | 未读 | [squeezellm](papers/squeezellm/meta.yaml) | — |
| [T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup](http://arxiv.org/abs/2511.11248) | 论文 | 统一查表布局、分块和流水线降低 NPU 低比特推理开销。 | backend/npu、system/lut、system/memory-layout、topic/inference、topic/quantization | 未读 | [t-man](papers/t-man/meta.yaml) | — |
| [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/adf7fa39d65e2983d724ff7da57f00ac-Abstract-Conference.html) | 论文 | 结合权重激活量化、逐层蒸馏与后端优化的推理流水线。 | precision/w8a8、scheme/ptq、scheme/weight-activation、topic/inference、topic/quantization | 未读 | [zeroquant](papers/zeroquant/meta.yaml) | — |

### transform/rotation · 旋转变换（1）

| 资源 | 类型 | 收藏理由 | 标签 | 状态 | 元数据 | 笔记 |
| --- | --- | --- | --- | --- | --- | --- |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](http://arxiv.org/abs/2404.00456) | 论文 | 旋转去除离群值，联合量化权重、激活和 KV cache。 | precision/w4a4、target/activation、target/kv-cache、target/weight、topic/quantization、transform/rotation | 未读 | [quarot](papers/quarot/meta.yaml) | — |
