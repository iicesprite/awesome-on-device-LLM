# ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs

来源：[ICML 2026 正式版](https://proceedings.mlr.press/v306/qi26b.html)（[PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/qi26b/qi26b.pdf)）。阅读正文 §1–6 与附录 B–D，核对了图 2、7–10 和表 2、3、5–7。本次想弄清两件事：长生成时粗检索为什么容易失效？原精度 KV 放在主存后，每步还要付多少检索和传输成本？

## 长生成会让按 prompt 聚类的质心过时（§1–3）

长上下文下，全注意力每步 decode 都要读取所有历史 KV，开销随上下文线性增长。按 query 只取相关 KV 可以降低带宽消耗。不过，若质心只用 prefill 的 key 拟合，后续生成的 key 越积越多，原先的分桶就不一定适合现在的数据。图 1 把 prefill 拟合的质心和用全程 key 重拟合的质心画在一起：两者有明显偏移。对短提示、长输出的推理任务，这比长提示、短输出更值得担心。

ParisKV 保留完整历史，每步找一批候选，只在候选中估计真正的 top-k。旧 KV 仍能取回；但候选漏掉的 token 不会在重排时补回来，即使最终 attention 用的是原精度数据。

## Prefill：用固定方向分桶，顺手建好重排摘要（§4.1）

对 query 和 key 做 L2 归一化，再用相同的随机正交变换（实现采用 SRHT）旋转。然后把旋转后的向量分为多个低维子空间，分别记录方向和半径。每个子空间的质心取坐标绝对值全相等的所有符号组合；最近质心的编号可从方向的符号得到。这样不用在 prefill 跑 K-means，也不用在 decode 重训质心。旋转不改变归一化向量之间的内积；近似各向同性是选这种码本的依据，不等于给任意 query 保证固定召回率。

每个 key 在 GPU 上留两份用于检索的信息：粗筛用的质心 ID，以及重排用的方向码和校准权重。方向码每坐标 4 bit（1 bit 符号、3 bit 幅值）；权重把原 key 的模、子空间半径和量化方向与真实方向的对齐因子合在一起。重排时再乘回 query 的模，估计的是原始内积，不是余弦相似度。主存中的原始 K、V 不按 4 bit 压缩。

## Decode：先用方向投票，再取回选中的 KV（§4.2–4.3）

最早的 sink token、最近的 local 窗口与新生成 token 的更新缓冲留在 GPU。缓冲攒满后向前滑动窗口，给移出窗口的 key 建索引，异步把对应原始 KV 搬去主存。这样既不会每步重建索引，也不会把刚生成的数据立即送去往返 PCIe。

检索时，query 在各子空间给方向相近的桶投票，累加后常只保留历史区约 5%–10% 的候选。融合 CUDA kernel 解包候选的低位码、计算并校准内积，再从中选 top-k。GPU 通过 UVA 直接读主存中这些 token 的原始 KV，避免每步由 CPU 发起细碎的拷贝。实验里检索预算为 100 个历史 token，不包括常驻的 sink/local 区；论文也按任务设定了短序列走全注意力的阈值（表 1）。

## 实验：分开看准确率、decode 和 prefill（§5、附录 D）

以下均是作者数据。准确率实验的模型包括 Qwen3-4B-Thinking-2507、Qwen3-8B 和 DeepSeek-R1-Llama-8B。AIME25 的输出上限是 38.9K token；Qwen3-8B 上，ParisKV 的 pass@8 是 73.33%，高于 PQCache 的 16.67% 和 MagicPIG 的 10.00%，低于全注意力的 83.33%。表 2 的 9 个长生成对照中有 7 个达到或超过全注意力；剩下两个都是 AIME25 对照。长输入结果也有分化：表 3 中，同一 Qwen3-8B 的 LongBench V2 总分为 33.07 对 33.59，长输入难题档为 19.67 对 28.57。检索结果接近全注意力的平均值，并不意味着每种任务都能无损切换。

| 作者测速条件 | 结果 |
| --- | --- |
| Llama 3.1 8B，1,024K token 上下文，batch=1，decode；§5.2、图 8 | ParisKV 49 ms/步，MagicPIG 830 ms/步，PQCache 2,179 ms/步，分别约为 16.9×、44.4× 的速度比。这里全注意力已 OOM，不能拿这些倍数与全注意力相比。 |
| Qwen3-8B，128K token 上下文；图 11、附录 D.2 | batch=1 时 ParisKV 和 FlashAttention-2 全注意力分别为 24.32、24.87 ms/步。ParisKV 到 batch=8 时是 58.92 ms/步；除以 batch 得 7.37 ms/token，是吞吐摊算值，不是单个请求两 token 间的等待时间。 |
| Llama 3.1 8B 和 Qwen3-8B，64K、128K、256K token，上下文固定、改变 batch；§5.2、图 7 | 在全注意力还能运行的范围，峰值 decode 吞吐优势为 2.1–2.8×；Qwen3-8B 的最大值是 2.8×。 |
| 附录表 7，128K token 输入，batch=1 | ParisKV 首 token 用时 43.00 s，表中全注意力为 33.30 s；该对照的 decode 为 24.42 对 23.13 ms/token，表 7 未单独标注模型。重排快不等于首 token 来得更早。 |

效率实验使用由 LongBench V2 派生的长输入、短生成负载；附录 D.1.2 未列明 GPU 具体型号和测速的固定输出 token 数。图 10 的消融把固定质心、归一化和旋转作为一组换掉：粗阶段平均 Recall@100 从 6.0% 到 16.1%，接精确重排时从 36.5% 到 64.3%。这组对照测的是粗筛候选的质量，不能用来衡量 4-bit 重排的影响。

## 读表时留意哪些条件

- 表 2 的 MATH500 列头和正文写 pass@1，图表题注却把它和 AIME25 放进 pass@8；读 MATH500 时以列头和正文为准。同一个百万 token 对照，摘要写 PQCache 约 44×、引言写 45×，上表采用 §5.2 的时延值计算。
- PQCache 按推荐的 20% 预算运行，MagicPIG 的有效预算随长度变化，ParisKV 固定取 100 个历史 token。这是整套方法的对比，不能把速度倍数都归给 CUDA kernel 或 4-bit 编码。正文也没有用百万 token 的准确率结果验证该长度下的 top-100 质量。
- [FreeKV](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5137e40ee586c4f1fc3615cf8cee4f87-Abstract-Conference.html) 把相邻 decode 步的检索与搬运排进流水线；ParisKV 则每步在 GPU 扫紧凑索引，减少要重排的历史 key 和要跨总线取回的 KV。对工作站或 UMA 设备，值得单独量的是粗筛 GPU 时间、实际传输量和首 token 时间。上述数据均来自论文，未做本机实测。
