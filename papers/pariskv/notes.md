# ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs

来源：[ICML 2026 正式版](https://proceedings.mlr.press/v306/qi26b.html)（[PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/qi26b/qi26b.pdf)）。阅读正文 §1–6 与附录 B–D，核对了图 2、7–10 和表 2、3、5–7。本次想弄清两件事：长生成时粗检索为什么容易失效？原精度 KV 放在主存后，每步还要付多少检索和传输成本？

## Drifting——漂移是什么？

生成过程中，新产生的 key 的分布逐渐偏离最初输入阶段（prefill）的 key 分布

## 长生成会让按 prompt 聚类的质心过时（§1–3）

长上下文下，全注意力每步 decode 都要读取所有历史 KV，开销随上下文线性增长。按 query 只取相关 KV 可以降低带宽消耗。不过，若质心只用 prefill 的 key 拟合，后续生成的 key 越积越多，原先的分桶就不一定适合现在的数据。图 1 把 prefill 拟合的质心和用全程 key 重拟合的质心画在一起：两者有明显偏移。对短提示、长输出的推理任务，这比长提示、短输出更值得担心。

ParisKV 保留完整历史，每步找一批候选，只在候选中估计真正的 top-k。旧 KV 仍能取回；但候选漏掉的 token 不会在重排时补回来，即使最终 attention 用的是原精度数据。

## Prefill：用固定方向分桶，顺手建好重排摘要（§4.1）

对 query 和 key 做 L2 归一化，再用相同的随机正交变换（实现采用 SRHT）旋转。然后把旋转后的向量分为**多个低维子空间**，分别记录方向和半径。每个子空间的质心取坐标绝对值全相等的所有符号组合；最近质心的编号可从方向的符号得到。这样不用在 prefill 跑 K-means，也不用在 decode 重训质心。旋转不改变归一化向量之间的内积；近似各向同性是选这种码本的依据，不等于给任意 query 保证固定召回率。

每个 key 在 GPU 上留两份用于检索的信息：粗筛用的质心 ID，以及重排用的方向码和校准权重。方向码每坐标 4 bit（1 bit 符号、3 bit 幅值）；权重把原 key 的模、子空间半径和量化方向与真实方向的对齐因子合在一起。重排时再乘回 query 的模，估计的是原始内积，不是余弦相似度。主存中的原始 K、V 不按 4 bit 压缩。

![ParisKV 流程：prefill 建立索引并卸载 KV，decode 粗筛、重排后按需取回](resources/assets/Pasted%20image%2020261116201425.png)

图 2（ICML 2026 正式版，PDF 第 4 页）：prefill 把检索摘要留在 GPU；decode 先用摘要粗筛和重排，再通过 UVA 取回选中的原始 KV。
### B.2.2精读

**B.2.2 解决的是：全精度 key 已经放在 CPU 上，GPU 如何只靠 4-bit 摘要，估计候选的真实 $q^\top k$ 并重新排序。**

它与 B.2.1 的区别是：

- **B.2.1：** 看 key 属于哪些中心，通过整数加分粗筛。
- **B.2.2：** 读取候选 key 更细致的量化表示，估计内积，选出最终 Top-k。

下面按原文公式（16）—（24）展开。

#### 1. 最终要估计什么？

目标是原始 key 与 query 的内积：

$$s_i=\langle k_i,q\rangle.$$

**不是只估计方向相似度。**因为两个 key 即使方向相同，长度不同，内积也可能不同。

例如：

$$k_1=(1,0),\qquad k_2=(3,0),\qquad q=(1,0).$$

两者与 query 的余弦相似度都是 1，但内积分别是 1 和 3。因此，前面归一化时去掉的长度信息，最后必须补回来。

#### 2. 先把精确内积分解成几个部分

##### ① 取出原始向量长度

定义单位向量：

$$\hat k_i=\frac{k_i}{\|k_i\|_2},
\qquad
\hat q=\frac{q}{\|q\|_2}.$$

于是：

$$\langle k_i,q\rangle
=
\|k_i\|_2\|q\|_2
\langle\hat k_i,\hat q\rangle.$$

##### ② 旋转，但不改变归一化后的内积

使用同一个正交矩阵 $R$：

$$\tilde k_i=R\hat k_i,\qquad
\tilde q=R\hat q.$$

由于正交旋转保持内积：

$$\langle\hat k_i,\hat q\rangle
=
\langle\tilde k_i,\tilde q\rangle.$$

##### ③ 拆成子空间，再分离半径与方向

对每个子空间 $b$：

$$\tilde k_{i,b}=r_{i,b}u_{i,b},$$

其中：

- $r_{i,b}=\|\tilde k_{i,b}\|_2$：这一块的长度；
- $u_{i,b}$：这一块的单位方向。

因此，精确的原始内积可以写成：

$$\boxed{
\langle k_i,q\rangle
=
\|k_i\|_2\|q\|_2
\sum_{b=1}^{B}
r_{i,b}
\langle u_{i,b},\tilde q_b\rangle
}$$

这就是后面估计器的起点：**长度信息单独保存，主要压缩方向 $u_{i,b}$。**

注意，$\tilde q_b$ 是“整个 query 归一化、旋转后的一个分块”，**不是再次归一化到单位长度的子向量**。

#### 3. 用 4 bit 编码方向的每个坐标

原文 Step 1 将 $u_{i,b}$ 的每个坐标编码为：

$$\underbrace{1\text{ bit}}_{\text{正负号}}
+
\underbrace{3\text{ bit}}_{\text{幅值档位}}.$$

##### 符号直接保留

$$s_{i,b,j}=\operatorname{sign}\big((u_{i,b})_j\big).$$

##### 绝对值分成 8 档

利用 7 个阈值，将坐标绝对值映射到编号：

$$t_{i,b,j}\in\{0,\ldots,7\}.$$

每个编号对应一个重建幅值 $a[t]$。解码后：

$$(v_{i,b})_j=s_{i,b,j}\,a[t_{i,b,j}].$$

**这里的 $v_{i,b}$ 是“量化后重建的 key 方向”，不是 KV 中的 value。**

例如，某坐标只需保存：

```
负号 + 幅值档位 5
```

重排时查表恢复成：

```
−a[5]
```

整个方向向量由这些坐标重建出来，但不需要保存原来的全精度坐标。

##### 这 8 个幅值怎么选？

B.1.2 根据旋转后单位方向坐标的理论分布，离线运行 Lloyd–Max 标量量化，得到阈值与重建值，而不是简单等间隔量化；这些值在相同子空间维度配置下共享。其 Beta 分布推导假设 Haar 随机正交旋转，实际 SRHT 实现则在近似各向同性假设下复用这些先验。

#### 4. 为什么还需要对齐修正 $\alpha$？

如果直接用重建方向 $v_{i,b}$ 代替真实方向 $u_{i,b}$，就会得到：

$$\langle u_{i,b},\tilde q_b\rangle
\approx
\langle v_{i,b},\tilde q_b\rangle.$$

但量化会改变向量，因此原文 Step 2 定义：

$$\alpha_{i,b}
=
\langle v_{i,b},u_{i,b}\rangle,$$

并采用修正估计：

$$\boxed{
\langle u_{i,b},\tilde q_b\rangle
\approx
\frac{\langle v_{i,b},\tilde q_b\rangle}
{\alpha_{i,b}}
}
\tag{19}$$

$\alpha$ 衡量重建向量在真实方向上的投影；由于 $v$ 不一定是单位向量，**它不应直接被理解为两个单位方向的余弦相似度**。

##### 用向量分解理解这个除法

下面是对公式（19）的数学解释。

因为 $u$ 是单位向量，可以精确地写成：

$$v=\alpha u+e,
\qquad e\perp u.$$

也就是说，量化后的 $v$ 包含：

- 沿真实方向的部分 $\alpha u$；
- 偏离真实方向的误差 $e$。

于是：

$$\frac{\langle v,\tilde q_b\rangle}{\alpha}
=
\langle u,\tilde q_b\rangle
+
\frac{\langle e,\tilde q_b\rangle}{\alpha}.$$

**除以 $\alpha$，能校正沿真实方向的缩放失真，但不能消除所有方向误差。** 所以它是近似内积估计，而不是恢复出原始 key。

##### 一个数值示例

以下数值仅用于说明修正机制，不是论文实际的量化级别。

设：

$$u=(0.6,0.8),\quad
v=(0.5,0.75),\quad
\tilde q_b=(0.8,0.6).$$

则：

$$\alpha=v^\top u=0.9.$$

真实方向内积为：

$$u^\top\tilde q_b=0.96.$$

直接使用量化方向得到：

$$v^\top\tilde q_b=0.85.$$

修正后为：

$$\frac{0.85}{0.9}\approx0.9444.$$

在这个例子中，修正让估计更接近真实值，但仍不是完全相等。

#### 5. 把方向估计变回原始内积

将修正后的方向内积代回第 2 节的分解：

$$\widehat{\langle k_i,q\rangle}
=
\|q\|_2
\sum_{b=1}^{B}
\left(
\|k_i\|_2
\frac{r_{i,b}}{\alpha_{i,b}}
\right)
\langle v_{i,b},\tilde q_b\rangle.$$

看括号里的部分：

$$\|k_i\|_2\frac{r_{i,b}}{\alpha_{i,b}}.$$

**它完全不依赖当前 query。** 所以论文将它预先合并为一个权重：

$$\boxed{
w_{i,b}
=
\|k_i\|_2
\frac{r_{i,b}}{\alpha_{i,b}}
}
\tag{23}$$

最终在线计算只剩下：

$$\boxed{
\hat s_i
=
\|q\|_2
\sum_{b=1}^{B}
w_{i,b}
\langle v_{i,b},\tilde q_b\rangle
}
\tag{24}$$

这几个因子的作用分别是：

|因子|作用|
|---|---|
|$\|k_i\|_2$|恢复原始 key 的长度|
|$r_{i,b}$|恢复该子空间的长度／贡献大小|
|$1/\alpha_{i,b}$|校正量化方向的对齐失真|
|$\langle v_{i,b},\tilde q_b\rangle$|当前 query 与重建方向的匹配程度|
|$\|q\|_2$|恢复原始 query 的长度|

这里的 $w_{i,b}$ 是**重排缩放权重**，与 B.2.1 的整数奖励档位权重不是一回事。

#### 6. Prefill 和 decode 分别做什么？

##### 建立摘要时：把只依赖 key 的计算提前做完

在完整 key 还可访问时：

```
归一化、旋转 key
    ↓
拆成子空间，得到半径 r 和真实方向 u
    ↓
把 u 量化成 4-bit 编码
    ↓
得到重建方向 v，计算 α = vᵀu
    ↓
合并计算 w = ||k|| × r / α
    ↓
GPU 保存 4-bit 编码和 w
```

因此，重排时不需要重新取得真实方向 $u$，也不需要单独保存 $\alpha$ 和 $r$：它们已经合并进 $w$。
##### Decode 重排时：只处理粗筛留下的候选

```
取出候选 key 的 4-bit 编码和 w
    ↓
解码每个坐标的符号、幅值
    ↓
与当前 query 的对应子空间做点积
    ↓
乘以 w，跨子空间累加
    ↓
按估计分数选最终 Top-k
```

由公式还可以直接推导：**对于同一个非零 query，$\|q\|_2$ 对所有候选都相同，因此仅为了确定 Top-k 排名时，可以省略这个公共正因子。** 但不同 key 的长度不同，$\|k_i\|_2$ 不能同样省掉。

**概括起来，B.2.2 就是：用 4-bit 编码保存 key 的细粒度方向，再用预计算权重补回长度并校正量化失真，让 GPU 不读取全精度 key，也能估计原始内积来精排。**

## Decode：先用方向投票，再取回选中的 KV（§4.2–4.3）

最早的 sink token、最近的 local 窗口与新生成 token 的更新缓冲留在 GPU。缓冲攒满后向前滑动窗口，给移出窗口的 key 建索引，异步把对应原始 KV 搬去主存。这样既不会每步重建索引，也不会把刚生成的数据立即送去往返 PCIe。

检索时，query 在各子空间给方向相近的桶投票，累加后常只保留历史区约 5%–10% 的候选。融合 CUDA kernel 解包候选的低位码、计算并校准内积，再从中选 top-k。GPU 通过 UVA 直接读主存中这些 token 的原始 KV，避免每步由 CPU 发起细碎的拷贝。实验里检索预算为 100 个历史 token，不包括常驻的 sink/local 区；论文也按任务设定了短序列走全注意力的阈值（表 1）。

### UVA：如何减小开销

论文给出的解释是：**GPU 内核直接按需读取 CPU 内存中的 Top-k KV，不必让 CPU 为每次检索结果组织拷贝。** 下面结合 CUDA 的内存机制解释这个过程。[作者实现说明](https://github.com/amy-77/ParisKV#how-it-works)

#### 1. UVA 让 GPU 能“找到”CPU 内存里的数据

UVA（统一虚拟寻址）把 CPU 和 GPU 的内存放进统一的虚拟地址空间；配合**映射的锁页主机内存（mapped pinned memory）**，GPU 内核就能通过指针访问 CPU 内存。**不是开启 UVA 后，任意普通 CPU 内存都自动能被 GPU 直接读取。**[CUDA 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#unified-virtual-address-space)

于是，GPU 内核里的读取逻辑可以概念性地写成：

```
// 示意：在 GPU 内核中执行
id = topk_indices[j];
value = host_kv[id];   // 指向已映射的 CPU 内存
```

这条读取指令会触发跨 CPU–GPU 互连的数据传输，而不需要 CPU 另外发出一次 `cudaMemcpy`。[CUDA 映射内存说明](https://docs.nvidia.com/cuda/archive/12.3.2/cuda-c-programming-guide/index.html#mapped-memory)

#### 2. 为什么能减少 CPU 调度？

对比下面两种**示意流程**：

```
CPU 组织搬运：
GPU 得到 Top-k 索引
    → CPU 获取索引
    → CPU 收集对应 KV、提交拷贝
    → GPU 使用 KV

GPU 按需读取：
GPU 得到 Top-k 索引
    → GPU 内核根据索引直接读取 CPU 中的 KV
    → GPU 使用 KV
```

关键不只是地址统一，而是 **“选哪些 KV”和“读取这些 KV”都由 GPU 完成**。ParisKV 的检索结果本来就在 GPU 上，因此不必为了搬运再把这份结果交给 CPU 处理；作者的实现把检索和 UVA 读取都放在 GPU 路径上。[作者实现说明](https://github.com/amy-77/ParisKV#how-it-works)

由此省掉的是这条路径中的**索引回传、CPU 端收集与拷贝提交，以及相应的协调等待**，而不是整个程序不再需要 CPU。

#### 3. “没有显式拷贝”不等于“没有传输”

KV 数据仍要经过 PCIe 或相应的 CPU–GPU 互连到达 GPU；改变的是**传输由谁发起**：

- 显式拷贝：CPU 提交搬运命令。
- 映射内存读取：GPU 的访存指令触发传输。

远程读取仍受互连带宽和延迟限制，也不等同于读取本地显存；因此这种方式并不保证对所有访问模式都更快。[CUDA 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#mapped-memory)

**一句话：UVA 配合映射内存，让 GPU 从“算好索引后等 CPU 搬数据”，变成“算好索引后自己去取数据”。**

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

## 一些锐评

### 关于rebuttal

#### 准确率基线问题

**没有针对“DS-R1-Llama-8B 原模型为什么只有13.12分”给出具体解释。** 我重新核对了你下载的全部 rebuttal，作者回应的是“为什么检索能超过全注意力”，但没有解释这个基线为何这么低。

Reviewer **pA8H**直接问：

> For DS-R1-Llama-8B, what is the reason for the improved accuracy relative to the baseline?

作者的回答是：

- 检索可能具有**隐式去噪作用**，排除低重要性KV，使注意力分布更集中；
- 增益很小，**小于0.5%**，处于正常评测波动范围；
- 主要结论是接近无损，而非稳定超过全注意力。[作者回复 pA8H](https://openreview.net/forum?id=wxD4wTYQXt&noteId=1b8cJrzBHf)

但这解释不了表3的差距

|DS-R1-Llama-8B，LongBench v2|Overall|
|---|--:|
|Full Attention|13.12|
|ParisKV|28.43|
|差值|**+15.31个百分点**|

**15.31个百分点显然不是作者所说的“小于0.5%”。Reviewer qwkG**也质疑了大幅超过全注意力的现象，作者仍使用相同的“小增益＋去噪＋正常波动”解释，没有针对这组数字展开。[作者回复 qwkG](https://openreview.net/forum?id=wxD4wTYQXt&noteId=OBOGtZ6RVN)

而且，补充实验里多种方法都超过该Full基线，例如Quest为23.51、SOCKET为21.51。这进一步说明，问题不只是“ParisKV为何更好”，而是**这条Full基线是否在正确、相同的评测协议下运行**。[补充结果](https://github.com/amy-77/ParisKV/blob/main/rebuttal_exp.md)

#### 与FreeKV的对比

**审稿人确实质疑了“只比 MagicPIG、PQCache，基线不够强”；作者在 rebuttal 中补上了 FreeKV，但 FreeKV 的速度数据仍存在前面发现的矛盾。**

审稿人从三个角度要求补充强基线：

|审稿人|主要质疑|
|---|---|
|**5xn6**|MagicPIG、PQCache 在延迟上不够有竞争力；应比较优化过的 GPU 检索方法，并明确询问与 Quest、Twilight 相比的实际速度。|
|**qwkG**|只比较 MagicPIG、PQCache，缺少近期 SOTA KV 检索方法，无法全面判断性能。|
|**JGZR**|缺少 RetroInfer、ShadowKV、RetrievalAttention、SOCKET 等方法；要求在可比预算、相同硬件下比较准确率与吞吐量。|

作者承认原稿基线覆盖有限，并补充 Quest、Twilight、RetroInfer、ShadowKV、SOCKET、FreeKV 等方法，声称使用相同硬件、对齐有效检索预算；回复中列出的目标预算为 LongBench-v2／GPQA 约 **420 token**，RULER 约 **2048 token**。

补充结果明确优于 FreeKV

| 任务／模型                         | FreeKV |   ParisKV |
| ----------------------------- | -----: | --------: |
| LongBench-v2 Overall，Qwen3-4B |  19.68 | **24.60** |
| RULER Avg，Llama3.1-8B         |  71.87 | **83.49** |
| GPQA-Diamond，Qwen3-4B         |  58.16 | **72.22** |

作者还特意将 **FreeKV 58.16** 列为 GPQA 上最强的非 ParisKV 基线，报告 ParisKV 高出 **14.06 个百分点**。[作者回复 qwkG](https://openreview.net/forum?id=wxD4wTYQXt&noteId=OBOGtZ6RVN)

但是速度对比中，页面本身也保留着矛盾

作者对 JGZR 的回复同时提供了两张表：

| 128K、batch size 1     |    FreeKV |   ParisKV |
| --------------------- | --------: | --------: |
| 标为吞吐量，tokens/s        |     21.78 | **41.10** |
| 标为 decode 延迟，ms/token | **21.78** |     24.42 |

两张表确实在你下载的讨论页中同时存在。[作者原回复](https://openreview.net/forum?id=wxD4wTYQXt&noteId=1Iocr5gUP2)

**这会导致相反结论：**

- 按吞吐量表：ParisKV 更快。
- 按延迟表：FreeKV 更快。

按单请求持续解码的换算，我计算：

\[ 21.78\ \text{ms/token} \quad\Longleftrightarrow\quad 45.91\ \text{tokens/s}, \]

并不对应表中的 21.78 tokens/s。MagicPIG、PQCache 两列也出现了吞吐量表与延迟表重复同一数值的现象，**提示可能有录入或单位混用，但页面没有说明哪张表有误**。

另外，作者把 FreeKV 在 256K、384K 的 decode 结果标为 NA；**仅凭这些 NA，不能推断 ParisKV 在这些长度上比 FreeKV 快**。[作者回复 5xn6](https://openreview.net/forum?id=wxD4wTYQXt&noteId=NLQJyi595u)

这里的数字矛盾是我核对页面发现的，**不是审稿人后续提出的问题**。

## 4. 审稿人是否继续追问？

页面中的后续 acknowledgement 显示：

- 5xn6 认为问题已充分解决，并表示会提高评分；
- qwkG 选择了“Fully resolved”；
- JGZR 也选择“Fully resolved”，后续主要关注代码开放。[5xn6 后续](https://openreview.net/forum?id=wxD4wTYQXt&noteId=uaeYwJiXi7)、[qwkG 后续](https://openreview.net/forum?id=wxD4wTYQXt&noteId=1lnCijqtz6)、[JGZR 后续](https://openreview.net/forum?id=wxD4wTYQXt&noteId=SXQM0VtmLk)

**没有看到针对 FreeKV 速度表矛盾的进一步追问或作者勘误。**

### 我的判断

**这份 rebuttal 能支持“作者补做了 FreeKV 比较，且所报告配置下 ParisKV 的质量更好”；但目前不能可靠支持“ParisKV 的 decode 比 FreeKV 更快”。**

至于为什么主文仍重点展示 MagicPIG、PQCache，这页能确认的是：**原稿确实主要比较这两者，强基线是在审稿要求后补充的**；作者没有进一步解释，为什么最终主文没有把这些新增强基线放到同等显著的位置。