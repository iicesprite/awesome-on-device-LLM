
# 1. 引言

大型语言模型（LLM）（Achiam et al., 2023；Touvron et al., 2023；Jiang et al., 2023；Yang et al., 2025）的上下文窗口正迅速扩展到数百万 token，这对推理效率提出了前所未有的挑战。KV 缓存保存了自回归解码所需的全部历史 token 的键和值。随着上下文增长，KV 缓存很快就会成为内存占用和延迟的主要来源（Zhang et al., 2023；Li et al., 2024；Chen et al., 2024）。关键在于，长上下文解码受内存带宽限制：每一步都需要读取大量 KV 向量，其带宽开销随上下文长度线性增长。这推动了稀疏注意力／选择性注意力的发展：利用实际观察到的注意力稀疏性，仅对相关历史 token 中的 Top-k 子集计算注意力。

**图 1：检索漂移实验结果。**（a）不同方法在 AIME 上的召回率比较。（b）解码阶段产生的 key 所引起的中心漂移，以仅使用 prefill 阶段 key 得到的中心（蓝色的原始中心），与使用 prefill 和解码阶段全部 key 聚类得到的参考中心（红色的正确中心）之间的不匹配来衡量。

在稀疏化设计中，KV 缓存丢弃／淘汰方法会永久移除 token；当早期 token 在后续重新变得重要时，这类方法可能表现脆弱。相比之下，KV 缓存检索保留完整的 KV 历史，并在每一步动态检索相关 key，因此更适合开放式的长上下文推理。

**挑战。** 尽管 KV 缓存检索具有准确性优势，但在对延迟敏感的长生成场景中，它仍面临以下效率挑战。

- **C1：速度与质量的权衡。** 轻量级近似检索，例如粗粒度聚类和低比特量化，会牺牲召回率以换取速度；若要恢复准确性，就需要增加检索预算，从而削弱稀疏化带来的收益。
- **C2：解码漂移。** 使用 prefill 阶段的 key 聚类得到的中心，会随着生成持续进行而过时。图 1(a) 展示了一个例子：先前方法（Zhang et al., 2025b；Chen et al., 2024）在长时间解码过程中出现严重的召回率下降，而 ParisKV 保持了稳定的召回率。图 1(b) 可视化了其根本原因：随着解码阶段的 key 不断积累，仅根据 prefill key 得到的中心（蓝色）与真实 key 分布（红色）之间的不匹配越来越严重。
- **C3：受 CPU 限制的检索与数据搬运。** 当完整 KV 缓存被卸载到 CPU 内存后，检索通常涉及 CPU 端搜索以及 CPU 到 GPU 的数据传输。GPU 只能访问中心或低比特编码等表示，这些表示会引入近似误差；与此同时，CPU 端的调度与协调成为端到端延迟的主要来源。

**我们的方法。** 我们提出 ParisKV，这是一种面向硬件特性的 KV 缓存检索算法—系统协同设计。即使存在漂移，它仍能在长输入和／或长生成场景中实现快速解码与稳定召回，并扩展到百万 token 规模。

不同于从 prefill key 中学习中心的先前方法，ParisKV 首先将 query 和 key 归一化到单位超球面上（见图 3），再对它们应用相同的随机正交旋转，将其变换到一个稳定、与数据无关的空间。在这个空间中，我们可以通过解析方式定义一组均匀分布的中心。这些中心均匀分布在单位超球面上，因此任何新生成的 key 都会接近至少一个中心。由此，ParisKV 避免了中心过时——即中心在解码过程中与数据分布不再匹配——并在长生成过程中保持稳定的检索质量。

ParisKV 采用完全在 GPU 上实现的两阶段流水线。粗筛阶段通过多子空间碰撞计数大幅削减候选集合；精排阶段则根据原始 key 的紧凑表示，使用经过校准的内积估计对候选重新排序，无需访问 CPU 上的全精度 key 即可实现高召回率。

此外，ParisKV 将完整 KV 缓存卸载到 CPU 内存，并利用统一虚拟寻址（Unified Virtual Addressing，UVA），让 GPU 内核按需直接读取最终选中的 Top-k KV 对，从而绕过显式内存拷贝（`memcpy`）以及 CPU 端调度的开销。

**贡献。**

1. 我们提出了抗漂移的解析中心，通过与数据无关的变换，即使在百万 token 规模、存在显著分布变化时，也能保持稳定的检索召回率。
2. 我们设计了 GPU 原生的由粗到细检索流水线，将基于碰撞的剪枝与校准重排结合起来，直接在压缩编码上实现高保真的 Top-k 选择，无需 CPU 端搜索。
3. 我们实现了基于 UVA 的可扩展卸载系统，只需极少的 CPU 干预即可按需获取 KV。
4. 大量实验表明，在百万 token 规模下，ParisKV 相较 MagicPIG 和 PQCache 分别实现了最高 **17 倍和 45 倍**的加速，同时在长输入与长篇推理基准上保持接近无损的质量。

**利益冲突披露。** 作者声明不存在财务利益冲突。

# 4. ParisKV 框架

ParisKV 是一种算法—系统协同设计，通过抗漂移检索实现快速的长上下文解码。下面先概述 ParisKV，再讨论具体细节。

在 **prefill 阶段**（见图 2），除了计算 KV 缓存之外，ParisKV 还会一次性构建 key 的摘要，以支持解码阶段快速的设备端检索。具体而言，我们：

1. 应用共享的归一化—旋转变换，使表示在解码漂移下保持稳定；
2. 为每个 key 构建紧凑的元数据，在全精度 KV 缓存卸载到 CPU 后，仍将这些元数据保留在 GPU 上。

这些元数据包含两部分。第一部分是小型码本表示，即用少量中心编号概括 key（图 2 中的 A.2），随后通过基于碰撞的投票快速生成候选（图 2 中的 B.2.1）。第二部分是为每个 key 的各个子空间保存紧凑的量化编码（A.3.1），以及用于校准的轻量级缩放因子（A.3.2）。这些量化编码和缩放因子用于高效估计 query–key 内积，并对候选进行重排（B.2.2），无需在 GPU 上访问全精度 key。

总体而言，prefill 阶段会生成保留在 GPU 上的小型检索摘要，而全精度 KV 缓存则异步卸载到 CPU 内存，以支持更长的上下文。

在 **decode 阶段**（见图 2 和图 4），ParisKV 使用两步检索流水线。

1. **粗粒度候选生成（B.2.1）。** 仅使用保留在 GPU 上的轻量级码本摘要来缩小搜索范围。对于一个 query，我们在每个子空间选择与其最相似的少数几个中心；位于这些被选中桶中的 key 各获得一票。随后跨子空间累加票数，将高票 key 保留为候选。
2. **候选重排。** 估计前一步候选的 query–key 内积，并选出最终用于注意力计算的 Top-k key。此时，GPU 保存着 key 的紧凑 4-bit 量化编码，但无法高效访问位于 CPU 内存中的全精度原始向量。因此，ParisKV 利用量化编码按子空间快速估计内积，并通过轻量级过程修正量化误差（见图 4 底部），从而实现高保真的估计以及高召回率的 Top-k 选择。由于 query–key 内积估计涉及许多细粒度操作，我们实现了融合重排内核，以简化计算流程。

**图 2：ParisKV 流水线。** 离线阶段，我们构建解析中心码本与量化配置。在 prefill 阶段，生成 KV 缓存并构建保留在 GPU 上的 key 摘要：用于第一阶段投票筛选的中心编号，以及用于第二阶段重排的低比特编码和轻量级权重；与此同时，将全精度 KV 异步卸载到 CPU 内存。在解码阶段，增量更新新生成 key 的摘要；GPU 仅使用这些摘要进行由粗到细的检索，即“投票 → 重排”，然后通过 UVA 从 CPU 读取选中的 Top-k KV 对以计算注意力。

## 4.1. Prefill 阶段

### 4.1.1. 预处理

ParisKV 应用共享的归一化—旋转变换，将向量拆分为旋转后的子空间，并使用极坐标形式，即“半径—方向”形式，概括每个子空间（图 2 中的 A.1）。

给定 prefill 阶段的 key $\{\mathbf{k}_i\}_{i=1}^{n}$，ParisKV 在 GPU 上为每个 key 构建紧凑元数据；query 也以同样方式处理。具体步骤如下。

**（1）归一化与旋转。**

归一化将所有 key 和 query 映射到同一个单位超球面空间，使相似性比较只依赖方向，而不依赖大小。共享旋转在保持内积不变的同时，将信息更均匀地分散到各个维度，使后续分桶与打分在漂移下更加稳定。

首先，对 key 和 query 进行 $\ell_2$ 归一化：

$$
\hat{\mathbf{k}}_i
=
\frac{\mathbf{k}_i}{\|\mathbf{k}_i\|_2},
\qquad
\hat{\mathbf{q}}
=
\frac{\mathbf{q}}{\|\mathbf{q}\|_2}.
$$然后，应用共享的正交旋转矩阵 $\mathbf{R}$，通过 SRHT 实现：
$$\tilde{\mathbf{k}}_i=\mathbf{R}\hat{\mathbf{k}}_i,
\qquad
\tilde{\mathbf{q}}=\mathbf{R}\hat{\mathbf{q}}.$$

由于 $\mathbf{R}$ 是正交矩阵，因此保持内积不变：

$$\langle\hat{\mathbf{k}}_i,\hat{\mathbf{q}}\rangle
=
\langle\tilde{\mathbf{k}}_i,\tilde{\mathbf{q}}\rangle.$$

**（2）子空间划分。**

子空间划分支持高效的并行处理，并为基于碰撞的候选筛选提供多个独立的投票。

我们将 $\tilde{\mathbf{k}}_i$ 和 $\tilde{\mathbf{q}}$ 划分为 $B$ 个连续子空间，每个子空间的维度为 $m=D/B$：

$$\tilde{\mathbf{k}}_i
=
[\tilde{\mathbf{k}}_{i,1};\ldots;\tilde{\mathbf{k}}_{i,B}],$$

$$\tilde{\mathbf{q}}
=
[\tilde{\mathbf{q}}_1;\ldots;\tilde{\mathbf{q}}_B],$$

其中，$\tilde{\mathbf{k}}_{i,b},\tilde{\mathbf{q}}_b\in\mathbb{R}^{m}$。

**（3）极坐标分解。**

极坐标形式进一步将方向与半径分离：方向用于投票，半径用于经过校准的内积估计。

对于每个 key $i$ 和子空间 $b$，定义：

$$r_{i,b}=\|\tilde{\mathbf{k}}_{i,b}\|_2,
\qquad
\mathbf{u}_{i,b}
=
\frac{\tilde{\mathbf{k}}_{i,b}}{r_{i,b}}.$$

因此：

$$\tilde{\mathbf{k}}_{i,b}=r_{i,b}\mathbf{u}_{i,b}.$$

对 $\tilde{\mathbf{q}}$ 也采用类似处理。由此，旋转后的内积可以写成各子空间贡献之和：

$$\begin{aligned}
\langle\tilde{\mathbf{k}}_i,\tilde{\mathbf{q}}\rangle
&=
\sum_{b=1}^{B}
\langle\tilde{\mathbf{k}}_{i,b},\tilde{\mathbf{q}}_b\rangle\\
&=
\sum_{b=1}^{B}
r_{i,b}\langle\mathbf{u}_{i,b},\tilde{\mathbf{q}}_b\rangle.
\end{aligned}
\tag{4}$$

### 4.1.2. 将方向分配给中心

我们将每个 $\mathbf{u}_{i,b}$ 映射到最近的方向中心 $\mathbf{c}_j$（图 2 中的 A.2），并保存对应的中心编号，用于解码阶段基于碰撞的候选生成。

**与数据无关的均匀方向中心。**

经过 $\ell_2$ 归一化与 SRHT 旋转后，子空间方向变得近似各向同性，因此可以有效地使用一组共享、与数据无关的方向中心，而不必从数据中学习中心。

因此，我们在每个 $m$ 维子空间中预先定义如下符号模式中心集合：

$$\Omega
\triangleq
\left\{\pm\frac{1}{\sqrt{m}}\right\}^{m},
\qquad
|\Omega|=2^{m}.
\tag{5}$$

其中，每个 $\boldsymbol{\omega}\in\Omega$ 都是各坐标绝对值相等的单位方向。对于子空间单位方向 $\mathbf{u}_{i,b}$，我们将其分配给 $\Omega$ 中最近的中心，并保存所得编号：

$$\mathrm{centroid\_id}_{i,b}
\triangleq
\arg\max_{\boldsymbol{\omega}\in\Omega}
\langle\mathbf{u}_{i,b},\boldsymbol{\omega}\rangle.
\tag{6}$$

**示例。** 假设 $m=3$，则码本包含 $2^3=8$ 个方向中心：

$$\Omega=
\left\{
\left(
\pm\frac{1}{\sqrt{3}},
\pm\frac{1}{\sqrt{3}},
\pm\frac{1}{\sqrt{3}}
\right)
\right\},
\qquad
|\Omega|=8.$$

这 8 个中心对应立方体的 8 个顶点。缩放到单位范数后，它们位于单位球面上，均匀覆盖 8 个卦限，即三维空间中的 8 个粗粒度方向。

给定旋转后的 key $\tilde{\mathbf{k}}$，我们只需将其分配到 $\Omega$ 中最近的中心。例如，如果 $\tilde{\mathbf{k}}$ 大致指向 $(+,+,-)$ 方向，就将其分配给：

$$\left(
\frac{1}{\sqrt{3}},
\frac{1}{\sqrt{3}},
-\frac{1}{\sqrt{3}}
\right).$$

具体示意见图 3。

**图 3：单位球面上基于旋转的码本分配示意图。**

### 4.1.3. A.3 低比特重排元数据

为了在不访问 CPU 内存中全精度 key 的情况下对候选进行重排，我们在 GPU 上为每个子空间保存轻量级摘要：

- **A.3.1：** 4-bit 方向编码 $\mathrm{code}_{i,b}$，反量化后得到重建方向 $\mathbf{v}_{i,b}$；
- **A.3.2：** 标量权重 $w_{i,b}$，用于预先计算候选重排中所有仅依赖 key 的因子。

具体而言，子空间半径为：

$$r_{i,b}=\|\tilde{\mathbf{k}}_{i,b}\|_2,$$

每个子空间的修正项为：

$$\alpha_{i,b}
\triangleq
\langle\mathbf{v}_{i,b},\mathbf{u}_{i,b}\rangle.
\tag{7}$$

其中，$\mathbf{u}_{k,b}$ 是 key $k$ 在子空间 $b$ 中的真实单位方向，在重排时未知；$\mathbf{v}_{k,b}$ 是其量化方向。

直观上，$\alpha_{k,b}$ 衡量量化方向与真实方向之间的对齐程度。量化通常会降低这种对齐程度，因此，若直接使用 $\langle\mathbf{v}_{k,b},\tilde{\mathbf{q}}_b\rangle$，就会系统性地低估 query–key 内积。具体而言，我们的重排估计器采用以下近似：

$$\langle\mathbf{u}_{k,b},\tilde{\mathbf{q}}_b\rangle
\approx
\frac{
\langle\mathbf{v}_{k,b},\tilde{\mathbf{q}}_b\rangle
}{
\alpha_{k,b}
}.
\tag{8}$$

我们为每个 key 的每个子空间定义缩放因子：

$$w_{i,b}
\triangleq
\|\mathbf{k}_i\|_2
\frac{r_{i,b}}{\alpha_{i,b}}.
\tag{9}$$

值得注意的是，$w_{i,b}$ 只依赖 key 和量化元数据，不涉及 query。因此，可以在 prefill 阶段一次性预计算并缓存。这样，在线重排时只需计算量化方向与旋转后 query 之间的轻量级点积，再使用缓存的 $w_{i,b}$ 进行缩放，即可高效估计 $\langle\mathbf{k}_i,\mathbf{q}\rangle$。

在线重排由此变为对各子空间进行简单的加权累加：

$$\widehat{\langle\mathbf{k}_i,\mathbf{q}\rangle}
=
\|\mathbf{q}\|_2
\sum_{b=1}^{B}
w_{i,b}
\langle\mathbf{v}_{i,b},\tilde{\mathbf{q}}_b\rangle.
\tag{10}$$

用于重排的内积估计详见附录 B.2.2。

总体而言，prefill 阶段在 GPU 上保存以下元数据：

$$\left\{
\left(
\mathrm{centroid\_id}_{i,b},
\mathrm{code}_{i,b},
w_{i,b}
\right)
\right\}_{i\leq n,\ b\leq B},$$

而全精度 $(\mathbf{k}_i,\mathbf{v}_i)$ 则保留在 CPU 内存中，供 UVA 按需读取。

## 4.2. 解码阶段

### 4.2.1. 缓冲区更新

随着解码推进，我们将 KV 缓存维护为四个连续区域（见图 5）：

- **Sink：** 保留在 GPU 上的少量早期、高注意力 token；
- **Retrieval：** 已卸载并建立索引的历史 token；
- **Local：** 保留在 GPU 上、用于稠密注意力计算的最近 `local_size` 个 token；
- **Update Buffer：** 位于 GPU 上、暂存新生成 token 的更新缓冲区。

我们将每个新 token 追加到 Update Buffer 中（图 2 中的 B.1）。当缓冲区达到 $m$ 个 token 时，触发滑动窗口更新：

1. 将 Local 区域中最旧的 $m$ 个 token 移入 Retrieval 区域；在启用卸载时，通过 GPU → CPU 传输完成。
2. 将缓冲区中的 token 移入 Local，并清空缓冲区，使 Local 窗口始终保留最近的 `local_size` 个 token。
3. 在 GPU 上为移出的 key 进行编码并建立索引，包括中心编号、4-bit 编码和 $w_{i,b}$；同时异步卸载对应的全精度 KV 对。

这种流式更新支持持续生成，同时保持检索元数据及时更新，并保留对近期上下文（Local）与长距离历史（Retrieval）的快速访问。

**图 5：滑动窗口 KV 缓存更新。**

### 4.2.2. 检索

在每个解码步骤中，ParisKV 都会执行完全在 GPU 上运行的两步检索流水线（图 4，以及图 2 中的 B.2），在排序时不访问全精度 key，实现近似的 Top-k 注意力选择。

**图 4：ParisKV 检索算法，对应图 2 中的 B.2。**

**（1）粗粒度候选生成（B.2.1）。**

我们使用仅依赖方向的代理评分，将全部 key 快速裁剪为一个小型候选池。

具体而言，将旋转后的 key 和 query 分解为 $B$ 个子空间，并通过累加各子空间的碰撞奖励，为每个 key 打分。在每个子空间中，将 query 与该 key 所属的中心进行比较，即计算开销较低的点积 $q^\top c$，仅允许排名前 $\rho$ 比例的项贡献非零奖励。

跨子空间求和后得到粗粒度整数分数。该分数对解码漂移具有鲁棒性，并支持快速筛选出排名前 $\beta$ 比例的 key 作为候选，通常 $\beta=5\%\text{–}10\%$。

我们设置 $\rho\geq\beta$，使每个子空间都能提供足够多的非零项，并使候选池明显大于重排阶段最终选择的 Top-k 集合，例如 $k=100$。具体细节见附录 B.2.1。

**（2）候选重排（B.2.2）。**

我们直接根据紧凑的 4-bit key 量化表示估计原始注意力分数 $\langle\mathbf{k},\mathbf{q}\rangle$，据此对候选重新排序。

每个子空间方向的每个坐标都使用 **1-bit 符号位 + 3-bit 幅值**进行编码。在线重排只需计算量化方向与旋转后 query 之间的轻量级点积，再结合预计算的逐子空间缩放因子，即可准确估计原始内积，同时避免在重排过程中访问任何全精度 key。只有最终选中的 Top-k token 才会读取其全精度 KV。

附录 B.2.2 详细介绍了重排使用的内积估计器，包括公式（7）中的修正项与公式（19）中的近似。归一化并旋转后的子空间方向，其理论量化级别的推导由命题 4.1 指导，完整推导见附录 B.1.1。

**命题 4.1：旋转诱导的子空间 Beta 先验。**

设 $\hat{\mathbf{k}}\in\mathbb{S}^{D-1}$ 为任意单位向量，$\mathbf{R}\in\mathbb{R}^{D\times D}$ 为服从 Haar 随机分布的正交矩阵。令：

$$\tilde{\mathbf{k}}=\mathbf{R}\hat{\mathbf{k}},$$

并将其划分为 $B$ 个连续子空间，每个子空间维度为 $m$，满足 $D=Bm$：

$$\tilde{\mathbf{k}}
=
[\tilde{\mathbf{k}}_1;\ldots;\tilde{\mathbf{k}}_B].$$

定义：

$$r_b=\|\tilde{\mathbf{k}}_b\|_2,
\qquad
z_b=r_b^2\in[0,1],
\qquad
\mathbf{u}_b
=
\frac{\tilde{\mathbf{k}}_b}{r_b}
\in\mathbb{S}^{m-1}.$$

其中，$z_b$ 是极坐标分解后的子空间能量占比，即半径的平方，用于设计半径量化中心，见公式（13）；$\mathbf{u}_b$ 是子空间 $b$ 内对应的单位方向，其逐坐标分布见公式（12），用于指导归一化—旋转表示的量化级别设计。

那么，对于任意固定的 $b\in[B]$ 和 $j\in[m]$，有：

$$z_b
\sim
\mathrm{Beta}
\left(
\frac{m}{2},
\frac{D-m}{2}
\right),
\tag{11}$$

$$(\mathbf{u}_b)_j^2
\sim
\mathrm{Beta}
\left(
\frac{1}{2},
\frac{m-1}{2}
\right).
\tag{12}$$

**实现。** 两个阶段均使用自定义 CUDA 内核实现：

1. 碰撞累加内核，以及用于 Top-$\beta$ 选择的小范围整数桶式 Top-k 内核；
2. 用于 RSQ-IP 的融合重排内核。

相关细节与内核级优化见第 4.3 节及附录。其他设计细节，包括多级碰撞权重、离线量化器推导，以及 SRHT／Beta 先验的依据，见附录 B。

### 4.2.3. 通过 UVA 按需读取

最后，GPU 通过 UVA 从 CPU 内存中仅获取选中的 Top-k KV 对，并计算注意力（图 2 中的 B.3）。

因此，CPU 内存作为提供存储容量的后备存储，而检索决策路径仍然保持 GPU 原生。

## 4.3. 实现优化与复杂度

ParisKV 采用由粗到细的检索流水线，其延迟主要来自 GPU 上的选择、重排和 KV 读取。我们实现了四个自定义 CUDA 内核，以消除排序开销、减少内核启动次数，并尽量降低内存流量：

1. 针对整数碰撞分数的桶式 Top-k 内核；
2. 并行碰撞内核；
3. 融合重排内核，将收集、解包和打分合并，即 `gather + unpack + score`；
4. 基于 UVA 的内核。

**表 1：不同任务的超参数配置。**

| 数据集          | Local | Update | 全注意力阈值（Full-thres.） | 最大生成长度（Max Gen） |
| ------------ | ----- | ------ | ------------------- | --------------- |
| AIME25       | 256   | 512    | 2K                  | 38.9K           |
| MATH500      | 256   | 256    | 1K                  | 38.9K           |
| GPQA-Diamond | 128   | 512    | 2K                  | 32.8K           |
| LongBench_V2 | 256   | 512    | 2K                  | 1024+512        |
| RULER        | 256   | 512    | 2K                  | 128             |

**复杂度。** 设 $|S|$ 为 GPU 上常驻／局部区域的注意力长度，$n=|R|$ 为检索区域长度。对整个缓存进行稠密打分，每一步的复杂度为：

$$\mathcal{O}\big((|S|+n)D\big).$$

ParisKV 仅对 $|S|$ 保持稠密注意力，并用基于紧凑元数据的由粗到细流水线替代全缓存打分：碰撞处理规模随 $\rho n$ 增长，重排规模为：

$$C=\lceil\beta n\rceil,$$

而最终只需读取 $k\ll C$ 个 KV 向量用于注意力计算。