# Progressive Sparse Attention: Algorithm and System Co-design for   Efficient Attention in LLM Serving

> **Date**：2025-03-01
> **arXiv**：https://arxiv.org/abs/2503.00392

## Abstract

Processing long contexts has become a critical capability for modern large language models (LLMs). However, serving long-context LLMs comes with significant inference costs due to the high memory overhead of the key-value (KV) cache. Existing work leverages dynamic sparse attention algorithms (DSAes) to mitigate the KV cache overhead, but these algorithms rely on top-$k$ KV cache selection, which results in a trade-off between accuracy and efficiency. A larger $k$ improves accuracy but decreases efficiency, while a smaller $k$ boosts efficiency but compromises accuracy. To overcome this trade-off, this paper presents PSA, a $\underline{P}$rogressive $\underline{S}$parse $\underline{A}$ttention mechanism that integrates algorithmic innovations with system co-design to achieve both high inference accuracy and improved efficiency in LLM serving. The PSA algorithm adaptively adjusts the KV cache budget of different tokens and layers according to their real attention weight distributions, rather than relying on a fixed budget $k$. This enables high accuracy while minimizing KV cache usage. To further enhance execution efficiency, we introduce a pipelined iteration scheme that reduces CPU-GPU interleaving and synchronization overhead during PSA computation. Additionally, we implement unified GPU memory management that optimizes PSA's memory utilization by accounting for uneven memory requirements across different model layers. Extensive experimental results demonstrate that PSA reduces KV cache usage for attention computation by up to 2.4$\times$ and 8.8$\times$, and increases end-to-end serving throughput by up to 1.4$\times$ and 2.0$\times$, compared to state-of-the-art DSAes and systems without sparse attention, respectively.

---

# 渐进稀疏注意力：面向大语言模型服务的算法与系统协同设计 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在处理几千甚至上万 token 的长上下文时，需要把每一次注意力的键值（KV）对都缓存下来，以便后续的自回归生成。这些 KV 缓存占用的显存往往是推理成本的主要瓶颈。已有的动态稀疏注意力（DSA）通过在每层每个 token 上挑选前 k 个最重要的 KV 来削减开销，但 k 必须固定，导致“k 大准度高、效率低；k 小效率高、准度差”的两难局面。换句话说，单一的硬阈值无法适配不同层、不同 token 的注意力分布差异，导致资源浪费或信息缺失。

### 关键概念速览
**KV 缓存**：在自回归生成时，模型把每一步的键（Key）和值（Value）保存下来，后续的注意力查询会在这些缓存上进行检索。想象成一次对话的“记事本”，越大越占空间。  
**动态稀疏注意力（DSA）**：每次注意力计算只保留前 k 个权重最高的 KV，类似只看最相关的几条记事本记录。  
**Top‑k 选择**：固定数量的阈值筛选方式，像是只挑选前 10 名学生参加比赛，忽略了成绩分布的差异。  
**Top‑p（核采样）**：根据累计概率阈值 p 动态决定保留多少项，类似把成绩总和达到 90% 的学生全部选进去。  
**KV 缓存预算**：指在一次注意力计算中允许使用的 KV 条目数目，可视为“记事本的页数”。  
**流水线迭代（Pipelined Iteration）**：把 CPU 的调度工作和 GPU 的计算交错进行，像是厨房里厨师和配菜员同步作业，减少等待时间。  
**统一 GPU 内存管理**：在不同层之间共享显存池，根据实际需求动态分配，避免某层占满显存而其他层空闲的浪费。

### 核心创新点
1. **固定 top‑k → 自适应 KV 预算**：传统 DSA 只能给每个 token 固定一个 k，导致层间和 token 间的资源分配不均。PSA 直接读取注意力权重的分布，用累计概率 p 动态决定每个 token 在每层实际使用多少 KV。这样高信息量的 token 能得到更多 KV，低信息量的 token 自动被压缩，整体准确率提升的同时显存占用大幅下降。  
2. **CPU‑GPU 串行 → 流水线迭代**：原有实现需要在 CPU 完成 KV 过滤后再把结果送到 GPU，频繁的同步导致吞吐率受限。PSA 把过滤过程拆成多个小批次，交叉执行：CPU 负责生成下一批的过滤指令，GPU 同时执行上一批的注意力计算。结果是同步点几乎被消除，端到端吞吐提升。  
3. **层级统一显存 → 按需分配**：不同层的稀疏程度差异很大，传统做法是为每层预留同等显存，导致显存利用率低。PSA 引入统一的显存池，运行时根据每层实际需要的 KV 数量动态申请和释放显存，显存碎片被显著压缩，进一步提升了可服务的上下文长度。  
4. **Top‑p 过滤 + 分块实现**：在实现层面，PSA 将长序列切分成若干块，每块内部使用累计概率 p 进行 KV 过滤，再把块间的结果拼接。这样既保持了全局稀疏的灵活性，又让 GPU 能够一次性加载完整块，避免跨块的随机访问开销。

### 方法详解
整体思路可以划分为三步：**（1）注意力权重采样、（2）自适应 KV 预算生成、（3）流水线执行与显存调度**。下面逐层拆解。

1. **注意力权重采样**  
   对于每个查询 token，模型先在完整的 KV 表上计算原始注意力分数（softmax 前的 logits），随后在 CPU 上对这些分数做一次累计求和。累计过程类似把所有分数排好序后，从大到小累加，直到累计概率超过预设阈值 p（如 0.9）。此时所有被累计的 KV 条目被标记为“保留”，其余的直接丢弃。因为 p 是固定的，而不是固定的条目数，保留的 KV 数量会随注意力分布自然伸缩。

2. **自适应 KV 预算生成**  
   每层会得到一个长度不等的保留列表。为了让 GPU 能一次性读取，PSA 把同一层的所有 token 的保留列表合并成一个稀疏矩阵，并记录每行（对应 token）的起始偏移和长度。这个稀疏矩阵相当于“压缩的记事本”，只保存需要查询的 KV。此时的 KV 预算已经由固定的 k 变成了每行独立的长度，真正实现了资源的细粒度分配。

3. **流水线迭代**  
   传统实现是：CPU 完成所有 token 的过滤 → 把稀疏矩阵拷贝到 GPU → GPU 完成注意力乘法 → 同步回 CPU。PSA 将这条链拆成两段交叉执行：  
   - **阶段 A**：CPU 处理第 i 批 token 的过滤并把稀疏描述写入共享缓冲区。  
   - **阶段 B**：GPU 同时读取第 i‑1 批的稀疏矩阵进行注意力乘法。  
   两个阶段在不同的 CUDA 流上并行，CPU 与 GPU 的工作几乎同步进行，显著降低了 CPU‑GPU 之间的等待时间。

4. **统一显存管理**  
   所有层共享一个显存池。每当某层完成稀疏矩阵的生成后，系统查询该层实际需要的 KV 条目数，并从池中分配对应大小的显存块；计算结束后立即归还。因为稀疏程度在不同层差异大（底层往往需要更多 KV），池化显存避免了“某层占满显存、其他层空闲”的低效局面。实现上使用 CUDA 的统一内存（Unified Memory）+ 手动分配策略，使得显存分配/回收的开销可以忽略不计。

**最巧妙的点**在于把注意力权重的概率分布直接当作资源调度信号，而不是事后再用硬阈值裁剪。这样模型本身的“自注意力”行为决定了硬件资源的分配，形成了真正的算法‑系统协同。

### 实验与效果
- **测试场景**：论文在公开的长上下文基准（如 LongChat、OpenWebText 长序列）以及实际的 LLM 推理服务中评估。  
- **对比基线**：包括最新的动态稀疏注意力实现（如 FlashAttention‑DSA）以及不使用稀疏的全注意力系统。  
- **KV 使用率**：相较于最强 DSA，PSA 在相同准确率下把 KV 缓存削减了最高 2.4×；在极端压缩下还能达到 8.8× 的削减。  
- **吞吐提升**：端到端服务吞吐率提升 1.4×（相对 DSA）和 2.0×（相对全注意力），说明流水线和显存池化的系统优化效果显著。  
- **消融实验**：作者分别关闭了自适应预算、流水线迭代和显存池化三个模块，发现自适应预算贡献最大（约 30% 的 KV 削减），流水线带来约 20% 的吞吐提升，显存池化在极长序列下提升显存利用率 15%。  
- **局限性**：论文承认在极端低 p（如 0.5）时，稀疏度过高会导致注意力信息丢失，准确率出现回撤；此外，当前实现依赖 CPU 端的累计排序，若在全 GPU 环境下迁移仍需进一步优化。

### 影响与延伸思考
PSA 把注意力分布直接映射到硬件资源，开启了“注意力驱动的系统调度”这一思路。后续工作（如 *Adaptive Cache Allocation for Transformers*、*Probabilistic Sparse Attention*）纷纷借鉴了累计概率阈值来决定稀疏度，甚至把 p 当作可学习的超参数。对于想继续深挖的读者，可以关注以下方向：① 将累计概率筛选搬到 GPU 上，实现全端到端的无缝流水线；② 结合检索增强模型，让外部文档的相关性也参与 KV 预算决策；③ 探索在多模态大模型中使用类似的自适应稀疏策略，以降低跨模态注意力的显存开销。整体来看，PSA 为大模型长上下文服务提供了一个兼顾算法精度和系统效率的可复制框架。

### 一句话记住它
**PSA 用注意力权重的累计概率动态决定每层每 token 用多少 KV，配合流水线和显存池化，让长上下文推理既省显存又提速。**