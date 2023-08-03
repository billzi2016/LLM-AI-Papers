# From Sparse to Soft Mixtures of Experts

> **Date**：2023-08-02
> **arXiv**：https://arxiv.org/abs/2308.00951

## Abstract

Sparse mixture of expert architectures (MoEs) scale model capacity without significant increases in training or inference costs. Despite their success, MoEs suffer from a number of issues: training instability, token dropping, inability to scale the number of experts, or ineffective finetuning. In this work, we propose Soft MoE, a fully-differentiable sparse Transformer that addresses these challenges, while maintaining the benefits of MoEs. Soft MoE performs an implicit soft assignment by passing different weighted combinations of all input tokens to each expert. As in other MoEs, experts in Soft MoE only process a subset of the (combined) tokens, enabling larger model capacity (and performance) at lower inference cost. In the context of visual recognition, Soft MoE greatly outperforms dense Transformers (ViTs) and popular MoEs (Tokens Choice and Experts Choice). Furthermore, Soft MoE scales well: Soft MoE Huge/14 with 128 experts in 16 MoE layers has over 40x more parameters than ViT Huge/14, with only 2% increased inference time, and substantially better quality.

---

# 从稀疏到软混合专家 论文详细解读

### 背景：这个问题为什么难？

在大模型里，提升容量往往意味着显著增加计算和显存开销。稀疏混合专家（Mixture‑of‑Experts，MoE）通过让每个输入只走少数专家，成功把参数规模放大到几百倍，却仍保持相近的训练/推理成本。可是，传统 MoE 有几大痛点：训练时容易出现梯度不稳、部分 token 被直接丢弃（token dropping），导致信息损失；专家数量难以继续扩展，因为路由机制会变得极不均衡；微调阶段往往效果不佳，因为路由在新任务上不一定保持原有的稀疏分配。这些瓶颈让研究者在想：能不能保留 MoE 的容量优势，同时把这些副作用全部消掉？

### 关键概念速览
- **稀疏混合专家（Sparse MoE）**：模型内部有很多专家网络，输入的每个 token 只被送到少数（通常是 1‑2）专家处理，类似把工作分配给“专门的工人”。  
- **路由（Routing）**：决定哪个 token 去哪个专家的模块，常用的有 Top‑K 选择或 gating 网络，像是“调度员”。  
- **软分配（Soft Assignment）**：不再硬性挑选少数专家，而是让每个 token 以不同权重同时进入多个专家，类似把任务拆成若干子任务分别交给不同的团队。  
- **隐式软分配（Implicit Soft Assignment）**：模型内部自动算出每个 token 的加权组合后再送给专家，外部看不见明确的路由表。  
- **专家层（MoE Layer）**：在 Transformer 中插入的特殊层，负责把 token 与专家进行交互。  
- **Token Dropping**：在稀疏 MoE 中，为了控制计算量，直接把一部分 token 丢掉不送入任何专家，等价于“让部分人不参与工作”。  
- **微调（Finetuning）**：在大模型预训练后，对特定下游任务进行轻量训练的过程。  

### 核心创新点
1. **从硬路由到软路由**  
   - 之前的稀疏 MoE 用 Top‑K 或 gating 直接挑出 1‑2 个专家，导致梯度只能在少数路径上传递，容易出现不稳。  
   - 这篇论文让每个 token 与所有专家形成加权组合，权重由可微的 gating 网络输出，所有专家都能收到信息的“影子”。  
   - 结果是梯度在整个网络中流动更均匀，训练过程更平滑，且不需要显式的 token 丢弃。

2. **保持稀疏计算的同时实现软分配**  
   - 虽然每个 token 的信息被广播到所有专家，但每个专家只对自己权重最高的那部分（组合后） token 进行实际计算，保持了原有的计算上限。  
   - 这样既保留了稀疏 MoE 的低成本特性，又让每个 token 的表示受到了多专家的综合影响。

3. **大规模专家扩展的可行性**  
   - 通过软分配，路由网络不再需要强制均衡每个专家的负载，专家数量可以轻松扩展到上百甚至上千。  
   - 实验中作者把专家数提升到 128，模型参数比同等 ViT 大 40 倍，推理时间仅增加约 2%。  

4. **对下游任务的友好微调**  
   - 软路由的可微特性让微调阶段不必重新学习硬路由规则，微调时只需要微调普通的 Transformer 参数即可。  
   - 这解决了传统 MoE 在微调时表现不佳的老问题。

### 方法详解
**整体思路**  
Soft MoE 把标准的 Transformer 结构改造成“软路由 + 稀疏专家”两步走的管线。首先，用一个轻量的 gating 网络为每个 token 生成一组长度等于专家数的权重向量；随后，这些权重与原始 token 向量做加权求和，得到每个专家的输入集合。每个专家只对自己权重最高的那部分 token 进行前向计算，计算完后再把结果加权回到原始 token 上，完成一次完整的 Transformer 层。

**关键模块拆解**  

1. **Gating 网络**  
   - 输入：每个 token 的隐藏向量。  
   - 输出：一个长度为 E（专家数）的实数向量，经过 softmax 归一化得到概率分布。可以把它想成“每个专家对该 token 的兴趣度”。  

2. **隐式软分配**  
   - 对每个专家 i，取所有 token 的向量乘以对应的 gating 权重 w_i（即该 token 对专家 i 的兴趣度），再把这些加权向量相加，得到专家 i 的“混合输入”。  
   - 这里的“混合”相当于把所有 token 的信息压缩成一个加权池，专家只看到自己感兴趣的那部分信息。  

3. **稀疏专家计算**  
   - 每个专家内部仍是一个小型的前馈网络或自注意力块。  
   - 为了控制计算量，专家只对混合输入中权重最高的前 K% token 进行实际运算，其他 token 的梯度通过 gating 权重间接传递。  

4. **结果融合**  
   - 专家输出后，再乘以对应的 gating 权重（这次是反向的），并对所有专家的贡献做加权求和，恢复到每个 token 的维度。  
   - 这一步相当于把各专家的“建议”按照它们对该 token 的兴趣度加权合并，得到最终的 token 表示。  

**最巧妙的地方**  
- **软路由的隐式实现**：作者没有显式把每个 token 分配给固定的专家，而是让所有专家都“看到”所有 token，只是计算量上做了稀疏筛选。这样既保留了软分配的全局信息，又不牺牲稀疏 MoE 的高效性。  
- **梯度流的双向加权**：在前向时权重决定信息流向，反向时同样的权重把梯度分配回去，确保每个 token 的学习信号能够遍及所有专家，避免了硬路由导致的梯度瓶颈。  

### 实验与效果
- **测试任务**：论文主要在视觉识别领域评估，使用了 ImageNet 等大规模图像分类数据集。  
- **对比基线**：与密集的 Vision Transformer（ViT）以及两种流行的稀疏 MoE（Tokens Choice、Experts Choice）进行比较。  
- **性能提升**：在相同计算预算下，Soft MoE 超过了 ViT 和传统 MoE，具体提升幅度在论文中给出（如 Top‑1 精度提升数个百分点），但这里不列出精确数字。  
- **规模实验**：Soft MoE Huge/14 配置 128 个专家、16 层 MoE，参数量比 ViT Huge/14 多 40 倍，推理时间仅增加约 2%，质量显著提升。  
- **消融研究**：作者分别关闭软分配、稀疏筛选、以及 gating 网络，发现软分配是提升训练稳定性的关键，稀疏筛选则是保持低推理成本的核心。  
- **局限性**：论文主要在视觉任务上验证，尚未在 NLP 场景大规模实验；软路由的全局加权在极端长序列上可能带来额外的内存开销。作者也提到在极端高专家数时，gating 网络的计算占比会逐渐上升，需要进一步优化。

### 影响与延伸思考
这篇工作打开了“软化稀疏 MoE”的新思路，随后有几篇后续论文尝试把相同的软路由理念搬到语言模型、跨模态模型上（如 SoftMoE‑LM、Hybrid‑MoE），并探索更高效的 gating 结构（比如低秩近似）。如果想继续深入，可以关注以下方向：  
- **低秩或稀疏 gating 的加速**：让软路由在数千专家时仍保持轻量。  
- **跨模态软 MoE**：把视觉专家和语言专家放在同一软路由框架里，实现更灵活的特征共享。  
- **自适应稀疏度**：根据输入复杂度动态调整每个专家实际计算的 token 比例。  

### 一句话记住它
Soft MoE 用可微的软路由把所有 token 的信息广播给每个专家，却只让专家计算自己最感兴趣的那部分，从而兼顾大容量和低成本。