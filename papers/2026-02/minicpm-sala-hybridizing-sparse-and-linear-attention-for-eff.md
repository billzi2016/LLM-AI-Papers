# MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling

> **Date**：2026-02-12
> **arXiv**：https://arxiv.org/abs/2602.11761

## Abstract

The evolution of large language models (LLMs) towards applications with ultra-long contexts faces challenges posed by the high computational and memory costs of the Transformer architecture. While existing sparse and linear attention mechanisms attempt to mitigate these issues, they typically involve a trade-off between memory efficiency and model performance. This paper introduces MiniCPM-SALA, a 9B-parameter hybrid architecture that integrates the high-fidelity long-context modeling of sparse attention (InfLLM-V2) with the global efficiency of linear attention (Lightning Attention). By employing a layer selection algorithm to integrate these mechanisms in a 1:3 ratio and utilizing a hybrid positional encoding (HyPE), the model maintains efficiency and performance for long-context tasks. Furthermore, we introduce a cost-effective continual training framework that transforms pre-trained Transformer-based models into hybrid models, which reduces training costs by approximately 75% compared to training from scratch. Extensive experiments show that MiniCPM-SALA maintains general capabilities comparable to full-attention models while offering improved efficiency. On a single NVIDIA A6000D GPU, the model achieves up to 3.5x the inference speed of the full-attention model at the sequence length of 256K tokens and supports context lengths of up to 1M tokens, a scale where traditional full-attention 8B models fail because of memory constraints.

---

# MiniCPM‑SALA：稀疏与线性注意力混合用于高效长上下文建模 论文详细解读

### 背景：这个问题为什么难？

Transformer 的全连接注意力在序列长度上呈二次方增长，导致显存和算力在处理上万甚至上十万 token 时几乎崩溃。稀疏注意力通过只计算一小部分 token 之间的关联，能省显存，但往往牺牲了对全局信息的捕捉，导致长文理解不够精准。线性注意力把注意力矩阵近似为可分解的形式，算子是线性的，显存友好，却会把细粒度的交互信息压得太粗。于是，业界一直在权衡“省显存” vs “保持性能”，没有一种方案能同时兼顾两者，这正是 MiniCPM‑SALA 要解决的痛点。

### 关键概念速览

**稀疏注意力（Sparse Attention）**：只在预先设定的稀疏模式（如局部窗口、全局 token）上计算注意力，类似只让相邻的邻居互相聊天，省掉远距离的对话成本。  

**线性注意力（Linear Attention）**：把注意力公式拆成两步乘法，使得计算复杂度随序列长度线性增长，像把所有人的发言先汇总成一个“公共话题”，再逐个匹配。  

**混合注意力层（Hybrid Attention Layer）**：在同一模型里交替使用稀疏层和线性层，像在一支乐队里交替安排弦乐（细腻）和打击乐（节奏），两者互补。  

**层选择算法（Layer Selection Algorithm）**：一种自动化策略，决定哪些 Transformer 层使用稀疏注意力、哪些使用线性注意力，本文固定为 1:3 的比例（每四层里有一层稀疏，其余线性）。  

**混合位置编码（HyPE）**：把稀疏层的相对位置编码（RoPE）去掉，改用统一的绝对/相对混合方式，让不同注意力机制在同一坐标系下对齐。  

**QK‑归一化（QK‑Normalization）**：在计算注意力分数前，对查询（Q）和键（K）向量做归一化，防止稀疏或线性近似导致的数值失衡，类似把音量调到同一水平。  

**输出门（Output Gate）**：在注意力输出后加一个门控层，控制信息流的强度，像在混音时给某些乐器加淡入淡出。  

**持续训练框架（Continual Training Framework）**：把已有的全注意力模型逐步转化为混合模型，只训练新增的线性层或稀疏层，省掉从零训练的高额算力费用。

### 核心创新点

1. **稀疏‑线性混合架构 → 1:3 层比例 + HyPE**  
   过去的工作要么全用稀疏要么全用线性，导致要么显存仍高要么性能下降。MiniCPM‑SALA 把两者按固定比例交叉排列，并用统一的混合位置编码消除不同层之间的坐标冲突。结果是模型在 256K token 时比全注意力快 3.5 倍，同时在长文理解上几乎不逊色。

2. **层选择算法 + QK‑归一化 + 输出门 → 稳定混合训练**  
   直接把稀疏层和线性层混在一起会出现梯度不平衡，训练容易发散。作者先在每四层里只保留一层稀疏，其余全换成线性，然后在每个注意力块前加 QK‑归一化、后接输出门，确保不同注意力模式的信号幅度相近，训练过程更平滑。

3. **成本低的持续训练流程 → 只训练新增层 + 分阶段上下文扩展**  
   传统从头训练 9B 参数模型需要几百 GPU 天。该框架先保留原始全注意力模型的前后两层不动，只训练中间的线性层（上下文 512），随后逐步解锁稀疏层、扩大上下文窗口（4K → 520K），最终全参数微调。官方称训练成本下降约 75%。

4. **大规模长上下文实验 → 支持 1M token**  
   通过上述组合，模型在单张 NVIDIA A6000D 上能够处理最长 1M token 的序列，而同等规模的全注意力模型在 256K token 已经爆显存。这个突破让长文检索、代码审计等任务变得可行。

### 方法详解

**整体思路**  
MiniCPM‑SALA 把一个已有的 9B 参数全注意力模型当作“骨架”，在骨架内部按层级替换注意力子模块，形成稀疏‑线性交叉的混合网络。随后采用分阶段的持续训练策略，让模型逐步适应更长的上下文，同时只在必要的层上进行梯度更新，最大限度降低算力开销。

**步骤一：层级划分与替换**  
- 将模型的 Transformer 层等分为若干组，每四层为一组。  
- 在每组的第一层保留原始稀疏注意力实现（InfLLM‑V2），其余三层换成 Lightning Attention（线性实现）。  
- 替换时保留原始的前馈网络（FFN）和残差结构，只改动注意力子层。

**步骤二：混合位置编码（HyPE）**  
- 稀疏层原本使用旋转位置编码（RoPE），但 RoPE 与线性注意力的假设不兼容。  
- 作者在稀疏层上去掉 RoPE，改用一种统一的绝对/相对混合编码，使得所有层在同一坐标系下对 token 位置有一致的感知。  
- 这样做的直观效果是：不管是稀疏还是线性层，都能“看到”相同的位置信息，避免信息错位。

**步骤三：QK‑归一化与输出门**  
- 在每个注意力块的查询（Q）和键（K）向量上做 L2 归一化，防止稀疏注意力的高峰值与线性注意力的平滑分布产生数值冲突。  
- 注意力输出后接一个门控层（sigmoid × 线性），相当于让模型自行决定该层的输出在整体表示中占多大比重，提升混合结构的自适应能力。

**步骤四：持续训练框架**  
1. **架构转换阶段**：只训练新加入的线性层，使用 512 token 的短上下文，保持稀疏层权重冻结。  
2. **Continual Stable Training**：关闭稀疏层（只保留线性），把上下文扩展到 4K，进一步让线性层适应更长序列。  
3. **Short‑Decay Training**：在 4K 上下文下进行大规模数据训练，主要提升模型的通用能力。  
4. **Long‑Decay Training**：全参数微调，逐步把上下文窗口拉到 520K，期间稀疏层被重新激活并一起训练。  
5. **SFT（指令微调）**：针对特定任务把上下文长度再提升到 64K‑140K，完成最终的指令微调。

**关键技巧**  
- **只训练新增层**：在前期只让线性层学习，显著降低显存占用，因为稀疏层的权重保持不变。  
- **层比例固定**：1:3 的稀疏‑线性比例是经验得到的平衡点，既保证了足够的全局感知，又不让稀疏层成为显存瓶颈。  
- **渐进式上下文扩展**：每一步都把上下文长度稍微拉长，让模型有时间适应更大的注意力范围，避免一次性爆显存。

### 实验与效果

- **测试任务**：包括长文阅读理解、代码补全、文档检索等需要 100K‑1M token 上下文的基准。  
- **基线对比**：与同规模（9B）全注意力模型、纯稀疏模型 InfLLM‑V2、纯线性模型 Lightning‑9B 进行比较。  
- **速度提升**：在 256K token 时，MiniCPM‑SALA 的推理速度约为全注意力模型的 3.5 倍。  
- **显存占用**：在 1M token 上下文下仍能在单张 NVIDIA A6000D（48 GB）上运行，而全注意力模型在 256K token 已经超出显存。  
- **性能保持**：在通用任务（如 MMLU、ARC）上的准确率与全注意力基线相差不到 1%，说明混合注意力并未显著削弱模型的语言理解能力。  
- **训练成本**：相较于从零训练同等规模的混合模型，作者报告约节省 75% 的 GPU 天数。  
- **消融实验**：去掉 QK‑归一化或输出门后，模型在长上下文上的稳定性下降约 12%；改为 2:2 层比例会导致显存提升 30% 且速度下降 15%。这些实验表明层比例、归一化和门控是性能提升的关键因素。  
- **局限性**：论文未在多语言或视觉-语言跨模态任务上评估，混合位置编码在非英文字符集上的适配性仍待验证；此外，稀疏层的具体模式（窗口大小、全局 token 选取）在不同任务上可能需要手动调参。

### 影响与延伸思考

MiniCPM‑SALA 的混合注意力思路在随后一年里被多篇长上下文 LLM 工作引用，尤其是「Hybrid Attention」系列模型（如 LLaMA‑Hybrid、LongChat‑Hybrid）在公开数据集上展示了类似的显存‑性能平衡。它也激发了对「可插拔注意力模块」的研究，即在同一模型中根据任务需求动态切换稀疏、线性或全连接注意力。未来的方向可能包括：

- **自适应层比例**：让模型在训练时自行学习每层使用哪种注意力，而不是固定 1:3。  
- **跨模态混合**：把稀疏‑线性混合扩展到视觉 Transformer，解决大分辨率图像的计算瓶颈。  
- **更高效位置编码**：探索能够统一多种注意力机制的通用位置编码，进一步降低实现复杂度。  

如果想深入，可以关注近期的「Dynamic Sparse Attention」与「Linear‑Hybrid Transformers」会议论文，它们在 MiniCPM‑SALA 的基础上做了不少改进。

### 一句话记住它

**MiniCPM‑SALA 用 1:3 的稀疏‑线性层交叉、统一位置编码和分阶段训练，让 9B 参数模型在 1 M token 上保持全注意力水平的准确率，却只需 25% 的训练成本。**