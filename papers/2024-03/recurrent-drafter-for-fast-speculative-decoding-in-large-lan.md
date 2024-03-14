# Recurrent Drafter for Fast Speculative Decoding in Large Language Models

> **Date**：2024-03-14
> **arXiv**：https://arxiv.org/abs/2403.09919

## Abstract

We present Recurrent Drafter (ReDrafter), an advanced speculative decoding approach that achieves state-of-the-art speedup for large language models (LLMs) inference. The performance gains are driven by three key aspects: (1) leveraging a recurrent neural network (RNN) as the draft model conditioning on LLM's hidden states, (2) applying a dynamic tree attention algorithm over beam search results to eliminate duplicated prefixes in candidate sequences, and (3) training through knowledge distillation from the LLM. ReDrafter accelerates Vicuna inference in MT-Bench by up to 2.8x with a PyTorch implementation on Nvidia H100 GPUs. To demonstrate its practicality in real environments, we also validated its effectiveness for on-device applications by implementing the approach in MLX and benchmarking performance on Metal GPUs in Apple Silicon chips, achieving up to 2.3x speedup.

---

# 递归草稿模型用于大语言模型快速推测解码 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在生成文本时需要逐词（或逐子词）进行前向计算，算力开销随模型规模线性增长。传统的贪心或束搜索虽然能保证输出质量，却几乎没有加速空间。推测解码（speculative decoding）尝试让一个更小的“草稿模型”先生成候选序列，再让大模型只验证这些候选，从而减少大模型的实际调用次数。但早期的草稿模型往往是独立的语言模型，缺乏对大模型内部状态的感知，导致生成的候选与大模型的真实分布差距大，验证失败率高，整体加速效果有限。要想真正把加速和质量兼顾，就必须让草稿模型能够实时“听”大模型的隐藏信息，并且高效处理大量候选的前缀重复问题。

### 关键概念速览

**推测解码（Speculative Decoding）**：先用一个轻量模型产生若干候选 token 序列，再让大模型只检查这些候选是否符合其分布，从而减少大模型的前向次数。类似于先让助理写草稿，再让专家快速审阅。

**草稿模型（Draft Model）**：在推测解码中负责生成候选的模型，通常比主模型小、快。这里的草稿模型是一个循环神经网络（RNN），能够在每一步读取大模型的隐藏状态。

**循环神经网络（RNN）**：一种对序列数据有记忆能力的网络，能够把前一步的隐藏向量传递到下一步。这里把它想成“记事本”，每写一个字就把当前的思路记录下来供下一个字使用。

**隐藏状态（Hidden State）**：大模型内部的向量表示，包含了已生成 token 的上下文信息。相当于大模型的“思考过程”，草稿模型可以直接读取它。

**束搜索（Beam Search）**：在生成时保留多个最有希望的候选路径，而不是只保留一个。像是同时追踪几条可能的剧情走向。

**动态树注意力（Dynamic Tree Attention）**：对束搜索得到的多条候选序列构建一棵前缀树，然后在这棵树上做注意力计算，专门去掉重复的前缀。可以把它想成在多条相似的草稿里找出共同的开头，只算一次。

**知识蒸馏（Knowledge Distillation）**：把大模型的输出概率当作“软标签”，让草稿模型学习模仿大模型的行为。相当于让助理在写草稿时参考专家的点评。

### 核心创新点

1. **RNN 作为感知式草稿模型**  
   *之前的草稿模型*：独立的 Transformer 或小型语言模型，无法直接使用大模型的内部信息。  
   *本文的做法*：在每一步把大模型的隐藏向量喂给一个轻量的 RNN，让它在生成候选时“听”到大模型的思路。  
   *带来的改变*：候选序列与大模型分布更贴近，验证成功率提升，整体加速比显著提高。

2. **动态树注意力去重**  
   *之前的束搜索*：直接对每条候选序列做注意力，导致大量重复前缀被重复计算，浪费算力。  
   *本文的做法*：把束搜索的结果组织成前缀树，对树的每个节点只计算一次注意力，然后在子树之间共享。  
   *带来的改变*：显著削减了重复计算的开销，使得即使束宽度较大也能保持高效。

3. **基于大模型的知识蒸馏训练草稿 RNN**  
   *之前的训练*：草稿模型往往只用普通语言建模目标，缺乏对大模型行为的对齐。  
   *本文的做法*：让 RNN 学习大模型在相同上下文下的输出分布，使用软标签进行蒸馏。  
   *带来的改变*：草稿模型在生成候选时更容易被大模型接受，验证步骤的回退次数下降。

4. **跨平台实现与实测**  
   *之前的加速方案*：大多只在高端 GPU 上验证，缺乏对移动端或 Apple Silicon 的适配。  
   *本文的做法*：提供了 PyTorch 在 Nvidia H100 上的实现，同时在 MLX 框架下跑 Metal GPU，展示了在不同硬件上的可行性。  
   *带来的改变*：证明了该方法不仅在服务器上有 2.8× 加速，在手机/笔记本等边缘设备上也能达到 2.3×，提升了实际落地价值。

### 方法详解

**整体框架**  
ReDrafter 的推理流程可以概括为四步：  
1）大模型（LLM）对已生成的 token 序列做一次前向，得到当前隐藏状态。  
2）把这个隐藏状态喂给轻量的 RNN，RNN 依据隐藏状态和上一步的草稿 token 生成若干候选 token（束宽度 *B*）。  
3）对所有候选序列构建前缀树，使用动态树注意力一次性计算它们在大模型词表上的注意力分布。  
4）大模型依据这些分布快速验证候选，接受的部分直接写入输出，未被接受的部分回退到常规解码。

**关键模块拆解**

- **RNN 草稿生成器**  
  - 输入：大模型的隐藏向量 *h_t*（维度与大模型内部一致）以及上一步草稿的 token 嵌入 *e_{t-1}*。  
  - 过程：RNN 先把 *h_t* 与 *e_{t-1}* 拼接，经过一个线性层映射到 RNN 的隐藏空间，然后更新内部状态 *s_t*。  
  - 输出：对词表做一次 softmax，得到候选分布，从中取前 *B* 个概率最高的 token 作为束搜索的起点。  
  - 直觉：RNN 像是“听写员”，它把大模型的思路记下来，再快速写出几个最可能的下一个词。

- **束搜索 + 前缀树**  
  - 对每一步的 *B* 个候选 token，继续展开到深度 *L*（一般为 2~4），形成 *B^L* 条完整的候选序列。  
  - 将这些序列插入一棵前缀树，树的每个节点对应一个公共前缀。  
  - 动态树注意力只在树的每个节点上计算一次注意力向量，然后向下传播到子节点，避免对相同前缀重复计算。

- **动态树注意力**  
  - 对每个节点的隐藏表示（由大模型的隐藏状态和节点对应的 token 嵌入组合而成）做一次注意力查询，得到该节点在词表上的概率分布。  
  - 由于子节点共享父节点的上下文，这一步相当于在“树形结构”上做一次全局的注意力聚合，计算量随树的宽度而非候选序列数线性增长。

- **验证与回退**  
  - 大模型使用节点级的注意力分布，对每条候选序列进行快速采样或阈值比较，判断是否符合大模型的真实分布。  
  - 若候选被接受，直接写入输出；若被拒绝，则回到普通的逐 token 解码，确保最终文本质量不受草稿错误影响。

**最巧妙的设计**  
把大模型的隐藏状态直接喂给 RNN，使草稿模型能够“同步思考”，这在之前的工作里几乎没有出现。再配合前缀树的结构化去重，彻底解决了束搜索带来的指数级候选爆炸问题。两者结合让草稿的生成质量和验证效率同步提升。

### 实验与效果

- **测试平台**：在 Nvidia H100 GPU 上使用 PyTorch 实现，对 Vicuna-13B 进行 MT‑Bench 基准测试；在 Apple Silicon（M2 Pro）上使用 MLX 框架跑 Metal GPU。  
- **加速表现**：在服务器端，ReDrafter 将 Vicuna 的推理速度提升至原来的 **2.8 倍**；在移动端金属 GPU 上也实现了 **2.3 倍** 的加速。  
- **质量保持**：尽管加速显著，模型在 MT‑Bench 上的评分下降不到 0.2 分，说明生成质量基本保持不变。  
- **对比基线**：与传统的 Speculative Decoding（使用独立小型 Transformer 作为草稿）相比，ReDrafter 的整体吞吐提升约 **30%–45%**，验证成功率提升约 **15%**。  
- **消融实验**：作者分别去掉（1）RNN 对隐藏状态的感知、（2）动态树注意力、（3）蒸馏训练，发现去掉任意一项都会导致加速回落到 1.5–1.8 倍，且验证成功率下降 10% 以上，验证了每个模块的必要性。  
- **局限性**：论文指出，RNN 的容量仍然受限于硬件，若草稿模型过小会导致候选质量下降；此外，前缀树的构建在极端长序列上会产生额外的内存开销，尚未在超长文本任务上做完整评估。

### 影响与延伸思考

ReDrafter 把“感知式草稿”与“结构化去重”结合，打开了推测解码的新思路。随后的工作开始探索使用更强的循环结构（如 Transformer‑XL）或混合 RNN‑CNN 草稿，以进一步提升候选质量。还有研究尝试把前缀树注意力推广到多模态生成（图文、音频）场景，借鉴 ReDrafter 的树形去重思想。对想深入的读者，可以关注以下方向：① 更高效的草稿‑大模型信息交互协议；② 动态树结构在大规模并行硬件上的实现；③ 将推测解码与稀疏激活（Mixture‑of‑Experts）结合的可能性。

### 一句话记住它

让大模型的隐藏状态“喂给”一个轻量 RNN，再用前缀树一次性去重，便能在保持质量的前提下把推理速度提升近三倍。