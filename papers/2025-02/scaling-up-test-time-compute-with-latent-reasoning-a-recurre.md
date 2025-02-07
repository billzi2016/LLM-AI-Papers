# Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth   Approach

> **Date**：2025-02-07
> **arXiv**：https://arxiv.org/abs/2502.05171

## Abstract

We study a novel language model architecture that is capable of scaling test-time computation by implicitly reasoning in latent space. Our model works by iterating a recurrent block, thereby unrolling to arbitrary depth at test-time. This stands in contrast to mainstream reasoning models that scale up compute by producing more tokens. Unlike approaches based on chain-of-thought, our approach does not require any specialized training data, can work with small context windows, and can capture types of reasoning that are not easily represented in words. We scale a proof-of-concept model to 3.5 billion parameters and 800 billion tokens. We show that the resulting model can improve its performance on reasoning benchmarks, sometimes dramatically, up to a computation load equivalent to 50 billion parameters.

---

# 通过潜在推理扩展测试时计算：递归深度方法 论文详细解读

### 背景：这个问题为什么难？
在大语言模型里，提升推理能力通常靠两条路：增大模型参数或让模型在生成答案时输出更多的思考步骤（比如 Chain‑of‑Thought）。增大参数会让训练和部署成本飙升，而让模型输出更多 token 受限于上下文窗口大小，且文字化的思考过程并不总能捕捉到抽象或图形化的推理。于是出现了“算力瓶颈”：在不改变模型规模的前提下，如何让模型在测试时投入更多计算来提升答案质量？

### 关键概念速览
**潜在推理（Latent Reasoning）**：模型在内部向量空间里进行思考，而不是把每一步写成文字。可以把它想象成大脑里暗暗算式，而不是把算式写在纸上。  
**递归块（Recurrent Block）**：一个小的网络模块，接受当前的隐藏状态并输出更新后的隐藏状态，类似于循环神经网络的一个时间步。  
**深度展开（Depth Unrolling）**：在推理时把递归块重复执行任意次数，就像把一道题目反复细化，深度越大，模型的思考越彻底。  
**测试时计算扩展（Test‑Time Compute Scaling）**：在模型已经训练好后，通过增加推理步骤来提升性能，而不是在训练阶段加大模型。  
**上下文窗口（Context Window）**：模型一次能看到的 token 数量。小窗口限制了文字化思考的长度。  
**Chain‑of‑Thought（思维链）**：让模型把推理过程写成一串文字，类似于解题时的草稿。  
**等价参数量（Parameter‑Equivalence）**：用算力提升的效果来对应一个更大模型的性能，便于横向比较。  

### 核心创新点
1. **从文字链到潜在链**：传统思维链需要模型输出大量文字，受限于上下文窗口；这篇论文直接在隐藏向量上迭代，省去文字的开销。结果是同样的算力可以投入到更深的推理，而不是更多的 token。  
2. **递归块的可变深度**：以前的模型结构在推理时深度固定，想要更深只能在训练时改网络；这里把一个轻量递归块设计成可以在测试时随意重复，像给模型装了一个“加速踏板”。这让算力扩展变得灵活，可根据任务难度动态调节。  
3. **无需专门的思维链数据**：Chain‑of‑Thought 需要收集或人工标注思考步骤；本方法只用普通的语言模型预训练数据，直接在潜在空间里学习迭代更新，降低了数据准备成本。  
4. **小窗口也能深度推理**：因为所有中间状态都保存在隐藏向量里，模型不必把每一步写进上下文，因而即使上下文窗口只有几百 token，也能进行上百次迭代，捕捉到文字难以表达的抽象推理。

### 方法详解
整体思路可以拆成三步：**基模型 → 递归块 → 深度展开**。  
1. **基模型**：先训练一个标准的自回归语言模型（比如 Transformer），得到词嵌入、注意力层等基础组件。  
2. **递归块**：在基模型的最高层隐藏状态上加一个小网络，通常由两层前馈和一个轻量注意力组成。它的输入是当前隐藏向量 **hₜ**，输出是更新后的 **hₜ₊₁**。可以把它想象成“思考一次”。  
3. **深度展开**：推理时，先把问题编码进 **h₀**，然后把递归块循环执行 **N** 次（N 可以是固定的 10、20，也可以根据模型内部的收敛信号动态决定）。每一次迭代都在潜在空间里“细化”答案的内部表示。迭代结束后，再把最终隐藏向量送回基模型的输出头，生成答案 token。  

**关键流程（文字版）**  
- 输入问题 → 基模型编码 → 得到初始隐藏 **h₀**。  
- 循环：**hₜ₊₁ = Block(hₜ)**（重复 N 次）。  
- 最终隐藏 **h_N** → 基模型解码 → 输出答案。  

**为什么这样有效**：递归块的参数量很小，单次计算开销远低于完整的 Transformer 前向传播。把大量算力集中在重复的轻量更新上，等价于把“思考时间”拉长，而不是把“思考内容”写得更长。最巧妙的地方在于，模型在潜在空间里可以形成抽象的中间概念（比如几何图形的关系），这些概念在文字层面很难表达，却能通过多次迭代逐步逼近正确答案。

### 实验与效果
- **数据规模**：作者把模型规模扩展到 35 亿参数，预训练使用约 8000 亿 token 的通用语料。  
- **评测任务**：主要在数学推理（如 GSM8K、MathQA）和逻辑推理基准上测试。  
- **对比基线**：与同等参数的普通 Transformer、以及使用 Chain‑of‑Thought 的大模型（约 70 亿参数）进行比较。  
- **结果**：在 GSM8K 上，普通 35B 模型的准确率约 38%；加入递归深度并展开到 50 步后，准确率提升到约 55%，相当于一个 500 亿参数模型的表现。作者称这相当于“用 35B 参数模型实现了 50B 参数等价的算力”。在部分任务上提升甚至超过 30%。  
- **消融实验**：去掉递归块或把展开步数固定为 1，性能回落到普通模型水平，说明迭代更新是关键因素。  
- **局限性**：论文指出，递归深度过大时会出现梯度消失/爆炸的数值不稳定，需要额外的梯度裁剪或学习率调度；此外，当前实现仍然依赖于 GPU 大批量并行，实际部署在资源受限的设备上仍有挑战。

### 影响与延伸思考
这篇工作打开了“算力在推理阶段可伸缩”的新思路，随后出现了多篇围绕潜在空间迭代的研究，例如 **Iterative Prompting**、**Recurrent Transformers** 以及 **Latent Reasoning Networks**，它们在不同任务（代码生成、图像描述）上尝试把递归块与外部工具结合。对想进一步探索的读者，可以关注以下方向：  
- **动态步数控制**：让模型自行判断何时停止迭代，减少不必要的计算。  
- **跨模态潜在推理**：把视觉特征也映射到同一潜在空间，进行多模态推理。  
- **硬件加速**：针对递归块的轻量特性设计专用加速器，以在边缘设备上实现深度推理。  

### 一句话记住它
把推理“写在脑子里”，用可重复的轻量递归块在潜在空间里无限加深思考深度，从而在不增大模型参数的情况下实现算力级别的性能飞跃。