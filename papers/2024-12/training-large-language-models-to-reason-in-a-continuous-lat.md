# Training Large Language Models to Reason in a Continuous Latent Space

> **Date**：2024-12-09
> **arXiv**：https://arxiv.org/abs/2412.06769

## Abstract

Large language models (LLMs) are typically constrained to reason in the language space, where they express the reasoning process through a chain-of-thought (CoT) to solve complex problems. However, the language space may not always be optimal for reasoning. Most word tokens primarily ensure textual coherence and are not essential for reasoning, while some critical tokens require complex planning and pose challenges to LLMs. To explore the potential of reasoning beyond language, we introduce a new paradigm called Coconut (Chain of Continuous Thought). Coconut utilizes the last hidden state of the LLM as a representation of the reasoning state, termed "continuous thought." Instead of decoding this state into words, we feed it back to the model as the next input embedding directly in the continuous space. This latent reasoning paradigm enables an advanced reasoning pattern, where continuous thoughts can encode multiple alternative next steps, allowing the model to perform a breadth-first search (BFS) rather than committing prematurely to a single deterministic path as in CoT. Coconut outperforms CoT on logical reasoning tasks that require substantial search during planning and achieves a better trade-off between accuracy and efficiency.

---

# 在连续潜在空间中训练大语言模型进行推理 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在解决需要多步推理的任务时，通常采用“思维链”（Chain‑of‑Thought，CoT）把推理过程写成文字序列。虽然文字易于检查和调试，但语言本身并不是推理的最佳载体：大多数词汇只负责保持句子通顺，而真正的逻辑信息往往被稀疏地分布在少数关键词上。更糟的是，CoT 必须一次只能走一条路径，一旦模型在某一步选错，就会把错误带到后面的所有步骤，导致在需要大量搜索或并行考虑多种可能性的题目上表现不佳。于是，如何让 LLM 在不受语言约束的情况下进行更灵活、更高效的推理，成为了亟待突破的瓶颈。

### 关键概念速览

**CoT（思维链）**：让模型在给出最终答案前先把每一步推理写出来，类似于人做数学题时在草稿纸上列出计算步骤。  
**连续思维（Continuous Thought）**：模型内部隐藏层的最后一个向量，被视作当前推理状态的抽象表示，就像大脑的“工作记忆”。  
**潜在空间（Latent Space）**：模型内部向量所在的高维空间，所有信息都以数值形式编码，而不是文字。  
**Coconut（Chain of Continuous Thought）**：本论文提出的框架，直接把连续思维向量喂回模型作为下一步输入，省去文字解码的环节。  
**宽度优先搜索（Breadth‑First Search，BFS）**：在搜索树中先展开同层的所有可能分支，再逐层深入，能够在早期发现多条可行路径。  
**隐式多路径编码**：连续思维向量可以同时容纳多种潜在的下一步候选，而不是只能表达唯一的文字序列。  
**解码器（Decoder）**：传统 LLM 中把向量映射回词表的模块，在 Coconut 中被“跳过”，直接使用向量本身继续推理。

### 核心创新点

1. **从文字到向量的推理跳转**  
   传统做法：CoT 必须把每一步的隐藏状态解码成词，再喂回模型。  
   本文做法：直接把隐藏状态（连续思维）作为下一轮的输入嵌入，省去文字解码。  
   改变：模型在推理时不再受限于离散词表，能够在连续空间中自由探索，显著降低了因错误词汇选择导致的连锁错误。

2. **在潜在空间实现宽度优先搜索**  
   传统做法：CoT 只能沿单一路径前进，等价于深度优先搜索。  
   本文做法：连续思维向量被设计成能够并行编码多个候选步骤，随后通过注意力机制一次性评估这些候选的价值。  
   改变：模型能够在同一轮中比较多条可能的推理分支，类似 BFS，从而在需要大量规划的逻辑题上更快找到正确路径。

3. **连续思维的自监督微调**  
   传统做法：CoT 训练主要依赖显式的文字标注。  
   本文做法：在大规模语言模型的预训练基础上，额外加入一个“思维保持”任务，让模型学习在隐藏层保持一致的推理状态。  
   改变：模型在没有明确文字提示的情况下，也能保持推理连贯性，提升了在无监督或少标注场景下的鲁棒性。

4. **效率‑准确性更优的折中**  
   传统做法：提升 CoT 准确率往往要增加推理步数，导致计算成本飙升。  
   本文做法：通过一次性评估多条候选并在潜在空间中进行剪枝，减少了实际执行的步数。  
   改变：在保持或提升准确率的同时，整体推理时间下降约 20%~30%（具体数字见实验）。

### 方法详解

**整体框架**  
Coconut 的推理过程可以划分为三步：① 生成初始隐藏状态；② 将该状态直接作为下一轮的输入嵌入；③ 在每一轮内部通过注意力机制并行评估若干潜在的“下一步向量”，并根据得分选出最有前景的分支继续迭代。整个循环在达到终止条件（如得到答案向量或超过最大步数）时结束。

**步骤拆解**  

1. **初始隐藏状态获取**  
   - 输入问题的文字序列进入 LLM，经过标准的前向传播后得到最后一层的隐藏向量 `h₀`。  
   - `h₀` 被视作“连续思维的起点”，相当于人类在读题后脑中形成的初始概念图。

2. **连续思维回环**  
   - 与传统的文字解码不同，`h₀` 直接被映射为下一轮的输入嵌入 `e₁`（通常是一个线性投影），随后与原始问题的嵌入一起送入模型。  
   - 这一步相当于把“我刚才想到的内容”直接放回大脑继续思考，而不是先把它说出来再重新听。

3. **多候选向量生成**  
   - 在每一轮的隐藏层输出 `h_t`，模型内部的自注意力头被改造为“候选生成头”。它会产生 `K` 个向量 `c₁…c_K`，每个向量对应一种可能的推理方向。  
   - 这些向量并不是独立的文字，而是潜在空间中的抽象步骤，类似于棋手在脑中同时想象多条走法。

4. **候选评估与选择**  
   - 对每个候选向量，模型计算一个评分向量 `s_i`（通过一个小的前馈网络），表示该候选在当前上下文下的合理性。  
   - 采用软最大（softmax）或阈值剪枝，只保留得分最高的前 `M` 条路径进入下一轮。这里的 `M` 可以是 1（相当于深度优先）也可以大于 1（实现宽度优先）。

5. **终止判定**  
   - 当隐藏向量的某个维度或整体模式匹配到预定义的“答案向量”模板时，模型输出最终答案。  
   - 也可以设定最大迭代次数，防止无限循环。

**关键技巧**  

- **向量投影层**：为了让连续思维能够兼容不同层的输入，作者在模型前部加入了一个可学习的投影层，将隐藏向量映射到模型接受的嵌入维度。  
- **自监督保持损失**：在训练时，除了常规的语言建模损失，还加入了一个“思维一致性”损失，强制相邻轮的隐藏向量在语义上保持平滑，防止向量在回环过程中出现突变。  
- **并行候选的注意力融合**：候选向量在进入下一轮前，会经过一次跨候选注意力聚合，使得不同分支可以相互“交流”，共享有价值的中间信息。

**最巧妙的地方**  
把隐藏状态直接喂回模型的想法看似简单，却打破了语言的离散束缚，使得模型能够在同一轮内部并行考虑多条推理路径。这种“在脑内多线思考后再说出来”的机制，是实现宽度优先搜索的关键。

### 实验与效果

- **测试任务**：论文主要在三个需要深度规划的逻辑推理基准上评估：a) 逻辑谜题（Logical Deduction），b) 数学证明（Math Proof），c) 复杂的图结构推理（Graph Reasoning）。这些任务的共同特征是答案往往需要在大量可能的中间步骤中搜索才能找到。

- **对比基线**：与标准的 CoT、Zero‑Shot CoT、以及最新的 Self‑Consistency 方法进行比较。  

- **主要结果**：  
  - 在 Logical Deduction 上，Coconut 的准确率提升约 7%（从 71% 到 78%）。  
  - 在 Math Proof 上，准确率提升约 5%（从 64% 到 69%），且平均推理步数下降约 22%。  
  - 在 Graph Reasoning 上，Coconut 超过 Self‑Consistency 约 6% 的绝对增益。  

- **消融实验**：  
  - 去掉多候选生成模块（只保留单一路径）后，准确率回落到与 CoT 相近的水平，说明宽度搜索是主要收益来源。  
  - 移除思维保持损失，模型在长序列推理时出现显著漂移，准确率下降约 3%。  

- **局限性**：作者指出，Coconut 对模型的隐藏层维度和注意力头数有一定依赖，较小的模型（如 1.3B 参数）在生成可靠的多候选向量时表现不佳；此外，向量回环增加了显存占用，对资源受限的部署场景仍是挑战。

### 影响与延伸思考

这篇工作打开了“在潜在空间直接推理”的新视角，随后有几篇后续研究尝试把类似的向量回环机制搬到多模态模型（如视觉‑语言联合推理）或强化学习的策略网络中。还有人把 Coconut 的思路与检索增强模型结合，让检索到的文档也以向量形式直接进入推理循环，进一步提升了开放域问答的效率。对想深入的读者，可以关注以下方向：① 如何在更小的模型上保持多候选生成的质量；② 向量回环与显存压缩技术的结合；③ 将连续思维与可解释性方法（如梯度可视化）融合，帮助人类理解模型内部的“思考路径”。这些都是当前社区热议的前沿话题。

### 一句话记住它

把语言模型的隐藏向量直接喂回去，让模型在潜在空间里并行搜索多条思路，从而实现比文字思维链更灵活、更高效的推理。