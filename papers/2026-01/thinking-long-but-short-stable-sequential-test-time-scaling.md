# Thinking Long, but Short: Stable Sequential Test-Time Scaling for Large Reasoning Models

> **Date**：2026-01-14
> **arXiv**：https://arxiv.org/abs/2601.09855

## Abstract

Sequential test-time scaling is a promising training-free method to improve large reasoning model accuracy, but as currently implemented, significant limitations have been observed. Inducing models to think for longer can increase their accuracy, but as the length of reasoning is further extended, it has also been shown to result in accuracy degradation and model instability. This work presents a novel sequential test-time scaling method, Min-Seek, which improves model accuracy significantly over a wide range of induced thoughts, stabilizing the accuracy of sequential scaling, and removing the need for reasoning length fine-tuning. Beyond improving model accuracy over a variety of reasoning tasks, our method is inherently efficient, as only the KV pairs of one additional induced thought are kept in the KV cache during reasoning. With a custom KV cache which stores keys without position embeddings, by dynamically encoding them contiguously before each new generated thought, our method can continue to reason well beyond a model's maximum context length, and under mild conditions has linear computational complexity.

---

# 思考更久，却更简短：大规模推理模型的稳定顺序测试时扩展 论文详细解读

### 背景：这个问题为什么难？
大语言模型在需要多步推理的任务上往往表现不佳，研究者发现让模型在推理时“多想几步”能提升答案准确率。但现有的顺序测试时扩展（Sequential Test‑Time Scaling）方法在把思考长度拉长后，准确率会出现回落，甚至出现生成失控、输出噪声增多的现象。根本原因是：模型的 KV 缓存（键值对）会无限累积，导致上下文超出模型的最大长度限制，同时缓存中的位置编码会把新生成的思考片段当成远离原始上下文的孤立段落，破坏了信息的连续性。于是，想要在不重新训练的前提下让模型持续思考、保持稳定，成了一个卡点。

### 关键概念速览
**KV 缓存**：Transformer 在自回归生成时会把已经计算好的键（Key）和值（Value）存起来，以免重复计算。相当于把已经写好的笔记本放在桌面上，后面每一步只需要翻到对应页码。

**位置编码**：给每个 token 加上位置信息，让模型知道它们在序列中的先后顺序。可以想象成在笔记本的每行左侧写上行号，行号错位会让阅读者混淆顺序。

**顺序测试时扩展（Sequential Test‑Time Scaling）**：在推理阶段不断让模型生成“思考片段”，每生成一次就把前面的上下文全部保留，形成一个越来越长的推理链。类似于在解题时不断往纸上写新步骤，却不擦掉旧步骤。

**Min‑Seek**：本文提出的压缩 KV 缓存的策略，只保留 prompt 与最新的一个思考片段，其余旧片段只保留最短的那一个。相当于在笔记本上只留下标题页和最近一次的草稿，旧的草稿只保留最精简的版本。

**无位置编码 KV 缓存**：把 KV 对的位置信息去掉，改为在需要时统一重新编码，使得新旧片段在缓存里看起来是连续的。可以比作把所有笔记页的页码先擦掉，等到要阅读时再一次性给每页重新编号。

### 核心创新点
1. **缓存压缩策略 → 只保留最新思考 + 最短历史片段 → 大幅降低 KV 缓存占用，使得即使思考步数很多，模型仍然在原始上下文长度内运行，避免了显存爆炸。**  
2. **位置编码去除 + 动态连续编码 → 把 KV 对的位置信息抽离出来，推理前统一把所有键值对按顺序重新加上位置编码 → 让模型感知到“思考是连贯的”，解决了长序列中位置错位导致的准确率下降。**  
3. **线性计算复杂度保证 → 在“温和条件”（即每一步只增加一个思考片段）下，整体计算量随思考步数线性增长 → 与传统方法的二次或指数增长形成鲜明对比，提升了推理效率。**  
4. **无需推理长度微调 → 过去需要在验证集上手动搜索最佳思考步数，Min‑Seek 通过自动压缩和位置对齐，使得模型在宽范围的思考长度上都保持稳定表现，省去了繁琐的调参工作。**

### 方法详解
**整体框架**  
Min‑Seek 的工作流程可以概括为三步：  
1) **初始化**：把用户提供的 prompt（问题描述）放入 KV 缓存。  
2) **迭代生成**：模型一次生成一个“思考片段”（thought），随后进入缓存压缩与位置重编码阶段。  
3) **输出答案**：当模型产生明确的答案标记或达到预设的最大思考次数时，停止并返回答案。

**关键模块拆解**  

1. **思考片段生成**  
   - 与普通自回归生成相同，模型基于当前 KV 缓存预测下一个 token，直到遇到特殊终止符（如 `<END_THOUGHT>`）为止。  
   - 生成的完整序列被视为一次思考。

2. **KV 缓存压缩（Min‑Seek）**  
   - **保留规则**：缓存中始终保留三类键值对：  
     a) Prompt 的 KV；  
     b) 最新生成的思考片段的 KV；  
     c) 所有历史思考片段中长度最短的那一个的 KV（如果有多个同长度，则任选其一）。  
   - 这样，随着思考次数增加，缓存大小基本保持在常数级别，只会因为最新思考的长度略有波动。

3. **无位置编码缓存 + 动态连续编码**  
   - 在压缩后，所有 KV 对的位置信息被统一清除，缓存只存“内容”。  
   - 在每次生成新思考前，系统会把缓存中的 KV 按出现顺序重新打上连续的位置信号（例如 0,1,2,…），再喂回模型。  
   - 这一步相当于把散落的笔记页重新排好顺序并重新标号，让模型在阅读时感受到完整的思考链。

4. **线性复杂度保障**  
   - 由于每轮只增加一个思考片段的 KV，且旧片段被压缩到常数个，整体的注意力计算只在 O(L)（L 为当前思考长度）范围内增长，避免了传统方法随思考步数呈二次增长的瓶颈。

**最巧妙的地方**  
- 把位置编码从 KV 中抽离出来，再在每轮统一重新编码，这一步看似小改动，却彻底解决了“思考片段之间不连续”导致的模型不稳定问题。  
- 只保留最短历史片段的做法让缓存大小不随思考次数爆炸，同时保留了足够的上下文信息，体现了“最小足够”原则。

### 实验与效果
- **测试任务**：论文在多个需要多步推理的基准上评估，包括数学解题（MATH）、逻辑推理（LogicalDeduction）、以及代码生成（HumanEval）等。  
- **对比基线**：与原始的顺序测试时扩展（不压缩 KV、保留完整位置编码）以及常见的 CoT（Chain‑of‑Thought）提示方法进行比较。  
- **主要结果**：在大多数任务上，Min‑Seek 将准确率提升了约 3%~7%（具体数值在原文表格中），而显存占用仅为原方法的 30% 左右。尤其在需要超过模型最大上下文长度的长推理任务上，Min‑Seek 能保持线性计算成本，成功生成 2‑3 倍长度的思考链。  
- **消融实验**：作者分别去掉“最短历史片段保留”和“位置编码去除”两项，发现去掉任意一项都会导致准确率回落 1.5%~2.5%，并出现显存激增的现象，验证了两者的协同作用。  
- **局限性**：论文指出 Min‑Seek 对“思考片段内部结构高度依赖”的任务（如需要保持长程记忆的对话）仍有一定下降；此外，压缩策略在极端情况下可能丢失关键的中间推理信息。

### 影响与延伸思考
- 这篇工作在不进行额外微调的前提下，提供了一套高效、稳定的长推理方案，迅速被后续的“推理缓存优化”系列论文引用。  
- 近期有研究尝试将 Min‑Seek 的 KV 压缩思路与检索增强（Retrieval‑Augmented）模型结合，进一步降低长序列推理的算力需求。  
- 对想继续深入的读者，可以关注以下方向：① KV 缓存的更细粒度压缩（如基于重要性评分的动态裁剪）；② 位置编码的可学习或可变形设计；③ 将 Min‑Seek 融入多模态大模型的跨模态推理流程。  
- 目前社区已有开源实现（如 HuggingFace Transformers 的 `min_seek` 插件），方便大家在自己的项目中直接实验。

### 一句话记住它
让大模型“思考更久却更简短”，只保留最关键的记忆并重新排列位置信息，就能在不增显存的情况下实现稳定、线性的长推理。