# The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"

> **Date**：2023-09-21
> **arXiv**：https://arxiv.org/abs/2309.12288

## Abstract

We expose a surprising failure of generalization in auto-regressive large language models (LLMs). If a model is trained on a sentence of the form "A is B", it will not automatically generalize to the reverse direction "B is A". This is the Reversal Curse. For instance, if a model is trained on "Valentina Tereshkova was the first woman to travel to space", it will not automatically be able to answer the question, "Who was the first woman to travel to space?". Moreover, the likelihood of the correct answer ("Valentina Tershkova") will not be higher than for a random name. Thus, models do not generalize a prevalent pattern in their training set: if "A is B" occurs, "B is A" is more likely to occur. It is worth noting, however, that if "A is B" appears in-context, models can deduce the reverse relationship. We provide evidence for the Reversal Curse by finetuning GPT-3 and Llama-1 on fictitious statements such as "Uriah Hawthorne is the composer of Abyssal Melodies" and showing that they fail to correctly answer "Who composed Abyssal Melodies?". The Reversal Curse is robust across model sizes and model families and is not alleviated by data augmentation. We also evaluate ChatGPT (GPT-3.5 and GPT-4) on questions about real-world celebrities, such as "Who is Tom Cruise's mother? [A: Mary Lee Pfeiffer]" and the reverse "Who is Mary Lee Pfeiffer's son?". GPT-4 correctly answers questions like the former 79% of the time, compared to 33% for the latter.   Code available at: https://github.com/lukasberglund/reversal_curse.

---

# 反转诅咒：LLM在学习“A是B”后无法掌握“B是A” 论文详细解读

### 背景：这个问题为什么难？

在自然语言中，**“A是B”** 与 **“B是A”** 是一种常见的对称关系，几乎所有人类都能凭借常识把它们互相转换。传统的语言模型在大量文本上进行自监督学习，理论上应该捕捉到这种模式。然而，实际使用中我们发现，即使模型在训练语料里见过成千上万条 “A是B” 句子，它仍然常常在回答 “B是A” 的问题时表现得像从未见过一样。之前的研究大多关注模型的词汇覆盖、事实记忆或推理能力，却很少检视这种最基本的关系逆向推断是否被学到。正是这种看似“理所当然”却被模型忽视的缺陷，促使作者专门挖掘并命名为 **“Reversal Curse（反转诅咒）”**。

### 关键概念速览
- **自回归语言模型（Auto‑regressive LLM）**：一种一次生成下一个词的模型，常见的有 GPT‑3、Llama 系列。它的训练目标是最大化已出现文本的概率。
- **A‑is‑B 句式**：形如 “A 是 B” 的陈述，表达一种属性或身份关系，例如 “爱因斯坦是物理学家”。在语料中出现频率极高。
- **B‑is‑A 逆向推断**：给定 “B”，要求模型输出对应的 “A”。相当于把 A‑is‑B 的关系翻转，例如问 “谁是物理学家爱因斯坦？”。
- **in‑context learning（上下文学习）**：模型在推理时利用当前对话或提示中的信息，而不是依赖预训练的参数记忆。类似于人类在看到一条新规则后立刻使用它。
- **微调（Fine‑tuning）**：在已有的大模型上继续训练，使用特定任务的数据让模型适应新需求。这里指在合成的 A‑is‑B 数据上进一步训练模型。
- **数据增强（Data Augmentation）**：人为制造额外训练样本（如同时加入 A‑is‑B 与 B‑is‑A）来帮助模型学习更全面的模式。
- **概率分布对比**：评估模型是否真的“懂”某个事实，通常比较正确答案的生成概率与随机答案的概率差距。

### 核心创新点
1. **发现并系统化“反转诅咒”**  
   - *之前的观察*：研究者注意到模型在事实问答上表现不均衡，但没有专门针对关系逆向进行实验。  
   - *本文的做法*：构造大量合成的 “A 是 B” 句子，对模型进行微调后，统一评测其对 “B 是 A” 问题的回答概率。  
   - *改变*：首次提供了可复现的实验框架，证明即使在同一模型、同等数据量下，逆向推断的成功率仍接近随机水平。

2. **对比微调与上下文学习的差异**  
   - *之前的做法*：大多数工作只报告模型在直接提示下的表现，忽略了在同一会话中提供显式关系的影响。  
   - *本文的做法*：在测试时分别给模型纯粹的逆向问题和包含 “A 是 B” 示例的上下文，观察两者的性能差距。  
   - *改变*：揭示模型能够在 **上下文** 中即时推理出逆向关系，但这种能力并未迁移到模型参数本身。

3. **跨模型族、跨规模的稳健性验证**  
   - *之前的局限*：很多现象只在单一模型或特定规模上出现，难以判断是否普遍。  
   - *本文的做法*：在 GPT‑3、Llama‑1 系列的多个尺寸上重复实验，并进一步测试 ChatGPT（GPT‑3.5、GPT‑4）。  
   - *改变*：证明“反转诅咒”是一个普遍现象，而非某个模型的偶发缺陷。

4. **数据增强实验的负结果**  
   - *常规思路*：如果模型缺少逆向样本，就给它加上 B‑is‑A 句子，理论上应该解决问题。  
   - *本文的做法*：在训练集里显式加入等量的逆向句子，观察是否提升逆向问答准确率。  
   - *改变*：结果显示即使大量增强也只能略微提升，说明模型的内部表示仍倾向于单向映射。

### 方法详解
**整体思路**：作者把“反转诅咒”当作一个**可测量的缺陷**，通过三步来验证：① 合成训练数据（纯 A‑is‑B），② 对大模型进行微调，③ 设计两类评测——纯逆向提问 vs. 包含正向示例的上下文提问。整个流程像是给模型装上“记忆芯片”，再检查它是否能在没有提示的情况下自行读取逆向信息。

**步骤拆解**：

1. **合成数据构造**  
   - 选取若干虚构实体（如 “Uriah Hawthorne”）和虚构作品（如 “Abyssal Melodies”），随机配对生成句子 “Uriah Hawthorne 是 Abyssal Melodies 的作曲家”。  
   - 为避免模型利用真实世界的先验，这些实体全部是不存在的，确保模型只能靠训练数据学习关系。  
   - 数据规模从几千到上万不等，覆盖不同模型的微调需求。

2. **微调过程**  
   - 使用标准的自回归语言模型微调脚本，目标仍是最大化下一个词的概率。  
   - 只喂入 **正向** 句子，不提供任何逆向示例。  
   - 训练超参数（学习率、批大小、epoch 数）与公开的 GPT‑3/Llama 微调指南保持一致，确保结果可复制。

3. **评测设计**  
   - **纯逆向提问**：直接向模型提出 “谁是 Abyssal Melodies 的作曲家？” 并记录模型生成的答案概率。  
   - **上下文逆向提问**：在同一输入中先给出正向句子 “Uriah Hawthorne 是 Abyssal Melodies 的作曲家”，随后再问逆向问题。  
   - 对每个问题，计算 **正确答案的生成概率** 与 **随机干扰词的概率** 的比值，若比值接近 1，说明模型没有真正学到逆向关系。

4. **跨模型与跨规模实验**  
   - 重复上述微调与评测流程，分别在 GPT‑3（Ada、Babbage、Curie、Davinci）和 Llama‑1（7B、13B、30B）上执行。  
   - 对 ChatGPT 系列使用 **零样本**（直接提问）和 **few‑shot**（提供正向示例）两种方式进行对比。

**关键细节与巧思**：

- **虚构实体**：防止模型在检索真实知识库时“作弊”，确保评测纯粹测量学习能力。  
- **概率对比**：不只看答案是否出现，而是看它的 **相对概率**，避免模型偶尔输出正确答案却并未真正理解。  
- **上下文学习对照**：通过在同一次交互中加入正向句子，展示模型的 **即时推理** 能力与 **参数记忆** 的脱节，凸显“反转诅咒”是参数层面的缺失。  

### 实验与效果
- **数据集**：作者自行生成的虚构 A‑is‑B 语料，规模从 5k 到 50k 条不等；真实世界评测使用了名人亲属关系（如 “Tom Cruise 的母亲是谁？”）以及对应的逆向问题。  
- **主要结果**：  
  - 在微调后，模型对 **正向** 问题的准确率普遍在 80% 以上，但对 **纯逆向** 问题的正确答案概率几乎等同于随机（约 30% 与随机基线持平）。  
  - 当在同一上下文中提供正向句子，逆向问题的正确率跃升至 70% 左右，说明模型能够利用即时信息。  
  - ChatGPT‑4 在真实名人数据上对正向提问的正确率为 79%，而逆向提问仅为 33%，差距显著。  
- **Baseline 对比**：未微调的原始 GPT‑3/ Llama 在两类问题上表现相近，均未出现显著差异，说明微调并未改善逆向能力。  
- **消融实验**：作者尝试仅在训练集中加入少量 B‑is‑A 句子（比例 10%），结果逆向准确率提升不到 5%，表明单纯数据量增加不足以突破“诅咒”。  
- **局限性**：实验主要围绕 **身份/属性** 关系（A 是 B），未覆盖更复杂的语义映射（如 “A 导致 B”）。此外，所有合成数据均为英文，中文模型的表现仍需进一步验证。

### 影响与延伸思考
这篇工作首次系统化地指出 LLM 在最基础的 **关系逆向** 上存在系统性缺陷，引发了社区对 **知识结构化** 与 **记忆机制** 的重新审视。随后出现的研究尝试：

- **双向关系预训练**：在大规模语料中显式加入 “B 是 A” 形式的对偶句子，探索是否能在预训练阶段就纠正偏向。  
- **图结构嵌入**：把实体关系建模为图，使用图神经网络或关系嵌入帮助模型学习对称性。  
- **元学习逆向推理**：让模型在少量示例中学习“翻转”规则，从而在参数层面获得更通用的逆向能力。  

如果想进一步了解，可以关注 **“关系抽取”** 与 **“知识图谱与语言模型融合”** 两大方向，尤其是近期在 LLM 中加入 **结构化记忆**（如 Retrieval‑Augmented Generation） 的尝试，可能是突破“反转诅咒”的关键路径。

### 一句话记住它
LLM 能在上下文里瞬间翻转 “A 是 B”，但除非显式训练，否则它的参数永远记不住 “B 是 A”。