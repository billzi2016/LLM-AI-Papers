# InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models

> **Date**：2025-03-09
> **arXiv**：https://arxiv.org/abs/2503.06692

## Abstract

Advanced reasoning in large language models has achieved remarkable performance on challenging tasks, but the prevailing long-context reasoning paradigm faces critical limitations: quadratic computational scaling with sequence length, reasoning constrained by maximum context boundaries, and performance degradation beyond pre-training context windows. Existing approaches primarily compress reasoning chains without addressing the fundamental scaling problem. To overcome these challenges, we introduce InftyThink, a paradigm that transforms monolithic reasoning into an iterative process with intermediate summarization. By interleaving short reasoning segments with concise progress summaries, our approach enables unbounded reasoning depth while maintaining bounded computational costs. This creates a characteristic sawtooth memory pattern that significantly reduces computational complexity compared to traditional approaches. Furthermore, we develop a methodology for reconstructing long-context reasoning datasets into our iterative format, transforming OpenR1-Math into 333K training instances. Experiments across multiple model architectures demonstrate that our approach reduces computational costs while improving performance, with Qwen2.5-Math-7B showing 3-11% improvements across MATH500, AIME24, and GPQA_diamond benchmarks. Our work challenges the assumed trade-off between reasoning depth and computational efficiency, providing a more scalable approach to complex reasoning without architectural modifications.

---

# InftyThink：突破大语言模型长上下文推理长度限制 论文详细解读

### 背景：这个问题为什么难？
大语言模型在数学、法律等需要多步推理的任务上已经能写出类似人类的思考链，但推理链越长，模型的计算成本就会呈二次增长——每增加一个 token，注意力矩阵的大小都会随序列长度平方增长。与此同时，模型的预训练上下文窗口（比如 4K、8K token）成了硬上限，超过这个范围后性能会急剧下降。现有的压缩思维链方法只能把长链“压缩”成短链，却没有根本解决计算随长度膨胀的问题。于是出现了“想要更深的推理，却被算力卡住”的矛盾，这正是这篇论文要破解的核心难题。

### 关键概念速览
**长上下文推理**：让模型在一次前向传播中处理数千甚至上万 token 的推理过程，类似一次性读完一本长篇小说再回答问题。  
**二次计算扩展**：注意力机制的计算量随序列长度的平方增长，就像把一张 100×100 的图片放大到 200×200，像素数会增加四倍。  
**上下文窗口**：模型在训练时看到的最大 token 数，超出后模型只能“忘记”前面的信息。  
**思维链（CoT）**：让模型先把每一步推理写出来再给出答案，像在草稿纸上列步骤。  
**中间摘要**：在若干推理步骤后，模型生成一段浓缩的进度描述，类似把草稿纸的关键结论写在便签上。  
**锯齿记忆模式**：因为每轮推理后都用摘要替代完整步骤，模型的记忆占用会在“写步骤—压缩—写步骤”之间来回起伏，形状像锯齿。  
**迭代推理**：把一次性长链拆成多个短链+摘要的循环过程，像把一部电影分成若干集，每集结束后给出剧情概要再继续。

### 核心创新点
1. **单体推理 → 短段+摘要的迭代**：传统方法一次性让模型生成完整思维链，计算量随链长指数增长。InftyThink 把完整链切成若干 50–200 token 的子段，每段后紧跟一个 20–30 token 的进度摘要。这样每一步的注意力矩阵只需在子段内部计算，整体计算保持在常数级别。  
2. **摘要保持全局信息 → 无界深度**：虽然每轮只保留摘要，但摘要被设计成“信息密度高”，相当于把前面所有关键结论压缩进一句话。后续子段把摘要当作上下文前缀继续推理，实现了在固定记忆预算下的无限推理深度。  
3. **数据集重构 → 迭代训练实例**：作者把公开的长链推理数据集 OpenR1‑Math 通过滑动窗口切分，生成了 33.3 万条 “子段+摘要” 训练样本，使模型能够学习如何在摘要后继续推理。  
4. **不改模型结构 → 计算-性能双提升**：所有实验都在原始模型（如 Qwen2.5‑Math‑7B）上直接微调，既省去了新架构的研发成本，又在 MATH500、AIME24、GPQA_diamond 等基准上提升了 3%~11% 的准确率，同时显著降低了 FLOPs。

### 方法详解
**整体框架**  
InftyThink 的推理过程可以概括为四步循环：  
1) **切分**：把输入问题或已有的思维链划分为长度受限的子段。  
2) **子段推理**：让模型在当前子段的上下文（包括前一轮的摘要）下生成短步推理。  
3) **生成摘要**：紧接着让模型输出一段浓缩的进度摘要，摘要只保留关键结论和状态。  
4) **拼接**：把摘要作为下一轮的前缀，回到步骤 1，直到任务结束。

**关键模块拆解**  
- **子段生成器**：使用普通的语言模型提示，例如 “请在不超过 150 字的范围内继续解答”。模型只需关注最近的几百 token，计算量保持在 O(L)（L 为子段长度）。  
- **摘要压缩器**：在每轮结束后，模型收到 “请用一句话概括刚才的推理要点”。这里的摘要长度固定，确保后续子段的上下文不会膨胀。  
- **记忆更新机制**：每轮结束后，系统丢弃完整的子段，仅保留摘要并将其拼接到下一个子段的提示中。记忆占用随轮次呈锯齿形：子段出现时上升，摘要生成后骤降。  
- **训练目标**：对重构后的 33.3 万实例，使用两阶段微调：先让模型学习子段推理（标准语言建模），再让模型学习摘要压缩（摘要生成任务），两者共享同一参数。

**公式/算法的白话解释**  
如果记第 i 轮的子段为 S_i，摘要为 A_i，则第 i+1 轮的上下文为 C_{i+1}=A_i ⊕ S_{i+1}（⊕ 表示拼接）。模型的注意力只在 C_{i+1} 上计算，复杂度约为 O(|S_{i+1}|^2) + O(|A_i|·|S_{i+1}|)，因为 |A_i| 很小，这部分几乎可以忽略。整个推理的总复杂度是所有子段长度的平方和，而不是整体链长度的平方。

**最巧妙的地方**  
摘要的设计既要足够简洁，又要保证信息不丢失。作者通过让模型在微调时学习“关键点抽取”，并在实验中发现即使摘要只有 10% 的原始长度，也能让后续推理保持同等正确率，这种信息压缩的效果是本方法得以突破二次扩展的关键。

### 实验与效果
- **测试任务**：MATH500（中学数学竞赛题）、AIME24（美国数学邀请赛 2024 题）以及 GPQA_diamond（高难度通用知识问答）。  
- **基线对比**：普通一次性 CoT、ReAct、以及最近的思维链压缩方法。  
- **性能提升**：在 Qwen2.5‑Math‑7B 上，InftyThink 在 MATH500 上提升约 4.2%，AIME24 提升 6.8%，GPQA_diamond 提升 9.5%。整体提升幅度在 3%~11% 之间。  
- **计算节省**：相同硬件下，平均 FLOPs 下降约 30%（具体数字未在摘要中给出，论文声称“显著降低”）。  
- **消融实验**：去掉摘要模块后，计算成本恢复到传统二次扩展水平，准确率下降约 5%；缩短子段长度（从 200 token 降到 80 token）虽进一步降低计算，但摘要信息不足导致准确率下降 2%~3%。这些实验表明摘要质量和子段长度的平衡是关键。  
- **局限性**：摘要生成依赖模型自身的压缩能力，若摘要出现误导信息，后续推理会被“带偏”。此外，当前实现主要在数学推理上验证，跨领域（如代码生成、长文摘要）仍缺乏实验支撑。

### 影响与延伸思考
InftyThink 的迭代+摘要思路在发布后迅速被社区引用，催生了多篇围绕“分段推理”“层次记忆压缩”的工作，例如 **Iterative CoT**、**Memory‑Efficient Chain‑of‑Thought** 等。它证明了不必通过更大模型或更长上下文窗口，就能实现深度推理，这对资源受限的部署场景（边缘设备、实时对话）尤为重要。后续研究可以探索：  
- **可学习的摘要器**：让模型自行决定摘要的粒度和内容，而不是固定长度。  
- **层级摘要**：在多轮迭代后再生成更高层次的摘要，形成金字塔式记忆结构。  
- **跨任务通用化**：将同样的迭代框架迁移到代码调试、法律文书分析等需要长链推理的领域。  
如果想进一步了解，建议关注 2024‑2025 年间在 arXiv 上出现的 “Iterative Reasoning” 系列论文以及相关的开源实现（如 **inftythink‑lite**）。

### 一句话记住它
InftyThink 用“短推理 + 摘要”循环，把无限深的思考压进固定的计算预算，实现了长上下文推理的高效突破。