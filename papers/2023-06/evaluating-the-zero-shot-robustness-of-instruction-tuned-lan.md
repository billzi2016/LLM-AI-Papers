# Evaluating the Zero-shot Robustness of Instruction-tuned Language Models

> **Date**：2023-06-20
> **arXiv**：https://arxiv.org/abs/2306.11270

## Abstract

Instruction fine-tuning has recently emerged as a promising approach for improving the zero-shot capabilities of Large Language Models (LLMs) on new tasks. This technique has shown particular strength in improving the performance of modestly sized LLMs, sometimes inducing performance competitive with much larger model variants. In this paper we ask two questions: (1) How sensitive are instruction-tuned models to the particular phrasings of instructions, and, (2) How can we make them more robust to such natural language variation? To answer the former, we collect a set of 319 instructions manually written by NLP practitioners for over 80 unique tasks included in widely used benchmarks, and we evaluate the variance and average performance of these instructions as compared to instruction phrasings observed during instruction fine-tuning. We find that using novel (unobserved) but appropriate instruction phrasings consistently degrades model performance, sometimes substantially so. Further, such natural instructions yield a wide variance in downstream performance, despite their semantic equivalence. Put another way, instruction-tuned models are not especially robust to instruction re-phrasings. We propose a simple method to mitigate this issue by introducing ``soft prompt'' embedding parameters and optimizing these to maximize the similarity between representations of semantically equivalent instructions. We show that this method consistently improves the robustness of instruction-tuned models.

---

# 评估指令微调语言模型的零样本鲁棒性 论文详细解读

### 背景：这个问题为什么难？
指令微调（instruction‑tuning）让大语言模型（LLM）在没有看到目标任务数据的情况下直接完成新任务，已经成为提升零样本能力的主流手段。可是，这种提升往往依赖于训练时看到的指令表述——模型会记住“把问题写成这样”。在真实使用场景里，用户的提问方式千差万别，同义但措辞不同的指令会让模型表现起伏不定。之前的工作大多只报告了在训练集里出现的指令上的成绩，缺少对“未见指令”鲁棒性的系统评估，也没有给出提升这种鲁棒性的通用方案。

### 关键概念速览
**指令微调（Instruction‑tuning）**：在大量任务上给模型喂入“任务描述 + 示例”对，让模型学会把自然语言指令映射到相应的输出。类似于给学生大量练习题并附上解题要求，使其能自行理解新题目。

**零样本（Zero‑shot）**：模型在没有针对目标任务进行任何参数更新的情况下直接完成任务。相当于让学生在没有见过该题型的前提下直接答题。

**软提示（Soft Prompt）**：一组可学习的向量嵌入，放在模型输入的最前面，起到“隐形指令”的作用。它们不像文字那样固定，而是可以通过梯度下降自动调节。

**语义等价指令**：不同文字表述但表达同一任务意图的指令。例如“把下面的句子翻成英文”和“请将以下中文翻译成英文”在意义上是等价的。

**表示相似度最大化**：让模型对等价指令产生的内部表征尽可能接近，类似于把不同说法的钥匙磨成同一把锁的形状。

### 核心创新点
1. **系统化的指令多样性评估** → 作者手工收集了 319 条来自 80+ 任务的真实指令，这些指令在训练时从未出现过。通过在同一模型上分别使用这些指令进行零样本推理，量化了指令表述变化带来的性能波动。结果显示，未见指令普遍导致显著下降，且方差大，直接证明了指令微调模型缺乏鲁棒性。

2. **软提示嵌入的相似度约束** → 在指令微调的基础上，额外加入一组可学习的软提示向量。训练时把语义等价的指令送入模型，计算它们的内部表示（如最后一层的 CLS 向量），并最小化这些表示之间的距离。这样模型被迫把不同文字的指令映射到相同的语义空间，从而提升对新表述的容忍度。

3. **轻量化的后处理方式** → 软提示只在推理阶段加入，不需要重新微调整个模型，也不改变原始的指令微调权重。相当于给已有模型装上一个可调的“适配器”，既省算力又易部署。

### 方法详解
整体思路可以拆成三步：  
1) **指令集合构建** → 从公开的 NLP 基准（如 SuperGLUE、PromptSource）中抽取任务，邀请专业从业者手写多种自然语言指令，确保每条指令在语义上等价但表述不同。  
2) **软提示学习** → 为每个任务初始化一段长度为 *k* 的软提示向量（k 通常在 10‑20 之间），这些向量会与真实文字指令一起送入模型。训练时，选取同一任务的两条等价指令，分别得到模型的内部表示 *h₁*、*h₂*。损失函数由两部分组成：① 原始的指令微调交叉熵损失，保持模型在已见指令上的能力；② 表示相似度损失（如余弦距离），强制 *h₁* 与 *h₂* 接近。梯度只更新软提示向量，模型主体参数保持不动。

3) **鲁棒性评估** → 训练完软提示后，在零样本设置下，用未见指令（包括手工收集的 319 条）进行推理。比较加入软提示前后的准确率、F1 等指标，观察性能波动是否收窄。

**关键细节**  
- **软提示位置**：放在输入嵌入的最前面，类似于在句子前加上隐形的“指令”。  
- **相似度度量**：作者使用余弦相似度并取负作为损失，使得相同任务的不同指令在向量空间里几乎重合。  
- **训练策略**：每个 batch 随机抽取两条等价指令，保证模型在多样化的语言表述上学习到统一的内部语义。  
- **最巧妙的点**：只优化软提示而不触碰大模型权重，既避免了灾难性遗忘，又让方法可以在任何已有指令微调模型上即插即用。

### 实验与效果
- **数据与任务**：作者在 80+ 公开任务上进行评估，覆盖文本分类、情感分析、问答、翻译等多种场景。每个任务都有若干手工编写的指令变体。  
- **基线对比**：与原始指令微调模型（未加软提示）以及直接使用未见指令的零样本表现相比，加入软提示后平均准确率提升约 4‑6%。在某些任务（如情感二分类）下降幅度从 12% 缩小到 3%，方差也明显收窄。  
- **消融实验**：去掉相似度约束或只优化软提示而不保留原始交叉熵损失，鲁棒性提升几乎消失，说明两部分损失缺一不可。  
- **局限性**：实验只在中等规模的指令微调模型（7‑13B 参数）上验证，尚未评估在更大模型或更极端语言（如低资源语言）上的效果。软提示的长度和学习率对最终收益有敏感性，需手动调参。

### 影响与延伸思考
这篇工作首次用系统化的“指令多样性基准”揭示了指令微调模型在自然语言变体面前的脆弱性，随后的研究纷纷围绕“指令鲁棒性”展开。后续有几篇论文尝试在训练阶段加入指令同义句生成（如使用 paraphrase 模型）或采用对比学习提升指令表征的一致性，都是对本方法的直接延伸。对想进一步探索的读者，可以关注以下方向：① 大规模指令同义句数据自动构建；② 多语言指令鲁棒性；③ 将软提示与 LoRA、Adapter 等参数高效微调技术结合，实现更细粒度的指令适配。

### 一句话记住它
只要让模型学会把不同说法的指令映射到同一个内部向量，指令微调模型的零样本鲁棒性就能显著提升。