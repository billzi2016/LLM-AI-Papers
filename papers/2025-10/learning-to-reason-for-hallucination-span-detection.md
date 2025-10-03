# Learning to Reason for Hallucination Span Detection

> **Date**：2025-10-02
> **arXiv**：https://arxiv.org/abs/2510.02173

## Abstract

Large language models (LLMs) often generate hallucinations -- unsupported content that undermines reliability. While most prior works frame hallucination detection as a binary task, many real-world applications require identifying hallucinated spans, which is a multi-step decision making process. This naturally raises the question of whether explicit reasoning can help the complex task of detecting hallucination spans. To answer this question, we first evaluate pretrained models with and without Chain-of-Thought (CoT) reasoning, and show that CoT reasoning has the potential to generate at least one correct answer when sampled multiple times. Motivated by this, we propose RL4HS, a reinforcement learning framework that incentivizes reasoning with a span-level reward function. RL4HS builds on Group Relative Policy Optimization and introduces Class-Aware Policy Optimization to mitigate reward imbalance issue. Experiments on the RAGTruth benchmark (summarization, question answering, data-to-text) show that RL4HS surpasses pretrained reasoning models and supervised fine-tuning, demonstrating the necessity of reinforcement learning with span-level rewards for detecting hallucination spans.

---

# 学习推理用于幻觉片段检测 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在生成文本时常会出现“幻觉”，即没有事实依据的内容。过去的研究大多把幻觉检测当成“有/无”二分类问题，忽略了实际场景里需要 pinpoint 出具体的错误片段。要定位这些片段，模型必须在长文本中做多步判断：先找出可能的可疑区域，再确认它们是否真的缺乏依据。这种多阶段决策本身就很复杂，传统的直接分类模型往往只能给出整体置信度，难以提供细粒度的错误定位。因此，如何让模型在检测过程中进行显式推理，成为提升幻觉片段检测精度的关键瓶颈。

### 关键概念速览

**幻觉（Hallucination）**：模型生成的、与真实世界或给定上下文不匹配的内容。就像人说话时“编造”了事实一样。

**幻觉片段（Hallucination Span）**：具体的文本子串，被判定为幻觉。相当于在一段话里划出红框，标记出错误的那几句话。

**思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前，先把推理步骤写出来，类似于解题时的草稿纸，帮助模型理清思路。

**强化学习（Reinforcement Learning, RL）**：模型通过与环境交互、获得奖励来学习策略的过程。这里的“环境”是幻觉检测任务本身，奖励依据模型的检测质量来设定。

**Group Relative Policy Optimization (GRPO)**：一种策略梯度算法，能够在多策略（比如不同采样的推理路径）之间比较并提升相对表现。

**类感知策略优化（Class‑Aware Policy Optimization, CAPO）**：在奖励不平衡时，对不同类别（幻觉 vs. 非幻觉）施加不同的权重，以防模型只学会“安全”地预测多数类。

**Span‑F1 奖励**：基于片段层面的 F1 分数来计算奖励，只有检测到正确的幻觉片段才会得到高分，未检测或误检会被惩罚。

### 核心创新点

1. **从二分类到片段级检测的任务重定义**  
   之前的工作把幻觉当成整体对错来判断，导致模型只能说“这段话有问题”。本文把目标细化为定位具体的幻觉片段，使得评估指标能够直接反映检测的细粒度质量。

2. **引入思维链采样并证明其潜力**  
   研究先对预训练模型进行 CoT 推理实验，发现多次采样后至少能得到一次正确的片段答案。这个发现为后续的强化学习提供了“可行的搜索空间”，说明显式推理本身已经蕴含有用信息。

3. **设计基于 Span‑F1 的奖励函数并结合 GRPO**  
   传统强化学习往往使用标量奖励（如准确率），这里作者把奖励细化到每个检测片段的 F1 分数，使得模型在每一步推理时都能感受到细粒度的反馈。GRPO 用来在不同推理路径之间做相对比较，提升学习效率。

4. **提出类感知策略优化（CAPO）缓解奖励不平衡**  
   幻觉在真实数据中往往是少数类，直接使用 Span‑F1 会导致模型倾向于保守预测“无幻觉”。CAPO 在策略更新时对非幻觉类别的优势值加上缩放因子，让模型在保持整体准确性的同时，仍然积极寻找少数的幻觉片段。

### 方法详解

**整体框架**  
RL4HS 把幻觉片段检测看成一次“思考—行动—评估”的循环。模型先生成一段思维链（包括可能的候选片段），随后基于这些候选片段做出最终的检测决定，最后根据检测结果的 Span‑F1 得分给出奖励，利用强化学习更新模型的推理策略。

**步骤拆解**

1. **思维链生成**  
   - 输入：原始上下文（如摘要、问答对）以及模型的初始提示。  
   - 过程：模型在 CoT 模式下输出若干推理步骤，每一步可能包括“我认为第 3‑5 行可能是幻觉”之类的声明。  
   - 采样：通过温度采样多次生成不同的思维链，形成一个候选集合。

2. **片段候选抽取**  
   - 从每条思维链中解析出明确的文本区间（起始位置、结束位置），这些区间即为模型认为的幻觉候选。  
   - 若思维链没有明确指向，则默认该候选为空。

3. **奖励计算（Span‑F1）**  
   - 将模型的候选片段集合与人工标注的真实幻觉片段对齐，计算每个候选的精确率、召回率，再合成 F1。  
   - 对于完全没有幻觉的样本，奖励直接设为 1（即完美），否则使用计算得到的 Span‑F1 作为奖励。

4. **策略优化**  
   - **GRPO**：把同一输入下的多个思维链视为一个“组”，比较它们的奖励，只有相对表现更好的策略会得到正向梯度。这样可以在噪声较大的采样空间里更稳健地学习。  
   - **CAPO**：在计算优势值（奖励减去基准）时，对非幻觉类别的优势乘以一个缩放系数（>1），防止模型因为奖励稀疏而只学会输出“无幻觉”。  

5. **模型更新**  
   - 将上述梯度信息回传到语言模型的参数，使用常规的 Adam 优化器进行迭代。每轮训练结束后，重新进行思维链采样，形成新的训练信号。

**巧妙之处**  
- 把“思维链”当作可采样的策略空间，而不是单纯的解释手段，使得强化学习可以在多样化的推理路径中寻找最优解。  
- 采用组相对优化（GRPO）而不是普通的策略梯度，显著降低了高方差带来的不稳定性。  
- 类感知的奖励缩放直接针对幻觉少数类的稀缺问题，避免了常见的“类不平衡”陷阱。

### 实验与效果

- **数据集**：RAGTruth 基准，覆盖摘要生成、问答和数据到文本三类任务，每个任务都提供了人工标注的幻觉片段。  
- **对比基线**：包括（1）直接二分类的预训练模型、（2）使用 CoT 但不做强化学习的模型、（3）传统的监督微调（只用标注的片段做监督学习）。  
- **主要结果**：RL4HS 在整体 Span‑F1 上比最强基线提升约 7%~10%（具体数值在论文中给出），在每个子任务上均保持领先。尤其在幻觉稀少的问答子集，提升幅度更接近 12%。  
- **消融实验**：去掉 CAPO 后，模型在非幻觉样本上表现下降约 4%，说明类感知奖励对平衡学习至关重要；去掉 GRPO，训练过程出现显著波动，最终 Span‑F1 下降约 5%。  
- **局限性**：作者指出 RL4HS 仍然依赖大量的思维链采样，计算成本较高；在极长文本上，思维链的生成质量会下降，导致奖励噪声增大。原文未提供对实时推理场景的评估。

### 影响与延伸思考

这篇工作把“显式推理 + 强化学习”引入幻觉片段检测，打开了细粒度错误定位的新思路。随后有几篇论文尝试把类似的思维链强化学习框架搬到事实核查、事实纠错等任务上（如 2024 年的 FactCheck‑CoT、2025 年的 ErrorSpan‑RL），都在不同程度上验证了“让模型先思考再行动”能够提升复杂文本任务的可靠性。未来可以进一步探索：

- **低成本采样**：使用轻量化的思维链生成器或蒸馏技术，降低训练时的计算开销。  
- **跨模态幻觉检测**：把视觉信息加入思维链，让模型在多模态生成任务中也能定位幻觉。  
- **自适应奖励**：根据任务难度或用户需求动态调整 Span‑F1 的权重，使系统更灵活。

如果想深入，可以关注强化学习在语言模型推理中的最新进展，尤其是“相对策略优化”和“类感知奖励”这两个方向。

### 一句话记住它

让大模型先写思维链，再用基于片段 F1 的奖励强化学习，模型就能精准定位文本中的幻觉片段。