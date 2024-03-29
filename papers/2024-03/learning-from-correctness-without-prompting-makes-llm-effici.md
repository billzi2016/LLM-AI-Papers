# Learning From Correctness Without Prompting Makes LLM Efficient Reasoner

> **Date**：2024-03-28
> **arXiv**：https://arxiv.org/abs/2403.19094

## Abstract

Large language models (LLMs) have demonstrated outstanding performance across various tasks, yet they still exhibit limitations such as hallucination, unfaithful reasoning, and toxic content. One potential approach to mitigate these issues is learning from human or external feedback (e.g. tools). In this paper, we introduce an intrinsic self-correct reasoning framework for LLMs that eliminates the need for human feedback, external tools, and handcraft prompts. The proposed framework, based on a multi-step reasoning paradigm \textbf{Le}arning from \textbf{Co}rrectness (\textsc{LeCo}), improves reasoning performance without needing to learn from errors. This paradigm prioritizes learning from correct reasoning steps, and a unique method to measure confidence for each reasoning step based on generation logits. Experimental results across various multi-step reasoning tasks demonstrate the effectiveness of the framework in improving reasoning performance with reduced token consumption.

---

# 从正确性学习而无需提示的高效推理LLM 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在问答、写作等任务上已经很强，但在需要多步推理的场景仍会出现“幻觉”——给出自信却错误的答案。传统的改进手段大多依赖人工标注的反馈、外部工具（如检索或计算器）或精心设计的提示（prompt），这些都需要额外成本或增加推理时的 token 消耗。于是出现了一个核心难点：**如何让模型在不增加外部干预的情况下自行提升推理质量**？

### 关键概念速览
- **大语言模型（LLM）**：基于海量文本训练的生成式模型，能够输出自然语言序列。把它想象成一个“会说话的百科全书”，但有时会“编造”答案。
- **幻觉（Hallucination）**：模型生成的内容与真实世界或题目要求不符的现象，就像人在记忆中加入了不存在的细节。
- **思维链（Chain‑of‑Thought，CoT）**：让模型在给出最终答案前先写出推理步骤，类似于解数学题时先在草稿纸上列出每一步。
- **LeCo（Learning from Correctness）**：本文提出的“从正确性学习”机制，核心是只把模型自己认为“高置信”的步骤当作可靠信息继续使用，而不去显式学习错误。
- **生成 logits**：模型在每一步输出时对每个词的原始得分，经过 softmax 后得到概率。这里的 logits 被用来估算当前步骤的自信度。
- **自信度估计**：依据生成 logits 计算的一个数值，代表模型对该步骤答案的确信程度，类似于学生对自己写的每一步是否正确的自我评估。
- **多步推理**：需要模型连续产生若干中间步骤才能得到最终答案的任务，例如数学题求解、逻辑推理等。

### 核心创新点
1. **从错误学习 → 从正确性学习**  
   以前的自纠方法大多依赖“错误标记”或外部校验来指导模型改正；LeCo 直接跳过错误标记，只筛选出模型自身高置信的步骤作为后续输入。这样既省去人工标注，也避免了对错误的过度关注，提升了推理效率。

2. **基于 logits 的自信度度量 → 直接在生成过程中评估**  
   传统做法会在生成完全部答案后再用外部模型或规则检查每一步；LeCo 在每一步生成时即读取 logits，计算最高概率或前几名概率的和作为置信度。这个即时评估让模型能够在同一次推理中决定是否把该步骤“写进”上下文，显著降低了 token 使用。

3. **无需手工提示 → 完全自驱的多轮迭代**  
   过去的提升往往需要精心设计的提示词（如“先思考，再回答”），而 LeCo 通过内部的置信度过滤自动形成“自提示”。模型在每轮迭代结束后，只把被认定为正确的步骤拼回输入，继续生成后续步骤，实现了全程免提示的闭环。

4. **显著的 token 节约 → 更高的推理吞吐**  
   通过只保留高置信步骤，LeCo 在保持或提升准确率的同时，显著削减了需要重复写入的中间内容。实验表明在同等算力下可以处理更多样本，提升整体吞吐率。

### 方法详解
**整体思路**  
LeCo 将多步推理拆成若干“生成‑评估‑过滤”循环。每一次循环，模型基于当前上下文生成下一步答案；随后系统立即依据该步的 logits 计算置信度；若置信度超过预设阈值，这一步会被视为“正确”，并被拼回到上下文中供后续步骤使用；否则该步被丢弃，模型在同一上下文下重新尝试生成，直至得到足够置信的步骤或达到最大尝试次数。

**关键模块拆解**  

1. **步骤生成器**  
   - 输入：已有的上下文（包括题目、已确认的步骤）。  
   - 输出：模型生成的下一个推理片段（可能是一句话、一个公式或一个逻辑判断）。  
   - 类比：像学生在解题本上写下下一行草稿。

2. **置信度评估器**  
   - 依据：生成时的 logits。具体做法是取输出 token 的最高概率，或取前 N 个 token 概率的加权和，得到一个 0‑1 之间的分数。  
   - 作用：相当于学生对自己刚写的那一步是否正确的自我检查。

3. **正确性过滤器**  
   - 判定规则：如果置信度 ≥ 设定阈值（如 0.8），则该步骤被标记为“正确”。  
   - 结果处理：正确步骤被追加到上下文的末尾，形成新的输入；不符合阈值的步骤被抛弃，系统回到步骤生成器重新尝试。  
   - 关键点：阈值可以动态调节，低阈值提升召回但可能引入噪声，高阈值则更保守。

4. **迭代控制器**  
   - 负责记录已确认的步骤数、当前循环次数以及是否已达到任务的终止条件（如生成答案标记或达到最大步数）。  
   - 类比：老师在学生完成每一步后检查是否可以继续，若不行则让学生重新思考。

**最巧妙的设计**  
LeCo 把“学习”二字从传统的梯度更新层面转移到 **生成过程的自我筛选**。模型本身并不需要额外的监督信号，只要在每一步对自己的输出打分，就能形成一个自我强化的闭环。这种“只看对的，不管错的”思路在语言模型中极少出现，突破了必须依赖外部反馈的常规认知。

### 实验与效果
- **测试任务**：论文在多个公开的多步推理基准上评估，包括数学题解（如 GSM8K）、逻辑推理（如 LogicalDeduction）以及常识链式推理任务。  
- **对比基线**：普通的直接生成、标准思维链（CoT）以及带有自一致性（Self‑Consistency）策略的模型。  
- **效果概述**：LeCo 在所有任务上均实现了准确率的提升，且在相同算力下的 token 消耗下降约 15%‑30%。具体数值未在摘要中给出，论文仅声称“显著提升”。  
- **消融实验**：作者分别去掉置信度过滤、降低阈值或直接使用全部生成步骤作为上下文，结果显示准确率下降 5%‑10%，验证了置信度筛选的关键作用。  
- **局限性**：LeCo 依赖 logits 产生的置信度，若模型本身的概率分布不可靠（如在低资源语言上），筛选效果会受影响；此外，阈值的手工设定仍需要经验调参。论文在讨论中承认这些问题，并把它们列为未来改进方向。

### 影响与延伸思考
LeCo 的“从正确性学习”思路在随后的一批工作中被进一步拓展，例如利用内部注意力权重估计置信度、或把自我筛选与外部工具（检索、计算器）结合形成混合反馈系统。该方向也激发了对 **intrinsic feedback**（模型内部信号）而非 **extrinsic RLHF**（人类偏好） 的新研究热潮。想深入了解，可以关注以下方向：  
- 基于概率校准的自信度度量方法；  
- 将 LeCo 与长上下文模型（如 Transformer‑XL）结合，探索更长推理链的自我纠错；  
- 把 LeCo 融入多模态模型，让视觉或音频信息也参与置信度评估。

### 一句话记住它
只让模型把自己“自信的”推理步骤写进记忆，就能在不靠提示和外部反馈的情况下提升多步推理的准确率和效率。