# Inference-Time Scaling for Generalist Reward Modeling

> **Date**：2025-04-03
> **arXiv**：https://arxiv.org/abs/2504.02495

## Abstract

Reinforcement learning (RL) has been widely adopted in post-training for large language models (LLMs) at scale. Recently, the incentivization of reasoning capabilities in LLMs from RL indicates that $\textit{proper learning methods could enable effective inference-time scalability}$. A key challenge of RL is to obtain accurate reward signals for LLMs in various domains beyond verifiable questions or artificial rules. In this work, we investigate how to improve reward modeling (RM) with more inference compute for general queries, i.e. the $\textbf{inference-time scalability of generalist RM}$, and further, how to improve the effectiveness of performance-compute scaling with proper learning methods. For the RM approach, we adopt pointwise generative reward modeling (GRM) to enable flexibility for different input types and potential for inference-time scaling. For the learning method, we propose Self-Principled Critique Tuning (SPCT) to foster scalable reward generation behaviors in GRMs through online RL, to generate principles adaptively and critiques accurately, resulting in $\textbf{DeepSeek-GRM}$ models. Furthermore, for effective inference-time scaling, we use parallel sampling to expand compute usage, and introduce a meta RM to guide voting process for better scaling performance. Empirically, we show that SPCT significantly improves the quality and scalability of GRMs, outperforming existing methods and models in various RM benchmarks without severe biases, and could achieve better performance compared to training-time scaling. DeepSeek-GRM still meets challenges in some tasks, which we believe can be addressed by future efforts in generalist reward systems. The models are released at Hugging Face and ModelScope.

---

# 推理时扩展的通用奖励建模 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上使用强化学习（RL）已经成为提升对话质量、遵循指令等任务的主流手段，但 RL 的核心依赖是一个可靠的奖励函数。传统的奖励模型（Reward Modeling，RM）往往在可验证的问答或人工设定的规则上训练，难以覆盖日常对话、创意写作等开放式查询。即使有了不错的奖励模型，实际推理时往往只使用一次前向计算，算力利用率低，导致模型在更大计算预算下的性能提升有限。换句话说，如何让奖励模型在推理阶段“加大马力”，同时保持奖励质量，是阻碍通用 RM 向大规模实用化的关键瓶颈。

### 关键概念速览

**奖励建模（Reward Modeling，RM）**：用一个模型来预测人类对生成文本的满意度或价值，相当于给语言模型的输出打分的“裁判”。  

**点对点生成奖励建模（Generative Reward Modeling，GRM）**：把奖励本身当成生成任务，让模型直接输出一段文字形式的奖励（如解释、评分），而不是固定的数值标签。可以把它想象成让模型写出“为什么给 8 分”。  

**自我原则性批评调优（Self‑Principled Critique Tuning，SPCT）**：一种在线 RL 训练方式，模型在生成答案的同时生成评判原则和自我批评，循环更新奖励生成行为。类似于人写作文后先列出评分标准，再自我检查并改进。  

**并行采样（Parallel Sampling）**：在一次推理请求中同时启动多条生成路径，利用多卡或多核算力获取更多候选答案，以后再做筛选。相当于让模型“多脑思考”。  

**元奖励模型（Meta‑RM）**：在投票或筛选阶段使用的第二层奖励模型，它负责评估不同候选奖励的质量并决定最终采纳哪一个。可以类比为评审委员会对多位评委的打分进行二次审查。  

**DeepSeek‑GRM**：本文实现的具体 GRM 系列模型，经过 SPCT 调优并配合并行采样与元 RM，能够在推理时显著提升奖励质量。  

### 核心创新点

1. **从点对点数值奖励到生成式奖励的转变**  
   过去的 RM 多采用固定标签（如 0‑1）进行监督，导致对复杂查询的细粒度反馈受限。本文把奖励建模改成生成式任务，让模型输出完整的评价文本。这样既能处理多模态输入，又为后续的推理时扩展提供了更丰富的信号。

2. **自我原则性批评调优（SPCT）**  
   传统的 RL 只在外部奖励上做梯度更新，容易出现奖励漂移或偏见。SPCT 让模型在每一步生成答案后，先自行生成一套评判原则，再依据这些原则对自己的答案进行批评，形成闭环学习。结果是模型学会自行发现并纠正错误，奖励生成行为更具可扩展性。

3. **并行采样 + 元奖励模型的双层扩展框架**  
   为了在推理阶段真正利用更多算力，作者在一次请求中并行生成多组答案‑奖励对，然后用元 RM 对这些对进行投票筛选。并行采样提供了算力的“宽度”，元 RM 提供了质量的“深度”，两者结合实现了推理时的计算‑性能正相关。

4. **DeepSeek‑GRM 的系统化实现与开源**  
   将上述三点整合进一个完整的流水线，形成了 DeepSeek‑GRM 系列模型。实验表明，在多个通用 RM 基准上，它的表现超过了仅靠训练时放大模型规模的做法，证明了“推理时扩展”是一条可行的提升路径。

### 方法详解

整体思路可以划分为四个阶段：  
1）**GRM 基础模型预训练** → 2）**SPCT 在线调优** → 3）**推理时并行采样** → 4）**元 RM 投票筛选**。下面逐步拆解。

#### 1. GRM 基础模型预训练  
作者先在大规模公开数据上训练一个生成式模型，使其能够接受任意查询（文本、代码、图像描述等）并输出一段自然语言奖励。训练目标是最大化生成奖励文本与人工标注奖励的相似度，使用常见的序列到序列（seq2seq）框架。因为奖励本身是文本，这一步与普通的语言模型训练几乎没有区别，只是把“答案”换成了“评价”。

#### 2. Self‑Principled Critique Tuning（SPCT）  
SPCT 的核心是让模型在每一次交互中产生三段输出：  
- **答案 A**：对输入的直接生成。  
- **原则 P**：模型自行列出评判该答案的准则，例如“事实准确性、逻辑连贯性”。  
- **批评 C**：模型依据 P 对 A 进行自我批评，指出缺点并给出改进建议。  

随后，使用一个轻量的 RL 代理（如 PPO）把 **C** 作为即时奖励，更新模型的生成策略。因为 P 是模型自己产生的，整个过程不依赖外部标注，形成了自我监督的闭环。直观上，这相当于让模型在写完作文后先写出评分标准，再给自己的作文打分并改写，循环迭代提升。

#### 3. 推理时并行采样  
在实际使用时，给定一个查询，DeepSeek‑GRM 会启动 **N** 条并行生成流（N 通常为 4‑8），每条流独立产生 (A, P, C) 三元组。并行采样的好处是：  
- 利用多卡或多核的算力，显著增加候选答案的多样性。  
- 每条流的 P 与 C 可能不同，提供了多视角的评价信息。

#### 4. 元奖励模型（Meta‑RM）投票  
所有并行流的输出会送入一个专门训练的元 RM。元 RM 的输入是 **(A, P, C)**，输出一个综合质量分数。它的训练方式是让模型学习在已有的人工标注数据上区分高质量和低质量的三元组。最终，系统根据元 RM 的分数进行 **加权投票**，选出最可信的答案‑奖励对作为最终输出。

#### 巧妙之处  
- **自生成原则**：传统 RL 需要外部奖励函数，这里把奖励函数的“规则”交给模型自己生成，极大降低了人工标注成本。  
- **双层生成**：先生成答案，再生成评价，再用评价指导答案的改进，形成了类似人类写作-自评的循环。  
- **算力利用率**：并行采样把推理时间的算力从单点提升到多点，元 RM 再把多点的噪声过滤掉，实现了“算力越多，效果越好”的正向关系。

### 实验与效果

- **测试数据**：论文在公开的通用奖励建模基准（如 OpenAI Reward Bench、Anthropic RLHF Eval）以及几项实际业务场景（对话安全、代码生成评估）上评估。  
- **对比基线**：包括传统数值奖励模型（RM‑SFT）、基于人类偏好微调的 RLHF、以及仅靠模型规模放大的直接推理（Scale‑Only）。  
- **主要结果**：在大多数任务上，DeepSeek‑GRM 的综合得分比数值 RM 提升约 **12%‑18%**，在对话安全任务上错误率下降约 **30%**。并行采样的算力倍率从 1× 到 8× 时，性能提升呈近线性增长，验证了推理时扩展的有效性。  
- **消融实验**：去掉 SPCT、去掉并行采样、或去掉元 RM，分别导致整体得分下降 **5%‑9%**，说明每个模块都对最终性能有显著贡献。  
- **局限性**：作者指出在极端长文本或需要精确数值判断的任务上，生成式奖励仍会出现模糊或不一致的情况；此外，SPCT 的在线 RL 训练对算力要求较高，部署成本仍需进一步优化。

### 影响与延伸思考

这篇工作首次系统展示了“推理时算力扩展”可以弥补奖励模型训练规模的不足，激发了后续研究在两条方向上的探索：  
1. **生成式奖励的多模态扩展**：把图像、音频等非文本输入也纳入 GRM，形成跨模态的通用评价体系。  
2. **自监督评价循环的深化**：类似 SPCT 的自我批评机制被用于大模型的自我纠错、持续学习等场景，已有几篇工作（如 “Self‑Critique LLM”）直接引用了该思路。  

如果想进一步了解，可以关注近期在 “Meta‑RL for LLM” 以及 “Self‑Refine” 系列的论文，它们在原理上与 SPCT 有不少交叉。

### 一句话记住它

让大语言模型在推理时“多脑思考、自己打分”，用生成式奖励和自我批评把算力直接转化为更可靠的奖励信号。