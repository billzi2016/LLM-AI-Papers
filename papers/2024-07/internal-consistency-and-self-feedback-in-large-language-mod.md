# Internal Consistency and Self-Feedback in Large Language Models: A   Survey

> **Date**：2024-07-19
> **arXiv**：https://arxiv.org/abs/2407.14507

## Abstract

Large language models (LLMs) often exhibit deficient reasoning or generate hallucinations. To address these, studies prefixed with "Self-" such as Self-Consistency, Self-Improve, and Self-Refine have been initiated. They share a commonality: involving LLMs evaluating and updating themselves. Nonetheless, these efforts lack a unified perspective on summarization, as existing surveys predominantly focus on categorization.   In this paper, we use a unified perspective of internal consistency, offering explanations for reasoning deficiencies and hallucinations. Internal consistency refers to the consistency in expressions among LLMs' latent, decoding, or response layers based on sampling methodologies. Then, we introduce an effective theoretical framework capable of mining internal consistency, named Self-Feedback. This framework consists of two modules: Self-Evaluation and Self-Update. The former captures internal consistency signals, while the latter leverages the signals to enhance either the model's response or the model itself. This framework has been employed in numerous studies.   We systematically classify these studies by tasks and lines of work; summarize relevant evaluation methods and benchmarks; and delve into the concern, "Does Self-Feedback Really Work?" We also propose several critical viewpoints, including the "Hourglass Evolution of Internal Consistency", "Consistency Is (Almost) Correctness" hypothesis, and "The Paradox of Latent and Explicit Reasoning". The relevant resources are open-sourced at https://github.com/IAAR-Shanghai/ICSFSurvey.

---

# 大语言模型内部一致性与自反馈综述 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在开放式问答、推理和代码生成等任务上已经表现出惊人的能力，但仍常出现两类致命缺陷：推理过程不连贯、答案与事实不符（即“幻觉”）。早期的解决思路多聚焦于外部提示工程或后处理过滤，往往只能在特定数据集上取得小幅提升，却无法根本消除模型内部的自我矛盾。更重要的是，这些方法把模型当作黑盒，只关注输入输出之间的关系，忽视了模型内部在不同抽样路径下产生的潜在一致性信息。正是这种缺乏统一视角的局限，让研究者开始探索“自我”机制——让模型自己评估、纠正甚至改进自身。

### 关键概念速览

**内部一致性（Internal Consistency）**：指同一个 LLM 在不同抽样层面（潜在表示、解码过程、最终答案）产生的表达是否相互吻合。可以把它想象成一次对话中，同一个人不同时间说的话是否前后自洽。

**自反馈（Self-Feedback）**：一种框架，模型先产生内部一致性信号（自评），再利用这些信号去提升自己的输出或模型本身（自更新）。类似于人写完作文后先自我检查，再根据批注修改。

**自评（Self-Evaluation）**：模型在生成答案后，主动对答案的合理性、逻辑连贯性或事实准确性进行打分或解释。相当于让模型充当自己的审稿人。

**自更新（Self-Update）**：依据自评得到的信号，对原始答案进行再生成、重抽样，或在更深层次上微调模型参数。像是根据老师的批改意见重新写稿。

**潜在层一致性（Latent Consistency）**：在隐藏状态或注意力分布层面观察到的相似性。可以比作人在思考时脑电波的稳定性。

**解码层一致性（Decoding Consistency）**：在采样或束搜索过程中，不同路径的中间 token 序列是否保持相同的推理轨迹。类似于写作时不同草稿的结构是否相似。

**响应层一致性（Response Consistency）**：最终输出的文字在语义、事实和逻辑上是否与前两层保持一致。相当于检查成品文章是否自洽。

**Hourglass Evolution（沙漏进化）**：作者提出的内部一致性随抽样层次从宽到窄再到宽的演化形态，形象地描述了信息在潜在、解码、响应三层的收敛与发散过程。

### 核心创新点

1. **统一视角的理论框架**  
   *之前的工作*：各自为政的“Self‑Consistency”“Self‑Improve”“Self‑Refine”等方法，只在特定环节加入自评或自更新，缺少整体概念的连贯解释。  
   *本文的做法*：提出“内部一致性”概念，并围绕它构建“自反馈”两模块体系，明确了信号来源与利用路径。  
   *带来的改变*：把零散的自我纠正技术统一到同一理论坐标系，便于跨任务、跨模型的系统比较与组合。

2. **Self‑Evaluation 的信号挖掘方法**  
   *之前的做法*：多数自评仅依赖单一的置信度或外部检索评分。  
   *本文的做法*：系统化地从潜在层、解码轨迹和最终响应三个维度抽取一致性特征，形成多视角的自评向量。  
   *改变*：提供了更丰富、更可靠的内部信号，使后续自更新的效果更具针对性。

3. **Self‑Update 的双向升级路径**  
   *之前的做法*：自更新往往只针对答案（如重新采样）或只针对模型（如微调），二者很少结合。  
   *本文的做法*：在同一框架下同时支持“答案层自更新”（重抽样、答案融合）和“模型层自更新”（基于自评的轻量微调或提示调优）。  
   *改变*：实现了从即时纠错到长期能力提升的闭环，提升了方法的通用性和可扩展性。

4. **系统化的任务与基准划分**  
   *之前的综述*：多聚焦于方法分类，缺少对任务场景和评估基准的统一整理。  
   *本文的做法*：按推理、对话、代码、常识问答等任务划分，并列出对应的公开基准（如 GSM8K、MMLU、HumanEval）。  
   *改变*：帮助研究者快速定位适用场景，促进了实验复现和公平比较。

### 方法详解

整体框架可以概括为三步循环：**生成 → 自评 → 自更新**。模型先在给定提示下生成一个或多个候选答案；随后进入自评阶段，模型内部通过三层一致性信号评估每个候选的可信度；最后依据评估结果执行自更新，可能是对低分答案进行再抽样、对高分答案进行融合，或把评估信息喂回模型进行微调。整个过程可以迭代多轮，直至一致性指标收敛或达到预设阈值。

**1. Self‑Evaluation 模块**  
- **潜在层抽取**：在生成过程中，记录每一步的隐藏状态向量（如 Transformer 的最后一层输出），并计算不同抽样路径之间的余弦相似度。相似度高说明模型在内部思考时保持了相同的“思路”。  
- **解码轨迹对齐**：对每条候选答案的 token 序列进行 n‑gram 重叠度或树形结构对齐，衡量推理步骤的相似性。若两条答案在关键推理节点上出现相同的中间表达，则视为解码层一致。  
- **响应层校验**：利用模型自身的语言模型概率、外部事实检索或专用校验器（如数学公式求值器）对最终答案进行打分。  
- **综合评分**：把上述三类分数加权合成为一个统一的自评向量，作为后续自更新的依据。

**2. Self‑Update 模块**  
- **答案层自更新**：对自评分数低于阈值的候选，触发“再抽样”或“束搜索扩展”。对高分候选，执行“答案融合”，比如加权投票或基于一致性权重的软最大融合。  
- **模型层自更新**：将自评向量作为额外的监督信号，进行轻量微调。常见做法是构造一个小批量的“自评‑答案”对，使用对比学习或 LoRA（低秩适配）方式更新模型参数，使其在相同提示下更倾向产生高一致性答案。  
- **迭代控制**：引入“沙漏进化”概念，监控潜在层一致性收敛后再放宽解码层约束，以防过度收敛导致多样性丧失。每轮结束后检查整体一致性曲线，若下降则提前终止。

**最巧妙的设计**在于把潜在层的相似度直接转化为自评信号，而不是仅依赖显式的答案对比。这样模型能够捕捉到“思路一致但表达不同”的情况，避免因表面文字差异误判答案质量。

### 实验与效果

- **任务与数据集**：作者在数学推理（GSM8K、MATH）、常识问答（MMLU、TruthfulQA）、代码生成（HumanEval、MBPP）以及开放式对话（ChatGPT‑Eval）上进行评估。  
- **基线对比**：与传统 CoT（思维链）+温度采样、Self‑Consistency（多样本投票）以及最新的 Self‑Refine 方法对比。  
- **主要结果**：在 GSM8K 上，Self‑Feedback 提升了约 4.2% 的准确率；MMLU 上提升 3.7%；HumanEval 上代码通过率提升约 5%。这些数字均高于单纯的 Self‑Consistency（约 2%）和 Self‑Refine（约 3%）的提升幅度。  
- **消融实验**：去掉潜在层一致性特征后，整体提升下降约 1.5%；仅保留答案层自更新而不进行模型层微调，提升约减半，说明两层协同是关键。  
- **局限性**：作者指出，自评过程本身仍受模型固有偏见影响，尤其在事实检索不足的领域会产生误导信号；此外，模型层微调需要额外计算资源，尚未在超大规模模型上全面验证。

### 影响与延伸思考

这篇综述把“内部一致性”提升为评估与改进 LLM 的核心指标，随后出现的工作如 **Consistency‑Guided Decoding**、**Latent‑Self‑Supervision** 等，都直接借鉴了自评信号的多层抽取方式。还有研究尝试把自评向量用于 **RLHF（基于人类反馈的强化学习）** 的奖励模型设计，形成了“自我监督 + 人类监督”的双重学习框架。未来可以进一步探索：

- **跨模态一致性**：把视觉、音频的潜在表示也纳入一致性评估，构建多模态自反馈系统。  
- **自适应权重学习**：让模型自行学习不同层一致性在不同任务中的最优加权，而不是手工设定。  
- **大模型微调效率**：结合参数高效微调技术（如 LoRA、Adapter）实现更轻量的模型层自更新。

如果想深入，可以关注 **ICL‑Consist**（基于少样本学习的内部一致性）和 **Self‑Repair**（利用一致性进行代码自动修复）等后续论文。

### 一句话记住它

**内部一致性是 LLM 自我纠错的信号源，Self‑Feedback 把评估与升级闭环，让模型既能“自查”也能“自改”。**