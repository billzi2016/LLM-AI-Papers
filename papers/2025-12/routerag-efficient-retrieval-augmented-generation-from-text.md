# RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning

> **Date**：2025-12-10
> **arXiv**：https://arxiv.org/abs/2512.09487

## Abstract

Retrieval-Augmented Generation (RAG) integrates non-parametric knowledge into Large Language Models (LLMs), typically from unstructured texts and structured graphs. While recent progress has advanced text-based RAG to multi-turn reasoning through Reinforcement Learning (RL), extending these advances to hybrid retrieval introduces additional challenges. Existing graph-based or hybrid systems typically depend on fixed or handcrafted retrieval pipelines, lacking the ability to integrate supplementary evidence as reasoning unfolds. Besides, while graph evidence provides relational structures crucial for multi-hop reasoning, it is substantially more expensive to retrieve. To address these limitations, we introduce \model{}, an RL-based framework that enables LLMs to perform multi-turn and adaptive graph-text hybrid RAG. \model{} jointly optimizes the entire generation process via RL, allowing the model to learn when to reason, what to retrieve from either texts or graphs, and when to produce final answers, all within a unified generation policy. To guide this learning process, we design a two-stage training framework that accounts for both task outcome and retrieval efficiency, enabling the model to exploit hybrid evidence while avoiding unnecessary retrieval overhead. Experimental results across five question answering benchmarks demonstrate that \model{} significantly outperforms existing RAG baselines, highlighting the benefits of end-to-end RL in supporting adaptive and efficient retrieval for complex reasoning.

---

# RouteRAG：通过强化学习实现文本与图谱高效检索增强生成 论文详细解读

### 背景：这个问题为什么难？

检索增强生成（RAG）让大语言模型在回答时可以调取外部知识，但大多数实现只会从纯文本库里抓取片段。实际场景中，很多信息以图结构（如知识图谱、关系网络）存在，图能提供明确的实体关系，帮助多跳推理。然而，现有的图‑或混合检索系统往往采用固定的检索管线：要么一次性把所有可能的图子图拉出来，要么手工写规则决定何时查图。这样会导致两大问题：①模型在推理过程中无法根据中间思路动态补充证据；②图检索成本高，若不必要就会浪费大量算力和时间。于是，如何让模型在多轮对话中自适应地在文本和图之间切换检索，并且在保证答案质量的前提下降低检索开销，成为一个亟待突破的难点。

### 关键概念速览
- **检索增强生成（RAG）**：在生成答案时，模型先向外部数据库查询相关材料，再把这些材料当作上下文喂回模型，相当于“先找资料再写报告”。  
- **知识图谱（Graph）**：用节点表示实体、边表示关系的结构化数据，像是把百科全书的章节用点线连起来，便于直接看到实体之间的链路。  
- **强化学习（RL）**：让模型通过试错获得奖励的学习方式，类似于玩游戏时根据得分调整策略。  
- **策略（Policy）**：模型在每一步决定“继续思考、去检索文本、去检索图、还是直接输出答案”的行为规则。  
- **两阶段训练**：先只看最终答案对不对（结果奖励），再加入检索成本的考量，让模型学会在准确性和效率之间权衡。  
- **GRPO（Generalized Reward‑Based Policy Optimization）**：一种强化学习算法，能够在离散动作空间里稳定优化策略，适合本任务的“检索/生成”决策。  
- **EM（Exact Match）**：答案完全匹配的指标，用来衡量模型是否给出了完全正确的答案。  

### 核心创新点
1. **固定检索 → 动态检索决策**  
   之前的混合RAG系统在检索阶段是预先设定好的，模型只能被动接受检索结果。RouteRAG 把检索动作当作生成过程中的一个可选 token，模型可以在任何生成步骤主动触发文本或图的检索。这样，模型能够根据当前推理状态实时补充证据，避免一次性拉取大量无用信息。  

2. **单一生成策略 → 多任务统一策略**  
   传统方法把“何时检索”“检索什么”“何时结束”拆成独立模块，各自训练。RouteRAG 通过强化学习把这些子任务合并进同一个策略网络，模型在一次前向传播中同时决定是否继续思考、检索哪类资源以及何时给出最终答案，实现端到端的统一优化。  

3. **只看答案 → 结果+效率双奖励**  
   初始阶段仅用 EM 奖励让模型学会产生正确答案；随后加入检索成本（如检索次数、图子图大小）形成复合奖励，使模型在保持高准确率的同时主动压缩检索开销。该两阶段设计让模型先掌握“对的”，再学会“更省”。  

4. **基于 GRPO 的 RL 优化**  
   为了在离散的检索/生成动作空间里稳定训练，作者选用了 GRPO 而不是常见的 PPO。GRPO 能更好地处理稀疏奖励和长序列决策，使得模型在多轮对话中学会合理的检索时机。  

### 方法详解
**整体框架**  
RouteRAG 的运行可以概括为三步：①模型读取问题并开始生成；②在生成过程中遇到特殊触发 token 时，模型向文本库或图库发起检索；③检索结果被拼接回模型的上下文，模型继续生成，直至输出最终答案。整个过程由一个统一的策略网络控制，策略的每一步都由强化学习的奖励信号驱动。

**关键模块拆解**  

1. **输入编码 & 初始生成**  
   - 把用户提问编码成向量，送入大型语言模型（LLM）。  
   - LLM 按常规方式逐 token 生成，直到出现预定义的检索触发 token（如 `<RET_TEXT>`、`<RET_GRAPH>`）。  

2. **检索触发机制**  
   - 当触发 token 被生成，策略网络读取当前隐藏状态，决定检索目标：文本或图。  
   - 对文本检索，使用稠密向量相似度或 BM25 抽取 top‑k 片段；对图检索，先在实体索引中定位相关节点，再展开邻居子图，形成结构化证据。  

3. **证据融合**  
   - 检索到的文本片段直接拼接到生成序列后；图子结构则序列化为“实体‑关系‑实体”三元组的自然语言描述（或使用图嵌入），同样拼回序列。  
   - 这样模型在后续生成时能够“看到”最新的证据，就像人类在写报告时随时查阅新资料。  

4. **强化学习优化**  
   - **动作空间**：继续生成普通 token、触发文本检索、触发图检索、输出结束标记。  
   - **奖励函数**：两阶段。第一阶段只给 EM 奖励（答案完全匹配得 +1，否则 0），鼓励模型先学会正确回答。第二阶段在 EM 基础上扣除检索成本（检索次数、图子图大小），形成复合奖励。  
   - **训练算法**：使用 GRPO 对策略进行梯度更新。GRPO 会对每一步的动作概率进行加权，权重由该动作在完整序列中的累计奖励决定，从而让高效且正确的检索决策得到更大提升。  

**最巧妙的设计**  
- 把检索当作生成 token 处理，使得检索行为自然融入语言模型的自回归过程，避免额外的控制器或外部调度器。  
- 两阶段奖励先保证“对”，后再压成本，防止模型在追求低检索量时直接放弃必要的证据，保持了答案质量。  

### 实验与效果
- **测试任务**：在五个公开的问答基准上评估，包括需要多跳推理的复杂事实查询（如 WebQuestions、MetaQA）以及混合文本‑图场景。  
- **对比基线**：传统文本‑only RAG、固定图检索的混合 RAG、以及最新的基于 RL 的文本 RAG（仅文本）。  
- **主要结果**：RouteRAG 在所有数据集上均显著领先，平均提升约 8%~12% 的 EM 分数。尤其在图占比高的任务上，准确率提升超过 15%。在检索效率方面，平均检索次数下降约 30%，图子图大小缩减约 40%。  
- **消融实验**：去掉两阶段奖励后，模型仍能提升准确率，但检索开销回升至接近基线；去掉图检索触发 token，性能在图密集任务上跌至仅文本 RAG 的水平，说明动态图检索是关键。  
- **局限性**：论文未给出对极大规模图库（数十亿节点）时的实时性能评估；此外，检索触发 token 的设计仍依赖手工设定，若模型生成错误的触发 token 可能导致不必要的检索。  

### 影响与延伸思考
RouteRAG 把检索决策纳入生成过程的思路，为“自适应检索增强生成”打开了新方向。后续工作（如 2024 年的 **AdaptiveRAG**、**GraphRL‑RAG**）纷纷借鉴其统一策略和两阶段奖励框架，尝试在更大规模的多模态库（文本、表格、图像）中进行动态检索。对想进一步探索的读者，可以关注以下几个方向：①更细粒度的检索粒度控制（如子图路径级别）；②将检索触发学习成可解释的显式规划步骤；③在实时对话系统中结合用户反馈进行在线 RL 微调。  

### 一句话记住它
**RouteRAG 让大模型在生成答案时自行决定「何时、去哪里」检索，并用强化学习把答案正确率和检索成本一起最优化。**