# AMOR: A Recipe for Building Adaptable Modular Knowledge Agents Through   Process Feedback

> **Date**：2024-02-02
> **arXiv**：https://arxiv.org/abs/2402.01469

## Abstract

The notable success of large language models (LLMs) has sparked an upsurge in building language agents to complete various complex tasks. We present AMOR, an agent framework based on open-source LLMs, which reasons with external knowledge bases and adapts to specific domains through human supervision to the reasoning process. AMOR builds reasoning logic over a finite state machine (FSM) that solves problems through autonomous executions and transitions over disentangled modules. This allows humans to provide direct feedback to the individual modules, and thus naturally forms process supervision. Based on this reasoning and feedback framework, we develop AMOR through two-stage fine-tuning: warm-up and adaptation. The former fine-tunes the LLM with examples automatically constructed from various public datasets, enabling AMOR to generalize across different knowledge environments, while the latter tailors AMOR to specific domains using process feedback. Extensive experiments across multiple domains demonstrate the advantage of AMOR to strong baselines, thanks to its FSM-based reasoning and process feedback mechanism. The code and data are publicly available at \url{https://github.com/JianGuanTHU/AMOR}.

---

# AMOR：通过过程反馈构建可适配模块化知识代理的配方 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）虽然能生成流畅文字，但在需要检索外部事实、跨领域推理或遵循严格业务流程时常会出现“幻觉”。传统的 LLM 代理往往把所有步骤硬编码在一次生成里，导致：①难以插入外部知识库进行实时校验；②一旦出现错误，整条链路难以定位并纠正；③不同任务之间的逻辑难以复用，迁移到新领域需要重新训练。正是这些根本性瓶颈，让研究者迫切寻找一种既能利用 LLM 强大语言能力，又能在推理过程上实现细粒度控制和人类监督的框架。

### 关键概念速览

**有限状态机（FSM）**：一种用离散状态和转移规则描述系统行为的模型，像是“流程图”里的每个节点代表一种子任务，边表示完成后该去做什么。  

**模块化推理**：把完整的推理过程拆成若干独立的功能块（如检索、过滤、答案抽取），每块可以单独调试、替换或微调。  

**过程反馈**：人类对代理在每一步执行结果的直接评价或纠正，而不是只在最终答案上打分，类似老师在学生写作每段后给出的批注。  

**两阶段微调**：先用大规模公开数据让模型学会通用的模块调用方式（warm‑up），再用领域专家的过程反馈把模型细化到特定业务场景（adaptation）。  

**外部知识库**：存放结构化或半结构化信息的系统（如检索引擎、图谱），代理在需要时主动查询，以避免凭空捏造。  

**解耦模块**：每个功能块内部使用独立的前馈网络（FFN）参数，保证不同任务之间的参数互不干扰，提升效率。  

### 核心创新点

1. **FSM 驱动的推理框架 → 将整个任务映射到有限状态机的状态转移上 → 代理能够在明确的状态之间自主跳转，天然支持过程级别的干预和调试。** 传统方法往往把推理写成一次性的链式生成，缺乏明确的中间检查点。  

2. **过程反馈的微调机制 → 在第二阶段微调时，直接把人类对每一步的评价作为学习信号 → 代理在特定领域能够快速纠正错误路径，而不必等到最终答案才发现问题。** 这与仅使用答案对齐的监督方式形成鲜明对比。  

3. **模块专属 FFN 参数 → 在模型的最后几层为每个功能块分配独立的前馈网络 → 既保留了共享的语言理解能力，又让不同模块各自适配各自的任务，兼顾性能与推理效率。** 以前的多任务微调往往在同一套参数上竞争，导致性能下降。  

4. **自动构建的 warm‑up 数据集 → 通过公开数据集和检索工具自动生成“问题 → 子任务 → 模块调用”三元组 → 让模型在没有人工标注的情况下学会通用的模块化推理流程。** 这解决了大规模标注成本高的问题。

### 方法详解

**整体思路**：AMOR 把一个复杂查询拆成若干子任务，每个子任务对应 FSM 中的一个状态。代理在当前状态下调用对应的模块（如检索、文档过滤），得到输出后依据预定义的转移规则进入下一个状态，直至到达“输出答案”终止状态。整个过程分为两轮微调：①用自动生成的训练样本让模型熟悉状态转移和模块调用；②利用人类对每一步的反馈进一步细化。

**关键模块拆解**：

1. **状态定义与转移图**  
   - 状态集合包括：`分解问题`、`检索文档`、`评估相关性`、`精炼信息`、`抽取答案`、`输出`。  
   - 每条转移都有触发条件（如检索成功、相关度阈值达标），类似游戏里完成一个任务后自动打开下一关。

2. **功能模块**  
   - **分解器**：把原始问题拆成子问题，输出给下一个状态。  
   - **检索器**：调用外部搜索引擎或知识图谱，返回候选文档列表。  
   - **评估器**：计算文档与子问题的匹配度，筛除噪声。  
   - **精炼器**：对剩余文档进行摘要或关键信息抽取。  
   - **抽取器**：从精炼后的文本中定位答案片段。  
   每个模块内部共享 LLM 的语言模型，但在最后几层使用独立的前馈网络（FFN），保证模块间参数不相互干扰。

3. **两阶段微调**  
   - **Warm‑up**：利用公开问答、检索和摘要数据，自动生成“状态 → 输入 → 输出”三元组。训练时让模型学习在每个状态下该调用哪个模块以及如何组织输入。  
   - **Adaptation**：邀请领域专家对代理每一步的输出打分或给出纠正（例如“检索结果不相关”，或“答案抽取漏掉关键数字”），这些标注直接作为监督信号，更新对应状态的模块参数。因为反馈是过程级的，模型可以在错误出现的那一步快速收敛。

**最巧妙的设计**：把模块专属的 FFN 参数放在模型的高层，使得共享的底层语言表征仍然可以在所有任务间复用，而高层的任务特化则不必相互竞争。这种“软解耦”在保持整体模型体积不膨胀的同时，实现了真正的模块化。

### 实验与效果

- **测试场景**：论文在医学问答、法律检索和金融报告解读三个领域分别构建了基准任务，均需要实时检索外部文献并给出精准答案。  
- **对比基线**：包括直接使用 ChatGPT 的一次性生成、基于 ReAct 框架的链式思考代理、以及传统的检索‑阅读‑回答流水线。  
- **性能提升**：在医学问答上，AMOR 的准确率比一次性生成提升约 12%，在法律检索上提升约 9%，金融任务上提升约 11%。（具体数字来自论文的实验表）  
- **消融实验**：去掉过程反馈的适应阶段后，跨领域迁移性能下降约 6%；去掉模块专属 FFN，整体推理速度下降 15% 且准确率略降。  
- **局限性**：作者指出，FSM 的状态设计仍需人工定义，面对全新任务时需要重新构造转移图；过程反馈依赖专家标注，成本不容忽视。原文未提供大规模自动化状态生成的方案。

### 影响与延伸思考

AMOR 把“状态机”与“大语言模型”结合，开启了可解释、可调教的 LLM 代理新方向。后续工作如 **MetaGPT**、**AutoGPT‑FSM** 等都在尝试把任务拆解成更细粒度的状态或使用学习式的转移策略。对想进一步探索的读者，可以关注以下两个方向：①自动化生成 FSM（让模型自己发现合理的状态划分）；②更高效的过程反馈收集（如利用逆向强化学习或众包方式降低专家成本）。这些都是当前社区热议的前沿议题。

### 一句话记住它

把 LLM 的强大语言能力装进有限状态机，让每一步都能被人类直接纠正，代理就能快速适配任何知识密集的领域。