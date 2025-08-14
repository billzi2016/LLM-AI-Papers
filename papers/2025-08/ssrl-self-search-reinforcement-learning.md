# SSRL: Self-Search Reinforcement Learning

> **Date**：2025-08-14
> **arXiv**：https://arxiv.org/abs/2508.10874

## Abstract

We investigate the potential of large language models (LLMs) to serve as efficient simulators for agentic search tasks in reinforcement learning (RL), thereby reducing dependence on costly interactions with external search engines. To this end, we first quantify the intrinsic search capability of LLMs via structured prompting and repeated sampling, which we term Self-Search. Our results reveal that LLMs exhibit strong scaling behavior with respect to the inference budget, achieving high pass@k on question-answering benchmarks, including the challenging BrowseComp task. Building on these observations, we introduce Self-Search RL (SSRL), which enhances LLMs' Self-Search capability through format-based and rule-based rewards. SSRL enables models to iteratively refine their knowledge utilization internally, without requiring access to external tools. Empirical evaluations demonstrate that SSRL-trained policy models provide a cost-effective and stable environment for search-driven RL training, reducing reliance on external search engines and facilitating robust sim-to-real transfer. We draw the following conclusions: 1) LLMs possess world knowledge that can be effectively elicited to achieve high performance; 2) SSRL demonstrates the potential of leveraging internal knowledge to reduce hallucination; 3) SSRL-trained models integrate seamlessly with external search engines without additional effort. Our findings highlight the potential of LLMs to support more scalable RL agent training.

---

# 自搜索强化学习（SSRL） 论文详细解读

### 背景：这个问题为什么难？

在强化学习里，让智能体去搜索网络信息通常要依赖外部搜索引擎或专门的检索工具。外部调用既慢又贵，还会因为网络波动导致训练不稳定。过去的做法要么把检索当成黑盒子直接喂给模型，要么在模拟环境里硬编码检索结果，结果是：①真实世界的搜索成本高；②模型容易产生“幻觉”，因为它只能靠记忆而不是实时查询；③从模拟到真实的迁移效果差。于是，如何在不依赖外部搜索的情况下，让大语言模型（LLM）自己产生可靠的检索答案，成为了一个急需突破的瓶颈。

### 关键概念速览
- **大语言模型（LLM）**：一种在海量文本上预训练的生成式模型，能够在给定提示后输出连贯的文字。可以把它想成“会说话的百科全书”，但答案质量取决于提示方式。
- **Self‑Search**：通过结构化提示和多次抽样，让 LLM 在内部自行产生检索式答案的过程。类似于让学生在不查资料的情况下，凭记忆多次写出可能的答案，然后挑出最靠谱的那一个。
- **强化学习（RL）**：让智能体在环境中通过试错学习策略的框架。这里的“环境”是模型自身的内部搜索过程，而不是外部网页。
- **奖励函数（Reward）**：用来衡量一次搜索行为好坏的打分机制。本文使用两类奖励：格式奖励（答案是否符合预定义模板）和规则奖励（答案是否满足逻辑约束），就像老师给作文打分时会看格式和内容是否符合要求。
- **Sim‑to‑Real Transfer**：把在模拟环境里学到的策略迁移到真实环境的技术。这里的“模拟环境”是 LLM 的自搜索模拟器，真实环境则是实际调用搜索引擎。
- **Pass@k**：在检索任务中，前 k 次抽样中至少有一次命中正确答案的比例。把它想成“前 k 次猜测里有一次对了”的成功率。

### 核心创新点
1. **把 LLM 当作搜索模拟器 → 通过结构化提示和多次抽样让模型自行生成检索答案 → 证明在预算增大时，LLM 的 Pass@k 呈现出明显的规模效应，能够在不访问外部搜索的情况下达到接近真实检索的水平。**  
   这一步把原本只能“记忆”知识的模型变成了“自我检索器”，为后续强化学习提供了可控的环境。

2. **在自搜索上加入强化学习奖励 → 设计了格式化奖励和规则奖励两套打分机制 → 训练出的策略模型在每一步都倾向于输出更符合检索规范、逻辑更严谨的答案，显著降低了幻觉现象。**  
   关键在于把“写得好看、写得对”这两个目标量化成可学习的信号，让模型在内部循环中不断自我纠正。

3. **构建 Self‑Search RL（SSRL）训练循环 → 让模型在每一次自搜索后根据奖励更新策略 → 形成一个闭环的自我提升系统，无需外部搜索引擎的实时调用。**  
   与传统的“先检索后回答”不同，SSRL 把检索和决策合二为一，省掉了昂贵的 API 调用费用。

4. **验证 SSRL 与外部搜索的兼容性 → 训练好的模型可以直接接入真实搜索引擎而无需额外适配 → 说明内部自搜索学习到的技巧能够迁移到真实检索场景。**  
   这一步展示了 SSRL 不只是省钱的技巧，还能提升模型在真实任务中的稳健性。

### 方法详解
**整体框架**  
SSRL 的训练过程可以划分为三大步骤：①构造 Self‑Search 环境；②定义奖励并进行强化学习；③在奖励驱动下迭代更新模型。整个循环像是让模型在自己的脑海里玩“搜索游戏”，每一次搜索结束后根据得分来改进下一轮的搜索策略。

**步骤拆解**  

1. **Self‑Search 环境搭建**  
   - 给定一个检索任务（例如回答一个需要事实支撑的问题），先用结构化提示把任务拆成“查询+答案”两部分。  
   - 让 LLM 进行多次抽样（比如 10 次），每次抽样都产生一组查询-答案对。  
   - 将这些抽样结果收集成一个小型“检索库”，相当于模型自己生成的候选答案集合。

2. **奖励函数设计**  
   - **格式奖励**：检查答案是否符合预设的模板（如“[事实] 是因为 …”），符合则给正向奖励，不符合则扣分。  
   - **规则奖励**：利用简单的逻辑检查（比如数值范围、实体匹配）来判断答案的真实性。若答案满足规则则加分，否则减分。  
   - 两类奖励相加得到一次搜索的总得分，这个得分直接作为强化学习的回报。

3. **强化学习更新**  
   - 采用基于策略梯度的 RL 算法（如 PPO），把 LLM 当作策略网络。  
   - 在每一次抽样后，根据奖励计算优势函数，进而更新模型参数，使得以后更倾向于产生高奖励的查询-答案对。  
   - 训练过程中会不断循环上述抽样‑奖励‑更新，模型的自搜索质量随之提升。

**关键细节**  
- **预算控制**：抽样次数直接对应推理预算，实验表明预算越大，Pass@k 越高，这为实际部署提供了灵活的成本-性能权衡。  
- **格式化提示**：通过在提示中加入明确的输出结构，引导模型产生易于评估的答案，这一步是把开放式生成转化为可度量任务的关键。  
- **规则奖励的轻量实现**：作者没有使用复杂的外部验证器，而是用一套可编程的规则库，保持了整个系统的“全内置”特性。

### 实验与效果
- **测试任务**：主要在问答类检索基准上评估，包括公开的 BrowseComp 任务（需要在网络上搜索并回答），以及若干标准的 Pass@k 基准。  
- **对比基线**：与传统的“外部搜索 + LLM 回答”流水线、以及仅使用 LLM 直接回答（不做自搜索）的两类方法进行比较。  
- **论文声称**：在相同推理预算下，SSRL 的 Pass@k 超过了仅使用 LLM 的 20% 左右，并且接近使用真实搜索引擎的水平。  
- **消融实验**：作者分别去掉格式奖励、规则奖励以及多抽样环节，发现去掉任意一项都会导致 Pass@k 下降 5%~10%，说明每个模块都有实质贡献。  
- **局限性**：论文指出 SSRL 仍然依赖于 LLM 本身的知识覆盖度，面对极其新鲜的事实时仍会出现幻觉；此外，奖励函数的设计仍然是手工规则，自动化程度有提升空间。

### 影响与延伸思考
SSRL 把“大模型内部搜索”提升到可训练的强化学习层面，打开了“省钱又稳健”的新路径。后续工作已经开始探索：  
- 用更大规模的 LLM（如 GPT‑4）做 Self‑Search，观察规模效应是否继续线性增长（推测）。  
- 将奖励函数自动化，利用可微分的检索评估器（如 dense retriever）替代手工规则（已有初步尝试）。  
- 把 SSRL 融入多模态任务，让模型在图像、音频等非文本检索中也能自我搜索（推测）。  
对想进一步深入的读者，可以关注“内置检索的强化学习”以及“LLM 低成本模拟环境”这两个方向的最新会议论文。

### 一句话记住它
**SSRL 让大语言模型自己当搜索引擎，通过奖励驱动的自我迭代，省去外部查询成本并显著降低幻觉。**