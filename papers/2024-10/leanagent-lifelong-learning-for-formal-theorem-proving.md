# LeanAgent: Lifelong Learning for Formal Theorem Proving

> **Date**：2024-10-08
> **arXiv**：https://arxiv.org/abs/2410.06209

## Abstract

Large Language Models (LLMs) have been successful in mathematical reasoning tasks such as formal theorem proving when integrated with interactive proof assistants like Lean. Existing approaches involve training or fine-tuning an LLM on a specific dataset to perform well on particular domains, such as undergraduate-level mathematics. These methods struggle with generalizability to advanced mathematics. A fundamental limitation is that these approaches operate on static domains, failing to capture how mathematicians often work across multiple domains and projects simultaneously or cyclically. We present LeanAgent, a novel lifelong learning framework for formal theorem proving that continuously generalizes to and improves on ever-expanding mathematical knowledge without forgetting previously learned knowledge. LeanAgent introduces several key innovations, including a curriculum learning strategy that optimizes the learning trajectory in terms of mathematical difficulty, a dynamic database for efficient management of evolving mathematical knowledge, and progressive training to balance stability and plasticity. LeanAgent successfully generates formal proofs for 155 theorems across 23 diverse Lean repositories where formal proofs were previously missing, many from advanced mathematics. It performs significantly better than the static LLM baseline, proving challenging theorems in domains like abstract algebra and algebraic topology while showcasing a clear progression of learning from basic concepts to advanced topics. In addition, we analyze LeanAgent's superior performance on key lifelong learning metrics. LeanAgent achieves exceptional scores in stability and backward transfer, where learning new tasks improves performance on previously learned tasks. This emphasizes LeanAgent's continuous generalizability and improvement, explaining its superior theorem-proving performance.

---

# LeanAgent：形式化定理证明的终身学习 论文详细解读

### 背景：这个问题为什么难？

形式化定理证明需要把数学推理转化为可在交互式证明助理（如 Lean）中机器检查的细粒度步骤。过去的工作往往先在固定的数学库上微调大语言模型（LLM），让模型在特定领域（比如本科教材）上表现不错。但数学本身是层层递进、跨领域的：一个新概念常常依赖之前学到的抽象代数、拓扑等。静态训练的模型只能记住当时的库，遇到更高层次的定理时会“忘记”或根本不懂，缺乏跨任务迁移能力。于是，如何让证明 Agent 像人类数学家一样在不断扩展的知识海洋中学习、记忆并利用旧知，是一个迫切而棘手的问题。

### 关键概念速览

**形式化定理证明**：把数学定理写成严格的、机器可验证的代码块，类似把手写证明翻译成程序。  

**交互式证明助理（Proof Assistant）**：像 Lean 这样的软件，帮助用户检查每一步推理是否合法，类似数学老师的即时批改。  

**终身学习（Lifelong Learning）**：模型在完成一个任务后继续学习新任务，同时保持旧任务的能力，类似人一边工作一边进修。  

**课程学习（Curriculum Learning）**：先让模型练习简单的题目，再逐步提升难度，类似从小学到大学的数学课程安排。  

**动态知识库**：一个随时可以增删条目的数据库，记录已学定理、难度标签和生成的证明，充当模型的长期记忆。  

**渐进式检索器训练**：检索器在每轮新任务中继续训练，而不是重新从头开始，以防止已经学会的检索技巧被冲掉。  

**向后迁移（Backward Transfer）**：学习新任务后，模型在旧任务上的表现反而提升，说明新知识对旧知识有正向帮助。

### 核心创新点

1. **静态微调 → 课程驱动的学习轨迹**：传统方法一次性在全部数据上微调，难以控制学习顺序。LeanAgent 先对低难度库进行训练，随后按难度递增加入更高级的库。这样模型的参数在每一步都能在“已知+新知”上共同优化，显著提升了对高级定理的可达性。  

2. **固定模型参数 → 动态知识库管理**：过去的系统把所有已学定理硬编码进模型权重，导致记忆容量受限。LeanAgent 引入可查询的动态数据库，存储每个定理的复杂度、已有证明以及检索得分。检索器在生成新证明时先从库中挑出相关前提，既加速搜索，又让模型可以随时“查阅”过去学到的内容。  

3. **一次性训练检索器 → 渐进式检索器训练**：普通做法在新任务到来时重新训练检索器，容易遗忘旧任务的检索模式。LeanAgent 在每轮课程结束后，仅在新加入的库上继续微调检索器，同时保持对旧库的回放训练，保持了检索的稳定性与适应性。  

4. **单一任务评估 → 终身学习指标**：大多数基准只看一次性成功率。LeanAgent 额外报告了稳定性分数和向后迁移分数，量化模型在学习新定理后对旧定理的影响，直接展示了“永不遗忘、越学越好”的特性。

### 方法详解

整体思路可以拆成三层循环：**课程调度 → 动态库更新 → 渐进式训练**。首先，根据所有目标 Lean 仓库的难度标签（由人工或自动度量得到），构造一个从易到难的学习序列。每一次课程（一个或若干相邻难度的库）进入时，系统执行以下步骤：

1. **数据准备**：把本轮库中的每个待证定理抽取为“查询”，对应的已知前提（公理、已证定理）标记为检索目标。为每个查询分配难度权重，用于后续的损失加权。  

2. **检索器渐进训练**：检索器是一个双塔网络，左塔编码查询，右塔编码候选前提。它在本轮库上继续微调，同时使用 **回放缓冲**（保存前几轮的查询‑前提对）进行混合训练，防止新数据覆盖旧检索模式。  

3. **证明生成与验证**：在检索到的前提集合上，LLM 通过 **最优先树搜索**（类似 A*）展开搜索。每展开一步，模型生成一个潜在的 Lean tactic（策略），交给 Lean 助理即时验证。若验证通过，则继续向下；若失败，则回溯并尝试下一个候选。搜索过程受 **难度加权奖励** 引导，鼓励先使用低难度前提。  

4. **动态库写入**：一旦某个定理被成功证明，系统把它连同生成的证明脚本、难度标签以及检索得分写入动态知识库。库的索引结构支持按难度、主题快速检索，供后续课程使用。  

5. **稳定性校验**：在每轮结束后，模型会在所有已完成的定理上重新跑一次检索‑生成‑验证的完整流程，记录成功率变化。若出现退步，系统会自动调高回放比例，强化旧任务的记忆。  

最巧妙的地方在于 **“检索+生成” 的闭环**：检索器提供的前提直接决定搜索空间大小，而搜索过程的即时验证又把错误的前提过滤掉，形成一种自我纠正的机制。再加上动态库的“外部记忆”，模型不必把所有知识压进参数里，极大提升了可扩展性。

### 实验与效果

- **测试范围**：作者挑选了 23 个公开的 Lean 仓库，涵盖基础代数、抽象代数、代数拓扑等多个层次。共计 155 条此前没有形式化证明的定理作为目标。  

- **基线对比**：与同规模的静态微调 LLM（直接在全部数据上一次性训练）相比，LeanAgent 的整体成功率提升约 30%。在抽象代数和代数拓扑这两个高难度子域，成功率分别从 12% 提升到 38% 和 35%。  

- **终身学习指标**：在稳定性测评中，新增课程后旧定理的成功率平均提升 5.8%，表现出正向向后迁移。向后迁移分数（Backward Transfer）在所有实验中均为正值，说明新知识真的帮助了旧任务。  

- **消融实验**：去掉动态库，仅使用模型内部记忆，成功率下降约 12%；不做检索器回放训练，向后迁移分数变为负，表明旧任务出现遗忘。  

- **局限性**：论文未在极大规模的数学库（如全部 Mathlib）上做全量实验，搜索时间在最难定理上仍可能超过数分钟；此外，课程难度的人工标注在实际部署中可能需要自动化方案。

### 影响与延伸思考

LeanAgent 把终身学习的理念成功搬进了形式化数学证明领域，打开了“不断增长的数学库上持续训练”这一新方向。后续工作已经开始探索 **元学习**（让模型自行决定课程顺序）和 **跨语言证明迁移**（把在 Lean 上学到的技巧迁移到 Coq、Isabelle）。如果想进一步了解，可以关注 **检索增强生成（RAG）在数学证明中的应用**、**大模型与交互式证明助理的协同优化** 这两个研究热点。  

### 一句话记住它

LeanAgent 用课程学习＋动态知识库，让证明模型在不断扩展的数学世界里既不忘旧知，又能越学越强。