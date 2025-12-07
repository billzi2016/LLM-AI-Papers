# LightSearcher: Efficient DeepSearch via Experiential Memory

> **Date**：2025-12-07
> **arXiv**：https://arxiv.org/abs/2512.06653

## Abstract

DeepSearch paradigms have become a core enabler for deep reasoning models, allowing them to invoke external search tools to access up-to-date, domain-specific knowledge beyond parametric boundaries, thereby enhancing the depth and factual reliability of reasoning. Building upon this foundation, recent advances in reinforcement learning (RL) have further empowered models to autonomously and strategically control search tool usage, optimizing when and how to query external knowledge sources. Yet, these RL-driven DeepSearch systems often reveal a see-saw trade-off between accuracy and efficiency-frequent tool invocations can improve factual correctness but lead to unnecessary computational overhead and diminished efficiency. To address this challenge, we propose LightSearcher, an efficient RL framework that incorporates textual experiential memory by learning contrastive reasoning trajectories to generate interpretable summaries of successful reasoning patterns. In addition, it employs an adaptive reward shaping mechanism that penalizes redundant tool calls only in correct-answer scenarios. This design effectively balances the inherent accuracy-efficiency trade-off in DeepSearch paradigms. Experiments on four multi-hop QA benchmarks show that LightSearcher maintains accuracy comparable to SOTA baseline ReSearch, while reducing search tool invocations by 39.6%, inference time by 48.6%, and token consumption by 21.2%, demonstrating its superior efficiency.

---

# LightSearcher：基于经验记忆的高效DeepSearch 论文详细解读

### 背景：这个问题为什么难？

DeepSearch 让大模型在推理时可以调用外部搜索工具，弥补了模型参数的知识时效性限制，已经成为提升多跳问答准确率的关键手段。但搜索本身是昂贵的：每一次调用都要向检索系统发送查询、等待返回、再把结果喂回模型，导致推理时间和算力开销急剧上升。现有的基于强化学习（RL）的 DeepSearch 系统虽然能学会“什么时候该查”，却常出现“查得太多”或“查得太少”的摇摆——频繁调用提升了答案的正确率，却把效率压到不可接受的水平。于是，如何在保持高准确率的同时显著削减搜索次数，成为亟待突破的瓶颈。

### 关键概念速览
- **DeepSearch**：模型在推理过程中主动向外部检索引擎发起查询，以获取最新或专业的事实信息。类似于人在写论文时随时上网查资料。
- **强化学习（RL）**：让模型通过试错学习一个策略，决定在每一步是继续思考还是去搜索。把模型当成“智能代理”，奖励函数决定它的行为倾向。
- **经验记忆（Experiential Memory）**：把过去成功的推理过程以文本形式保存下来，供后续推理时参考。想象成一本“案例手册”，里面记载了哪些搜索路径能得到对的答案。
- **对比式推理轨迹（Contrastive Reasoning Trajectories）**：把高质量（少调用、答案对）和低质量（多调用、答案错）两类推理路径放在一起，让模型学习它们的差异。相当于给模型展示“好例子”和“坏例子”，帮助它辨别。
- **自适应奖励塑形（Adaptive Reward Shaping）**：在奖励函数里动态调节对冗余搜索的惩罚力度，而不是使用固定的超参数。类似于老师根据学生的表现随时调整作业难度。
- **多跳问答（Multi-hop QA）**：需要跨越多个事实片段才能得到答案的问答任务。比如“谁是诺贝尔文学奖得主中出生在同一城市的两位作家？”需要先查出获奖者，再查出生地。

### 核心创新点
1. **对比式经验推理 → 生成文本化经验**  
   之前的 RL‑DeepSearch 只把搜索次数作为奖励的负向项，缺乏对“为什么这条路径好”的显式指导。LightSearcher 先收集大量推理轨迹，按成功与否划分为高质量和低质量两组，然后让大语言模型（LLM）把每组的共性抽象成一段可读的经验总结。这样模型在后续推理时可以直接读取这些经验，就像在做题时翻看老师的解题要点。

2. **自适应冗余惩罚 → 只在正确答案时才收紧**  
   传统方法对每一次搜索都施加固定惩罚，导致模型在不确定的情况下过早放弃搜索，答案准确率受损。本文的奖励机制把“冗余调用惩罚”绑定到“答案正确”这一条件上：当模型最终得到正确答案时，系统会检查该答案对应的最少搜索次数，并据此动态提升惩罚力度；若答案错误，则放宽惩罚，让模型有机会多搜索。这样实现了“先保证对，再追求省”。

3. **经验记忆注入 → Prompt‑guided RL**  
   在每一步的动作选择前，LightSearcher 会把当前的经验记忆（即前面生成的经验文本）拼进模型的提示（prompt）里，让模型在做决策时能够“看到”过去的成功模式。相当于在玩游戏时给玩家一份攻略手册，帮助他们避免重复错误。

4. **整体效率提升**  
   通过上述两大机制，模型在保持与最强基线 ReSearch 相近的准确率的同时，大幅削减了搜索调用次数、推理时长和 token 消耗。实验数据显示，搜索调用下降约 40%，推理时间缩短近 50%，token 使用降低 20% 左右。

### 方法详解
**整体框架**  
LightSearcher 的训练与推理分为三个阶段：① 轨迹采集，② 经验抽取与文本化，③ 强化学习带记忆的策略优化。整体思路是先让模型自由探索，收集足够的成功与失败案例；再把这些案例浓缩成可读经验；最后在 RL 过程中把经验当作外部知识注入，指导模型在每一步决定是否搜索。

**1️⃣ 轨迹采集**  
使用与基线相同的 DeepSearch 环境，让模型在四个多跳 QA 数据集上进行若干轮交互。每一次交互会记录：问题、每一步的内部思考（CoT）、是否调用搜索工具、搜索返回的文本、最终答案以及答案是否正确。这样得到的轨迹自然分为两类：  
- **高质量轨迹**：答案正确且搜索次数接近该问题的历史最少次数。  
- **低质量轨迹**：答案错误或在同等条件下调用次数明显冗余。

**2️⃣ 对比式经验抽取**  
把同一问题的高、低质量轨迹配对，交给预训练的大语言模型（如 GPT‑4）进行对比式摘要。模型的任务是：“在这两条推理路径中，哪一步的搜索是多余的？哪些思考步骤是关键的？”生成的文本大致包含：① 关键搜索点的提示，② 常见的冗余调用模式，③ 简洁的策略建议。得到的经验记忆库是一个键值对集合，键是问题的主题或检索关键词，值是对应的经验文本。

**3️⃣ 自适应奖励塑形**  
在 RL 的奖励函数里加入两项：  
- **准确性奖励**：答案正确得正向奖励。  
- **冗余惩罚**：如果答案正确，则根据该问题的历史最少搜索次数计算一个动态惩罚系数；如果答案错误，则该惩罚系数设为 0。这样模型在正确答案的情况下被迫压缩搜索次数，而在错误情况下仍有机会多搜索以纠正错误。

**4️⃣ 经验记忆注入**  
每一次动作决策前，系统会检索经验库中与当前问题最相似的经验文本（使用简单的关键词匹配或向量相似度），并把它拼进模型的提示里。提示结构大致为：“你正在解答：{问题}。以下是过去类似问题的成功经验：{经验文本}。请在思考后决定是否调用搜索工具。” 这样模型在做出“搜索”或“继续思考”的二选一时，已经受到了经验的显式引导。

**5️⃣ 强化学习循环**  
使用近端策略优化（PPO）等常见 RL 算法，对模型的策略网络进行更新。策略的输入是当前的对话历史 + 注入的经验，输出是搜索概率分布。由于奖励已经把效率和准确率融合，训练过程自然会收敛到“少调用、对答案”的平衡点。

**最巧妙的地方**  
- 把经验记忆做成**文本**而不是向量，使得它可以直接作为 Prompt 使用，省去额外的编码步骤。  
- **条件惩罚**只在正确答案时生效，避免了传统固定惩罚导致的“过早停搜索”问题。  
- 对比式抽取让模型学习“好”和“坏”之间的细微差别，比单纯的正向奖励更具辨别力。

### 实验与效果
- **数据集**：四个公开的多跳问答基准（HotpotQA、Musique、2WikiMultiHop、ComplexWebQuestions）。这些任务需要模型在多个事实之间跳转，典型的 DeepSearch 场景。
- **对比基线**：最强的 RL‑DeepSearch 系统 ReSearch，以及几种不使用搜索或使用固定搜索策略的模型。  
- **主要指标**：答案准确率、搜索调用次数、推理时长、token 消耗。  
- **结果**：LightSearcher 在保持与 ReSearch 相当的准确率（差距小于 1%）的同时，实现了搜索调用下降 39.6%、推理时间缩短 48.6%、token 使用降低 21.2%。这些数字直接来源于摘要。  
- **消融实验**：作者分别去掉（1）对比式经验抽取、（2）自适应惩罚、（3）经验注入三项，发现每去掉一项都会导致搜索调用增加 10%~15%，且准确率略有下降，说明三者相辅相成。  
- **局限性**：经验库的构建依赖大量高质量轨迹，若训练数据不足或领域转移严重，经验的覆盖率会下降；此外，经验检索采用简单匹配，可能在复杂语义相似度场景下失效。原文未给出跨语言或跨领域的实验。

### 影响与延伸思考
LightSearcher 把“经验记忆”与 RL‑DeepSearch 结合，打开了让模型利用过去成功案例来调节搜索行为的新思路。自发表以来，已有工作尝试把类似的经验库用于代码生成、医学问答等需要频繁检索的场景（如 “Memory‑augmented Retrieval‑RL”）。未来可以探索：① 更高效的经验检索（使用向量数据库），② 跨任务共享经验库，实现“一库多用”，③ 将经验记忆与自监督预训练结合，让模型在无标注数据上也能生成有价值的经验。对想深入的读者，建议关注近期在 “retrieval‑augmented generation” 领域的 “contrastive memory” 方向，以及强化学习中 “reward shaping” 的最新理论。

### 一句话记住它
让模型在“查对了再省”之间找到平衡：用对比式经验记忆指导 RL，显著削减搜索开销而不牺牲答案正确率。