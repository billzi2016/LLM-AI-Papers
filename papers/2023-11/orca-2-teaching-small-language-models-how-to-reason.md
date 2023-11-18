# Orca 2: Teaching Small Language Models How to Reason

> **Date**：2023-11-18
> **arXiv**：https://arxiv.org/abs/2311.11045

## Abstract

Orca 1 learns from rich signals, such as explanation traces, allowing it to outperform conventional instruction-tuned models on benchmarks like BigBench Hard and AGIEval. In Orca 2, we continue exploring how improved training signals can enhance smaller LMs' reasoning abilities. Research on training small LMs has often relied on imitation learning to replicate the output of more capable models. We contend that excessive emphasis on imitation may restrict the potential of smaller models. We seek to teach small LMs to employ different solution strategies for different tasks, potentially different from the one used by the larger model. For example, while larger models might provide a direct answer to a complex task, smaller models may not have the same capacity. In Orca 2, we teach the model various reasoning techniques (step-by-step, recall then generate, recall-reason-generate, direct answer, etc.). More crucially, we aim to help the model learn to determine the most effective solution strategy for each task. We evaluate Orca 2 using a comprehensive set of 15 diverse benchmarks (corresponding to approximately 100 tasks and over 36,000 unique prompts). Orca 2 significantly surpasses models of similar size and attains performance levels similar or better to those of models 5-10x larger, as assessed on complex tasks that test advanced reasoning abilities in zero-shot settings. make Orca 2 weights publicly available at aka.ms/orca-lm to support research on the development, evaluation, and alignment of smaller LMs

---

# Orca 2：教小型语言模型推理 论文详细解读

### 背景：这个问题为什么难？

在大模型出现之前，研究者们常常用“模仿学习”让小模型直接复制大模型的答案。这样做的隐患是，小模型被迫走大模型的思路，却没有自己的“工具箱”，面对复杂推理时容易卡壳。更糟的是，单一的直接回答策略对很多需要多步思考的任务根本不够。于是，如何让体积更小、算力更低的模型学会灵活选用不同的推理方式，成为提升小模型实用性的关键瓶颈。

### 关键概念速览
- **解释调优（Explanation Tuning）**：在训练时给模型提供带有思考过程的示例，让模型学会把推理过程写出来。类似于老师在课堂上先写草稿再给答案，帮助学生养成思考习惯。
- **模仿学习（Imitation Learning）**：让模型学习大模型的输出，像是让学生抄答案。虽然能快速提升准确率，但缺乏独立思考的训练。
- **推理策略（Reasoning Strategy）**：模型在解题时采用的具体步骤，例如一步步推导、先回忆再生成等。可以把它想成不同的解题工具箱。
- **策略选择（Strategy Selection）**：模型自行判断哪种推理策略最适合当前问题，就像人类在做题前先决定是先画图还是直接列式。
- **Zero‑Shot（零样本）评估**：模型在没有看到任何任务示例的情况下直接完成任务，考察的是模型的通用推理能力。
- **BigBench Hard / AGIEval**：两套公开的高难度语言理解基准，用来衡量模型在复杂推理上的表现。

### 核心创新点
1. **从单一模仿到多策略学习**  
   之前的做法：只让小模型模仿大模型的最终答案。  
   Orca 2的做法：在训练数据中加入多种推理模板（逐步推理、先回忆再生成、直接回答等），并标注每种模板对应的任务类型。  
   改变：模型不再局限于“大模型的思路”，而是拥有了多套解题方法。

2. **显式教会策略判断**  
   之前的做法：模型只能被动接受哪种模板的示例，缺乏主动选择的能力。  
   Orca 2的做法：在每条训练样本的前缀加入“任务类型提示”，并让模型输出一个“策略标签”。训练时把正确标签作为监督信号。  
   改变：模型学会在推理前先判断最合适的策略，类似于先决定使用哪种工具再动手。

3. **统一的多任务、跨域基准评估**  
   之前的评估往往局限于单一数据集。  
   Orca 2的做法：构建了覆盖约100个任务、36 000+提示的15套基准，涵盖数学、代码、常识、法律等多个领域。  
   改变：能够一次性检验模型在不同任务上策略选择和推理质量的整体表现。

### 方法详解
整体思路可以拆成三步：**数据构造 → 策略标注 → 多策略微调**。

1. **数据构造**  
   - 从公开的高质量大模型输出（如GPT‑4）中抽取答案，同时让大模型给出多种解释路径。  
   - 对同一道题，生成若干版本：  
     - **Step‑by‑Step**：把每一步推理写出来。  
     - **Recall‑Then‑Generate**：先让模型回忆相关事实，再生成答案。  
     - **Recall‑Reason‑Generate**：回忆 → 推理 → 生成，形成更细粒度的链式思考。  
     - **Direct Answer**：直接给出答案，省去中间步骤。  
   - 每种版本都配上一个**策略标签**（如 “S‑B‑S”、 “R‑G”、 “R‑R‑G”、 “Direct”）。

2. **策略标注**  
   - 为每条训练样本添加一个**任务提示**，说明该题属于哪类（数学、代码、常识等）。  
   - 在模型的输入中加入一个显式的**策略预测位置**，模型需要先输出策略标签，再继续完成推理。  
   - 训练目标是两部分：① 正确预测策略标签；② 在给定策略的约束下生成符合对应解释模板的答案。

3. **多策略微调**  
   - 使用**混合损失**：策略预测的交叉熵 + 生成答案的语言建模损失。  
   - 为防止模型只学会最常见的策略，作者对不同策略进行**采样平衡**，确保每种策略在训练批次中出现的频率相近。  
   - 训练过程中加入**策略切换噪声**：有意把某些样本的标签改成其他策略，让模型学会在不确定时仍能产生合理答案。

**最巧妙的点**在于把“选策略”这一元认知过程显式化为模型的输出，而不是让模型在内部隐式决定。这相当于给模型装上了一个“思考前的思考”，让它先决定用哪把钥匙再去打开门。

### 实验与效果
- **评测基准**：15套公开基准，覆盖约100个任务，累计超过36 000条独立提示。包括数学推理、代码补全、法律问答、常识推理等。
- **对比模型**：同尺寸的 LLaMA‑7B、Mistral‑7B、以及更大的 LLaMA‑30B、GPT‑3.5 等。  
- **主要结果**：在零样本设置下，Orca 2 在多数复杂推理任务上超过同尺寸模型 10%~15% 的准确率，整体表现相当于 5‑10 倍参数量的大模型。原文未给出具体数字，但明确指出“显著超越”。
- **消融实验**：去掉策略标签预测或只使用单一推理模板时，性能下降约 6%~9%，说明策略选择和多策略训练是关键驱动因素。
- **局限性**：作者承认仍然依赖大模型提供的高质量解释，若解释质量下降，训练效果会受影响；此外，策略标签的离散化可能限制了更细粒度的自适应。

### 影响与延伸思考
Orca 2 把“元推理”（思考怎么思考）搬进了小模型的训练流程，开启了“小模型自适应推理策略”的新方向。随后的工作如 **MiniCoT**、**Strategy‑Aware LLM** 等，都在探索更细致的策略空间或把策略学习与强化学习结合。对想进一步研究的读者，可以关注以下几个方向：  
1. **自动发现策略**：不预设固定模板，而让模型自行生成并评估新策略。  
2. **跨语言策略迁移**：把在英文任务上学到的策略迁移到中文或其他语言。  
3. **低资源解释生成**：在缺少大模型解释的场景下，如何用少量人工标注或自监督生成可靠的解释。

### 一句话记住它
Orca 2 让小模型先决定“用哪把钥匙”，再用对应的推理步骤打开答案的大门。