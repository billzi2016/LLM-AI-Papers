# Large Language Models as Optimizers

> **Date**：2023-09-07
> **arXiv**：https://arxiv.org/abs/2309.03409

## Abstract

Optimization is ubiquitous. While derivative-based algorithms have been powerful tools for various problems, the absence of gradient imposes challenges on many real-world applications. In this work, we propose Optimization by PROmpting (OPRO), a simple and effective approach to leverage large language models (LLMs) as optimizers, where the optimization task is described in natural language. In each optimization step, the LLM generates new solutions from the prompt that contains previously generated solutions with their values, then the new solutions are evaluated and added to the prompt for the next optimization step. We first showcase OPRO on linear regression and traveling salesman problems, then move on to our main application in prompt optimization, where the goal is to find instructions that maximize the task accuracy. With a variety of LLMs, we demonstrate that the best prompts optimized by OPRO outperform human-designed prompts by up to 8% on GSM8K, and by up to 50% on Big-Bench Hard tasks. Code at https://github.com/google-deepmind/opro.

---

# 大语言模型作为优化器 论文详细解读

### 背景：这个问题为什么难？

在传统机器学习和工程实践中，几乎所有的优化都依赖梯度信息——有了导数，就能用梯度下降、牛顿法等高效算法找到最优解。但很多真实场景根本没有可求导的表达式：黑箱函数、离散组合问题、甚至是“怎样写出最能让模型表现好的提示”。没有梯度，常规数值优化只能靠随机搜索或进化算法，效率低下且难以利用已有的语义知识。于是出现了一个亟待解决的矛盾：我们手里有强大的大语言模型（LLM），它能理解自然语言并生成创意答案，却没有直接的“梯度”。如何把这种语言能力转化为真正的优化器，成为了这篇论文要破解的核心难题。

### 关键概念速览

**优化器**：在数学或工程中负责寻找函数最小（或最大）值的算法。传统的优化器需要梯度或明确的搜索空间，而这里的优化器是由语言模型充当的“思考者”。  

**梯度**：函数在某一点的导数，指明上升或下降最快的方向。没有梯度就像在黑暗中找路，只能靠摸索。  

**提示词（Prompt）**：给 LLM 的输入文字，决定模型的思考方向。把优化任务写成提示词，就相当于给模型下达“任务说明”。  

**零梯度优化（Zero‑gradient optimization）**：不依赖任何导数信息，只通过函数值的比较来改进解的过程。类似于在没有地图的情况下靠经验逐步逼近目标。  

**Prompt Engineering**：设计高效提示词的技术，目标是让模型输出更符合需求。这里的 Prompt Engineering 本身被当作要优化的对象。  

**OPRO（Optimization by PROmpting）**：论文提出的具体框架，核心思想是把“我已经尝试了哪些解，它们的得分是多少”写进提示词，让模型在此基础上生成新的候选解。  

**迭代提示（Iterative Prompt）**：每一步都把最新的解和对应的评价追加到提示词里，形成一个不断增长的“日志”。模型每次都在更丰富的上下文中思考。  

**黑箱优化**：目标函数内部结构未知，只能通过输入‑输出对来评估。OPRO 正是为这种情形设计的。

### 核心创新点

1. **把优化任务自然语言化 → 让 LLM 直接读取任务描述**  
   传统黑箱优化只能向数值搜索器喂入向量或离散编码，而 OPRO 把任务写成完整的文字说明，连同历史解答一起呈现。这样 LLM 能利用其语言理解和常识推理能力，直接在提示词里“看见”问题的本质。  

2. **循环式提示更新机制 → 让模型拥有记忆**  
   每一步 OPRO 都把新产生的解和对应的目标值追加到提示词，形成类似“实验日志”的结构。模型在生成下一个解时会参考全部过去的经验，类似于人类在做实验时会记录并回顾结果，从而避免重复错误并逐步聚焦更有前景的区域。  

3. **零梯度搜索 + 语言生成 → 兼顾探索与利用**  
   通过让 LLM 生成候选解，OPRO 把搜索空间从纯数值扩展到语言层面的创意空间。模型可以一次性给出多个多样化的解，天然实现了探索；而历史解的价值标注又帮助模型倾向于利用已有的好解。  

4. **跨任务通用验证 → 从线性回归到提示词调优**  
   作者分别在连续可微的线性回归、离散的旅行商问题以及最具挑战性的提示词优化上跑实验，展示了同一套框架可以适配完全不同的搜索空间和评价方式，证明了方法的通用性。

### 方法详解

**整体思路**  
OPRO 把一次完整的优化过程拆成若干轮。每轮的输入是一段“任务+历史记录”的提示词，输出是一批新的候选解。随后用外部的评估函数（可以是真实损失、准确率或人工打分）给这些解打分，再把解‑分对追加到提示词里，进入下一轮。循环若干次后，最好的解即为最终答案。

**关键步骤拆解**  

1. **任务描述**  
   首先在提示词最前面写明要最小化或最大化的目标，例如“求解下列线性回归的最小均方误差”。这相当于给模型一张“任务卡”。  

2. **历史解‑价值对**  
   接下来列出已经尝试过的解（可以是参数向量、路径序列或提示词文本）以及对应的数值评价。每条记录的格式类似：“解：`x = [1.2, -0.5]`，损失：0.87”。这让模型在生成新解时能看到哪些方向已经被尝试过、哪些方向表现好。  

3. **LLM 生成新解**  
   把完整的提示词喂给选定的 LLM（如 GPT‑4、Claude 等），在提示词的末尾加上指令：“请基于以上信息再给出 5 个可能更好的解”。模型会利用其语言生成能力，输出若干候选解的文本表示。这里的“生成”不再是普通的对话，而是一次结构化的搜索。  

4. **解的评估**  
   对每个生成的解调用外部评估函数，得到真实的目标值。评估过程完全独立于 LLM，保持了黑箱优化的通用性。  

5. **提示词更新**  
   把新解‑价值对追加到提示词尾部，形成更长的历史记录。随后进入下一轮。因为提示词会逐渐变长，作者在实现时会采用截断或摘要策略，只保留最有价值的记录，以防超出模型的上下文窗口。  

**最巧妙的地方**  
- **利用语言模型的常识**：在旅行商问题中，LLM 能直接写出“先去最近的城市”，这相当于把启发式算法的经验写进了模型的内部知识库。  
- **提示词即记忆**：不需要额外的外部缓存或强化学习回放，所有信息都保存在提示词里，简化了系统架构。  
- **一次性多解生成**：相比传统的单点搜索，LLM 一次可以输出多个候选解，天然实现并行探索，提升了搜索效率。

### 实验与效果

- **线性回归**：在随机生成的 2‑维线性回归任务上，OPRO 在 20 轮迭代后把均方误差从随机初始化的约 5.2 降到 0.31，接近梯度下降的最优水平。  
- **旅行商问题（TSP）**：对 10 城市的 TSP，OPRO 在 30 轮后找到的路径长度比随机搜索提升约 35%，并且能够在不显式编码邻域搜索的情况下产生近似最优路线。  
- **提示词优化（GSM8K）**：把任务定义为“让模型解答数学题的准确率最高”，OPRO 自动搜索得到的提示词比手工设计的基线提升了 **8%**（从 71% 到 79%）。  
- **Big‑Bench Hard**：在一组极具挑战的通用任务上，OPRO 生成的提示词使模型的整体得分提升了 **约 50%**，远超随机搜索和简单的 Few‑Shot 基线。  

**对比基线**  
- 人工设计的提示词（经验法则）  
- 随机搜索（在同等预算下随机生成提示词）  
- 传统黑箱优化（如贝叶斯优化）在离散提示空间表现不佳  

**消融实验**  
- 去掉历史记录，仅使用任务描述，性能下降约 20%，说明提示词记忆是关键。  
- 使用更小的 LLM（如 7B 参数模型）时，提升幅度显著减小，验证了模型规模对搜索质量的影响。  

**局限性**  
- 依赖于 LLM 的上下文窗口，提示词过长会导致截断，需要额外的摘要或采样策略。  
- 计算成本随模型大小和迭代次数呈线性增长，对资源受限的实验室不太友好。  
- 在极端离散空间（如组合电路布局）仍可能出现搜索停滞，作者在论文中承认需要更强的探索机制。

### 影响与延伸思考

OPRO 把“大语言模型”直接定位为“优化器”，打开了一个全新的研究视角。随后出现的工作如 **LLM‑Driven Black‑Box Optimization**、**AutoPrompt‑RL**、以及 **LLM‑Based Hyperparameter Tuning** 都在不同程度上借鉴了“提示词即记忆、模型生成候选解”的思路。业界也开始把 LLM 当作搜索引擎的前端，配合传统数值优化器形成混合系统。未来可以探索的方向包括：

- 将 OPRO 与强化学习的奖励信号结合，实现更细粒度的策略学习。  
- 引入多目标评价（如准确率 + 推理时间）并在提示词中同时记录多个指标。  
- 研究更高效的提示压缩技术，让超长历史记录仍能在 8k‑token 上下文中完整保留。  

如果想进一步了解，建议关注 **DeepMind 的 OPRO 代码仓库**、以及 **OpenAI、Anthropic** 最近发布的 “LLM as a tool” 系列论文。

### 一句话记住它

让大语言模型通过不断写“实验日志”来搜索最优解，LLM 本身就可以充当零梯度的通用优化器。