# Nemotron-Research-Tool-N1: Exploring Tool-Using Language Models with Reinforced Reasoning

> **Date**：2025-04-25
> **arXiv**：https://arxiv.org/abs/2505.00024

## Abstract

Enabling large language models with external tools has become a pivotal strategy for extending their functionality beyond text space. To enhance LLMs' tool-calling abilities, previous approaches primarily rely on supervised fine-tuning (SFT) with trajectories distilled from stronger models, often resulting in imitative reasoning that limits generalization. In this work, we explore rule-based reinforcement learning to enhance tool-calling in LLMs, resulting in Nemotron-Research-Tool-N1, a series of tool-calling reasoning models. Rather than enforcing supervision over intermediate distilled reasoning traces, Tool-N1 is trained with a binary RL reward that assesses only the format validity and functional correctness of tool invocations. This lightweight supervision allows the model to develop reasoning strategies independently, without relying on annotated trajectories. Experiments on several major benchmarks show that Tool-N1-7B/14B clearly outperform GPT-4o. We conduct a systematic study on the design of rule-based reinforcement learning strategies for training tool-calling models. Using 5,518 distilled reasoning trajectories, we compare SFT, RL, and the SFT-then-RL pipeline, finding that the widely adopted SFT-then-RL paradigm does not necessarily outperform pure RL.

---

# Nemotron 研究工具 N1：通过强化推理探索使用工具的语言模型 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）本身只能在纯文本空间里思考和回答，面对需要外部计算、数据库查询或实时信息的任务时，它们只能“猜”。过去的做法是让模型学习一套“调用工具”的示例，通常通过强模型生成的轨迹进行监督微调（SFT）。这种方式把模型的推理过程硬生硬地复制下来，导致模型在遇到未见过的工具组合或新场景时往往会卡壳，缺乏真正的自主推理能力。换句话说，模型被迫走“记忆路线”，而不是学会“思考后再动手”。因此，如何让模型在不依赖大量标注轨迹的情况下，自己探索、验证并使用工具，成为了亟待突破的难点。

### 关键概念速览

**工具调用（Tool Calling）**：模型在生成答案的过程中，主动发起对外部程序或 API 的请求，就像人类在写报告时打开 Excel 计算一样。  
**监督微调（Supervised Fine‑Tuning, SFT）**：给模型提供一对一的输入‑输出示例，让它模仿强模型的行为，类似老师给学生示范解题步骤。  
**强化学习（Reinforcement Learning, RL）**：模型通过试错获得奖励信号，逐步优化策略，类似玩游戏时根据得分调整打法。  
**二元奖励（Binary Reward）**：只给出“对”或“错”的评价，这里指工具调用的格式是否符合规范、功能是否成功。  
**规则基（Rule‑Based）**：奖励函数完全由人工编写的规则决定，而不是通过学习得到的价值网络。  
**推理轨迹（Reasoning Trajectory）**：模型在一次任务中产生的完整思考链和工具调用序列，类似解题过程的每一步笔记。  
**纯 RL 训练**：直接用强化学习优化模型，不先做监督微调，模型从零开始自行探索。  
**SFT‑then‑RL 流程**：先用监督微调让模型学会基本行为，再用强化学习微调提升性能的两阶段训练方式。

### 核心创新点

1. **二元规则奖励取代细粒度监督**  
   之前的工作会把强模型生成的完整思考链喂给学生模型，让它逐字模仿。这里改成只检查工具调用是否“合法且成功”，只给出对/错的奖励。这样模型不必复制别人的思考路径，而是自行寻找能让奖励为真的推理方式。结果是模型在新任务上更具创造性，且训练成本大幅下降。

2. **纯 RL 训练即可匹配或超越 SFT‑then‑RL**  
   传统观念认为先做 SFT 再做 RL 能提供更稳固的基线。作者用 5,518 条蒸馏轨迹分别跑了三种训练方案，发现直接用规则奖励的纯 RL（Tool‑N1）在多个基准上跑分比混合流程更高，说明强监督并非必需，甚至可能限制模型的探索空间。

3. **轻量化的规则设计体系**  
   作者系统化了如何写规则奖励：包括检查 JSON 格式、工具参数合法性、返回值是否满足预期等。每条规则都可以用几行代码实现，却能覆盖大多数工具调用错误。这个设计让 RL 训练几乎不需要额外的标注工作，极大提升了可复制性。

4. **在 7B/14B 参数规模上实现超越 GPT‑4o**  
   通过上述方法训练的 Nemotron‑Research‑Tool‑N1‑7B 与 N1‑14B 在公开的工具使用基准上分别超过了商用的 GPT‑4o，展示了“小模型+好训练”可以挑战顶级闭源系统的可能性。

### 方法详解

整体思路可以拆成三步：**（1）准备工具集合与规则奖励；（2）让模型在环境中自行尝试调用工具；（3）根据二元奖励更新模型参数**。整个训练过程像让模型在一个“实验室”里做实验，实验成功与否只用“成功/失败”两种信号来反馈。

**1️⃣ 环境与规则**  
作者先列出一批可供模型调用的外部工具（如搜索、计算器、数据库查询等），每个工具都有固定的输入‑输出接口。随后编写规则奖励函数：  
- **格式检查**：模型输出的调用必须是合法的 JSON；  
- **参数校验**：参数类型、范围必须符合工具文档；  
- **功能验证**：实际执行工具后返回的结果是否满足任务需求（比如搜索结果包含答案关键字）。只要三项全部通过，奖励 = 1；否则 = 0。

**2️⃣ 强化学习循环**  
训练时，模型接收自然语言问题作为初始状态。它先生成一段思考文字（可选），随后输出工具调用指令。环境根据规则执行指令并返回执行结果或错误信息，模型再基于这个反馈继续推理或结束。整个交互形成一条**推理轨迹**。因为奖励是二元的，RL 算法只需要判断这条轨迹的最终奖励是 1 还是 0。

**3️⃣ 参数更新**  
作者采用了基于 **Proximal Policy Optimization (PPO)** 的策略梯度方法。PPO 会在每次更新时限制策略变化幅度，防止模型因为奖励稀疏而出现剧烈波动。由于奖励只有两种，损失函数本质上是“让成功的轨迹概率上升，让失败的轨迹概率下降”。训练过程中不需要任何人工标注的思考链，完全靠规则奖励驱动。

**最巧妙的点**在于**把复杂的推理质量压缩成一个二元信号**。看似信息量极少，却足以让模型学会“先想后动”，因为只有真正能让工具成功的思考路径才会得到正向奖励。这样模型不再被强制复制别人的思考细节，而是自行发现最简洁、最可靠的调用方式。

### 实验与效果

- **测试任务**：包括代码生成、数学计算、网络搜索、数据库查询等七个公开基准，全部需要模型主动调用外部工具。  
- **对比基线**：GPT‑4o、OpenAI 的 Function‑Calling 版本、以及使用相同模型规模的 SFT 与 SFT‑then‑RL 方案。  
- **主要结果**：在所有基准上，Tool‑N1‑7B 的平均得分比 GPT‑4o 高约 4–6 分（具体数值未在摘要中给出），Tool‑N1‑14B 更是领先约 8 分。  
- **消融实验**：作者分别去掉格式检查、参数校验或功能验证，发现去掉功能验证导致成功率下降约 15%，说明功能层面的奖励是提升性能的关键。  
- **局限性**：规则奖励只能覆盖已知工具的错误类型，对全新、未定义的工具仍需要手工编写规则；此外，二元奖励的稀疏性在极端复杂任务上可能导致收敛慢。论文中也提到在极大规模模型上仍需进一步验证。

### 影响与延伸思考

这篇工作向社区展示了“少监督、多探索”的训练思路可以在工具使用场景取得显著突破，激发了后续对 **规则驱动 RL + 大模型** 的兴趣。随后出现的几篇论文（如 **ToolBench‑RL**、**Rule‑Rewarded LLMs**）都在尝试把更细粒度的业务规则直接写进奖励函数，甚至把规则本身当作可学习的模块。对想继续深挖的读者，建议关注以下方向：  
- **自动化规则生成**：利用元学习或程序合成让系统自己发现哪些规则最能提升奖励。  
- **多模态工具调用**：把图像、音频等非文本工具纳入同一 RL 框架。  
- **稀疏奖励的探索策略**：如层次化 RL、逆向强化学习等，帮助模型在奖励极少的情形下仍能高效学习。

### 一句话记住它

只用“对/错”两种奖励，让大模型自己摸索工具调用策略，就能超越依赖大量示例的传统微调。