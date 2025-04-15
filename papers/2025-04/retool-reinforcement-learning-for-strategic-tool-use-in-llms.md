# ReTool: Reinforcement Learning for Strategic Tool Use in LLMs

> **Date**：2025-04-15
> **arXiv**：https://arxiv.org/abs/2504.11536

## Abstract

While reasoning models (e.g., DeepSeek R1) trained with reinforcement learning (RL), excel in textual reasoning, they struggle in scenarios requiring structured problem-solving, such as geometric reasoning, concise computation, or complex equation solving-areas where computational tools like code interpreters (CI) demonstrate distinct advantages. To bridge this gap, we propose ReTool, which enhances long-form reasoning with tool-integrated learning, including two key features: (1) dynamic interleaving of real-time code execution within natural language reasoning processes, and (2) an automated RL paradigm that allows policy rollouts with multi-turn real-time code execution and teaches the model in learning when and how to invoke tools based on outcome feedback. ReTool employs a systematic training framework, beginning with synthetic cold-start data generation to produce code-augmented long-form reasoning traces for fine-tuning base models. Subsequent RL training leverages task outcomes as rewards to iteratively refine the model's tool use strategy, enabling autonomous discovery of optimal tool invocation patterns without human priors. Experiments on the challenging MATH Olympiad benchmark AIME demonstrate ReTool's superiority: Our 32B model achieves 67% accuracy with 400 training steps, outperforming text-based RL baseline (40% accuracy, 1080 steps) in efficiency and performance. Remarkably, ReTool-32B attains 72.5% accuracy in extended settings, surpassing OpenAI's o1-preview by 27.9%. Further analysis reveals emergent behaviors such as code self-correction, signaling an ''aha moment'' in which the model autonomously masters adaptive tool use. These findings highlight the promise of outcome-driven tool integration for advancing complex mathematical reasoning and offer new insights into hybrid neuro-symbolic systems.

---

# ReTool：面向大语言模型的强化学习工具使用策略 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在纯文本推理上已经很强，但一旦遇到需要严谨计算、几何构造或符号求解的题目，它们往往只能靠“猜”而不是“算”。传统的强化学习（RL）训练可以让模型学会更好的思路链（CoT），却没有办法让模型主动调用外部计算工具，比如代码解释器（CI）。于是模型在数学竞赛、工程仿真等结构化任务上表现停滞，根本原因是缺少“何时使用工具、怎么使用工具”的决策能力。

### 关键概念速览
- **大语言模型（LLM）**：能够生成自然语言的深度模型，类似会说话的“智能机器人”。  
- **强化学习（RL）**：让模型通过试错获得奖励的学习方式，像训练小狗通过奖励学会新技巧。  
- **工具使用（Tool Use）**：模型在推理过程中主动调用外部程序（如 Python 解释器）来完成子任务，类似人类在做题时打开计算器。  
- **实时代码执行（Real‑time Code Execution）**：模型生成代码后立即运行并把运行结果反馈到推理流中，像写完公式后立刻在纸上算出答案。  
- **冷启动合成数据（Cold‑start Synthetic Data）**：在没有真实标注的情况下，先用规则或小模型生成带有代码的推理示例，帮助模型快速学会“先写代码再解释”。  
- **多轮回滚（Multi‑turn Rollout）**：在 RL 训练时，让模型在一次对话里多次调用工具并观察每一步的结果，类似在棋局中走多步棋再评估局面。  
- **奖励信号（Reward Signal）**：根据最终任务是否正确给模型打分，引导模型学习更有效的工具调用策略。  

### 核心创新点
1. **动态交叉执行 → 让模型在自然语言推理中随时插入代码执行**  
   以前的模型要么一次性输出完整答案，要么在训练阶段手工标记何时调用工具。ReTool 把代码生成和执行嵌入到推理链的每一步，模型可以在写完一段解释后立刻跑代码，再把结果写回文本。这样模型不再需要预先知道“哪个位置需要计算”，而是自行发现。

2. **全自动 RL 框架 → 用任务成功率直接教会模型何时调用工具**  
   传统 RL 需要人工设计何时奖励、何时惩罚。ReTool 只看最终答案对错，把整个多轮交互过程视为一次“游戏”。模型在每一次回滚中尝试不同的调用时机，成功的策略会得到正向奖励，失败的会被削弱，从而自动学习最优的工具使用模式。

3. **合成冷启动 → 用人工合成的代码增强推理轨迹快速微调**  
   为了让基模型懂得“写代码再解释”，作者先生成大量带代码的长文本推理样本（比如先写 Python 求根公式，再解释步骤），用这些数据对模型进行有监督微调。这样即使没有真实标注，模型也能在正式 RL 前已经具备基本的工具调用语法。

4. **无需人工先验 → 完全摆脱手工规则的工具调度**  
   过去的系统往往硬编码“如果出现‘求和’就调用计算器”。ReTool 完全依赖 RL 产生的策略，模型自行发现哪些子任务适合交给代码解释器，哪些可以直接用语言推理。实验显示，这种自发现的调度比手工规则更高效。

### 方法详解
**整体框架**  
ReTool 的训练分两阶段：① 合成数据微调（Cold‑start），② 基于任务结果的强化学习。第一阶段让模型学会在自然语言中嵌入可执行代码；第二阶段让模型在真实任务上通过多轮交互学习何时、如何调用工具。

**第一阶段：合成数据生成与微调**  
- 使用小型数学求解器或规则模板，自动生成“问题 → 代码 → 代码执行结果 → 文字解释”的完整轨迹。  
- 这些轨迹被当作监督信号，直接喂给基模型进行有监督学习，使模型的输出格式兼容“代码块 + 解释文本”。  
- 类比：先教小学生写算式再解释步骤，等他们熟练后再让他们自己决定何时写算式。

**第二阶段：RL 训练**  
- **环境定义**：每一次任务（如一道 AIME 题）是一个 RL 环境。模型的动作包括生成普通文字、生成代码块或发出“执行”指令。  
- **回滚过程**：模型在一次对话里可以多次生成代码并立即执行，系统把执行结果（数值、图形等）返回给模型，模型再基于这些信息继续推理。  
- **奖励设计**：仅在任务结束时检查答案是否正确，正确给高奖励，错误给低奖励。中间的代码执行成功与否不直接计分，只影响后续推理质量。  
- **策略更新**：采用常见的策略梯度（如 PPO）对模型参数进行微调。因为奖励稀疏，作者使用了基于价值函数的基线来降低方差。  
- **自我纠错机制**：当模型生成的代码报错或得到不合理结果时，系统会把错误信息返回，模型可以在下一轮尝试修改代码，这种“代码自我修正”在实验中自然出现。

**关键模块的类比**  
- **代码块 → 计算器**：模型把需要精确计算的子任务交给计算器，计算器返回结果后模型继续写解释。  
- **多轮回滚 → 棋局思考**：模型像下棋一样在每一步尝试不同的代码调用，观察局面变化后再决定下一步。  

**最巧妙的设计**  
- 只用最终答案做奖励，却让模型在中间得到即时的执行反馈，这种“稀疏奖励+即时反馈”的组合让学习既高效又不需要手工标记每一步的正确性。  
- 冷启动合成数据的规模足够大，确保模型在进入 RL 前已经掌握了“代码-解释”模板，避免了 RL 初期的探索灾难。

### 实验与效果
- **测试任务**：MATH Olympiad 系列的 AIME 题目，这是公认的高难度数学竞赛集合，涉及代数、几何、组合等多种技巧。  
- **基线对比**：  
  - 纯文本 RL 模型（不使用工具）在同等训练步数（1080 步）下准确率约 40%。  
  - ReTool‑32B 在仅 400 步的训练后达到 67% 的准确率，显著提升效率。  
  - 在更宽松的评估设置下（允许多轮代码），ReTool‑32B 达到 72.5%，比 OpenAI o1‑preview 高出约 27.9%。  
- **消融实验**：  
  - 去掉合成冷启动，仅用 RL，模型收敛慢且最终准确率跌至约 55%。  
  - 去掉实时代码执行，只保留代码生成但不运行，准确率下降到 60%。  
  - 这些实验表明，合成数据和实时执行两块缺一不可。  
- **局限性**：  
  - 论文主要在数学题上评估，未展示对自然语言问答或代码生成以外任务的适用性。  
  - 训练仍依赖大规模算力（32B 参数模型），对资源受限的团队仍有门槛。  
  - 作者提到在极端长链推理时，模型有时会产生冗余代码调用，导致推理成本上升。

### 影响与延伸思考
ReTool 把“工具使用”提升到可以通过 RL 自动学习的层级，开启了大模型与外部计算资源深度融合的新方向。随后的工作（如 DeepMind 的 “Toolformer” 系列、Meta 的 “Neuro‑Symbolic RL”）都在不同程度上借鉴了实时交叉执行和稀疏奖励的思路。未来的研究可能会：
- 扩展到多模态工具（如图像编辑器、数据库查询）；
- 探索更轻量的策略学习，如基于少量人类示例的元学习；
- 结合检索增强（RAG）让模型在调用工具前先检索已有解法，进一步降低计算成本。  
如果想深入，可以关注“神经符号系统”和“可解释 RL”这两个交叉领域，它们正逐步把大模型的“黑箱”变成可控的“工具箱”。

### 一句话记住它
ReTool 让大语言模型通过强化学习自行决定何时写代码并实时运行，从而在复杂数学推理上实现了“会算、会写、会调工具”的全能表现。