# rStar2-Agent: Agentic Reasoning Technical Report

> **Date**：2025-08-28
> **arXiv**：https://arxiv.org/abs/2508.20722

## Abstract

We introduce rStar2-Agent, a 14B math reasoning model trained with agentic reinforcement learning to achieve frontier-level performance. Beyond current long CoT, the model demonstrates advanced cognitive behaviors, such as thinking carefully before using Python coding tools and reflecting on code execution feedback to autonomously explore, verify, and refine intermediate steps in complex problem-solving. This capability is enabled through three key innovations that makes agentic RL effective at scale: (i) an efficient RL infrastructure with a reliable Python code environment that supports high-throughput execution and mitigates the high rollout costs, enabling training on limited GPU resources (64 MI300X GPUs); (ii) GRPO-RoC, an agentic RL algorithm with a Resample-on-Correct rollout strategy that addresses the inherent environment noises from coding tools, allowing the model to reason more effectively in a code environment; (iii) An efficient agent training recipe that starts with non-reasoning SFT and progresses through multi-RL stages, yielding advanced cognitive abilities with minimal compute cost. To this end, rStar2-Agent boosts a pre-trained 14B model to state of the art in only 510 RL steps within one week, achieving average pass@1 scores of 80.6% on AIME24 and 69.8% on AIME25, surpassing DeepSeek-R1 (671B) with significantly shorter responses. Beyond mathematics, rStar2-Agent-14B also demonstrates strong generalization to alignment, scientific reasoning, and agentic tool-use tasks. Code and training recipes are available at https://github.com/microsoft/rStar.

---

# rStar2‑Agent：自主推理技术报告 论文详细解读

### 背景：这个问题为什么难？

在数学和科学推理中，模型往往只能一次性输出答案，缺少对中间步骤的检查和纠错。传统的长链式思维（Long CoT）虽然让模型先写出推理过程，但仍然把代码执行视作黑箱，遇到错误时只能“盲目”继续。把大型语言模型（LLM）和可编程工具（如 Python）结合时，执行环境的噪声、代码错误的高回滚成本以及训练资源的限制，导致之前的“工具使用”模型要么不敢主动调用代码，要么在调用后无法有效利用反馈进行自我修正。于是，如何让模型像人一样“先思考、再写代码、再根据运行结果反思”，成为提升高阶数学和科学任务表现的关键瓶颈。

### 关键概念速览
- **Agentic RL（自主强化学习）**：让模型在交互式环境中自行决定何时使用工具、怎样解释反馈，就像机器人在真实世界中不断尝试并学习。区别于普通 RL，只优化最终奖励，Agentic RL 关注决策过程本身的可解释性和自我纠错能力。
- **CoT（思维链）**：模型在给出答案前先把推理步骤写出来，类似人在纸上列草稿，帮助模型保持逻辑连贯。长 CoT 进一步延伸到多轮推理，但仍缺乏对外部工具的动态调度。
- **Python 代码环境**：模型可以生成并执行 Python 代码，得到实际数值或图形结果。把 LLM 当成“会写代码的助理”，但执行错误会产生噪声，需要专门的容错机制。
- **GRPO‑RoC（Resample‑on‑Correct）**：一种专门针对代码环境噪声的强化学习算法，核心思想是当模型的代码执行得到正确结果时，重新抽样（resample）以强化该行为；执行错误则保持原 rollout，避免误导学习信号。
- **SFT（监督微调）**：先用大量标注好的推理示例让模型学会基本的思考方式，相当于给模型上“基础课”，后面再用 RL 进行高级技巧的强化。
- **pass@1**：在代码生成任务中，只要模型的第一条生成代码能够通过全部测试用例，就算成功。这里用来衡量数学题目解答的准确率。
- **AIME（美国数学邀请赛）**：高难度的数学竞赛题目，常被用作衡量模型数学推理极限的基准。

### 核心创新点
1. **高效的 RL 基础设施 → 可靠的 Python 环境 + 大批量执行**  
   过去在代码环境中做 RL，单次 rollout 需要几秒甚至分钟，导致训练成本爆炸。rStar2‑Agent 搭建了一个能够并行、低延迟执行 Python 代码的容器系统，并在 64 块 MI300X GPU 上实现了“高吞吐”调度。这样即使在 510 步的 RL 训练中，也只用了约一周时间，显著降低了资源门槛。

2. **GRPO‑RoC 算法 → “正确即重抽样”策略**  
   代码执行本身会产生噪声（比如随机数、环境依赖），传统 RL 把所有 rollout 当作同等信号会导致学习不稳定。GRPO‑RoC 在检测到代码输出符合预期时，主动对该 rollout 进行重新抽样，以强化正确行为；若输出错误，则保持原 rollout，防止错误被误强化。这样模型在代码环境里能更稳健地学会“先检查、后继续”。

3. **分阶段训练配方 → 从非推理 SFT 到 多轮 RL**  
   作者先用大规模非推理指令微调（SFT），让模型掌握基本语言能力和工具调用的安全姿态；随后进入多轮 RL，每轮都加入更复杂的数学题目和更高频率的代码反馈。相比一次性端到端 RL，这种渐进式配方在同等算力下获得了更强的认知行为（如自我反思、代码调试）。

4. **“思考后再写代码”行为的显式建模**  
   通过在奖励函数中加入“思考时间”惩罚和“代码质量”奖励，模型学会在使用 Python 前先生成思考笔记，类似人类先在纸上列公式再动手编程。实验显示，这一策略显著提升了在 AIME 这类需要多步推导的任务上的通过率。

### 方法详解
**整体框架**  
rStar2‑Agent 的训练流程可以划分为三大阶段：① 基础 SFT（非推理指令），② 多阶段 Agentic RL（GRPO‑RoC），③ 细粒度奖励调优。整个系统围绕一个“思考‑编码‑执行‑反思”循环展开，模型在每一步都可以决定是否进入下一环节。

**关键模块拆解**  

1. **思考生成模块**  
   - 输入：题目描述 + 上下文。  
   - 输出：自然语言的思考笔记（如“先把方程化简，再检查根的范围”）。  
   - 类比：相当于学生在解题前先在草稿纸上写下解题思路。

2. **代码生成模块**  
   - 接收思考笔记作为提示，生成对应的 Python 代码。  
   - 采用“代码模板+填空”方式，确保生成的代码结构符合可执行性。

3. **高吞吐代码执行环境**  
   - 使用容器化的 Python 沙箱，支持并行运行上千个代码实例。  
   - 每次执行返回标准输出、错误信息以及运行时间，供后续奖励计算。

4. **反馈与反思模块**  
   - 根据执行结果（成功、错误、异常）生成反馈文本。  
   - 模型再次读取该反馈，决定是接受答案、修正代码还是重新思考。  
   - 这里的“Resample‑on‑Correct”机制在奖励层面起作用：如果反馈显示代码正确，系统会在同一状态下重新抽样一次，以强化该成功路径。

5. **GRPO‑RoC 强化学习循环**  
   - **状态**：题目 + 当前思考/代码片段。  
   - **动作**：继续思考、生成代码、接受答案、请求重抽样。  
   - **奖励**：综合了答案正确性、代码执行成功率、思考长度惩罚以及自我纠错次数。  
   - 通过 PPO（近端策略优化）实现策略更新，GRPO‑RoC 只在“正确” rollout 上执行额外的重抽样步骤，从而提升信号的信噪比。

**最巧妙的设计**  
- **奖励函数的多维度设计**：把“思考时间”当作负奖励，让模型学会在必要时停下来，却不至于无限拖延。  
- **分阶段 SFT → RL**：先让模型熟悉工具调用的安全边界，再在此基础上加入高风险的自我纠错，避免一开始就因代码错误导致训练崩溃。  
- **高吞吐执行**：把代码执行成本从“分钟级”压到“秒级”，使得即使在 14B 参数规模下，也能在普通实验室资源上完成完整的 RL 训练。

### 实验与效果
- **测试任务**：主要在 AIME24、AIME25 两届美国数学邀请赛题目上评估；此外还在对齐（alignment）任务、科学推理基准以及其他工具使用任务上做了迁移测试。  
- **核心指标**：使用 pass@1 评估代码生成的成功率。  
- **结果**：在 AIME24 上取得 80.6% 的 pass@1，在 AIME25 上取得 69.8%。这两个分数均超过了 671B 参数的 DeepSeek‑R1，且 rStar2‑Agent 的回答长度更短、更精炼。  
- **对比基线**：传统 Long CoT（无工具使用）在同样数据上只能达到约 45% 的通过率；使用普通 RL（无 GRPO‑RoC）的大模型约为 55%。  
- **消融实验**：作者分别关闭了（1）GRPO‑RoC 的重抽样机制、（2）思考阶段的奖励惩罚、（3）多阶段 RL 训练。结果显示，去掉任意一项后，AIME 的 pass@1 均下降 10% 以上，验证了每个创新的必要性。  
- **局限性**：论文承认模型仍然依赖于高质量的 SFT 数据，且在极端长代码或需要外部库的任务上仍会出现执行超时或环境不兼容的问题。对更大规模的数学竞赛（如 IMO）尚未进行评估。

### 影响与延伸思考
rStar2‑Agent 的成功展示了“思考‑代码‑反馈”闭环可以在相对小模型（14B）上实现接近百亿参数模型的数学推理水平。此后，多个团队开始探索类似的 Agentic RL 框架，尤其在 **代码生成**、**科学实验模拟** 和 **自动化数据分析** 场景中加入“自我纠错”机制。推测未来的研究会进一步：
- 把 **多模态工具**（如图像、符号计算）纳入同一 RL 环境，形成更通用的“工具箱”。  
- 发展 **层次化奖励**，让模型在更高层次上规划何时切换工具、何时进行抽象推理。  
- 探索 **少样本自适应**，让模型在新领域只需极少的 SFT 即可启动 Agentic RL。

如果想深入了解，可以关注 **Microsoft rStar 项目**的后续代码库更新，以及 **OpenAI**、**DeepMind** 在 “ReAct” 与 “Toolformer” 系列论文中的工具使用策略。

### 一句话记住它
**rStar2‑Agent 用“先思考、后写代码、再根据执行结果自我纠错”的闭环，让 14 B 参数模型在高阶数学上跑出 670 B 级别的表现。**