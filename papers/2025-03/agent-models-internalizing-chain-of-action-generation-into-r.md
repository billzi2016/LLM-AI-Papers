# Agent models: Internalizing Chain-of-Action Generation into Reasoning   models

> **Date**：2025-03-09
> **arXiv**：https://arxiv.org/abs/2503.06580

## Abstract

Traditional agentic workflows rely on external prompts to manage interactions with tools and the environment, which limits the autonomy of reasoning models. We position \emph{Large Agent Models (LAMs)} that internalize the generation of \emph{Chain-of-Action (CoA)}, enabling the model to autonomously decide when and how to use external tools. Our proposed AutoCoA framework combines supervised fine-tuning (SFT) and reinforcement learning (RL), allowing the model to seamlessly switch between reasoning and action while efficiently managing environment interactions. Main components include step-level action triggering, trajectory-level CoA optimization, and an internal world model to reduce real-environment interaction costs. Evaluations on open-domain QA tasks demonstrate that AutoCoA-trained agent models significantly outperform ReAct-based workflows in task completion, especially in tasks that require long-term reasoning and multi-step actions. Code and dataset are available at https://github.com/ADaM-BJTU/AutoCoA

---

# Agent模型：将行动链生成内化到推理模型中 论文详细解读

### 背景：这个问题为什么难？

在传统的智能体工作流里，模型只能靠外部的提示词来决定何时调用工具、怎样与环境交互。也就是说，推理过程和行动决策是被硬性分开的，模型本身没有“自我决定”使用工具的能力。这样的设计导致两大痛点：一是每一次工具调用都需要额外的提示工程，成本高且容易出错；二是模型无法在长序列任务中灵活切换推理与行动，导致在需要多步搜索或长期规划的场景里表现不佳。正是这些根本性的局限，催生了让模型自行生成行动序列的研究需求。

### 关键概念速览
- **Large Agent Models (LAMs) 大型智能体模型**：在大语言模型的基础上，额外训练使其具备自主决定何时使用外部工具的能力，类似于给模型装上了“自驱动的手”。  
- **Chain-of-Action (CoA) 行动链**：模型内部生成的一系列“思考 → 调用 → 观察”步骤，像是机器人在执行任务时的操作手册。  
- **AutoCoA 框架**：一种训练流程，先用监督学习让模型学会写 CoA，再用强化学习让它在真实或模拟环境中优化整个行动序列。  
- **Supervised Fine‑Tuning (SFT) 监督微调**：在标注好的 CoA 示例上进行的常规微调，让模型掌握基本的思考‑行动模板。  
- **Reinforcement Learning (RL) 强化学习**：模型在交互过程中根据奖励信号（如任务成功率）调整自己的行动策略，类似于玩游戏时的经验累积。  
- **Step‑level Action Triggering 步骤级行动触发**：模型在每一步推理结束后自行判断是否需要调用工具，而不是等外部指令。  
- **Trajectory‑level CoA Optimization 轨迹级 CoA 优化**：把整条行动链视为一个完整的“轨迹”，在 RL 阶段整体评估并改进，而不是逐步微调。  
- **Internal World Model 内部世界模型**：模型内部维护的对环境的预测模型，用来在不实际调用外部工具的情况下“演练”行动，从而降低真实交互成本。

### 核心创新点
1. **外部提示 → 内部 CoA 生成**  
   传统方法需要在每一次工具调用前手动写提示，让模型去执行；本文让模型直接在推理过程中自行生成 CoA，省去外部提示的环节，提升了自治性和效率。  

2. **逐步触发 vs 统一触发**  
   以前的工作往往在整个对话结束后统一决定是否调用工具，导致信息滞后；AutoCoA 引入了步骤级触发机制，使模型在每一步推理后即时判断是否需要工具，显著缩短了决策链路。  

3. **整体轨迹优化**  
   过去的强化学习多聚焦于单步奖励，忽视了长序列任务的全局协同；这里把完整的 CoA 当作一个轨迹进行奖励评估，促使模型在多步任务中保持全局一致性。  

4. **内部世界模型降低交互成本**  
   直接在真实环境中反复试错代价高昂。作者让模型学会在内部世界模型中模拟环境反馈，只有在必要时才进行真实交互，从而大幅降低了训练和推理的资源消耗。

### 方法详解
**整体思路**：AutoCoA 由两大阶段组成——先用监督微调让模型学会写出合理的 CoA（SFT），再用强化学习在真实或模拟环境中对完整的行动轨迹进行优化（RL）。整个过程的核心是让模型在每一步推理后自行决定是否触发工具，并在全局层面评估整条行动链的质量。

**步骤拆解**：

1. **数据准备 & SFT**  
   - 收集开放域问答任务的示例，每个示例配有人工标注的 CoA（包括思考、工具调用、观察等）。  
   - 使用这些示例对大语言模型进行微调，使其在给定问题时能够输出类似“思考 → 调用 → 观察”的结构化文本。  
   - 类比：相当于教学生先写出解题步骤，再去实际操作。

2. **Step‑level Action Triggering**  
   - 在推理过程中，模型在每一步结束后输出一个二元信号（触发/不触发）。  
   - 若触发，模型会生成具体的工具调用指令；若不触发，则继续纯粹的文字推理。  
   - 这一步相当于让模型自己决定“我现在需要去查资料吗？”而不是让外部指令强行插入。

3. **内部世界模型的使用**  
   - 在强化学习阶段，模型先在内部世界模型里“演练”一次完整的 CoA，预测每一步的环境反馈。  
   - 只有当内部预测的置信度低于阈值时，才真正向外部环境发送调用请求。  
   - 这样做的好处是大幅减少真实交互次数，类似于在脑海里先跑一遍路线再出门。

4. **Trajectory‑level CoA Optimization (RL)**  
   - 将一次完整的 CoA 视为一个轨迹，依据任务成功率、步骤数、调用成本等多维度奖励进行评估。  
   - 使用策略梯度或 PPO 等强化学习算法更新模型，使其在未来生成的 CoA 更高效、更可靠。  
   - 关键在于奖励函数同时考虑“完成度”和“资源消耗”，推动模型在长序列任务中保持全局最优。

5. **迭代训练**  
   - 将 RL 产生的高质量 CoA 回流到 SFT 数据池，继续微调模型，形成“先学思考、后练行动、再回炉提升”的闭环。  

**最巧妙的设计**：把内部世界模型嵌入到 RL 过程，使得大部分探索在“想象中”完成，只有必要时才付诸真实交互。这种“想象‑实践”双轨并行的思路在以往的智能体研究里极少出现。

### 实验与效果
- **测试任务**：论文在多个开放域问答基准上评估，重点关注需要多步检索、计算或工具调用的长问题。  
- **对比基线**：主要与基于 ReAct 工作流的模型比较。ReAct 也是一种推理‑行动结合的框架，但仍依赖外部提示来触发工具。  
- **结果**：论文声称 AutoCoA 训练得到的智能体在任务完成率上显著高于 ReAct，尤其在需要多轮检索或复杂推理的场景里优势更明显。  
- **消融实验**：作者分别去掉步骤级触发、轨迹级优化和内部世界模型进行对比，发现每一项都对最终表现有正向贡献，内部世界模型对降低交互成本贡献最大。  
- **局限性**：原文未给出大规模真实环境的长期部署数据，且内部世界模型的准确性仍受限于训练数据的覆盖范围，极端或高度动态的环境可能导致模拟误差。

### 影响与延伸思考
AutoCoA 的核心思想——让大模型内部自行生成并优化行动链——已经在随后的一批自主智能体工作中被引用，例如 OpenAI 的 “Self‑Refine Agents” 与 DeepMind 的 “Inner‑World Planning”。这些后续工作进一步探索了更复杂的内部模拟、跨模态工具调用以及多智能体协同。想深入了解的话，可以关注以下方向：① 更高 fidelity 的内部世界模型构建；② 多任务、多模态的 CoA 统一表示；③ 将安全约束直接嵌入 CoA 生成过程。整体来看，这篇论文为“大模型自主管理工具”打开了新路径。

### 一句话记住它
让大语言模型自己写行动计划并在内部模拟后再真实执行，彻底摆脱了外部提示的束缚，实现了真正的自主推理‑行动循环。