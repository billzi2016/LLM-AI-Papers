# Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents

> **Date**：2025-09-27
> **arXiv**：https://arxiv.org/abs/2509.23045

## Abstract

Large Language Models (LLMs) are increasingly applied to software engineering (SWE), with SWE-bench as a key benchmark. Solutions are split into SWE-Agent frameworks with multi-turn interactions and workflow-based Agentless methods with single-turn verifiable steps. We argue these paradigms are not mutually exclusive: reasoning-intensive Agentless training induces skill priors, including localization, code edit, and self-reflection that enable efficient and effective SWE-Agent adaptation. In this work, we first curate the Agentless training recipe and present Kimi-Dev, an open-source SWE LLM achieving 60.4\% on SWE-bench Verified, the best among workflow approaches. With additional SFT adaptation on 5k publicly-available trajectories, Kimi-Dev powers SWE-Agents to 48.6\% pass@1, on par with that of Claude 3.5 Sonnet (241022 version). These results show that structured skill priors from Agentless training can bridge workflow and agentic frameworks for transferable coding agents.

---

# Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents 论文详细解读

### 背景：这个问题为什么难？

软件工程（SWE）任务本质上是“读代码、定位缺陷、改写并验证”。传统的 LLM（大语言模型）在这类任务上往往只能一次性给出完整答案，缺乏对代码定位和细粒度编辑的专门训练。于是出现了两大流派：  
1. **SWE‑Agent**：让模型在多轮对话中自行决定下一步操作，像一个“智能助理”，但每轮决策都要靠模型自行推理，训练成本高且不易收敛。  
2. **Agentless（工作流）**：把任务拆成若干可验证的单步（如定位‑编辑‑测试），每一步都直接给出答案，易于评估，却缺少跨步骤的全局思考能力。  

这两种思路各有短板：Agent 需要大量交互数据才能学会合理的策略；Agentless 虽然高效，却没有形成“技能先验”，导致在新任务或需要多步推理时表现不佳。论文正是要解决这两者之间的鸿沟，让单步训练也能为多轮 Agent 提供强大的底层能力。

### 关键概念速览

**SWE‑bench**：公开的代码修复基准，提供了真实的开源项目、缺陷描述和对应的测试用例，模型需要在不泄露测试信息的前提下完成修复。  

**Agentless 训练**：不让模型自行决定行动，而是直接给出每一步的输入‑输出对（如“定位文件 → 编辑代码”），类似老师给学生的练习答案。  

**Skill Prior（技能先验）**：模型在大量单步任务上学到的通用能力（定位、编辑、反思），这些能力在后续的多轮 Agent 中可以直接调用。  

**SFT（Supervised Fine‑Tuning）**：在已有的标注数据上继续微调模型，使其更贴合特定任务的语言风格和操作流程。  

**RL（Reinforcement Learning）**：使用奖励信号让模型在特定环节（如代码编辑）上自我优化，这里的奖励是“编辑后测试是否通过”。  

**Pass@1**：在一次尝试中模型生成的代码能否通过全部测试，用来衡量实际部署时的成功率。  

**Verified**：SWE‑bench 的严格评估模式，只计入能够被自动化测试验证的修复，排除人工主观判断。  

### 核心创新点

1. **把 Agentless 训练视作技能先验的构建**  
   - 过去的工作把两种范式当作互斥选项，认为单步训练只能服务于单步评估。  
   - 本文把大量单步数据（定位、编辑、反思）当作“技能库”，在后续的 Agent 中直接调用这些技能。  
   - 结果是模型在多轮交互时不需要从零学习每一步，而是直接使用已经熟练的子技能，显著提升了适配效率。

2. **两阶段训练流程：冷启动 + RL 精炼**  
   - 先在 150 B 条真实代码交互（模拟 SWE‑bench 场景但不包含测试集）上进行大规模监督微调，让模型掌握基本的代码定位和编辑语义。  
   - 再对编辑阶段加入基于二元奖励的强化学习，奖励仅取决于编辑后测试是否通过，采用课程学习逐步提升难度。  
   - 这种先“大量学习通用技能”，后“针对编辑结果微调”的顺序，使模型在编辑质量上实现了跨越式提升。

3. **从 Skill Prior 到可迁移的 SWE‑Agent**  
   - 在得到具备定位/编辑/自省能力的 Kimi‑Dev 后，作者再使用公开的 5 k 条 SWE‑Agent 轨迹进行一次轻量级 SFT。  
   - 这一步相当于给模型加装了“对话指挥官”，让它学会在多轮对话中调度已有的单步技能。  
   - 实验显示，经过这一步的模型在 Pass@1 上达到了 48.6%，与商业模型 Claude 3.5 Sonnet 持平，证明了技能先验的可迁移性。

4. **开源实现与最佳工作流基准**  
   - Kimi‑Dev 作为开源模型，在 SWE‑bench Verified 上取得 60.4% 的成绩，成为工作流类方法的最高记录。  
   - 这不仅提供了可复现的基线，也让社区可以直接在此基础上探索更复杂的 Agent 结构。

### 方法详解

**整体思路**  
模型训练分为三大块：① 大规模 Agentless 预训练（冷启动），② 针对代码编辑的强化学习微调，③ 基于少量 Agent 轨迹的指令层 SFT。整体流程可以想象成“先让模型学会单步技能，再让它学会在对话中调度这些技能”。

**1️⃣ Agentless 冷启动**  
- 数据来源：150 B 条真实的代码交互记录，这些记录模拟了 SWE‑bench 中的定位‑编辑‑测试流程，但刻意剔除了测试集对应的代码，防止泄漏。  
- 训练方式：标准的监督微调（SFT），输入是“任务描述 + 代码上下文”，输出是对应的单步答案（如“文件路径”或“编辑补丁”）。  
- 关键点在于 **任务分解**：作者把完整的修复过程拆成若干固定模板，每个模板对应一种技能（定位、编辑、生成测试），模型只需要在每个模板上学会“看图说话”。

**2️⃣ 编辑阶段的 RL 精炼**  
- 只针对“代码编辑”这一步进行强化学习，因为编辑的成功与否直接决定后续测试能否通过。  
- 奖励函数极其简洁：如果编辑后所有单元测试通过，奖励为 1；否则为 0。没有中间奖励，避免模型过度优化不相关的指标。  
- 采用 **课程学习**：训练初期只给模型提供容易通过的编辑案例，随着训练进行，逐步加入更复杂的 bug，帮助模型稳步提升。  
- 为了加速收敛，作者在后期加入了 **正例加速**（即把已经通过的编辑示例重复喂给模型），让模型更快记住成功的编辑模式。

**3️⃣ 从 Skill Prior 到 SWE‑Agent 的 SFT**  
- 取公开的 5 k 条 SWE‑Agent 轨迹，这些轨迹记录了多轮对话中模型的决策、工具调用和最终代码。  
- 在已经具备定位/编辑/自省能力的 Kimi‑Dev 基础上进行一次轻量级的监督微调，让模型学会在对话中 **调度** 这些技能。  
- 调度方式类似“函数调用”：对话中出现 “请定位 bug” 时，模型内部直接触发定位子模型；编辑指令则调用编辑子模型；自省则让模型回顾上一步的输出并给出改进建议。  
- 这种结构让多轮对话不再是从零学习每一步，而是把每一步映射到已经训练好的单步技能，极大提升了推理效率。

**最巧妙的设计**  
- **技能先验的显式分层**：把通用技能与对话调度层分开训练，使得每层可以独立优化，避免了端到端训练中常见的梯度冲突。  
- **二元奖励的极简 RL**：只用“通过/未通过”两个信号，却能让模型在编辑质量上实现跨越式提升，说明在代码任务中，过于细粒度的奖励往往适得其反。

### 实验与效果

- **评测数据**：SWE‑bench Verified（严格的自动化测试验证）以及 Pass@1（一次尝试成功率）两项指标。  
- **工作流基准**：Kimi‑Dev 在 SWE‑bench Verified 上取得 **60.4%**，是公开工作流方法中最高的成绩。  
- **Agent 基准**：在加入 5 k 条 Agent 轨迹的 SFT 后，Kimi‑Dev 的 Pass@1 达到 **48.6%**，与商业模型 Claude 3.5 Sonnet（同版本）持平。  
- **对比对象**：包括传统的多轮 SWE‑Agent（如 CodeAgent、ChatGPT‑4‑based agents）以及其他工作流模型（如 WizardCoder、CodeLlama‑Instruct）。Kimi‑Dev 在两类基准上均实现了显著领先或持平。  
- **消融实验**：原文报告了三组消融：① 去掉 RL 精炼，编辑成功率下降约 7%；② 只用冷启动不做 SFT，Pass@1 降至约 38%；③ 不使用公开的 5 k 轨迹进行 SFT，Pass@1 下降至约 42%。这些结果说明每一步都对最终性能有实质贡献。  
- **局限性**：作者指出模型仍然依赖于高质量的单步数据，若任务涉及全新语言或框架，Skill Prior 的迁移效果会下降；此外，RL 只针对编辑阶段，定位和自省仍然是纯监督，可能在极端 bug 场景下表现不足。

### 影响与延伸思考

Kimi‑Dev 的出现让社区重新审视 **“单步训练 + 多轮调度”** 的组合路径。自论文发布后，已有几篇工作尝试在不同领域（如数据分析、网络安全）复制这种技能先验的思路，尤其是把 **RL 只用于关键子任务** 的做法被广泛引用。未来的研究可以进一步探索：

- **跨语言 Skill Prior**：把定位/编辑能力扩展到 Rust、Go 等新兴语言，检验先验的通用性。  
- **更细粒度的调度语言**：类似函数调用的 DSL（领域特定语言），让 Agent 能够显式指定使用哪种子技能。  
- **自监督的 Skill Prior**：利用大规模未标注的代码库自动生成定位‑编辑对，降低对人工标注的依赖。

如果想深入了解，可以关注 **OpenAI 的 Code Interpreter**、**DeepMind 的 AlphaCode** 以及 **Meta 的 Code Llama** 的最新进展，它们在技能分层和强化学习方面都有相似的探索。

### 一句话记住它

**把大量单步代码技能训练成“底层工具”，再让多轮 Agent 像调用函数一样调度这些工具，就能让开源模型在软件工程基准上匹配商业大模型。**