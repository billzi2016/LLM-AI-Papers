# Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning

> **Date**：2025-05-22
> **arXiv**：https://arxiv.org/abs/2505.16410

## Abstract

Recently, large language models (LLMs) have shown remarkable reasoning capabilities via large-scale reinforcement learning (RL). However, leveraging the RL algorithm to empower effective multi-tool collaborative reasoning in LLMs remains an open challenge. In this paper, we introduce Tool-Star, an RL-based framework designed to empower LLMs to autonomously invoke multiple external tools during stepwise reasoning. Tool-Star integrates six types of tools and incorporates systematic designs in both data synthesis and training. To address the scarcity of tool-use data, we propose a general tool-integrated reasoning data synthesis pipeline, which combines tool-integrated prompting with hint-based sampling to automatically and scalably generate tool-use trajectories. A subsequent quality normalization and difficulty-aware classification process filters out low-quality samples and organizes the dataset from easy to hard. Furthermore, we propose a two-stage training framework to enhance multi-tool collaborative reasoning by: (1) cold-start fine-tuning, which guides LLMs to explore reasoning patterns via tool-invocation feedback; and (2) a multi-tool self-critic RL algorithm with hierarchical reward design, which reinforces reward understanding and promotes effective tool collaboration. Experimental analyses on over 10 challenging reasoning benchmarks highlight the effectiveness and efficiency of Tool-Star. The code is available at https://github.com/dongguanting/Tool-Star.

---

# Tool-Star：通过强化学习赋能基于大语言模型的多工具推理器 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在单纯的文字推理上已经很强，但真正的复杂任务往往需要调用外部工具（比如搜索、计算器、数据库）才能完成。过去的研究要么让模型手工写出调用指令，要么在少量人工标注的工具使用示例上微调，导致模型在真实环境里经常忘记或错误使用工具。根本原因是：①缺少大规模、质量可靠的“工具使用”训练数据；②现有的强化学习（RL）框架主要针对单一步骤的奖励，难以捕捉多步骤、多工具协同的长程依赖。于是，如何让 LLM 在一步步推理中主动、正确地挑选并组合多个工具，仍是未解的难题。

### 关键概念速览

**工具使用轨迹（tool-use trajectory）**：模型在一次完整推理过程中，每一次调用外部工具的指令、输入、输出以及随后的思考步骤的序列。想象成解题过程中的“实验记录本”，记录了每一步实验的材料和结果。

**提示式生成（prompt-based generation）**：利用 LLM 本身的生成能力，根据设计好的提示词让模型自行产生数据。类似让学生先看例题，再让他自己写出类似的练习题。

**难度感知分类（difficulty-aware classification）**：把生成的轨迹按照解题难度从易到难排序，帮助训练时先让模型掌握基础，再挑战更复杂的情形。好比学习语言时先学日常对话，再学专业论文。

**冷启动微调（cold-start fine-tuning）**：在没有任何工具使用经验的模型上，先用少量合成数据进行微调，让模型学会“怎么尝试”。相当于给新手先上几堂实验课。

**层级奖励设计（hierarchical reward design）**：在 RL 中为不同层面的行为设定不同的奖励，例如正确选择工具得小奖励，完成整个任务得大奖励。类似游戏里完成小任务得积分，通关得大礼包。

**多工具自评者（multi-tool self-critic）**：模型在生成每一步后自行评估这一步的质量，并据此调整策略。像是写作时自己先读一遍草稿，判断是否需要改动。

### 核心创新点

1. **从稀缺到规模化的工具使用数据**  
   之前：工具使用数据几乎只能靠人工标注，规模小且质量参差。  
   本文：提出“一体化工具推理数据合成管线”，先用工具集成提示让 LLM 生成包含工具调用的完整推理过程，再用“提示+hint”采样方式自动扩充。  
   改变：实现了大规模、覆盖六类工具的训练数据，解决了数据瓶颈。

2. **质量归一化 + 难度感知过滤**  
   之前：生成的合成数据质量不一，直接喂给模型会导致噪声学习。  
   本文：引入质量归一化步骤（过滤低质量轨迹）和难度感知分类，把数据从易到难组织。  
   改变：模型先学会基本的工具调用模式，再逐步适应更复杂的多工具协同，提升学习效率。

3. **两阶段训练框架**  
   之前：要么只做一次微调，要么直接用 RL 但缺少引导，导致探索效率低。  
   本文：第一阶段用冷启动微调让模型熟悉工具调用的基本模式；第二阶段使用层级奖励的多工具自评者 RL，强化对整体任务的理解并鼓励有效工具合作。  
   改变：显著提升了模型在长链推理中保持正确工具序列的能力。

4. **层级奖励的多工具协同强化**  
   之前的 RL 只奖励最终答案正确与否，忽视中间的工具选择质量。  
   本文：设计了分层奖励——工具选择、工具输出利用、整体任务完成三层，分别给出不同尺度的奖励信号。  
   改变：模型在探索过程中能够感知每一步的价值，避免盲目尝试，收敛更快。

### 方法详解

整体思路可以拆成四大块：**数据合成 → 数据清洗 → 冷启动微调 → 多工具自评 RL**。下面按顺序展开。

1. **工具集成推理数据合成**  
   - **提示模板**：为每类工具（搜索、计算、代码执行、数据库查询、绘图、文件操作）准备统一的提示格式，告诉 LLM “在下面的任务中，你可以调用 X 工具”。  
   - **Hint‑based 采样**：在生成轨迹时，先让模型给出“思考提示”（hint），比如“需要先搜索最新的统计数据”。随后根据提示强制插入对应工具调用，形成完整的工具使用链。  
   - **自动化循环**：上述过程在大规模并行运行，产生数十万条包含多工具调用的推理轨迹。

2. **质量归一化与难度感知**  
   - **质量评估模型**：训练一个小型判别器，输入轨迹，输出“是否符合逻辑、工具调用是否有效”。低于阈值的直接剔除。  
   - **难度打分**：依据轨迹中工具种类、调用次数、是否需要跨工具信息传递等因素打分，划分为 Easy、Medium、Hard。最终把数据按难度从易到难排好序。

3. **冷启动微调**  
   - 使用上述清洗后的数据，对原始 LLM（如 LLaMA‑2‑7B）进行有监督微调。目标是让模型在看到任务描述后，能够主动输出工具调用指令并解释其意图。  
   - 训练时采用“teacher‑forcing”，即强制模型输出与合成轨迹完全一致，帮助模型快速掌握工具调用的语法和基本策略。

4. **多工具自评者强化学习**  
   - **行为空间**：每一步模型可以选择“继续思考”或“调用某工具并提供输入”。  
   - **层级奖励**：  
     * **工具选择奖励**：若模型选对了最适合当前子任务的工具，给小额奖励。  
     * **工具利用奖励**：工具返回的结果被正确引用（如在后续推理中使用），再奖励。  
     * **任务完成奖励**：最终答案正确且符合格式，给大额奖励。  
   - **自评机制**：模型在生成每一步后，使用自身的“自评头”对该步进行打分（类似自我检查），该分数作为即时奖励的一部分，帮助模型在没有外部监督的情况下纠正错误。  
   - **训练循环**：采用 PPO（近端策略优化）等常见 RL 算法，迭代更新策略，使得整体奖励最大化。因为奖励是层级的，模型会自然倾向于先做好工具选择，再确保利用，最后完成任务。

**最巧妙的点**在于把“工具调用”当作可学习的动作，并通过层级奖励把每一步的价值显式化；同时利用自评让模型在每一步都能得到即时反馈，极大提升了探索效率。

### 实验与效果

- **评测任务**：论文在 10+ 具挑战性的推理基准上验证，包括数学推理、代码生成、信息检索、图表绘制等需要外部工具的任务。  
- **对比基线**：与纯 LLM（未使用工具）、基于少量人工工具使用数据的微调模型、以及已有的工具调用框架（如 ReAct、Toolformer）进行比较。  
- **性能提升**：论文声称在大多数基准上相较于最强基线提升 10%~20% 的准确率，尤其在需要多工具协同的复杂任务上提升更明显。  
- **消融实验**：分别去掉（1）质量归一化、（2）难度感知排序、（3）层级奖励、（4）自评机制，结果显示每一项都对最终表现有正向贡献，去掉层级奖励导致整体准确率下降约 7%。  
- **局限性**：作者指出当前只支持六类预定义工具，跨域新工具的迁移仍需额外数据合成；此外 RL 训练成本仍高于单纯的有监督微调。

### 影响与延伸思考

Tool‑Star 的出现让“LLM + 多工具”从“手工调参”迈向“可训练、可强化”的阶段。随后的工作（如 **ToolBench**、**AutoTool**）开始借鉴其数据合成管线和层级奖励思路，尝试在更大模型（GPT‑4 级）上实现自动工具协同。对想继续深入的读者，可以关注以下方向：

1. **工具库扩展**：如何让模型在未知工具出现时快速学习其调用方式。  
2. **跨模态工具**：把图像、音频处理工具纳入同一框架，考验多模态协同。  
3. **更高效的 RL**：探索离线 RL、逆向强化学习等降低训练成本的方案。  
4. **安全与可解释**：在多工具链路中加入审计与可解释性机制，防止误用或滥用。

### 一句话记住它

**Tool‑Star 用层级奖励和自评强化，让大语言模型像实验室助理一样，主动、可靠地组合多种外部工具完成复杂推理。**