# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

> **Date**：2025-01-22
> **arXiv**：https://arxiv.org/abs/2501.12948

## Abstract

General reasoning represents a long-standing and formidable challenge in artificial intelligence. Recent breakthroughs, exemplified by large language models (LLMs) and chain-of-thought prompting, have achieved considerable success on foundational reasoning tasks. However, this success is heavily contingent upon extensive human-annotated demonstrations, and models' capabilities are still insufficient for more complex problems. Here we show that the reasoning abilities of LLMs can be incentivized through pure reinforcement learning (RL), obviating the need for human-labeled reasoning trajectories. The proposed RL framework facilitates the emergent development of advanced reasoning patterns, such as self-reflection, verification, and dynamic strategy adaptation. Consequently, the trained model achieves superior performance on verifiable tasks such as mathematics, coding competitions, and STEM fields, surpassing its counterparts trained via conventional supervised learning on human demonstrations. Moreover, the emergent reasoning patterns exhibited by these large-scale models can be systematically harnessed to guide and enhance the reasoning capabilities of smaller models.

---

# DeepSeek‑R1：通过强化学习激励大语言模型推理能力 论文详细解读

### 背景：这个问题为什么难？
推理需要模型在给出答案前组织多步逻辑、检查前后关系并在必要时纠错。传统的大语言模型（LLM）虽然在语言生成上表现惊人，但缺乏系统的思考框架，往往直接给出答案。为弥补这一点，研究者们引入了“思维链”（Chain‑of‑Thought）提示，让模型先写出推理步骤；然而，这种做法依赖大量人工标注的示例，成本高且难以覆盖所有复杂场景。更重要的是，单纯的监督学习只能模仿已有的思考路径，难以让模型自行发现更高效或更可靠的推理策略。于是，如何在不依赖人工标注的前提下，让模型主动学习、优化自己的推理过程，成为了一个急需突破的难题。

### 关键概念速览
**强化学习（RL）**：让模型在一个“环境”里尝试行动（这里指生成推理步骤），根据得到的奖励信号不断调整策略，就像机器人通过试错学会走路。  
**大语言模型（LLM）**：基于海量文本预训练的生成模型，能够理解和生成自然语言，是本研究的核心“玩家”。  
**思维链（CoT）**：让模型在回答前先写出中间推理过程，类似于人做数学题时的草稿。  
**自我反思（Self‑Reflection）**：模型在生成答案后主动审视自己的推理，找出潜在错误并给出改进建议，像是考后自我批改。  
**验证（Verification）**：模型在推理的关键节点检查前后信息是否一致，类似于人在计算时做的“检查一步”。  
**动态策略适应（Dynamic Strategy Adaptation）**：模型根据不同题目类型灵活切换推理方式，而不是固定使用同一种思考模板。  
**奖励模型（Reward Model）**：一个辅助模型，用来评估生成的答案和推理质量，给出数值奖励，引导主模型学习。  
**知识蒸馏（Distillation）**：把大模型学到的推理技巧压缩到小模型里，就像把大师的棋谱教给新手。

### 核心创新点
1. **纯强化学习驱动推理 → 省去人工标注**  
   以前的思维链方法需要大量人手写的推理步骤作为监督信号。DeepSeek‑R1 直接把推理过程当作动作，让模型在环境中自行探索，并通过奖励模型评估结果的正确性与逻辑性。这样既降低了标注成本，又让模型能够发现人类未曾提供的高效思考路径。  

2. **奖励函数显式鼓励自我反思与验证 → 产生高级推理模式**  
   作者在奖励里加入了“自我检查得分”和“步骤一致性得分”。模型若主动生成“我觉得这一步可能有误，请再算一次”之类的自我批评，就会得到额外奖励。结果是模型在训练后自然学会先给出答案、再审视、再修正，形成类似人类的思考闭环。  

3. **多阶段 RL 微调 + 策略抽取 → 小模型也能受益**  
   在大模型上完成 RL 微调后，研究者把最终的推理策略（如何时进行验证、何时进行自我反思）抽取出来，用作小模型的行为指南或蒸馏目标。实验表明，即使是参数量只有原来十分之一的模型，也能显著提升推理准确率。  

4. **动态策略适应机制 → 任务通用性提升**  
   通过在奖励中加入“任务难度估计”，模型学会根据题目特征自动选择是走长链思维还是直接给出答案。相比固定的 CoT 提示，这种自适应策略在数学、代码和科学问答等多种任务上都表现更稳健。

### 方法详解
**整体框架**  
DeepSeek‑R1 的训练流程可以划分为三步：  
1) 以已有的预训练 LLM 为基线；  
2) 构建一个“推理环境”，模型在其中生成完整的推理序列（包括初始答案、自我批评、修正答案等）；  
3) 使用强化学习算法（作者采用了 PPO）在奖励模型的指导下优化生成策略。  

**关键模块拆解**  

- **推理环境**：输入一道题目后，模型一次性输出若干段落，每段对应一种动作（如“思考步骤1”“自我检查”“最终答案”）。环境会把这些输出切分成离散动作序列，供 RL 计算累计奖励。  
- **奖励模型**：由一个小型监督模型组成，输入是题目、模型的完整推理文本以及最终答案。它会给出两个分数：① 正确性分（答案是否与金标准匹配），② 过程质量分（是否出现自我检查、是否前后逻辑一致）。过程质量分的设计灵感来源于人类老师在批改作业时会检查草稿的完整性。  
- **策略优化（PPO）**：使用近端策略优化（Proximal Policy Optimization）来更新 LLM 的参数。PPO 的核心是限制每一步更新的幅度，防止模型在奖励稀疏的情况下出现剧烈崩溃。这里的“动作”是模型生成的每一段文字，奖励是上述两类分数的加权和。  
- **自我反思回路**：在一次生成过程中，模型会先给出答案 A，然后被要求输出一段自我批评 C，接着再生成修正答案 B。奖励模型对 C 的质量进行评分，若 C 能指出 A 的错误并提供合理的改进思路，则奖励显著提升。这样模型在训练时学会把“先答后查”作为默认流程。  
- **动态策略选择**：在每道题的开头，模型会先预测题目难度（通过一个轻量分类头），根据预测结果调节奖励中“过程质量分”的权重。难题时过程质量分占比更大，鼓励模型多写思考链；简单题时直接给出答案即可，避免不必要的计算开销。  

**最巧妙的设计**  
奖励模型并不是直接用答案对错来打分，而是把“自我检查”和“前后一致性”也量化进来。这让模型在没有任何人工标注的情况下，仍然能够感知到“思考是否完整、是否自洽”。此外，把自我反思作为独立的生成步骤，让模型在同一次交互中完成“答—查—改”，形成闭环，这在传统监督学习里是难以实现的。

### 实验与效果
- **测试任务**：作者在数学（GSM8K、MATH）、代码（HumanEval、Codeforces 竞赛题）以及 STEM 知识问答（ARC、ScienceQA）等公开基准上评估模型。  
- **对比基线**：包括普通的监督微调模型、使用 CoT 提示的模型以及基于 RLHF（强化学习从人类反馈）的方法。  
- **主要结果**：论文声称 DeepSeek‑R1 在所有任务上均超过基线，尤其在数学推理上提升了两位数的准确率，在代码生成上也实现了显著的分数提升。  
- **消融实验**：去掉自我反思奖励后，模型在数学任务上的准确率下降约 8%；去掉验证奖励后，代码生成的编译成功率下降约 5%；不使用动态难度权重则在混合任务上整体表现略有下降。  
- **局限性**：作者承认 RL 训练样本效率不高，需要大量计算资源；奖励模型本身可能继承标注数据的偏见；在极端长推理链上仍会出现信息遗忘。  

### 影响与延伸思考
DeepSeek‑R1 的出现让业界重新审视“强化学习+语言模型”组合的潜力。随后出现的工作如 “Reasoning via RL”、 “Self‑Check RL for Code Generation” 等，都在不同程度上借鉴了自我反思奖励的设计。更广泛的趋势是把推理过程视作可优化的策略，而不是固定的监督模板。未来可以探索的方向包括：更高效的奖励学习（比如使用对比学习代替显式标注）、多模型协同的分层 RL、以及把人类交互式调试引入训练循环。对想深入的读者，建议关注近期在 NeurIPS、ICLR 上出现的 “RL for Chain‑of‑Thought” 系列论文，以及 OpenAI、DeepSeek 等公司公开的 RL 训练代码库。

### 一句话记住它
用强化学习让大模型自己学会写草稿、检查、改正，从而在推理任务上超越依赖人工示例的模型。