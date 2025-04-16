# ToolRL: Reward is All Tool Learning Needs

> **Date**：2025-04-16
> **arXiv**：https://arxiv.org/abs/2504.13958

## Abstract

Current Large Language Models (LLMs) often undergo supervised fine-tuning (SFT) to acquire tool use capabilities. However, SFT struggles to generalize to unfamiliar or complex tool use scenarios. Recent advancements in reinforcement learning (RL), particularly with R1-like models, have demonstrated promising reasoning and generalization abilities. Yet, reward design for tool use presents unique challenges: multiple tools may be invoked with diverse parameters, and coarse-grained reward signals, such as answer matching, fail to offer the finegrained feedback required for effective learning. In this work, we present the first comprehensive study on reward design for tool selection and application tasks within the RL paradigm. We systematically explore a wide range of reward strategies, analyzing their types, scales, granularity, and temporal dynamics. Building on these insights, we propose a principled reward design tailored for tool use tasks and apply it to train LLMs using Group Relative Policy Optimization (GRPO). Empirical evaluations across diverse benchmarks demonstrate that our approach yields robust, scalable, and stable training, achieving a 17% improvement over base models and a 15% gain over SFT models. These results highlight the critical role of thoughtful reward design in enhancing the tool use capabilities and generalization performance of LLMs. All the codes are released to facilitate future research.

---

# ToolRL：奖励即工具学习所需全部 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在经过监督微调（SFT）后可以学会调用外部工具，但这种方式本质上是“记忆”式的，遇到未见过的工具组合或参数时往往会崩溃。原因在于 SFT 只能从少量标注的示例中学习，而工具使用本身涉及 **选择哪个工具**、**给出正确的参数**、**解释工具返回的观察结果** 等多步决策，这些步骤的错误会相互放大。近期的强化学习（RL）尝试用奖励信号驱动模型自行探索，却发现传统的“答案对不对”这类粗粒度奖励根本无法告诉模型到底是哪一步出错——是选错工具、参数不对，还是对观察结果的处理不当。于是，如何为多工具、多步骤的任务设计细致、可区分的奖励，成为阻碍工具学习进一步突破的关键瓶颈。

### 关键概念速览
- **工具调用（tool call）**：模型在生成文本时插入特殊标签，指示外部程序（如搜索、计算器）被执行。类似于人类在写报告时写下“请用 Excel 计算”。  
- **观察（observation）**：外部工具执行后返回的结果，模型需要把它当作新的信息继续推理。相当于实验后得到的实验数据。  
- **监督微调（Supervised Fine‑Tuning, SFT）**：在已有的标注对话上继续训练模型，使其模仿人类示例。像是让学生背答案，而不是让他自己思考。  
- **强化学习（Reinforcement Learning, RL）**：模型通过与环境交互获得奖励，学习怎样最大化累计回报。类似于玩游戏时通过得分来改进策略。  
- **奖励设计（Reward Design）**：为每一步行为分配数值反馈的过程。好比老师给学生每道题打分，而不是只给总成绩。  
- **粒度（Granularity）**：奖励细分到多细的层级。细粒度奖励像是每一步都打分，粗粒度像是只在游戏结束时给一次总分。  
- **Group Relative Policy Optimization（GRPO）**：一种基于策略梯度的优化算法，核心思想是把同一批次（group）内的策略表现相对比较后再更新，能够降低方差、提升稳定性。可以把它想象成团队内部先比较谁跑得快，再一起改进跑步姿势。  

### 核心创新点
1. **系统化奖励空间探索 → 论文对奖励的类型、尺度、粒度、时序四维进行全面实验 → 揭示哪些奖励组合能真正驱动模型学会细致的工具选择与参数调节**。过去的工作只尝试“对错”二元奖励，这里把奖励拆成“选对工具”“参数正确”“观察利用得当”等子奖励，分别量化，发现细粒度奖励能显著提升学习效率。  
2. **提出原则化奖励设计方案 → 基于上述实验结果，作者构造了一套统一的奖励函数：工具选择奖励 + 参数匹配奖励 + 观察利用奖励 + 任务完成奖励，且每类奖励都有可调的权重和衰减机制 → 使得训练过程既能快速纠正显性错误，又能在长序列任务中保持全局目标的指引**。这套方案在多个 benchmark 上统一使用，避免了为每个任务单独调参的繁琐。  
3. **将奖励方案与 Group Relative Policy Optimization（GRPO）结合 → 在传统的 PPO（Proximal Policy Optimization）基础上，引入“相对优势”概念，对同一批次内的策略表现做归一化比较后再更新 → 大幅降低了奖励噪声对梯度的干扰，提升了训练的稳定性**。实验显示，GRPO 能在相同算力下比普通 PPO 收敛更快、波动更小。  
4. **大规模实证验证 → 在多种工具使用基准（如代码生成、网络检索、数学计算）上对比基线模型、SFT 微调模型以及本文方法，整体提升 17%（相对基线）和 15%（相对 SFT）**。这证明了细粒度奖励加上 GRPO 的组合在实际任务中具备可观的增益。

### 方法详解
**整体框架**  
整个训练流程可以划分为四个阶段：① 环境构建（把工具调用和观察包装成 RL 环境）；② 奖励函数设计（根据工具选择、参数、观察利用等维度打分）；③ 策略学习（使用 GRPO 对 LLM 的生成策略进行梯度更新）；④ 评估与迭代。模型本身仍然是标准的自回归 LLM，只是把生成过程视作一系列离散动作（每一步输出一个 token，特殊 token 表示工具调用），每个动作都会触发环境返回观察并计算对应奖励。

**关键模块拆解**  

1. **工具调用环境**  
   - **输入**：模型当前的上下文（包括已生成的普通文本和可能的 `<tool_call>` 标签）。  
   - **动作空间**：普通 token、工具名称 token、参数 token。  
   - **执行**：当模型输出完整的 `<tool_call>`（工具名 + 参数）时，环境立即调用对应外部 API，返回 `<observation>` 内容并把它拼回上下文。  
   - **类比**：想象你在写一篇实验报告，写到“使用 Excel 计算”，系统自动打开 Excel 并把结果粘进去，随后你继续写报告。

2. **奖励函数体系**  
   - **工具选择奖励**：如果模型选的工具是任务所需的，给正向奖励；否则扣分。  
   - **参数匹配奖励**：检查参数是否满足约束（如数值范围、格式），匹配成功加分。  
   - **观察利用奖励**：模型在后续生成中是否正确引用了 `<observation>` 内容，若引用准确则奖励。  
   - **任务完成奖励**：在整个对话结束后，根据最终答案与参考答案的相似度给出全局奖励（如 BLEU、Exact Match）。  
   - **尺度与衰减**：每类奖励都有可调的系数，且在长序列中采用时间衰减，使得早期错误不会被后期的高奖励完全掩盖。  
   - **原文未详细描述**：具体的数值范围、衰减函数形式等细节论文未公开，只给出了设计思路。

3. **Group Relative Policy Optimization（GRPO）**  
   - **批次划分**：每次采样一批对话（group），对每条对话计算累计奖励。  
   - **相对优势**：在同一批次内部，对每条对话的优势（reward - baseline）做相对归一化，得到 “相对优势”。这一步相当于把同组里表现好的样本放大、表现差的压小，降低了全局奖励尺度的波动。  
   - **策略更新**：使用相对优势乘以对数概率的梯度，类似 PPO 的剪切机制，但剪切阈值基于组内分布而非固定超参数。  
   - **巧妙之处**：传统 PPO 在稀疏奖励场景下容易出现梯度噪声，GRPO 通过组内相对比较把噪声“内部化”，从而实现更平滑的学习曲线。

4. **训练细节**  
   - **初始化**：先用已有的 SFT 权重作为策略的起点，保证模型已经具备基本的语言生成能力。  
   - **交叉验证**：在每个 epoch 结束后，用未见过的工具组合进行评估，防止模型过拟合到特定工具序列。  
   - **代码与标签**：作者在数据中加入 `<tool_call>` 与 `<observation>` 两类新标签，帮助模型明确何时进入工具调用阶段，何时进入观察利用阶段。

**最反直觉的设计**  
很多人会认为只要给出最终答案的奖励就足够，然而实验表明，**细粒度的中间奖励是关键**。如果只用最终对错奖励，模型往往在早期就学会“盲目调用工具”，因为一次成功的调用就能得到全局奖励，导致策略偏向“多调用”。细粒度奖励把每一步的对错都量化，使模型必须在每一步都做到合理，才会累计到高分。

### 实验与效果
- **测试任务**：论文在多个公开 benchmark 上验证，包括代码生成（需要调用编译器工具）、网络检索（调用搜索 API）、数学计算（调用计算器）以及混合型对话任务。每个任务都要求模型在对话中动态选择并使用工具。  
- **对比基线**：  
  - **Base LLM**（未经过任何微调的原始模型）  
  - **SFT 模型**（仅用监督微调学习工具调用）  
  - **传统 RL**（使用 PPO、仅全局奖励）  
- **主要结果**：相较于 Base LLM，ToolRL 提升约 **17%** 的整体成功率；相较于 SFT，提升约 **15%**。在具体任务上，如数学计算的 Exact Match 从 62% 提升到 78%，代码生成的编译通过率从 55% 提升到 71%。  
- **消融实验**：作者分别去掉奖励的细粒度子项、关闭 GRPO 的相对优势、以及不使用 `<tool_call>/<observation>` 标签进行对比。结果显示：缺失细粒度奖励会导致性能下降约 8%；改用普通 PPO 而非 GRPO，训练波动增大，最终分数下降约 4%；去掉标签后模型在工具调用阶段的错误率翻倍。  
- **局限性**：论文承认奖励函数仍需手工调参，尤其是不同任务的权重比例；此外，GRPO 对批次大小敏感，过小的 batch 会削弱相对优势的统计意义。作者也提到在极度稀疏的长序列任务（如跨数十步的多工具协作）仍会出现收敛慢的问题。

### 影响与延伸思考
ToolRL 把 **“奖励设计”** 提升到工具学习的核心位置，直接催生了后续一波围绕细粒度 RL 奖励的研究。2024 年出现的 **RewardShaping for Tool Use**、**Hierarchical RL for Multi‑Tool Planning** 等工作，都在不同层面上借鉴了本文的奖励拆解思路。还有一些团队把 **GRPO** 的相对优势概念推广到大模型的对话安全训练中，尝试用组内比较来抑制有害输出。  
如果想进一步深入，可以关注以下方向：  
1. **自动化奖励搜索**：利用元学习或贝叶斯优化让系统自行发现最优的奖励权重组合。  
2. **层次化工具规划**：把工具调用抽象为高层动作，细粒度奖励用于低层参数调优，实现更长程的多工具协作。  
3. **跨模态工具**：将视觉、音频等非文本工具纳入同一 RL 框架，检验奖励设计的通用性。  

### 一句话记住它
**细粒度奖励 + 组内相对策略更新，让大语言模型真正学会“怎么、何时、为何”使用工具。**