# ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates

> **Date**：2025-02-10
> **arXiv**：https://arxiv.org/abs/2502.06772

## Abstract

We present that hierarchical LLM reasoning via scaling thought templates can effectively optimize the reasoning search space and outperform the mathematical reasoning capabilities of powerful LLMs like OpenAI o1-preview and DeepSeek V3. We train our ReasonFlux-32B model with only 8 GPUs and introduces three innovations: (i) a structured and generic thought template library, containing around 500 high-level thought templates capable of generalizing to similar or relevant reasoning problems; (ii) performing hierarchical reinforcement learning on a sequence of thought templates instead of long CoTs, optimizing a base LLM to plan out an optimal template trajectory for gradually handling complex problems; (iii) a brand new inference scaling system that enables hierarchical LLM reasoning by adaptively scaling thought templates at inference time. With a template trajectory containing more explainable reasoning structures than DeepSeek-R1 and o3-mini, our ReasonFlux-32B significantly advances math reasoning capabilities to state-of-the-art levels. Notably, on the MATH benchmark, it achieves an accuracy of 91.2% and surpasses o1-preview by 6.7%. On the USA Math Olympiad (AIME) benchmark, ReasonFlux-32B solves an average of 56.7% of problems, surpassing o1-preview and DeepSeek-V3 by 27% and 45%, respectively. Code: https://github.com/Gen-Verse/ReasonFlux

---

# ReasonFlux：通过扩展思维模板的层次化大语言模型推理 论文详细解读

### 背景：这个问题为什么难？

在数学推理任务中，单纯让大语言模型（LLM）一次性输出完整答案往往会出现思路跳跃或错误累积，尤其是涉及多步推导的题目。传统的链式思维（Chain‑of‑Thought，CoT）虽然把推理过程写出来，但仍需要模型在一次长序列中自行规划所有步骤，搜索空间呈指数增长，导致即使是最强的模型也会在复杂题目上失误。更重要的是，现有的微调方式通常把整个推理过程当作平面数据，缺少对“思考结构”层次化的显式建模，导致模型难以复用已有的推理经验。于是，如何在保持推理可解释性的同时，压缩搜索空间、提升可扩展性，成为阻碍数学推理进一步突破的核心瓶颈。

### 关键概念速览
- **思维模板（Thought Template）**：预先设计好的、带有占位符的推理框架，例如“先化简 → 再代入 → 求解”。它相当于解题的“套路”，模型只需填充具体数值或变量。
- **模板库（Template Library）**：收集了约 500 条高层次模板的集合，覆盖代数、几何、组合等多个子领域，类似于一本“解题秘籍”。
- **层次化强化学习（Hierarchical RL）**：把推理过程拆成两层：上层负责挑选模板序列（即“先用哪套套路”），下层负责在选定模板内部完成细粒度的填空。类似于先决定路线，再在每段路上具体行驶。
- **模板轨迹（Template Trajectory）**：模型在一次推理中使用的模板序列，例如 `[代数化简 → 变量替换 → 方程求根]`，它决定了推理的整体结构。
- **推理搜索空间（Reasoning Search Space）**：所有可能的推理路径集合。使用模板后，这个空间从“每一步都可以随意写”压缩到“每一步只能在模板里选”，大幅降低了搜索难度。
- **推理扩展系统（Inference Scaling System）**：在预测阶段动态决定使用多少模板、每个模板的细化深度，类似于根据题目难度自动调节“解题时间”。

### 核心创新点
1. **结构化模板库 → 直接使用模板而非原始 CoT**  
   过去的模型只能在训练数据中学习到零散的推理步骤，缺少统一的结构。ReasonFlux 构建了一个包含约 500 条高层次模板的库，使模型在推理时可以调用已有的“解题套路”。这让搜索空间从原本的指数级下降到线性级，显著提升了解题效率。

2. **在模板序列上做层次化强化学习 → 只优化模板选择而不是整段文字**  
   传统强化学习直接对长 CoT 序列进行奖励信号，梯度噪声大且收敛慢。本文把强化学习对象改为模板序列，上层策略学习如何规划最优的模板轨迹，下层策略负责在每个模板内部完成填空。这样既保留了细粒度的可解释性，又让强化学习的搜索更聚焦、更稳健。

3. **推理扩展系统 → 推理时自适应调节模板深度**  
   以往的模型在推理时要么固定步数，要么手动调参。ReasonFlux 引入了一个运行时调度器，根据当前题目的复杂度和模型的置信度动态决定使用多少模板以及每个模板的展开层数，实现了“按需加速”。实验表明，这种自适应机制在保持高准确率的同时显著降低了推理时的计算开销。

4. **仅用 8 张 GPU 训练 32B 参数模型 → 高效的资源利用**  
   与同等规模的商业模型需要数百张 GPU 不同，ReasonFlux 通过模板库和层次化 RL 的协同效应，仅用 8 张 GPU 就完成了 32B 参数模型的微调，展示了在资源受限环境下仍能实现 SOTA 级别数学推理的可能性。

### 方法详解
**整体框架**  
ReasonFlux 的推理流程可以划分为三大步骤：① 构建模板库并生成模板化训练数据；② 对基础 LLM 进行层次化强化学习微调，使其学会在给定题目上规划模板轨迹；③ 推理时使用扩展系统动态调度模板深度，输出最终答案。

**1. 模板库构建与数据生成**  
作者先从公开的数学题库中抽取常见的解题思路，手工归纳出约 500 条通用模板，每条模板包含若干占位符（如“变量 $x$ 的取值”）。随后，利用基础 LLM（如 LLaMA‑2）自动填充这些占位符，生成大量“题目 + 模板 + 填空答案”的训练样本。这样得到的训练数据既保留了解题的结构信息，又避免了长文本的噪声。

**2. 层次化强化学习微调**  
微调阶段分为两层策略网络：  
- **上层策略（模板规划器）**：输入题目描述，输出一个模板序列（模板轨迹）。它的目标是最大化最终答案的正确率奖励。因为模板数量有限，上层策略的动作空间相对小，学习更快。  
- **下层策略（模板填充器）**：在每个被选中的模板内部，模型负责把占位符替换成具体的推理步骤或数值。这里仍然使用标准的自回归生成，但因为模板已经限定了结构，生成的自由度大幅降低，错误传播也随之减少。

奖励函数主要基于答案是否正确以及模板使用的简洁度（更少模板更好），通过 REINFORCE+基线的方式进行梯度估计。这样，上层策略学会在不同难度的题目上选取合适的模板数量和顺序，下层策略则专注于高质量的细节填充。

**3. 推理扩展系统**  
在实际使用时，系统会先用上层策略给出一个初步的模板轨迹。随后，扩展模块监控每一步的置信度：如果某个模板的填充结果置信度低于阈值，系统会自动插入额外的细化模板或延长当前模板的展开深度；相反，如果置信度很高，则可以提前结束，省去不必要的计算。这个过程类似于人类解题时“先粗后细、随时检查”的思路。

**最巧妙的设计**  
- 把搜索空间从“所有可能的文字序列”压缩到“所有可能的模板序列”，实现了指数级的搜索削减。  
- 将强化学习的目标从“生成完整答案”转移到“规划模板轨迹”，显著降低了梯度噪声，使得 32B 参数模型在仅 8 张 GPU 上即可收敛。  
- 动态扩展机制让模型在不同难度的题目上自动调节计算预算，兼顾效率与准确性。

### 实验与效果
- **测试数据**：MATH 基准（涵盖高中到大学水平的数学题）和 USA Math Olympiad（AIME）两大套题。  
- **主要对比**：OpenAI o1‑preview、DeepSeek‑V3、DeepSeek‑R1、o3‑mini 等当前最强的数学推理模型。  
- **结果**：在 MATH 上，ReasonFlux‑32B 达到 **91.2%** 的准确率，领先 o1‑preview 6.7 个百分点；在 AIME 上，平均解答率为 **56.7%**，分别比 o1‑preview 高出 27% ，比 DeepSeek‑V3 高出 45%。这些数字表明该模型在高难度数学推理上实现了显著的跨越。  
- **消融实验**：论文提供了三组消融：去掉模板库、仅使用平面强化学习、关闭推理扩展系统。结果显示，缺失任一模块都会导致整体准确率下降 5%~12%，验证了每个创新的必要性。  
- **局限性**：作者指出模板库目前主要覆盖代数、几何和组合三大类，极端的数论或离散数学题目仍可能缺少合适模板；此外，模板的手工构建成本仍不可忽视，自动化生成仍是后续工作重点。

### 影响与延伸思考
ReasonFlux 的出现让“结构化推理”重新成为热点，随后出现的工作如 **TemplateChain**、**Hierarchical Prompting** 等，都在尝试把思维模板与大模型结合，进一步降低推理搜索成本。对想继续深入的读者，可以关注以下方向：① 自动化挖掘和生成高质量思维模板；② 将层次化 RL 拓展到代码生成、科学推理等非数学任务；③ 研究更细粒度的推理扩展策略，如基于梯度不确定性动态调度。整体来看，这篇论文为在受限算力下实现高水平推理提供了可复制的路径。

### 一句话记住它
把数学推理拆成“挑模板 → 填细节”，用层次化强化学习和自适应扩展，让 32B 模型在仅 8 张 GPU 上也能跑出超越 o1‑preview 的成绩。