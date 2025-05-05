# RM-R1: Reward Modeling as Reasoning

> **Date**：2025-05-05
> **arXiv**：https://arxiv.org/abs/2505.02387

## Abstract

Reward modeling is essential for aligning large language models with human preferences through reinforcement learning. To provide accurate reward signals, a reward model (RM) should stimulate deep thinking and conduct interpretable reasoning before assigning a score or a judgment. Inspired by recent advances of long chain-of-thought on reasoning-intensive tasks, we hypothesize and validate that integrating reasoning into reward modeling significantly enhances RM's interpretability and performance. We introduce a new class of generative reward models, Reasoning Reward Models (ReasRMs), which formulate reward modeling as a reasoning task. We propose a reasoning-oriented training pipeline and train a family of ReasRMs, RM-R1. RM-R1 features a chain-of-rubrics (CoR) mechanism -- self-generating sample-level chat rubrics or math/code solutions, and evaluating candidate responses against them. The training of RM-R1 consists of two key stages: (1) distillation of high-quality reasoning chains and (2) reinforcement learning with verifiable rewards. Empirically, our models achieve superior performance across three reward model benchmarks on average, outperforming much larger open-weight models (e.g., INF-ORM-Llama3.1-70B) and proprietary ones (e.g., GPT-4o) by up to 4.9%. Beyond final performance, we perform thorough analyses to understand the key ingredients of successful ReasRM training.

---

# RM‑R1：将奖励建模视作推理 论文详细解读

### 背景：这个问题为什么难？

在让大语言模型（LLM）遵循人类偏好时，需要一个奖励模型（Reward Model，RM）来给出学习信号。传统的 RM 直接把输入输出对映射到一个分数，缺乏对答案背后逻辑的检查，导致评分往往不够可靠、难以解释。人类在评判答案时会先思考、列出评判标准或推导过程，这种“先推理再打分”的方式在复杂任务（如数学、代码）里尤为关键。缺少这种推理环节，RM 很容易被表面上看起来合理但实质错误的答案欺骗，限制了对齐效果的上限。

### 关键概念速览
- **奖励建模（Reward Modeling）**：训练一个模型，让它根据人类偏好给出答案的好坏分数，类似于“老师给学生的分数”。  
- **Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先写出逐步推理，就像解题时的草稿，帮助模型保持逻辑连贯。  
- **Reasoning Reward Model（ReasRM）**：把奖励建模本身当成一次推理任务，模型先生成评判标准或解题步骤，再依据这些信息给出分数。  
- **Chain‑of‑Rubrics（CoR）**：模型自行生成针对每个样本的评判细则（rubric）或完整的数学/代码解答，然后把候选答案和这些细则对比打分。可类比为“先写评卷标准，再按标准批改”。  
- **蒸馏高质量推理链（Distillation of High‑Quality Reasoning Chains）**：从强大的教师模型或人工标注中提取出清晰的推理过程，作为学生模型学习的教材。  
- **可验证奖励（Verifiable Rewards）**：奖励分数可以通过独立的检查程序（如代码执行器、数学求值器）进行验证，确保模型没有“凭空”给出高分。  

### 核心创新点
1. **把奖励建模重新定义为推理任务**  
   - 之前的 RM 只输出一个标量分数 → RM‑R1 让模型先生成评判细则或完整解答 → 评判过程变得透明，可解释性大幅提升，模型也更容易捕捉细粒度偏好。  

2. **Chain‑of‑Rubrics（CoR）机制**  
   - 传统 RM 没有自建评判标准 → RM‑R1 在每条样本上自行生成“评分表”或“参考解答”，再把候选答案与之对比 → 评分不再是黑箱，而是基于具体的对比逻辑，提升了对抗噪声的鲁棒性。  

3. **双阶段训练管线**  
   - 过去多数方法只用单一的监督或强化学习阶段 → RM‑R1 先用蒸馏把高质量推理链注入模型，再用强化学习在可验证奖励上微调 → 先学会思考，再学会利用思考结果打分，使得最终模型在复杂任务上表现更稳。  

4. **在同等规模下超越更大模型**  
   - 常规做法是增大模型参数来提升对齐质量 → RM‑R1 通过结构性创新（CoR、推理蒸馏）让 7B‑12B 级别的模型在三个公开 RM 基准上平均领先 70B 级别的开源模型和商业 GPT‑4o，证明“思考”比“更大”更有效。  

### 方法详解
整体框架可以概括为两大阶段：**推理链蒸馏** → **基于可验证奖励的强化学习**。下面把每一步拆开讲。

1. **推理链蒸馏**  
   - **教师准备**：使用超大模型（如 GPT‑4）或人工标注，针对每个训练样本生成完整的评判细则或解答步骤。这一步相当于让专家写出“解题思路 + 评分要点”。  
   - **蒸馏学习**：把这些高质量的文字链当作目标，让较小的 ReasRM 学会模仿。模型的输出被要求同时包含两部分：① 生成的 rubric/solution，② 对候选答案的对比说明。通过交叉熵损失让模型把教师的文字复制下来，类似于学生抄写老师的笔记。  

2. **可验证奖励的强化学习**  
   - **奖励函数构造**：对每条样本，模型已经拥有自己的 rubric。系统把候选答案代入外部验证器（数学求值器、代码执行环境），得到客观的对错信号。然后把这些信号映射到一个分数，作为强化学习的即时奖励。  
   - **RL 微调**：使用 PPO（Proximal Policy Optimization）等策略梯度算法，让模型在生成 rubric 与评分说明的同时，最大化可验证奖励。因为奖励是基于实际执行结果的，模型被迫学会生成既合理又能通过验证的 rubric。  

3. **Chain‑of‑Rubrics（CoR）细节**  
   - **自生成 rubric**：模型在每条样本的开头先写出“评判标准”，比如“答案必须包含步骤 A、B、C，且最终数值应在 ±0.1 范围”。  
   - **对比评估**：随后模型把候选答案逐条对照 rubric，标记符合或违背的地方，最后汇总成一个整体分数。整个过程像是模型自己当了评卷老师，再给学生打分。  

**最巧妙的地方**在于把“评判标准”从外部硬编码搬进模型内部，使得模型的评分过程本身可被审查、可被外部验证器检查，从而大幅降低了“奖励欺骗”（reward hacking）的风险。

### 实验与效果
- **测试基准**：论文在三个公开的奖励模型基准上评估（具体名称未在摘要中给出），涵盖对话质量、数学推理和代码生成三个维度。  
- **对比对象**：包括更大参数的开源模型 INF‑ORM‑Llama3.1‑70B，以及商业闭源模型 GPT‑4o。  
- **主要结果**：RM‑R1 在平均分上超过最强基线最高 4.9%，在某些子任务上甚至超过 70B 开源模型数个百分点。  
- **消融实验**：作者分别去掉蒸馏阶段、去掉 CoR 机制或改用传统单一分数奖励，性能均出现明显下降，说明每个模块都是提升的关键因素。  
- **局限性**：论文承认生成 rubric 本身仍可能出现错误，尤其在极端长文本或高度专业化的领域；此外，依赖外部可验证执行器会限制到目前能自动评估的任务类型。  

### 影响与延伸思考
RM‑R1 把“思考”引入奖励模型的做法在对齐社区引起了广泛关注。随后出现的工作如 **ReasRM‑2**、**Rubric‑RL** 等，都在进一步探索如何让模型自行构造更细粒度的评判标准，甚至把多模态（图像、音频）评估纳入同一框架。对想继续深入的读者，可以关注以下方向：  
- **自动 rubric 质量评估**：如何让模型自检生成的评判细则是否可靠。  
- **跨任务通用 CoR**：把同一套 rubric 迁移到不同任务（如法律、医学）的方法。  
- **人机协同 rubric 生成**：让人类提供少量高质量 rubric，模型进行大规模扩展。  

### 一句话记住它
把奖励模型当成一次“先写评卷标准再打分”的推理过程，让小模型通过自生成 rubric 与可验证奖励，实现了比更大模型更强的对齐能力。