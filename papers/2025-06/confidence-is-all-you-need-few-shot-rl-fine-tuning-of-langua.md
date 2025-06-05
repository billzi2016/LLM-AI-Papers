# Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models

> **Date**：2025-06-05
> **arXiv**：https://arxiv.org/abs/2506.06395

## Abstract

Large language models (LLMs) excel at reasoning, yet post-training remains critical for aligning their behavior with task goals. Existing reinforcement learning (RL) methods often depend on costly human annotations or external reward models. We propose Reinforcement Learning via Self-Confidence (RLSC), which uses the model's own confidence as reward signals-eliminating the need for labels, preference models, or reward engineering. Applied to Qwen2.5-Math-7B with only 16 samples per question and 10 or 20 training steps, RLSC improves accuracy by +13.4% on AIME2024, +21.2% on MATH500, +21.7% on Minerva Math, +20.8% on Olympiadbench, and +9.7% on AMC23. RLSC provides a simple, scalable post-training method for inference models, requiring only a small number of samples and unlabelled supervision.

---

# 置信度即一切：少样本强化学习微调语言模型 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在推理上已经很强，但要让它们在特定任务上表现得更精准，仍需要后训练（fine‑tuning）。传统的强化学习（RL）微调往往依赖人工标注的奖励、偏好模型或精心设计的奖励函数，这些资源既昂贵又难以规模化。尤其在数学推理等高精度任务上，获取足够的标注成本高、质量难保，导致很多模型在实际使用时仍然会出现明显错误。于是，如何在几乎不需要额外标签的情况下，让模型自行提升表现，成为一个迫切的技术瓶颈。

### 关键概念速览
**置信度（confidence）**：模型对自己输出的概率估计，数值越高表示越“自信”。可以把它想成学生在答题时对答案的把握程度。  
**强化学习（RL）**：通过奖励信号让模型学会更好地行动，就像训练机器人通过奖赏学会走路。  
**少样本微调（few‑shot fine‑tuning）**：只用极少量的示例（这里是每道题 16 条）就进行参数更新，类似于只给学生几道练习题就让他提升。  
**奖励模型（reward model）**：专门训练来评估模型输出好坏的网络，常用于 RLHF（人类反馈强化学习）。  
**RLHF（Reinforcement Learning from Human Feedback）**：利用人类偏好构造奖励，再用 RL 优化模型的技术。  
**自监督奖励（self‑supervised reward）**：奖励直接来源于模型自身信息，而不是外部标注，像是学生自己给自己的答案打分。  
**数学推理基准（Math benchmarks）**：如 AIME、MATH500、Minerva Math 等，用来衡量模型在数学题目上的解题能力。  

### 核心创新点
1. **奖励来源从外部转向模型自身置信度**  
   之前的 RL 方法需要人工标注或训练专门的奖励模型；这篇论文直接把模型对答案的置信度当作奖励信号。这样省去了标签成本，也避免了奖励模型可能的偏差。  
2. **极少样本、极少步数即可实现显著提升**  
   传统微调往往需要上千甚至上万条标注数据并进行大量梯度更新；作者只用了每题 16 条样本，训练 10–20 步，就在多个数学基准上实现了两位数的提升。  
3. **统一的后训练框架适用于任何推理型 LLM**  
   方法不依赖特定模型结构，只要模型能够输出置信度（即概率分布），就可以直接套用。相比于需要专门构建奖励模型的 RLHF，这种“一键式”方案更易推广。  

### 方法详解
整体思路可以分为三步：**采样 → 置信度评估 → RL 更新**。先让模型在少量示例上生成答案并记录每个答案的置信度，然后把置信度当作即时奖励，最后用经典的强化学习算法（如 PPO）对模型参数做小幅度梯度更新。

**步骤 1：少样本采样**  
对每道数学题，随机抽取 16 条不同的提示（prompt）变体，让模型生成答案。这里的提示可以是不同的题目表述或轻微的上下文变化，目的是让模型在同一任务上产生多样的输出。

**步骤 2：置信度计算**  
模型在生成每个答案时会输出一个概率分布，最高概率对应的数值即为该答案的置信度。直观上，这相当于学生对自己答案的自信程度。作者把这个数值直接作为奖励 r，置信度越高，奖励越大。

**步骤 3：强化学习微调**  
使用近端策略优化（PPO）等安全的 RL 算法，以置信度奖励为目标进行参数更新。因为奖励本身已经是模型内部的信号，梯度方向自然指向“更自信的正确答案”。训练只进行 10–20 步，足以让模型在置信度分布上产生微调，而不会破坏原有的语言能力。

**关键细节**  
- **奖励归一化**：为了防止置信度数值跨度过大，作者对奖励做了简单的归一化处理，使得 PPO 的优势函数保持数值稳定。  
- **KL 散度约束**：在 PPO 中加入对旧策略的 KL 散度约束，确保模型在微调过程中不会偏离原始分布太远，保持推理能力。  
- **无标签的自监督**：整个流程不需要任何人工标注或外部偏好模型，唯一的“监督”来自模型自身的概率输出，这一点在实验设计上极具创新性。

### 实验与效果
- **测试基准**：AIME2024、MATH500、Minerva Math、Olympiadbench、AMC23，都是公开的数学推理评测集合。  
- **对比基线**：与原始 Qwen2.5‑Math‑7B（未微调）以及常见的 RLHF 微调方案相比，RLSC 在所有基准上都有两位数的提升。具体提升幅度为：AIME2024 +13.4%、MATH500 +21.2%、Minerva Math +21.7%、Olympiadbench +20.8%、AMC23 +9.7%。  
- **消融实验**：论文报告了去掉置信度奖励、仅使用随机奖励或增加更多微调步数的实验，结果显示置信度奖励是提升的主要驱动力，步数过多反而会导致性能回落。  
- **局限性**：作者指出方法依赖模型能够给出可靠的置信度；在置信度本身不准确的模型上可能会产生误导性奖励。此外，实验仅在数学推理任务上验证，其他类型的生成任务（如对话、代码）仍需进一步探索。

### 影响与延伸思考
RLSC 把“自信”直接转化为奖励的思路，为后训练提供了一条低成本、易扩展的路径。自发表以来，已有工作尝试把模型内部不确定性（如熵、方差）用于自监督奖励，进一步推广到代码生成、长文摘要等领域。对想深入的读者，可以关注以下方向：① 如何在置信度不可靠的模型上校准奖励；② 将置信度奖励与外部评价（如人类评分）结合的混合策略；③ 在多模态模型中利用视觉置信度进行类似的自监督 RL。  

### 一句话记住它
把模型自己的置信度当作奖励，几乎不需要标注，就能让大语言模型在数学推理上实现显著提升。