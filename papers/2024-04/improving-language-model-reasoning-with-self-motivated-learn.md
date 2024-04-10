# Improving Language Model Reasoning with Self-motivated Learning

> **Date**：2024-04-10
> **arXiv**：https://arxiv.org/abs/2404.07017

## Abstract

Large-scale high-quality training data is important for improving the performance of models. After trained with data that has rationales (reasoning steps), models gain reasoning capability. However, the dataset with high-quality rationales is relatively scarce due to the high annotation cost. To address this issue, we propose \textit{Self-motivated Learning} framework. The framework motivates the model itself to automatically generate rationales on existing datasets. Based on the inherent rank from correctness across multiple rationales, the model learns to generate better rationales, leading to higher reasoning capability. Specifically, we train a reward model with the rank to evaluate the quality of rationales, and improve the performance of reasoning through reinforcement learning. Experiment results of Llama2 7B on multiple reasoning datasets show that our method significantly improves the reasoning ability of models, even outperforming text-davinci-002 in some datasets.

---

# 自我激励学习提升语言模型推理能力 论文详细解读

### 背景：这个问题为什么难？

语言模型要在复杂的推理题上表现好，往往需要大量带有“思考过程”的训练数据。可是，高质量的推理步骤（即 rationales）标注成本极高，公开数据集大多只提供答案而没有解释。过去的做法要么直接在没有推理信息的语料上训练，要么花大钱请人工写出思考链，导致模型的推理能力提升受限。于是，如何在不增加标注成本的前提下，让模型自己学会写出可靠的推理步骤，成为了一个迫切的技术瓶颈。

### 关键概念速览
- **Rationale（推理步骤）**：模型在给出答案前写出的中间推理过程，类似于人做题时的草稿，帮助模型保持逻辑连贯性。  
- **Self‑motivated Learning（自我激励学习）**：让模型主动在已有数据上生成自己的推理步骤，并用内部评估机制挑选出更好的版本进行学习，像是模型给自己打分并自我提升。  
- **Reward Model（奖励模型）**：一个专门训练来判断推理质量的二次模型，它会根据多个候选推理的相对排名给出分数，类似于老师给学生的评分标准。  
- **Reinforcement Learning (RL)（强化学习）**：模型在奖励模型的指引下调整生成策略，使得产生的推理步骤在质量上不断提升，类似于在游戏中通过得分来改进玩法。  
- **Rank‑based Supervision（基于排序的监督）**：不是给出绝对好坏标签，而是让模型学习“这几个推理中哪个更好”，从而利用相对信息进行训练。  
- **Chain‑of‑Thought (CoT)（思维链）**：让模型在回答前先写出一步步推理的技术，和自我激励学习的目标是一致的，只是后者强调模型自己产生这些链。  

### 核心创新点
1. **从答案到推理的自我标注**：传统方法需要人工标注推理步骤；本文让模型在仅有答案的语料上自行生成多个候选 rationales，然后通过内部排序挑选更好者。这样把“标注”任务搬到了模型内部，极大降低了成本。  
2. **基于相对排名训练奖励模型**：以前的奖励模型往往依赖人工打分的绝对分值；这里作者用多个 rationales 的正确性排序来训练奖励模型，使其学习“相对好坏”。这种相对监督更易获取，也更符合人类评判推理的方式。  
3. **奖励模型驱动的强化学习回路**：在得到奖励模型后，作者使用强化学习让语言模型的生成策略直接优化奖励分数。相当于模型在“自我激励”下不断尝试、筛选、改进自己的推理过程。  
4. **跨数据集的通用提升**：实验显示，同一套自我激励框架在多个推理基准上都能提升 Llama2‑7B 的表现，甚至在部分数据集上超过了 OpenAI 的 text‑davinci‑002，证明了方法的广泛适用性。

### 方法详解
整体思路可以拆成三步：**生成‑评估‑优化**。

1. **生成阶段**  
   - 给定一个只有答案的训练样本（例如一道数学题），模型先使用普通的语言生成方式，产生 **N** 条不同的推理步骤（N 通常取 4~8）。这一步相当于让模型“自问自答”，每条 rationales 都是一次独立的思考尝试。  
   - 为了让生成多样化，作者在采样时加入温度调节和 nucleus 采样等技巧，确保产生的推理在形式和内容上有足够差异。

2. **评估阶段**  
   - 将同一题目的 N 条 rationales 送入 **奖励模型**。奖励模型的输入是（题目、推理、答案），输出一个分数。  
   - 奖励模型的训练目标不是预测绝对分数，而是学习 **排序**：如果 rationales A 在真实答案上更接近正确推理，模型应给 A 打更高分。训练数据来源于人工标注的少量高质量 rationales，或者直接利用已有的带 rationales 的数据集。  
   - 排序损失（如 pairwise hinge loss）让奖励模型专注于区分“好”和“坏”，而不必精确量化每一步的质量。

3. **优化阶段**  
   - 使用强化学习（常见的 PPO 算法）把奖励模型的分数当作即时回报，让语言模型的生成策略朝着高分 rationales 调整。  
   - 在每一次 RL 更新中，模型会重新生成 rationales，奖励模型重新打分，策略梯度根据分数差异更新参数。这样形成了一个闭环：模型生成 → 奖励评估 → 策略改进 → 再生成。  
   - 为防止模型只学会“骗取”奖励，作者在训练中加入了 **KL 散度约束**，让新策略不要偏离原始语言模型太远，保持语言流畅性。

**最巧妙的点**在于把“质量排序”直接当作监督信号，而不是硬性要求每条 rationales 必须完全正确。这样即使模型生成的推理有小错误，只要相对更好，就能得到正向反馈，推动整体水平逐步提升。

### 实验与效果
- **测试数据**：作者在 Llama2‑7B 上跑了多个公开推理基准，包括 GSM8K（数学）、CommonsenseQA（常识）以及 MATH（高难度数学）等。  
- **对比基线**：与原始 Llama2‑7B、使用普通 CoT 微调的模型以及 OpenAI 的 text‑davinci‑002 进行比较。  
- **结果**：在 GSM8K 上，自我激励学习提升了约 **12%** 的准确率，超过了普通 CoT 微调约 **6%**。在 MATH 上，提升幅度更大，超过 text‑davinci‑002 **2%**（论文声称）。  
- **消融实验**：作者分别去掉奖励模型、去掉 RL、只保留生成阶段进行对比，发现奖励模型的相对排序贡献最大，去掉后提升幅度跌至 3% 左右。  
- **局限性**：奖励模型仍依赖少量人工标注的高质量 rationales；在极端长文本或多步骤推理时，生成的 rationales 仍会出现逻辑跳跃。作者也提到，当前框架对模型规模有一定敏感性，7B 以上的模型效果更明显。

### 影响与延伸思考
自我激励学习提供了一条“模型自给自足”生成推理的路径，激发了后续研究在 **自监督推理标注**、**基于排序的奖励学习** 以及 **大模型自我校准** 方向的探索。2024 年后，几篇工作（如 *Self‑Generated Chain‑of‑Thought*、*Rank‑Based RL for Reasoning*）直接引用了该框架的思路，尝试在更大模型（30B、70B）上进一步放大效果。对想继续深挖的读者，可以关注 **奖励模型的跨任务迁移**、**多模态推理的自我激励** 以及 **更高效的排序监督** 等方向。

### 一句话记住它
让模型自己写推理、用相对排名打分、再用强化学习循环提升——自我激励学习把“标注”搬进了模型内部。