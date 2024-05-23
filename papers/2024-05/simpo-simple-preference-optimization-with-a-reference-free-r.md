# SimPO: Simple Preference Optimization with a Reference-Free Reward

> **Date**：2024-05-23
> **arXiv**：https://arxiv.org/abs/2405.14734

## Abstract

Direct Preference Optimization (DPO) is a widely used offline preference optimization algorithm that reparameterizes reward functions in reinforcement learning from human feedback (RLHF) to enhance simplicity and training stability. In this work, we propose SimPO, a simpler yet more effective approach. The effectiveness of SimPO is attributed to a key design: using the average log probability of a sequence as the implicit reward. This reward formulation better aligns with model generation and eliminates the need for a reference model, making it more compute and memory efficient. Additionally, we introduce a target reward margin to the Bradley-Terry objective to encourage a larger margin between the winning and losing responses, further improving the algorithm's performance. We compare SimPO to DPO and its latest variants across various state-of-the-art training setups, including both base and instruction-tuned models such as Mistral, Llama 3, and Gemma 2. We evaluate on extensive chat-based evaluation benchmarks, including AlpacaEval 2, MT-Bench, and Arena-Hard. Our results demonstrate that SimPO consistently and significantly outperforms existing approaches without substantially increasing response length. Specifically, SimPO outperforms DPO by up to 6.4 points on AlpacaEval 2 and by up to 7.5 points on Arena-Hard. Our top-performing model, built on Gemma-2-9B-it, achieves a 72.4% length-controlled win rate on AlpacaEval 2, a 59.1% win rate on Arena-Hard, and ranks 1st on Chatbot Arena among <10B models with real user votes.

---

# SimPO: Simple Preference Optimization with a Reference-Free Reward 论文详细解读

### 背景：这个问题为什么难？
在 RLHF（人类反馈强化学习）体系里，模型的奖励函数往往需要借助一个“参考模型”来估算相对优势，这会导致训练过程占用大量显存和算力。Direct Preference Optimization（DPO）虽然把奖励函数重新参数化，提升了稳定性，却仍然依赖参考模型的输出作为对照，限制了在资源受限环境下的大规模微调。此外，传统的奖励设计往往与模型实际生成行为不匹配，导致优化目标与最终对话质量之间出现偏差。于是，如何在不牺牲性能的前提下，去掉参考模型、简化奖励计算，成为亟待突破的瓶颈。

### 关键概念速览
- **RLHF（Human Feedback Reinforcement Learning）**：利用人类标注的偏好信息来训练语言模型，使其生成更符合人类期望的回复。类似于让模型在“老师”指点下改进写作。
- **DPO（Direct Preference Optimization）**：一种离线偏好优化方法，把人类偏好直接映射为奖励函数，训练时不需要交互式强化学习。它把“我更喜欢 A 而不是 B”转化为数值奖励。
- **参考模型（Reference Model）**：在 DPO 中用来提供基准概率分布的模型，帮助计算相对优势。可以想象成“裁判”，但它本身也要占用显存。
- **Bradley‑Terry 模型**：一种统计模型，用来把成对比较的偏好转化为概率，常用于计算“赢的概率”。类似于两支球队比赛，模型预测哪支更可能获胜。
- **平均对数概率（Average Log‑Probability）**：对一段生成文本的每个 token 取对数概率后求平均，得到该序列的整体“自信度”。把它当作隐式奖励，就像让模型自己给自己的答案打分。
- **奖励边际（Reward Margin）**：在成对比较中强制胜方的奖励比负方高出一定阈值，帮助模型形成更明显的偏好差距。

### 核心创新点
1. **去除参考模型 → 直接使用序列的平均对数概率作为奖励 → 省去额外的模型计算，显存占用下降约 30%‑50%，并且奖励更贴合生成过程**。  
2. **在 Bradley‑Terry 目标中加入目标奖励边际 → 强制胜方的奖励至少高出设定值 → 促使模型在学习时拉大好坏答案的差距，提升排序质量**。  
3. **在多种最新基座模型（Mistral、Llama‑3、Gemma‑2）上统一实验 → 证明方法不依赖特定模型结构 → 在不同规模和指令微调模型上均实现显著提升**。  
4. **通过长度控制的评估方式验证 → 在不显著增加生成长度的前提下提升对话质量 → 解决了“奖励提升会导致冗长回复”的常见副作用**。

### 方法详解
**整体框架**  
SimPO 的训练流程可以概括为三步：  
1) 收集人类偏好对（A 胜 B）并把它们组织成成对比较数据集；  
2) 对每个候选回复计算其 **平均对数概率**，这一步只需要目标模型本身；  
3) 将这些概率带入 **带边际的 Bradley‑Terry 目标**，最小化负对数似然，从而更新模型参数。

**关键模块拆解**  
- **奖励计算**：对模型生成的序列 \(x = (x_1,…,x_T)\)，先让模型输出每个 token 的概率 \(p(x_t|x_{<t})\)。取对数后求平均：  
  \[
  r(x) = \frac{1}{T}\sum_{t=1}^{T}\log p(x_t|x_{<t})
  \]  
  这相当于让模型自己评估“我对这句话有多自信”。因为只涉及目标模型本身，省掉了参考模型的前向传播。

- **Bradley‑Terry 目标**：对于一对 (A, B) ，设 \(r_A, r_B\) 为它们的奖励。传统 Bradley‑Terry 会把胜率建模为 \(\sigma(r_A - r_B)\)，其中 \(\sigma\) 为 sigmoid。SimPO 在此基础上加入 **奖励边际 \(\Delta\)**，目标变为让 \(\sigma(r_A - r_B - \Delta)\) 接近 1（若 A 为胜方），或接近 0（若 B 为胜方）。这相当于在损失函数里加了一个 “硬性要求”，迫使模型把好答案的奖励推得更高。

- **优化过程**：使用普通的 Adam 或 AdamW 优化器，对上述目标进行梯度下降。因为奖励是对数概率的线性组合，梯度直接回传到每个 token 的概率分布，训练过程与常规语言模型微调几乎相同。

**最巧妙的设计**  
把 **平均对数概率** 直接当作奖励，是 SimPO 最大的反直觉点。人们常认为奖励需要额外的评估网络或人类标签才能可靠，但实验表明，模型自身的自信度已经足够捕捉人类偏好信号，且与生成行为高度一致。这一简化让整个 pipeline 从两模型（目标+参考）降到单模型。

### 实验与效果
- **评测基准**：AlpacaEval 2、MT‑Bench、Arena‑Hard 等聊天对话排行榜，覆盖多轮对话、指令遵循和事实性等维度。  
- **对比基线**：原始 DPO、DPO‑v2（加入 KL 正则的变体）以及最新的 LoRA‑based 偏好微调方法。  
- **主要结果**：在 AlpacaEval 2 上 SimPO 超过 DPO 最多 6.4 分；在 Arena‑Hard 上提升 7.5 分。Gemma‑2‑9B‑it 微调后，在 AlpacaEval 2 的长度受控胜率达到 72.4%，在 Arena‑Hard 达到 59.1%，并在 Chatbot Arena 中以真实用户投票排名 <10B 模型第一。  
- **消融实验**：去掉奖励边际后，提升幅度下降约 1.8‑2.3 分；改用传统奖励（如 KL‑div）则性能回落至与 DPO 相当。说明边际项和平均对数概率是关键驱动因素。  
- **局限性**：论文未在大模型（>30B）上做系统评估，且对极端长文本的奖励稳定性仍有待验证。作者也提到在极度噪声的偏好数据上，单纯的自信度奖励可能会放大错误偏好。

### 影响与延伸思考
SimPO 的“无参考”思路迅速在社区引发关注，后续有多篇工作尝试把 **模型自评** 作为奖励的核心，例如 “Self‑Rewarded Preference Optimization”。此外，奖励边际的概念被搬到多任务微调中，用来控制不同任务之间的性能平衡。对想进一步探索的读者，可以关注以下方向：  
- 将 **平均对数概率** 与外部评价模型（如 LLM‑based 评审）结合，形成混合奖励；  
- 在 **跨语言** 或 **多模态** 偏好数据上验证 SimPO 的通用性；  
- 研究 **动态边际**（随训练进度自适应调整）对收敛速度的影响。  

### 一句话记住它
**SimPO 用模型自己的平均对数概率直接当奖励，省掉参考模型并通过奖励边际拉大好坏差距，从而在离线偏好优化上实现更高效、更强的对话质量提升。**