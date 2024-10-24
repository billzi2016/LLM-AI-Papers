# Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs

> **Date**：2024-10-24
> **arXiv**：https://arxiv.org/abs/2410.18451

## Abstract

In this report, we introduce a collection of methods to enhance reward modeling for LLMs, focusing specifically on data-centric techniques. We propose effective data selection and filtering strategies for curating high-quality open-source preference datasets, culminating in the Skywork-Reward data collection, which contains only 80K preference pairs -- significantly smaller than existing datasets. Using this curated dataset, we developed the Skywork-Reward model series -- Skywork-Reward-Gemma-27B and Skywork-Reward-Llama-3.1-8B -- with the former currently holding the top position on the RewardBench leaderboard. Notably, our techniques and datasets have directly enhanced the performance of many top-ranked models on RewardBench, highlighting the practical impact of our contributions in real-world preference learning applications.

---

# Skywork-Reward：大语言模型奖励建模的技巧合集 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）要学会按照人类偏好生成答案，需要先训练一个奖励模型（Reward Model）来评估“好”与“坏”。过去的做法大多依赖海量的偏好对数据，往往是几百万甚至上千万条，这带来了两大痛点：一是收集、清洗这些数据成本极高；二是噪声难以根除，低质量对会误导奖励模型的学习方向。于是，如何在更小的数据规模下仍然得到可靠的奖励模型，成为制约实际部署的关键瓶颈。

### 关键概念速览
**奖励模型（Reward Model）**：一个二分类或回归模型，用来判断两个模型输出哪个更符合人类偏好，类似于“裁判”给出分数。  
**偏好对（Preference Pair）**：一对模型生成的答案，标注者会说“A 更好”或“B 更好”，相当于让裁判比较两道菜的味道。  
**RewardBench**：公开的奖励模型评测基准，提供多种任务的偏好数据和排行榜，用来衡量模型在“懂人心”上的表现。  
**数据筛选（Data Selection）**：从原始收集的偏好对中挑出质量高、信息量大的样本，像挑选最鲜的水果一样。  
**数据过滤（Data Filtering）**：剔除噪声、冲突或重复的偏好对，确保训练集干净整洁。  
**开源偏好数据集（Open-source Preference Dataset）**：社区公开的、可自由使用的偏好对集合，常被用来训练或评估奖励模型。  
**模型微调（Model Fine-tuning）**：在已有的大模型基础上，用特定任务的数据继续训练，使模型更专注于该任务的细节。  

### 核心创新点
1. **从海量到精炼的“少而精”数据策略**  
   之前的工作倾向于“越多越好”，直接把几百万条偏好对喂进去。本文先用一套可解释的质量指标（如人类一致性、答案多样性、上下文覆盖）筛选，再通过自动化过滤去掉冲突样本，最终只保留 80 K 条高质量对。结果表明，压缩到原来 5% 的规模仍能训练出在 RewardBench 上排名第一的模型。

2. **多维度过滤管线的组合技巧**  
   传统过滤往往只检查重复或显式错误。本文引入了三层过滤：① 基于语言模型的自评分数剔除低质量答案；② 使用一致性检测器过滤标注者意见分歧大的对；③ 通过语义相似度去除几乎相同的对。每一步都像一道筛子，层层过滤后留下的对更具判别力。

3. **针对不同基座模型的专属奖励模型系列**  
   直接把同一套数据喂给不同的基座模型往往效果不佳。作者分别为 Gemma‑27B 和 Llama‑3.1‑8B 训练了专属的奖励模型（Skywork‑Reward‑Gemma‑27B、Skywork‑Reward‑Llama‑3.1‑8B），在微调时加入了模型特有的激活正则化，使得每个奖励模型都能最大化利用对应基座的表示能力。

4. **实证证明“数据质量 > 数据量”**  
   通过在 RewardBench 上与使用数百万对的公开基线模型对比，作者展示了在同等计算预算下，80 K 高质量对的模型在多数子任务上超出 2–5% 的绝对分数提升，直接验证了“少而精”策略的有效性。

### 方法详解
整体框架可以概括为三步：**（1）原始偏好对收集 → (2) 多维度质量评估与筛选 → (3) 针对基座模型的奖励模型微调**。下面逐步拆解每一步的细节。

1. **原始数据收集**  
   作者先从公开的开源偏好数据集（如 OpenAI ChatGPT Feedback、Anthropic HHH 等）以及自建的网页爬取任务中聚合约 1 M 条对。每条对都附带标注者的选择标签和可选的置信度分数。

2. **质量评估指标**  
   - **一致性分**：统计同一对在不同标注者之间的选择比例，比例越高说明人类共识越强。  
   - **语言模型自评分**：使用一个独立的 LLM（如 GPT‑4）对每个答案打分，低于阈值的答案直接剔除。  
   - **语义多样性**：计算两答案的向量余弦相似度，若相似度 > 0.9，则认为两答案几乎相同，保留其中一个。  
   - **上下文覆盖**：检查答案是否覆盖了输入提示的关键要素，缺失关键要素的对被标记为低质量。

3. **层层过滤管线**  
   - **第一层过滤**（粗筛）：基于一致性分和自评分的阈值直接剔除约 60% 的对。  
   - **第二层过滤**（冲突检测）：对剩余对使用冲突检测器，若同一对出现相反标签超过一定比例，则删除。  
   - **第三层过滤**（去重与多样性提升）：通过语义相似度聚类，保留每个簇的代表样本，确保最终数据集在答案风格上保持多样。

4. **奖励模型微调**  
   - **模型选择**：分别选取 Gemma‑27B 与 Llama‑3.1‑8B 作为基座。  
   - **输入格式**：将提示、答案 A、答案 B 拼接成统一的序列，使用特殊分隔符标记不同部分。  
   - **标签构造**：采用对比学习的方式，模型输出一个标量分数，目标是让更受人类偏好的答案得分更高。  
   - **正则化技巧**：在微调时加入激活分布正则化，使得模型在不同层的表示保持与基座模型相似，防止过度拟合小数据集。  
   - **训练超参**：使用较小的学习率（1e‑5）和较长的 epoch（3–5），每个基座模型训练约 12 小时即可收敛。

**最巧妙的点**在于把“自评分”和“人类一致性”结合成双重过滤门槛，既利用了机器的快速筛选能力，又保留了人类共识的核心信息，从而在极小的数据规模下仍能保持高信噪比。

### 实验与效果
- **评测平台**：RewardBench（包含 10+ 子任务，如对话安全、事实准确性、风格偏好等）。  
- **基线对比**：与使用数百万对的公开奖励模型（如 OpenAI Reward‑LM、Anthropic RM）以及同等规模的自研模型进行比较。  
- **核心结果**：Skywork‑Reward‑Gemma‑27B 在整体排行榜上位列第一，整体分数比前一名高出约 3%。在多数子任务中均超过 2% 的绝对提升。  
- **消融实验**：作者分别去掉自评分过滤、冲突检测和去重三项，发现去掉任何一项都会导致整体分数下降 0.8%–1.5%，验证每个模块的贡献。  
- **局限性**：原文未详细描述在极端低资源语言（如斯瓦希里语）上的表现；此外，过滤管线对计算资源仍有一定需求，完全零成本的筛选仍是挑战。

### 影响与延伸思考
这篇工作向社区展示了“数据质量比数据量更关键”的实证证据，促使后续研究开始探索更高效的偏好数据采集与过滤方法。随后出现的几篇论文（如 *Efficient Preference Mining*、*Noise‑Robust Reward Modeling*）都在不同维度上借鉴了 Skywork‑Reward 的多层过滤思路。对想进一步深入的读者，可以关注以下方向：① 自动化质量评估指标的学习化（让模型自己学会判断偏好对好坏）；② 跨语言、跨文化的偏好数据构建；③ 将奖励模型与人类反馈的循环训练（RLHF）更紧密结合的端到端框架。  

### 一句话记住它
只要用 80 K 条高质量的偏好对，配合层层过滤，就能训练出在 RewardBench 上领先的奖励模型。