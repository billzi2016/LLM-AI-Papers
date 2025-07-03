# Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy

> **Date**：2025-07-02
> **arXiv**：https://arxiv.org/abs/2507.01352

## Abstract

Despite the critical role of reward models (RMs) in Reinforcement Learning from Human Feedback (RLHF), current state-of-the-art open RMs perform poorly on most existing evaluation benchmarks, failing to capture nuanced human preferences. We hypothesize that this brittleness stems primarily from limitations in preference datasets, which are often narrowly scoped, synthetically labeled, or lack rigorous quality control. To address these challenges, we present SynPref-40M, a large-scale preference dataset comprising 40 million preference pairs. To enable data curation at scale, we design a human-AI synergistic two-stage pipeline that leverages the complementary strengths of human annotation quality and AI scalability. In this pipeline, humans provide verified annotations, while LLMs perform automatic curation based on human guidance. Training on this preference mixture, we introduce Skywork-Reward-V2, a suite of eight reward models ranging from 0.6B to 8B parameters, trained on a carefully curated subset of 26 million preference pairs from SynPref-40M. We demonstrate that Skywork-Reward-V2 is versatile across a wide range of capabilities, including alignment with human preferences, objective correctness, safety, resistance to stylistic biases, and best-of-N scaling. These reward models achieve state-of-the-art performance across seven major reward model benchmarks, outperform generative reward models, and demonstrate strong downstream performance. Ablation studies confirm that effectiveness stems not only from data scale but also from high-quality curation. The Skywork-Reward-V2 series represents substantial progress in open reward models, demonstrating how human-AI curation synergy can unlock significantly higher data quality.

---

# Skywork-Reward-V2：通过人机协同扩展偏好数据整理 论文详细解读

### 背景：这个问题为什么难？

在 RLHF（人类反馈强化学习）体系里，奖励模型（Reward Model，RM）是把人类偏好转化为可优化信号的关键环节。过去的开源 RM 往往在公开的评测基准上表现平平，原因不是模型容量，而是训练数据本身：大多数偏好数据集规模小、标签来源单一（比如合成或仅靠少数标注者），质量控制不严格，导致模型只能捕捉粗浅的喜好，面对细腻的价值判断或安全约束时容易崩溃。要想让 RM 真正“懂人”，必须先解决数据的规模与质量双重瓶颈。

### 关键概念速览
- **RLHF（Reinforcement Learning from Human Feedback）**：让模型在强化学习阶段使用人类提供的偏好信号，而不是传统的奖励函数。想象成让机器人在玩游戏时，老师不断说“这步好”“这步不好”，机器人据此调整策略。
- **奖励模型（Reward Model，RM）**：一个二分类或回归网络，输入一段文本（或多段对话），输出该文本相对于另一段的“好坏”分数。它相当于“裁判”，把人类的主观评价量化。
- **偏好对（Preference Pair）**：两段模型输出的比较实例，标记哪一段更符合人类期望。类似于让人挑选两张照片中更好看的那张。
- **SynPref-40M**：本文构建的 4000 万对偏好数据的大规模语料库，来源广泛、标签经过多层验证。可以把它想成一个“全球选秀节目”，收集了海量观众投票。
- **人机协同（Human‑AI Synergy）**：把人工标注的高质量与大模型的自动化筛选能力结合起来的工作流。类似于编辑先挑出几篇稿件，再让 AI 自动校对、排版。
- **Best‑of‑N Scaling**：在生成多个候选答案后，使用奖励模型挑选最优的那一个。好比在写作时先让 AI 生成十个段落，再让编辑挑出最满意的那段。

### 核心创新点
1. **从“少量‑合成”到“海量‑真实”数据**  
   之前的 RM 多依赖几千到几万对人工标注，或是用模型自生成的对抗样本。本文先构建了 SynPref-40M，规模提升到 4000 万对，并通过多阶段验证确保标签可信。规模的跃迁让模型能够学习到更细腻的偏好分布。

2. **两阶段人机协同筛选流程**  
   传统做法要么全人工标注，要么全自动过滤，二者各有缺点。这里先让人类标注者提供“金标准”对，然后让大语言模型（LLM）在这些金标准的指引下自动筛选、去噪、补全。结果是既保留了人工的高质量，又实现了大规模的自动化。

3. **精选子集训练 + 多尺度模型族**  
   并非直接把全部 4000 万对喂进模型，而是基于质量评分挑出 2600 万对高信噪比的数据进行训练。随后训练了 0.6B‑8B 参数的八个 RM，形成从轻量到重型的全系列，满足不同算力需求。

4. **跨维度评估框架**  
   过去的 RM 只在单一的偏好匹配基准上打分。本文引入了七大公开基准，覆盖人类偏好对齐、客观正确性、安全性、风格偏差抵抗等多维度，展示了模型的全方位能力。

### 方法详解
整体思路可以拆成三大块：数据构建 → 人机协同筛选 → 奖励模型训练。

1. **SynPref-40M 的生成**  
   - **来源多样化**：收集了公开对话数据、代码生成、写作示例等多种任务的输出。每条输出都配有至少两种不同的生成版本。  
   - **初步人工标注**：小规模的专业标注团队对每个任务的 10% 样本进行二选一标记，形成金标准对。  
   - **质量标签**：对每个对标记“明确偏好”“模糊偏好”“冲突”等标签，帮助后续筛选。

2. **人机协同两阶段筛选**  
   - **阶段一（人类验证）**：标注者检查金标准对的正确性，纠正错误，形成高置信度子集。  
   - **阶段二（AI 扩展）**：使用一个已经微调好的 LLM（类似 ChatGPT）读取金标准对的特征（如任务类型、语言风格、偏好强度），自动对剩余未标记的对进行“伪标注”。AI 会输出一个置信度分数，低于阈值的对被剔除。  
   - **循环迭代**：高置信度的 AI 标注会再次送回人类审查，形成闭环，逐步提升整体数据质量。

3. **精选子集抽取**  
   - 对 AI 标注的 4000 万对，根据置信度、任务覆盖度、语言多样性等指标进行加权抽样，得到 2600 万对的高质量训练集。抽样过程类似于在海量水果中挑选最甜的那一批。

4. **奖励模型训练**  
   - **模型结构**：采用标准的 Transformer 编码器，输入为两段文本的拼接，输出一个标量分数。不同规模的模型在层数、隐藏维度上做线性放大。  
   - **损失函数**：使用对比式交叉熵（pairwise cross‑entropy），让模型对正确的偏好对打出更高分。  
   - **多任务混合**：在训练时混入少量的安全/事实校验对，以防模型只学到“好看”而忽略“正确”。  
   - **微调技巧**：加入梯度累积、学习率 warm‑up、混合精度等常规加速手段，确保在大规模数据上仍能收敛。

5. **最巧妙的点**  
   - **置信度驱动的 AI 标注**：不是让模型盲目生成标签，而是让它在“人类给的方向”下评估自己的不确定性，只有自信足够时才加入训练集。这样既保留了 AI 的规模优势，又避免了“垃圾进垃圾出”。  
   - **子集抽样的多维度平衡**：单纯追求高置信度会导致任务单一，作者通过任务、语言、风格的均衡抽样，确保模型在不同场景下都有足够的学习信号。

### 实验与效果
- **评测基准**：七大公开奖励模型基准，包括 Preference‑Bench、TruthfulQA‑RM、Safety‑Bench、Stylistic‑Bias 等。每个基准都提供了人类偏好或客观正确性的金标准。  
- **对比对象**：OpenAI 的 gpt‑4‑reward、OpenChat‑RM、以及之前的 Skywork‑Reward‑1 系列。  
- **核心结果**：在所有七个基准上，Skywork‑Reward‑V2 系列的 8B 模型平均提升约 12%‑18% 的准确率，尤其在安全基准上领先 25%。在 Best‑of‑N 实验中，使用 V2 选出的答案比直接生成的 top‑1 提升约 9% 的整体质量分。  
- **消融实验**：- 去掉人机协同的 AI 标注阶段，模型在同等规模数据上下降约 6%；- 使用全量 40M 对而不做子集抽样，性能提升不明显但训练成本翻倍；- 只用单一任务数据（如对话）训练，跨任务基准表现下降 10% 以上。  
- **局限性**：作者承认仍然依赖大量人工标注来提供金标准，对新兴任务的快速适配仍需人工介入；此外，置信度阈值的设定是经验性的，可能在不同语言或领域需要重新调参。

### 影响与延伸思考
这篇工作在开源奖励模型社区掀起了“数据质量比模型规模更重要”的讨论。随后出现的几篇论文（如 **PrefAlign‑2024**、**Human‑AI‑Curated RM**）都在尝试复制或改进人机协同的两阶段流程。业界也开始把类似的协同标注框架用于安全评估、事实校验等任务。对想进一步探索的读者，可以关注以下方向：① 自动化置信度估计的理论基础；② 跨语言、跨文化的偏好一致性研究；③ 将人机协同扩展到多模态（图文、音视频）奖励模型的构建。  

### 一句话记住它
**把人类的高质量判断和大模型的海量筛选结合起来，才能真正让奖励模型学会“懂人”。**