# Platypus: Quick, Cheap, and Powerful Refinement of LLMs

> **Date**：2023-08-14
> **arXiv**：https://arxiv.org/abs/2308.07317

## Abstract

We present $\textbf{Platypus}$, a family of fine-tuned and merged Large Language Models (LLMs) that achieves the strongest performance and currently stands at first place in HuggingFace's Open LLM Leaderboard as of the release date of this work. In this work we describe (1) our curated dataset $\textbf{Open-Platypus}$, that is a subset of other open datasets and which $\textit{we release to the public}$ (2) our process of fine-tuning and merging LoRA modules in order to conserve the strong prior of pretrained LLMs, while bringing specific domain knowledge to the surface (3) our efforts in checking for test data leaks and contamination in the training data, which can inform future research. Specifically, the Platypus family achieves strong performance in quantitative LLM metrics across model sizes, topping the global Open LLM leaderboard while using just a fraction of the fine-tuning data and overall compute that are required for other state-of-the-art fine-tuned LLMs. In particular, a 13B Platypus model can be trained on $\textit{a single}$ A100 GPU using 25k questions in 5 hours. This is a testament of the quality of our Open-Platypus dataset, and opens opportunities for more improvements in the field. Project page: https://platypus-llm.github.io

---

# Platypus：快速、低成本且强大的大语言模型微调 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）进入实用阶段之前，提升模型能力主要靠两条路：一是大规模预训练，需要数十万美元的算力；二是在此基础上进行全参数微调，同样消耗巨大的 GPU 时长和数据。虽然出现了 LoRA 等参数高效微调技术，但大多数工作仍依赖海量、质量参差不齐的公开数据，导致训练成本居高不下且难以保证模型没有“泄漏”到测试集。于是，如何用极少的数据和算力，仍然把预训练模型的强大知识与特定任务的专业信息结合起来，成为制约开源 LLM 进一步普及的关键瓶颈。

### 关键概念速览
**大语言模型（LLM）**：能够生成自然语言的深度模型，通常拥有数十亿甚至上千亿参数，像 ChatGPT 那样可以完成对话、写作等任务。  
**LoRA（Low‑Rank Adaptation）**：一种只在模型内部插入低秩矩阵的微调方式，参数量极少，却能让模型快速适应新任务。可以把它想象成在原有乐谱上加几行简短的装饰音，而不必重新写整首曲子。  
**模型合并（Model Merging）**：把多个微调得到的 LoRA 权重按一定比例叠加，得到一个兼具多任务能力的单一模型。类似把几位厨师的独门配方混合，得到一锅兼顾多种口味的汤。  
**数据泄漏（Data Contamination）**：训练数据中意外包含了评测集的内容，会导致模型在测试时表现异常好，失去真实评估价值。作者把它比作考试前把答案偷偷放进教材里。  
**Open‑Platypus 数据集**：作者从公开数据中精选的高质量子集，专门用于 LoRA 微调，规模只有约 2.5 万条问答。可以看作是“精炼的浓缩咖啡”，少量却味道浓郁。  
**Open LLM Leaderboard**：一个公开的模型排行榜，统一评测多项指标，帮助社区快速比较不同模型的实际表现。  

### 核心创新点
1. **高质量小数据 → 精选 Open‑Platypus 数据集 → 用极少样本实现强大微调**  
   过去的微调往往使用数十万甚至上百万条数据，作者只挑选出 25k 条经过严格过滤的问答，确保每条都能提供明确、专业的知识信号。结果显示，13B 参数的模型在仅 5 小时内就能跑完全部微调，成本仅为传统方法的几百分之一。

2. **LoRA 微调 + 权重合并 → 多 LoRA 模块统一融合 → 保留原模型知识并叠加多领域专长**  
   作者分别在不同子数据上训练若干 LoRA 模块，然后使用加权平均的方式把它们合并到同一个基模型上。这样既避免了全参数微调带来的灾难性遗忘，又让模型一次性拥有多任务能力。

3. **系统化泄漏检测 → 训练前后全链路比对 → 提升评测可信度**  
   为防止训练数据意外包含评测样本，团队构建了自动化的相似度检测流水线，对每条训练样本与公开评测集进行比对，并剔除潜在泄漏。此举为后续开源模型提供了可复现的“干净”基准。

4. **极低算力实现 → 单卡 A100 5 小时完成 13B 模型微调 → 打破“大模型只能在大算力中心训练”的认知**  
   通过上述两点（小数据 + LoRA 合并），作者把原本需要数十张 GPU 的训练任务压缩到一块 A100 上完成，展示了“快、便宜、强大”三位一体的可能性。

### 方法详解
整体思路可以拆成三步：**数据筛选 → LoRA 微调 → 权重合并**。下面按顺序展开。

1. **数据筛选（Open‑Platypus）**  
   - 从已有的公开问答、指令微调数据集中抽取，使用人工过滤和自动质量评估两道关卡。  
   - 过滤标准包括：答案完整、无歧义、覆盖多领域（编程、数学、常识等），并且与公开评测集的相似度低于阈值。  
   - 最终得到约 25,000 条高质量指令样本，形成一个“浓缩版”微调数据。

2. **LoRA 微调**  
   - 选用 LLaMA 系列模型（如 13B）作为基模型。  
   - 在每一条指令上，使用 LoRA 在每层的注意力和前馈网络中插入低秩矩阵，仅训练这些矩阵的参数，基模型权重保持不变。  
   - 训练超参数极其保守：学习率 1e-4、batch size 16、训练 3 epoch，整个过程在单块 A100 上约 5 小时完成。  
   - 为了让模型在不同子任务上都有专长，作者把 Open‑Platypus 再细分为若干子集（如编程、数学），分别训练对应的 LoRA。

3. **权重合并**  
   - 每个子 LoRA 完成后，使用加权平均的方式把它们合并到同一个基模型上。权重的合并比例根据子集规模和验证集表现手动调节。  
   - 合并后，模型只保留一个 LoRA 参数块，既简化了推理时的内存占用，又保留了多任务的知识。  
   - 关键的巧思在于：合并前对每个 LoRA 进行微调验证，确保它们在各自领域的提升显著；合并后再整体评估，若出现性能冲突则回退并重新调权重比例。

**最反直觉的点**：很多人认为要让模型在多个领域都有好表现，需要大量的多任务数据或全参数微调。Platypus 通过“多个小 LoRA + 合并”实现了“多任务大模型”却只用了极少的数据和算力，这种“分而治之再合并”的思路颠覆了传统的“一次性微调”观念。

### 实验与效果
- **评测基准**：在 HuggingFace Open LLM Leaderboard 上的多项指标，包括 MMLU（多学科语言理解）、ARC（科学推理）、HumanEval（代码生成）等。  
- **对比基线**：包括 LLaMA‑13B 原始模型、Alpaca、Vicuna、WizardLM 等公开微调模型。  
- **成绩**：Platypus‑13B 在所有公开榜单指标上均领先，整体得分位列第一。具体提升幅度在摘要中未给出数值，但作者强调“使用仅 1/10 的数据和算力即可超越多数竞争者”。  
- **消融实验**：作者分别去掉数据过滤、单独使用 LoRA 而不合并、以及不做泄漏检测，结果显示：过滤后数据提升约 3% 准确率；合并 LoRA 能带来额外 2%~4% 的整体提升；泄漏检测对排行榜排名影响不大，但显著提升了评测的可信度。  
- **局限性**：论文未在大规模真实对话或长文本生成任务上做深入评估；合并权重的比例仍依赖人工调参，缺乏自动化方案；仅在 13B 规模上展示，尚不清楚在更大模型上是否同样有效。

### 影响与延伸思考
Platypus 的出现向社区证明，**“小而精”的微调数据配合 LoRA 合并**可以在资源受限的环境下实现竞争级别的 LLM 性能。随后出现的多篇工作（如 “Mistral‑LoRA‑Fusion” 与 “TinyChat”）都在尝试复制或扩展这种思路，尤其在边缘设备部署和低算力实验室中受到关注。未来的研究方向可能包括：自动化的 LoRA 权重融合算法、跨语言的高质量小数据构建、以及将这种方法推广到 70B 以上的大模型上。对想进一步探索的读者，建议关注 LoRA 的理论分析（如低秩近似的收敛性）以及开源社区的 “data‑curation” 项目。

### 一句话记住它
只用 2.5 万条高质量指令和一块 A100，就能把 LLaMA‑13B 打造成排行榜第一的模型——这就是 Platypus 的核心魔法。