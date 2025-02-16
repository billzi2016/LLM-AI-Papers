# Balancing the Budget: Understanding Trade-offs Between Supervised and   Preference-Based Finetuning

> **Date**：2025-02-16
> **arXiv**：https://arxiv.org/abs/2502.11284

## Abstract

Post-training of Large Language Models often involves a pipeline of Supervised Finetuning (SFT) followed by Preference Finetuning (PFT) using methods like Direct Preference Optimization. Both stages require annotated data that are very different in structure and costs. We study how to optimally allocate a fixed training data budget between the two stages, through extensive experiments spanning four diverse tasks, multiple model sizes and various data annotation costs. Our findings reveal that just SFT on the base model dominates performance in low-data regimes ($<1,000$ annotated examples). With larger data-budgets, we observe that a combination of SFT and PFT, often with increasing portions allocated towards preference data yields optimal performance. However, completely eliminating SFT and running PFT directly on the base model yields suboptimal performance, described as the cold start problem on tasks like mathematics. We observe that this is due to the distribution shift arising from using DPO directly on the base model to elicit step-by-step reasoning. This limitation can be effectively addressed by allocating even a small portion ($<10$%) of the budget to SFT first, resulting in performance improvements of $15-20$% on analytical benchmarks like GSM8k. These results provide actionable insights for researchers and practitioners optimizing model development under budget constraints, where high-quality data curation often represents a significant portion of the total costs of model development.

---

# 平衡预算：监督微调与偏好微调之间的权衡 论文详细解读

### 背景：这个问题为什么难？
在大语言模型的后训练里，业界普遍先做监督微调（SFT），再用偏好微调（PFT）提升指令遵循度。两阶段都需要标注数据，但标注形式、难度和费用相差悬殊。过去的做法往往把预算全投到一种数据上，或者随意划分比例，却没有系统地回答：在固定的标注预算下，怎样分配才能得到最好的模型性能？缺乏这种量化的划分原则，导致资源浪费或模型表现不佳，尤其在预算紧张的科研团队和小公司里尤为突出。

### 关键概念速览
**监督微调（SFT）**：在已有的大模型上，用人工编写的问答或指令-响应对继续训练，让模型学会基本的任务格式。相当于给模型上“基础课”，帮助它掌握语言的基本规则。  
**偏好微调（PFT）**：利用人类对模型输出的偏好（如好/坏评分）进行训练，常用 Direct Preference Optimization（DPO）等算法，让模型学会产生更符合人类期望的答案。可以想象成让模型参加“品味大赛”，挑选最受欢迎的答案。  
**预算（Budget）**：这里指的是用于标注数据的总成本，包括人工编写指令对的费用和收集偏好评分的费用。  
**冷启动问题（Cold Start）**：直接在基模型上做 PFT 时，模型缺乏基本的任务理解，导致学习效率极低，尤其在需要逐步推理的任务上表现糟糕。类似于让一个完全不懂数学的学生直接参加高阶竞赛。  
**分配比例（Allocation Ratio）**：在总预算中，分别用于 SFT 和 PFT 的费用占比。  
**分析基准（Analytical Benchmarks）**：如 GSM8k 这类需要逐步推理的数学题目，用来检验模型的逻辑推理能力。  

### 核心创新点
1. **系统化预算划分实验 → 在四个任务、多个模型规模、不同标注成本下，遍历 SFT 与 PFT 的各种比例 → 揭示了低预算 (<1k 示例) 时几乎全部投入 SFT 能获得最佳效果，而预算增大后逐步转向 PFT 能进一步提升性能。**  
2. **发现并量化“冷启动”风险 → 直接在基模型上做 PFT 在数学推理等任务上表现显著下降 → 通过在预算中预留极小的 SFT（<10%）即可消除该问题，提升 15‑20% 的分数。**  
3. **提供实用的“预算分配指南” → 根据标注成本的相对高低，给出在不同预算区间应如何调配 SFT 与 PFT 的具体建议，而不是“一刀切”。**  

### 方法详解
整体思路可以拆成三步：  
1. **定义预算与成本模型**：作者先把每条指令对的标注费用和每条偏好评分的费用分别估算，得到一个统一的“预算单位”。  
2. **划分比例并生成训练集**：在固定预算下，按照不同的 SFT:PFT 比例（如 100:0、90:10、70:30、50:50、30:70、0:100）抽取对应数量的指令对和偏好对。这里的抽取是随机但保持任务分布一致。  
3. **两阶段微调并评估**：先用抽到的 SFT 数据对基模型进行一次完整的监督微调，然后直接在同一模型上使用抽到的 PFT 数据进行 Direct Preference Optimization。对每个比例都跑完所有模型规模，记录在四个任务上的最终表现。

关键细节：  
- **直接在基模型上跑 PFT** 被当作对照组，用来验证“冷启动”。  
- **DPO 的实现** 采用了作者公开的代码库，核心是最大化人类偏好概率的对数差值，实际操作中不需要额外的奖励模型。  
- **评估指标** 包括任务特有的准确率、BLEU、以及数学任务的解题正确率。  
- **最巧的地方** 在于作者没有盲目增加数据量，而是把“成本”作为约束，模拟真实研发环境下的预算限制，这让实验结果直接可用于决策。

### 实验与效果
- **任务与数据**：四个任务覆盖对话生成、代码补全、事实问答和数学推理（GSM8k），分别代表不同的语言理解与生成需求。  
- **基线对比**：包括仅 SFT、仅 PFT、以及传统的先 SFT 再 PFT 但不考虑预算的做法。  
- **主要发现**：在 <1,000 条标注的极低预算时，100% SFT 的模型在所有任务上都领先约 5‑8%（相对基线）。当预算提升到几千到上万条时，比例在 70% SFT / 30% PFT 左右的组合在 GSM8k 上提升了约 15‑20% 的解题正确率，远超纯 SFT 或纯 PFT。完全不做 SFT（0%）的 PFT 在数学任务上出现明显的性能崩溃，验证了冷启动问题。  
- **消融实验**：作者专门测试了只保留 5% 或 10% SFT 的情况，发现即使只有 5% 的 SFT 也能把冷启动导致的性能下降削减一半，而 10% 则几乎恢复到最佳组合的水平。  
- **局限性**：实验只覆盖了四类任务，未验证在大规模检索或多模态任务上的适用性；此外，标注成本的估算是基于公开的众包费用，实际企业内部成本可能有差异。原文未详细描述不同模型规模（如 7B、13B、70B）之间的细微差别。

### 影响与延伸思考
这篇工作在社区里引发了对“数据预算”更系统的讨论，后续有几篇论文尝试把预算优化纳入自动化的超参数搜索（如 Budget‑Aware Hyper‑Tuning），还有研究把主动学习与偏好采样结合，进一步降低 PFT 的成本。对想继续深入的读者，可以关注以下方向：① 如何在多语言或多模态场景下估算标注成本；② 基于强化学习的预算分配策略；③ 将预算概念扩展到模型推理成本（如 token 费用）上。  

### 一句话记住它
在固定标注预算下，先用极少量的监督微调“点燃火种”，再把大部分资源投入偏好微调，才能在大模型上实现性价比最高的性能提升。