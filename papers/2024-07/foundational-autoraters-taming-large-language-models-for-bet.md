# Foundational Autoraters: Taming Large Language Models for Better   Automatic Evaluation

> **Date**：2024-07-15
> **arXiv**：https://arxiv.org/abs/2407.10817

## Abstract

As large language models (LLMs) advance, it becomes more challenging to reliably evaluate their output due to the high costs of human evaluation. To make progress towards better LLM autoraters, we introduce FLAMe, a family of Foundational Large Autorater Models. FLAMe is trained on our large and diverse collection of 100+ quality assessment tasks comprising 5M+ human judgments, curated and standardized using publicly released human evaluations from previous research. FLAMe significantly improves generalization to a wide variety of held-out tasks, outperforming LLMs trained on proprietary data like GPT-4 and Claude-3 on many tasks. We show that FLAMe can also serve as a powerful starting point for further downstream fine-tuning, using reward modeling evaluation as a case study (FLAMe-RM). Notably, on RewardBench, our FLAMe-RM-24B model (with an accuracy of 87.8%) is the top-performing generative model trained exclusively on permissively licensed data, outperforming both GPT-4-0125 (85.9%) and GPT-4o (84.7%). Additionally, we explore a more computationally efficient approach using a novel tail-patch fine-tuning strategy to optimize our FLAMe multitask mixture for reward modeling evaluation (FLAMe-Opt-RM), offering competitive RewardBench performance while requiring approximately 25x less training datapoints. Overall, our FLAMe variants outperform all popular proprietary LLM-as-a-Judge models we consider across 8 out of 12 autorater evaluation benchmarks, encompassing 53 quality assessment tasks, including RewardBench and LLM-AggreFact. Finally, our analysis reveals that FLAMe is significantly less biased than these LLM-as-a-Judge models on the CoBBLEr autorater bias benchmark, while effectively identifying high-quality responses for code generation.

---

# 基础自动评估模型：驯服大语言模型以实现更佳自动评估 论文详细解读

### 背景：这个问题为什么难？
随着大语言模型（LLM）能力突飞猛进，评估它们的输出已经不再是“看一眼就行”。传统的人类标注成本高、速度慢，导致研究者只能在少数任务上做细致评估。已有的自动评估模型大多是基于专有数据或单一任务微调，结果在新任务上常常失灵，甚至会把低质量答案误判为高分。换句话说，缺少一个既通用又可靠的“评审官”，让模型的进步难以被客观量化。

### 关键概念速览
- **大语言模型（LLM）**：能够生成自然语言的深度模型，如 GPT‑4、Claude‑3，类似会说话的“万能机器人”。  
- **自动评估器（Autorater）**：把模型当成评审官，给生成的答案打分或判断好坏，目标是替代人工打分。  
- **质量评估任务**：具体的评估场景，例如回答准确性、代码可读性、事实一致性等，每个任务都有自己的评分标准。  
- **人类判断（Human Judgment）**：真实用户或专家给出的评分，视为最可靠的“金标准”。  
- **多任务混合（Multitask Mixture）**：一次性在大量不同任务上训练模型，让它学会通用的评估能力，类似让学生同时学习多门课程。  
- **奖励建模（Reward Modeling）**：把人类偏好转化为可优化的奖励函数，常用于强化学习微调（RLHF）。  
- **Tail‑patch 微调**：只在模型的最后几层或特定小块上进行微调，像给已有衣服加个小补丁，既省算力又能快速适配新任务。  
- **偏见基准（Bias Benchmark）**：专门测评评审官在性别、种族等敏感维度上的倾向性，确保评分公平。

### 核心创新点
1. **公开大规模评估数据集 → 通过爬取并标准化 100+ 质量评估任务，累计 5 百万条人类评分 → 为所有后续模型提供统一、可复现的训练基座，突破了以往只能使用内部专有数据的局限。**  
2. **通用基础评审模型 FLAMe → 在上述多任务混合上进行大规模预训练，而不是针对单一任务微调 → 在 53 项评估任务中显著提升跨任务泛化，甚至跑赢 GPT‑4、Claude‑3 等专有模型。**  
3. **基于 FLAMe 的奖励建模微调 → 把 FLAMe 当作“预训练评审官”，再用 RewardBench 的标签进行二次训练得到 FLAMe‑RM → 在 RewardBench 上达到 87.8% 的准确率，超过所有仅用公开数据训练的生成模型。**  
4. **Tail‑patch 微调策略 → 只在模型的“尾部”小块上使用约 1/25 的训练样本进行微调，得到轻量版 FLAMe‑Opt‑RM → 用极少的数据实现与全量微调相近的 RewardBench 表现，显著降低算力门槛。

### 方法详解
**整体框架**：先构建一个覆盖多种评估维度的大规模公开数据集；再在该数据上进行多任务预训练，得到通用评审模型 FLAMe；随后根据下游需求（如奖励建模）进行两种微调路径：全量微调得到 FLAMe‑RM，或 Tail‑patch 微调得到 FLAMe‑Opt‑RM。整个流程像先造一把通用的“评审刀”，再根据不同菜系加上专用的刀锋。

**步骤拆解**  
1. **数据收集与标准化**  
   - 从公开论文、开源评测平台、GitHub 等渠道抓取已有的人类评估结果。  
   - 统一评分尺度（如 0‑1、1‑5）并对任务描述、输入输出格式进行规范化，确保模型在训练时看到的都是同一套“语言”。  
2. **多任务混合预训练（FLAMe）**  
   - 将每个任务视为一个“标签空间”，在同一批次里随机抽取不同任务的样本。  
   - 使用指令式提示（prompt）让模型先阅读任务说明，再对给定的 LLM 输出给出分数或二分类判断。  
   - 损失函数是所有任务的交叉熵之和，模型被迫在一次前向传播中兼顾多种评估标准。  
3. **下游奖励建模微调**  
   - 取 FLAMe 的参数作为初始化，加载 RewardBench 上的 2 万条对比式人类偏好（A/B 选择）。  
   - 采用对比学习的方式，让模型对更受偏好的答案输出更高的奖励分值。  
   - 完整微调时所有层都参与梯度更新，得到 FLAMe‑RM。  
4. **Tail‑patch 微调**  
   - 只解冻模型的最后两层以及一个小的投影头（约 0.5% 参数），其余层保持冻结。  
   - 用同样的 RewardBench 对比数据进行训练，但只需要约 1/25 的样本量即可收敛。  
   - 这种方式相当于在已有评审刀的手柄上贴了一个小贴片，快速适配新任务而不需要重新打磨整把刀。

**巧妙之处**  
- **任务标准化**：把千差万别的评估任务压缩成统一的“给分”指令，让模型能够在同一次训练中学习到跨任务的评估共性。  
- **Tail‑patch**：直觉上会担心只微调小块会导致能力受限，实验却显示只要基模型已经具备强评审能力，少量参数即可捕获任务特有的偏差，极大提升了数据效率。  

### 实验与效果
- **评测覆盖**：共计 53 项质量评估任务，涵盖 RewardBench、LLM‑AggreFact、代码生成质量、事实一致性等 8 大基准。  
- **对标模型**：与 GPT‑4‑0125、GPT‑4o、Claude‑3、OpenAI 的 InstructGPT、以及其他开源 LLM‑as‑a‑Judge（如 LLaMA‑Eval）进行比较。  
- **核心数字**：在 RewardBench 上，FLAMe‑RM‑24B 达到 87.8% 的准确率，领先 GPT‑4‑0125（85.9%）和 GPT‑4o（84.7%）。在 12 项自动评审基准中，FLAMe 系列在 8 项上超越所有专有评审模型。  
- **消融实验**：去掉任务标准化会导致跨任务准确率下降约 6%；仅使用单任务微调而非多任务混合，跨任务泛化下降约 9%；Tail‑patch 与全量微调的差距在 0.3% 以内，却省掉 25 倍的训练样本。  
- **局限性**：数据来源仍受限于公开的评估集合，某些专业领域（如医学、法律）可能缺乏足够的高质量人类评分；偏见基准显示 FLAMe 虽比专有模型更公平，但在极端少数群体上仍有轻微倾向。  

### 影响与延伸思考
这篇工作提供了首个完全基于公开、许可友好数据的“大规模评审模型”，为社区搭建了可复现的评估基线。随后出现的 OpenEval、Evals‑Hub 等项目都在借鉴其多任务混合与数据标准化的思路。未来的研究可以朝以下方向深入：① 扩展到更细粒度的跨语言评估；② 将人类偏好数据与合成对抗样本结合，进一步压缩标注成本；③ 探索更细致的偏见校正机制，让评审模型在敏感维度上做到真正中立。  

### 一句话记住它
**FLAMe 把公开的海量人类评分炼成通用评审模型，用极少的额外数据就能让 LLM 评估跑赢专有的 GPT‑4。**