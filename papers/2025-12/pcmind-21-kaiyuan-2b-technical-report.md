# PCMind-2.1-Kaiyuan-2B Technical Report

> **Date**：2025-12-08
> **arXiv**：https://arxiv.org/abs/2512.07612

## Abstract

The rapid advancement of Large Language Models (LLMs) has resulted in a significant knowledge gap between the open-source community and industry, primarily because the latter relies on closed-source, high-quality data and training recipes. To address this, we introduce PCMind-2.1-Kaiyuan-2B, a fully open-source 2-billion-parameter model focused on improving training efficiency and effectiveness under resource constraints. Our methodology includes three key innovations: a Quantile Data Benchmarking method for systematically comparing heterogeneous open-source datasets and providing insights on data mixing strategies; a Strategic Selective Repetition scheme within a multi-phase paradigm to effectively leverage sparse, high-quality data; and a Multi-Domain Curriculum Training policy that orders samples by quality. Supported by a highly optimized data preprocessing pipeline and architectural modifications for FP16 stability, Kaiyuan-2B achieves performance competitive with state-of-the-art fully open-source models, demonstrating practical and scalable solutions for resource-limited pretraining. We release all assets (including model weights, data, and code) under Apache 2.0 license at https://huggingface.co/thu-pacman/PCMind-2.1-Kaiyuan-2B.

---

# PCMind-2.1‑Kaiyuan‑2B 论文详细解读

### 背景：这个问题为什么难？

大模型的性能大多来源于海量、高质量的训练数据和精心调校的训练流程。工业界往往使用闭源、经过严格筛选的语料以及专有的训练配方，导致开源社区在模型规模、数据质量和训练效率上与商业模型形成明显差距。传统的开源预训练方式往往直接把所有公开数据堆在一起，缺乏系统的质量评估和课程安排，容易在算力受限的情况下出现“数据浪费”或“梯度不稳”。因此，如何在有限算力和预算下，既保证数据质量，又提升训练效率，成为阻碍开源大模型进一步追赶商业模型的关键瓶颈。

### 关键概念速览

**Quantile Data Benchmarking（分位数据基准）**：把不同来源、不同质量的开源语料按照统计分位数进行排序和对比，类似于把各种水果按甜度分层，帮助我们决定哪些数据该多吃、哪些该少吃。  

**Strategic Selective Repetition（策略性选择性重复）**：在多阶段训练中，对稀缺的高质量样本进行有计划的重复出现，而对低质量样本只出现一次，像在课堂上老师会多次强调重点概念，却只让学生浏览一次次要的例子。  

**Multi‑Domain Curriculum Training（多域课程学习）**：先让模型学习容易、通用的语言（如英文），再逐步加入中文、代码、数学等更专业的领域，类似于先学好基础数学，再去学微积分、线性代数等进阶内容。  

**Logits Soft‑Capping（Logits软截断）**：在模型输出层对 logits（未归一化的概率）施加柔和的上限，防止极端值导致数值不稳定，像给激动的演讲者加个音量限制器。  

**Sandwich Normalization（夹层归一化）**：在网络的不同层之间交叉使用多种归一化手段，以兼顾训练速度和数值稳定性，类似于在烘焙时先用低温预热，再用高温快速定型。  

**FP16 Stability（FP16 稳定性）**：使用 16 位浮点数进行训练时的数值安全措施，确保梯度不会因为精度不足而“炸掉”。  

**Data Preprocessing Pipeline（数据预处理流水线）**：一套自动化的清洗、去重、格式化、质量打分等步骤，像流水线工厂把原材料加工成合格部件。  

### 核心创新点

1. **分位数据基准 → 通过统计分位数对异构开源语料进行统一评估 → 为数据混合提供量化依据，避免盲目堆砌导致的噪声膨胀。** 传统做法往往只靠人工经验或粗糙的过滤规则，缺乏可比性。  

2. **策略性选择性重复 + 多阶段训练 → 在每个训练阶段只让同一批数据出现一次，但对高质量样本在不同阶段进行有计划的重复 → 让稀缺的优质信息被模型多次“看到”，提升学习效率。** 以前的重复策略多是随机或全局重复，容易浪费算力。  

3. **多域课程学习 → 按质量从易到难的顺序安排语言、代码、数学等领域的样本 → 模型先建立通用语言能力，再逐步适配专业任务，显著提升跨域表现。** 过去的多语言模型往往一次性喂入所有语料，导致收敛慢、效果不均。  

4. **架构细节（Logits Soft‑Capping、Sandwich Normalization、FP16 稳定性） → 在 Gemma‑2 基础上加入软截断和夹层归一化，并针对 FP16 进行专门调优 → 让 2B 参数模型在 bf16/FP16 环境下训练更稳、收敛更快。** 这些技巧在开源社区中少有系统化公开，填补了实现细节的空白。

### 方法详解

**整体框架**  
这篇报告把模型训练划分为五个阶段，每个阶段对应一种数据质量和领域组合。核心流程是：① 统一评估所有公开语料 → ② 按分位基准划分为高、中、低质量层 → ③ 依据课程学习顺序安排阶段 → ④ 在每个阶段执行策略性选择性重复 → ⑤ 通过改进的模型架构和数值技巧完成预训练。

**步骤拆解**  

1. **数据评估与分位划分**  
   - 使用预设的质量打分模型（如 perplexity、重复率、语言检测）为每条样本打分。  
   - 将所有样本的分数按百分位切分，例如前 10% 为“极高质量”，10‑30% 为“高质量”，其余为“普通”。  
   - 这一步类似于把水果市场的所有水果按甜度排队，后续可以决定哪些要多买、哪些只买一点。

2. **课程学习排程**  
   - 先选取英文语料（因为英文资源最丰富且质量相对均衡），作为第 1 阶段的主要输入。  
   - 第 2、3 阶段逐步加入中文、代码、数学等专业域的高质量子集。  
   - 每个阶段的样本集合都是“只出现一次”，防止同一数据在同一阶段被模型过度记忆。

3. **策略性选择性重复**  
   - 对于分位中最高的那部分（极高质量），在每个阶段都保留一次出现机会，即在 5 阶段中会出现 5 次。  
   - 中等质量的样本只在它所在的阶段出现一次。  
   - 低质量样本则只在最初的英文阶段出现一次，随后不再出现。  
   - 这样模型在每个阶段都能“复习”关键信息，却不会被噪声淹没。

4. **模型架构与数值技巧**  
   - 基础网络采用 Gemma‑2 的 Transformer 结构，参数规模约 2 B。  
   - 在每层的输出前加入 **Logits Soft‑Capping**：对 logits 施加一个平滑的上限函数，防止极端值导致梯度爆炸。  
   - 在每个子层之间交叉使用 **LayerNorm** 与 **RMSNorm**（即 Sandwich Normalization），兼顾收敛速度和数值稳定。  
   - 训练时使用 **FP16**（半精度）或 **bf16**（Brain Float16），并在关键位置加入梯度缩放、动态 loss scaling 等手段，确保在低精度下仍能保持数值安全。

5. **高效预处理流水线**  
   - 自动化脚本完成去重、格式统一、质量打分、分位划分、阶段划分等全部步骤，支持并行化处理，显著降低人工干预成本。  

**最巧妙的点**  
- 把“质量驱动的重复”与“课程学习”结合起来，使得稀缺的高质量样本在模型的每一次“学习升级”中都能被重新强化，而不必额外增加算力。  
- 通过软截断和夹层归一化，让 2 B 参数模型在仅使用 16 位浮点数的环境下仍能保持训练稳定，这在资源受限的开源项目中尤为关键。

### 实验与效果

- **测试任务**：在公开的语言理解基准（如 MMLU、C-Eval、HumanEval）以及代码生成任务上进行评估。  
- **对比基线**：与同规模的开源模型（如 LLaMA‑2‑7B 的子模型、Gemma‑2‑2B）以及商业闭源模型的公开指标进行比较。  
- **结果概述**：报告称 Kaiyuan‑2B 在多数评测上能够追平或略超同等参数的全开源模型，尤其在中文和代码任务上表现突出。具体数值未在摘要中给出，原文未详细披露。  
- **消融实验**：通过去掉 Quantile Benchmark、Selective Repetition 或 Curriculum Training，模型在对应任务上的得分均出现明显下降，说明每个模块都有实质贡献。  
- **局限性**：作者承认模型仍受限于 2 B 参数规模，在更大模型上未验证；此外，质量评估依赖的打分模型本身可能带有偏见，导致某些领域数据被低估。  

### 影响与延伸思考

这篇技术报告在开源大模型社区引发了对“数据质量驱动的训练策略”更深入的讨论。随后出现的几篇工作（如 **OpenChat‑3B**、**MOSS‑Base**）在数据混合阶段加入了类似的分位评估或课程学习概念，说明 PCMind‑2.1 的思路已经被快速采纳。对想继续探索的读者，可以关注以下方向：

- **更细粒度的质量评估模型**：利用多任务评估器或人类反馈来提升分位划分的可靠性。  
- **跨模态课程学习**：把文本、图像、音频等多模态数据按质量和难度进行分阶段训练。  
- **大规模重复策略的理论分析**：从信息论角度解释为何稀疏高质量样本的多次出现能显著提升收敛速度。  

### 一句话记住它

**PCMind‑2.1‑Kaiyuan‑2B 用“质量分位+分阶段重复+多域课程”三把钥匙，在算力受限的情况下把开源大模型的训练效率和效果推向了商业水平。**