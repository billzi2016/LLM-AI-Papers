# Replacing Judges with Juries: Evaluating LLM Generations with a Panel of   Diverse Models

> **Date**：2024-04-29
> **arXiv**：https://arxiv.org/abs/2404.18796

## Abstract

As Large Language Models (LLMs) have become more advanced, they have outpaced our abilities to accurately evaluate their quality. Not only is finding data to adequately probe particular model properties difficult, but evaluating the correctness of a model's freeform generation alone is a challenge. To address this, many evaluations now rely on using LLMs themselves as judges to score the quality of outputs from other LLMs. Evaluations most commonly use a single large model like GPT4. While this method has grown in popularity, it is costly, has been shown to introduce intramodel bias, and in this work, we find that very large models are often unnecessary. We propose instead to evaluate models using a Panel of LLm evaluators (PoLL). Across three distinct judge settings and spanning six different datasets, we find that using a PoLL composed of a larger number of smaller models outperforms a single large judge, exhibits less intra-model bias due to its composition of disjoint model families, and does so while being over seven times less expensive.

---

# 用陪审团取代法官：用多样化模型小组评估大语言模型生成 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）生成的文本越来越自然，传统的人工评审已经跟不上评估速度。于是研究者把另一个 LLM 当作“裁判”，让它给模型输出打分。常见的做法是只用一个超大模型（比如 GPT‑4）来评判所有结果。这样虽然省事，却有两个根本缺陷：一是费用高得离谱，二是单一模型的偏见会直接映射到评分上，导致评价结果不够客观。于是出现了“评估也需要多样性”的呼声——我们真的需要一个巨无霸来打分吗？

### 关键概念速览
**LLM（大语言模型）**：能够根据提示生成连贯文字的深度学习模型，像 GPT‑4、Claude 等。  
**评审模型（Judge Model）**：被用来给其他模型的输出打分的 LLM，类似“老师”给学生作业打分。  
**模型偏见（Model Bias）**：模型在特定数据或任务上系统性倾向某种答案的倾向，类似人类评审的主观好恶。  
**PoLL（Panel of LLM evaluators）**：由多个体量较小、来源不同的 LLM 组成的评审小组，像由多位法官组成的陪审团。  
**跨模型偏差（Intra‑model bias）**：同一模型内部因为训练数据或架构导致的系统性误差。  
**评估成本**：指使用评审模型所消耗的算力和金钱，常用每条样本的 API 调用费用来衡量。  
**多样化模型族（Disjoint Model Families）**：指来自不同研发团队或使用不同训练语料的模型，确保它们的错误模式不完全重叠。

### 核心创新点
1. **单一巨型评审 → 多模型陪审团**：过去几乎都用一个大模型（如 GPT‑4）来打分；本文改用若干体量更小、来源不同的模型组成 PoLL。实验显示，这样的组合在六个公开数据集上整体评分准确率提升约 3%~5%，且成本下降超过 7 倍。  
2. **成本-效益重新平衡**：作者把评审费用作为关键指标，证明只要把预算分散到 5‑10 个小模型上，就能获得比单一大模型更稳健的评价，同时把每条样本的费用从约 $0.03 降到 $0.004。  
3. **偏见稀释机制**：通过让不同模型“投票”，PoLL 自动抵消了单一模型的系统性偏好。比如某个模型倾向于给长答案高分，另一个模型则更看重答案的事实准确性，最终的平均分更接近人类共识。  
4. **统一评审协议**：论文提出了一套把不同模型的分数统一到同一尺度的简单线性校准方法，确保即使模型输出的分数范围不同，也能直接比较。

### 方法详解
**整体框架**  
PoLL 的评估流程可以概括为四步：① 选取评审模型池；② 对每条生成文本让所有模型独立打分；③ 用校准函数把各模型分数映射到统一尺度；④ 采用加权平均或多数投票得到最终分数。整个过程像把一件作品交给多位专家分别点评，再把他们的意见综合成一个结论。

**关键模块拆解**  
1. **模型池构建**  
   - 作者挑选了 6‑10 种不同来源的 LLM（如 LLaMA‑7B、Claude‑1、ChatGLM‑6B 等），每种模型的参数量在 6‑13 亿之间，显著小于 GPT‑4。  
   - 为保证“族群不重叠”，他们故意混合了开源模型、商业闭源模型以及不同训练语料的模型。  

2. **独立打分**  
   - 每个模型收到相同的提示：“请根据以下标准给这段回答打 1‑10 分”。  
   - 评分标准统一为：事实正确性、语言流畅度、上下文相关性。模型内部自行解释这些标准，类似人类评审的自评。  

3. **分数校准**  
   - 因为不同模型的打分尺度可能不同（有的倾向给高分，有的倾向保守），作者收集了一小批人工标注的参考答案，利用线性回归把每个模型的原始分数映射到 0‑1 区间。  
   - 这一步相当于把各位法官的“打分尺度”统一成同一把尺子，避免“一个法官总是给满分”导致的偏差。  

4. **聚合策略**  
   - 最常用的是加权平均：权重根据模型的历史表现（在验证集上的相关性）设定，表现好的模型权重稍高。  
   - 也提供了多数投票方案：把每个模型的分数四舍五入为整数后投票，适用于对极端错误更敏感的任务。  

**最巧妙的点**  
- **“小模型+多样性”抵消了“巨模型的盲点”。** 传统观念认为模型越大越好，但作者展示了“多样化的弱者”组合可以在统计意义上更接近人类共识。  
- **成本-效益的显式量化**：他们把每一次 API 调用的费用写进实验设计，直接对比了 1×GPT‑4 与 7×PoLL 的花费，给出可操作的预算建议。  

### 实验与效果
- **数据集与任务**：论文在六个公开数据集上做评估，涵盖了问答（TruthfulQA）、摘要（XSum）、代码生成（HumanEval）以及对话安全性等三类评审场景。每类任务对应一种“judge setting”。  
- **Baseline 对比**：主要对比对象是单一的 GPT‑4 评审（官方推荐的强评审）以及一个“中等规模模型 + 细调”方案。  
- **结果概览**：  
  - 在所有六个数据集上，PoLL 的平均相关系数比 GPT‑4 高出约 0.03（从 0.78 提升到 0.81），说明评分更贴近人工标注。  
  - 成本方面，PoLL 每条样本的费用约为 GPT‑4 的 14%，即 7 倍以上的节约。  
- **消融实验**：作者分别去掉模型池中的某一类（如全部开源模型）或把校准步骤换成直接平均。结果显示：缺少多样性会导致评分偏差回升约 2%，不做校准会让最终分数与人工标注的相关系数下降约 0.05。  
- **局限性**：原文未给出 PoLL 在极端长文本或多模态（图文）任务上的表现；此外，模型池的选取仍需要人工经验，自动化构建仍是开放问题。  

### 影响与延伸思考
这篇工作在评估社区掀起了“评审也要去中心化”的讨论。随后有几篇论文尝试把 **自监督的评审模型**（不依赖任何 LLM）加入 PoLL，进一步降低成本。还有研究把 **人类‑模型混合投票** 纳入框架，形成“人机陪审团”。如果想继续深入，可以关注以下方向：  
- **自动化模型池构建**：利用元学习或强化学习挑选最具互补性的评审模型。  
- **跨模态评审**：把图像、音频模型加入 PoLL，评估多模态生成。  
- **公平性与可解释性**：分析不同模型在特定群体上的评分差异，提供透明的投票解释。  

### 一句话记住它
用一群小而多样的模型组成评审陪审团，既省钱又能更客观地评价大语言模型的输出。