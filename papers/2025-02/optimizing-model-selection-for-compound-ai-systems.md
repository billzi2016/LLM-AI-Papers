# Optimizing Model Selection for Compound AI Systems

> **Date**：2025-02-20
> **arXiv**：https://arxiv.org/abs/2502.14815

## Abstract

Compound AI systems that combine multiple LLM calls, such as self-refine and multi-agent-debate, achieve strong performance on many AI tasks. We address a core question in optimizing compound systems: for each LLM call or module in the system, how should one decide which LLM to use? We show that these LLM choices have a large effect on quality, but the search space is exponential. We propose LLMSelector, an efficient framework for model selection in compound systems, which leverages two key empirical insights: (i) end-to-end performance is often monotonic in how well each module performs, with all other modules held fixed, and (ii) per-module performance can be estimated accurately by an LLM. Building upon these insights, LLMSelector iteratively selects one module and allocates to it the model with the highest module-wise performance, as estimated by an LLM, until no further gain is possible. LLMSelector is applicable to any compound system with a bounded number of modules, and its number of API calls scales linearly with the number of modules, achieving high-quality model allocation both empirically and theoretically. Experiments with popular compound systems such as multi-agent debate and self-refine using LLMs such as GPT-4o, Claude 3.5 Sonnet and Gemini 1.5 show that LLMSelector confers 5%-70% accuracy gains compared to using the same LLM for all modules.

---

# 复合 AI 系统的模型选择优化 论文详细解读

### 背景：这个问题为什么难？
复合 AI 系统会把多个大语言模型（LLM）调用串起来完成任务，例如先让模型生成草稿，再让另一个模型评审或改写。不同模块选用的模型性能差异很大，但每个模块可以挑选的模型种类很多，组合起来的搜索空间呈指数级增长。过去的做法往往把同一个模型硬塞进所有环节，或者靠人工经验逐个调试，既耗时又难以保证全局最优。因此，如何在保持系统可用性的前提下，系统化、自动化地为每个模块挑选最合适的模型，成为制约复合系统进一步提升的瓶颈。

### 关键概念速览
**LLM（大语言模型）**：能够理解并生成自然语言的深度学习模型，如 GPT‑4o、Claude 3.5 Sonnet。把它想象成会说话的“机器人专家”。  
**复合系统**：由多个 LLM 调用组成的工作流，每一步都是一个独立的“模块”。类似于流水线上的不同工位，每个工位可以配备不同的机器。  
**模块**：系统中完成特定子任务的单个 LLM 调用，例如“生成答案草稿”或“评审草稿”。  
**单模块性能**：该模块在固定输入下的输出质量，通常用准确率、BLEU 等指标衡量。相当于评估单个工位的产出好坏。  
**单调性（Monotonicity）**：在保持其他模块不变的情况下，某个模块的模型越好，整体系统的表现不会下降。就像把更好的机器装到工位上，整体产量只会不变或提升。  
**LLM 估计器**：利用一个强大的 LLM（如 GPT‑4）对其他模型的单模块表现进行预测，而不必实际跑实验。类似于让资深工程师先给出“预估产能”。  
**LLMSelector**：本文提出的自动化模型分配框架，基于单调性和 LLM 估计器迭代挑选每个模块的最佳模型。  
**多代理辩论**、**自我细化**：两类典型的复合系统，前者让多个模型相互争论后取胜者答案，后者让模型自行改写并检查自己的输出。

### 核心创新点
1. **从经验调参到基于单调性的大局搜索**  
   之前的做法是人工经验或穷举尝试，成本高且难以保证全局最优。本文发现，只要其他模块保持不变，提升单个模块的模型质量，整体性能几乎总是单调递增。利用这一性质，LLMSelector 把原本的指数搜索压缩成线性搜索，只需要逐个检查每个模块的提升空间。  
2. **用强 LLM 直接预测单模块表现**  
   传统做法需要实际跑每个模型的评测来得到单模块分数，代价巨大。作者提出让一个强 LLM（如 GPT‑4）充当“性能预言家”，根据模型描述、任务提示等信息估算该模型在该模块的表现。实验表明，这种估计与真实评测高度相关，省去了大量 API 调用。  
3. **迭代式分配策略**  
   LLMSelector 不是一次性决定所有模块，而是循环：先挑选当前提升最大的模块并为其分配最优模型，更新系统后再重新评估剩余模块。这样可以在每一步都确保收益最大化，直至再无提升空间。相比一次性全局分配，迭代式更灵活，也更容易在实际部署中中途停止。  
4. **理论与实证双重保证**  
   作者给出线性上界的调用复杂度证明，说明在模块数目有限的情况下，所需的 API 调用次数只随模块数线性增长。实验则在多代理辩论和自我细化两大场景中验证了该理论，展示了 5%–70% 的准确率提升。

### 方法详解
**整体框架**  
LLMSelector 的工作流程可以概括为三步：① 初始化所有模块使用同一基线模型；② 用强 LLM 估算每个候选模型在每个模块的单独表现；③ 迭代挑选提升最大的 (模块, 模型) 对并替换，直到没有进一步提升。整个过程只需要对每个模块做一次“性能预测”，因此调用次数与模块数呈线性关系。

**关键模块拆解**  
1. **基线设定**：系统先用最常用的模型（如 GPT‑4o）填满所有模块，得到一个可运行的基线系统。  
2. **性能估计器**：调用一个强 LLM，输入包括（a）目标模块的任务描述，（b）候选模型的公开规格（参数量、训练数据、已知强项），以及（c）基线系统的输出示例。LLM 根据这些信息输出一个预估分数（如准确率或 BLEU），相当于让它“打分”。  
3. **单调性检验**：因为单调性假设成立，系统只需要比较“提升幅度”。如果某个模块的候选模型预估分数高于当前模型，则该替换必然不会让整体性能下降。  
4. **迭代替换**：在所有 (模块, 候选模型) 对中挑选提升最大的那一对，实际把该模型部署到对应模块，更新系统状态。随后重新调用性能估计器，重新评估剩余模块的提升空间。循环直到所有提升幅度均 ≤ 0。

**算法背后的直觉**  
最巧妙的地方在于把“真实评测”交给了一个更强的 LLM 来做“预测”。这相当于让经验更丰富的工程师先给出改进建议，而不是每次都实际改装机器再跑实验。只要预测足够可靠，整体搜索就可以在极少的实际调用下完成。

### 实验与效果
- **测试任务**：论文选取了两类典型复合系统——多代理辩论（多个模型相互争论后取胜者答案）和自我细化（模型自行改写并检查输出）。这两者分别代表“协作式”与“自循环”两大范式。  
- **使用模型**：候选模型包括 GPT‑4o、Claude 3.5 Sonnet、Gemini 1.5 等主流商用 LLM。  
- **对比基线**：全部模块统一使用同一模型的“单模型基线”。  
- **效果**：论文声称，在上述任务上，LLMSelector 能带来 5% 到 70% 不等的准确率提升，具体数值随任务难度和模型差异而变化。  
- **消融实验**：作者分别关闭“单调性假设”和“LLM 估计器”，发现没有这两个核心组件时，提升幅度显著下降，验证了它们的必要性。  
- **局限性**：论文承认，LLM 估计器本身依赖于强模型的可用性和成本；如果估计误差较大，可能导致次优分配。此外，单调性在极端情况下（如模型之间出现负迁移）可能不完全成立。

### 影响与延伸思考
LLMSelector 为复合系统的自动化调优提供了可扩展的思路，随后出现的工作开始探索把“性能预测”交给专门训练的小模型或使用少量真实评测进行校准，以降低对大型 LLM 的依赖。还有研究尝试把单调性概念推广到更复杂的工作流，如包含检索、工具调用等多模态步骤的系统。想进一步了解，可关注“LLM‑as‑Evaluator”系列论文以及最近在机器学习系统优化（AutoML）与大模型协同调度（LLM‑Orchestration）交叉点的研究。

### 一句话记住它
让最强的 LLM 先“打分”，再用单调性指引的迭代替换，几行 API 调用就能把复合系统的每个环节配上最合适的模型。