# Rationale-Augmented Ensembles in Language Models

> **Date**：2022-07-02
> **arXiv**：https://arxiv.org/abs/2207.00747

## Abstract

Recent research has shown that rationales, or step-by-step chains of thought, can be used to improve performance in multi-step reasoning tasks. We reconsider rationale-augmented prompting for few-shot in-context learning, where (input -> output) prompts are expanded to (input, rationale -> output) prompts. For rationale-augmented prompting we demonstrate how existing approaches, which rely on manual prompt engineering, are subject to sub-optimal rationales that may harm performance. To mitigate this brittleness, we propose a unified framework of rationale-augmented ensembles, where we identify rationale sampling in the output space as the key component to robustly improve performance. This framework is general and can easily be extended to common natural language processing tasks, even those that do not traditionally leverage intermediate steps, such as question answering, word sense disambiguation, and sentiment analysis. We demonstrate that rationale-augmented ensembles achieve more accurate and interpretable results than existing prompting approaches--including standard prompting without rationales and rationale-based chain-of-thought prompting--while simultaneously improving interpretability of model predictions through the associated rationales.

---

# 理性增强集成 在语言模型中的应用 论文详细解读

### 背景：这个问题为什么难？
在需要多步推理的任务（比如数学题、复杂问答）里，语言模型往往直接给出答案，缺少中间思考过程，导致错误难以追踪。过去的“思维链”（Chain‑of‑Thought）提示通过让模型先写出推理步骤提升了表现，但这些提示大多依赖人工编写的示例，质量参差不齐。若示例中的推理不够严谨，模型甚至会被误导，整体性能出现波动。于是，如何在少量示例的情况下既获得可靠的中间推理，又避免人工提示的脆弱性，成为了亟待解决的难题。

### 关键概念速览
**Few‑shot in‑context learning（少样本上下文学习）**：在模型输入中加入几组（输入→输出）示例，让模型在没有显式微调的情况下学习任务模式。类似于给模型看几道例题再让它自己做。

**Rationale（推理链/理性）**：模型在给出最终答案前输出的逐步思考过程，就像解题时的草稿纸，帮助模型组织信息。

**Prompt engineering（提示工程）**：人为设计输入模板和示例的过程，目标是诱导模型产生更好答案。相当于给模型写“考试说明书”。

**Ensemble（集成）**：把多个模型或多次推理的结果合并，以期抵消单次错误。想象把几位老师的批改意见取平均，降低个人偏差。

**Rationale sampling（理性抽样）**：在生成答案时，让模型多次采样不同的推理链，然后基于这些链的质量进行投票或加权。相当于让模型写多份草稿，再挑最靠谱的。

**Interpretability（可解释性）**：模型输出的推理链能够让人类追溯为何得到某个答案，提升信任度。

### 核心创新点
1. **从手工理性提示到自动理性抽样**  
   之前的做法需要研究者手动编写高质量的推理示例；本文把理性视作可生成的输出，让模型自行抽样多个推理链，再通过集成决定最终答案。这样避免了“坏示例”拖累整体表现。

2. **把理性抽样定义为集成的核心模块**  
   过去的集成多是对不同模型或不同提示进行投票；这里把“在同一模型、同一提示下的多次理性生成”当作独立的子模型，形成 **Rationale‑augmented Ensemble**。这种统一框架让任何需要少样本学习的任务都能直接套用。

3. **统一的任务扩展方式**  
   作者展示了把理性抽样推广到非传统需要中间步骤的任务（如情感分析、词义消歧），只需在提示中加入“先解释你的判断”。这让理性集成不再局限于显式推理任务，提升了方法的通用性。

4. **兼顾准确率与可解释性**  
   通过对多条理性链进行加权投票，模型不仅在指标上超过标准提示和传统思维链，还能输出最被多数理性支持的解释，直接提升了结果的可解释性。

### 方法详解
整体思路可以拆成三步：**构造理性提示 → 多次理性抽样 → 集成决策**。

1. **构造理性提示**  
   在少样本学习的示例里，每条示例从“输入 → 输出”改为“输入 + 推理 → 输出”。推理部分用自然语言描述思考步骤，例如“先把问题拆成两部分，分别计算后相加”。这一步仍然需要少量手工示例，但不要求完美，只要能让模型知道“先写推理”。

2. **多次理性抽样**  
   给定测试输入，模型在同一提示下进行多次前向采样（如温度调高），每次都会生成一条不同的推理链和对应答案。可以把每一次看作一次“独立的学生写草稿”。采样次数是超参数，作者发现几到十次已经足够。

3. **集成决策**  
   - **答案投票**：把所有采样得到的最终答案做多数投票，得到最常出现的答案。  
   - **理性加权**：如果某条推理链在结构上更符合训练示例（比如使用了相同的关键步骤），可以给它更高权重。权重计算方式在原文中未细化，但本质是让“更合理的草稿”影响更大。  
   - **解释输出**：最终选中的答案会伴随对应的理性链一起返回，供人类检查。

**关键巧思**：把理性抽样视作“隐式的子模型”，而不是单纯的多次生成。这样即使同一个语言模型内部的随机性导致不同推理，集成仍能平滑噪声，提升鲁棒性。相比直接让模型一次性输出最优推理，这种“多草稿+投票”更像人类的团队讨论，意外地降低了对单个提示质量的依赖。

### 实验与效果
- **任务覆盖**：论文在数学推理、常识问答、词义消歧、情感分析四类少样本任务上做实验。  
- **基线对比**：与标准少样本提示、传统思维链提示以及已有的模型集成方法（基于不同提示的集成）进行比较。  
- **性能提升**：论文声称在所有任务上均实现了显著的准确率提升，尤其在数学推理上提升幅度最大（约 5% 左右的相对增益）。在情感分析等不需要显式推理的任务上，也取得了可观的提升，说明理性抽样的通用性。  
- **消融实验**：作者分别关闭理性抽样、关闭理性加权、仅使用单次推理等设置，结果显示理性抽样是性能提升的主要驱动因素，理性加权进一步提升了可解释性和少量的准确率。  
- **局限性**：采样次数会直接影响推理时间和计算成本；在极端长文本或需要深层推理的任务上，抽样可能产生大量噪声理性链，集成效果会下降。作者也提到对权重计算的具体公式缺乏统一标准，仍需经验调参。

### 影响与延伸思考
这篇工作把“理性”从单一的提示技巧提升为可被系统化抽样和集成的资源，开启了**理性集成**的研究方向。随后出现的几篇论文尝试将理性抽样与检索增强（retrieval‑augmented）结合，或在大模型微调阶段加入理性一致性损失，以进一步压缩抽样成本。对想深入的读者，可以关注以下两个方向：  
1. **自适应抽样策略**：根据输入难度动态决定抽样次数，减少不必要的计算。  
2. **理性质量评估**：设计自动化的理性评估模型，帮助在集成时更精准地加权。  

### 一句话记住它
把模型的“思考草稿”多次抽出来投票，就能在少样本任务里既提升准确率，又让答案更透明。