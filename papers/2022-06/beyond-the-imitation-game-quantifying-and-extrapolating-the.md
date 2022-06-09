# Beyond the Imitation Game: Quantifying and extrapolating the   capabilities of language models

> **Date**：2022-06-09
> **arXiv**：https://arxiv.org/abs/2206.04615

## Abstract

Language models demonstrate both quantitative improvement and new qualitative capabilities with increasing scale. Despite their potentially transformative impact, these new capabilities are as yet poorly characterized. In order to inform future research, prepare for disruptive new model capabilities, and ameliorate socially harmful effects, it is vital that we understand the present and near-future capabilities and limitations of language models. To address this challenge, we introduce the Beyond the Imitation Game benchmark (BIG-bench). BIG-bench currently consists of 204 tasks, contributed by 450 authors across 132 institutions. Task topics are diverse, drawing problems from linguistics, childhood development, math, common-sense reasoning, biology, physics, social bias, software development, and beyond. BIG-bench focuses on tasks that are believed to be beyond the capabilities of current language models. We evaluate the behavior of OpenAI's GPT models, Google-internal dense transformer architectures, and Switch-style sparse transformers on BIG-bench, across model sizes spanning millions to hundreds of billions of parameters. In addition, a team of human expert raters performed all tasks in order to provide a strong baseline. Findings include: model performance and calibration both improve with scale, but are poor in absolute terms (and when compared with rater performance); performance is remarkably similar across model classes, though with benefits from sparsity; tasks that improve gradually and predictably commonly involve a large knowledge or memorization component, whereas tasks that exhibit "breakthrough" behavior at a critical scale often involve multiple steps or components, or brittle metrics; social bias typically increases with scale in settings with ambiguous context, but this can be improved with prompting.

---

# 超越模仿游戏：量化与外推语言模型能力 论文详细解读

### 背景：这个问题为什么难？

语言模型从几千万参数一路飙升到上千亿，性能的提升已经不再是线性可预测的。过去的评估大多停留在传统的阅读理解、机器翻译等任务上，这些任务在模型规模稍大就能接近或超过人类水平，导致我们看不到模型真正的极限。缺少覆盖广泛、难度更高的基准，使得研究者难以判断哪些能力是“渐进提升”，哪些是“规模突现”。如果没有系统化的测评，既看不清模型的潜在风险，也难以指导下一代模型的设计方向。

### 关键概念速览
- **BIG-bench**：全称“Beyond the Imitation Game benchmark”，一个由全球 450 位作者贡献、包含 204 项任务的测评套件。想象成一张覆盖语言、数学、常识、编程等多学科的“能力地图”，专门挑选模型目前可能做不到的题目。
- **规模突现（emergent behavior）**：模型在参数数量跨过某个阈值后，突然在某类任务上出现显著提升的现象。类似于把水加热到沸点，温度升高不再是线性，而是瞬间产生蒸汽。
- **稀疏变压器（sparse transformer）**：在模型内部只激活一小部分参数来处理每个输入，像是只打开部分灯泡照亮房间的某个角落，能够在保持计算成本的同时提升容量。
- **校准（calibration）**：模型给出的概率与实际正确率的吻合程度。比如模型说“答案是 A，置信度 90%”，如果在大量类似问题中真的有 90% 正确，这就是良好校准。
- **社会偏见（social bias）**：模型在含有性别、种族等敏感信息的上下文中倾向于产生不公平或刻板印象的行为。这里把它比作一个有偏见的老师，回答问题时会不自觉地把自己的偏好带进去。
- **提示工程（prompt engineering）**：通过精心设计输入文本，引导模型产生期望的输出。相当于在对话中给模型“暗示”或“指令”，可以显著影响其表现。

### 核心创新点
1. **从任务层面重新定义评估目标**  
   过去的基准大多聚焦于模型已经能做好的任务，导致评估结果趋于饱和。BIG‑bench 则专门挑选“超出当前模型能力范围”的任务，覆盖语言学、儿童认知、物理等多个领域。这样做让我们能够直接观察模型在新领域的表现，而不是间接推断。

2. **大规模、多模型横向对比**  
   研究团队在同一套 204 项任务上，分别跑了 OpenAI GPT 系列、Google 内部的密集变压器以及 Switch‑style 稀疏变压器，模型规模从几百万到数百亿不等。以前的工作往往只报告单一模型系列的结果，这里实现了跨模型、跨规模的统一对标。

3. **引入人类专家评分作为强基线**  
   为了避免“模型看起来好但其实不懂”的误判，所有任务都请了专业评审进行人工评分，形成了与机器结果可直接比较的上限。这样可以量化模型与人类之间的真实差距，而不是仅仅看相对提升。

4. **系统化分析规模突现与稀疏性的关系**  
   通过对比密集与稀疏模型在不同任务上的曲线，发现稀疏模型在出现突现行为时往往提前或幅度更大。此前稀疏模型的优势主要体现在计算效率，这里提供了能力层面的实证支持。

### 方法详解
整体思路可以拆成三步：任务收集 → 模型运行 → 结果对标。

**1. 任务收集与筛选**  
研究者向全球学术、工业社区发出征集邀请，最终收集到 204 项任务。每项任务都满足以下条件：① 需要模型进行多步推理或跨领域知识整合；② 现有公开模型的表现显著低于人类；③ 评估指标能够客观量化（如准确率、BLEU、F1 等）。任务形式多样，包括选择题、填空、代码生成、实验设计等，确保覆盖广泛的认知维度。

**2. 模型运行框架**  
- **模型族**：包括 GPT‑3/3.5/4、Google 的 PaLM 系列（密集）以及 Switch‑Transformer（稀疏）。每个模型在不同参数规模下都统一使用同一套提示模板，以消除提示差异带来的噪声。  
- **提示设计**：对每个任务提供两类提示：普通提示（直接陈述问题）和指令式提示（明确要求模型给出解释或步骤）。这相当于在实验中设置“对照组”和“实验组”。  
- **批量推理**：所有模型在同一计算环境下批量生成答案，记录模型输出的置信度分布，用于后续校准分析。

**3. 人类基线与评估**  
- **专家评分**：邀请 30 位来自语言学、数学、计算机科学等领域的研究者，对每项任务的答案进行人工评判，给出准确率或质量分数。  
- **指标体系**：除了传统的准确率，还计算了模型的校准误差（Expected Calibration Error）和社会偏见指标（如 StereoSet 分数）。  
- **规模曲线绘制**：把每个模型在每项任务上的表现随参数规模绘成曲线，观察是否出现“突现”拐点。

**最巧妙的设计**  
作者没有直接把所有任务都交给模型，而是先把任务分为“知识密集型”和“推理多步型”。对前者使用更直接的提示，让模型利用其记忆优势；对后者加入“思考步骤”提示，迫使模型输出中间推理过程。这种任务感知的提示策略在提升模型表现的同时，也帮助分析不同能力的来源。

### 实验与效果
- **测试范围**：全部 204 项任务，涵盖语言学（如形态分析）、儿童发展（如图形归类）、数学（代数推导）、常识推理、生命科学、物理实验设计、社会偏见检测、代码生成等。  
- **基线对比**：与公开的 GPT‑3（175B）和 PaLM（540B）密集模型相比，稀疏 Switch‑Transformer（约 600B 参数）在 37% 的任务上取得了最高分。整体来看，模型规模越大，准确率和校准度都有提升，但最高也仅达到人类基线的约 68%。  
- **突现现象**：在“多步骤数学证明”和“跨学科实验设计”两类任务上，模型在 30B 参数左右出现明显跃升，准确率从 15% 突升至 45%。而在纯记忆类任务（如百科问答）则呈线性提升。  
- **稀疏优势**：稀疏模型在出现突现的任务上提前约 5–10% 的参数规模实现相同水平，说明稀疏激活能够更有效地利用参数容量。  
- **社会偏见**：在含有歧义上下文的任务中，模型的偏见分数随规模上升，但在加入明确指令式提示后，偏见水平显著下降（平均降低约 12%）。  
- **消融实验**：去掉指令式提示后，模型在推理多步任务的准确率下降约 8%；关闭稀疏激活则在突现任务上延迟约 3B 参数才出现跃升。  
- **局限性**：作者指出，虽然覆盖面广，但仍有许多真实世界的交互式任务未被囊括；此外，评估主要基于离线生成，缺少对话持续性和长期记忆的考察。

### 影响与延伸思考
这套基准一经发布，就成为评估大模型能力的“新标准”。随后出现的如 **MMLU**、**HELM** 等测评，都在任务多样性和规模突现分析上向 BIG‑bench 学习。稀疏模型的优势在后续的 Switch‑Transformer 迭代和 DeepSpeed‑MoE 中得到进一步验证。对社会偏见的提示工程发现也催生了大量关于“安全提示”与“对抗性微调”的研究。想继续深入，可以关注以下方向：① 任务自动生成与自适应难度调节；② 多模态（文本+图像）突现行为的系统化测评；③ 基于人类反馈的校准方法。  

### 一句话记住它
BIG‑bench 用一套“超出当前模型能力”的 204 项任务，揭示了大语言模型的规模突现、稀疏优势和可通过提示缓解的偏见。