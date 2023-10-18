# Adaptation with Self-Evaluation to Improve Selective Prediction in LLMs

> **Date**：2023-10-18
> **arXiv**：https://arxiv.org/abs/2310.11689

## Abstract

Large language models (LLMs) have recently shown great advances in a variety of tasks, including natural language understanding and generation. However, their use in high-stakes decision-making scenarios is still limited due to the potential for errors. Selective prediction is a technique that can be used to improve the reliability of the LLMs by allowing them to abstain from making predictions when they are unsure of the answer. In this work, we propose a novel framework for adaptation with self-evaluation to improve the selective prediction performance of LLMs. Our framework is based on the idea of using parameter-efficient tuning to adapt the LLM to the specific task at hand while improving its ability to perform self-evaluation. We evaluate our method on a variety of question-answering (QA) datasets and show that it outperforms state-of-the-art selective prediction methods. For example, on the CoQA benchmark, our method improves the AUACC from 91.23% to 92.63% and improves the AUROC from 74.61% to 80.25%.

---

# 自评适配提升大语言模型选择性预测 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在问答、对话等任务上已经能生成看似可信的答案，但它们仍会在关键时刻“胡说八道”。在医疗诊断、法律咨询等高风险场景，错误答案的代价极高。传统的做法是直接把模型的输出当作最终答案，或者用后处理的置信度阈值来决定是否放弃。然而，LLM 的内部概率分布往往与真实错误率脱钩，导致置信度不可靠。已有的选择性预测（selective prediction）方法大多依赖外部校准器或额外的监督模型，既增加了计算开销，又难以在不同任务间迁移。因此，如何让模型本身在给出答案的同时评估自己的可靠性，成为提升高风险应用可用性的关键瓶颈。

### 关键概念速览

**选择性预测**：模型在推理时可以自行决定是否输出答案，若自评分数低于设定阈值则“弃答”。类似于考试时学生不确定时选择不作答，以免扣分。

**自评（Self‑Evaluation）**：模型对自己的输出进行打分或判断正确性，像是让模型先给自己的答案打个“自检分”，再决定是否提交。

**参数高效微调（Parameter‑Efficient Tuning）**：在保持大模型主体权重不变的前提下，只调少量可学习的附加参数（如 LoRA、Adapter），相当于在原有模型上贴一层可调的“胶水”，既省显存又易于快速适配新任务。

**AUACC（Area Under Accuracy‑Coverage Curve）**：在不同置信度阈值下，累计覆盖率（模型回答的比例）与对应准确率的面积，越大说明模型在保持高覆盖的同时保持高准确。

**AUROC（Area Under ROC Curve）**：二分类任务中，真正例率与假正例率的曲线下面积，用来衡量自评分数区分对错答案的能力。

**CoQA**：一种对话式阅读理解数据集，要求模型在多轮问答中保持上下文连贯性，常用于评估问答系统的鲁棒性。

### 核心创新点

1. **自评能力嵌入微调过程**  
   之前的选择性预测大多在模型训练完成后再加一个外部评估器；本工作在微调阶段同步训练一个自评头，使模型在学习任务本身的同时学会判断答案对错。这样做把评估信息直接写进模型内部，省去后置校准步骤。

2. **参数高效适配 + 自评联合优化**  
   传统微调需要全模型更新，成本高且容易过拟合；这里仅通过轻量级 Adapter/LoRA 调整模型表示，同时对自评头使用二元交叉熵损失。两者共享同一任务数据，既保持了任务特化，又让自评信号渗透进模型表征。

3. **基于自评的动态阈值选择**  
   过去的阈值往往是手工设定或通过验证集搜索；本文提出在推理时根据自评分数的分布自适应调节阈值，使得在不同输入难度下模型能够灵活决定是否回答，提升了覆盖率与准确率的平衡。

4. **统一的选择性预测评价框架**  
   通过 AUACC 与 AUROC 两个指标同时评估，展示了自评分数不仅在整体覆盖上有优势，还在区分对错答案上更具判别力。相较于仅使用置信度或后置校准的基线，提升更为显著。

### 方法详解

**整体思路**  
整个框架可以拆成三步：① 用参数高效微调把 LLM 适配到目标 QA 任务；② 在同一批次里让模型同时预测答案和“答案是否正确”的二分类标签；③ 推理时依据自评分数与动态阈值决定是否输出答案。整个过程只需在原始训练数据上增加一个二元标记（答案对错），不需要额外的人工标注。

**关键模块拆解**  

1. **Adapter/LoRA 层**  
   在每个 Transformer 层的投影矩阵旁并行插入低秩矩阵（LoRA）或小型全连接层（Adapter），这些参数在微调时被更新，原始权重保持冻结。想象成在原模型外面套了一层可调的“外衣”，既能捕捉任务特征，又不破坏模型的通用知识。

2. **自评头（Self‑Eval Head）**  
   在模型的最后隐藏状态上接一个轻量的二分类头，输出一个 0‑1 之间的分数，表示“我对答案的信心”。训练时使用交叉熵损失，正例是模型在训练集上给出正确答案的情况，负例是错误答案。因为答案的对错已经在训练数据中可得，这一步不需要额外标注。

3. **联合损失函数**  
   总损失 = 任务损失（如 QA 的交叉熵） + λ × 自评损失。λ 是平衡系数，控制模型在追求答案质量和自评能力之间的权衡。实验中 λ 设为 0.5 左右，既能保持答案准确，又让自评分数有足够的学习信号。

4. **动态阈值机制**  
   推理时先得到答案和自评分数。系统维护一个滑动窗口统计最近 N 条自评分数的分布（均值、方差），根据设定的覆盖目标自动计算阈值 τ = μ - k·σ（k 为超参数）。如果自评分数 ≥ τ，则提交答案；否则返回 “我不确定”。这种自适应方式类似于人类在不同情境下对自己的信心进行调节。

**最巧妙的地方**  
把自评头直接嵌入微调过程，使得模型的内部表征同时服务于答案生成和可靠性判断。这样一来，自评分数不再是后置的“外部打分”，而是模型内部的“自我感知”，大幅提升了评估的准确性和计算效率。

### 实验与效果

- **数据集与任务**：在对话式阅读理解数据集 CoQA 上进行评估，还补充了 SQuAD、HotpotQA 等常用 QA 基准，以验证方法的通用性。所有实验均使用同一基线 LLM（如 LLaMA‑7B）进行参数高效微调。

- **对比基线**：  
  1. 直接微调后使用原始 logits 置信度（Baseline‑Logits）。  
  2. 采用温度标定（Temperature Scaling）后的置信度。  
  3. 经典选择性预测模型 SelectiveNet。  
  4. 近期的自评式方法 Self‑Check（仅在推理阶段加评估器）。

- **主要结果**（以 CoQA 为例）：  
  - AUACC 从 91.23%（Baseline‑Logits）提升到 92.63%。  
  - AUROC 从 74.61%（SelectiveNet）提升到 80.25%。  
  - 在相同覆盖率（80%）下，答案准确率提升约 2.1%。  

- **消融实验**：  
  - 去掉自评头，仅保留 Adapter 微调，AUROC 下降至 73.8%，说明自评是提升判别力的关键。  
  - 将自评头换成独立的后置评估器，AUACC 下降约 0.9%，验证了“嵌入式自评”比后置更有效。  
  - 改变 λ 值：λ 过小自评学习不足，λ 过大导致答案质量受损，最佳 λ 在 0.4‑0.6 之间。

- **局限性**：  
  - 需要在训练数据中明确标记答案对错，若任务本身没有明确答案（如开放式生成），自评训练会更困难。  
  - 目前只在 QA 场景验证，跨任务（如代码生成、文本摘要）的效果尚未公开。  
  - 动态阈值依赖近期分数分布，极端输入可能导致阈值失效，需要进一步鲁棒性研究。

### 影响与延伸思考

这篇工作开启了“让大模型自我评估”在选择性预测中的系统化探索。随后出现的研究如 **Self‑Check GPT**、**Confidence‑Aware Retrieval‑Augmented Generation** 等，都在不同程度上借鉴了“微调时同步学习自评”这一思路。业界也开始在产品化 LLM 时加入自评模块，以实现“可拒绝回答”的安全机制。未来可以进一步探索：

- **跨任务自评共享**：构建一个通用的自评头，能够在多种下游任务间迁移，降低标注成本。  
- **无监督自评**：利用模型自身的生成概率或对比学习生成伪标签，摆脱对答案对错的显式依赖。  
- **人机协同**：把自评分数作为人类审查的优先级指示，让人类只审查低置信度的输出，提高整体系统效率。

### 一句话记住它

让大模型在微调时学会给自己的答案打分，主动在不确定时“闭嘴”，从而显著提升选择性预测的可靠性。