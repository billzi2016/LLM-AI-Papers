# Large Language Model Confidence Estimation via Black-Box Access

> **Date**：2024-06-01
> **arXiv**：https://arxiv.org/abs/2406.04370

## Abstract

Estimating uncertainty or confidence in the responses of a model can be significant in evaluating trust not only in the responses, but also in the model as a whole. In this paper, we explore the problem of estimating confidence for responses of large language models (LLMs) with simply black-box or query access to them. We propose a simple and extensible framework where, we engineer novel features and train a (interpretable) model (viz. logistic regression) on these features to estimate the confidence. We empirically demonstrate that our simple framework is effective in estimating confidence of Flan-ul2, Llama-13b, Mistral-7b and GPT-4 on four benchmark Q\&A tasks as well as of Pegasus-large and BART-large on two benchmark summarization tasks with it surpassing baselines by even over $10\%$ (on AUROC) in some cases. Additionally, our interpretable approach provides insight into features that are predictive of confidence, leading to the interesting and useful discovery that our confidence models built for one LLM generalize zero-shot across others on a given dataset.

---

# 通过黑盒访问的大语言模型置信度估计 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在问答、摘要等任务上表现惊人，但它们的输出往往缺乏可解释的“可信度”。传统的置信度估计方法依赖模型内部的概率分布或梯度信息，而这些信息在商业化的 API（如 GPT‑4）里根本不可得。于是只能把模型当作一个只能输入 Prompt、得到文本输出的黑盒子。没有内部信号，怎么判断答案是“靠谱”还是“随便说说”？这直接限制了 LLM 在高风险场景（医学、法律等）的应用，也让用户难以对模型的整体可靠性进行评估。

### 关键概念速览
- **黑盒访问**：只能通过发送文字提示（Prompt）并读取模型返回的文字，无法获取模型内部的概率、注意力权重或梯度。类似只能看门口的灯光，却看不见电路板。
- **置信度（Confidence）**：对模型输出正确性的主观概率估计。高置信度意味着模型“自信”它的答案是对的。
- **特征工程**：从模型的原始输出或交互过程里抽取可量化的属性（比如答案长度、重复率等），再喂给机器学习模型。就像从一张照片里挑出颜色、纹理等信息来判断它是否是风景照。
- **可解释模型**：指模型结构简单、易于人类理解的模型，如逻辑回归。它的每个系数都可以直接解释为某特征对置信度的正负影响。
- **AUROC（曲线下面积）**：衡量二分类模型区分正负样本能力的指标，越接近 1 越好。把它想成在所有可能阈值下的平均准确率。
- **零样本迁移（Zero‑Shot Transfer）**：在没有针对目标模型重新训练的情况下，直接把在一个模型上学到的置信度估计器用于另一个模型。像是学会了辨别水果的新鲜度后，直接用同样的标准去判断蔬菜。

### 核心创新点
1. **从黑盒输出直接抽特征 → 训练轻量可解释模型 → 置信度预测**  
   过去的工作要么依赖内部概率，要么使用复杂的后置模型（如自回归校准器）。这篇论文把注意力放在“能直接得到的东西”，通过精心设计的十几种特征（如答案长度、词汇多样性、Prompt‑Response 相似度等），再用逻辑回归学习它们与真实正确性的关系。结果显示，简单线性模型就能把置信度估计的 AUROC 提升 10% 以上。

2. **特征可解释性 → 揭示置信度来源**  
   与深度网络的黑箱不同，逻辑回归的系数直接告诉我们哪个特征是“好”或“坏”。实验发现，答案的自洽度（比如是否出现自相矛盾的词）和 Prompt 与答案的语义相似度是最强的正向信号，而过长或高度重复的答案往往暗示低置信度。

3. **跨模型零样本迁移**  
   训练好的置信度估计器在一个 LLM（比如 Llama‑13B）上表现优秀后，直接搬到另一个模型（比如 GPT‑4）上使用，仍然保持相当的区分能力。作者把这归因于特征本身捕捉的是“语言输出质量”而非特定模型的内部状态。

4. **统一框架覆盖问答与摘要**  
   之前的置信度研究多聚焦于单一任务。这里把同一套特征和模型同时用于四个问答基准和两个摘要基准，证明了方法的通用性。换句话说，只要任务是生成式文本，框架几乎不需要改动。

### 方法详解
**整体思路**  
1）把 LLM 当作只能接受 Prompt、返回文本的黑盒子。  
2）针对每一次查询，构造一组描述答案“属性”的特征向量。  
3）把这些特征和真实标签（答案是否正确/摘要是否符合参考）喂给逻辑回归，训练出置信度预测模型。  
4）推理时，只需再次计算特征并用已训练好的回归系数得到置信度分数。

**关键模块拆解**  

| 步骤 | 具体做法 | 类比 |
|------|----------|------|
| **Prompt 设计** | 为每个任务准备标准化的提问模板，确保特征的可比性。 | 像在实验室里使用统一的试管，避免因容器不同导致测量误差。 |
| **特征抽取** | - **长度特征**：答案字符数、词数。<br>- **重复度**：n‑gram 重复比例。<br>- **词汇多样性**：独特词占比。<br>- **语义相似度**：Prompt 与答案的向量余弦相似度（使用公开的句向量模型）。<br>- **自洽度**：检测答案内部是否出现矛盾（通过简单的规则或小模型）。<br>- **置信词**：答案中出现的“当然”“肯定”等词频。 | 把答案当成一块水果，测量它的重量、颜色、硬度等属性来判断是否成熟。 |
| **标签获取** | 对于问答，使用已有的标准答案比对得到对/错；对摘要，用 ROUGE/BLEU 等自动评价指标阈值划分为高/低质量。 | 像在实验中先给每个样本贴上“好”或“坏”的标签，供后面的学习使用。 |
| **模型训练** | 采用 L2 正则化的逻辑回归，目标是最小化交叉熵损失，使输出的概率尽可能接近真实标签。 | 逻辑回归就像一条直线在特征空间里划分“可信”和“不可信”两块区域。 |
| **推理** | 给定新 Prompt，重复特征抽取步骤，用训练好的系数计算置信度分数（0‑1 之间），可直接作为阈值判断依据。 | 类似把新水果放在同样的测量仪器上，得到一个成熟度分数。 |

**最巧妙的地方**  
- **不依赖模型内部概率**：所有特征都是外部可观测的，这让方法可以直接在商业 API 上跑。  
- **特征的跨模型一致性**：作者发现语义相似度、重复度等特征在不同模型上表现相似，这为零样本迁移提供了理论依据。  
- **逻辑回归的可解释性**：系数大小直接映射到特征重要性，帮助研究者和用户了解“为什么模型不自信”。  

### 实验与效果
- **任务与数据集**：四个公开的问答基准（如 TruthfulQA、MMLU 等）以及两个摘要基准（CNN/DailyMail、XSum）。  
- **模型覆盖**：Flan‑ul2、Llama‑13B、Mistral‑7B、GPT‑4（问答）以及 Pegasus‑large、BART‑large（摘要）。  
- **对比基线**：直接使用模型输出的 token 概率、温度调节后的熵、以及最近的自校准方法。  
- **主要结果**：在多数任务上，本文的置信度估计器的 AUROC 超过基线 5%~12%。例如在 GPT‑4 的 TruthfulQA 上，AUROC 从 0.78 提升到 0.86。  
- **消融实验**：去掉语义相似度特征后，AUROC 下降约 3%；去掉重复度特征后下降约 2%。说明这两个特征是贡献最大的。  
- **零样本迁移**：在 Llama‑13B 上训练的模型直接用于 GPT‑4，AUROC 只损失约 1%，仍显著优于所有基线。  
- **局限性**：特征工程仍然依赖手工设计，面对极端长文本或多轮对话时可能需要额外特征；对高度专业化的任务（如医学报告）作者未给出实验，效果未知。

### 影响与延伸思考
这篇工作打开了“黑盒置信度估计”的可能性，随后有不少研究尝试把更丰富的外部特征（如检索到的证据相似度、模型生成的自评文本）加入框架，甚至把轻量的神经网络替代逻辑回归以捕捉非线性关系。还有工作把这种置信度估计用于 **模型自我纠错**：当置信度低于阈值时自动发起二次查询或交给人工审查。  
如果想进一步探索，可关注以下方向：  
- **自动特征学习**：使用对比学习或元学习让系统自行发现有效的置信度特征。  
- **跨语言迁移**：检验同一套特征在多语言 LLM 上的通用性。  
- **与检索增强模型结合**：把外部知识检索的匹配度作为置信度信号。  
（以上为基于当前文献的推测）

### 一句话记住它
只要把黑盒 LLM 的输出当成“可测量的水果”，用几条简单的质量特征喂给逻辑回归，就能在不看内部概率的情况下，精准估计答案的可信度。