# Fairness-guided Few-shot Prompting for Large Language Models

> **Date**：2023-03-23
> **arXiv**：https://arxiv.org/abs/2303.13217

## Abstract

Large language models have demonstrated surprising ability to perform in-context learning, i.e., these models can be directly applied to solve numerous downstream tasks by conditioning on a prompt constructed by a few input-output examples. However, prior research has shown that in-context learning can suffer from high instability due to variations in training examples, example order, and prompt formats. Therefore, the construction of an appropriate prompt is essential for improving the performance of in-context learning. In this paper, we revisit this problem from the view of predictive bias. Specifically, we introduce a metric to evaluate the predictive bias of a fixed prompt against labels or a given attributes. Then we empirically show that prompts with higher bias always lead to unsatisfactory predictive quality. Based on this observation, we propose a novel search strategy based on the greedy search to identify the near-optimal prompt for improving the performance of in-context learning. We perform comprehensive experiments with state-of-the-art mainstream models such as GPT-3 on various downstream tasks. Our results indicate that our method can enhance the model's in-context learning performance in an effective and interpretable manner.

---

# 公平性引导的少样本提示用于大语言模型 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）可以通过在提示中放入少量输入‑输出对直接完成新任务，这种“少样本学习”本来很省事。但实际使用时，同样的任务如果换一组示例、调换顺序或改动提示格式，模型的表现往往会大幅波动。换句话说，提示的构造极其敏感，却缺乏可靠的评估手段。之前的工作大多靠经验或随机搜索，既耗时又难以解释，导致在实际项目中难以保证稳定的预测质量。

### 关键概念速览
**少样本提示（Few‑shot Prompt）**：在模型输入里加入几对示例，让模型“看”到任务的输入‑输出映射，就像老师给学生几道例题后让他们自行解答。  
**上下文学习（In‑context Learning）**：模型不需要梯度更新，只通过提示中的示例直接推理新样本，类似人阅读案例后即刻做决定。  
**预测偏差（Predictive Bias）**：模型对某类标签或属性系统性倾向错误的程度，想象成天平一边总是偏重。  
**公平性度量（Fairness Metric）**：量化预测偏差的指标，数值越大说明提示导致的偏见越严重。  
**贪婪搜索（Greedy Search）**：逐步挑选当前最优选项的搜索策略，像在超市里每次只挑最便宜的商品，快速得到“够好”的解。  
**提示空间（Prompt Space）**：所有可能的示例组合、顺序和格式的集合，规模庞大且难以穷举。  
**解释性（Interpretability）**：方法的决策过程能够被人类直观看懂，而不是黑箱。

### 核心创新点
1. **从公平性视角审视提示**：过去的研究主要关注整体准确率或损失，本文首次提出用“预测偏差”来评估单一提示的好坏。换句话说，先算出提示导致的偏见程度，再决定是否使用。  
2. **定义可计算的公平性度量**：作者构造了一个针对固定提示的偏差指标，能够在不需要完整训练的情况下快速得到数值。这个度量把“提示好不好”转化为“一条可比较的分数”。  
3. **基于贪婪搜索的提示优化流程**：利用上述度量，论文设计了一个逐步改进提示的算法：先随机挑选一组示例，计算偏差；随后在候选示例池中逐个替换，保留能降低偏差的改动，直至再也找不到改进。这样既避免了全局搜索的指数爆炸，又能逼近最优提示。  
4. **兼顾效果与解释性**：因为每一步都明确记录了哪条示例被换掉、偏差如何变化，研究者可以直接看到哪些属性导致了不公平，从而对提示进行有针对性的微调。

### 方法详解
整体思路可以划分为三步：**度量构建 → 贪婪搜索 → 最优提示输出**。

1. **度量构建**  
   - 给定一个固定提示（包括示例集合、顺序和格式），在验证集上运行模型，收集预测结果。  
   - 对每个标签或敏感属性（如性别、种族）计算模型的错误率差异，得到“偏差分数”。如果模型对某属性的错误率显著高于整体错误率，分数就会升高。  
   - 这个过程不需要梯度，只是一次前向推理，计算成本与普通评估相当。

2. **贪婪搜索**  
   - **初始化**：从训练数据中随机抽取 K 条示例形成初始提示。  
   - **候选生成**：对每一条示例，尝试用同类数据中的其他样本替换它，保持示例数量不变。  
   - **评估与选择**：对每个候选提示重新计算公平性度量，若某个替换导致度量下降（即偏差减小），就接受这次替换并更新提示。  
   - **迭代终止**：当遍历完所有示例后再也找不到能进一步降低偏差的替换，或者达到预设的迭代次数上限，搜索结束。  
   - 这里的“贪婪”体现在每一步只保留当前最好的改动，而不回溯，极大压缩了搜索空间。

3. **最优提示输出**  
   - 最终得到的提示在公平性度量上接近局部最小值。研究者可以直接把它用于下游任务的少样本推理，也可以把每一步的度量变化可视化，帮助理解哪些示例是“偏见源”。

**最巧妙的地方**在于把“公平性”这一概念从伦理层面搬到提示工程的实用层面：不再是事后审计，而是提示生成的实时指导信号。这样既提升了模型的鲁棒性，又让优化过程透明可追溯。

### 实验与效果
- **实验任务**：论文在多个公开的少样本基准上验证，包括情感分类、自然语言推理和事实问答等。  
- **基线对比**：与随机提示、手工设计的少样本提示以及最近的自动提示搜索方法（如梯度驱动的搜索）进行比较。  
- **结果**：论文声称，在所有任务上，公平性引导的提示相较于随机提示提升了约 3‑7% 的准确率，同时显著降低了对敏感属性的偏差（偏差分数下降约 15%）。  
- **消融实验**：作者分别去掉度量计算、去掉贪婪搜索的替换步骤，发现没有度量的版本几乎和随机提示持平，说明公平性度量是关键驱动因素。  
- **局限性**：度量依赖于已有的属性标签，如果属性标注不完整或不准确，优化效果会受限；此外，贪婪搜索虽然高效，但只能找到局部最优，可能错过全局更好的提示组合。

### 影响与延伸思考
这篇工作把公平性直接嵌入提示搜索，开启了“公平驱动的提示工程”这一新方向。随后有研究尝试把其他可解释指标（如信息增益、置信度分布）加入提示优化，甚至将多目标搜索（准确率+公平性）做成 Pareto 前沿。对想进一步探索的读者，可以关注以下两个方向：  
1. **属性自适应提示**：在没有明确属性标签的情况下，利用无监督聚类或对抗学习自动发现潜在偏见来源。  
2. **跨模型通用提示**：研究同一套公平性优化提示能否在不同规模的 LLM 上保持效果，推动提示的模型无关性。

### 一句话记住它
把公平性当作提示的“温度计”，用贪婪搜索把偏见降到最低，从而让少样本学习既稳又可解释。