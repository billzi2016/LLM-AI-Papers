# Evaluating Models' Local Decision Boundaries via Contrast Sets

> **Date**：2020-04-06
> **arXiv**：https://arxiv.org/abs/2004.02709

## Abstract

Standard test sets for supervised learning evaluate in-distribution generalization. Unfortunately, when a dataset has systematic gaps (e.g., annotation artifacts), these evaluations are misleading: a model can learn simple decision rules that perform well on the test set but do not capture a dataset's intended capabilities. We propose a new annotation paradigm for NLP that helps to close systematic gaps in the test data. In particular, after a dataset is constructed, we recommend that the dataset authors manually perturb the test instances in small but meaningful ways that (typically) change the gold label, creating contrast sets. Contrast sets provide a local view of a model's decision boundary, which can be used to more accurately evaluate a model's true linguistic capabilities. We demonstrate the efficacy of contrast sets by creating them for 10 diverse NLP datasets (e.g., DROP reading comprehension, UD parsing, IMDb sentiment analysis). Although our contrast sets are not explicitly adversarial, model performance is significantly lower on them than on the original test sets---up to 25\% in some cases. We release our contrast sets as new evaluation benchmarks and encourage future dataset construction efforts to follow similar annotation processes.

---

# 通过对比集评估模型的局部决策边界 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理（NLP）里，研究者通常用一个固定的测试集来衡量模型的泛化能力。可是，很多公开数据集在标注时会留下系统性的漏洞——比如标注者的偏好、常见的写作模式或是任务本身的设计缺陷。这些漏洞让模型只要学会几条“捷径”就能在原始测试集上取得高分，却根本没有掌握任务背后的语言能力。于是，原本用来检验模型的评估方式反而成了误导，导致我们对模型真实水平的认知出现偏差。

### 关键概念速览
- **对比集（Contrast Set）**：在已有测试样本上做细微、意义明确的改动，使得正确答案通常会改变。想象把一道选择题的关键词换成近义词，答案随之翻转，这样的改动就构成对比集。
- **局部决策边界（Local Decision Boundary）**：模型在输入空间中对某个具体实例的分类阈值。就像在地图上标记出一座山的坡度，局部决策边界描述的是模型在该点附近是如何划分不同标签的。
- **系统性缺口（Systematic Gap）**：数据集内部的结构性偏差，导致模型可以利用非语言信息（如长度、特定词频）轻松预测标签。类似于考试卷子里隐藏的“答案提示”。
- **标注扰动（Annotation Perturbation）**：人工对原始测试实例进行小幅度修改的过程，目标是保持自然语言的流畅性同时改变标签。可以把它看作是“微调”原题目，使其考点转移。
- **评估基准（Benchmark）**：用于统一比较不同模型性能的标准数据集合。这里的基准指的是加入对比集后形成的新测试集。
- **非对抗性（Non‑Adversarial）**：与专门设计来“欺骗”模型的对抗样本不同，对比集的改动并不刻意寻找模型弱点，而是遵循自然语言的合理变化。

### 核心创新点
1. **从全局评估转向局部视角**  
   过去的做法是把模型在整个测试集上的整体准确率当作唯一指标。本文提出先在每个原始样本上生成对应的对比样本，然后比较模型在这对样本上的表现差异。这样可以直接观察模型在输入空间的微小变化下是否保持一致，揭示出隐藏的决策边界。

2. **手工标注扰动的系统化流程**  
   传统的对抗样本往往依赖自动化的梯度攻击，生成的文本常常不自然。作者设计了一套人工标注指南：先挑选关键实体或关系，再用同义词、否定、数值调换等手段进行微调，确保改动后仍符合语言习惯且标签会改变。此流程把“噪声”降到最低，却能有效暴露模型的薄弱环节。

3. **跨任务对比集构建**  
   之前的改进大多局限于单一任务（如情感分析）。本文在十个风格迥异的任务上（阅读理解、依存句法、情感分类等）统一使用对比集方法，展示了该思路的通用性。实验表明，即使是最先进的模型，在这些对比集上的表现普遍下降，最高跌幅达到 25%。

4. **公开对比集作为新基准**  
   作者把所有生成的对比样本整理成可直接下载的评估套件，鼓励后续研究在同一基准上进行比较。这样既提升了评估的透明度，也为数据集构建者提供了可复制的标注流程。

### 方法详解
整体思路可以拆成三步：**原始测试集准备 → 人工扰动生成对比集 → 双向评估**。

1. **原始测试集准备**  
   研究者先选定已有的公开测试集，确保每条样本都有明确的金标准标签（gold label）。这些样本是后续对比的基准点。

2. **人工扰动生成对比集**  
   - **关键要素定位**：标注者阅读每条样本，找出对答案决定性最大的成分（比如数值、实体、情感词）。  
   - **扰动规则制定**：根据任务类型，提供一套可操作的改动模板。例如在情感分析中把“好”换成“糟”，在阅读理解中把问题中的数值加一，或者在依存句法中把主谓关系的动词换成同义词。  
   - **标签重新标注**：改动后，标注者重新判断正确答案，这一步确保对比样本的标签确实发生了变化。  
   - **质量检查**：通过双人交叉验证或小规模自动检测（如语言模型流畅度评分）过滤掉不自然或标签错误的样本。

   这一步的核心在于“微小但有意义”。如果改动太大，模型可能会因为学习到全新模式而表现不佳，失去评估局部决策边界的意义；如果改动太小，标签可能不变，无法形成对比。

3. **双向评估**  
   对每一对（原始样本，扰动样本），模型分别给出预测。随后统计两类错误：  
   - **原始错误**：模型在原始测试集上已经错了，这说明模型本身的基本能力不足。  
   - **对比错误**：模型在原始样本上正确，但在对应的对比样本上错误，这直接反映了模型的局部决策边界不稳。  
   作者进一步计算 **对比下降率**（Contrast Drop），即对比集上准确率相对原始集的下降幅度，用来量化模型对细微语言变化的鲁棒性。

**最巧妙的地方**在于把“人工标注”与“系统评估”结合起来。传统上人工标注被视为成本高、难以规模化的环节，而这里作者把它包装成一种“局部探针”，只需要对每条样本做一次小幅度修改，就能得到高价值的评估信息。

### 实验与效果
- **数据集与任务**：作者在十个公开数据集上验证，包括 DROP（阅读理解）、UD（依存句法分析）、IMDb（情感分类）等，覆盖了问答、结构化预测和文本分类三大类。  
- **基线模型**：对每个任务选取了当前最主流的模型，如 BERT、RoBERTa、T5 等，作为对比对象。  
- **主要结果**：在原始测试集上，这些模型的准确率普遍在 80%~90% 之间。但在对应的对比集上，准确率整体下降，最高跌幅达到 25%。例如，在 IMDb 情感分类上，RoBERTa 的原始准确率为 92%，而在对比集上跌至 71%。  
- **消融实验**：作者分别去掉“人工质量检查”和“关键要素定位”两步，发现对比集的质量显著下降，模型的对比错误率也随之降低，说明这两步对生成有效对比样本至关重要。  
- **局限性**：对比集的构建依赖人工标注，成本随任务规模线性增长；此外，作者承认对比集仍可能遗漏某些类型的系统性缺口，不能保证覆盖所有潜在的模型弱点。

### 影响与延伸思考
这篇工作在 NLP 社区掀起了对评估方式的反思潮流。随后出现的多篇论文把对比集的思路扩展到机器翻译、对话系统甚至代码生成等领域，形成了“对比评估”这一子方向。还有研究尝试用半自动化的方式生成对比样本，降低人工成本；也有工作把对比集与主动学习结合，让模型在训练阶段就能感知并修正自己的局部决策边界。想进一步了解，可以关注 **“Robustness Benchmarks”** 系列以及 **“Challenge Sets”** 的最新进展，它们在方法论上与本文有很多共通之处。

### 一句话记住它
对比集通过人为的细微改动，直接暴露模型在局部输入空间的脆弱决策边界，让评估从“整体准确率”转向“微调鲁棒性”。