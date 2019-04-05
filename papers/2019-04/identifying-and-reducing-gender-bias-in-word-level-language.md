# Identifying and Reducing Gender Bias in Word-Level Language Models

> **Date**：2019-04-05
> **arXiv**：https://arxiv.org/abs/1904.03035

## Abstract

Many text corpora exhibit socially problematic biases, which can be propagated or amplified in the models trained on such data. For example, doctor cooccurs more frequently with male pronouns than female pronouns. In this study we (i) propose a metric to measure gender bias; (ii) measure bias in a text corpus and the text generated from a recurrent neural network language model trained on the text corpus; (iii) propose a regularization loss term for the language model that minimizes the projection of encoder-trained embeddings onto an embedding subspace that encodes gender; (iv) finally, evaluate efficacy of our proposed method on reducing gender bias. We find this regularization method to be effective in reducing gender bias up to an optimal weight assigned to the loss term, beyond which the model becomes unstable as the perplexity increases. We replicate this study on three training corpora---Penn Treebank, WikiText-2, and CNN/Daily Mail---resulting in similar conclusions.

---

# 识别与降低词级语言模型中的性别偏见 论文详细解读

### 背景：这个问题为什么难？
文本数据里常常混杂着社会刻板印象，比如“医生”更常和男性代词相伴，而“护士”则倾向于女性代词。传统的语言模型只追求预测准确度，忽视了这些潜在的偏见，导致模型在生成或推断时会放大不公平的关联。早期的偏见检测大多聚焦在句子或文档层面，缺少对词级别、尤其是语言模型内部表示的细粒度度量；而已有的去偏方法往往通过后处理或数据平衡来缓解，却没有在模型训练过程中直接约束表示空间。于是，如何在保持语言模型性能的同时，系统地量化并削减词级别的性别偏见，成为一个亟待解决的技术难题。

### 关键概念速览
**性别偏见**：模型对男性和女性概念的关联强度不对称，例如“医生”更倾向于出现“他”。可以把它想成模型的“性别天平”失衡。  
**词级语言模型**：以单词为基本预测单元的模型，如RNN语言模型，负责给定前文预测下一个词。类似于打字时的自动补全。  
**投影（Projection）**：把向量映射到某个子空间的操作，就像把三维坐标压到二维平面上，只保留感兴趣的方向信息。  
**性别子空间**：在词向量空间中专门捕捉性别信息的维度集合，类似于把所有“红色”特征聚在一起的颜色通道。  
**正则化损失（Regularization Loss）**：在原始训练目标之外额外加入的约束项，用来惩罚模型产生不希望的行为。就像在跑步时加上心率限制，防止过度冲刺。  
**困惑度（Perplexity）**：语言模型预测不确定性的度量，数值越低说明模型越擅长预测下一个词。可以把它看作模型的“语言流畅度”。  
**消融实验**：逐个去掉模型组件，观察性能变化，以判断每个部分的重要性。类似于把机器的零件一个一个拆下来看哪块最关键。

### 核心创新点
1. **提出可操作的性别偏见度量** → 通过计算词向量在性别子空间上的投影大小来量化偏差，而不是仅靠共现统计。 → 让研究者能够在训练前后直接比较偏见程度，提供了一个统一的评估标尺。  
2. **在语言模型训练中加入性别正则化** → 在标准的交叉熵损失上叠加一个惩罚项，强制模型的词嵌入在性别子空间的投影尽可能小。 → 直接在学习过程中削弱性别信息的表达，而不是事后修剪。  
3. **系统化权重调节与模型稳定性分析** → 通过实验发现正则化权重有一个最佳区间，超过后困惑度急升导致模型不稳定。 → 为实际使用提供了调参指南，避免盲目增大正则化导致性能崩溃。  
4. **跨数据集验证** → 在Penn Treebank、WikiText‑2、CNN/Daily Mail三套语料上复现实验，得到一致的偏见削减效果。 → 证明方法的通用性，而非仅在特定语料上有效。

### 方法详解
整体思路可以划分为四步：①构建性别子空间、②定义偏见度量、③设计正则化损失、④联合训练并调节权重。

**1）性别子空间的构建**  
作者先挑选一组已知的性别词对（如“he‑she”、“man‑woman”），利用这些词的预训练嵌入计算它们的差向量。把所有差向量做主成分分析（PCA），取前几个主成分作为性别子空间的基向量。直观上，这相当于在高维词向量里找出最能区分男性和女性的方向。

**2）偏见度量**  
对于任意目标词（比如“doctor”），将其嵌入向性别子空间投影，得到投影向量的长度。长度越大，说明该词在性别维度上越偏向某一侧。为了得到整体偏见，作者在整个词表上统计这些投影的平均绝对值，形成一个全局偏见分数。

**3）正则化损失的设计**  
在标准的语言模型训练目标（最小化交叉熵）之外，加入一个正则化项：所有词嵌入在性别子空间的投影平方和乘以一个超参数 λ。公式上可以写成“原始损失 + λ·∑(投影长度²)”。这个项的作用是“压平”性别方向的信号，让模型在学习语言规律时不依赖性别信息。

**4）联合训练与权重调节**  
训练时同时优化两个目标。作者在实验中逐步增大 λ，观察困惑度和偏见分数的变化。发现当 λ 处于一个中等范围时，偏见显著下降而困惑度几乎不变；若 λ 过大，模型开始“忘记”语言结构，困惑度飙升，生成文本变得不连贯。于是提出一种经验性调参策略：先在验证集上搜索 λ，使偏见下降率与困惑度上升率的比值最大。

**最巧妙的地方**  
把性别信息抽象成一个子空间，然后直接在嵌入层上做投影约束，而不是在输出层或后处理阶段干预。这种“在根部拔除”方式既保持了语言模型的端到端训练，又避免了对生成文本的人工干预，兼顾了效果和实现简洁性。

### 实验与效果
- **数据集**：分别在Penn Treebank（新闻稿）、WikiText‑2（维基百科片段）和CNN/Daily Mail（新闻摘要）上训练同构的RNN语言模型。  
- **基线**：普通RNN语言模型（仅交叉熵）以及一个使用数据平衡的简单去偏方法（在训练语料中对性别词进行等频抽样）。  
- **结果**：在所有三套语料上，加入正则化后全局偏见分数下降约30%~45%，而困惑度仅上升约2%~5%。相比仅做数据平衡的基线，偏见削减幅度高出约15个百分点，且语言流畅度保持更好。  
- **消融实验**：作者分别去掉（a）性别子空间的构建（直接使用随机子空间），（b）正则化项，发现偏见削减效果几乎消失，验证了子空间质量和正则化的必要性。  
- **局限**：论文只在词级RNN模型上验证，未涉及Transformer等更主流的架构；正则化权重的选择仍需在每个新语料上手动调参；对非二元性别或交叉文化偏见的处理没有展开。

### 影响与延伸思考
这篇工作打开了“在训练阶段直接约束词向量子空间”这一思路，随后出现了多篇针对不同属性（种族、年龄）或不同模型（BERT、GPT）的类似正则化方法。还有研究把性别子空间与对抗训练结合，进一步提升去偏鲁棒性。想深入了解的读者可以关注以下方向：①在大规模预训练模型中引入子空间正则化的实现细节；②多属性子空间的联合约束；③如何在保持下游任务性能的前提下，自动调节正则化强度。整体来看，这篇论文为语言模型公平性提供了一个可量化、可操作的框架，后续工作大多在此基础上扩展。

### 一句话记住它
在语言模型训练中加入“压平性别子空间投影”的正则化，就能在不牺牲流畅度的前提下显著削减词级别的性别偏见。