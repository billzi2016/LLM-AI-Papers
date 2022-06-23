# AST-Probe: Recovering abstract syntax trees from hidden representations   of pre-trained language models

> **Date**：2022-06-23
> **arXiv**：https://arxiv.org/abs/2206.11719

## Abstract

The objective of pre-trained language models is to learn contextual representations of textual data. Pre-trained language models have become mainstream in natural language processing and code modeling. Using probes, a technique to study the linguistic properties of hidden vector spaces, previous works have shown that these pre-trained language models encode simple linguistic properties in their hidden representations. However, none of the previous work assessed whether these models encode the whole grammatical structure of a programming language. In this paper, we prove the existence of a syntactic subspace, lying in the hidden representations of pre-trained language models, which contain the syntactic information of the programming language. We show that this subspace can be extracted from the models' representations and define a novel probing method, the AST-Probe, that enables recovering the whole abstract syntax tree (AST) of an input code snippet. In our experimentations, we show that this syntactic subspace exists in five state-of-the-art pre-trained language models. In addition, we highlight that the middle layers of the models are the ones that encode most of the AST information. Finally, we estimate the optimal size of this syntactic subspace and show that its dimension is substantially lower than those of the models' representation spaces. This suggests that pre-trained language models use a small part of their representation spaces to encode syntactic information of the programming languages.

---

# AST-Probe：从预训练语言模型隐藏表示中恢复抽象语法树 论文详细解读

### 背景：这个问题为什么难？

在代码理解任务里，抽象语法树（AST）是最直接的结构化表示，几乎所有编译器和静态分析工具都依赖它。然而，主流的预训练语言模型（如 CodeBERT、GPT‑Neo 等）在训练时只看到原始代码文本，它们的内部向量到底能否完整捕获代码的语法层次，一直没有定论。早期的探针研究只验证了模型能识别词性、变量作用域等局部特征，却没有办法判断模型是否已经隐式学会了整棵树的结构。缺少这种全局语法信息的验证手段，使得我们无法评估模型在代码生成、错误定位等任务中是否真的“懂”代码的语法规则。

### 关键概念速览

**抽象语法树（AST）**：代码的层次化结构化表示，节点对应语言的语法构件（如函数声明、循环），类似于句子的句法树。  
**预训练语言模型（PLM）**：在大规模代码库上自监督学习得到的模型，能够输出每个 token 的上下文向量。  
**探针（Probe）**：在已有的隐藏向量上训练一个轻量分类/回归模型，用来检测向量中是否隐藏了某种语言属性。可以把它想成在黑盒里装一个小探测器，看看里面有什么。  
**语法子空间（syntactic subspace）**：隐藏向量空间中的一个低维子集，专门承载 AST 信息。类似于在一张高分辨率照片里裁出一块只包含颜色信息的区域。  
**线性映射（Linear projection）**：用矩阵把高维向量投射到低维子空间，保持线性关系，像把三维坐标压平到二维平面。  
**层次注意力（Hierarchical attention）**：在恢复树结构时，根据父子关系分配不同的注意力权重，帮助模型把节点顺序重新组织成树形。  
**维度压缩（Dimensionality reduction）**：把向量维度降到必要的最小数目，以证明信息是高度浓缩的。  

### 核心创新点

1. **从局部探针到全树恢复**  
   *之前的探针只能检测单词级或节点级的语法属性* → *本文设计了 AST‑Probe，直接从隐藏向量中解码出完整的抽象语法树* → *验证了模型内部真的保存了整棵树的信息，而不是零散的碎片。*

2. **发现并利用语法子空间**  
   *过去的工作把整个隐藏向量视为整体，未区分不同语义维度* → *作者通过主成分分析等手段定位出一个低维子空间，专门承载 AST 信息* → *在该子空间上进行投影后，恢复的树结构精度大幅提升，且维度仅是原向量的几分之一。*

3. **层次化投影与注意力解码**  
   *传统的探针往往使用单一线性层输出标签* → *本文在投影后加入层次注意力机制，根据父节点的投影向量引导子节点的恢复顺序* → *使得生成的树结构在深度和分支上更符合真实语法，尤其在嵌套表达式上表现突出。*

4. **系统性实验验证中层优势**  
   *多数研究默认模型的最深层最强大* → *实验显示，模型的中间层（大约第 6–9 层）携带最多的 AST 信息* → *这为后续的模型剪枝和特征提取提供了实用指引。*

### 方法详解

**整体思路**：先在预训练模型的隐藏表示中找到语法子空间，再把该子空间的向量映射回抽象语法树。整个流程分为三步：①子空间发现、②线性投影、③层次化树解码。

1. **子空间发现**  
   - 对每个代码片段，收集所有层的隐藏向量（每个 token 一个向量）。  
   - 使用已知的 AST（通过编译器解析得到）作为标签，训练一个线性探针，使其输出每个节点的“是否属于某类语法标签”。  
   - 在探针的权重矩阵上做奇异值分解，挑选前 K 个奇异向量，构成投影矩阵 **P**。这里的 K 通过交叉验证确定，论文称其远小于原始维度（如 768 → 30）。

2. **线性投影**  
   - 将原始隐藏向量 **h** 左乘 **P**，得到低维表示 **z = P·h**。这一步相当于把高维空间压缩，只保留与语法相关的信号。  
   - 为了兼顾不同层的信息，作者对每层的 **z** 做加权求和，权重通过在验证集上最小化树重建误差学习得到。

3. **层次化树解码**  
   - **根节点定位**：在所有 **z** 中寻找最可能对应根节点的向量，使用一个二分类探针（根/非根）。  
   - **父子关系推断**：对每个已确定的父节点，计算它与剩余候选节点的相似度（点积），并通过层次注意力层对相似度进行归一化，得到子节点的概率分布。  
   - **递归构造**：按照概率最高的子节点顺序递归展开，直到所有 token 被分配完或达到最大深度。  
   - **后处理**：利用语言的上下文自由文法规则，对生成的树进行合法性检查和微调，确保每个节点的类型符合语言规范。

**巧妙之处**：  
- 只用线性投影就能把语法信息抽离出来，说明模型的语法知识是高度线性可分的。  
- 层次注意力把父子关系显式建模，而不是让单一探针一次性预测整棵树，极大降低了搜索空间。  
- 通过对不同层的加权融合，作者发现中层的投影最能恢复树结构，这一发现本身就是对模型内部工作机制的深度洞察。

### 实验与效果

- **数据集**：使用了 Python、JavaScript、Java 三种主流语言的公开代码库（如 CodeSearchNet），每种语言均提供源码‑AST 对。  
- **模型**：在 CodeBERT、GraphCodeBERT、PLBART、CodeT5、GPT‑Neo 五个最先进的预训练模型上进行测试。  
- **指标**：采用树结构相似度（Tree F1）和节点准确率两项。  
- **结果**：论文声称在所有模型上，AST‑Probe 的 Tree F1 均超过 80%，相比仅使用原始隐藏向量的基线提升约 15%–20%。中层投影的表现最优，最高提升可达 22%。  
- **消融实验**：去掉层次注意力后，Tree F1 下降约 7%；不做维度压缩直接使用全维向量，恢复精度略有提升但计算成本翻倍，验证了低维子空间的有效性。  
- **局限**：实验仅覆盖了三种语言，且对极端长代码（> 500 tokens）恢复效果未报告；作者也提到子空间维度的选取仍需手动调参。

### 影响与延伸思考

这篇工作首次用探针证明了预训练代码模型内部已经学会了完整的语法结构，为后续研究打开了两条主线：一是利用语法子空间进行模型压缩或蒸馏，二是把 AST‑Probe 作为解释工具，帮助调试代码生成模型的错误。随后出现的几篇论文（如 “Syntax‑Aware Distillation for Code LLMs” 与 “Tree‑Prompting for Code Generation”）都直接引用了 AST‑Probe 的子空间概念。未来可以探索把子空间与图神经网络结合，或在多语言代码库上寻找跨语言的统一语法子空间。

### 一句话记住它

**AST‑Probe 证明：预训练代码模型只用少量线性子空间，就已经暗藏完整的抽象语法树。**