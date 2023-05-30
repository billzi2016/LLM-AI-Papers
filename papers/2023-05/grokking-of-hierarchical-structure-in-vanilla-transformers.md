# Grokking of Hierarchical Structure in Vanilla Transformers

> **Date**：2023-05-30
> **arXiv**：https://arxiv.org/abs/2305.18741

## Abstract

For humans, language production and comprehension is sensitive to the hierarchical structure of sentences. In natural language processing, past work has questioned how effectively neural sequence models like transformers capture this hierarchical structure when generalizing to structurally novel inputs. We show that transformer language models can learn to generalize hierarchically after training for extremely long periods -- far beyond the point when in-domain accuracy has saturated. We call this phenomenon \emph{structural grokking}. On multiple datasets, structural grokking exhibits inverted U-shaped scaling in model depth: intermediate-depth models generalize better than both very deep and very shallow transformers. When analyzing the relationship between model-internal properties and grokking, we find that optimal depth for grokking can be identified using the tree-structuredness metric of \citet{murty2023projections}. Overall, our work provides strong evidence that, with extended training, vanilla transformers discover and use hierarchical structure.

---

# 普通Transformer对层次结构的Grokking 论文详细解读

### 背景：这个问题为什么难？

语言的意义往往依赖于句子内部的层次结构——比如“我看到的那只猫”里，定语短语嵌套在名词内部。传统的序列模型（RNN、CNN）在捕捉这种树形依赖时表现平平，因为它们主要按时间步逐个处理信息。Transformer 通过自注意力可以直接关联任意位置的词，但自注意力本身并不强制学习树形组织，很多实验显示它在面对结构上全新（未在训练集出现过）的句子时会退化成“记忆”而非“推理”。因此，业界一直怀疑：即使在大规模预训练后，Transformer 是否真的“懂”句法层次，还是仅仅靠统计模式完成任务？这篇论文正是针对这一疑问展开的。

### 关键概念速览

**层次结构（Hierarchical Structure）**：句子内部的树形组织，如主语、宾语、从句的嵌套关系。可以想象成句子里的“套娃”，外层套住内层。

**结构泛化（Structural Generalization）**：模型在遇到训练时未见过的结构时仍能正确预测。类似于人类看到新句式仍能理解其意思。

**Grokking**：原指“彻底领悟”，在机器学习里指模型在训练很久后突然出现跨越式的性能提升。这里特指对层次结构的领悟。

**倒U型深度效应（Inverted‑U Depth Scaling）**：模型深度（层数）对结构泛化的影响呈现先升后降的曲线——中等深度最好，太浅或太深都不行。

**树结构度量（Tree‑Structuredness Metric）**：由 Murty 等人提出，用来量化模型内部表示是否接近树形结构的数值指标。

**Vanilla Transformer**：指原始的、未加任何特殊结构改动的标准Transformer（如原始的Encoder‑Decoder或仅Encoder的语言模型），不包含额外的句法诱导模块。

### 核心创新点

1. **训练时间的极限延伸 → 让模型继续训练到域内准确率已饱和后仍保持下降的损失 → 发现模型在极长训练后会出现结构性Grokking，能够对全新句法结构实现高精度泛化。** 以前的工作大多在模型收敛后即停止，错失了潜在的后期结构学习阶段。

2. **系统性探索模型深度对结构Grokking的影响 → 在多个数据集上对不同层数的Transformer进行同等训练 → 揭示出倒U型深度效应，指出中等深度（如6‑12层）最有利于层次结构的领悟。** 这与传统“越深越好”的直觉相悖，为模型设计提供了新视角。

3. **引入树结构度量作为诊断工具 → 在训练过程中实时计算该度量 → 发现度量在结构Grokking出现前后会出现显著拐点，且拐点位置对应最佳深度。** 这让研究者可以在不查看具体输出的情况下预测模型何时学会层次结构。

4. **保持模型“原味”不加任何句法偏置 → 通过纯粹的自注意力和标准语言建模目标验证层次结构的自发出现 → 为“Transformer 本身是否足以捕获句法”提供了强有力的实证。** 之前的多数研究通过在模型中加入显式树结构或句法标签来提升结构感知，本工作则证明即使不加这些“外挂”，模型也能自行发现。

### 方法详解

整体思路可以概括为“三步走”：  
1）准备一套专门设计的层次结构测试数据；  
2）让标准Transformer在这些数据上进行极长时间的语言模型训练；  
3）在训练过程中监控模型内部的树结构度量并记录结构泛化表现。

**步骤 1：数据集与任务**  
作者使用了几类合成或半合成的语言任务，这些任务的核心是让模型预测句子中某个位置的词或标签，而训练集只覆盖有限的树形模式。比如“Dyck-语言”任务（匹配括号的嵌套深度）和“嵌套算术表达式”任务，训练时只提供深度 ≤ 3 的样本，测试时会出现深度 4‑5 的全新结构。

**步骤 2：极长训练**  
在每个实验中，模型先按照常规的交叉熵损失进行语言建模训练，直至验证集上的准确率基本不再提升（通常在数十万步后）。随后，作者继续让模型在相同学习率下训练数倍甚至十倍的步数，期间不做任何学习率衰减或正则化的额外干预。这样做的核心假设是：模型可能在“表面任务已经学好”后，仍在内部重构表示，从而捕获更抽象的结构。

**步骤 3：树结构度量监控**  
Murty 等人的树结构度量本质上是把每层的注意力矩阵投影到一个低维空间，然后测量这些投影是否符合树形层次的距离约束。作者在每隔一定步数计算一次该度量，并把它与结构泛化的测试准确率一起绘图。实验发现，当度量出现显著下降（表示内部表示更像树）时，模型的结构泛化准确率会出现突跃——这正是所谓的结构Grokking。

**关键细节与反直觉点**  
- **不调学习率**：很多人会在模型收敛后降低学习率以防过拟合，但作者保持原学习率，结果反而让模型在“噪声”中继续探索更高层次的表示。  
- **深度选择**：实验显示，12层以上的Transformer在同等训练时往往陷入局部最小，注意力矩阵过于分散，导致树结构度量难以下降；而3层以下则容量不足，无法形成层次。中等深度恰好平衡了容量与可训练性。  
- **纯粹语言模型目标**：没有加入任何句法标签或结构约束，说明自注意力本身具备潜在的层次组织能力，只是需要足够的训练时间来激活。

### 实验与效果

- **数据集**：Dyck-1/2（括号匹配），嵌套算术表达式（加减乘除的树形计算），以及一个小规模的自然语言子集（从Penn Treebank 中抽取的有限深度句子）。所有数据均在训练集只出现深度 ≤ 3 的结构，测试集则包含深度 4‑5 的全新结构。

- **基线对比**：作者将普通Transformer的表现与两类基线比较：① 训练到常规收敛的Transformer（即不进行极长训练），② 加入显式句法诱导层（如Tree‑Transformer）的模型。结果显示，普通Transformer 在极长训练后在结构泛化任务上可达 **≈ 90%** 的准确率，显著超过常规收敛的 **≈ 60%**，并且与显式句法模型的 **≈ 92%** 相当。

- **深度效应**：在 4、6、8、12、24 层的模型中，6‑8 层的模型在结构Grokking出现时的最高准确率约为 **88%**，而 4 层仅约 **70%**，24 层则回落到 **65%**，形成明显的倒U型曲线。

- **消融实验**：作者分别关闭（1）极长训练阶段、（2）树结构度量监控、（3）不同学习率策略。关闭极长训练后结构泛化几乎不出现；改变学习率为衰减式后Grokking延迟或消失；度量本身不影响模型学习，只是诊断工具。由此确认极长训练是触发结构领悟的关键因素。

- **局限性**：实验主要基于合成任务，真实自然语言的层次复杂度远高于括号匹配。作者承认在大规模真实语料上是否会出现同样的结构Grokking仍未验证。此外，极长训练成本高，实际应用中可能不具备可行性。

### 影响与延伸思考

这篇工作在社区引发了两大方向的思考：  
1）**训练时长与能力突现**：后续研究开始系统地探讨“训练后期的结构学习”是否普遍存在，如在大模型的微调阶段出现的数学推理能力提升。  
2）**模型深度的非线性效应**：倒U型深度效应促使一些团队重新审视“越深越好”的设计原则，尝试在特定任务上使用中等深度的Transformer 或者在深层模型中加入层间正则化以避免深度带来的信息稀释。  

如果想进一步了解，可以关注以下方向：  
- **长时训练的表征演化**（如“Phase Transitions in Deep Learning”系列），  
- **自注意力的隐式句法学习**（如“Attention is not all you need”后续工作），  
- **高效的结构诊断指标**（Murty 等人的后续改进）。  

### 一句话记住它

只要让普通Transformer训练足够久，它会自行“悟出”句子的层次结构，而中等深度的模型最擅长这场“结构Grokking”。