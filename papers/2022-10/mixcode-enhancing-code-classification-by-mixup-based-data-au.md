# MIXCODE: Enhancing Code Classification by Mixup-Based Data Augmentation

> **Date**：2022-10-06
> **arXiv**：https://arxiv.org/abs/2210.03003

## Abstract

Inspired by the great success of Deep Neural Networks (DNNs) in natural language processing (NLP), DNNs have been increasingly applied in source code analysis and attracted significant attention from the software engineering community. Due to its data-driven nature, a DNN model requires massive and high-quality labeled training data to achieve expert-level performance. Collecting such data is often not hard, but the labeling process is notoriously laborious. The task of DNN-based code analysis even worsens the situation because source code labeling also demands sophisticated expertise. Data augmentation has been a popular approach to supplement training data in domains such as computer vision and NLP. However, existing data augmentation approaches in code analysis adopt simple methods, such as data transformation and adversarial example generation, thus bringing limited performance superiority. In this paper, we propose a data augmentation approach MIXCODE that aims to effectively supplement valid training data, inspired by the recent advance named Mixup in computer vision. Specifically, we first utilize multiple code refactoring methods to generate transformed code that holds consistent labels with the original data. Then, we adapt the Mixup technique to mix the original code with the transformed code to augment the training data. We evaluate MIXCODE on two programming languages (Java and Python), two code tasks (problem classification and bug detection), four benchmark datasets (JAVA250, Python800, CodRep1, and Refactory), and seven model architectures (including two pre-trained models CodeBERT and GraphCodeBERT). Experimental results demonstrate that MIXCODE outperforms the baseline data augmentation approach by up to 6.24% in accuracy and 26.06% in robustness.

---

# MIXCODE：基于 Mixup 的代码分类增强 论文详细解读

### 背景：这个问题为什么难？

在代码分析领域，深度神经网络（DNN）已经展示出强大的特征学习能力，但它们对大量、标注精准的训练样本有极高依赖。获取代码标签往往需要专业的程序员或安全专家，成本远高于普通文本标注。现有的代码数据增强手段大多停留在“代码改写”或“对抗样本”层面，这类方法只能产生表面上不同的代码，却难以显著丰富特征空间，导致模型在真实场景下的鲁棒性提升有限。于是，如何在不增加标注成本的前提下，生成既保持语义标签又能多样化的训练样本，成为制约代码分类性能提升的关键瓶颈。

### 关键概念速览

- **代码分类**：把一段源码映射到预定义的类别（如题目类型、是否含 bug），类似于把图片分到不同的标签，只不过输入是程序文本或其结构化表示。  
- **数据增强**：在已有训练集上人为制造新样本，以扩大数据量、提升模型泛化能力，常见于图像的旋转、裁剪等操作。  
- **代码重构（refactoring）**：对代码进行等价的结构或风格改动（如变量重命名、提取函数），不改变功能但能产生不同的代码文本。  
- **Mixup**：一种在图像领域流行的增强技巧，将两张图片的像素和对应标签按比例线性混合，得到“混合图”，帮助模型学习更平滑的决策边界。  
- **标签混合**：在 Mixup 中，原始标签也按同样比例混合，得到一个软标签（如 0.7 × 类A + 0.3 × 类B），模型需要预测这种概率分布。  
- **软标签（soft label）**：相较于硬标签的“全对或全错”，软标签提供了不确定性信息，让模型在训练时更宽容、学习更稳健。  
- **预训练模型（如 CodeBERT）**：先在大规模代码语料上进行自监督学习，再在特定任务上微调的模型，类似于先学会“通用语言”，再专注于“专业任务”。  

### 核心创新点

1. **从单一重构到混合重构**  
   - 之前的增强方法往往只把原始代码直接改写一次，得到的 B 池代码与原始 A 池代码之间差异有限。  
   - MIXCODE 先使用多种重构手段（变量改名、代码块抽取、顺序调换等）一次性生成多个等价变体，形成更丰富的 B 池。  
   - 这种多样化的等价代码让后续的混合操作拥有更大的“混合空间”，提升了生成样本的多样性。

2. **将 Mixup 思想直接搬到代码文本**  
   - Mixup 在视觉任务中是对像素进行线性插值，代码文本没有像素概念，直接插值会破坏语法。  
   - 作者创新性地在 **特征层面**（即模型的隐藏向量）进行线性混合：先把原始代码和重构代码分别送入编码器得到向量，再按随机比例混合向量，同时混合对应的软标签。  
   - 这样既保留了代码的语法合法性，又实现了 Mixup 的平滑决策边界效果。

3. **统一的任务适配框架**  
   - 过去的增强往往针对特定任务（如 bug 检测），难以复用。  
   - MIXCODE 将重构+Mixup 的流程抽象为“数据层面”操作，几乎不需要改动下游模型结构，直接在训练数据加载阶段完成。  
   - 这种即插即用的设计让它能够在问题分类、缺陷检测等多种任务上统一使用。

4. **兼容多种模型架构**  
   - 通过在特征层面混合，MIXCODE 能够兼容基于 Transformer 的预训练模型（CodeBERT、GraphCodeBERT）以及传统的基于图的模型。  
   - 与仅针对特定模型的增强手段相比，提升了方法的通用性和推广价值。

### 方法详解

**整体框架**  
MIXCODE 的训练管线可以划分为四步：① 原始代码收集（A 池）；② 多样化代码重构生成等价代码（B 池）；③ 编码器将 A、B 两端代码映射到向量空间；④ 在向量空间进行 Mixup，生成混合向量及软标签，送入下游分类头进行训练。

**步骤拆解**

1. **代码重构模块**  
   - 采用一组预定义的重构策略，如变量/函数重命名、代码块抽取成独立函数、语句顺序调换（在不改变控制流的前提下）等。  
   - 每条原始样本会随机挑选 1~3 种策略组合，生成若干等价代码片段。  
   - 这些重构代码在语义上与原始代码保持一致，因此标签不变。

2. **特征编码**  
   - 任选一个已有的代码编码器（如 CodeBERT），对原始代码和每个重构代码分别进行前向传播，得到对应的隐藏向量（通常取 `[CLS]` token 表示）。  
   - 这里的向量是模型对代码语义的抽象表示，已经在大规模代码语料上学习到通用的语法/语义特征。

3. **向量层 Mixup**  
   - 随机抽取一对向量 `(v_A, v_B)`，并从均匀分布 `λ ~ U(0,1)` 采样混合系数。  
   - 计算混合向量 `v_mix = λ * v_A + (1-λ) * v_B`。  
   - 同时对标签进行线性混合：`y_mix = λ * y_A + (1-λ) * y_B`（因为标签相同，软标签仍是原标签，只是加入了少量噪声，提升鲁棒性）。  
   - 生成的 `v_mix` 与 `y_mix` 直接送入分类层，完成一次前向训练。

4. **训练与优化**  
   - 采用交叉熵损失对软标签进行监督，梯度会同时更新编码器和分类头。  
   - 为防止混合比例极端导致训练不稳定，作者在实现中对 `λ` 做了裁剪（如限制在 `[0.2, 0.8]`），确保每个样本都有足够的信息量。

**巧妙之处**  
- **在特征层混合**：直接在代码文本上做线性插值会产生非法代码，作者把混合搬到模型内部的向量空间，既保持了语法合法性，又实现了 Mixup 的核心思想。  
- **软标签的噪声作用**：虽然原始标签相同，混合后仍产生微小的概率分布偏移，这种“标签平滑”帮助模型在面对噪声或未见代码时更不易过拟合。  
- **多策略重构**：通过组合多种等价变换，B 池的分布更接近真实代码的多样性，避免了单一重构导致的特征偏移。

### 实验与效果

- **数据集与任务**：在四个公开基准上评估：Java250（Java 题目分类）、Python800（Python 题目分类）、CodRep1（代码补全任务的缺陷定位子任务）以及 Refactory（Java bug 检测）。  
- **模型对比**：分别在七种模型上跑实验，包括两种预训练 Transformer（CodeBERT、GraphCodeBERT）以及传统基于图的模型。baseline 为不使用任何增强以及使用传统代码转换增强。  
- **性能提升**：论文报告在最好的组合上，准确率提升最高 **6.24%**，鲁棒性提升最高 **26.06%**（在对抗噪声或代码混淆测试集上的表现）。  
- **消融实验**：作者分别去掉（1）多策略重构，仅保留单一重构；（2）向量层 Mixup，仅使用原始+重构的硬标签；结果显示，去掉任一环节都会导致准确率下降约 2%~4%，验证了两者的协同增益。  
- **局限性**：实验主要集中在两种语言（Java、Python）和相对中等规模的数据集，未在大规模工业代码库上验证；此外，Mixup 依赖于编码器的特征质量，若编码器本身表现不佳，混合效果会受限。论文也提到对极端长代码（超过模型最大长度）仍需截断或分块处理。

### 影响与延伸思考

MIXCODE 把视觉领域的 Mixup 成功迁移到代码分析，开启了“特征层混合”在软件工程任务中的新思路。随后的工作（如 CodeMix、GraphMix 等）开始探索在抽象语法树（AST）或图神经网络的节点表示上进行线性混合，进一步提升对结构化代码的鲁棒性。还有研究尝试将 Mixup 与对抗训练结合，生成更具挑战性的混合样本，以提升安全相关任务的防御能力。对想深入的读者，可以关注以下方向：① 更丰富的等价变换（如自动化重构工具生成的多样化补丁）；② 在多模态模型（代码+文档）上做跨模态 Mixup；③ 将 Mixup 与自监督预训练目标结合，提升预训练阶段的特征平滑性。  

### 一句话记住它

**MIXCODE 用多种等价重构生成的代码向量做 Mixup，让模型在特征空间里“混合代码”，从而显著提升代码分类的准确率和鲁棒性。**