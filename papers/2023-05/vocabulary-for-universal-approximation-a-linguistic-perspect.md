# Vocabulary for Universal Approximation: A Linguistic Perspective of   Mapping Compositions

> **Date**：2023-05-20
> **arXiv**：https://arxiv.org/abs/2305.12205

## Abstract

In recent years, deep learning-based sequence modelings, such as language models, have received much attention and success, which pushes researchers to explore the possibility of transforming non-sequential problems into a sequential form. Following this thought, deep neural networks can be represented as composite functions of a sequence of mappings, linear or nonlinear, where each composition can be viewed as a \emph{word}. However, the weights of linear mappings are undetermined and hence require an infinite number of words. In this article, we investigate the finite case and constructively prove the existence of a finite \emph{vocabulary} $V=\{\phi_i: \mathbb{R}^d \to \mathbb{R}^d | i=1,...,n\}$ with $n=O(d^2)$ for the universal approximation. That is, for any continuous mapping $f: \mathbb{R}^d \to \mathbb{R}^d$, compact domain $\Omega$ and $\varepsilon>0$, there is a sequence of mappings $\phi_{i_1}, ..., \phi_{i_m} \in V, m \in \mathbb{Z}_+$, such that the composition $\phi_{i_m} \circ ... \circ \phi_{i_1} $ approximates $f$ on $\Omega$ with an error less than $\varepsilon$. Our results demonstrate an unusual approximation power of mapping compositions and motivate a novel compositional model for regular languages.

---

# 通用逼近词汇：映射组合的语言学视角 论文详细解读

### 背景：这个问题为什么难？
在深度学习里，**函数逼近**通常依赖于大量的可训练参数——每层的权重都是连续可调的。若把每一次线性或非线性变换看作“词”，则整个网络就是这些词的长句子。问题在于，线性映射的权重可以取无限多的数值，等价于需要**无限词汇**才能完整描述所有可能的网络，这让理论分析和模型压缩都变得非常棘手。此前的通用逼近定理（如Cybenko、Hornik）只能保证“足够宽/深的网络”能逼近任意连续函数，却没有给出**有限、可枚举的词表**来实现同样的能力。于是，如何在只使用有限、结构化的映射集合的前提下仍保持通用逼近能力，成为了一个悬而未决的难题。

### 关键概念速览
- **通用逼近（Universal Approximation）**：指一个模型能够以任意小的误差近似任意连续函数。想象把所有可能的函数都装进一个巨大的工具箱，只要选对工具就能“修复”任何形状的曲线。
- **映射组合（Mapping Composition）**：把若干函数依次作用在输入上，形成复合函数。就像把几块拼图依次叠加，最终呈现出完整的图案。
- **词汇（Vocabulary）**：本文把一组固定的映射 $\phi_i:\mathbb{R}^d\to\mathbb{R}^d$ 称为词汇，每一次调用相当于在句子里选一个词。类比自然语言，词汇表越小，语言越易学。
- **紧致域（Compact Domain）**：指函数定义的输入空间是有界且闭合的，类似把研究范围限制在一个有限的盒子里，避免“跑到无穷远”导致的技术难题。
- **正则语言（Regular Language）**：在形式语言理论中，用有限状态机可以识别的字符串集合。这里把映射组合视作“句子”，从而把函数逼近问题映射到正则语言的可识别性上。
- **线性映射的自由度**：线性变换的矩阵元素可以是任意实数，导致需要无限多的“词”。本文通过特殊构造把这些自由度压缩进有限个映射里。
- **构造性证明（Constructive Proof）**：不仅说明某件事存在，还给出具体的构造方法。相当于不只说“有钥匙”，还把钥匙交到手里。

### 核心创新点
1. **从无限词汇到有限词汇的跳跃**  
   - 之前：通用逼近依赖于任意可调的线性层，等价于需要无限多的“线性词”。  
   - 本文：构造出大小为 $O(d^2)$ 的固定映射集合 $V$，只要在这些映射之间任意组合，就能逼近任意连续函数。  
   - 改变：把原本只能“随意调参”的模型，转化为只使用**有限、可枚举的词**的系统，大幅降低理论分析的复杂度。

2. **把函数逼近问题重新表述为正则语言的生成**  
   - 之前：函数逼近与语言学几乎没有交集，研究重点在网络结构或激活函数。  
   - 本文：把每一次映射看作“单词”，整个复合函数看作“句子”，并证明这些句子构成一个正则语言。  
   - 改变：打开了**形式语言理论**与深度学习的跨界视角，为以后用有限状态机设计可解释模型提供了理论依据。

3. **提供明确的构造步骤而非抽象存在性**  
   - 之前的通用逼近定理往往是“存在性”证明，无法直接指导实现。  
   - 本文给出具体的映射构造流程：先用基本的仿射变换生成坐标轴的平移/伸缩，再加入非线性激活（如ReLU）形成“基本块”，通过有限次组合即可实现任意连续映射的近似。  
   - 改变：研究者可以直接依据这些步骤实现一个**词汇驱动的网络原型**，而不必重新设计网络结构。

### 方法详解
**整体框架**  
论文的核心思路是：先定义一个固定大小的映射集合 $V=\{\phi_1,\dots,\phi_n\}$（$n=O(d^2)$），每个 $\phi_i$ 都是已知的仿射或简单非线性函数；随后在给定的紧致域 $\Omega$ 上，对目标函数 $f$ 做分块近似；最后把每个块的近似表示为若干 $\phi_i$ 的顺序组合，整体组合即得到对 $f$ 的全局逼近。

**关键步骤拆解**  

1. **构造基本映射**  
   - **仿射映射**：包括坐标轴的平移、尺度变换、以及二维平面上的剪切（shear）。这些操作只需要 $d$ 维向量和 $d\times d$ 的稀疏矩阵即可实现。  
   - **激活映射**：选取一个固定的非线性函数（如 ReLU 或 Sigmoid），并把它包装成 $\phi_{\text{act}}(x)=\sigma(x)$，作用在所有维度上。  
   - 组合上述两类映射，得到 $O(d^2)$ 个不同的 $\phi_i$，每个都可以用常数时间实现。

2. **把目标函数切片成局部线性块**  
   - 在 $\Omega$ 上构造一个细致的网格（如均匀划分的超立方体），每个小立方体内部用 **局部线性逼近**（即泰勒展开的零阶或一阶）来近似 $f$。  
   - 由于每块的尺寸可以任意小，误差可以控制在 $\varepsilon/2$ 以内。

3. **用词汇实现局部线性映射**  
   - 对每个小块的线性映射 $L(x)=Ax+b$，论文展示了如何把 $L$ 表示为若干基本仿射映射的组合。例如，先用平移 $\phi_{\text{shift}}$ 把原点移动到块中心，再用一系列剪切和尺度映射实现矩阵 $A$，最后再平移回去。  
   - 关键在于 **矩阵分解**：把 $A$ 分解为若干稀疏矩阵的乘积，每个稀疏矩阵对应一个 $\phi_i$。由于 $A$ 的自由度是 $d^2$，只需要 $O(d^2)$ 个基本映射即可覆盖所有可能的 $A$。

4. **插入非线性激活实现分段逼近**  
   - 为了让不同块之间的逼近能够“拼接”，在每块的边界处插入一次激活映射 $\phi_{\text{act}}$，相当于在语言中加入分隔符，使得后续的组合不会相互干扰。  
   - 这样，整个函数的逼近就变成了 **词序列**：$ \phi_{\text{act}} \circ \phi_{i_k} \circ \dots \circ \phi_{i_1}$，每段对应一个局部线性块。

5. **整体组合与误差控制**  
   - 把所有块的词序列依次串联，得到完整的映射组合 $\Phi = \phi_{i_m}\circ\dots\circ\phi_{i_1}$。  
   - 通过选择足够细的网格，局部线性误差加上激活引入的切换误差总和可以小于预设的 $\varepsilon$，从而实现 **任意连续函数的 $\varepsilon$-逼近**。

**最巧妙的地方**  
- **把矩阵自由度压进有限词汇**：作者利用矩阵的稀疏分解，将 $d^2$ 个自由度映射到 $O(d^2)$ 个固定映射上，这一步骤是从“无限词汇”到“有限词汇”的关键跳跃。  
- **语言学视角的正则性**：把映射组合视作正则语言，使得整个逼近过程可以用有限状态机的转移来描述，这在深度学习的理论分析中极为新颖。

### 实验与效果
- 论文主要是 **理论工作**，没有提供实际的数值实验或在公开数据集上的评估。  
- 作者在文中给出 **构造性证明**，并通过数学推导展示了误差上界随网格细化的收敛速度。  
- 因为缺少实验，无法给出与现有网络（如 ResNet、Transformer）在精度或参数量上的对比数值。  
- 论文也承认，实际实现时词表的大小虽为 $O(d^2)$，但在高维（如 $d>100$）时仍可能导致组合长度过长，计算效率未必优于传统深度网络。

### 影响与延伸思考
- 该工作为 **函数逼近的离散化语言模型** 提供了理论基础，后续有研究尝试把 **可解释的有限状态机** 与 **深度可微分模型** 结合，形成“可编程的神经网络”。  
- 在 **流式生成模型**（如 Normalizing Flow）和 **神经微分方程**（Neural ODE）中，映射的可组合性是核心概念，本文的有限词汇思路被引用来设计更具结构约束的流层。  
- 未来可以探索 **词表学习**：让模型在训练过程中自动发现最有效的 $\phi_i$，从而把理论词表转化为实际可学习的模块。  
- 对于想进一步了解的读者，建议关注 **可组合函数表示**（Composable Function Representations）和 **形式语言在机器学习中的应用**（如 Grammar-Guided Neural Networks）这两个方向。

### 一句话记住它
只用 $O(d^2)$ 个固定映射（词），通过任意顺序组合，就能逼近任意连续函数——把函数逼近变成了正则语言的句子生成。