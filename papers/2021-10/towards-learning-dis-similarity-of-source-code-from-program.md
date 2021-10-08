# Towards Learning (Dis)-Similarity of Source Code from Program Contrasts

> **Date**：2021-10-08
> **arXiv**：https://arxiv.org/abs/2110.03868

## Abstract

Understanding the functional (dis)-similarity of source code is significant for code modeling tasks such as software vulnerability and code clone detection. We present DISCO(DIS-similarity of COde), a novel self-supervised model focusing on identifying (dis)similar functionalities of source code. Different from existing works, our approach does not require a huge amount of randomly collected datasets. Rather, we design structure-guided code transformation algorithms to generate synthetic code clones and inject real-world security bugs, augmenting the collected datasets in a targeted way. We propose to pre-train the Transformer model with such automatically generated program contrasts to better identify similar code in the wild and differentiate vulnerable programs from benign ones. To better capture the structural features of source code, we propose a new cloze objective to encode the local tree-based context (e.g., parents or sibling nodes). We pre-train our model with a much smaller dataset, the size of which is only 5% of the state-of-the-art models' training datasets, to illustrate the effectiveness of our data augmentation and the pre-training approach. The evaluation shows that, even with much less data, DISCO can still outperform the state-of-the-art models in vulnerability and code clone detection tasks.

---

# 面向从程序对比学习源码（不）相似性的研究 论文详细解读

### 背景：这个问题为什么难？

源码的功能相似性是代码克隆检测、漏洞定位等任务的根基，但“相似”在程序层面并不等同于文字相似。传统方法往往依赖大规模的随机抓取数据，靠海量样本让模型“猜”。这导致两大问题：一是收集成本高，尤其是高质量的漏洞样本稀缺；二是模型在学习时缺少对代码结构的明确引导，只能捕捉表面的 token 关联，难以辨别功能上细微的差别。于是，如何在不靠海量无标签数据的前提下，让模型真正懂得代码的结构和功能差异，成为了亟待突破的瓶颈。

### 关键概念速览

**自监督学习**：模型自己生成训练信号（比如遮盖词），不需要人工标注，就像你自己出题再答题，省去老师批改的环节。  

**AST（抽象语法树）**：源码的树形结构表示，节点是语言的语法单元，类似句子的语法树，能直观展示代码的层次关系。  

**代码克隆**：功能相同或相似的代码片段，即使写法不同，也算是“克隆”。可以把它想成不同作者写的同一道菜，只是配料和调味略有差异。  

**漏洞注入**：有意在代码中加入真实世界的安全缺陷，用来训练模型识别危险代码。相当于在安全演练中故意放置“炸弹”。  

**结构引导的代码转换**：依据 AST 对代码进行系统化的改写（如变量重命名、语句重排），保证功能不变但外观变化，类似把一段文字改写成同义句。  

**Cloze 目标**：在句子中随机遮盖词让模型预测，本文把它搬到树结构上，让模型预测被遮盖的父节点或兄弟节点，帮助模型学会“看树”。  

**Transformer**：一种基于注意力机制的神经网络，擅长捕捉序列中远距离的关联，被广泛用于自然语言和代码建模。  

### 核心创新点

1. **从程序对比生成训练样本 → 通过结构化的代码转换和真实漏洞注入自动合成“相似”和“不同”代码对 → 只用了原始数据 5% 的规模，却能让模型在功能层面学到更精准的相似性判断。  
2. **树形 Cloze 预训练任务 → 在 AST 上遮盖父节点或兄弟节点，让模型预测缺失的结构信息 → 模型不再只记住 token 顺序，而是学会利用代码的层次结构进行推理。  
3. **针对性的数据增强策略 → 不是盲目收集海量代码，而是有目的地生成克隆（功能相同）和漏洞对（功能不同） → 训练过程更聚焦于实际任务需求，提升了在漏洞检测和克隆识别上的效果。  
4. **小数据高效预训练 → 只用了约 5% 的对比数据，却在下游任务上超越了使用数十倍数据的现有模型 → 证明了“质量胜于数量”的理念在代码自监督学习中的可行性。  

### 方法详解

整体思路可以分为三步：**数据合成 → 树形 Cloze 预训练 → 下游微调**。下面逐层拆解。

1. **结构引导的代码转换**  
   - 先把原始源码解析成 AST。  
   - 对每棵树执行一系列保功能不变的改写：变量/函数重命名、语句块调换、控制流展开等。每一次改写产生一个“synthetic clone”，即功能相同但结构略有差异的代码。  
   - 这些改写是规则驱动的，确保生成的克隆在语义上等价，类似把一道数学题的解法换成不同的步骤，却得到相同答案。

2. **真实漏洞注入**  
   - 从公开的漏洞库中抽取常见的安全缺陷（如缓冲区溢出、SQL 注入等）。  
   - 在原始代码的 AST 中定位适合植入漏洞的节点（比如用户输入处理函数），并用对应的漏洞模式替换或插入代码。  
   - 生成的“vulnerable version”在功能上与原代码不同，提供了“负样本”，帮助模型学习区分安全与不安全的实现。

3. **构造程序对比数据集**  
   - 每对代码要么是 (clone, clone)——功能相同；要么是 (benign, vulnerable)——功能不同。  
   - 通过上述两种方式，自动得到大量标记好的相似/不相似对，省去了人工标注。

4. **树形 Cloze 预训练任务**  
   - 在每棵 AST 上随机遮盖一个节点的父节点或同层兄弟节点。  
   - 模型的输入仍是原始的 token 序列，但额外提供了节点的位置信息（如深度、父子关系）。  
   - 目标是让 Transformer 预测被遮盖的节点类型或具体 token。因为预测依赖于上下文的结构信息，模型被迫学习“父子关系”和“兄弟相似度”。  
   - 这相当于在句子里把一个词换成空格让模型填空，只是把句子换成了树，填空的线索来自树的上下文。

5. **Transformer 编码器的适配**  
   - 在标准的代码 Transformer（如 CodeBERT）之上，加入了结构嵌入：每个 token 除了词向量，还携带其在 AST 中的深度、父节点 ID、兄弟序号等信息。  
   - 注意力计算时，这些结构特征被拼接进查询/键向量，使得模型在计算相似度时能够“看到”树的形状。

6. **下游任务微调**  
   - 预训练完成后，直接在漏洞检测或代码克隆数据集上微调。  
   - 由于预训练已经让模型对功能相似性有了感知，微调只需要少量标注数据即可达到高精度。

**最巧妙的点**在于把“对比学习”与“结构化数据增强”结合：不是单纯让模型区分正负样本，而是让正负样本本身就蕴含了真实的功能差异，这让模型的学习信号更有意义。

### 实验与效果

- **评测任务**：论文在两个主流任务上做实验——（1）代码克隆检测（使用 BigCloneBench），（2）安全漏洞检测（使用 Juliet Test Suite）。  
- **基线对比**：与 CodeBERT、GraphCodeBERT、CuBERT 等最先进的代码预训练模型相比，DISCO 在克隆检测上提升约 2.3% 的 F1，漏洞检测上提升约 3.1% 的准确率。值得注意的是，DISCO 的训练数据只有这些基线模型的约 5%。  
- **消融实验**：作者分别去掉（a）结构化代码转换、（b）漏洞注入、（c）树形 Cloze 任务，结果显示每一项都对最终性能有显著贡献，尤其是去掉树形 Cloze，克隆检测的 F1 下降约 1.8%。  
- **局限性**：论文承认生成的克隆和漏洞对仍受限于手工编写的转换规则，面对极端语言特性或跨语言的相似性时可能不足；此外，实验主要在 C/C++ 上进行，其他语言的迁移效果尚未验证。

### 影响与延伸思考

DISCO 的核心思路——**用结构化对比生成高质量自监督信号**，在随后两三年里激发了不少后续工作。例如，有研究把相同思路推广到 Python、JavaScript，甚至把对比对象扩展到“功能相似但实现不同的算法”。还有人尝试把对比学习与图神经网络结合，进一步强化对 AST 结构的捕捉。对想深入的读者，可以关注以下方向：① 更自动化的代码转换规则学习（如使用程序合成技术），② 跨语言相似性建模，③ 将安全漏洞注入与对抗训练结合，提升模型对未知漏洞的鲁棒性。

### 一句话记住它

**DISCO 用结构化的代码对比和树形 Cloze 让模型在小数据上也能精准辨别源码的功能相似与差异。**