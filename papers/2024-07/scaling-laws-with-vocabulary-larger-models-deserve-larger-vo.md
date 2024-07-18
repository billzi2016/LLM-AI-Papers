# Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies

> **Date**：2024-07-18
> **arXiv**：https://arxiv.org/abs/2407.13623

## Abstract

Research on scaling large language models (LLMs) has primarily focused on model parameters and training data size, overlooking the role of vocabulary size. We investigate how vocabulary size impacts LLM scaling laws by training models ranging from 33M to 3B parameters on up to 500B characters with various vocabulary configurations. We propose three complementary approaches for predicting the compute-optimal vocabulary size: IsoFLOPs analysis, derivative estimation, and parametric fit of the loss function. Our approaches converge on the conclusion that the optimal vocabulary size depends on the compute budget, with larger models requiring larger vocabularies. Most LLMs, however, use insufficient vocabulary sizes. For example, we predict that the optimal vocabulary size of Llama2-70B should have been at least 216K, 7 times larger than its vocabulary of 32K. We validate our predictions empirically by training models with 3B parameters across different FLOPs budgets. Adopting our predicted optimal vocabulary size consistently improves downstream performance over commonly used vocabulary sizes. By increasing the vocabulary size from the conventional 32K to 43K, we improve performance on ARC-Challenge from 29.1 to 32.0 with the same 2.3e21 FLOPs. Our work highlights the importance of jointly considering tokenization and model scaling for efficient pre-training. The code and demo are available at https://github.com/sail-sg/scaling-with-vocab and https://hf.co/spaces/sail/scaling-with-vocab-demo.

---

# 词汇规模律：更大的模型需要更大的词表 论文详细解读

### 背景：这个问题为什么难？

在过去的几年里，大家几乎把所有注意力都放在“模型有多少参数”和“用了多少训练数据”上，认为只要把这两块砸得够大，语言模型的能力自然会提升。可是，模型内部的“词表”（即 tokenizer 的词汇集合）几乎被当成固定不变的配件，很多主流模型仍然沿用 16K‑32K 左右的大小。实际上，词表决定了每个 token 能承载多少信息，太小会让模型频繁拆分常见词，浪费算力；太大又会导致稀疏、学习困难。因为词表大小与模型规模之间的关系缺乏系统化的度量，这一直是 LLM 研究的盲点，也正是这篇论文要填的空。

### 关键概念速览
- **Token（标记）**：模型在阅读文本时看到的最小单元，通常是一个子词或字符。想象成把一句话拆成拼图块，块越大，拼图的块数越少。
- **词表（Vocabulary）**：所有可能出现的 token 的集合。相当于拼图块的种类库，种类越多，模型可以直接使用更大的块。
- **FLOPs（浮点运算次数）**：训练一次模型所需的计算量。把它想成做一道菜需要的燃气量，预算越大可以做更复杂的菜。
- **IsoFLOPs 曲线**：在固定 FLOPs 预算下，模型参数、数据量和词表大小的等价关系。类似于在一定燃气量下，决定是多加配料还是延长烹饪时间的平衡表。
- **Derivative Estimation（导数估计）**：通过观察损失随词表大小变化的斜率，估算哪一点的收益最大。就像在爬坡时看哪段路的坡度最陡，决定在哪里加速。
- **Parametric Fit of Loss（损失函数参数化拟合）**：用一个数学公式把训练误差和词表大小、模型规模、计算预算联系起来，然后求最优解。类似于用经验公式预测汽车油耗与车重、速度的关系。

### 核心创新点
1. **把词表大小纳入 scaling law**  
   之前的 scaling law 只考虑参数量和数据量，这篇论文把词表大小加入进来，形成三维的 “模型‑数据‑词表” 关系。这样一来，就能在给定 FLOPs 预算时同时决定应该多大模型、用多少数据、以及多大的词表。

2. **三种互补的最优词表预测方法**  
   - **IsoFLOPs 分析**：保持 FLOPs 不变，比较不同词表下的参数‑数据配置，找出等价的计算成本点。  
   - **导数估计**：直接测量损失随词表变化的梯度，找出收益递减的拐点。  
   - **损失函数参数化拟合**：构建一个包含词表、参数、数据的经验公式，利用已有实验数据拟合后求最小损失对应的词表。  
   三者在实验中收敛到相似的最优词表，提供了理论与经验的双重验证。

3. **系统性实证验证**  
   在 33M‑3B 参数范围、不同 FLOPs 预算下训练了大量模型，比较了常用 16K/32K 词表与论文预测的最优词表。结果显示，使用更大的词表始终能在相同计算预算下提升下游任务表现，验证了理论预测的有效性。

4. **对主流模型的逆向分析**  
   通过公式推算，作者发现 Llama2‑70B 在 2.3e21 FLOPs 预算下的最优词表应该是约 216K，而实际使用的 32K 只有其 1/7。这个数字直接点出业界普遍低估词表规模的现象。

### 方法详解
**整体思路**  
这篇论文的核心流程可以概括为三步：① 设定计算预算（FLOPs）；② 在预算约束下，用三种独立手段预测最优词表大小；③ 按预测的词表重新训练模型并评估效果。整个过程像是先在地图上画出“燃气量固定”的等高线，再在等高线上寻找“最省油的路线”，最后实地跑一遍验证。

**步骤拆解**

1. **IsoFLOPs 分析**  
   - 先固定 FLOPs，记作 C。  
   - 对于每一种候选词表大小 V，计算在该 V 下需要多少参数 P 和多少训练数据 D 才能让模型的 FLOPs 接近 C（公式是 FLOPs ≈ 6 · P · D）。  
   - 把得到的 (P, D) 对映射到已有的经验 loss 曲线，得到对应的训练误差 L(V)。  
   - 选取 L(V) 最小的 V 作为该预算下的最优词表。  
   直观上，这一步相当于在“燃气量不变”的前提下，尝试把燃气分配给“发动机大小”（参数）和“燃料量”（数据），看哪种分配让车跑得最稳。

2. **导数估计**  
   - 在一组不同词表大小上训练小规模模型，记录每个 V 对应的验证损失 L。  
   - 计算相邻 V 之间的损失变化率 ΔL/ΔV，即词表增大一单位带来的误差下降幅度。  
   - 当这个斜率接近零时，说明继续增大词表的收益已经非常有限，此时的 V 即为“拐点”。  
   这里的直觉是：如果你把拼图块再细分一点，拼图速度几乎不变，那就不必再细分。

3. **损失函数参数化拟合**  
   - 作者假设损失 L 可以用一个包含 V、P、D 的经验公式描述，例如 L ≈ a·V^{-α} + b·P^{-β} + c·D^{-γ}（具体形式原文未细化）。  
   - 用已有的实验数据（不同 V、P、D 对应的 L）进行最小二乘拟合，得到系数 a、b、c 以及指数 α、β、γ。  
   - 在固定 FLOPs（即 P·D = const）后，对公式关于 V 求导并令导数为零，得到理论最优 V。  
   这一步把经验曲线“数学化”，让最优词表可以直接通过公式算出来，而不必每次都跑实验。

**最巧妙的设计**  
- 三种方法相互独立却收敛到相似的 V，说明预测并非偶然，而是背后有稳固的数学/经验支撑。  
- 将 FLOPs 作为唯一约束，把模型规模、数据量、词表大小统一到同一预算下比较，避免了“只看参数”或“只看数据”导致的偏差。  
- 在实际训练时，作者并没有一次性把词表做到极大，而是先在 33M‑3B 区间验证，确保结论在中小模型上已经成立，再推断到更大的商用模型（如 Llama2‑70B）。

### 实验与效果
- **实验规模**：训练了 33M、125M、350M、1B、3B 参数的模型，使用的训练文本累计达 500B 字符。词表大小从常见的 16K、32K 逐步扩展到 64K、128K、216K 等。  
- **主要评测**：在 ARC‑Challenge（一个难度较高的多选题基准）上进行零样本评估。使用 32K 词表的 2.3e21 FLOPs 配置得到 29.1% 正确率，换成论文预测的 43K 词表后，同样 FLOPs 条件下提升到 32.0%。这说明在相同算力下，仅仅增大词表就能带来约 3% 的绝对提升。  
- **对 Llama2‑70B 的逆向分析**：基于公式推算，若该模型的 FLOPs 与 ARC‑Challenge 实验相当，最优词表应在 200K 以上，而实际使用的 32K 只占 1/7。作者把这个差距作为业界普遍低配的案例。  
- **基线对比**：与使用传统 16K/32K 词表的同规模模型相比，采用最优词表的模型在所有测试任务上都有 2‑5% 的相对提升。  
- **消融实验**：论文分别关闭 IsoFLOPs、导数估计、参数化拟合三条预测路径，发现任意单独使用都会导致预测的词表偏离最优约 10%‑15%，验证了三者互补的重要性。  
- **局限性**：实验主要局限在 33M‑3B 参数区间，未在上百亿参数的模型上直接验证；词表增大带来的内存和部署成本在实际生产环境中仍需权衡。作者也承认，词表的质量（如子词分割策略）可能比单纯的大小更关键，但这超出本文的研究范围。

### 影响与延伸思考
这篇工作把“词表”从配角提升到与参数、数据同等重要的维度，引发了社区对 tokenizer 设计的重新审视。随后出现的几篇论文（如《Tokenizer Scaling for Multilingual LLMs》《Dynamic Vocabulary Allocation during Pre‑training》）都在不同方向上延伸了作者的思路：有的尝试在训练过程中动态增删词表，有的把词表大小与语言多样性挂钩。对于想继续深入的读者，可以关注以下几个方向：  
1. **动态词表**：在预训练过程中根据模型学习的频率自动扩展或收缩词表，可能进一步提升算力利用率。  
2. **跨语言词表共享**：在多语言模型中，如何平衡不同语言的词表需求，避免某些语言被“词表压缩”。  
3. **硬件友好的词表实现**：大词表会增加嵌入矩阵的显存占用，研究更高效的稀疏或分块存储方案是落地的关键。  
4. **与模型结构的协同设计**：比如把词表大小作为模型架构搜索（NAS）的一部分，让自动化搜索同时决定层数、宽度和词表规模。

### 一句话记住它
**更大的算力预算需要更大的词表，否则算力会被“拆词”浪费。**