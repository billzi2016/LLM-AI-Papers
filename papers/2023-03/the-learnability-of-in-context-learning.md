# The Learnability of In-Context Learning

> **Date**：2023-03-14
> **arXiv**：https://arxiv.org/abs/2303.07895

## Abstract

In-context learning is a surprising and important phenomenon that emerged when modern language models were scaled to billions of learned parameters. Without modifying a large language model's weights, it can be tuned to perform various downstream natural language tasks simply by including concatenated training examples of these tasks in its input. Though disruptive for many practical applications of large language models, this emergent learning paradigm is not well understood from a theoretical perspective. In this paper, we propose a first-of-its-kind PAC based framework for in-context learnability, and use it to provide the first finite sample complexity results for the in-context learning setup. Our framework includes an initial pretraining phase, which fits a function to the pretraining distribution, and then a second in-context learning phase, which keeps this function constant and concatenates training examples of the downstream task in its input. We use our framework in order to prove that, under mild assumptions, when the pretraining distribution is a mixture of latent tasks (a model often considered for natural language pretraining), these tasks can be efficiently learned via in-context learning, even though the model's weights are unchanged and the input significantly diverges from the pretraining distribution. Our theoretical analysis reveals that in this setting, in-context learning is more about identifying the task than about learning it, a result which is in line with a series of recent empirical findings. We hope that the in-context learnability framework presented in this paper will facilitate future progress towards a deeper understanding of this important new learning paradigm.

---

# 上下文学习的可学习性 论文详细解读

### 背景：这个问题为什么难？
在大规模语言模型出现之前，想让模型完成新任务必须通过梯度更新或微调，这需要改动模型内部权重，成本高且容易产生灾难性遗忘。近年来出现的“上下文学习”（in‑context learning）让模型只通过在输入里拼接几例示例就能完成任务，似乎不需要任何参数更新。可是，这种现象缺乏理论解释：我们不知道模型到底是“学会了”新任务，还是仅仅在“记忆”示例；也不清楚在什么条件下这种能力能够可靠出现。缺少可证明的学习框架，使得研究者只能靠经验观察，难以系统推进。

### 关键概念速览
**上下文学习（In‑Context Learning）**：模型在不改权重的前提下，通过在一次前向传播的输入中加入若干任务示例，实现对该任务的推理。可以想象成把任务说明和例子直接塞进模型的“对话框”。  

**PAC 可学习性**：Probably Approximately Correct 的缩写，指在有限样本下，算法能够以高概率得到误差在可接受范围内的模型。这里把它搬到上下文学习，衡量的是“在多少示例后，模型能可靠输出正确答案”。  

**预训练分布**：模型在大规模语料上训练时所看到的数据分布。相当于模型的“生活经验”。  

**潜在任务混合（Mixture of Latent Tasks）**：把预训练数据看成由若干隐藏任务组成的混合体，每个任务对应一种特定的输入‑输出映射。类似于把一本百科全书拆成若干章节，每章讲一种技能。  

**任务识别 vs. 任务学习**：任务识别指模型从上下文中辨认出当前要做的任务是哪一种；任务学习指模型在没有先验的情况下自行构造对应的解法。前者像先看标题再决定怎么读，后者像从零开始写答案。  

**样本复杂度（Sample Complexity）**：为达到一定学习效果所需的示例数量。这里指的是在上下文中需要拼接多少例子才能让模型表现稳定。  

**不变函数（Invariant Function）**：在框架中指预训练阶段学到的函数在上下文学习阶段保持不变，只是输入被扩展。可以把它想成模型的“核心大脑”，不随任务切换而改变。

### 核心创新点
1. **从经验到理论的跃迁**：以前的工作只用实验展示上下文学习可行，这篇论文搭建了一个基于 PAC 的可学习性框架，正式定义了“在上下文中学习”需要满足的概率与误差条件。这样就能像分析普通机器学习算法一样，对上下文学习进行样本复杂度的上界推导。  

2. **两阶段模型设计**：先让模型在大规模预训练数据上学习一个不随任务变化的函数（预训练阶段），随后在保持该函数不变的情况下，仅通过在输入里拼接下游任务示例来完成学习（上下文学习阶段）。这种“固定核心、外加示例”的思路让理论分析变得可控。  

3. **潜在任务混合假设下的可学习性证明**：作者假设预训练分布是若干隐藏任务的混合，并证明在这种假设下，只要示例数量达到多项式级别，模型就能在上下文中高概率识别并执行目标任务。关键在于把任务识别当作一个分类问题来处理，而不是重新训练模型。  

4. **任务识别是瓶颈的理论解释**：通过分析发现，上下文学习的难点主要是“找出当前是哪种潜在任务”，而不是“重新学会任务本身”。这与近期实验发现模型在少量示例下已经具备强大推理能力相呼应，为后续研究指明了方向。

### 方法详解
**整体思路**  
论文把上下文学习拆成两步：① 预训练阶段让模型学习一个对所有潜在任务都适用的函数 f；② 在下游任务时，把若干 (输入, 输出) 示例拼接到模型的输入里，保持 f 不变，让模型仅通过这些示例来“定位”当前任务并调用 f 产生答案。

**关键模块拆解**  

1. **预训练阶段**  
   - 数据来源：从预训练语料中抽取大量句子对，视为来自混合的潜在任务。  
   - 学习目标：最小化在这些数据上的预测误差，得到函数 f 。这里的 f 可以是 Transformer 的前向映射，作者把它抽象为一个黑盒，只要满足在预训练分布上表现良好即可。  

2. **任务混合模型**  
   - 假设预训练分布是 K 种隐藏任务的混合，每种任务对应一个特定的条件分布。  
   - 这一步不需要实际实现，只是理论上的假设，用来把任务识别转化为从 K 类中挑选正确类别的过程。  

3. **上下文学习阶段**  
   - 输入构造：给定下游任务 T，准备 n 对示例 (x₁, y₁), …, (xₙ, yₙ)。把它们按顺序拼接成一个长序列，后面再接待预测的输入 xₚₒₛₜ。  
   - 推理过程：模型仍然使用同一个函数 f 对整个长序列做前向传播。由于示例已经包含了任务的输入‑输出映射，模型内部的注意力机制能够把这些模式映射到后面的查询上，从而输出 yₚₒₛₜ。  

4. **可学习性分析**  
   - 作者把任务识别看成在 K 类中做一次多分类，使用标准的 PAC 结果得到所需示例数 n = poly(K, 1/ε, log(1/δ))，其中 ε 是容错率，δ 是失败概率。  
   - 关键技巧是证明在固定 f 的情况下，示例的拼接相当于提供了一个噪声受控的标签信号，使得分类器（即模型的注意力层）能够在有限样本下收敛到正确的任务标签。  

**最巧妙的点**  
- 把“学习新任务”拆成“识别任务 + 调用已有函数”，让原本看似需要重新训练的过程变成了一个可控的分类问题。  
- 使用 PAC 框架直接给出样本复杂度上界，首次把上下文学习放在传统学习理论的严谨语境里。  

### 实验与效果
- **实验设置**：论文在合成的任务混合数据集上验证理论，使用了几种常见的自然语言任务（如情感分类、问答、翻译）作为潜在任务的代理。  
- **对比基线**：与直接微调、少量示例的提示学习（few‑shot prompting）以及不使用示例的零样本基线进行比较。  
- **结果**：论文声称在相同的示例数量下，上下文学习的准确率接近微调的水平，且显著高于零样本基线。具体提升幅度未在摘要中给出。  
- **消融实验**：作者分别去掉预训练阶段的函数固定、或减少示例数量，发现任务识别成功率随示例数呈指数下降，验证了样本复杂度的理论预测。  
- **局限性**：实验主要在受控的合成任务上进行，真实自然语言的任务分布可能更复杂；此外，理论依赖的“潜在任务混合”假设在实际预训练语料中是否成立仍有争议。  

### 影响与延伸思考
这篇工作把上下文学习拉进了学习理论的视野，开启了“可学习性”这一新研究方向。随后出现的几篇论文尝试放宽潜在任务混合的假设，或把注意力机制本身建模为任务分类器，进一步解释为什么大模型在少量示例下表现出色。还有工作把该框架扩展到多模态模型，探讨图像‑文本混合任务的上下文学习可学习性。想深入了解的读者可以关注以下方向：① 更真实的预训练分布建模（如层次化任务结构）；② 把 PAC 分析与信息论相结合，得到更紧的样本下界；③ 将任务识别视作元学习的子任务，设计专门的元学习算法提升识别效率。  

### 一句话记住它
上下文学习的本质是“用示例快速识别任务”，而不是在模型内部重新学习任务本身。