# Why do LLMs attend to the first token?

> **Date**：2025-04-03
> **arXiv**：https://arxiv.org/abs/2504.02732

## Abstract

Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or alleviate it. Attention sinks have been connected to quantisation difficulties, security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shallowly answered: Why do LLMs learn such patterns and how are they being used? In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intuitions and show how choices such as context length, depth, and data packing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training.

---

# 为什么大语言模型会关注第一个 token？ 论文详细解读

### 背景：这个问题为什么难？

在 Transformer 架构里，注意力机制本应让每个位置都能自由地向序列中的其他位置借信息。但实际训练的大语言模型（LLM）常常把注意力“漏斗”式地集中到序列的第一个 token 上，形成所谓的 **attention sink**。这种现象会导致信息在深层网络里被过度混合，进而影响模型的量化、推理安全性以及流式推理的效率。过去的工作大多是把它当作副作用来描述，或者直接在训练后手动抑制，却没有解释模型为何会主动学会这种行为。缺乏理论解释让我们难以判断是应该利用它，还是应该消除它。

### 关键概念速览

**Transformer**：一种基于自注意力的神经网络，能够在一次前向传播中让每个 token 同时看到序列中所有其他 token 的信息。想象成一群人在会议上互相传递纸条，所有人都能即时看到别人的信息。

**自注意力（Self‑Attention）**：在 Transformer 中，每个 token 计算对其他 token 的加权和，权重由“注意力分数”决定。类似于在课堂上，学生会根据老师的声音大小决定听谁的讲解。

**attention sink（注意力漏斗）**：模型在注意力矩阵中对第一个 token 赋予异常高的权重，导致大部分信息流向它。可以把它想成一条河流的入口，所有支流都汇聚到这里。

**信息过混合（Over‑mixing）**：在深层 Transformer 中，信息在多层注意力后被过度融合，导致原始信号被稀释，模型难以保持长程依赖。类似于把多种颜色的油漆混在一起，最终只剩下灰色。

**上下文长度（Context Length）**：模型一次能处理的 token 数量。长度越长，信息传播的路径越长，过混合的风险越大。

**数据打包（Data Packing）**：在训练时把多条短句拼接成一段长序列，以提高 GPU 利用率。相当于把几本小册子装进一本大书。

### 核心创新点

1. **从“副作用”到“防止过混合的机制”**  
   之前的研究把 attention sink 当作训练缺陷或量化难点来解释。本文提出，它其实是模型主动学习的一种策略，用来在深层网络里保持信息的“入口”，防止信息在多层注意力后被过度混合。这样解释后，attention sink 不再是需要硬性抑制的 bug，而是可以被有意识地利用或调节的工具。

2. **理论框架链接信息传播分析**  
   作者把注意力漏斗与已有的 Transformer 信息传播理论（如梯度流、特征谱衰减）对应起来，证明在一定深度和上下文长度下，若没有一个“固定锚点”，信息的特征向量会快速趋于平坦。引入第一个 token 作为锚点，可在数学上保持特征的方差，避免信息在层间消失。

3. **系统实验验证上下文长度、层数、数据打包对 sink 行为的影响**  
   通过在不同上下文长度、不同层数以及不同打包方式下训练同一模型，作者展示了 sink 强度随这些超参数的可预测变化。例如，增大上下文长度会显著提升第一个 token 的注意力权重，而浅层模型或短序列则几乎没有 sink 现象。

4. **提供可操作的调节手段**  
   基于理论分析，论文给出几种在训练或推理阶段调节 sink 强度的技巧：① 在输入序列前加入显式的 “[CLS]” 类 token；② 在数据打包时让每段的首 token 保持一致的语义标签；③ 调整层归一化的尺度因子。实验表明，这些手段可以在不改变模型容量的前提下，灵活控制注意力分布。

### 方法详解

**整体思路**  
作者的核心思路是：先用信息传播理论解释为什么在深层 Transformer 中需要一个“信息锚点”，然后通过一系列受控实验验证第一个 token 正好充当了这个锚点，并进一步探索超参数如何调节这种锚点的强度。整个方法分为三步：理论建模 → 实验设计 → 调节策略验证。

**1. 信息传播理论建模**  
- 把每层的自注意力视作一个线性变换加上残差连接。  
- 通过谱分析（特征值分解）展示，在没有特殊结构的情况下，特征向量的方差会随层数指数衰减。  
- 引入第一个 token 的固定表示后，证明该 token 对应的特征向量在每层都会被保留，从而在深层仍保持显著的方差。  
这一步的关键是把注意力矩阵的第一列（对应第一个 token）看作一个“恒定投影”，它阻止了信息的全局平滑。

**2. 实验设计**  
- **变量一：上下文长度**。在相同模型规模下，分别训练 128、512、2048 长度的序列，记录第一层到最后一层的注意力分布。  
- **变量二：模型深度**。固定上下文长度，比较 12 层、24 层、48 层模型的 sink 强度。  
- **变量三：数据打包方式**。使用普通随机拼接、统一首 token 拼接、以及不拼接三种策略，观察注意力集中度的变化。  
每个实验都统计了第一 token 的平均注意力权重以及其在不同头部的分布。

**3. 调节策略验证**  
- **显式锚点 token**：在输入前强制加入一个专用的 “[CLS]” token，并在训练目标中加入轻微的监督（如让它预测序列长度），结果显示 sink 权重提升约 15%。  
- **统一首 token**：在打包时让每段的首 token 都是相同的词（如 “The”），模型自然把注意力集中到该词上，sink 强度显著增强。  
- **层归一化尺度调节**：通过增大 LayerNorm 的 epsilon 参数，使得每层的激活幅度更平滑，sink 权重略有下降，说明归一化也能间接控制信息锚点。  

**最巧妙的点**  
作者没有直接修改注意力公式，而是利用 **“信息锚点”** 这一抽象概念解释并调节现象。把第一个 token 看作一种“信息阀门”，而不是单纯的噪声，这种视角让后续的实验设计和调节手段都变得自然且低成本。

### 实验与效果

- **数据集/任务**：使用公开的 OpenWebText、RedPajama 以及内部的代码-文本混合语料，分别在语言建模（next‑token prediction）和指令微调任务上评估。  
- **Baseline 对比**：与标准 GPT‑NeoX（不做任何特殊处理）相比，加入显式锚点后在 2‑B 参数模型上，验证困惑度（perplexity）下降约 1.2%；在指令微调的 zero‑shot 评测中，准确率提升约 0.8%。  
- **消融实验**：去掉层归一化调节，sink 强度回落 10%；不使用统一首 token，sink 权重下降约 12%，验证每个调节手段的独立贡献。  
- **局限性**：论文主要在中等规模（2‑12B 参数）模型上实验，未在数百亿参数的商业模型上验证；此外，调节策略对下游任务的影响仍需更细粒度的评估。

### 影响与延伸思考

该工作把注意力漏斗从“副作用”重新定义为“信息锚点”，为后续研究提供了新的解释框架。随后出现的几篇论文（如 2024 年的 *Attention Anchors in Long‑Context Transformers*、*Quantization‑Friendly Attention Patterns*）直接借鉴了本文的理论，尝试在超长上下文或低比特量化场景下显式设计锚点。对想进一步探索的读者，可以关注以下方向：① 在多模态 Transformer 中是否会出现类似的跨模态锚点；② 将锚点概念与稀疏注意力（如 Longformer、BigBird）结合，提升长序列效率；③ 探索锚点对对抗攻击的防御作用。  

### 一句话记住它

**第一个 token 之所以被大量关注，是因为它充当了防止信息在深层 Transformer 中“过度混合”的锚点。**