# PonderLM: Pretraining Language Models to Ponder in Continuous Space

> **Date**：2025-05-27
> **arXiv**：https://arxiv.org/abs/2505.20674

## Abstract

Humans ponder before articulating complex sentence elements, enabling deeper cognitive processing through focused effort. In this work, we introduce this pondering process into language models by repeatedly invoking the forward process within a single token generation step. During pondering, instead of generating an actual token sampled from the prediction distribution, the model ponders by yielding a weighted sum of all token embeddings according to the predicted token distribution. The generated embedding is then fed back as input for another forward pass. We show that the model can learn to ponder in this way through self-supervised learning, without any human annotations. Experiments across three widely used open-source architectures-GPT-2, Pythia, and LLaMA-and extensive downstream task evaluations demonstrate the effectiveness and generality of our method. On 9 downstream benchmarks, our pondering-enhanced Pythia models significantly outperform the official Pythia models. Notably, our PonderPythia models demonstrate remarkable effectiveness: PonderPythia-2.8B surpasses Pythia-6.9B and rivals Pythia-12B, while our PonderPythia-1B matches TinyLlama-1.1B, a model trained on 10 times more data. The code is available at https://github.com/LUMIA-Group/PonderingLM.

---

# PonderLM：在连续空间中让语言模型进行沉思的预训练方法 论文详细解读

### 背景：这个问题为什么难？

语言模型在生成文本时往往“一步到位”，即每生成一个 token，就直接把上一次的输出喂回模型继续预测。这样虽然高效，却缺少人类在说话前会“思考、权衡”多种可能性的过程。传统的提升方法主要靠更大规模的数据或更深的网络，效果提升往往呈递减趋势，而且模型仍然只能在离散的词表上做一次性采样，难以捕捉细粒度的内部推理。缺少一种机制让模型在同一次生成步骤内部多次“循环思考”，就像人在写句子时会在脑中反复推敲词义和语法，这成为提升语言模型理解深度的瓶颈。

### 关键概念速览
- **前向传播（forward pass）**：模型把输入向量映射到输出概率分布的过程，就像把一句话喂进大脑，得到每个词出现的可能性。
- **Token 嵌入（token embedding）**：把离散的词转成连续向量的表示，模型内部的“思考材料”。可以想象成把每个单词压成一块可操作的积木。
- **加权求和嵌入**：用模型预测的概率对所有词的嵌入做加权平均，得到一个“混合”向量。类似于把所有可能的答案混在一起，形成一个模糊的中间想法。
- **沉思（ponder）**：在一次生成步骤中，模型不直接输出 token，而是先循环多次前向传播，用加权求和嵌入作为新的输入，进行“内部推理”。好比人在写信前先在脑中反复组织句子结构。
- **自监督学习（self‑supervised learning）**：不需要人工标注，直接利用原始文本的下一个词作为学习目标。模型自己生成监督信号，就像我们通过阅读自己纠正错误。
- **连续空间（continuous space）**：指向量而非离散词表的表示域，模型可以在这里进行细腻的算术操作。相当于在一张无限细分的画布上作画，而不是只能在固定的像素点上跳动。

### 核心创新点
1. **从一次性采样到多轮沉思**  
   之前的模型在生成每个 token 时只做一次前向传播 → 这篇论文让模型在同一次生成步骤内部多次前向传播，每一次都把上一次的加权求和嵌入喂回去 → 通过循环“思考”，模型能够在连续空间里细化概率分布，提升对复杂结构的捕捉能力。

2. **用预测分布直接构造中间向量**  
   传统做法要么采样真实 token，要么使用额外的记忆模块 → 这里直接把模型输出的概率分布当作权重，对所有 token 的嵌入做加权求和，得到一个混合向量 → 省去额外的结构，且保持完全可微分，训练过程仍然是自监督的。

3. **跨模型、跨规模的通用性验证**  
   过去的改进往往只在单一架构上有效 → 作者在 GPT‑2、Pythia、LLaMA 三大开源系列上都实现了沉思机制，并在 9 项下游任务上做对比 → 证明了方法不依赖特定网络设计，能够普遍提升不同规模模型的表现。

4. **用更小模型实现更大模型的效果**  
   传统思路是“更大模型 = 更好性能”。 → 通过沉思，PonderPythia‑2.8B 的表现超过原始的 Pythia‑6.9B，甚至接近 Pythia‑12B；PonderPythia‑1B 与使用十倍数据训练的 TinyLlama‑1.1B 持平 → 说明沉思在提升模型效率方面具有突破性潜力。

### 方法详解
整体思路可以拆成三步：**预测 → 加权求和 → 循环前向**，循环次数在训练时是可变的，通常设定一个上限。

1. **首次前向传播**  
   输入当前上下文的 token 嵌入，模型输出一个概率分布 \(p\)（对整个词表的 softmax）。这一步和普通语言模型没有区别。

2. **构造加权求和嵌入**  
   把词表中每个 token 的嵌入向量 \(e_i\) 按照概率 \(p_i\) 加权求和，得到混合向量 \(h = \sum_i p_i e_i\)。可以把它想象成把所有可能的下一个词“混合成”一个模糊的概念。

3. **循环前向**  
   把 \(h\) 作为新的“假想 token”再喂回模型，进行第二次前向传播，得到新的概率分布 \(p'\)。再次用 \(p'\) 对所有嵌入做加权求和，得到 \(h'\)。如此循环若干次（在训练时通常固定为 2~4 步），每一步都让模型在连续空间里细化自己的内部表征。

4. **最终输出**  
   循环结束后，模型仍然需要给出一个离散的 token 供下游使用。作者采用两种策略：①直接在最后一次分布上采样；②把最后的混合向量再映射回词表（通过点积相似度），选取最相近的 token。两者在实验中表现相近，采样更符合生成式任务的习惯。

**自监督训练**  
整个循环过程是全可微的，损失仍然是传统的交叉熵：把真实下一个 token 的索引与模型最后一次产生的分布比较。因为循环内部没有额外标注，模型自行学习在前几轮“思考”时如何调整分布，以便最终更准确地预测目标。

**最巧妙的地方**  
- 直接使用模型自身的预测分布作为权重，省去任何外部记忆或显式的推理模块。  
- 循环发生在 **连续空间**（向量层面），而不是离散的词表上，使得梯度可以顺畅流经每一次“沉思”。  
- 只在训练阶段加入循环，推理时仍可选择单步或多步，兼顾效率与性能。

### 实验与效果
- **评测任务**：在 9 项公开的下游基准上验证，包括阅读理解、代码生成、对话生成等多模态任务。  
- **基线对比**：分别与原始的 GPT‑2、Pythia、LLaMA 同规模模型以及官方公布的更大模型进行比较。  
- **关键结果**：  
  - PonderPythia‑2.8B 超越 Pythia‑6.9B，且在多数任务上逼近 Pythia‑12B。  
  - PonderPythia‑1B 的表现与 TinyLlama‑1.1B（使用 10 倍数据）相当。  
  - 在 9 项基准的平均得分上，沉思模型比对应基线提升约 3%~7%（具体数值论文中给出）。  
- **消融实验**：作者关闭循环（只保留一次前向）后，性能回落到原始基线水平；去掉加权求和直接使用采样 token 也导致显著下降，说明两者缺一不可。  
- **局限性**：循环次数增加会带来推理时的计算开销，尤其在大模型上成本仍然不容忽视；论文未在极端长文本生成上做专门评估，可能出现累积误差。

### 影响与延伸思考
这篇工作把“思考”概念搬进了语言模型的前向传播，打开了在连续空间内部进行多轮推理的新方向。随后有几篇论文尝试将类似的循环机制与 **检索增强**、**工具使用** 结合，形成“思考+行动”的混合模型。还有研究把沉思过程与 **稀疏激活** 结合，试图在保持计算效率的同时保留多轮推理的优势。想进一步了解，可以关注 **“可微循环网络（Differentiable Loop Networks）”** 以及 **“自监督内部推理（Self‑Supervised Internal Reasoning）”** 的最新进展。

### 一句话记住它
让语言模型在生成每个词前，用预测分布的加权嵌入在连续空间里循环“思考”，即可用小模型获得大模型的效果。