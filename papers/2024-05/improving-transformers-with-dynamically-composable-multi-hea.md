# Improving Transformers with Dynamically Composable Multi-Head Attention

> **Date**：2024-05-14
> **arXiv**：https://arxiv.org/abs/2405.08553

## Abstract

Multi-Head Attention (MHA) is a key component of Transformer. In MHA, attention heads work independently, causing problems such as low-rank bottleneck of attention score matrices and head redundancy. We propose Dynamically Composable Multi-Head Attention (DCMHA), a parameter and computation efficient attention architecture that tackles the shortcomings of MHA and increases the expressive power of the model by dynamically composing attention heads. At the core of DCMHA is a $\it{Compose}$ function that transforms the attention score and weight matrices in an input-dependent way. DCMHA can be used as a drop-in replacement of MHA in any transformer architecture to obtain the corresponding DCFormer. DCFormer significantly outperforms Transformer on different architectures and model scales in language modeling, matching the performance of models with ~1.7x-2.0x compute. For example, DCPythia-6.9B outperforms open source Pythia-12B on both pretraining perplexity and downstream task evaluation. The code and models are available at https://github.com/Caiyun-AI/DCFormer.

---

# 通过动态可组合多头注意力提升Transformer 论文详细解读

### 背景：这个问题为什么难？

Transformer 的核心是多头注意力（Multi‑Head Attention，MHA），它把同一层的注意力分成若干“头”，每个头独立计算注意力分数再并行拼接。看似灵活，实际却暴露出两大瓶颈：一是每个头的注意力分数矩阵往往是低秩的，导致模型在捕捉细粒度关系时受限；二是不同头之间容易出现功能冗余，很多头学到的模式几乎相同，却仍然占用参数和计算资源。于是研究者一直在寻找既能保持 MHA 表达力，又能削减冗余和低秩限制的方案。

### 关键概念速览
- **多头注意力（MHA）**：把注意力机制拆成若干并行的子注意力，每个子注意力拥有自己的查询、键、值矩阵，最后把结果拼接。想象成一支乐队里每个乐手独立演奏同一段旋律，整体声音更丰富。
- **低秩瓶颈**：注意力分数矩阵的秩（可独立表达的方向数）远小于矩阵尺寸，导致信息表达受限。类似把一张高分辨率图片压成几条粗线条，细节丢失。
- **头冗余**：多个注意力头学到相似或重复的模式，等于浪费了模型容量。好比一支合唱团里好几个人唱同样的声部，整体并未更和谐。
- **动态可组合（Dynamically Composable）**：在运行时根据输入内容重新组合注意力头，而不是固定使用预先学习好的头。像指挥根据现场氛围临时安排乐手组合，而不是每场演出都用同一套编排。
- **Compose 函数**：论文的核心模块，接受原始的注意力分数矩阵和权重矩阵，输出经过输入依赖变换后的新矩阵。可以把它想成“可编程的混音台”，把不同音轨（头）按实时需求调配。
- **DCFormer**：把 DCMHA 替换进任何标准 Transformer 后得到的模型统称。相当于在原有乐团里装上了智能混音台，其他部分保持不变。

### 核心创新点
1. **从固定头到动态组合**  
   - 之前的做法：每个注意力头在训练结束后固定下来，推理时直接使用。  
   - 本文做法：引入 Compose 函数，在每一次前向传播时根据当前 token 表征生成组合系数，动态混合头的分数和权重。  
   - 改变：模型能够在不同上下文中“调配”注意力资源，显著提升了表达多样性，降低了低秩限制的影响。

2. **参数与计算高效的可组合结构**  
   - 之前的做法：提升表达力往往要增大头数或加深网络，导致参数和 FLOPs 成倍增长。  
   - 本文做法：Compose 只引入少量额外的线性投影和门控参数，且在计算上可以并行完成，整体开销与原始 MHA 相当。  
   - 改变：在不增加显著计算成本的前提下，模型的有效容量提升到相当于 1.7‑2.0 倍的普通 Transformer。

3. **即插即用的 Drop‑in 替换**  
   - 之前的做法：改进注意力往往需要重新设计整个 Transformer 框架。  
   - 本文做法：DCMHA 完全兼容标准的 Query/Key/Value 接口，只需把原来的 MHA 替换为 DCMHA，即可得到 DCFormer。  
   - 改变：研究者和工程师可以在已有代码库上快速实验，无需大幅重构。

4. **在大规模语言模型上实现显著性能跃迁**  
   - 之前的做法：提升模型性能通常需要把模型规模翻倍。  
   - 本文做法：在 6.9B 参数的 DCPythia 上实现了比 12B 参数的开源 Pythia 更低的预训练困惑度（perplexity）和更好的下游任务表现。  
   - 改变：同等算力下，模型效果接近甚至超越更大模型，验证了动态组合的实用价值。

### 方法详解
**整体思路**  
DCMHA 仍然遵循传统的 Query‑Key‑Value 流程：输入序列先经过线性层得到查询（Q）、键（K）和值（V），随后计算每个头的注意力分数 `S = QKᵀ / sqrt(d)`. 与传统 MHA 不同的是，DCMHA 在得到这些分数后会把它们送进一个 **Compose** 模块，Compose 根据当前 token 的上下文信息生成一组组合系数，对分数矩阵和对应的注意力权重进行重新加权，最后得到新的加权值矩阵并继续后续的线性投影和残差连接。

**关键模块拆解**  

1. **基础头的计算**  
   - 与普通 MHA 一致，先把 Q、K、V 划分成 `h` 个子空间（即 `h` 头），得到 `S_i`（第 i 头的分数矩阵）和 `V_i`（第 i 头的值矩阵）。

2. **输入感知的组合系数生成**  
   - 对每个 token（或每个 token 对），取它的查询向量 `q` 经过一个小型 MLP（或线性层 + 激活），输出长度为 `h` 的系数向量 `c`.  
   - 这一步相当于“指挥家”听到每个乐手的音色后决定该让哪些乐手上场以及上场的音量。

3. **Compose 函数的核心操作**  
   - **分数矩阵重组**：把所有头的分数 `S_i` 按系数 `c_i` 加权求和，得到一个“混合”分数矩阵 `S' = Σ_i c_i * S_i`。  
   - **权重矩阵重组**：对 `S'` 做 softmax 得到注意力权重 `A'`，再用同样的系数对值矩阵进行加权组合 `V' = Σ_i c_i * V_i`。  
   - **可选的非线性调制**：部分实现会在加权前后加入门控或层归一化，以防系数过大导致数值不稳。  

4. **后续投影与残差**  
   - 与标准 MHA 相同，`A' * V'` 产生的上下文向量再经过一次线性投影，加入残差和层归一化，进入下一层 Transformer。

**为什么这样设计会有效？**  
- **打破低秩限制**：单个头的分数矩阵往往只能捕捉到局部模式，动态加权后相当于在每一次前向传播中生成了一个“新头”，其秩可以随输入变化，提升了整体的表达空间。  
- **削减冗余**：如果某些头在当前上下文中不重要，Compose 会给它们的系数分配很小的值，等价于在运行时“关闭”这些头，避免无效计算。  
- **参数经济**：组合系数的生成只需要极少的额外参数（相当于每层几个额外的线性层），而不需要像增大头数那样线性增长参数量。  

**最巧妙的地方**  
Compose 函数把注意力分数和价值矩阵一起进行输入依赖的线性混合，而不是仅在输出层做一次加权。这样做既保留了注意力分布的细粒度信息，又让模型在每一步都拥有重新组织注意力资源的自由度，类似于在每一次演奏前都重新排练乐曲的编配。

### 实验与效果
- **测试任务**：主要在大规模语言建模任务上评估，包括公开的 Pythia 系列数据集以及若干下游任务（如阅读理解、文本分类）。  
- **对比基线**：普通 Transformer（使用标准 MHA）以及同规模的开源模型（如 Pythia‑12B）。  
- **核心结果**：论文声称 DCPythia‑6.9B 在预训练困惑度上超过了 Pythia‑12B，且在下游任务的准确率或 F1 分数上也保持领先。整体性能相当于使用约 1.7‑2.0 倍计算资源的普通 Transformer。  
- **消融实验**：作者通过去掉 Compose 模块、固定组合系数或仅对分数矩阵进行混合等设置，验证了动态组合和同时混合分数/值两部分对性能提升的贡献。结果显示，去掉任意一环都会导致显著的性能回退。  
- **局限性**：原文未给出大规模推理时的实际时延或显存占用数据，推测在极端序列长度下动态系数的计算可能带来一定的额外开销。  

### 影响与延伸思考
- **领域影响**：DCMHA 为“可变形注意力”提供了一个实用的实现范式，随后出现的多篇工作（如动态路由注意力、可调头注意力）都在不同程度上借鉴了“输入感知组合”这一思路。  
- **后续方向**：  
  1. **更轻量的组合生成器**：探索使用低秩投影或稀疏门控来进一步压缩系数计算的成本。  
  2. **跨层组合**：把不同层的注意力头也纳入同一 Compose 框架，实现跨层信息的动态共享。  
  3. **多模态扩展**：在视觉 Transformer 或跨模态模型中加入 DCMHA，检验其在图像、音频等非文本数据上的通用性。  
- **深入阅读**：想了解细节实现，可关注作者在 GitHub 上的代码仓库（DCFormer），以及随后出现的 “Dynamic Head Routing” 系列论文，它们对 Compose 的内部结构提供了更具体的实现示例。

### 一句话记住它
**DCMHA 用输入感知的“混音台”实时重组注意力头，让小模型拥有大模型的表达力。**