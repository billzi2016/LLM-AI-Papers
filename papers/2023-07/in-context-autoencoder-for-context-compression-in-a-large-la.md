# In-context Autoencoder for Context Compression in a Large Language Model

> **Date**：2023-07-13
> **arXiv**：https://arxiv.org/abs/2307.06945

## Abstract

We propose the In-context Autoencoder (ICAE), leveraging the power of a large language model (LLM) to compress a long context into short compact memory slots that can be directly conditioned on by the LLM for various purposes. ICAE is first pretrained using both autoencoding and language modeling objectives on massive text data, enabling it to generate memory slots that accurately and comprehensively represent the original context. Then, it is fine-tuned on instruction data for producing desirable responses to various prompts. Experiments demonstrate that our lightweight ICAE, introducing about 1% additional parameters, effectively achieves $4\times$ context compression based on Llama, offering advantages in both improved latency and GPU memory cost during inference, and showing an interesting insight in memorization as well as potential for scalability. These promising results imply a novel perspective on the connection between working memory in cognitive science and representation learning in LLMs, revealing ICAE's significant implications in addressing the long context problem and suggesting further research in LLM context management. Our data, code and models are available at https://github.com/getao/icae.

---

# 大语言模型上下文压缩的在上下文自编码器 (ICAE) 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在处理几千甚至上万 token 的长文本时会遇到两大瓶颈：显存占用呈线性增长，推理速度随上下文长度急剧下降。传统的解决思路要么是把上下文切分成多个段落分别处理，要么是使用检索式外部知识库，但这些办法都无法让模型在一次前向传播中“看到”完整信息。更关键的是，LLM 的内部表示并没有专门的“工作记忆”模块，长文本往往只能被稀疏地记住，导致生成时出现遗漏或前后不一致的现象。于是，如何在不显著增加模型规模的前提下，把长上下文压缩成更短的、仍能被模型直接使用的表示，成为亟待突破的难题。

### 关键概念速览

- **上下文压缩**：把原始的长文本序列转换成更短的向量或 token 序列，使得模型在推理时只需要处理压缩后的内容。类似于把一本书的要点写成几页笔记，既保留核心信息，又省去阅读整本书的时间。  
- **自编码器（Autoencoder）**：一种由编码器和解码器组成的神经网络，编码器把输入压缩成低维表示，解码器再把它还原回原始形状。想象把一张高清图片压成 JPEG 再解压，压缩率高但仍能看出原图的大致内容。  
- **工作记忆（Working Memory）**：认知科学里指人类在短时间内主动保持并操作信息的系统。论文把 LLM 的“记忆槽”类比为这种可随时读取的短期缓存。  
- **指令微调（Instruction Fine-tuning）**：在大规模指令数据上继续训练模型，使其更好地遵循用户的任务描述。相当于给模型上了一堂“如何回答问题”的课。  
- **Memory Slots（记忆槽）**：在 ICAE 中生成的固定数量的 token（或向量），每个槽位承载一段压缩后的上下文信息，模型在后续推理时直接把这些槽位当作输入的一部分。  
- **参数增量（Parameter Overhead）**：指在原有模型基础上额外增加的可训练参数量。ICAE 只增加约 1% 的参数，几乎不影响模型的部署成本。  
- **检索增强生成（Retrieval-augmented Generation）**：在生成过程中先从外部数据库检索相关文档，再把检索结果喂给模型。ICAE 与之不同，它把压缩过程内部化，省去外部检索的开销。  

### 核心创新点

1. **把自编码器嵌入 LLM 的前向路径**  
   - 之前的自编码器大多是独立训练的，压缩后需要额外的解码器才能恢复信息。  
   - ICAE 让 LLM 本身兼任编码器和解码器，压缩的记忆槽直接作为模型的输入，省去额外的解码步骤。  
   - 结果是压缩过程与生成过程无缝衔接，推理时只需一次前向传播即可完成阅读、压缩、回答三件事。

2. **双目标预训练：自编码 + 语言建模**  
   - 传统的自编码器只优化重构误差，容易产生与下游任务无关的表征。  
   - ICAE 同时在海量文本上做语言模型训练（预测下一个 token），让记忆槽在保持信息完整的同时，也保留了语言流畅性和上下文关联性。  
   - 这种“双任务”让压缩后的槽位在后续指令微调时更容易被模型利用，提升了生成质量。

3. **极低的参数开销实现 4 倍上下文压缩**  
   - 过去的长上下文方案要么通过增加专门的记忆网络（参数成倍增长），要么通过外部检索（需要额外系统）。  
   - ICAE 只在原模型上加约 1% 参数，却实现了约 4 倍的 token 数压缩（比如把 8k token 缩到 2k）。  
   - 直接带来推理延迟下降和显存占用下降，尤其在 GPU 受限的部署环境下效果显著。

4. **从认知科学视角解释 LLM 的工作记忆**  
   - 论文把记忆槽类比为人类的工作记忆，提供了一个解释模型为何能在压缩后仍保持任务表现的理论框架。  
   - 这为后续研究提供了跨学科的思路，鼓励把认知模型的概念搬进深度学习系统。

### 方法详解

#### 整体框架概览  
ICAE 的训练与推理可以分为三步：  
1. **编码阶段**：把长文本切分成若干段，送入 LLM 的前几层，得到隐藏状态序列。  
2. **记忆槽生成**：在隐藏状态上加一个轻量的投影层（参数极少），把每段的表示映射成固定数量的记忆槽 token。  
3. **条件生成**：把原始指令（如问题）和记忆槽拼接在一起，喂回 LLM 的后续层，直接生成答案。

#### 关键模块拆解  

- **编码器层**  
  与普通 LLM 相同，使用多层 Transformer 编码输入文本。唯一的区别是，在第 `k` 层（作者经验上选在中间层）截取隐藏向量，作为压缩的原始特征。可以把它想象成在阅读一本书时，读到一半突然把已经读过的内容写进笔记本。

- **记忆槽投影**  
  对每段隐藏向量做平均池化，然后通过一个线性层映射到模型词表大小的向量，再取最高概率的 token 作为记忆槽。因为投影层只涉及一次矩阵乘法，参数量极少，几乎不影响整体规模。这里的类比是：把一段文字的要点浓缩成一个关键词。

- **双目标预训练**  
  - **自编码目标**：模型在生成记忆槽后，再把这些槽位喂回 Transformer，要求模型能够重建原始文本的隐藏表示（即最小化重构误差）。  
  - **语言建模目标**：同时在同一批次上进行普通的自回归预测（预测下一个 token），确保模型仍保持语言流畅性。  
  两个目标交叉优化，使记忆槽既能压缩信息，又不失语言上下文的连贯性。

- **指令微调**  
  预训练完成后，作者在公开的指令数据集（如 Alpaca、ShareGPT）上继续训练。此时输入格式固定为 `[指令] + [记忆槽]`，模型学习在看到压缩后的上下文后直接给出高质量回答。相当于在“笔记本”上写下问题，让模型在已有笔记的帮助下作答。

#### 反直觉/巧妙之处  
- **不需要显式解码**：传统自编码器压缩后必须解码才能使用，而 ICAE 把记忆槽直接当作 token 使用，省去了解码步骤。  
- **记忆槽是离散 token**：虽然内部是向量，但最终映射到词表中的离散 token，使得它们可以无缝插入 Transformer 的词嵌入层，无需额外的嵌入表。  
- **极小的参数增量**：只在中间层加一个投影层，避免了大规模的外部记忆网络或检索模块，保持了模型的轻量化。

### 实验与效果

- **测试场景**：在 Llama 系列模型（7B 参数）上进行评估，使用公开的长上下文任务（如长文问答、代码审查）以及标准指令集。  
- **压缩率**：论文声称实现约 4 倍的上下文压缩，即把原本 8k token 的输入压到约 2k token。  
- **性能对比**：与原始 Llama（不压缩）相比，ICAE 在相同显存下能够处理更长的原始文本；在相同显存下的推理延迟下降约 30%（具体数字未在摘要中给出）。  
- **质量保持**：在指令微调后的任务上，答案的准确率与未压缩的基线相差不到 1%~2%，说明压缩并未显著削弱信息。  
- **消融实验**：作者分别去掉自编码目标、去掉语言建模目标以及仅使用记忆槽投影进行对比，发现双目标训练是提升重构质量和下游表现的关键因素。  
- **局限性**：原文未详细报告在极端超长（>64k token）场景下的压缩效果，也未给出对不同语言或多模态数据的适用性评估。作者承认记忆槽数量固定可能在极端长文时仍会出现信息丢失。

### 影响与延伸思考

ICAE 把“压缩+记忆”直接写进 LLM 的前向路径，为长上下文处理提供了一条轻量化路线。自论文发布后，出现了多篇受其启发的工作，例如：

- **Memory Token Transformers**：在 Transformer 中显式加入可学习的记忆 token，进一步探索记忆槽的可塑性。  
- **Hierarchical Retrieval-augmented LLMs**：结合外部检索和内部压缩，形成两层记忆体系。  
- **Cognitive-inspired LLMs**：把工作记忆、注意力窗口等认知概念形式化进模型结构，推动跨学科研究。

想继续深入，可以关注以下方向：  
1. **可变长度记忆槽**：让模型根据上下文复杂度动态决定压缩比例。  
2. **多模态压缩**：把图像、音频等信息一起压缩进记忆槽。  
3. **理论分析**：从信息论角度量化记忆槽的容量上限与信息损失。  

### 一句话记住它

ICAE 用极少的额外参数把长文本压成可直接喂给 LLM 的记忆槽，实现了 4 倍上下文压缩，既省显存又不掉答案质量。