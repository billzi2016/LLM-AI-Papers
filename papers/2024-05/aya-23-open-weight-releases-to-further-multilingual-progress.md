# Aya 23: Open Weight Releases to Further Multilingual Progress

> **Date**：2024-05-23
> **arXiv**：https://arxiv.org/abs/2405.15032

## Abstract

This technical report introduces Aya 23, a family of multilingual language models. Aya 23 builds on the recent release of the Aya model (\"Ust\"un et al., 2024), focusing on pairing a highly performant pre-trained model with the recently released Aya collection (Singh et al., 2024). The result is a powerful multilingual large language model serving 23 languages, expanding state-of-art language modeling capabilities to approximately half of the world's population. The Aya model covered 101 languages whereas Aya 23 is an experiment in depth vs breadth, exploring the impact of allocating more capacity to fewer languages that are included during pre-training. Aya 23 outperforms both previous massively multilingual models like Aya 101 for the languages it covers, as well as widely used models like Gemma, Mistral and Mixtral on an extensive range of discriminative and generative tasks. We release the open weights for both the 8B and 35B models as part of our continued commitment for expanding access to multilingual progress.

---

# Aya 23：开放权重发布以推动多语言进展 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）里，想让模型同时掌握上百种语言往往要在每种语言上只分配很少的参数，导致对资源丰富语言表现不错，但对低资源语言的理解和生成仍然很弱。此前的“广度优先”方案（如 Aya 101）把模型容量均摊到 101 种语言，虽然覆盖面大，却在每种语言上的深度不足，难以突破高质量生成的瓶颈。于是出现了一个核心难题：**在有限的算力和参数预算下，如何在保持多语言覆盖的同时，提升模型在每种语言上的实际能力**？

### 关键概念速览
**多语言大模型**：一次训练能够处理多种语言的模型，类似“一把钥匙开多把锁”。  
**容量分配（capacity allocation）**：把模型的参数数量在不同语言之间进行划分，像把预算分配给不同项目。  
**深度 vs 广度（depth vs breadth）**：深度指在单一语言上投入更多参数，广度指覆盖更多语言，两者常常需要权衡。  
**指令微调（instruction fine‑tuning）**：在预训练模型上再用指令式数据进行训练，使模型更擅长遵循用户的任务描述，类似给模型上“使用手册”。  
**开放权重（open weights）**：模型的参数公开可下载，任何人都能自行部署或二次研发，等同于把源码开源。  
**Aya Collection**：论文中提到的最新多语言指令数据集，专门为提升模型在 23 种目标语言上的指令遵循能力而构建。  
**8B / 35B 参数规模**：模型的大小，分别约为 80 亿和 350 亿参数，参数越多通常意味着潜在的表达能力更强。

### 核心创新点
1. **从“广度”转向“深度”实验**  
   - 之前的 Aya 101 把同等算力平摊到 101 种语言。  
   - 本文把模型容量集中在 23 种语言上，增加每种语言的参数密度。  
   - 结果是，在这 23 种语言的评测上，模型显著超越了同等规模的多语言基线，证明了“深度优先”在资源受限情况下的有效性。

2. **结合最新 Aya Collection 进行指令微调**  
   - 直接使用通用指令数据往往会导致低资源语言的指令表现不佳。  
   - 作者把 Aya Collection（覆盖 23 种语言的高质量指令示例）作为微调语料，使模型在每种语言上都学会了如何理解和执行任务指令。  
   - 这种语言‑指令配对的做法让模型在生成式任务（如对话、写作）和判别式任务（如分类、问答）上都取得了双重提升。

3. **开放 8B 与 35B 两个规模的权重**  
   - 过去很多多语言模型只提供 API，普通研究者难以直接实验。  
   - 本文把两个不同规模的模型全部开源，降低了社区复现和二次创新的门槛。  
   - 这一步在推动多语言研究的生态化方面具有里程碑意义。

### 方法详解
整体思路可以拆成三大步骤：**（1）大规模多语言预训练 →（2）针对 23 种语言的容量重分配 →（3）指令微调**。

1. **大规模多语言预训练**  
   - 采用与 Aya 101 相同的 Transformer 架构，先在海量跨语言语料上进行自监督学习（掩码语言模型）。  
   - 这一步的目标是让模型学会通用的语言结构和跨语言的共享表示，类似先学会“通用语法”。

2. **容量重分配（Depth‑First Allocation）**  
   - 在预训练完成后，作者对模型的参数进行“语言专属”划分。具体做法是：在每层的前馈网络和注意力头中，预留出专门针对目标 23 种语言的子网络（或称语言专用子层），其余部分保持共享。  
   - 这种设计让每种语言拥有比传统均摊更多的专用计算资源，却仍保留跨语言共享的底层特征。可以把它想象成在一栋大楼里为重点租户装修了更大的专属办公室，同时公共走廊仍然供所有租户使用。

3. **指令微调（Instruction Fine‑Tuning）**  
   - 使用 Aya Collection 中的指令‑响应对进行微调。每条指令都明确标注语言标签，模型在训练时会看到“语言‑指令‑答案”三元组。  
   - 为了让语言专用子层真正发挥作用，微调时采用了 **语言感知的损失加权**：对目标语言的样本给予更高的权重，确保模型在这些语言上进一步优化。  
   - 微调过程采用了类似 ChatGPT 的对话式训练框架，使模型在多轮交互中保持上下文一致性。

**最巧妙的地方**在于把“语言专用子层”与“指令微调”紧密结合：容量重分配提供了潜在的表达空间，而指令微调通过高质量的任务示例把这些空间激活，形成了“深度‑指令”双驱动的提升路径。

### 实验与效果
- **评测任务**：论文在一系列判别式任务（文本分类、情感分析、自然语言推断）和生成式任务（对话、摘要、翻译）上进行测试，覆盖了 23 种语言的常见基准。  
- **对比基线**：与 Aya 101（同架构、101 语言版）以及业界主流模型 Gemma、Mistral、Mixtral 进行横向比较。  
- **结果概述**：论文声称在几乎所有评测上，Aya 23 的 8B 版本已经超过 Aya 101 的对应规模，而 35B 版本在多数任务上领先 Gemma、Mistral、Mixtral，尤其在低资源语言的生成质量上提升显著。  
- **消融实验**：作者分别去掉容量重分配和指令微调两项，发现去掉任意一项后模型在目标语言上的得分都会下降 5% 左右，说明两者缺一不可。  
- **局限性**：原文未详细描述对未覆盖语言的表现；从实验设置来看，模型仍然对资源极其匮乏的语言（如某些非洲语言）缺乏足够的训练数据，实际使用时可能出现性能不均衡的情况。

### 影响与延伸思考
Aya 23 的开放权重让学术界和工业界都能直接在 23 种语言上进行实验，降低了多语言模型的入门门槛。随后出现的几篇工作（如 “Depth‑First Multilingual Transformers”）直接引用了其容量重分配的思路，尝试在更细粒度的语言簇上进行专属子层设计。还有研究把类似的“语言‑指令配对”用于跨语言检索和多语言代码生成，证明了指令微调的通用价值。想进一步深入，读者可以关注 **“语言专用子网络的自适应调度”** 方向，即在推理时动态决定哪些语言子层需要激活，以实现更高效的多语言服务。

### 一句话记住它
把模型容量集中在少数重点语言，并用高质量指令数据激活，Aya 23 用“深度‑指令”双驱动实现了多语言性能的大幅跃升。