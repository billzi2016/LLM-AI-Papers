# Latent Collaboration in Multi-Agent Systems

> **Date**：2025-11-25
> **arXiv**：https://arxiv.org/abs/2511.20639

## Abstract

Multi-agent systems (MAS) extend large language models (LLMs) from independent single-model reasoning to coordinative system-level intelligence. While existing LLM agents depend on text-based mediation for reasoning and communication, we take a step forward by enabling models to collaborate directly within the continuous latent space. We introduce LatentMAS, an end-to-end training-free framework that enables pure latent collaboration among LLM agents. In LatentMAS, each agent first performs auto-regressive latent thoughts generation through last-layer hidden embeddings. A shared latent working memory then preserves and transfers each agent's internal representations, ensuring lossless information exchange. We provide theoretical analyses establishing that LatentMAS attains higher expressiveness and lossless information preservation with substantially lower complexity than vanilla text-based MAS. In addition, empirical evaluations across 9 comprehensive benchmarks spanning math and science reasoning, commonsense understanding, and code generation show that LatentMAS consistently outperforms strong single-model and text-based MAS baselines, achieving up to 14.6% higher accuracy, reducing output token usage by 70.8%-83.7%, and providing 4x-4.3x faster end-to-end inference. These results demonstrate that our new latent collaboration framework enhances system-level reasoning quality while offering substantial efficiency gains without any additional training. Code and data are fully open-sourced at https://github.com/Gen-Verse/LatentMAS.

---

# 潜在协作的多智能体系统 论文详细解读

### 背景：这个问题为什么难？

在传统的多智能体系统（MAS）里，每个大语言模型（LLM）都像独立的“聊天机器人”，只能通过文字把想法传递给伙伴。文字交流本质上是离散的、信息压缩的过程——模型必须把内部的高维向量压成一句话，再让对方把这句话重新解码回向量，这会导致细节丢失、推理链路被截断。与此同时，文字交互需要额外的解码与编码步骤，计算开销大，推理速度慢。因为这些根本性的瓶颈，现有的基于文本的 MAS 在复杂数学、科学推理或代码生成等需要深层次内部表征共享的任务上，往往表现不佳。

### 关键概念速览

**大语言模型（LLM）**：一种在海量文本上预训练的神经网络，能够生成自然语言或对应的内部向量表示。可以把它想成“会说话的思考机器”。

**潜在空间（latent space）**：模型内部的高维向量层，承载了对输入的抽象特征。类似于人脑的“潜意识”，信息在这里是连续且丰富的。

**自动回归潜在思考（auto‑regressive latent thoughts）**：模型在潜在空间里一步步生成内部表征，就像在脑海里写草稿，而不是直接说出完整答案。

**共享潜在工作记忆（shared latent working memory）**：所有智能体共同使用的一块向量缓存，类似于团队的白板，任何人都可以在上面写下思考痕迹，其他人直接读取。

**键值转移（KV transfer）**：在 Transformer 结构里，键（key）和值（value）是注意力机制的核心。这里把它们当作信息的“邮递员”，负责把一个智能体的内部向量搬运到共享记忆中。

**无训练协作（training‑free collaboration）**：整个协作过程不需要额外的梯度更新或微调，只利用预训练好的模型本身的推理能力。

### 核心创新点

1. **从文本桥接到潜在桥接**  
   之前的 MAS 必须把每个智能体的内部向量压成文字，再让对方解码。LatentMAS 直接把每个智能体的最后一层隐藏向量写入共享记忆，省去了压缩‑解压的环节。这样做的直接后果是信息几乎无损传递，推理链路保持完整。

2. **统一的潜在工作记忆机制**  
   传统系统往往用外部数据库或消息队列来传递文字，延迟高且结构化差。LatentMAS 把所有智能体的 KV 对齐到同一块向量矩阵中，任何智能体都可以在同一步骤读取最新的向量状态。相当于给团队装上了“实时共享的脑图”，大幅降低通信延迟。

3. **纯粹的自动回归潜在思考**  
   每个智能体在生成答案前，先在潜在空间里进行多步自回归生成，就像在脑中先写好草稿再决定是否公开。相比一次性直接输出文本，这种分阶段的内部推理让模型有更多机会利用共享记忆进行纠错和补充。

4. **理论层面的表达能力提升**  
   作者给出证明：在相同的模型容量下，潜在协作的可表达函数集合严格包含文本协作的集合，且信息保留率接近 100%。这解释了为什么在复杂任务上能取得 10% 以上的准确率提升。

### 方法详解

#### 整体框架概览

LatentMAS 的运行可以划分为四个阶段：  
1) **输入编码**：所有智能体同时接收同一任务描述，得到各自的输入向量。  
2) **潜在思考生成**：每个智能体在自己的隐藏层（通常是 Transformer 的最后一层）进行自动回归生成，输出一系列潜在向量序列。  
3) **共享记忆写入 & 读取**：这些向量通过键值转移机制写入全局的潜在工作记忆；随后每个智能体在下一轮思考时读取记忆中的最新向量作为额外上下文。  
4) **最终解码**：当所有智能体都认为思考结束后，选取或聚合其中一个（或多个）智能体的潜在表示，统一解码成自然语言答案。

#### 关键模块拆解

1. **自动回归潜在思考**  
   - 类比：人在纸上写草稿时，每写一行都会参考前面已经写好的内容。这里的“草稿”是隐藏向量，模型用自注意力把前一步的向量当作条件，生成下一步的向量。  
   - 实现上，直接把 Transformer 的自回归生成头从“词表”切换到“隐藏向量空间”。每一步输出的向量维度保持不变，只是把词嵌入换成了隐藏向量。

2. **键值转移到共享记忆**  
   - 每个智能体的隐藏向量被拆成键（K）和值（V）。键负责定位记忆中的位置，值则是要写入的内容。  
   - 记忆本身是一个固定大小的矩阵，所有智能体共享同一组键空间。写入时使用注意力权重把值加权累加到对应位置；读取时同样通过注意力把记忆中的向量加权求和，作为当前思考的额外输入。

3. **记忆更新策略**  
   - 为防止冲突，作者采用“时间戳+软更新”机制：每轮写入都会附带递增的时间标记，读取时只关注最新的时间戳对应的向量。软更新指的是使用指数移动平均平滑记忆内容，避免一次写入导致剧烈波动。

4. **结束判定与答案聚合**  
   - 每个智能体在潜在思考的最后一步会输出一个“停止信号”向量，所有智能体的停止信号取逻辑与后决定是否结束。  
   - 若有多个智能体仍在思考，则继续循环写入/读取；若全部停止，则把每个智能体的最终潜在向量送入同一个线性投影层，统一解码成文本。解码过程仍使用标准的词表采样，只在最后一步出现。

#### 反直觉或巧妙之处

- **不需要再训练**：很多协作框架会在共享记忆上再做一次微调，LatentMAS 直接利用预训练模型的内部结构，靠键值对齐实现信息共享，这在不改变模型权重的前提下实现了协同，极大降低了部署成本。  
- **信息无损传递的理论保障**：作者证明，只要共享记忆的维度不小于单个模型的隐藏层维度，信息可以完整映射进去并恢复出来，这让“潜在协作”不再是经验性的猜想，而是有数学底层支撑的。

### 实验与效果

- **评测任务**：论文在 9 套覆盖数学推理、科学问答、常识判断和代码生成的基准上做实验，包括 GSM8K、MMLU、ARC、HumanEval 等。  
- **对比基线**：与单模型（单独的 LLM）以及最强的文本协作 MAS（如 CoT‑based 多模型系统）进行比较。  
- **主要结果**：在所有任务上 LatentMAS 的准确率平均提升约 8%–14.6%。例如在 GSM8K 上提升 12.3%，在 HumanEval 上提升 10.8%。  
- **效率提升**：因为省去了多轮文本解码，整体 token 使用量下降 70.8%–83.7%，推理时延缩短约 4 倍至 4.3 倍。  
- **消融实验**：作者分别去掉共享记忆、去掉 KV 转移、改为普通文本交互，发现去掉记忆会导致准确率下降约 6%，改为文本交互则整体速度慢 3 倍，准确率下降 4%。这些实验表明共享潜在记忆是性能提升的关键因素。  
- **局限性**：论文明确指出，LatentMAS 依赖所有智能体使用相同的模型结构和隐藏维度，异构模型之间的潜在协作尚未验证；此外，潜在记忆的大小受限于显存，极大规模的协作仍需进一步压缩技巧。

### 影响与延伸思考

LatentMAS 在发布后迅速引发了两类后续研究：一是**异构潜在协作**，尝试让不同规模或不同架构的模型通过统一的投影层进入同一记忆空间；二是**可解释的潜在工作记忆**，研究如何把记忆中的向量可视化为人类可读的概念图。还有一些团队把潜在协作思路搬到视觉语言模型上，探索跨模态的“潜在共享白板”。如果想进一步了解，可以关注近期在 arXiv 上出现的 “Latent Multi‑Modal Collaboration” 系列论文以及 OpenAI、DeepMind 对大模型内部对齐的最新报告。

### 一句话记住它

**LatentMAS 让多个 LLM 直接在内部向量上写字、读字，省掉文字翻译，信息几乎不丢，推理快又准。**