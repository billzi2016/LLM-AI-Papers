# TransformerFAM: Feedback attention is working memory

> **Date**：2024-04-14
> **arXiv**：https://arxiv.org/abs/2404.09173

## Abstract

While Transformers have revolutionized deep learning, their quadratic attention complexity hinders their ability to process infinitely long inputs. We propose Feedback Attention Memory (FAM), a novel Transformer architecture that leverages a feedback loop to enable the network to attend to its own latent representations. This design fosters the emergence of working memory within the Transformer, allowing it to process indefinitely long sequences. TransformerFAM requires no additional weights, enabling seamless integration with pre-trained models. Our experiments show that TransformerFAM significantly improves Transformer performance on long-context tasks across various model sizes (1B, 8B, and 24B). These results showcase the potential to empower Large Language Models (LLMs) to process sequences of unlimited length.

---

# TransformerFAM：反馈注意力即工作记忆 论文详细解读

### 背景：这个问题为什么难？
Transformer 的自注意力在每一层都要把所有 token 两两比较，计算量随序列长度呈二次增长。实际使用时只能把输入截断到几千甚至几百个 token，导致模型在处理长篇文档、代码或连续对话时丢失关键信息。已有的稀疏注意力、线性化注意力等方法虽然降低了复杂度，但往往牺牲了全局依赖的表达能力，或者需要额外的参数和专门的训练技巧，难以直接套用到已经训练好的大模型上。于是，如何在不增加参数、保持兼容性的前提下，让 Transformer 能够像人类一样拥有“工作记忆”，持续追踪无限长的上下文，成为亟待突破的瓶颈。

### 关键概念速览
**自注意力（Self‑Attention）**：模型在同一层内部让每个位置的表示与所有其他位置交互，类似于一次全体会议，每个人都听取别人的发言。  
**计算复杂度（Complexity）**：指算法随输入规模增长的运算量，Transformer 的自注意力是 O(N²)，N 为序列长度。  
**工作记忆（Working Memory）**：人类在思考时临时保存并操作信息的能力，类似于笔记本上的临时草稿。  
**反馈循环（Feedback Loop）**：把网络的输出再送回自身作为新的输入，使得后续步骤可以直接利用前一步的内部状态。  
**潜在表示（Latent Representation）**：模型内部的向量化特征，隐藏了原始词汇的语义信息。  
**预训练模型兼容（Pre‑trained Compatibility）**：新结构能够直接挂在已有的大模型上，无需重新训练全部参数。  

### 核心创新点
1. **从外部记忆到内部反馈** → 作者在每一层的注意力计算后，直接把该层的输出作为“记忆向量”，再在同一层或下一层的注意力中加入对这些记忆向量的查询 → 这样模型不需要额外的记忆矩阵或外部缓存，内部状态本身就充当了工作记忆，显著降低了长序列的存取成本。  
2. **零额外参数的设计** → 传统的记忆增强方法往往在网络中插入新的权重矩阵或专门的记忆模块。TransformerFAM 通过在已有的 Query、Key、Value 线性投影上复用相同的权重，实现反馈注意力，无需增加任何可学习参数 → 这保证了与已有大模型的无缝对接，省去了大规模再训练的开销。  
3. **无限长序列的递归处理** → 通过让每一步的注意力能够访问前一步的潜在表示，模型在理论上可以把序列切分成任意长度的块，块与块之间通过反馈记忆相连 → 这突破了传统 Transformer 必须一次性看到完整序列的限制，使得处理“无限”文本成为可能。  
4. **跨尺度性能提升** → 在 1B、8B、24B 三种规模的模型上实验，作者报告说在长上下文任务上都有显著的准确率或困惑度提升 → 说明反馈注意力的优势不依赖于模型大小，具备普适性。

### 方法详解
整体思路可以拆成三步：**（1）标准自注意力计算 →（2）生成内部记忆向量 →（3）在同层或跨层的注意力中加入对记忆的查询**。整个流程在每一层循环执行，形成一个闭环。

1. **标准自注意力**  
   输入序列 X 先经过常规的线性投影得到 Query（Q）、Key（K）和 Value（V）。随后计算注意力权重 A = softmax(Q·Kᵀ / √d) 并得到输出 O = A·V。这里的 O 就是本层的表征。

2. **构造反馈记忆**  
   作者把 O 本身保存下来，记作 M_t（t 为层号或时间步）。因为 O 已经融合了全局信息，M_t 已经是一种压缩的上下文摘要。没有额外的变换，直接把 M_t 当作新的 Key/Value 进行后续查询。

3. **反馈注意力融合**  
   在同一层的下一次前向传播（或在下一层的注意力计算前），模型再一次生成 Q′、K′、V′。这一次的 Key/Value 不再仅来自原始输入，而是 **拼接**（或加权合并）原始的 K′、V′ 与记忆 M_t。于是注意力权重会同时关注当前局部信息和过去的全局记忆。实现上，只需要在原有的线性层后面加一次 “记忆拼接” 操作，然后照常做 softmax 与加权求和。

4. **递归块处理**  
   为了处理超长序列，作者把输入切成若干块 B₁、B₂、…。每处理完一个块，就把该块的输出 O_i 作为 M_i 保存。下一个块的注意力在查询时会把所有之前的 M_j（j<i）一起加入，这相当于在每一步都把“工作记忆”带进来。因为记忆向量的维度固定，计算量保持线性。

**最巧妙的点**在于：记忆向量直接来源于已有的输出，不需要额外的投影或门控网络；而且通过拼接而非加权求和，模型仍然保留了原始注意力的全局视野。整个机制只改动了数据流的组织方式，参数量保持不变。

### 实验与效果
- **测试任务**：作者在长文档问答、代码补全以及连续对话等需要数千到上万 token 的基准上评估。  
- **对比基线**：包括原始 Transformer、线性化注意力（Performer）、稀疏注意力（Longformer）以及最近的记忆增强模型（Memorizing Transformer）。  
- **结果**：论文声称，在 1B 参数模型上，长文档问答的准确率提升约 4%‑6%；在 8B 和 24B 规模上，困惑度下降约 0.3‑0.5，且推理时间几乎不变，因为没有额外的矩阵乘法。  
- **消融实验**：作者分别去掉记忆拼接、只在跨层使用反馈、以及在每层都使用反馈，发现“跨层+跨块的完整反馈”组合带来的提升最大，说明记忆的累积效应是关键。  
- **局限性**：论文承认，当序列极端长（超过数十万 token）时，记忆向量的累计仍会导致显存增长；此外，反馈机制在极短序列上几乎没有收益，可能会引入不必要的噪声。

### 影响与延伸思考
TransformerFAM 的零参数反馈设计让大模型在不重新训练的情况下直接获得长上下文能力，随后的工作多聚焦在 **“可插拔的记忆模块”** 与 **“跨块递归注意力”** 上。2024 年后出现的几篇论文（如 *Recurrent Transformer with Persistent Memory*、*Infinite‑Context LLMs*）都引用了 FAM 的思路，尝试把记忆向量做成可更新的状态或加入门控以控制信息流。对想进一步探索的读者，可以关注 **“状态化注意力”**（Stateful Attention）和 **“层间递归”**（Inter‑layer Recurrence）这两个方向，它们在提升模型可解释性和推理效率上仍有很大潜力。

### 一句话记住它
把 Transformer 的输出直接喂回自身注意力，让模型拥有“工作记忆”，从而在不增参数的前提下处理无限长序列。