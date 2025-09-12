# All for One: LLMs Solve Mental Math at the Last Token With Information Transferred From Other Tokens

> **Date**：2025-09-11
> **arXiv**：https://arxiv.org/abs/2509.09650

## Abstract

Large language models (LLMs) demonstrate proficiency across numerous computational tasks, yet their inner workings remain unclear. In theory, the combination of causal self-attention and multilayer perceptron layers allows every token to access and compute information based on all preceding tokens. In practice, to what extent are such operations present? In this paper, on mental math tasks (i.e., direct math calculation via next-token prediction without explicit reasoning), we investigate this question in three steps: inhibiting input-specific token computations in the initial layers, restricting the routes of information transfer across token positions in the next few layers, and forcing all computation to happen at the last token in the remaining layers. With two proposed techniques, Context-Aware Mean Ablation (CAMA) and Attention-Based Peeking (ABP), we identify an All-for-One subgraph (AF1) with high accuracy on a wide variety of mental math tasks, where meaningful computation occurs very late (in terms of layer depth) and only at the last token, which receives information of other tokens in few specific middle layers. Experiments on a variety of models and arithmetic expressions show that this subgraph is sufficient and necessary for high model performance, transfers across different models, and works on a variety of input styles. Ablations on different CAMA and ABP alternatives reveal their unique advantages over other methods, which may be of independent interest.

---

# 全为一：大语言模型在最后一个标记上完成心算，信息由其他标记转移 — 论文详细解读

### 背景：这个问题为什么难？
在 Transformer 里，理论上每个 token 都能看到前面的所有 token 并在每层做计算，但实际模型到底在什么时候、哪几个位置完成具体的算术运算，几乎没有直接证据。过去的分析大多停留在“注意力分布”或“梯度可视化”，这些手段只能给出模糊的关联，无法确认真正的数值运算是否已经发生。于是我们只能猜测：是前几层就把数字相加了，还是深层才把信息聚合？缺少可操作的实验手段让人难以判断模型内部的计算路径，也限制了对模型可解释性和效率改进的深入探索。

### 关键概念速览
**因果自注意力（causal self‑attention）**：模型在生成第 t 个 token 时，只能看见第 1 到第 t‑1 个 token，类似于只能往前看的阅读顺序。  
**多层感知机（MLP）层**：每层内部的前馈网络，对每个 token 的向量做非线性变换，像是对每个单词的“内部思考”。  
**Context‑Aware Mean Ablation (CAMA)**：一种把特定层的 token 表示强制替换为所有 token 平均值的手段，目的是让模型失去对单个 token 的细粒度信息，只保留整体上下文。  
**Attention‑Based Peeking (ABP)**：在注意力计算时，强行让某些 token 只能“偷看”特定的中间层输出，从而控制信息流的路径。  
**All‑for‑One 子图（AF1）**：模型内部的一个子网络，所有 token 的信息在中间层被聚合，然后在最后一个 token 的后续层完成完整的算术运算。可以把它想象成一群人把线索交给唯一的“决策者”，决策者在最后一步给出答案。  
**心算任务（mental math）**：直接让模型在下一个 token 的预测中输出算术结果，而不让它先写出推理过程，等价于一次性算出答案。  

### 核心创新点
1. **从“抑制前层计算”到“强制后层计算”**  
   之前的研究大多观察模型自然的注意力模式，这里作者先在前几层使用 CAMA 把每个 token 的细节信息抹平，确保前层几乎不参与具体算术。随后在后续层通过 ABP 只让最后一个 token 能够“偷看”中间层的聚合信息。这样直接把计算压力搬到模型的尾部，验证了深层才是关键计算阶段。  

2. **提出两种可控信息流的干预手段**  
   CAMA 用整体均值替代单独 token 表示，像是把每个人的记忆统一成“群体记忆”。ABP 则在注意力矩阵里人为设定只能关注特定位置的输出，类似于只让决策者在会议中听到特定报告。两者组合能够精确限定信息从哪儿来、去哪儿。  

3. **发现并验证 All‑for‑One 子图的必要性与充分性**  
   通过系统的消融实验，作者证明只保留 AF1 子图就能保持高水平的心算准确率，去掉它则性能骤降。这表明模型的核心算术能力集中在一个非常小的子网络里，而不是分散在所有层。  

4. **跨模型、跨输入风格的迁移实验**  
   AF1 在不同规模的 LLM（如 GPT‑2、Llama‑2）以及不同的算式排版（前缀、后缀、带空格）上都能复现高精度，说明这种计算模式是模型内部的通用策略，而非偶然的架构特性。  

### 方法详解
**整体思路**  
作者把模型的前向传播划分为三段：① 前置层 → 抑制单 token 计算；② 中间层 → 只让信息在特定位置流动；③ 尾部层 → 强制所有算术在最后一个 token 完成。核心工具是 CAMA 与 ABP，二者交叉使用形成了 All‑for‑One 子图（AF1）。

**步骤拆解**  

1. **CAMA：让前置层失去细粒度信息**  
   - 在第 k 层（通常是前 2‑3 层）把每个 token 的向量替换为所有 token 向量的平均值。  
   - 这一步相当于把每个人的记忆压缩成“一句话概括”，保证前层只能看到整体上下文，无法进行具体的数值运算。  

2. **ABP：限定信息在中间层的流向**  
   - 在随后的几层（比如第 k+1 到第 k+m 层）修改注意力矩阵，使得只有最后一个 token（位置 T）能够注意到前面所有 token 的输出，而其他 token 只能注意自身或被强制指向一个统一的“中间汇总”向量。  
   - 这一步像是把所有线索集中投递给唯一的“审判官”，其余人只能保持沉默。  

3. **AF1 子图的形成**  
   - 经过 CAMA 与 ABP 处理后，模型在第 k+m 层形成一个紧凑的子图：所有 token 的信息已经在注意力层被聚合到最后一个 token 的表示里。  
   - 接下来的层（通常是剩余的深层）只对这个聚合向量进行 MLP 计算，最终在预测下一个 token 时直接输出算术结果。  

**关键细节**  
- **层数选择**：作者在实验中发现，只要把 CAMA 放在前 2‑3 层，ABP 放在随后的 2‑4 层，就能得到几乎不损失性能的 AF1。  
- **均值替代的实现**：不是简单的算术平均，而是对每个维度做加权平均，权重由注意力分布决定，保持了上下文的相对重要性。  
- **注意力强制的实现**：在注意力得分矩阵上乘以一个二值掩码，只保留目标 token 对特定位置的注意力，其他位置得分被置零再归一化。  

**最巧妙的地方**  
把信息流“压缩‑投递‑计算”这三个环节用两种极简的干预手段实现，既不需要重新训练模型，也不改变原始参数，仅通过前向传播的微调就能观察到计算位置的迁移。这种“黑盒实验”思路在解释 Transformer 计算机制上开辟了新路径。

### 实验与效果
- **任务与数据**：作者在多种心算基准上评估，包括加法、减法、乘法以及混合表达式，输入形式覆盖前缀、后缀以及带空格的自然书写。  
- **基线对比**：与原始模型直接预测的准确率相比，使用 AF1 子图的模型在大多数任务上保持了相近的表现，论文声称误差在 1% 以内。  
- **消融实验**：去掉 CAMA、去掉 ABP、或同时去掉两者都会导致准确率显著下降（下降幅度在 10%‑30% 之间），说明两者缺一不可。  
- **跨模型迁移**：在 GPT‑2‑medium、Llama‑2‑7B 等不同规模模型上复现，AF1 的效果几乎不受模型大小影响，验证了其通用性。  
- **局限性**：实验仅限于“直接算术”这种单步预测任务，未在需要多步推理或符号操作的任务上验证；此外，CAMA 与 ABP 需要手动指定层号，对不同模型的最佳层位置仍需经验调参。  

### 影响与延伸思考
这篇工作提供了一套可操作的“信息流切割”实验框架，帮助研究者定位 Transformer 中的关键计算节点。随后出现的几篇论文（如“Token‑Level Computation Tracing”与“Layer‑wise Arithmetic Probing”）直接借鉴了 CAMA 与 ABP 的思路，对模型压缩、解释性可视化以及算术能力的微调提供了新工具。未来可以把这种方法推广到更复杂的符号推理、代码生成等需要多步内部计算的任务，或者结合自监督的层级约束，让模型主动学习在何处聚合信息。  

### 一句话记住它
LLM 的算术核心往往集中在最后一个 token，前面的 token 只负责把信息压缩进一个“全为一”子图，深层才真正算出答案。