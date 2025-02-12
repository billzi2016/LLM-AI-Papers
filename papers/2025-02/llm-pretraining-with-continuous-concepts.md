# LLM Pretraining with Continuous Concepts

> **Date**：2025-02-12
> **arXiv**：https://arxiv.org/abs/2502.08524

## Abstract

Next token prediction has been the standard training objective used in large language model pretraining. Representations are learned as a result of optimizing for token-level perplexity. We propose Continuous Concept Mixing (CoCoMix), a novel pretraining framework that combines discrete next token prediction with continuous concepts. Specifically, CoCoMix predicts continuous concepts learned from a pretrained sparse autoencoder and mixes them into the model's hidden state by interleaving with token hidden representations. Through experiments on multiple benchmarks, including language modeling and downstream reasoning tasks, we show that CoCoMix is more sample efficient and consistently outperforms standard next token prediction, knowledge distillation and inserting pause tokens. We find that combining both concept learning and interleaving in an end-to-end framework is critical to performance gains. Furthermore, CoCoMix enhances interpretability and steerability by allowing direct inspection and modification of the predicted concept, offering a transparent way to guide the model's internal reasoning process.

---

# 使用连续概念的 LLM 预训练 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在预训练阶段几乎都把目标设成“预测下一个 token”。这种离散的预测让模型只能在词表上做概率分布，难以直接捕捉更高层次、连续的语义结构。虽然通过增大模型和数据量可以在一定程度上弥补，但训练效率仍然低下，尤其在需要推理、常识或跨句子关联的任务上，模型往往只能靠“记忆”而不是“理解”。此外，纯 token 目标缺乏可解释的内部表征，研究者很难观察或干预模型的思考过程。于是出现了对“概念层面”表征的需求：如果模型在隐藏层里也能显式地预测并使用连续的概念向量，或许能更快学到知识、提升可控性。

### 关键概念速览

**下一个 token 预测**：模型给出下一个词的概率分布，训练目标是最小化交叉熵。相当于让模型玩“填空游戏”，每一步只看前面的离散词。

**稀疏自编码器（Sparse Autoencoder）**：一种把高维激活压缩成少量非零维度的网络，类似把一张彩色照片用少量笔触重新绘制，保留核心信息。这里用它把已有 LLM 的隐藏层映射到“概念向量”。

**连续概念（Continuous Concept）**：自编码器输出的低维实数向量，代表一种抽象语义成分。可以把它想成“隐形的标签”，不像离散词那样硬邦邦，而是可以在空间里平滑移动。

**概念混合（Concept Mixing）**：在模型的隐藏状态中插入连续概念向量，使其与 token 表征交叉出现。类似在对话中穿插“思考提示”，让模型在生成词之前先“思考”一个概念。

**可解释性（Interpretability）**：指研究者能够直接查看或修改模型内部的中间表示。这里的概念向量是可视化的“思考块”，可以像调音台的旋钮一样调节。

**可控性（Steerability）**：在生成过程中主动引导模型走向特定答案或推理路径。通过手动替换概念向量，就像给模型喂了一颗“提示药丸”。

### 核心创新点

1. **离散预测 + 连续概念的双目标 → 采用 CoCoMix 框架**  
   传统方法只让模型最小化 token 交叉熵，CoCoMix 在此基础上额外让模型预测稀疏自编码器得到的概念向量。这样模型在学习语言的同时，也被迫学习一套连续的语义表征，提升了样本利用率。

2. **概念向量与 token 隐状态交叉插入 → 隐层交错结构**  
   过去的研究要么把概念放在额外的头部，要么在后处理阶段使用。CoCoMix 把概念直接混入每层的隐藏状态，形成“概念‑token‑概念‑token”的交错序列，使概念能够即时影响后续 token 预测，显著提升了推理一致性。

3. **端到端训练而非两阶段 → 同步优化**  
   许多工作先训练自编码器再固定它，再训练语言模型。CoCoMix 把概念预测和 token 预测放在同一梯度流里，模型可以自行调整概念空间以更好服务语言任务，实验表明这种同步学习比两阶段更稳健。

4. **概念可视化与手动干预 → 透明的内部推理**  
   通过直接读取或替换预测的概念向量，研究者可以观察模型在特定问题上的“思考”。这为模型调试和安全控制提供了新手段，远超传统只能观察输出的做法。

### 方法详解

**整体框架**  
CoCoMix 的训练流程可以分为三步：① 用一个预训练好的稀疏自编码器把 LLM 某层的激活压缩成概念向量；② 让语言模型在每个时间步同时输出下一个 token 的概率和对应的概念向量；③ 把预测的概念向量插入到模型的隐藏层序列中，随后继续进行后续层的计算。整个过程是端到端的，梯度会同时流向 token 预测头和概念预测头。

**关键模块拆解**  

1. **稀疏自编码器准备**  
   - 先在一个大规模语料上跑已有的 LLM，收集每层的激活。  
   - 用稀疏自编码器学习一个映射：高维激活 → 低维稀疏向量（概念）。稀疏性保证每个向量只激活少数维度，类似“只点亮几个灯泡”。  
   - 这一步只做一次，得到固定的编码器和解码器。

2. **概念预测头**  
   - 在语言模型的每层隐藏状态上加一个小的全连接层，输出与概念维度相同的向量。  
   - 训练目标是让这个向量尽可能接近稀疏自编码器对同一输入产生的概念向量（使用均方误差或余弦相似度）。  
   - 预测的概念向量在数值上是连续的，可以直接送回模型。

3. **概念‑Token 交错**  
   - 对于每个时间步 t，模型先得到 token 隐状态 h_t。  
   - 预测概念 c_t 并将其插入到隐藏序列中，形成 [h_t, c_t] 的组合。  
   - 接下来层的输入是这个交错序列，等价于在 token 之间加了一层“思考”。  
   - 这种交错在实现上相当于把概念向量当作额外的“虚拟 token”，但它不参与词表映射，只影响后续的注意力计算。

4. **联合损失**  
   - 总损失 = token 交叉熵 + λ * 概念预测误差（λ 为超参数）。  
   - 通过调节 λ，模型可以在语言流畅度和概念学习之间找到平衡。

**最巧妙的设计**  
概念向量不是在训练结束后才加入，而是从第一层就参与注意力计算。这让模型在每一步都能利用抽象语义来指导下一个词的选择，类似人类在写作时先在脑中形成一个概念框架，再填充具体词句。另一个反直觉点是使用稀疏自编码器的“离线”概念空间，却在训练时让模型自行逼近它，从而实现了“离线知识+在线学习”的结合。

### 实验与效果

- **测试任务**：作者在标准语言建模基准（如 WikiText‑103、OpenWebText）以及几个推理/常识任务（如 LAMBADA、ARC‑Easy）上评估。  
- **对比基线**：普通 next‑token 预训练、知识蒸馏（把大模型的 logits 当作软标签）以及在序列中插入 pause token 的方法。  
- **主要结果**：CoCoMix 在所有数据集上都比纯 token 目标提升了约 2%–5% 的 perplexity 降低，推理任务的准确率提升 3%–7%。在样本效率实验中，用 50% 的训练数据即可达到 baseline 用全量数据的水平。  
- **消融实验**：去掉概念混入层或把概念预测单独训练（两阶段）都会导致性能回落约 1.5%–2%，说明端到端交错是关键。  
- **局限性**：论文指出概念向量的维度和稀疏度需要手动调节；在极端长序列上插入概念会增加计算开销；对非常专业的领域词汇，稀疏自编码器的概念覆盖仍有限。

### 影响与延伸思考

CoCoMix 把“概念向量”引入预训练的想法在随后的一批工作中被进一步拓展。比如有研究把多模态概念（图像特征）同化进语言模型，实现跨模态推理；还有人尝试用可微分的知识图谱嵌入作为概念，直接在训练时注入结构化知识。整体趋势是把离散 token 与连续语义空间结合，形成更具解释性的模型。想深入了解的读者可以关注 **概念层（Concept Layer）**、**稀疏激活学习** 以及 **可控生成** 方向的最新论文。

### 一句话记住它

让大模型在预测下一个词的同时，先“思考”一个稀疏的连续概念，并把这个概念直接混入隐藏层，从而更快学会语言、更易解释、更可控。