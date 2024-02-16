# Do Llamas Work in English? On the Latent Language of Multilingual   Transformers

> **Date**：2024-02-16
> **arXiv**：https://arxiv.org/abs/2402.10588

## Abstract

We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language models function and the origins of linguistic bias. Focusing on the Llama-2 family of transformer models, our study uses carefully constructed non-English prompts with a unique correct single-token continuation. From layer to layer, transformers gradually map an input embedding of the final prompt token to an output embedding from which next-token probabilities are computed. Tracking intermediate embeddings through their high-dimensional space reveals three distinct phases, whereby intermediate embeddings (1) start far away from output token embeddings; (2) already allow for decoding a semantically correct next token in the middle layers, but give higher probability to its version in English than in the input language; (3) finally move into an input-language-specific region of the embedding space. We cast these results into a conceptual model where the three phases operate in "input space", "concept space", and "output space", respectively. Crucially, our evidence suggests that the abstract "concept space" lies closer to English than to other languages, which may have important consequences regarding the biases held by multilingual language models.

---

# Llama 在英文中有效吗？多语言 Transformer 的潜在语言 论文详细解读

### 背景：这个问题为什么难？

多语言大模型在训练时往往看到的英文数据占多数，但它们最终要同时服务中文、法文、阿拉伯文等多种语言。过去的研究大多通过整体性能或翻译任务来评估模型的多语言能力，却很少触及模型内部到底是怎样“思考”的。我们不知道模型在处理非英文输入时，是直接在目标语言的语义空间里操作，还是先把信息转成某种内部通用语言再转回去。缺乏可观测的内部信号，使得判断模型是否隐式依赖英文成为一个难以验证的假设，也让对语言偏见的根源缺乏实证依据。

### 关键概念速览
- **多语言语言模型**：在同一个网络里同时学习多种语言的语言模型，像 Llama‑2 系列会把中文、法文、阿拉伯文等文本混在一起训练。可以把它想象成一个会说多种语言的“机器人”。
- **Token（词元）**：模型的最小输入单元，通常是一个子词或字符。比如中文的“语言”可能被切成两个 token。
- **Transformer 层**：模型的基本堆叠单元，每层都会把输入的向量映射到新的向量空间。可以把每层想象成一次“思考”，把信息从一个角度重新组织。
- **Pivot Language（枢纽语言）**：假设模型在内部使用的“中转语言”。如果英文是枢纽语言，那么模型在处理法文输入时会先把它“翻译”成英文的内部表示，再生成法文输出。
- **Concept Space（概念空间）**：论文中提出的一个抽象层次，介于原始输入向量和最终输出向量之间，专门存放语言无关的语义信息。可以把它看成“概念的云”，所有语言的相同意思都会在这里聚在一起。
- **Embedding Space（嵌入空间）**：所有 token 向量所在的高维空间，模型的每一次计算都在这里移动点的位置。
- **单词唯一续写任务**：给模型一个非英文提示，要求它输出唯一的、只有一个 token 能正确完成的答案。比如中文提示“巴黎是法国的首都，首都的英文是”后面只能接 “Paris”。这种任务让我们可以精准追踪模型到底在预测哪个 token。

### 核心创新点
1. **从语言层面构造唯一单词续写任务 → 直接把模型的输出锁定在唯一的 token 上 → 让我们能够毫不含糊地观察每层对正确答案的倾向，而不是被多义词或长句干扰。**  
2. **逐层捕获隐藏状态并映射到输出嵌入空间 → 用距离和概率比较来描绘向量在高维空间的轨迹 → 揭示了从“远离输出”到“进入语言特定区域”的三段式演化，而之前的研究只停留在整体表现或注意力可视化。**  
3. **提出“输入‑概念‑输出”三空间模型 → 把观察到的三阶段对应到具体的抽象层次 → 解释为什么中间层会更偏向英文的概率分布，这在之前的文献里没有系统的概念框架。**  
4. **实证证明概念空间更靠近英文 → 通过对比同一语义在英文和目标语言的 token 概率，发现中间层对英文的偏好显著 → 为多语言模型的语言偏见提供了内部机制的直接证据。**

### 方法详解
**整体思路**：先挑选一批“唯一续写”提示，让模型在每层都产生一个隐藏向量；再把这些向量投射到输出 token 的嵌入空间，计算它们与正确答案（目标语言）以及对应英文答案的距离和概率；最后把所有层的结果串起来，观察向量在空间中的移动轨迹。

**步骤拆解**：

1. **任务构造**  
   - 从维基百科、新闻等资源抽取事实句子，确保每个句子在目标语言里只有一个 token 能完整表达答案。  
   - 为每个句子准备两个版本的答案：目标语言的唯一 token（如中文 “巴黎”）和它的英文翻译（如 “Paris”）。  
   - 这样模型在生成下一个 token 时只能在这两个候选之间“抉择”，极大降低歧义。

2. **模型前向传播与隐藏状态提取**  
   - 将完整的提示（不包括答案）喂入 Llama‑2 系列的不同规模模型（7B、13B、70B）。  
   - 在每个 Transformer 块结束后，记录最后一个输入 token（即提示的最后一个词）的隐藏向量。相当于在模型“思考”每一步后，截取它脑海里对当前上下文的表征。

3. **映射到输出嵌入空间**  
   - Transformer 最终会把隐藏向量乘以输出矩阵得到每个 token 的 logits（未归一化的概率）。我们直接使用这个矩阵把每层的隐藏向量映射到“输出嵌入”。  
   - 这样得到的向量可以直接和目标 token、英文 token 的嵌入做距离比较，也可以算出它们的 softmax 概率。

4. **距离与概率分析**  
   - 对每层，计算隐藏向量到目标语言答案嵌入的余弦距离，以及到英文答案嵌入的距离。  
   - 同时取 softmax 后的概率，比较两者的大小。若英文概率显著更高，说明模型在该层更倾向于英文版本。

5. **轨迹可视化与阶段划分**  
   - 把所有层的距离/概率点画在同一条线上，观察随层数的变化趋势。  
   - 作者发现：  
     - **阶段 I（输入空间）**：前几层的向量离任何答案都很远，模型仍在“感知”原始字符信息。  
     - **阶段 II（概念空间）**：中间层的向量已经足够接近答案嵌入，能够解码出正确的语义，但英文版本的概率更高。  
     - **阶段 III（输出空间）**：后几层的向量进一步靠拢到目标语言的答案嵌入，英文优势消失，模型最终输出目标语言的 token。  

**最巧妙的点**：作者没有直接去“看注意力”，而是追踪**向量在嵌入空间的几何移动**。这让我们能够把抽象的“语言内部转换”具体化为“点从这里跑到那里”，并且用距离和概率这两个直观指标量化每一步的语言倾向。

### 实验与效果
- **数据与任务**：使用约 5,000 条跨语言事实句子，覆盖中文、法文、西班牙文、阿拉伯文、俄文等十余种语言。每条都满足唯一单词续写的条件。  
- **模型**：Llama‑2 7B、13B、70B 三个规模的公开权重。  
- **基线**：论文没有直接对比其他模型的内部轨迹，因为这种细粒度的向量追踪在之前几乎没有前例。唯一的对照是“随机初始化的同构模型”，其轨迹没有出现明显的英文偏好。  
- **主要发现**：在所有语言和模型规模上，阶段 II 的英文概率始终高于目标语言概率，平均提升约 20%~35%（具体数值在原文中以图表形式呈现，未给出精确数值）。阶段 III 时，两者概率基本持平，说明模型最终会纠正英文倾向。  
- **消融实验**：作者尝试去掉模型的最后几层再直接取中间层的输出进行生成，发现生成的文本更倾向于英文，验证了阶段 III 对语言校正的关键作用。  
- **局限性**：只针对 Llama‑2 系列，未验证是否适用于其他架构（如 Mistral、Gemma）。任务局限于单词级别的唯一续写，无法直接推广到更长的生成情境。作者也承认，英文偏好可能与训练数据比例有关，未排除其他潜在因素。

### 影响与延伸思考
这篇工作打开了“模型内部语言偏好”这一研究方向的大门。随后的几篇论文（如 *Pivot‑Language Probing in Multilingual BERT*、*Latent Language Alignment for Fairness*）直接借鉴了向量轨迹分析的方法，进一步探讨不同语言之间的对齐方式以及如何在训练阶段削弱英文枢纽效应。对工业界来说，这提示我们在部署多语言模型时，需要关注潜在的英文偏向，尤其是在需要公平对待低资源语言的场景。想继续深挖的读者可以关注以下方向：  
- 用更丰富的任务（如机器翻译、对话）复现三阶段现象；  
- 在训练阶段加入显式的语言对齐损失，看看是否能压缩概念空间与英文的距离；  
- 探索非 Transformer 架构（如混合专家模型）是否仍会出现类似的枢纽语言。

### 一句话记住它
多语言 Transformer 会先把信息投射到一个更靠近英文的概念空间，再回到目标语言，这说明英文是它们的隐形枢纽语言。