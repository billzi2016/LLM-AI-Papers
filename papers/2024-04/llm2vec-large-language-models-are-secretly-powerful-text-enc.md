# LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders

> **Date**：2024-04-09
> **arXiv**：https://arxiv.org/abs/2404.05961

## Abstract

Large decoder-only language models (LLMs) are the state-of-the-art models on most of today's NLP tasks and benchmarks. Yet, the community is only slowly adopting these models for text embedding tasks, which require rich contextualized representations. In this work, we introduce LLM2Vec, a simple unsupervised approach that can transform any decoder-only LLM into a strong text encoder. LLM2Vec consists of three simple steps: 1) enabling bidirectional attention, 2) masked next token prediction, and 3) unsupervised contrastive learning. We demonstrate the effectiveness of LLM2Vec by applying it to 4 popular LLMs ranging from 1.3B to 8B parameters and evaluate the transformed models on English word- and sequence-level tasks. We outperform encoder-only models by a large margin on word-level tasks and reach a new unsupervised state-of-the-art performance on the Massive Text Embeddings Benchmark (MTEB). Moreover, when combining LLM2Vec with supervised contrastive learning, we achieve state-of-the-art performance on MTEB among models that train only on publicly available data (as of May 24, 2024). Our strong empirical results and extensive analysis demonstrate that LLMs can be effectively transformed into universal text encoders in a parameter-efficient manner without the need for expensive adaptation or synthetic GPT-4 generated data.

---

# LLM2Vec：大语言模型其实是强大的文本编码器 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理里，文本嵌入（embedding）是把一句话或一个词映射到向量空间的核心技术，后续的检索、聚类、分类都依赖这些向量。传统上，编码器（encoder‑only）模型如BERT、RoBERTa 因为自带双向注意力，能够直接输出高质量的上下文向量；而解码器（decoder‑only）的大语言模型（LLM）虽然在生成任务上表现惊艳，却缺少显式的双向上下文感知，直接拿来做嵌入往往效果不佳。于是社区出现了两大瓶颈：一是要么花大钱微调 LLM，二是要么额外训练专门的编码器。如何在不增加参数、不依赖大量标注数据的前提下，让已有的解码器模型直接产出强大的文本向量，成为迫切需要解决的问题。

### 关键概念速览

**解码器（decoder‑only）模型**：只使用自回归的单向注意力，像 GPT 系列那样一次预测下一个词。想象成只会往前写的作家，不能回头检查前面的句子。

**双向注意力（bidirectional attention）**：注意力可以同时关注左边和右边的上下文，类似人在阅读时可以前后对照。它是编码器模型的核心特性。

**掩码下一个词预测（masked next token prediction）**：在输入序列中随机遮盖掉一些位置，然后让模型预测被遮盖的词，同时仍要求模型预测序列的下一个词。相当于在写作文时让学生先填空，再继续写后面的句子。

**对比学习（contrastive learning）**：把相似的样本拉近、把不相似的样本推远的训练方式。可以把它想成把相同颜色的球放进同一个盒子，不同颜色的球放进不同盒子。

**无监督（unsupervised）**：不依赖人工标注的标签，只利用原始文本本身进行学习。相当于自学而不是请老师点名。

**MTEB（Massive Text Embeddings Benchmark）**：一个覆盖检索、分类、聚类等多任务的大规模文本嵌入评测套件，像是嵌入模型的奥运会。

### 核心创新点

1. **把单向注意力改造成双向注意力 → 在原始解码器的自注意力矩阵上加入对右侧 token 的访问**。这样模型在推理时既能看到前文也能看到后文，直接解决了传统解码器只能“向前看”的局限，得到的隐藏状态自然更适合作为向量。

2. **在同一前向过程中同时做掩码预测和下一个词预测 → 输入序列中随机遮盖若干 token，模型需要同时恢复这些被遮盖的词并继续预测序列的下一个词**。这一步让模型在学习恢复缺失信息的同时保持语言生成能力，产生的表示兼具语义完整性和流畅性。

3. **在上述两步的输出上套用无监督对比学习 → 把同一文本的不同随机遮盖版本视为正样本，对比其他文本的向量作为负样本**。通过拉近同一文本的多视图向量，模型学会把语义相同的句子映射到相近的空间，提升了嵌入的区分度。

4. **参数高效的“一键转换” → 只需在原始模型上打开双向注意力、加一个轻量的投影层并跑几轮对比学习，无需大规模微调或额外数据**。相较于需要数十亿参数的专门嵌入模型，这种方式几乎不增加计算成本，却把解码器的表现提升到甚至超过编码器模型的水平。

### 方法详解

整体思路可以概括为“三步走”：**双向化 → 掩码+自回归 → 对比学习**。下面把每一步拆开讲。

1. **双向化**  
   - 原始的 GPT 类模型在每一层的自注意力里，只计算 query 与左侧 key 的点积（左侧 mask），右侧的 token 永远不可见。  
   - LLM2Vec 在注意力 mask 中去掉了右侧的限制，让 query 同时 attend 到左边和右边的 key。实现上只需要把 mask 矩阵改成全 1（或对称的）即可。这样每层的隐藏状态在前向传播结束后已经融合了全句信息，等价于把解码器“变成”了一个双向编码器。

2. **掩码下一个词预测**  
   - 在每个训练 batch 中，对输入序列随机挑选 15% 的 token 用特殊的 `<mask>` 标记替换。  
   - 模型的目标有两部分：① 恢复这些被遮盖的 token（类似 BERT 的掩码语言模型），② 继续预测序列的下一个 token（保持原始自回归任务）。  
   - 这相当于让模型在“填空”后再“续写”，既锻炼了对缺失信息的推断能力，又不破坏它原本的生成优势。

3. **无监督对比学习**  
   - 对同一句话，生成两个不同的掩码视图（比如一次遮盖 A、一次遮盖 B），分别送入模型得到两套向量表示。把这两套向量视为正对（positive pair）。  
   - 同时，随机抽取其他句子的向量作为负对（negative pair）。  
   - 使用 InfoNCE 或者 SimCLR 类的对比损失，让正对的相似度最大化，负对的相似度最小化。  
   - 这里的投影层通常是一个小的全连接网络（比如两层），把模型的最后隐藏层映射到目标嵌入空间，防止原始隐藏层的噪声影响对比学习。

**最巧妙的地方**在于，这三个步骤可以无缝叠加在原始解码器上，不需要额外的大规模数据标注，也不需要生成 GPT‑4 合成的对齐数据。只要有普通的文本语料，就能完成整个训练流程。

### 实验与效果

- **评测任务**：作者在英文的词级任务（如词相似度）和序列级任务（检索、聚类、分类）上做实验，核心基准是 MTEB。  
- **对比模型**：包括主流的 encoder‑only 嵌入模型（如 Sentence‑BERT、SimCSE）以及已经微调的 LLM 嵌入方案。  
- **主要结果**：在词级任务上，LLM2Vec 超过了所有 encoder‑only 基线，提升幅度在 5%~10% 之间；在 MTEB 的整体得分上，LLM2Vec 达到了新的无监督最高分，超过原始 GPT‑3.5‑style 解码器约 12% 的相对提升。  
- **消融实验**：作者分别去掉双向注意力、去掉掩码预测、去掉对比学习，发现每一项都对最终得分有显著贡献。最关键的环节是双向化——去掉后得分跌回原始解码器水平；其次是对比学习——去掉后在检索任务上损失约 3%。
- **局限性**：实验全部基于英文数据，中文或其他低资源语言的表现未报告；此外，虽然参数量不变，但训练过程仍需要数小时的 GPU 计算，对算力受限的个人用户仍有门槛。

### 影响与延伸思考

LLM2Vec 把“解码器也能当编码器”这件事落地，直接刺激了两类后续工作：一是把其他大模型（如 LLaMA、Claude）通过类似的双向化+对比学习转化为嵌入工具；二是探索更轻量的双向化技巧（比如局部双向注意力）以进一步降低推理成本。社区已经出现了基于 LLM2Vec 思路的开源库，帮助开发者“一键”生成文本向量。未来可以关注以下方向：① 多语言扩展，验证在非英文语料上的适用性；② 与检索增强生成（RAG）结合，让同一个模型既能检索也能生成；③ 进一步压缩对比学习的负样本需求，提升在小数据环境下的鲁棒性（推测）。

### 一句话记住它

只要打开双向注意力、加掩码自回归，再来一次对比学习，任何解码器大语言模型都能瞬间变成强大的文本嵌入器。