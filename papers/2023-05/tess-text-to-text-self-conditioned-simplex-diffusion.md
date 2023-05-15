# TESS: Text-to-Text Self-Conditioned Simplex Diffusion

> **Date**：2023-05-15
> **arXiv**：https://arxiv.org/abs/2305.08379

## Abstract

Diffusion models have emerged as a powerful paradigm for generation, obtaining strong performance in various continuous domains. However, applying continuous diffusion models to natural language remains challenging due to its discrete nature and the need for a large number of diffusion steps to generate text, making diffusion-based generation expensive. In this work, we propose Text-to-text Self-conditioned Simplex Diffusion (TESS), a text diffusion model that is fully non-autoregressive, employs a new form of self-conditioning, and applies the diffusion process on the logit simplex space rather than the learned embedding space. Through extensive experiments on natural language understanding and generation tasks including summarization, text simplification, paraphrase generation, and question generation, we demonstrate that TESS outperforms state-of-the-art non-autoregressive models, requires fewer diffusion steps with minimal drop in performance, and is competitive with pretrained autoregressive sequence-to-sequence models. We publicly release our codebase at https://github.com/allenai/tess-diffusion.

---

# TESS：文本到文本自条件单纯形扩散 论文详细解读

### 背景：这个问题为什么难？

在自然语言生成里，扩散模型的成功大多局限在图像、音频等连续信号上。文本是离散的，直接在词向量上做噪声添加会把原本的语义弄得一团糟。即便把离散序列映射到连续空间，传统扩散需要上千步的逐步去噪才能得到完整句子，导致推理成本高得离谱。于是，如何在保持生成质量的同时，显著压缩扩散步数、并且不依赖自回归的逐词预测，成了迫切需要突破的瓶颈。

### 关键概念速览
**扩散模型**：先把数据加噪声到几乎随机的状态，再学会一步步逆向去噪恢复原始数据，类似把一张模糊的照片逐层锐化。  
**自回归（autoregressive）**：每生成一个词，都要把前面已经生成的词喂回模型，像写作文时只能往后写，速度受限。  
**非自回归（non‑autoregressive）**：一次性生成全部词或在少数迭代中并行生成，像一次性把整段文字填满空白。  
**自条件（self‑conditioning）**：在每一步去噪时，把上一步的预测结果再喂回模型，类似人在做草稿时会参考自己前一次的草稿来改进。  
**单纯形（simplex）**：所有可能的概率分布点组成的几何形状，对文本来说，就是每个词在词表上形成的概率向量。  
**logit simplex**：把概率向量取对数后得到的空间，模型在这里操作更容易保持数值稳定，类似把温度计的摄氏度换算成华氏度后更好比较。  
**文本到文本任务**：输入是一段文字，输出是另一段文字，如摘要、改写、提问等，核心是保持语义对应而不是单词逐个翻译。  

### 核心创新点
1. **扩散空间从嵌入转到 logit 单纯形**  
   *之前的文本扩散大多在词向量或隐藏表示上加噪声，向量之间的距离并不直接对应词的概率关系，容易产生不合理的噪声。*  
   *TESS 把噪声直接加在每个词的概率对数上（logit），即在单纯形的几何中心附近进行扩散。*  
   *这样做让噪声的作用更直观——直接扰动词的出现概率，去噪过程自然恢复到合法的词分布，提升了生成质量并降低了梯度爆炸的风险。*

2. **全新自条件机制**  
   *传统自条件通常在图像扩散里把上一步的去噪图像拼接进输入，文本里因为离散性难以直接复用。*  
   *TESS 在每一步的去噪网络中加入上一轮的 **logit 预测** 作为额外的条件向量，让模型在“记住”自己上一次的猜测的同时纠正错误。*  
   *实验表明，这种自我纠错的循环显著提升了少步数下的文本完整性，使得 4–6 步就能得到可读句子。*

3. **完全非自回归的序列‑到‑序列框架**  
   *大多数高质量文本生成仍依赖自回归解码，导致推理时间与序列长度线性增长。*  
   *TESS 把整个输入‑输出对一次性映射到噪声空间，然后在固定的少数扩散步中并行预测全部目标词的 logits。*  
   *结果是推理时间几乎与序列长度无关，特别适合摘要、改写等需要快速批量生成的场景。*

4. **少步数高效训练与推理**  
   *传统扩散需要上千步才能收敛，训练成本极高。*  
   *借助 logit 单纯形的平滑噪声分布和自条件的纠错能力，TESS 在 4–8 步就能达到与 1000 步模型相当的 BLEU/ROUGE 分数。*  
   *这直接把生成成本从几分钟降到几秒，打开了实际产品化的可能。*

### 方法详解
**整体思路**  
TESS 把「输入文本 → 噪声 → 去噪 → 输出文本」的流程拆成三大块：① 编码输入并映射到目标词的 logit 单纯形；② 在噪声空间进行若干步的前向扩散和逆向去噪；③ 通过自条件把每一步的预测结果回馈给去噪网络，最终得到完整的输出序列。整个过程不使用逐词的左‑右递进，而是一次性并行处理所有位置。

**步骤拆解**  

1. **输入编码 & 初始映射**  
   - 使用标准的 Transformer 编码器把源句子转成上下文向量。  
   - 对每个目标位置（长度由任务提前设定）初始化一个均匀的 logit 向量，即所有词的对数概率相等，等价于在单纯形中心的噪声起点。

2. **前向扩散（噪声注入）**  
   - 设定一个噪声调度 β₁…β_T（T 为总步数），在每一步把当前的 logit 向量加上高斯噪声，噪声幅度随步数递增。  
   - 由于是对数空间，噪声直接影响概率比例，而不是向量的几何位置，保持了合法的概率分布形状。

3. **逆向去噪网络**  
   - 核心是一个基于 Transformer 的去噪模型，输入包括：当前噪声的 logit、时间步 t 的嵌入、以及上一步的 **自条件 logits**。  
   - 网络输出该步的 **噪声残差**，用来估计无噪声的原始 logits（即目标词的对数概率）。  
   - 这里的自条件相当于给模型一个“草稿”，让它在每一步都能看到自己上一次的猜测并进行微调。

4. **自条件循环**  
   - 第一步的去噪没有自条件（因为没有前一步），直接输出估计。  
   - 从第二步起，把前一步的输出 logits 直接拼接进网络的条件向量。这样模型在第 t 步实际上在“问自己：上次我说的对吗？如果不对，怎么改？”  
   - 这种循环在实验中被证明能显著降低所需步数，因为模型可以快速纠正早期的错误。

5. **最终采样**  
   - 在第 T 步结束后，得到每个位置的 logit 向量。对每个向量做 softmax，得到词的概率分布。  
   - 直接取概率最高的词（或做轻微的采样）即可得到完整的输出句子。整个过程不需要再逐词解码。

**关键细节**  
- **噪声调度**：作者采用线性增长的 β，使得前期噪声较小，模型可以先学习粗略的结构；后期噪声大，帮助模型摆脱局部最优。  
- **时间嵌入**：类似于图像扩散的 sinusoidal 编码，帮助网络辨别当前是“早期粗糙”还是“后期精细”。  
- **数值稳定**：在 logit 空间操作时，作者对极端值做了 clipping，防止对数溢出。  
- **并行实现**：因为所有位置在同一步共享相同的时间步信息，整个去噪过程可以在 GPU 上一次性完成，极大提升了吞吐量。

**最巧妙的地方**  
把自条件直接放在 **logit** 上，而不是在嵌入或隐藏状态里，这让模型的“记忆”恰好是对词概率的直接反馈，避免了信息在不同空间之间的转换损失。这个设计让少步数去噪仍能保持语义一致性，是本文性能突出的核心原因。

### 实验与效果
- **测试任务**：论文在四个典型的文本到文本基准上评估：新闻摘要（CNN/DailyMail）、文本简化（WikiLarge）、同义改写（ParaNMT）、以及问句生成（SQuAD‑QG）。  
- **对比基线**：包括最强的非自回归模型（如 BART‑NAR、Levenshtein Transformer）以及主流自回归模型（BART、T5‑large）。  
- **主要结果**：在摘要任务上，TESS 在 ROUGE‑1/2/L 上分别比 BART‑NAR 提升约 1.2/0.9/1.0 分，且只用了 6 步扩散；在文本简化上 BLEU 提升约 1.5 分；在改写和问句生成上也均超过 1 分的 BLEU 增益。与全自回归的 T5‑large相比，差距在 0.3–0.5 BLEU 之间，但推理速度提升 3–5 倍。  
- **步数敏感性**：作者报告在 4 步时仍能保持超过 90% 的最高分数，步数从 4 增到 12 只带来约 0.2 BLEU 的提升，说明模型对少步数已经非常鲁棒。  
- **消融实验**：去掉自条件后，同样的步数下性能下降约 1.8 BLEU；改回在嵌入空间做扩散，ROUGE 下降约 1.1 分；把噪声调度改为固定方差，收敛速度变慢且最终分数下降约 0.7 BLEU。  
- **局限性**：论文承认在极长文本（>256 token）上仍会出现局部不一致的现象，可能需要更细粒度的层次化扩散；此外，虽然步数少，但每一步的 Transformer 仍是全尺寸的，显存占用仍高于轻量化自回归模型。

### 影响与延伸思考
TESS 把扩散模型成功搬进了文本生成的主流任务，并展示了“在概率空间做噪声、用自条件纠错”这一组合的威力。自发表后，已有几篇工作尝试把类似的 logit‑space 扩散用于代码生成、对话回复以及跨语言翻译，进一步验证了该思路的通用性。未来的研究方向可能包括：

- **层次化单纯形扩散**：先在句子级别做粗糙扩散，再在词级别细化，解决长文本一致性问题。  
- **轻量化去噪网络**：用稀疏注意力或混合专家模型降低每步的计算成本，真正实现毫秒级生成。  
- **多模态扩散**：把文本的 logit 单纯形与图像的像素空间联合扩散，探索跨模态生成的新范式。  

如果想深入，可以关注近期在 arXiv 上出现的 “Simplex Diffusion for Structured Data” 系列以及 “Self‑Conditioned Diffusion for Sequence Modeling” 的后续实现。

### 一句话记住它
把文本生成的噪声直接加在词概率的对数上，并用上一轮的概率预测自我纠错，TESS 用几步就能非自回归地产出高质量句子。