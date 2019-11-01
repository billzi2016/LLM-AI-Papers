# Generalization through Memorization: Nearest Neighbor Language Models

> **Date**：2019-11-01
> **arXiv**：https://arxiv.org/abs/1911.00172

## Abstract

We introduce $k$NN-LMs, which extend a pre-trained neural language model (LM) by linearly interpolating it with a $k$-nearest neighbors ($k$NN) model. The nearest neighbors are computed according to distance in the pre-trained LM embedding space, and can be drawn from any text collection, including the original LM training data. Applying this augmentation to a strong Wikitext-103 LM, with neighbors drawn from the original training set, our $k$NN-LM achieves a new state-of-the-art perplexity of 15.79 - a 2.9 point improvement with no additional training. We also show that this approach has implications for efficiently scaling up to larger training sets and allows for effective domain adaptation, by simply varying the nearest neighbor datastore, again without further training. Qualitatively, the model is particularly helpful in predicting rare patterns, such as factual knowledge. Together, these results strongly suggest that learning similarity between sequences of text is easier than predicting the next word, and that nearest neighbor search is an effective approach for language modeling in the long tail.

---

# 通过记忆实现泛化：最近邻语言模型 论文详细解读

### 背景：这个问题为什么难？
语言模型要在海量文本上学会预测下一个词，但实际分布极度不均——常见词占大头，稀有事实、专有名词等“长尾”信息却很少出现。传统的神经语言模型（如Transformer）只能靠参数记忆这些稀有模式，往往需要更大的模型、更长的训练才能稍有提升，却仍会在罕见情形上出错。换句话说，模型在“记住”细节和“概括”规律之间存在瓶颈：直接预测下一个词的任务对稀有模式的学习成本极高。

### 关键概念速览
**语言模型（LM）**：给定前面的文字序列，输出下一个词的概率分布，就像打字时的自动补全。  
**嵌入空间（embedding space）**：模型把每个上下文映射成一个向量，向量之间的距离反映语义相似度，类似于把句子投射到一个多维坐标系里。  
**k‑最近邻（k‑NN）**：在向量空间里找离当前向量最近的k个已知向量，然后把它们的标签（这里是下一个词）投票或加权，像在图书馆里找最相似的几本书再参考它们的目录。  
**线性插值（linear interpolation）**：把两个概率分布按比例混合，例如 0.7×神经网络输出 + 0.3×k‑NN 预测，类似调配两种调味料得到更好的味道。  
**数据存储（datastore）**：预先把所有训练文本的上下文向量和对应的目标词保存下来，供检索使用，像把所有书的目录做成索引卡。  
**困惑度（perplexity）**：语言模型预测的好坏指标，数值越低表示模型越“懂”语言，类似于拼写检查的错误率。

### 核心创新点
1. **把神经语言模型和k‑NN检索直接混合 → 在每个预测时，先用预训练模型得到上下文向量，再在存储的向量库里找k个最近邻，取它们的目标词分布 → 通过线性插值得到最终概率。** 这一步把“记忆”外部化，让模型不必把所有稀有模式都压进参数里，而是直接查表。

2. **使用模型自身的嵌入空间作为相似度度量 → 计算向量距离时直接使用预训练LM的隐藏层输出，而不是额外训练相似度网络 → 检索过程既高效又保持语义一致性。** 这样避免了额外的相似度学习成本，直接利用已有的表示。

3. **无需再训练即可实现大规模数据扩展和领域适配 → 只要换掉或追加存储的向量库（比如加入特定领域的文本），模型即可在新领域表现更好 → 通过更换datastore实现快速迁移。** 这让模型在不同任务之间的切换成本几乎为零。

4. **在长尾事实预测上显著提升 → 通过检索到包含相同稀有实体或事实的上下文，模型能够更准确地给出对应词 → 证明“相似序列的记忆”比纯粹的参数预测更容易。** 这解释了为什么在预测罕见词时k‑NN的贡献尤为突出。

### 方法详解
整体思路可以拆成三步：**（1）预训练语言模型 →（2）构建向量存储 →（3）预测时混合检索结果**。

1. **预训练语言模型**  
   先训练一个标准的自回归语言模型（如Transformer），得到每个时间步的隐藏向量 h_t。这里的模型不需要任何改动，只是普通的语言模型。

2. **构建向量存储（datastore）**  
   - 对训练语料的每个位置 i，记录下对应的隐藏向量 h_i（通常取最后一层的输出）以及该位置的真实下一个词 w_{i+1}。  
   - 把 (h_i, w_{i+1}) 以键值对形式写入磁盘或内存索引结构（如FAISS），形成一个巨大的“查询表”。  
   - 这个表可以来自原始训练集，也可以加入额外的文本（比如专业文档），只要把对应的向量算出来并加入即可。

3. **预测时的混合**  
   - 给定上下文 x_{1:t}，模型先输出当前隐藏向量 h_t。  
   - 在datastore里用 h_t 进行近似最近邻搜索，找出 k 个最相似的键 h_{i1},…,h_{ik}。  
   - 对每个邻居取它们对应的目标词 w_{i_j+1}，并根据向量距离计算一个权重（距离越近权重越大），得到一个基于检索的词分布 p_{kNN}(·)。  
   - 同时，模型本身给出一个基于参数的词分布 p_{LM}(·)。  
   - 最终概率是两者的线性插值： p = λ·p_{LM} + (1‑λ)·p_{kNN}，λ 是超参数（常在0.2‑0.5之间调），相当于在“记忆”和“推理”之间调配比例。

**关键细节**  
- **距离度量**：使用欧氏距离或内积的负数，FAISS 能在数十亿向量上实现毫秒级检索。  
- **权重计算**：常用软最大（softmax）把距离转成概率，确保近邻的贡献占主导。  
- **k 的选取**：k 较小时检索更快但信息有限，k 较大时会引入噪声，实验中发现 64~128 是一个折中。  
- **插值系数 λ**：不需要重新训练模型，只在验证集上调一次即可，保持了“零训练”优势。  
- **最巧妙的点**：把检索过程放在模型的隐藏空间里，而不是原始词表或字符层面，这让相似度天然捕捉语法和语义信息，几乎不需要额外的特征工程。

### 实验与效果
- **数据集**：主要在 Wikitext‑103 上评估，这是一套约 100M 单词的英文维基百科语料，常用作语言模型基准。  
- **基线**：使用同等规模的强 Transformer LM（原始 perplexity 大约 18.7）。  
- **结果**：加入 k‑NN 检索后，整体 perplexity 降到 15.79，提升约 2.9 点，创下当时的最优记录。  
- **长尾表现**：在包含稀有实体或专业术语的句子上，k‑NN 的贡献尤为明显，模型能够直接检索到相同事实的上下文，显著降低错误率。  
- **消融实验**：作者分别关闭检索、改用随机向量库、或把 λ 设为 0，发现没有检索或检索库不相关时性能回落到基线，说明检索质量和插值比例是关键因素。  
- **局限**：检索需要额外的存储（每个训练 token 一个向量）和显存/磁盘 I/O，虽然可以用压缩或分块技术缓解，但在极大规模数据上仍会有成本；此外，检索速度受限于硬件，实时生成任务可能需要加速方案。

### 影响与延伸思考
这篇工作打开了“记忆+推理”混合的思路，随后出现了多篇基于检索增强的语言模型，如 REALM、RAG、RETRO 等，它们把外部文档检索与生成结合，进一步提升了开放域问答和代码生成等任务的表现。还有研究把向量库动态更新、使用多模态检索或把检索过程端到端微调，都是对 k‑NN‑LM 思路的延伸。想深入了解，可以关注以下方向：  
- **可扩展的向量检索系统**（FAISS、ScaNN）在超大规模语料上的优化；  
- **检索-生成端到端训练**，让模型学会在检索时产生更有用的查询向量；  
- **跨语言或跨模态检索**，把文本、图像、代码等多源信息统一到同一嵌入空间。  

### 一句话记住它
把语言模型的“记忆”外包给向量检索，只要在隐藏空间里找相似上下文并线性混合，就能用几行代码把 perplexity 降到新纪录。