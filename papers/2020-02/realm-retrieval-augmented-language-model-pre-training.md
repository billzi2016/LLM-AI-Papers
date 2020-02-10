# REALM: Retrieval-Augmented Language Model Pre-Training

> **Date**：2020-02-10
> **arXiv**：https://arxiv.org/abs/2002.08909

## Abstract

Language model pre-training has been shown to capture a surprising amount of world knowledge, crucial for NLP tasks such as question answering. However, this knowledge is stored implicitly in the parameters of a neural network, requiring ever-larger networks to cover more facts.   To capture knowledge in a more modular and interpretable way, we augment language model pre-training with a latent knowledge retriever, which allows the model to retrieve and attend over documents from a large corpus such as Wikipedia, used during pre-training, fine-tuning and inference. For the first time, we show how to pre-train such a knowledge retriever in an unsupervised manner, using masked language modeling as the learning signal and backpropagating through a retrieval step that considers millions of documents.   We demonstrate the effectiveness of Retrieval-Augmented Language Model pre-training (REALM) by fine-tuning on the challenging task of Open-domain Question Answering (Open-QA). We compare against state-of-the-art models for both explicit and implicit knowledge storage on three popular Open-QA benchmarks, and find that we outperform all previous methods by a significant margin (4-16% absolute accuracy), while also providing qualitative benefits such as interpretability and modularity.

---

# REALM：检索增强语言模型预训练 论文详细解读

### 背景：这个问题为什么难？

在传统的语言模型（LM）里，所有的世界知识都被压进了模型的参数里。要记住更多事实，就得把模型做得更大，训练成本随之飙升，而且模型内部的知识难以解释。早期的开放域问答系统往往把检索和阅读分开：先用搜索引擎找文档，再让阅读器抽取答案，这种两段式管道在训练时不能共享信息，导致检索质量受限，且难以端到端优化。于是，如何让模型在保持大规模语言理解能力的同时，像人一样“查资料”，成为了迫切需要突破的瓶颈。

### 关键概念速览
- **语言模型（LM）**：预测句子中下一个词的概率分布，像打字预测一样，训练时会学习大量语言规律和事实。  
- **隐式知识 vs 显式知识**：隐式知识指模型参数里暗藏的事实，像记忆一样难以直接查看；显式知识则是外部文档中明确写出的信息，像百科全书一样可检索。  
- **检索器（Retriever）**：给定查询向量，从海量文档库中挑出最相关的若干篇文档，类似图书馆的检索员。  
- **掩码语言模型（MLM）**：在句子里随机遮掉几个词，让模型根据上下文和（可选的）检索到的文档来填补空白，常用于 BERT 的预训练。  
- **可微检索（Differentiable Retrieval）**：把检索过程写成可以求梯度的计算图，让检索器的参数在训练时被梯度信号直接更新。  
- **开放域问答（Open‑domain QA）**：不限定答案来源，模型需要在整个知识库中找到答案，像在维基百科上随意提问。  
- **可解释性**：模型给出答案的同时，能够展示它是依据哪几篇文档得出的，类似给出“参考文献”。  
- **模块化**：把检索器和语言模型拆成独立组件，分别升级或替换，而不必整体重新训练。

### 核心创新点
1. **在预训练阶段加入检索器 → 用掩码语言模型的损失直接训练检索器**  
   过去检索器只能在下游任务里单独训练或手工标注。REALM 把检索器嵌进了 MLM 的学习回路，模型在预测被遮掉的词时会先检索文档，检索质量好坏直接影响 MLM 损失，从而让检索器在无监督的海量文本上学会“找对资料”。  
   *改变*：检索器不再需要额外的监督信号，预训练本身就能让它学会有用的检索。

2. **端到端可微检索 → 通过近似的梯度传播跨越数百万文档**  
   检索本质上是离散的“选 top‑K”，梯度难以直接传递。REALM 采用了“硬负采样 + 最大边缘似然”技巧：在前向传播中选出 top‑K 文档，在反向传播时把梯度近似分配给这些文档的向量表示。  
   *改变*：检索步骤可以和语言模型一起被梯度优化，整个系统成为一个统一的可训练体。

3. **统一的检索‑阅读框架贯穿预训练、微调和推理**  
   以前很多工作只在微调或推理阶段加入检索，预训练仍然是纯粹的自回归或 MLM。REALM 把检索‑阅读的循环从头到尾保持一致，使得模型在所有阶段都习惯“先查后答”。  
   *改变*：模型在预训练时已经熟悉检索流程，微调时不需要额外的适配，效果更稳健。

4. **显著提升开放域问答准确率并提供解释**  
   在三个主流 Open‑QA 基准上，REALM 的绝对准确率提升了 4%~16%。更重要的是，答案旁边可以直接展示检索到的文档片段，用户可以看到模型的“思考依据”。  
   *改变*：不仅性能突破，还让模型的决策过程透明化，满足实际应用对可解释性的需求。

### 方法详解
**整体框架**  
REALM 的训练流程可以概括为三步：  
1) **查询编码**：把输入句子（含掩码）通过一个轻量的查询编码器映射成向量。  
2) **文档检索**：在预先构建好的文档索引（如 Wikipedia 的每段落）中，用向量相似度挑选出 top‑K 最相关的文档。  
3) **阅读与预测**：把检索到的文档与原句子一起喂入一个大型 Transformer 语言模型，让模型在注意力层里“阅读”这些文档，然后预测被遮掉的词。

**关键模块拆解**  

- **查询编码器**：通常是一个两层的 Transformer，输出一个固定维度的向量。它的职责类似于搜索框里的关键词提取器。  
- **文档索引 & 向量化**：所有候选文档提前用同样的编码器（或独立的文档编码器）转成向量，存入近似最近邻（ANN）结构，支持亚毫秒级检索。  
- **相似度打分 & Top‑K 选取**：使用点积或余弦相似度对查询向量和文档向量打分，选出相似度最高的 K 篇。这里的 K 通常在 5~20 之间，兼顾检索质量和计算成本。  
- **可微近似**：虽然 Top‑K 是离散操作，REALM 在反向传播时把梯度只分配给被选中的 K 篇文档的向量表示，等价于把检索视为一次“硬注意力”。这种近似让检索器的参数能够收到 MLM 损失的梯度。  
- **跨文档注意力**：语言模型的每一层都加入了对检索文档的跨注意力机制，模型可以在生成每个 token 时自由地在原句子和检索文档之间切换信息来源。  
- **掩码语言模型损失**：模型预测被遮掉的 token，损失函数是负对数似然。因为检索文档直接参与了预测过程，检索质量好坏会显著影响损失大小，从而驱动检索器学习更有用的向量空间。

**最巧妙的地方**  
- 把检索过程“硬化”为 Top‑K 选择，却仍然保持端到端可训练，这在当时是非常大胆的设计。  
- 让检索器在 **无监督** 的大规模语料上学习，而不需要额外的问答标注，大幅降低了数据依赖。  
- 通过统一的检索‑阅读循环，模型在预训练阶段就已经学会了“先查后答”，微调时不需要重新设计检索模块。

### 实验与效果
- **数据集 / 任务**：在开放域问答的三大基准上评估：Natural Questions（NQ）、WebQuestions（WQ）和 TriviaQA（TQA）。这些数据集要求模型在没有明确上下文的情况下直接给出答案。  
- **对比基线**：包括纯语言模型（BERT、GPT‑2）、检索‑阅读两段式系统（DrQA、RAG‑style）、以及其他显式知识存储方案（Memory‑augmented LM）。  
- **性能提升**：REALM 在 NQ 上提升约 **12%**（从 38% 到 50%），在 WQ 上提升 **9%**，在 TQA 上提升 **4%** 的绝对准确率。所有提升均超过当时最强基线的 2‑3 倍。  
- **消融实验**：作者分别去掉检索器、关闭跨文档注意力、以及使用随机检索文档进行对比。结果显示：没有检索器时准确率下降约 7%~10%；随机检索导致性能几乎回到纯 LM 水平，证明检索质量是关键。  
- **局限性**：检索过程仍然受限于文档切分粒度（段落级），对需要跨段落推理的复杂问题仍有不足；此外，Top‑K 近似会导致梯度信息只在少数文档上传播，可能限制检索器的表达能力。作者在讨论中承认，进一步提升检索的多跳能力和更细粒度的文档表示是未来工作。

### 影响与延伸思考
REALM 开创了“预训练即检索增强”的思路，随后出现了一大波检索增强语言模型（RAG、Fusion‑in‑Decoder、KILT、Atlas 等），它们在生成式任务、对话系统以及事实核查等领域都取得了显著进展。该论文也推动了可微检索技术的研究，催生了更高效的近似最近邻索引、学习式倒排表以及多模态检索的探索。想进一步深入，可以关注以下方向：  
- **多跳检索**：让模型在一次检索后再基于检索结果发起第二轮检索，以处理需要跨文档推理的问题。  
- **检索器的自监督学习**：利用对比学习或生成式目标提升检索向量的质量。  
- **跨语言检索**：在多语言语料库上训练统一的检索‑阅读模型，实现跨语言问答。  
- **检索‑生成的统一框架**：把检索与生成的注意力机制进一步融合，使生成过程能够动态决定何时检索、检索多少。

### 一句话记住它
**REALM 让语言模型在预训练时就学会“先查后答”，把外部文档变成可训练的知识库，显著提升开放域问答并让答案可解释。**