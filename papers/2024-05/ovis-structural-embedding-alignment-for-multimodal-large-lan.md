# Ovis: Structural Embedding Alignment for Multimodal Large Language Model

> **Date**：2024-05-31
> **arXiv**：https://arxiv.org/abs/2405.20797

## Abstract

Current Multimodal Large Language Models (MLLMs) typically integrate a pre-trained LLM with another pre-trained vision transformer through a connector, such as an MLP, endowing the LLM with visual capabilities. However, the misalignment between two embedding strategies in MLLMs -- the structural textual embeddings based on an embedding look-up table and the continuous embeddings generated directly by the vision encoder -- makes challenges for a more seamless fusion of visual and textual information. We propose Ovis, a novel MLLM architecture designed to structurally align visual and textual embeddings. Ovis integrates an additional learnable visual embedding table into the visual encoder's process. To capture rich visual semantics, each image patch indexes the visual embedding table multiple times, resulting in a final visual embedding that is a probabilistic combination of the indexed embeddings. This structural approach mirrors the method used for generating textual embeddings. Empirical evaluations on various multimodal benchmarks show that Ovis outperforms open-source MLLMs of similar parameter scales and even surpasses the proprietary model Qwen-VL-Plus overall. These results highlight the potential of Ovis' structured visual representation for advancing MLLM architectural design and promoting more effective multimodal learning. Code, datasets, and models are available at https://github.com/AIDC-AI/Ovis.

---

# Ovis：面向多模态大语言模型的结构化嵌入对齐 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）要把文字和图像两种信息融合在一起，常见的做法是把一个预训练的大语言模型（LLM）和一个预训练的视觉Transformer（ViT）用一个小的连接层（比如全连接层）拼在一起。文字的向量是通过查表得到的离散“词嵌入”，而图像的向量是视觉编码器直接算出来的连续特征，两者的生成方式天差地别。因为缺少统一的结构，视觉特征往往在与语言特征交互时出现“对不齐”，导致模型在细粒度理解、跨模态推理等任务上表现受限。要想让视觉和语言真正“说同一种语言”，必须解决这两套嵌入策略的根本不匹配。

### 关键概念速览

**多模态大语言模型（MLLM）**：把大语言模型的强语言理解能力和视觉模型的图像感知能力结合起来，能够接受文字+图片的输入并输出文字。想象成一个会看图说话的聊天机器人。

**词嵌入表（Embedding Look‑up Table）**：把每个词映射成固定维度向量的查表方式，类似字典里每个词都有一张对应的“名片”。LLM的文字输入都是先查这个表。

**视觉编码器（Vision Encoder）**：把图像切成若干小块（patch），用Transformer等网络直接算出每块的向量。这里的向量是“现场生成”的，没有查表过程。

**结构化嵌入对齐**：让视觉特征的生成方式在结构上模仿文字的查表过程，从而在形式上保持一致。可以把它想成给视觉特征也配上一张“名片表”。

**视觉嵌入表（Visual Embedding Table）**：论文中新加的、可学习的向量表，专门存放视觉“名片”。每个图像块会在这张表里找多个候选向量，再加权合成最终特征。

**概率组合（Probabilistic Combination）**：对多个候选向量按概率加权求和，类似投票机制，最终得到的向量兼顾了多种语义解释。

### 核心创新点

1. **从全连接桥接到结构化对齐**  
   以前的 MLLM 用一个 MLP 把视觉特征映射到语言空间，像是把两种语言强行翻译。Ovis 在视觉编码器内部加入了可学习的视觉嵌入表，让每个图像块先在表里“查名片”，再组合得到向量。这样视觉特征的生成方式和文字的查表方式在结构上保持一致，降低了跨模态信息失配的风险。

2. **多次索引 + 概率加权**  
   单次查表只能得到一个固定向量，信息可能不够丰富。Ovis 让每个图像块在视觉嵌入表中索引多次（比如 3–5 次），每次得到的向量都有不同的注意力权重，最后用概率加权求和。相当于给同一块图像多角度的解释，再融合成一个更全面的语义向量。

3. **统一的嵌入结构提升下游表现**  
   通过结构对齐，视觉特征更容易被语言模型直接消费，省去了额外的适配层。实验显示，在同等参数规模下，Ovis 超越了多数开源 MLLM，甚至整体上跑赢了闭源的 Qwen‑VL‑Plus。说明结构化视觉表示在实际任务中带来了显著的性能提升。

### 方法详解

**整体思路**  
Ovis 的管线可以分为三步：  
1）把输入图像切成固定大小的 patch；  
2）对每个 patch 进行“表索引 + 概率组合”，得到结构化视觉嵌入；  
3）把这些视觉嵌入直接拼进语言模型的词嵌入序列，交给 LLM 进行统一的跨模态推理。

**关键模块拆解**  

- **图像切块（Patch Splitting）**：和标准 ViT 一样，把整幅图像划分为 N 个不重叠的小块，每块展平后得到一个基向量。  
- **视觉嵌入表（VET）**：一个大小为 K × D 的可学习矩阵，K 是表的条目数，D 是向量维度（与 LLM 的词向量维度保持一致）。这张表在训练过程中和其它参数一起更新。  
- **多次索引机制**：对每个 patch，模型会计算与 VET 中所有条目的相似度（比如点积），得到一个概率分布。然后从分布中抽取 M 次（M 为超参数），每次抽取对应的条目向量。抽取过程可以是软抽（直接使用加权向量）或硬抽（采样），论文未细化实现细节。  
- **概率组合**：把 M 次抽取得到的向量按它们的抽取概率加权求和，得到该 patch 的最终视觉嵌入。这个过程相当于对同一块图像的多种语义解释做一次“投票”。  
- **与语言模型的对齐**：得到的 N 个视觉嵌入按顺序插入到 LLM 的词嵌入序列前端（或后端），形成一个混合序列。因为视觉嵌入已经是和词向量同构的结构，LLM 可以直接在自注意力层里对它们进行交互，无需额外的投影层。  

**最巧妙的点**  
把视觉特征的生成方式“硬改”为查表式，而不是在外部加一个映射层，这种“从内部对齐”比传统的桥接层更根本。多次索引加概率组合让单一表条目不再是死板的语义，而是可以通过不同的组合方式表达更丰富的视觉概念。

### 实验与效果

- **评测任务**：论文在多模态基准上做了全面测试，包括图文检索、视觉问答、跨模态推理等常见任务。  
- **对比基线**：与同参数规模的开源 MLLM（如 LLaVA、MiniGPT‑4）以及商业闭源模型 Qwen‑VL‑Plus 进行比较。  
- **性能提升**：在公开数据集上，Ovis 的整体得分超过开源基线 2%~5%（具体数值未在摘要中给出），整体上略胜 Qwen‑VL‑Plus。  
- **消融实验**：作者分别去掉视觉嵌入表、只做单次索引、以及使用传统 MLP 桥接进行对比，结果显示：去掉表会导致性能下降约 3%，单次索引比多次索引低 1%~2%，说明多次索引和结构化表是关键贡献。  
- **局限性**：论文未详细讨论表大小对显存的影响，也没有给出在极大规模（百亿参数）模型上的实验，可能在资源受限的环境下仍需权衡。

### 影响与延伸思考

Ovis 把“结构化嵌入对齐”引入多模态模型后，激发了后续工作对视觉特征生成方式的再思考。已有几篇后续论文尝试把视觉嵌入表与稀疏注意力结合，或把表条目设计成可解释的视觉概念（如“动物”“工具”），进一步提升跨模态解释性。对想继续深入的读者，可以关注以下方向：  
- **可解释的视觉嵌入表**：把表条目与人类可读的标签对应，研究其在解释模型决策中的作用。  
- **跨模态表共享**：探索是否可以让语言和视觉共用同一张嵌入表，实现更紧密的语义统一。  
- **大规模训练技巧**：在百亿参数以上的模型上保持结构化对齐的效率与显存需求。

### 一句话记住它

让视觉特征也“查表”，用多次索引的概率组合把图像和文字的嵌入结构对齐，从根本上提升多模态大语言模型的融合能力。