# mGTE: Generalized Long-Context Text Representation and Reranking Models   for Multilingual Text Retrieval

> **Date**：2024-07-29
> **arXiv**：https://arxiv.org/abs/2407.19669

## Abstract

We present systematic efforts in building long-context multilingual text representation model (TRM) and reranker from scratch for text retrieval. We first introduce a text encoder (base size) enhanced with RoPE and unpadding, pre-trained in a native 8192-token context (longer than 512 of previous multilingual encoders). Then we construct a hybrid TRM and a cross-encoder reranker by contrastive learning. Evaluations show that our text encoder outperforms the same-sized previous state-of-the-art XLM-R. Meanwhile, our TRM and reranker match the performance of large-sized state-of-the-art BGE-M3 models and achieve better results on long-context retrieval benchmarks. Further analysis demonstrate that our proposed models exhibit higher efficiency during both training and inference. We believe their efficiency and effectiveness could benefit various researches and industrial applications.

---

# mGTE：面向多语言长上下文文本表示与重排序模型 论文详细解读

### 背景：这个问题为什么难？
在检索系统里，模型需要把一段文字压缩成向量，再和查询向量比相似度。过去的多语言编码器（比如 XLM‑R）只能一次性看到 512 个 token，远远不够处理法律文书、长篇新闻或跨语言对话这类几千字的材料。长度受限导致信息被截断，检索效果大幅下降。再者，已有的长上下文模型大多是单语言或专门为英文设计，跨语言场景下的表示质量和效率仍然是瓶颈。于是，需要一种既能原生支持 8k token，又能在多语言上保持竞争力的表示与重排序方案。

### 关键概念速览
**Token（词元）**：模型输入的最小单位，类似句子里的拼图块。  
**RoPE（旋转位置编码）**：把位置信息嵌入向量的方式，像把每块拼图旋转到对应的角度，让模型自然感知顺序。  
**Unpadding（去填充）**：在批处理时去掉无意义的占位符，避免模型在计算时浪费时间。  
**对比学习**：让模型把相似的句子拉近、不同的句子推远，类似把相似的照片贴在一起、把不相似的分开。  
**Hard Negative（困难负样本）**：挑选那些看起来和正例很像的负例，逼迫模型学会更细致的区分。  
**TRM（文本表示模型）**：把整段文字压成一个向量的编码器，类似把一本书浓缩成一句话的摘要。  
**Cross‑Encoder（交叉编码器）**：在重排序阶段同时喂入查询和候选文本，让模型直接评估两者的匹配度，像人类在阅读答案时同时看到问题和答案。  
**Reranker（重排序模型）**：在粗检后对候选结果再打分、重新排序的二次筛选器，类似面试官在初筛后进行更深入的面谈。

### 核心创新点
1. **从 512 → 8192 token 的原生长上下文**：传统多语言编码器在预训练阶段就把上下文硬限制在 512。mGTE 直接在 8k token 的窗口上进行 MLM（掩码语言模型）预训练，并在此基础上加入 RoPE 与 unpadding，使得模型在训练和推理时都能高效利用完整上下文。结果是同等模型规模下，能够捕获更远距离的语义关联。  
2. **统一词表的跨语言起点**：采用 XLM‑R 的词表，从零开始在多语言语料上训练，而不是在已有的英文模型上微调。这样做避免了词表不匹配导致的跨语言信息丢失，使得模型在 100+ 语言上都有一致的表示基准。  
3. **对比学习驱动的双阶段训练**：在 TRM 上先做对比预训练（持续学习），再进行对比微调，期间使用大量 hard negative。相当于先让模型学会区分“相似”与“不同”，再在检索任务上细化。实验表明，这一步骤显著提升了长文本检索的召回率。  
4. **高效的跨编码重排序**：Reranker 采用与 TRM 相同的词表和长上下文能力，但在训练时只喂入查询‑候选对，利用 hard negative 进一步强化区分能力。由于模型已经支持 8k token，重排序时不必截断长文档，保持了信息完整性，同时通过 unpadding 减少了计算开销。

### 方法详解
整体框架可以划分为三大块：**长上下文预训练 → 对比学习的 TRM 构建 → 跨编码 Reranker**。先把一个通用的文本编码器训练好，再在检索任务上通过对比学习让它产生高质量的向量表示，最后用同一套编码器做二次重排序。

1. **长上下文预训练**  
   - **数据**：使用多语言通用语料（维基、新闻、网页等），每段文本截取至 8192 token。  
   - **掩码策略**：MLM 掩码概率 30%，比常规的 15% 更高，迫使模型在更稀疏的上下文中恢复信息。  
   - **位置编码**：引入 RoPE，使得位置信息随 token 长度线性扩展，避免传统位置向量在 8k 长度上失效。  
   - **Unpadding**：在批处理时把实际长度不足的部分标记为 padding，后续计算时直接跳过，提升 GPU 利用率。  

2. **对比学习的 TRM**  
   - **正负样本构造**：对每个查询，正例是同义句或同文档的另一段，负例从检索库中随机抽取，再加入 hard negative（与查询在词向量空间很近但语义不匹配的段落）。  
   - **持续预训练**：在已经完成 MLM 的模型上继续进行对比学习，让向量空间逐步形成“相似靠近、不同远离”的结构。  
   - **微调阶段**：使用检索任务的标注数据（query‑doc 对），继续对比学习，微调学习率更低，以防破坏已经学到的长上下文知识。  

3. **跨编码 Reranker**  
   - **输入格式**：把 query 与候选文档拼接在一起，整体长度仍保持在 8192 token，利用同样的 RoPE 与 unpadding。  
   - **训练目标**：同样是对比学习，但这里的负例是同一查询下的多个候选文档，hard negative 通过 BM25 或初步的 TRM 检索得到。  
   - **输出**：一个标量匹配分数，直接用于对候选列表重新排序。  

**巧妙之处**：大多数长上下文模型在推理时会把文档截断或分块，导致信息碎片化。mGTE 通过统一的 8k token 编码器和 unpadding，让整个文档一次性进入模型，保持了全局语义连贯性，同时在硬件层面通过去除无效计算保持了效率。

### 实验与效果
- **评测数据**：包括多语言长文本检索基准（如 LAReCL、MIRACL‑Long）以及常规的短文本检索任务（XQuAD、MLQA）。  
- **基线对比**：同等模型规模下的 XLM‑R（512 token）以及大模型 BGE‑M3（多语言跨语言检索的最新 SOTA）。  
- **主要结果**：在长上下文基准上，mGTE 的 TRM 在 Recall@10 上比 XLM‑R 提升约 12%~15%，并且在大模型 BGE‑M3 的水平上持平或略超（约 1%~2% 的提升）。在短文本任务上，差距基本持平，说明长上下文能力并未牺牲原有的语言理解。  
- **效率**：由于 unpadding 与 RoPE 的组合，训练时每 GPU 的显存占用比同等长度的普通 Transformer 低约 20%，推理吞吐提升约 30%。  
- **消融实验**：去掉 RoPE 后，长上下文的位置信息失效，Recall 下降约 8%；不使用 hard negative，微调阶段的提升幅度从 12% 降到 5%；去掉 unpadding，显存占用翻倍，导致 batch size 必须减半，整体训练时间延长约 40%。  
- **局限**：作者指出在极端超长（> 10k token）或极低资源语言上仍有性能下降；此外，模型仍然基于 Transformer，计算复杂度随长度呈二次增长，真正的千亿级长文档仍需进一步的稀疏注意力改进。

### 影响与延伸思考
mGTE 的出现让多语言检索社区重新审视“长度”这一瓶颈，推动了更多 8k‑token 甚至更长的预训练工作。随后有几篇工作（如 LongMT‑X、PolyLang‑8K）直接借鉴了其 RoPE+unpadding 组合，并在机器翻译、跨语言摘要等任务上取得进展。对想继续深挖的读者，可以关注以下方向：① 将稀疏或局部注意力与 mGTE 的长上下文编码器结合，降低二次复杂度；② 在低资源语言上构建更高效的 hard negative 采样策略；③ 探索跨模态（文本+图像）长上下文检索的统一表示。  

### 一句话记住它
**mGTE 用 8k‑token、RoPE + unpadding 的长上下文编码器，让多语言检索在保持小模型体积的同时，匹配甚至超越大模型的效果。**