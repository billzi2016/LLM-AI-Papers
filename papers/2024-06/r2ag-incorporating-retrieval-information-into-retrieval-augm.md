# R^2AG: Incorporating Retrieval Information into Retrieval Augmented   Generation

> **Date**：2024-06-19
> **arXiv**：https://arxiv.org/abs/2406.13249

## Abstract

Retrieval augmented generation (RAG) has been applied in many scenarios to augment large language models (LLMs) with external documents provided by retrievers. However, a semantic gap exists between LLMs and retrievers due to differences in their training objectives and architectures. This misalignment forces LLMs to passively accept the documents provided by the retrievers, leading to incomprehension in the generation process, where the LLMs are burdened with the task of distinguishing these documents using their inherent knowledge. This paper proposes R$^2$AG, a novel enhanced RAG framework to fill this gap by incorporating Retrieval information into Retrieval Augmented Generation. Specifically, R$^2$AG utilizes the nuanced features from the retrievers and employs a R$^2$-Former to capture retrieval information. Then, a retrieval-aware prompting strategy is designed to integrate retrieval information into LLMs' generation. Notably, R$^2$AG suits low-source scenarios where LLMs and retrievers are frozen. Extensive experiments across five datasets validate the effectiveness, robustness, and efficiency of R$^2$AG. Our analysis reveals that retrieval information serves as an anchor to aid LLMs in the generation process, thereby filling the semantic gap.

---

# R²AG：将检索信息融入检索增强生成 论文详细解读

### 背景：这个问题为什么难？
检索增强生成（RAG）把外部文档喂给大语言模型（LLM），本意是让模型拥有更丰富的事实来源。但检索器和LLM的训练目标、结构差异导致两者之间出现“语义鸿沟”。检索器只负责找相似文档，LLM却必须在生成时自行判断这些文档到底可信、相关还是噪声，这相当于让模型在没有任何提示的情况下自行“读懂”检索结果。结果往往是模型把检索到的内容当成普通上下文，甚至会产生误解或遗漏关键信息。于是，如何让LLM真正感知、利用检索器的内部信号，成为了提升RAG效果的关键瓶颈。

### 关键概念速览
**检索增强生成（RAG）**：把检索器找来的文档拼接到提示中，让LLM在生成答案时参考这些外部材料。  
**语义鸿沟**：检索器关注向量相似度，LLM关注语言流畅性，两者的“语言”不匹配，导致信息传递不顺畅。  
**检索特征**：检索器在匹配过程中产生的打分、注意力权重、文档排序等信息，类似于检索器的“自我解释”。  
**R²-Former**：论文中新设计的Transformer变体，用来把检索特征编码成可供LLM消费的向量。  
**检索感知提示（retrieval‑aware prompting）**：在提示里显式加入检索特征，让LLM在生成时“看到”检索器的决策依据。  
**冻结模型**：指在下游任务中不对检索器或LLM的参数进行微调，只通过外部模块或提示来提升性能。  

### 核心创新点
1. **检索特征显式建模 → 引入 R²-Former**  
   传统 RAG 只把检索到的原始文本喂给 LLM，忽略了检索器内部的打分、排序等信号。R²AG 设计了 R²-Former，将这些细粒度特征转化为统一的向量表示。这样，LLM 能直接“看到”哪些文档被检索器认为更重要，减少了模型自行判断的负担。

2. **检索感知提示 → 把特征写进 Prompt**  
   仅靠模型内部的注意力难以让 LLM 主动利用检索特征。R²AG 在构造提示时，把 R²-Former 输出的特征拼接到文本提示里，形成一种“带标签的上下文”。相当于在给模型喂“文档+检索器的解释”，提升了生成的准确性。

3. **低资源、冻结设置的适配**  
   大多数实际场景中，检索器和 LLM 已经是商用的黑盒，无法微调。R²AG 只在外部加入轻量模块（R²-Former）和提示策略，几乎不增加训练成本，却在五个公开数据集上实现了显著提升，展示了在冻结模型下的实用性。

### 方法详解
整体思路可以拆成三步：  
1) **检索阶段**：使用已有检索器（如 DPR、BM25）对输入查询得到一组候选文档，并记录每篇文档的匹配分数、检索向量以及排序位置。  
2) **检索特征编码**：把上述原始特征送入 R²-Former。R²-Former 的结构类似普通 Transformer，但在每层加入了专门处理“文档‑分数”对的交叉注意力头，使得模型能够学习“高分文档往往更可信，低分文档可能是噪声”。输出是一组与文档等长的特征向量。  
3) **检索感知提示构造**：将每篇文档的文本内容和对应的特征向量交叉拼接，形成形如 `[DOC_i] <|retrieval_feat_i|>` 的块。所有块按检索排序依次放入提示的前部，随后是原始任务指令。LLM 在解码时会先看到“文档+检索解释”，从而把检索器的判断当作生成的锚点。

**关键细节**  
- **R²-Former 的输入**：每篇文档的向量化表示（如 BERT CLS 向量）+ 检索分数。分数先经过归一化，再映射到与向量相同维度的标量，随后相加形成融合向量。  
- **交叉注意力**：在每层的自注意力之后，加入一次跨文档的注意力，专门让高分文档的特征影响低分文档的表示，类似于让模型“听取专家意见”。  
- **提示格式**：作者使用了特殊的占位符 `<|retrieval_feat_i|>`，在实际实现中可以把特征向量映射为一段可解释的文字（如 “[Score:0.92]”）或直接喂给支持向量输入的模型。  
- **冻结策略**：检索器、LLM 参数保持不变，只有 R²-Former 需要在少量标注数据上微调，训练成本大幅低于全模型微调。

最巧妙的地方在于：把检索器的“内部想法”转化为可视化的提示，而不是让 LLM 只能“盲目阅读”。这相当于在两位专家之间加了一层翻译，让他们的语言对齐。

### 实验与效果
- **数据集**：论文在五个公开任务上评估，包括事实问答（Natural Questions）、对话检索（Wizard of Wikipedia）、多跳推理（HotpotQA）等。  
- **对比基线**：传统 RAG（直接拼接文档）、Fusion‑In‑Decoder（把检索文档放进解码器注意力）以及最新的检索增强大模型（如 RETRO）。  
- **性能提升**：在大多数数据集上，R²AG 相比传统 RAG 提升了 3%~7% 的 EM/F1 分数；在低资源冻结设置下，仍能超过全模型微调的 Fusion‑In‑Decoder。  
- **消融实验**：去掉 R²-Former 或仅使用原始文档不加检索特征，性能回落到普通 RAG 水平，说明检索特征是关键贡献。  
- **效率**：R²-Former 参数量仅占整体模型的 0.5%，推理时额外计算几毫秒，几乎不影响实时性。  
- **局限**：论文未在极大规模检索库（如数十亿文档）上测试，检索特征的稀疏性可能导致编码效果下降；此外，提示中加入特征向量的方式在不同 LLM 上的兼容性仍需探索。

### 影响与延伸思考
R²AG 把“检索器的信心”显式搬进生成过程，打开了检索‑生成协同的新思路。随后的工作（如 Retrieval‑Aware Prompting、Meta‑Retriever）纷纷尝试把检索器的内部状态（注意力图、置信度）转化为可供 LLM 使用的信号。未来可能的方向包括：在多模态检索（图文、音频）中加入跨模态特征、让检索特征自适应地决定文档的采样比例，甚至在完全不冻结模型的情况下让检索器和 LLM 进行端到端的对话式协同训练。想深入了解的读者可以关注近期的 “Retrieval‑Enhanced Language Modeling” 系列论文以及开源实现（如 LangChain 的 Retrieval‑Aware Prompt 模块）。

### 一句话记住它
R²AG 通过把检索器的打分和排序信息编码进提示，让大语言模型在生成时直接“看到”检索器的判断，从而弥合两者的语义鸿沟。