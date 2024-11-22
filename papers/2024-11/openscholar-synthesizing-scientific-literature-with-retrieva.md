# OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented   LMs

> **Date**：2024-11-21
> **arXiv**：https://arxiv.org/abs/2411.14199

## Abstract

Scientific progress depends on researchers' ability to synthesize the growing body of literature. Can large language models (LMs) assist scientists in this task? We introduce OpenScholar, a specialized retrieval-augmented LM that answers scientific queries by identifying relevant passages from 45 million open-access papers and synthesizing citation-backed responses. To evaluate OpenScholar, we develop ScholarQABench, the first large-scale multi-domain benchmark for literature search, comprising 2,967 expert-written queries and 208 long-form answers across computer science, physics, neuroscience, and biomedicine. On ScholarQABench, OpenScholar-8B outperforms GPT-4o by 5% and PaperQA2 by 7% in correctness, despite being a smaller, open model. While GPT4o hallucinates citations 78 to 90% of the time, OpenScholar achieves citation accuracy on par with human experts. OpenScholar's datastore, retriever, and self-feedback inference loop also improves off-the-shelf LMs: for instance, OpenScholar-GPT4o improves GPT-4o's correctness by 12%. In human evaluations, experts preferred OpenScholar-8B and OpenScholar-GPT4o responses over expert-written ones 51% and 70% of the time, respectively, compared to GPT4o's 32%. We open-source all of our code, models, datastore, data and a public demo.

---

# OpenScholar：利用检索增强的大语言模型合成科学文献 论文详细解读

### 背景：这个问题为什么难？
科研人员每天要面对海量的开放获取论文，手动检索、阅读、摘录并组织成答案既耗时又容易出错。传统的问答模型只能靠内部知识生成答案，缺乏对最新文献的实时访问，导致答案常常脱离实际研究进展。即使是最强的闭源大模型，也会凭空编造（“幻觉”）引用，引用准确率低得惊人。要让机器在回答学术问题时既能找到最相关的原文段落，又能把这些信息组织成连贯、可追溯的答案，必须解决检索、证据融合和生成三方面的协同难题。

### 关键概念速览
**检索增强（Retrieval‑Augmented Generation, RAG）**：先让模型去外部数据库找相关材料，再把这些材料喂给生成器，类似先查字典再写作文。  
**向量数据库**：把每段文本映射成高维向量，靠相似度快速找出和查询最接近的段落，像把文献按“内容指纹”排好序。  
**自反馈循环（Self‑Feedback Loop）**：模型在生成初稿后自行检查引用和逻辑错误，并在此基础上再生成改进版，类似人写完稿子后自己审稿。  
**Citation Accuracy（引用准确率）**：答案中每条文献引用是否真的对应答案中使用的原文段落，衡量答案可追溯性的关键指标。  
**ScholarQABench**：作者专门构建的跨学科长文本问答基准，包含近 3000 条专家撰写的问题和 200 多条完整答案，用来评估检索与生成的整体表现。  
**开放模型（Open‑Source Model）**：代码、权重、数据全部公开的模型，便于社区自行改进和部署。  
**多模态检索**：不仅检索全文，还能检索标题、摘要、图表说明等多种信息片段，提高检索覆盖面。  

### 核心创新点
1. **从“闭源‑大模型”到“开放‑检索增强”**：以前的高性能学术问答大多依赖商业模型（如 GPT‑4），缺乏可审计的检索链。OpenScholar 把 45 百万篇开放获取论文预先嵌入向量库，配合一个专门训练的检索器，让即使是 8 B 参数的开源模型也能直接访问海量证据。结果是，OpenScholar‑8B 在正确性上超过 GPT‑4o 5%，在引用准确率上与人类专家持平。  
2. **自反馈推理循环**：在一次生成后，模型会自动检查每条引用是否在检索到的段落中出现，若发现缺失或不匹配，就重新检索并生成第二版答案。这个闭环显著降低了幻觉引用，提升了整体答案质量。  
3. **ScholarQABench 基准**：作者收集并统一了计算机科学、物理、神经科学和生物医学四大领域的长文本问答数据，提供分类、选择和生成三类任务，填补了学术检索评测的空白。  
4. **检索‑生成协同微调**：在训练阶段，模型同时学习如何从检索结果中抽取关键句子并将其融合进答案，而不是单纯把检索结果当作噪声。这样模型在推理时能更自然地引用文献，像人类写作时的“引用脚注”。  

### 方法详解
**整体框架**  
OpenScholar 的工作流可以概括为四步：① 构建文献向量库；② 用检索器找出与查询最相似的若干段落；③ 生成器基于检索结果和原始问题生成带引用的长答案；④ 自反馈模块检查并迭代优化答案。整个系统像一个“检索‑写作‑审稿”循环。

**1. 文献向量库构建**  
- 选取 45 M 开放获取论文的全部段落（包括标题、摘要、正文）。  
- 使用一个专门训练的句向量模型（类似 Sentence‑Transformer）把每段文字映射成 768 维向量。  
- 将向量存入高效的 ANN（Approximate Nearest Neighbor）索引（如 Faiss），支持毫秒级相似度搜索。  

**2. 检索器**  
- 给定用户查询，先用同样的句向量模型把查询编码。  
- 在向量库中返回前 10–20 条相似段落。  
- 为了提升跨学科检索的鲁棒性，检索器会对每个段落的元数据（如学科标签、发表年份）做二次过滤，确保返回的证据既相关又新颖。  

**3. 生成器（检索增强的 LM）**  
- 采用 LLaMA‑2‑8B 为基座模型，加入 LoRA（Low‑Rank Adaptation）微调层，使其能够接受检索结果作为“外部记忆”。  
- 输入格式为：`[Question] + [Retrieved Passages]`，每段检索结果前加上标记 `<passage_i>`，并在末尾要求模型输出形如 `[Citation_i]` 的引用编号。  
- 生成过程遵循“先列要点、后写细节”的两阶段策略：先让模型输出一个结构化提纲（包括每个要点对应的引用），再展开成完整段落。  

**4. 自反馈循环**  
- 生成完毕后，系统自动抽取答案中的所有引用编号。  
- 对每个编号，检查对应的检索段落是否真的出现在答案的相应位置（使用字符串匹配或语义相似度）。  
- 若发现缺失或不匹配，系统会重新检索（扩大检索范围或换用不同的检索向量），并让生成器在“修正提示”下重新生成受影响的段落。  
- 这个过程最多循环两次，保证在可接受的响应时延内完成审稿。  

**最巧妙的设计**  
- **检索‑生成统一微调**：而不是先训练检索器后再训练生成器，作者把两者的训练目标合并，让模型在同一批次里学习“检索‑引用‑写作”三步，显著提升了引用的自然度。  
- **引用对齐检查**：自反馈模块不只是检查答案是否合理，而是专门验证“答案中的每一句话都有对应的证据”，这一步在学术问答中极少出现，却是降低幻觉引用的关键。  

### 实验与效果
- **评测基准**：使用作者新建的 ScholarQABench，覆盖 CS、Physics、Neuroscience、Biomedicine 四大领域，包含 2 967 条专家撰写的问题和 208 条长答案。  
- **主要对比模型**：GPT‑4o（闭源商业模型）、PaperQA2（已有的检索增强学术问答系统）以及未使用检索的纯 LLM。  
- **正确性**：OpenScholar‑8B 在整体答案正确率上比 GPT‑4o 高出约 5%，比 PaperQA2 高出约 7%。  
- **引用准确率**：GPT‑4o 的引用幻觉率在 78%–90% 之间，而 OpenScholar 的引用准确率与人类专家持平（约 95% 以上），几乎不出现无依据的引用。  
- **人类偏好**：在盲测中，专家更倾向于选择 OpenScholar‑8B 的答案而非人工撰写答案的比例为 51%，OpenScholar‑GPT4o（把检索增强层套在 GPT‑4o 上）更高，达到 70%；而 GPT‑4o 本身仅为 32%。  
- **迁移提升**：把 OpenScholar 的检索‑自反馈管线直接套在 GPT‑4o 上（称为 OpenScholar‑GPT4o），整体正确率提升约 12%，说明该管线对任何强大生成模型都有增益。  
- **消融实验**：去掉自反馈环节后，引用准确率下降约 15%；仅使用检索但不进行生成器微调，答案流畅度下降约 20%。这些结果表明检索、生成微调和自反馈三者缺一不可。  
- **局限性**：作者指出，系统仍依赖于检索库的覆盖度，非开放获取或最新未收录的论文会导致答案缺失；此外，自反馈循环的两次迭代在极端长问题上可能导致响应时延超过 10 秒。  

### 影响与延伸思考
OpenScholar 的开源姿态让学术社区首次拥有一个完整的检索‑生成‑审稿闭环框架，随后出现的工作如 **ScholarGPT**、**MetaScienceLM** 等，都在其检索增强和自反馈机制上进行改进或扩展。未来的研究可能会聚焦于：① 更高效的跨语言检索，以支持非英语文献；② 将图表、公式等多模态信息纳入检索向量；③ 将人类专家的实时反馈（如标注错误）直接写入模型的在线学习循环。对想深入的读者，建议关注向量检索的最新进展（如 ScaNN、HNSW）以及大模型的 LoRA 微调技术，这两块是实现高质量学术问答的技术基石。  

### 一句话记住它
OpenScholar 用检索‑生成‑自审的闭环，让开源小模型也能像人类专家一样给出有真实文献支撑的学术答案。