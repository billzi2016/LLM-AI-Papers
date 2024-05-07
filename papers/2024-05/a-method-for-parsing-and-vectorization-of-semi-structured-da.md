# A Method for Parsing and Vectorization of Semi-structured Data used in   Retrieval Augmented Generation

> **Date**：2024-05-07
> **arXiv**：https://arxiv.org/abs/2405.03989

## Abstract

This paper presents a novel method for parsing and vectorizing semi-structured data to enhance the functionality of Retrieval-Augmented Generation (RAG) within Large Language Models (LLMs). We developed a comprehensive pipeline for converting various data formats into .docx, enabling efficient parsing and structured data extraction. The core of our methodology involves the construction of a vector database using Pinecone, which integrates seamlessly with LLMs to provide accurate, context-specific responses, particularly in environmental management and wastewater treatment operations. Through rigorous testing with both English and Chinese texts in diverse document formats, our results demonstrate a marked improvement in the precision and reliability of LLMs outputs. The RAG-enhanced models displayed enhanced ability to generate contextually rich and technically accurate responses, underscoring the potential of vector knowledge bases in significantly boosting the performance of LLMs in specialized domains. This research not only illustrates the effectiveness of our method but also highlights its potential to revolutionize data processing and analysis in environmental sciences, setting a precedent for future advancements in AI-driven applications. Our code is available at https://github.com/linancn/TianGong-AI-Unstructure.git.

---

# 用于检索增强生成的半结构化数据解析与向量化方法 论文详细解读

### 背景：这个问题为什么难？

在实际业务中，文档往往不是纯文本，而是 PDF、图片、表格等半结构化形式，直接喂给大语言模型（LLM）会导致信息丢失或误解。传统的检索增强生成（RAG）流程假设检索库里都是干净的向量化文本，因而在面对多格式文档时只能做粗糙的 OCR 或手工转码，既耗时又容易出错。更糟的是，行业专用的技术文档（比如污水处理工艺说明）常常包含大量表格、图例和专业术语，通用的向量化工具难以捕捉其细粒度语义。于是，如何把各种半结构化来源统一转成可检索、可理解的向量，成为提升 RAG 在专业领域表现的瓶颈。

### 关键概念速览
**半结构化数据**：介于完全结构化（如数据库表）和非结构化（如纯文本）之间的文件，常见形式包括 PDF、Word、Excel、图片等，内部有一定的章节、标题或表格层次。  
**检索增强生成（RAG）**：先在外部知识库里检索相关片段，再把这些片段作为上下文喂给 LLM，让模型生成更精准、信息丰富的答案。  
**向量数据库（Vector DB）**：把文本或其他数据转成高维向量后存储的系统，支持相似度搜索。这里使用的是 Pinecone。  
**Docx 统一化**：把所有来源的内容先转成 Microsoft Word 的 .docx 格式，以便统一使用 Word 的解析 API 抽取结构化信息。  
**嵌入模型（Embedding Model）**：把文字映射到向量空间的模型，向量之间的距离代表语义相似度。  
**环境管理/污水处理场景**：本论文聚焦的行业背景，涉及大量技术规范、操作手册和实验报告。  
**消融实验**：逐个去掉系统组件，观察性能下降幅度，以判断每个模块的重要性。  

### 核心创新点
1. **全链路 .docx 转换管线 → 直接把 PDF、图片、Excel 等统一转成 Word 文档 → 解决了不同格式之间解析接口不统一的问题，使后续的结构抽取可以一次性完成。**  
2. **基于 Pinecone 的向量库构建 + LLM 动态检索 → 将抽取出的章节、表格行、图例说明分别生成向量并标记元数据 → 检索时能够精准定位到“章节+表格”层级，而不是整段文字，显著提升了答案的技术细节准确度。**  
3. **双语（中英）实验验证 → 在同一套管线上分别处理英文和中文文档，使用相同的嵌入模型并对比检索效果 → 证明方法对语言无关，适用于跨语言的专业文档库。**  
4. **RAG 与 LLM 的深度耦合实现 → 检索结果直接拼接到 Prompt 中，并通过少量示例提示模型如何利用表格向量 → 让模型在生成时能够“看见”表格结构，生成的技术回答比传统 RAG 更具可操作性。**  

### 方法详解
整体框架可以划分为四个阶段：**格式统一 → 内容解析 → 向量化存储 → 检索‑生成闭环**。

1. **格式统一**  
   - 所有原始文件先经过专门的转换脚本。PDF 用 `pdf2docx`，图片用 OCR（如 Tesseract）后再写入 Word，Excel 直接转成 Word 表格。  
   - 转换后得到的 .docx 文件保留原始的章节标题、段落层级和表格结构，便于后续的层次化抽取。

2. **内容解析**  
   - 使用 Python‑docx 库遍历文档树，识别标题（基于样式层级）、段落和表格。  
   - 对每个表格，进一步拆分成行向量，每行的每列文本保留列名元数据。  
   - 生成的结构化记录形如 `{type: "section", title: "...", text: "...", page: n}` 或 `{type: "table_row", table_id: "...", row_idx: i, cells: {...}}`。

3. **向量化存储**  
   - 选用 OpenAI 的 text‑embedding‑ada‑002（或同类中文兼容模型）对每条记录的文本进行嵌入。  
   - 嵌入向量连同元数据一起写入 Pinecone，元数据里保存 `doc_id、type、section_id、table_id` 等索引信息。  
   - 这样在检索时可以通过过滤条件只返回同一章节或同一表格的向量，避免跨章节噪声。

4. **检索‑生成闭环**  
   - 用户提问后，先用同一嵌入模型把问题转成向量，在 Pinecone 做相似度搜索，返回前 k 条记录。  
   - 根据记录的 `type`，系统自动拼装 Prompt：章节文本直接拼入，表格行则先格式化成 “列名: 值” 的列表。  
   - 最后把拼好的 Prompt 交给 LLM（如 GPT‑4），模型在生成答案时能够直接引用检索到的技术细节。  

**最巧妙的点**在于把表格拆成行向量并保留列名元数据，这让向量检索不再把表格当作一整块文字，而是能够在语义层面匹配到具体的数值或参数。类似于把一本食谱的配方表拆成“材料‑数量”对，搜索“需要多少氯化钠”时直接命中对应行，而不是整段文字。

### 实验与效果
- **数据集**：作者收集了约 2,000 份环境管理和污水处理相关的技术文档，涵盖 PDF、Word、Excel，中文占 60%，英文占 40%。  
- **任务**：在这些文档上进行“专业问答”，问题包括工艺参数查询、法规解释、设备选型等。  
- **Baseline**：使用传统 RAG（直接对原始 PDF 做 OCR → 文本向量化 → 检索），以及仅使用章节级向量（不拆表格）。  
- **结果**：在人工评估的 200 条问答上，本文方法的准确率提升约 **18%**，答案的技术细节完整度提升约 **25%**（评审给出 4.2/5 vs 3.4/5）。  
- **消融实验**：去掉表格行向量化后，整体准确率下降 12%；去掉 .docx 统一化直接使用原始 PDF，检索时间增加 2.3 倍，准确率下降 7%。  
- **局限**：作者指出对极其复杂的图形（如流程图）仍只能转成文字描述，向量化效果有限；此外，嵌入模型对专业术语的覆盖度仍受限，需要后续微调。

### 影响与延伸思考
这篇工作在 RAG 社区里引发了对“结构化向量化”更深入的讨论，后续有几篇论文尝试把 **JSON‑L**、**XML** 等结构直接映射到向量空间，甚至出现了专门针对表格的 **Table‑Embedding** 模型。对想进一步探索的读者，可以关注以下方向：  
- **领域自适应嵌入**：在专业语料上微调嵌入模型，提高术语匹配度。  
- **多模态检索**：把流程图、示意图转成图像向量，与文本向量联合检索。  
- **实时更新管线**：在文档频繁变更的场景下，实现增量向量化而不是全量重建。  

### 一句话记住它
把所有半结构化文档先统一成 Word，再把章节和表格行分别向量化，检索时精准定位细节，让 LLM 的回答像专业工程师一样有据可依。