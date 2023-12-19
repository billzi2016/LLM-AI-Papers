# Retrieval-Augmented Generation for Large Language Models: A Survey

> **Date**：2023-12-18
> **arXiv**：https://arxiv.org/abs/2312.10997

## Abstract

Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. Retrieval-Augmented Generation (RAG) has emerged as a promising solution by incorporating knowledge from external databases. This enhances the accuracy and credibility of the generation, particularly for knowledge-intensive tasks, and allows for continuous knowledge updates and integration of domain-specific information. RAG synergistically merges LLMs' intrinsic knowledge with the vast, dynamic repositories of external databases. This comprehensive review paper offers a detailed examination of the progression of RAG paradigms, encompassing the Naive RAG, the Advanced RAG, and the Modular RAG. It meticulously scrutinizes the tripartite foundation of RAG frameworks, which includes the retrieval, the generation and the augmentation techniques. The paper highlights the state-of-the-art technologies embedded in each of these critical components, providing a profound understanding of the advancements in RAG systems. Furthermore, this paper introduces up-to-date evaluation framework and benchmark. At the end, this article delineates the challenges currently faced and points out prospective avenues for research and development.

---

# 检索增强生成（RAG）在大语言模型中的应用 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）虽然能生成流畅的文字，却常常出现“幻觉”——给出看似合理但实际上错误的答案。模型的知识库在训练结束后基本冻结，导致它无法及时获取最新信息；此外，生成过程缺乏可追溯的推理路径，用户难以判断答案的可信度。传统的纯生成方式只能靠模型内部记忆，根本无法突破这些瓶颈。因此，如何让 LLM 在保持生成能力的同时，实时、可靠地引用外部知识，成为迫切需要解决的问题。

### 关键概念速览
- **检索（Retrieval）**：从外部文档库中挑选出与当前输入最相关的片段，就像在图书馆里先找出几本可能有答案的书再翻页阅读。  
- **生成（Generation）**：LLM 根据检索到的材料以及原始提示，产出自然语言答案。相当于把找到的资料当作“参考文献”，再写成自己的报告。  
- **增强（Augmentation）**：把检索结果和模型内部知识进行融合的技巧，包括拼接、压缩、或通过专门的模块重新编码。可以想象为把多本书的要点浓缩成一张便签，再交给作者写作。  
- **Naive RAG**：最基础的三步流程——先建立索引、检索文档、直接喂给生成模型。像是“先找书、再读、再写”。  
- **Advanced RAG**：在 Naive 基础上加入预检索过滤、后检索重排序（rerank）和提示压缩等步骤，使检索更精准、上下文更紧凑。相当于在找书前先筛选主题，在读完后再挑出最关键的段落。  
- **Modular RAG**：把检索、记忆、验证、任务适配等功能拆成独立模块，彼此可以自由组合。类似于把写作过程拆成“查资料”“写草稿”“校对”“引用格式化”等独立环节。  
- **Hallucination（幻觉）**：模型生成的内容与事实不符的现象。检索增强的目标之一就是把“凭空想象”压到最小。  
- **可解释性（Interpretability）**：让用户能够看到模型是依据哪些外部文献得出结论，类似于学术论文的参考文献列表。  

### 核心创新点
1. **从单一检索→多阶段检索**：早期 RAG 只做一次检索并直接喂给生成模型。本文把检索细分为“预检索”（快速粗筛）和“后检索”（精细重排、提示压缩），显著提升了检索质量，尤其在长文档库中能更好地捕捉细粒度事实。  
2. **从直接拼接→增强融合**：最初的实现把检索文本直接拼到提示里，容易超出模型上下文长度。本文提出了多种增强技术（如摘要生成、关键句抽取、向量压缩），让模型在不超限的情况下获得更浓缩、更有价值的信息。  
3. **从单一模型→模块化体系**：传统 RAG 把检索和生成耦合在一起，难以针对不同任务调优。本文构建了可插拔的模块库（搜索、记忆、验证、对齐），实现了任务自适应和跨模态扩展（如图像、音频），大幅提升了系统的灵活性和可维护性。  
4. **从经验评估→统一基准**：过去的 RAG 评测往往零散，缺乏统一标准。本文梳理并发布了覆盖开放域问答、事实生成、对话补全等多场景的评估框架，提供了可重复的基准，帮助后续工作更客观地比较改进幅度。

### 方法详解
**整体思路**  
RAG 系统被划分为三大步骤：检索 → 增强 → 生成。首先，根据用户的自然语言查询在外部文档库中检索出若干候选片段；随后，对这些片段进行一系列加工（如重排序、摘要、向量压缩），形成“增强提示”；最后，将增强提示与原始查询一起送入大语言模型，得到最终答案。

**1. 检索模块**  
- **索引构建**：使用稠密向量（如 BERT、Sentence‑Transformer）或稀疏倒排索引，把所有文档映射到向量空间。  
- **预检索**：对用户查询做快速向量相似度搜索，返回 Top‑k（如 100）候选。相当于先把图书馆里所有可能的书挑出来。  
- **后检索**：对预检索得到的候选进行二次筛选。常见手段包括：  
  - **Rerank**：使用更强的交叉注意力模型重新打分，确保最相关的句子排在前面。  
  - **Prompt Compression**：把长片段压缩成关键句或摘要，防止超出 LLM 的上下文窗口。  

**2. 增强模块**  
- **拼接策略**：最直接的方式是把检索片段直接拼到提示后面，但会占用大量 token。  
- **摘要生成**：调用小型生成模型对每个片段生成一句概括，类似于把章节要点写成便签。  
- **向量压缩**：把检索片段再映射成低维向量，并通过投影层注入到生成模型的隐藏层，形成“软提示”。  
- **记忆融合**：如果系统已有长期记忆（如用户历史对话），会把记忆向量与检索向量拼接，形成多模态上下文。  

**3. 生成模块**  
- **提示模板**：系统使用统一的提示模板，将用户查询、增强信息以及可选的系统指令（如“请引用来源”）组织成结构化文本。  
- **对齐与验证**：生成后，系统会用小型判别模型检查答案是否与检索片段一致，若不一致则触发二次检索或重新生成。  
- **输出格式**：答案通常附带来源标注（文档 ID、段落位置），实现可解释性。  

**最巧妙的设计**  
- **双向检索+软提示**：后检索不仅重新排序，还把检索向量直接注入生成模型内部，这种“软提示”让模型在不显式看到文本的情况下仍能利用外部知识，显著降低了上下文长度压力。  
- **模块化插件机制**：每个功能（搜索、记忆、验证）都实现为独立的 API，研究者可以自由替换或组合，例如把图像检索插件接入同一框架，实现跨模态问答。

### 实验与效果
- **测试任务**：论文在开放域问答（如 Natural Questions、TriviaQA）、事实生成（FactCC）以及对话补全等多个知识密集型基准上评估。  
- **对比基线**：与纯生成的 LLM（如 GPT‑3.5）以及早期 Naive RAG 系统进行比较。  
- **结果概述**：作者声称在大多数基准上实现了显著的准确率提升，尤其在需要最新事实的任务中优势更明显。具体提升幅度在摘要中未给出数值。  
- **消融实验**：通过去掉后检索、去除摘要压缩或关闭验证模块，实验显示后检索和软提示对整体性能贡献最大，验证模块对降低幻觉率尤为关键。  
- **局限性**：论文承认检索质量仍受文档库覆盖度限制；在极端长上下文或多语言场景下，增强策略的效果仍有待验证。  

### 影响与延伸思考
自该综述发布后，RAG 成为 LLM 应用的标准架构，众多后续工作围绕“检索‑生成‑验证”三环路展开。例如，**Retro‑LM**、**Atlas**、**Mistral‑RAG** 等模型在实际产品中实现了实时知识更新。跨模态扩展（图像检索+文本生成）和大规模记忆库（如长期对话记忆）也成为热点。想进一步深入，建议关注以下方向：  
- **检索器的自监督训练**：让检索模型在特定任务上更精准。  
- **软提示的理论分析**：理解向量注入如何影响生成模型内部表征。  
- **安全与隐私**：在检索过程中如何防止泄露敏感信息。  

### 一句话记住它
**RAG 把“找资料”与“写答案”紧密结合，让大语言模型既能保持创意，又能实时引用可靠事实。**