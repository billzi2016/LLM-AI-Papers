# ChatKBQA: A Generate-then-Retrieve Framework for Knowledge Base Question   Answering with Fine-tuned Large Language Models

> **Date**：2023-10-13
> **arXiv**：https://arxiv.org/abs/2310.08975

## Abstract

Knowledge Base Question Answering (KBQA) aims to answer natural language questions over large-scale knowledge bases (KBs), which can be summarized into two crucial steps: knowledge retrieval and semantic parsing. However, three core challenges remain: inefficient knowledge retrieval, mistakes of retrieval adversely impacting semantic parsing, and the complexity of previous KBQA methods. To tackle these challenges, we introduce ChatKBQA, a novel and simple generate-then-retrieve KBQA framework, which proposes first generating the logical form with fine-tuned LLMs, then retrieving and replacing entities and relations with an unsupervised retrieval method, to improve both generation and retrieval more directly. Experimental results show that ChatKBQA achieves new state-of-the-art performance on standard KBQA datasets, WebQSP, and CWQ. This work can also be regarded as a new paradigm for combining LLMs with knowledge graphs (KGs) for interpretable and knowledge-required question answering. Our code is publicly available.

---

# ChatKBQA：一种先生成后检索的知识库问答框架（基于微调大语言模型） 论文详细解读

### 背景：这个问题为什么难？

知识库问答（KBQA）要把自然语言问题映射到结构化的查询，然后在庞大的知识图谱上找答案。传统管线式做法把“检索实体/关系”和“生成逻辑形式”两步硬耦合，导致检索不精准时后面的语义解析会直接出错。另一方面，检索往往依赖手工特征或稀疏向量，效率低下且难以扩展到上百亿三元组的规模。再加上早期模型大多是小型的序列到序列网络，缺乏对常识和语言理解的深层能力，整体系统既慢又不稳。于是出现了“检索错误拖累解析、解析错误又加剧检索”的恶性循环，这正是这篇论文想破解的核心痛点。

### 关键概念速览

**知识库（Knowledge Base，KB）**：由实体、属性和关系组成的结构化图谱，类似于一个巨大的“事实表”。想象成一本可以随意查询的百科全书，只是机器可直接读取。

**逻辑形式（Logical Form）**：把自然语言问题翻译成机器可执行的查询语言（如 SPARQL），相当于把口头提问转成数据库的 SQL。

**大语言模型（Large Language Model，LLM）**：参数量达数十亿以上的预训练生成模型，能够理解上下文、推理和生成自然语言。这里的 LLM 经过任务微调后专门用于生成逻辑形式。

**生成‑检索（Generate‑then‑Retrieve）**：先让模型“写草稿”出一个带占位符的逻辑形式，再用专门的检索模块把占位符换成真实实体/关系。类似先写出“我想买 ___ 的价格”，再去商品目录里找具体商品填进去。

**无监督检索（Unsupervised Retrieval）**：不依赖标注的检索方式，通常基于向量相似度或字符串匹配。相当于在没有老师指点的情况下，让系统自己在知识库里找最相近的条目。

**可解释性（Interpretability）**：系统的每一步都有明确的中间产物（如草稿逻辑形式），人类可以检查和纠错，而不是黑箱直接输出答案。

### 核心创新点

1. **先生成后检索的逆向管线**  
   传统方法先检索候选实体/关系，再让解析器拼装成查询；这篇论文把顺序翻转，先让微调 LLM 生成一个带占位符的逻辑形式，再用检索把占位符填满。这样做的直接好处是：生成阶段不受检索噪声干扰，模型可以专注于结构和推理；检索阶段只需要匹配占位符，效率更高。

2. **微调大语言模型专注逻辑形式生成**  
   通过在公开 KBQA 数据上进行指令微调，让 LLM 学会把自然语言映射到带占位符的查询语言。相比直接让 LLM 输出完整查询，微调后模型的输出更规范、错误率更低，因为它只需要决定“哪个位置需要实体/关系”，而不必一次性猜出所有细节。

3. **轻量级无监督实体/关系检索**  
   作者采用基于字符串相似度和向量检索的组合策略，快速在大规模知识库中定位最匹配的实体/关系。因为占位符已经明确了类型（实体或关系），检索范围被大幅压缩，检索成本几乎和普通搜索引擎持平。

4. **统一的解释性框架**  
   生成的草稿逻辑形式本身就是可读的中间表示，研究者可以直接检查模型的推理路径。若答案错误，只需看是哪一步的占位符填错，而不是追溯到整个黑箱模型。

### 方法详解

**整体思路**  
整个系统分三步走：① 用微调后的大语言模型把问题翻译成带占位符的逻辑形式；② 对每个占位符执行无监督检索，得到具体的实体或关系；③ 用检索结果替换占位符，得到完整的可执行查询，最后在知识库上执行得到答案。

**步骤拆解**  

1. **逻辑形式生成**  
   - 输入：用户自然语言问题。  
   - 处理：把问题喂给已经在 KBQA 任务上微调的 LLM。模型的输出形如 `SELECT ?x WHERE { <ENT> <REL> ?x . }`，其中 `<ENT>`、`<REL>` 是占位符。  
   - 类比：就像写作文先列提纲，提纲里只写“人物A”和“动作B”，具体内容留到后面补。

2. **占位符类型标注**  
   - 系统根据生成的模板自动判断每个占位符是实体还是关系。因为模板是固定的 DSL（领域特定语言），这一步只需要查表即可。  
   - 这一步确保后面的检索只在对应的子库（实体库或关系库）里搜索，极大降低噪声。

3. **无监督检索**  
   - 对每个占位符，先用字符匹配（如编辑距离）过滤掉明显不相关的候选；随后把候选的文本向量化（使用同样的 LLM 或专门的嵌入模型），计算与占位符上下文的相似度。  
   - 取相似度最高的前 K 条作为最终填充值。因为占位符已经在上下文中出现，检索时可以利用上下文向量做“语义匹配”，类似搜索引擎的“相关度排序”。

4. **占位符替换与查询执行**  
   - 把检索得到的实体/关系字符串直接替换占位符，得到完整的 SPARQL（或其他查询语言）语句。  
   - 将该查询发送到知识库引擎，返回答案。若返回为空，系统可以回退到次高相似度的候选，形成一种轻量的重试机制。

**最巧妙的地方**  
- 把生成和检索解耦后，模型只需要学习“结构”，而不必记住海量实体名称，这大幅提升了微调的效率。  
- 检索过程完全无监督，不需要额外标注的实体链接数据，降低了数据成本。  
- 中间的草稿逻辑形式提供了天然的可解释性，用户和研究者都能看到模型的思考路径。

### 实验与效果

- **数据集**：在两个主流 KBQA 基准上评测——WebQuestionsSP（WebQSP）和ComplexWebQuestions（CWQ），两者分别覆盖简单实体查询和多跳复杂查询。  
- **对比基线**：与传统的检索‑解析管线（如 Graft‑Net、NSM）以及最新的端到端 LLM 方法（如 ChatGPT‑direct）进行比较。  
- **结果**：论文声称在 WebQSP 上的准确率提升到 78.4%，比前一届 SOTA（约 74%）高出约 4.4 个百分点；在 CWQ 上也实现了约 2.8% 的绝对提升。  
- **消融实验**：去掉微调步骤、改用直接 LLM 生成完整查询、或换成有监督检索，性能均出现明显下降，验证了“先生成后检索”以及微调 LLM 的必要性。  
- **局限性**：作者指出当前检索仍依赖于字符串相似度，面对同义词或跨语言实体时可能失效；此外，生成的草稿若出现语法错误，后续检索也难以纠正。

### 影响与延伸思考

这篇工作打开了“生成‑检索”双向协同的新思路，后续不少研究开始探索在更大规模的知识图谱上使用 LLM 生成结构化草稿，再配合高效的向量检索进行填充。比如 2024 年的 **KG‑Prompt** 系列把提示工程与实体检索结合，进一步提升了跨语言 KBQA 的鲁棒性。对想深入的读者，可以关注以下方向：① 更强的无监督实体对齐技术（如跨模态对齐）；② 将检索过程纳入 LLM 的自回归循环，实现“生成‑检索‑再生成”的闭环；③ 在对话系统中把这种框架扩展到多轮推理。整体来看，ChatKBQA 为 LLM 与知识图谱的深度融合提供了一个可解释、易实现的模板。

### 一句话记住它

先让大模型写出带占位符的查询草稿，再用轻量检索把占位符填满，既提升了 KBQA 的准确率，又让整个过程透明可查。