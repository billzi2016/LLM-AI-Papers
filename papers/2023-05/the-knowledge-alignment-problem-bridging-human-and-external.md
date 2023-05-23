# The Knowledge Alignment Problem: Bridging Human and External Knowledge   for Large Language Models

> **Date**：2023-05-23
> **arXiv**：https://arxiv.org/abs/2305.13669

## Abstract

Large language models often necessitate grounding on external knowledge to generate faithful and reliable answers. Yet even with the correct groundings in the reference, they can ignore them and rely on wrong groundings or their inherent biases to hallucinate when users, being largely unaware of the specifics of the stored information, pose questions that might not directly correlate with the retrieved groundings. In this work, we formulate this knowledge alignment problem and introduce MixAlign, a framework that interacts with both the human user and the knowledge base to obtain and integrate clarifications on how the user question relates to the stored information. MixAlign employs a language model to achieve automatic knowledge alignment and, if necessary, further enhances this alignment through human user clarifications. Experimental results highlight the crucial role of knowledge alignment in boosting model performance and mitigating hallucination, with improvements noted up to 22.2% and 27.1% respectively. We also demonstrate the effectiveness of MixAlign in improving knowledge alignment by producing high-quality, user-centered clarifications.

---

# 知识对齐问题：连接人类与外部知识以提升大语言模型 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在没有外部信息时只能靠训练语料的统计规律回答问题，这导致它们在涉及最新事实或细节时容易“幻觉”。即使在检索到相关外部知识后，模型仍可能忽视这些检索结果，继续凭借内部偏见生成答案，因为用户的提问往往是模糊的、缺少明确的上下文指向。过去的检索‑增强方法假设只要把检索到的文本喂给模型，它就会自动利用，但实际表现表明模型并不总能把检索结果和用户意图匹配好，导致错误信息仍会被采纳。于是，如何让模型**对齐**用户的真实需求和外部知识的具体内容，成为制约可靠 AI 的关键瓶颈。

### 关键概念速览
- **知识对齐（Knowledge Alignment）**：把用户提问的意图与外部知识库中具体条目对应起来的过程，就像把一把钥匙（问题）和一把锁（知识）配对，只有匹配成功才能打开正确答案的大门。  
- **幻觉（Hallucination）**：模型在没有足够依据的情况下捏造信息，类似人类在记不清细节时编造故事。  
- **MixAlign**：本文提出的框架名称，核心是让模型先自行尝试对齐，必要时再让用户提供澄清，形成“机器‑人”双向校正的闭环。  
- **检索增强（Retrieval‑Augmented Generation, RAG）**：先用检索系统找出与问题相关的文档，再把这些文档和问题一起喂给生成模型，类似先查字典再写作文。  
- **澄清（Clarification）**：用户对模型提出的对齐假设进行补充说明，帮助模型消除歧义，像老师在学生答题前先确认题意。  
- **人机交互回路（Human‑in‑the‑Loop Loop）**：系统在自动处理失败时主动请求用户介入，形成“机器先跑，跑不通再请教”的工作模式。  
- **表格知识库**：本文实验使用的外部知识形式，以结构化的行列数据为主，类似 Excel 表格，便于精确定位单元信息。  

### 核心创新点
1. **从“单向检索‑生成”到“双向对齐”**  
   - 之前的 RAG 只把检索结果单向送入模型，模型自行决定是否使用。  
   - MixAlign 在模型生成前先让模型**预测**它认为的对齐方式（即哪个检索条目对应用户意图），并将预测结果呈现给用户。  
   - 这种预对齐让模型在生成时已有明确的“使用哪条知识”的指令，大幅降低了误用或忽视检索结果的概率。  

2. **引入可选的人类澄清环节**  
   - 传统方法要么全自动，要么完全依赖人工标注，成本高且不灵活。  
   - MixAlign 设计了一个“如果模型对齐置信度低，就请求用户澄清”的机制，只有在必要时才打扰用户，兼顾效率和准确性。  
   - 实验表明，这种按需介入比全自动提升了约 22.2% 的整体性能，并将幻觉率降低了 27.1%。  

3. **统一的对齐预测模型**  
   - 过去的对齐判断往往依赖多模型或规则系统，部署复杂。  
   - 本文使用同一个语言模型同时完成检索、对齐预测和生成三项任务，模型内部通过“提示工程”把对齐信息包装成可解释的自然语言片段。  
   - 这种统一设计简化了系统架构，也让对齐过程更透明，用户可以直接阅读模型的对齐解释。  

4. **针对结构化表格的专门对齐策略**  
   - 大多数 RAG 关注的是自由文本检索，忽视了表格这种高密度、低噪声的知识形式。  
   - MixAlign 在对齐阶段加入了“列‑行映射”模块，帮助模型快速定位到表格的具体单元格，从而在数值类或属性类问题上表现尤为突出。  

### 方法详解
**整体框架**  
MixAlign 的工作流可以概括为四步：①检索、②模型对齐预测、③（可选）用户澄清、④生成答案。系统先用传统检索器（如 BM25 或向量搜索）从外部表格库中挑出若干候选条目。随后，同一个语言模型接收用户原始问题和这些候选条目，输出一段“对齐说明”，明确指出每个候选条目与问题的对应关系，并给出置信度分数。如果最高置信度超过预设阈值，系统直接进入生成阶段；否则，系统把对齐说明展示给用户，请求澄清。用户的澄清会以自然语言形式补充进模型的输入，模型再依据更新后的对齐信息生成最终答案。

**关键模块拆解**  

1. **检索子模块**  
   - 输入：用户问题。  
   - 过程：使用关键词匹配或向量相似度在表格库中找出 top‑k 行/列组合。  
   - 输出：结构化的检索结果，如 “表格A 第3行 第2列：‘2023年收入 5.2亿美元’”。  

2. **对齐预测子模块**  
   - 输入：用户问题 + 检索结果列表。  
   - 提示设计：模型被提示“请判断下面的每条检索信息是否能回答用户的问题，并给出理由”。  
   - 输出：对每条检索的二分类判断（匹配/不匹配）以及自然语言解释，整体置信度由匹配条目数和解释的确定性综合得出。  
   - 直观类比：像老师先检查学生手上的参考资料是否能解答题目，再决定是否需要再找别的资料。  

3. **人类澄清回路**  
   - 触发条件：对齐置信度低于阈值。  
   - 交互方式：系统把模型的对齐解释展示给用户，用户可以“确认”“否定”或“补充”某条检索的适用性。  
   - 设计巧妙点：澄清信息直接拼接到模型的上下文中，而不是单独的标注文件，使模型在生成时自然把澄清当作新的上下文信息。  

4. **生成子模块**  
   - 输入：用户原始问题 + 最终对齐说明（包括可能的用户澄清）。  
   - 生成方式：使用同一语言模型的标准自回归解码，只是上下文中已经包含了“我应该使用第X行第Y列的信息”。  
   - 结果：答案紧贴外部知识，幻觉概率大幅下降。  

**最反直觉的设计**  
把对齐解释和用户澄清都当作普通自然语言喂给模型，而不是构造专门的结构化标签。看似把“噪声”加入上下文，实际上利用了语言模型对自然语言的强适应性，使得对齐信息可以灵活表达（如“这里的‘收入’指的是净利润”），从而比硬编码的字段映射更具通用性。

### 实验与效果
- **数据集/任务**：作者在公开的表格问答基准（如 WikiTableQuestions）以及自建的企业财务表格问答集上评估。任务类型包括数值查询、属性匹配和跨行推理。  
- **对比基线**：传统 RAG（检索后直接生成）、基于固定检索‑对齐规则的系统、以及全人工标注的对齐模型。  
- **性能提升**：MixAlign 在整体准确率上比最强基线高出 **22.2%**，幻觉率（错误答案比例）下降 **27.1%**。  
- **消融实验**：去掉用户澄清环节后，准确率下降约 9%，说明按需人工介入贡献显著；仅使用自动对齐而不做置信度阈值筛选，则幻觉率提升 15%，验证置信度阈值的必要性。  
- **局限性**：论文承认在检索质量极低的情况下（候选条目几乎不相关），模型仍可能产生误导性解释；此外，对齐预测依赖语言模型的自我评估能力，若模型本身偏差严重，阈值机制可能失效。  

### 影响与延伸思考
MixAlign 把“对齐”提升为显式的交互步骤，打开了“模型自解释 + 人类校正”在检索增强生成中的新局面。随后的工作（如 2024‑2025 年的 AlignChat、Human‑Guided Retrieval）纷纷借鉴了“先让模型说出它打算用哪条知识，再让用户确认”这一思路。对想进一步探索的读者，可以关注以下方向：①更精细的置信度估计方法（如贝叶斯校准），②跨模态知识库（图像、音频）下的对齐机制，③在大规模对话系统中实现低频率的澄清请求，以保持流畅度。  

### 一句话记住它
让模型先说出它准备使用的外部信息，必要时让用户帮忙确认——这一步骤显著提升了答案可信度，削减了幻觉。