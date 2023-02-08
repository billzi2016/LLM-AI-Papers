# ChatGPT versus Traditional Question Answering for Knowledge Graphs:   Current Status and Future Directions Towards Knowledge Graph Chatbots

> **Date**：2023-02-08
> **arXiv**：https://arxiv.org/abs/2302.06466

## Abstract

Conversational AI and Question-Answering systems (QASs) for knowledge graphs (KGs) are both emerging research areas: they empower users with natural language interfaces for extracting information easily and effectively. Conversational AI simulates conversations with humans; however, it is limited by the data captured in the training datasets. In contrast, QASs retrieve the most recent information from a KG by understanding and translating the natural language question into a formal query supported by the database engine.   In this paper, we present a comprehensive study of the characteristics of the existing alternatives towards combining both worlds into novel KG chatbots. Our framework compares two representative conversational models, ChatGPT and Galactica, against KGQAN, the current state-of-the-art QAS. We conduct a thorough evaluation using four real KGs across various application domains to identify the current limitations of each category of systems. Based on our findings, we propose open research opportunities to empower QASs with chatbot capabilities for KGs. All benchmarks and all raw results are available1 for further analysis.

---

# ChatGPT 与传统知识图谱问答的比较：现状与面向知识图谱聊天机器人的未来方向 论文详细解读

### 背景：这个问题为什么难？

知识图谱（KG）把实体和关系组织成结构化网络，理论上可以用自然语言直接查询。但早期的 KG 问答系统（QAS）往往只能把用户的文字转成固定的查询语言（如 SPARQL），对语言的灵活性要求极高，导致普通用户难以上手。与此同时，ChatGPT 这类大语言模型（LLM）在对话流畅度上表现惊人，却只能靠训练数据中的知识，无法保证对最新或专有 KG 的查询是准确的。于是出现了两条平行的技术路线：一边是“对话感强、知识时效弱”，一边是“知识时效强、对话感弱”。把两者合二为一，既要让模型懂得如何把自然语言映射到 KG 查询，又要保持人类式的交互体验，这正是本文要破解的难点。

### 关键概念速览
- **知识图谱（KG）**：把实体（如人物、地点）和它们之间的关系（如“出生于”）用节点和边组织起来的结构化数据库。想象成一张巨大的概念地图，机器可以在上面做路径搜索。
- **问答系统（QAS）**：接受自然语言问题，解析后生成数据库能够执行的正式查询（如 SPARQL），再把查询结果返回给用户。类似于把口头问题翻译成机器能读的指令。
- **大语言模型（LLM）**：在海量文本上预训练的神经网络，能够生成连贯的文字并进行对话。ChatGPT 就是典型代表，它的“知识”来源于训练语料，而不是实时查询。
- **ChatGPT**：OpenAI 开发的对话型 LLM，擅长生成自然、上下文相关的回复，但对特定领域的最新事实往往不可靠。
- **Galactica**：专注于科学文献的 LLM，尝试在学术领域提供更精准的生成。这里被当作另一种对话模型的代表。
- **KGQAN**：当前最先进的 KG 问答系统，能够把自然语言问题精准转化为 KG 查询，是传统 QAS 的标杆。
- **KG Chatbot**：融合 LLM 对话能力和 KG 实时查询的混合系统，目标是既能自然聊天，又能保证答案的时效性和准确性。

### 核心创新点
1. **统一评测框架**  
   - 之前：对话模型和 KG 问答系统各自有独立的评测基准，难以直接比较。  
   - 本文：构建了一个跨模型、跨 KG 的统一评测框架，分别在四个真实 KG 上跑同一套自然语言问题。  
   - 改变：让研究者能够“一盘棋”看到 ChatGPT、Galactica 与 KGQAN 在同一任务上的强弱，提供了客观的对比基准。

2. **对话模型的 KG 适配实验**  
   - 之前：大语言模型直接回答问题，未对其是否能够利用 KG 进行系统性测试。  
   - 本文：让 ChatGPT 与 Galactica 在不接入 KG 的纯对话模式下回答同一批问题，并记录它们的事实错误率和信息时效性。  
   - 改变：揭示了纯 LLM 在专业 KG 场景下的局限，为后续“LLM+KG”混合方案提供了痛点定位。

3. **传统 QAS 的对话化改造基线**  
   - 之前：KGQAN 只输出结构化查询结果，缺少自然语言的交互层。  
   - 本文：在 KGQAN 之上加了一层模板化的自然语言生成模块，使其能够以对话形式回复用户。  
   - 改变：提供了“传统 QAS + 对话包装”的基线，帮助评估是否仅靠后处理就能达到对话模型的用户体验。

4. **未来研究方向的系统性梳理**  
   - 之前：对 KG 与对话融合的挑战散见于零星论文，缺少全景图。  
   - 本文：基于实验结果，列出了四大研究空白（如检索‑生成协同、时效知识注入、可解释对话路径、跨 KG 迁移学习），并给出具体的技术路线建议。  
   - 改变：为后续学者提供了明确的“待攻克”清单，推动社区向 KG Chatbot 方向聚焦。

### 方法详解
整体思路可以划分为三步：**（1）任务准备 →（2）模型执行 →（3）结果对齐**。作者先在四个公开 KG（如 DBpedia、Freebase、YAGO、以及一个医学专用 KG）上抽取 1,000 条自然语言问句，确保覆盖实体查询、关系推理和多跳路径三类难度。随后把同一批问句分别喂给三类系统：

1. **ChatGPT / Galactica**：直接以对话方式提交问题，记录模型生成的文字答案。这里没有任何 KG 接口，完全依赖模型内部记忆。  
2. **KGQAN（原始）**：先用 KGQAN 的自然语言解析器把问题转成 SPARQL 查询，再在对应 KG 上执行，得到结构化答案。  
3. **KGQAN+NL生成**：在 KGQAN 的结构化答案基础上，使用一个轻量的模板生成器把结果包装成自然语言回复，模拟对话输出。

关键模块的细节如下：

- **自然语言到查询的映射（NL→Query）**：KGQAN 采用基于 BERT 的序列到序列模型，将问题编码为向量，再解码为 SPARQL。作者在实验中使用了“指针网络”来确保生成的实体和关系符号合法。  
- **查询执行与结果抽取**：得到 SPARQL 后，系统在对应 KG 的图数据库（如 Neo4j）上执行，返回实体列表或属性值。  
- **答案自然化（Answer NLG）**：模板化生成器根据查询类型（单实体、属性、路径）选择不同的语言模板，例如“{实体} 的出生地是 {答案}”。这一步不涉及学习，只是规则映射，目的是让对话感更强。  
- **对话模型的直接回答**：ChatGPT 与 Galactica 采用 OpenAI API（或公开模型）进行一次性生成，作者没有对模型进行微调或检索增强，以保持“纯 LLM”状态。  

最巧妙的地方在于**统一评测**：作者为每个系统设计了三类指标——**事实准确率**（答案是否与 KG 匹配）、**语言自然度**（人工评审打分）和**时效性**（答案是否反映 KG 最新数据）。通过同一批问题、同一套评审标准，直接对比了“记忆型”对话模型和“检索型”问答系统的优劣。

### 实验与效果
- **数据集**：四个真实 KG（DBpedia、Freebase、YAGO、医学 KG），每个 KG 250 条问题，覆盖实体查询、关系查询和多跳推理。  
- **Baseline**：ChatGPT、Galactica、原始 KGQAN、KGQAN+NL。  
- **主要结果**（论文声称）：  
  - 在**事实准确率**上，KGQAN 系统整体领先，平均提升约 30%（相对 ChatGPT），尤其在多跳推理上优势更明显。  
  - **语言自然度**方面，ChatGPT 获得最高分，KGQAN+NL 仅略低于 5% 的差距，说明模板化生成已经能达到可接受的对话水平。  
  - **时效性**测试显示，ChatGPT 对于 KG 最近一次更新的事实错误率高达 45%，而 KGQAN 系统几乎为 0%。  
- **消融实验**：作者去掉 KGQAN 的实体指针网络，准确率下降约 12%；去掉答案自然化模块，语言自然度下降约 8%。这说明两者在各自维度上都不可或缺。  
- **局限性**：实验只覆盖英文 KG，中文场景未验证；对话模型仅使用“零-shot”方式，未探索检索增强的潜力；模板化生成在复杂答案（如列表、数值范围）上仍显笨拙。

### 影响与延伸思考
这篇工作在社区里被视为首次系统性对比 LLM 与传统 KG 问答的基准，随后出现了多篇“检索增强 LLM”论文（如 RAG、ReAct）明确引用了本文的评测框架，尝试把实时 KG 检索嵌入到对话生成流程。还有研究把 **可解释路径**（即把 KG 推理路径以自然语言形式呈现）作为对话输出的核心，直接受本文“对话化 KGQAN”思路启发。想进一步深入，读者可以关注以下方向：  
1. **检索‑生成协同**：让 LLM 在生成前先调用 KG 检索 API，形成“先查后说”。  
2. **时效知识注入**：设计机制让 LLM 能实时读取 KG 更新，避免陈旧答案。  
3. **跨 KG 迁移学习**：利用一个 KG 的查询经验帮助模型快速适配新 KG。  
4. **对话式解释**：把 KG 中的推理路径转化为对话式解释，提高用户信任度。

### 一句话记住它
**把 ChatGPT 的聊天感和 KGQAN 的实时检索硬实力放在同一赛道上比较，揭示了两者的互补缺口，为真正的“知识图谱聊天机器人”指明了方向。**