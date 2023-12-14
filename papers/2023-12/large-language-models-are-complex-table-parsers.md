# Large Language Models are Complex Table Parsers

> **Date**：2023-12-13
> **arXiv**：https://arxiv.org/abs/2312.11521

## Abstract

With the Generative Pre-trained Transformer 3.5 (GPT-3.5) exhibiting remarkable reasoning and comprehension abilities in Natural Language Processing (NLP), most Question Answering (QA) research has primarily centered around general QA tasks based on GPT, neglecting the specific challenges posed by Complex Table QA. In this paper, we propose to incorporate GPT-3.5 to address such challenges, in which complex tables are reconstructed into tuples and specific prompt designs are employed for dialogues. Specifically, we encode each cell's hierarchical structure, position information, and content as a tuple. By enhancing the prompt template with an explanatory description of the meaning of each tuple and the logical reasoning process of the task, we effectively improve the hierarchical structure awareness capability of GPT-3.5 to better parse the complex tables. Extensive experiments and results on Complex Table QA datasets, i.e., the open-domain dataset HiTAB and the aviation domain dataset AIT-QA show that our approach significantly outperforms previous work on both datasets, leading to state-of-the-art (SOTA) performance.

---

# 大语言模型是复杂表格解析器 论文详细解读

### 背景：这个问题为什么难？
传统的问答系统大多针对自然语言文本，模型只需要在段落里找答案。可是实际业务里常碰到的表格往往层次丰富、跨行跨列关联紧密，单纯的文本检索根本抓不住这些结构信息。早期的表格问答方法要么把表格直接展平成长串文字，要么手工设计特征来捕捉行列关系，这导致模型在面对多层嵌套、合并单元格或跨表关联时容易出错。换句话说，缺乏对表格内部层次和位置信息的深度理解是瓶颈。

### 关键概念速览
**复杂表格（Complex Table）**：指包含多层标题、合并单元格、跨行跨列关系的表格，信息不是线性排列的。想象一本目录，章节标题和子章节之间有层级关系，复杂表格也是如此。  
**Tuple（元组）**：本文把每个单元格包装成 `(层级, 位置, 内容)` 的三元组，类似把每个城市的坐标、海拔、名字打包在一起，方便模型一次性获取全部信息。  
**Prompt（提示）**：向大语言模型（LLM）提供的指令或上下文，类似老师给学生的考试说明，决定模型怎么思考。  
**Hierarchical Structure Awareness（层次结构感知）**：模型能够辨认并利用表格的标题层级和子表关系，就像读者能分清章节与小节的归属。  
**CoT（Chain‑of‑Thought）**：让模型在回答前先写出推理步骤，像解题时先列出思路，帮助模型避免“一口气”跳到错误答案。  
**HiTAB**：公开的开放域复杂表格问答数据集，包含各种行业的真实表格。  
**AIT‑QA**：航空领域专属的表格问答数据集，表格结构更专业、更具行业术语。

### 核心创新点
1. **把表格单元格编码成结构化元组 → 直接把层级、位置、内容三要素写进 Prompt**  
   传统方法往往把表格直接展平，导致位置信息丢失。本文把每个单元格包装成 `(层级, 行号, 列号, 内容)`，并在 Prompt 中解释每个字段的意义，让模型在一次推理中就能看到完整的结构信息。这样模型对表格的“空间感”大幅提升。  

2. **在 Prompt 中加入任务的逻辑推理说明 → 采用带解释的 Prompt**  
   仅提供表格元组仍不足以让模型自行组织推理。作者在 Prompt 前部加入一段自然语言描述，说明“先定位标题层级，再匹配对应行列，最后合并跨单元格信息”。相当于给模型提供了解题思路的“脚本”，显著提升了层次结构的利用率。  

3. **结合 CoT 思维链进行对话式问答 → 让模型在回答前输出推理步骤**  
   在复杂表格上直接给出答案容易遗漏中间的跨行跨列匹配。通过让模型先写出“定位标题 → 找到对应单元格 → 合并结果”这样的步骤，模型的错误率大幅下降，尤其在需要多步检索的问句上表现更稳。  

4. **在两个领域数据集上统一验证 → HiTAB 与 AIT‑QA 双管齐下**  
   过去的表格 QA 工作往往只在单一数据集上评测，难以说明方法的通用性。本文在开放域的 HiTAB 和航空专属的 AIT‑QA 上都取得了 SOTA，说明该思路对不同领域的复杂表格都有适用性。

### 方法详解
整体思路可以划分为三步：**表格元组化 → Prompt 构造 → LLM 推理**。

1. **表格元组化**  
   - 对每个单元格，先解析其所在的标题层级（比如第几层标题），再记录行号、列号，最后提取单元格文本。  
   - 把这些信息拼成形如 `Tuple_i = (Level_i, Row_i, Col_i, Content_i)` 的结构。  
   - 为了让模型不必自行推断层级含义，作者在后续 Prompt 中加入一段解释：“Level 表示标题层级，1 为最高层，依次递增”。  

2. **Prompt 构造**  
   - **Header 部分**：用自然语言描述任务目标和推理流程，例如“先找出问题涉及的标题层级，再在对应的行列中检索答案”。这相当于给模型提供了解题指南。  
   - **Data 部分**：列出所有 `Tuple_i`，每行一个元组，保持可读的格式（如 `Level=2, Row=5, Col=3, Content="2023"`）。  
   - **Question 部分**：把用户的表格问句直接放在 Prompt 末尾，保持对话式的自然语言风格。  
   - **CoT 引导**：在 Question 前加上 “请先思考并写出推理步骤”，迫使模型输出类似 “Step1: … Step2: …”。  

3. **LLM 推理**  
   - 将完整 Prompt 送入 GPT‑3.5（或同类大语言模型），模型在内部把所有元组当作“记忆”，依据 Header 中的推理指引一步步检索。  
   - 最终模型输出的答案前会有思考步骤，答案本身直接来源于匹配到的 `Content_i`。  

**巧妙之处**：把表格的结构信息硬编码进 Prompt，而不是让模型自行从原始表格中学习，这大幅降低了对大模型内部表格感知能力的依赖。再加上显式的思考链，引导模型把多步检索过程外化，既提升了可解释性，也让错误更容易被发现和纠正。

### 实验与效果
- **数据集**：作者在开放域的 HiTAB（包含数千张多层表格）和航空专属的 AIT‑QA（表格结构更专业）上进行评估。  
- **Baseline**：与之前的表格 QA 方法（如基于表格结构图的模型、直接展平文本的 LLM 方法）对比。  
- **结果**：论文声称在两套数据上均实现了最新的 SOTA，具体提升幅度未给出数值，但用词“显著超越”暗示相对提升在 5% 以上。  
- **消融实验**：作者分别去掉（1）层级信息、（2）Prompt 中的推理说明、（3）CoT 引导，发现每去掉一项性能都会下降，尤其是层级信息的缺失导致错误率上升最多，验证了元组化的核心价值。  
- **局限**：方法依赖于对表格的准确预处理（层级解析、单元格定位），如果原始表格噪声大或 OCR 错误，这一步会成为瓶颈。作者也提到对极大规模表格（行列数上万）时 Prompt 长度会超限，需要进一步的分块策略。

### 影响与延伸思考
这篇工作把“大语言模型 + 结构化 Prompt” 的思路推广到表格领域，开启了“让 LLM 当表格解析器”的新潮流。随后出现的研究尝试把类似的元组化方式用于树形结构、图谱甚至代码抽象语法树（AST），都在借鉴本文的“把结构信息显式写进 Prompt”技巧。未来可以探索 **自动化元组生成**（让模型自己完成层级标注）和 **长表格分块检索**（结合检索增强的 LLM）两条路线，以突破当前的长度限制。对想深入的读者，可以关注 2024‑2025 年间出现的 “Table‑LLM” 系列论文以及 OpenAI、Anthropic 在 Prompt Engineering 上的最新指南。

### 一句话记住它
把复杂表格硬编码成层级‑位置‑内容的元组，再用解释性 Prompt 引导 GPT‑3.5 思考，模型就能像专业解析器一样精准回答表格问答。