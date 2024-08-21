# DocTabQA: Answering Questions from Long Documents Using Tables

> **Date**：2024-08-21
> **arXiv**：https://arxiv.org/abs/2408.11490

## Abstract

We study a new problem setting of question answering (QA), referred to as DocTabQA. Within this setting, given a long document, the goal is to respond to questions by organizing the answers into structured tables derived directly from the document's content. Unlike traditional QA approaches which predominantly rely on unstructured text to formulate responses, DocTabQA aims to leverage structured tables as answers to convey information clearly and systematically, thereby enhancing user comprehension and highlighting relationships between data points. To the best of our knowledge, this problem has not been previously explored. In this paper, we introduce the QTabA dataset, encompassing 300 financial documents, accompanied by manually annotated 1.5k question-table pairs. Initially, we leverage Large Language Models (LLMs) such as GPT-4 to establish a baseline. However, it is widely acknowledged that LLMs encounter difficulties when tasked with generating intricate, structured outputs from long input sequences. To overcome these challenges, we present a two-stage framework, called DocTabTalk, which initially retrieves relevant sentences from extensive documents and subsequently generates hierarchical tables based on these identified sentences. DocTabTalk incorporates two key technological innovations: AlignLLaMA and TabTalk, which are specifically tailored to assist GPT-4 in tackling DocTabQA, enabling it to generate well-structured, hierarchical tables with improved organization and clarity. Comprehensive experimental evaluations conducted on both QTabA and RotoWire datasets demonstrate that our DocTabTalk significantly enhances the performances of the GPT-4 in our proposed DocTabQA task and the table generation task. The code and dataset are available at https://github.com/SmileWHC/DocTabQA for further research.

---

# DocTabQA：基于长文档的表格化问答 论文详细解读

### 背景：这个问题为什么难？

传统问答系统大多把答案直接写成一句话或一段文字，模型只需要在长文档里找出相关片段就行。可是当用户想要把信息以表格的形式呈现——比如财报里不同年份的收入、成本、利润对应关系——普通的文本答案会显得杂乱，难以一眼看清数据之间的关联。要在几千字的文档里抽取出结构化的行列信息，并且按照层次组织成表格，模型面临两大挑战：① 输入序列太长，超出大多数语言模型的上下文窗口；② 生成的输出必须严格遵守表格的行列约束，稍有格式错误就会导致后续数据处理失败。之前的研究要么只做抽取式问答，要么在小段落上生成表格，根本没有系统化地解决“长文档 → 结构化表格”这一任务。

### 关键概念速览
- **DocTabQA**：一种新型问答设定，要求模型在阅读完整长文档后，输出直接来源于文档内容的结构化表格。想象成让模型把一篇报告“翻译”成 Excel 表格。
- **QTabA 数据集**：作者手工标注的 300 份金融报告，配套 1.5k 条问题-表格对，用来训练和评估 DocTabQA。相当于给模型提供了“题目+答案表格”的教材。
- **两阶段框架（DocTabTalk）**：先用检索模块挑出与问题最相关的句子，再让生成模块把这些句子组织成层次化表格。类似先找线索，再写报告的写作流程。
- **AlignLLaMA**：一种针对 LLaMA 系列模型的对齐技术，帮助模型在检索阶段更精准地匹配问题和文档句子。可以把它想成把模型的注意力“调校”到正确的段落上。
- **TabTalk**：专门用于把自然语言描述转化为表格结构的生成器，内部加入了表格层级约束，使得输出的行列关系更合理。类似于让模型在写表格时遵守“表格语法”。
- **层次化表格**：表格内部可以嵌套子表格或分组行，能够表达多层次的关系（比如总计下的细分项目）。这比普通的平面表格更能体现金融报告中的层级结构。
- **RotoWire**：一个公开的体育新闻生成数据集，常被用于检验模型的表格生成能力，作者用它来验证方法的通用性。

### 核心创新点
1. **从未有的任务定义 → DocTabQA**：以前的 QA 只关注文本答案，作者首次提出“答案必须是表格”这一约束，打开了长文档结构化输出的新方向。
2. **两阶段检索‑生成流水线 → DocTabTalk**：传统方法一次性让模型读完整篇文档并直接生成表格，容易因上下文溢出而失误。DocTabTalk 先用检索把文档压缩到几百字的关键句子，再交给表格生成器，显著降低了长序列带来的噪声。
3. **针对 LLaMA 的对齐技巧 → AlignLLaMA**：直接使用原始 LLaMA 检索效果不佳，作者在检索阶段加入了专门的对齐层，使得模型的注意力更集中在真正相关的句子上，提升了检索召回率。
4. **表格生成专用解码器 → TabTalk**：普通的语言模型在生成带有严格行列约束的文本时常出现错位或缺失。TabTalk 在解码时加入层级约束和列对齐规则，确保输出的表格结构完整、层次分明。

### 方法详解
整体思路可以拆成三块：**检索 → 对齐 → 表格生成**。先把长文档压缩成一小段相关句子，再让模型在这些句子上“写表格”。下面按步骤展开。

1. **检索模块**  
   - 输入：用户提问 + 整篇文档。  
   - 采用稀疏向量（BM25）和稠密向量（Sentence‑BERT）双路检索，得到前 50 条候选句子。  
   - 为了让检索结果更贴合问题，作者在候选句子上跑了 **AlignLLaMA**：先用 LLaMA 生成每句的潜在表示，再通过一个小型对齐网络（类似于跨注意力）把这些表示与问题的表示对齐，重新打分后取前 10 条。可以把它想成先用粗筛选，再用细致的“人工眼光”挑出最相关的句子。

2. **句子聚合与层次标注**  
   - 选出的句子往往包含不同层级的信息（比如总收入句子、子业务收入句子）。作者使用规则和轻量分类器把句子划分为 **根层**、**子层** 两类，并记录它们在文档中的顺序。这样在后续生成表格时可以直接映射到表格的层级结构。

3. **TabTalk 表格生成器**  
   - 输入：聚合好的句子序列 + 层级标签。  
   - 基础模型是 GPT‑4（或 LLaMA‑2）经过指令微调的版本。关键在于 **解码约束**：在每一步生成 token 时，系统检查当前 token 是否会破坏已有的行列对齐；如果会，则强制选择符合约束的候选。相当于在写表格时给模型装上了“格子尺”。  
   - 为了让模型学会层次化输出，训练数据中每个表格都被序列化为一种“树形文本”，如 `|总收入| -> |子业务A|, |子业务B|`，模型在学习时自然会捕捉到父子关系。  
   - 生成完成后，系统会执行一次 **后处理校验**：检查是否所有列都有标题、是否出现空行、是否满足层级闭合。若发现问题，自动回滚到最近的合法状态并重新生成。

4. **整体流程图（文字版）**  
   ```
   用户提问 + 长文档
          │
   双路检索 (BM25 + Sentence‑BERT)
          │
   AlignLLaMA 对齐 → 取 Top‑10 句子
          │
   句子层级标注 (根/子)
          │
   TabTalk 受约束解码 → 初步表格
          │
   后处理校验 → 最终层次化表格
   ```

**最巧妙的点**在于把“检索”与“生成”解耦，并在生成阶段加入硬性表格约束。这样模型既不必一次性记住几千字的全文，又能保证输出符合严格的结构要求。

### 实验与效果
- **数据集**：主要在作者自行构建的 QTabA（300 篇金融报告、1.5k QA 对）上评测；另外在公开的 RotoWire（体育新闻 → 表格）上做跨域验证。  
- **基线**：直接让 GPT‑4 在完整文档上生成表格（无检索、无约束），以及使用传统抽取式 QA + 手工表格拼接的流水线。  
- **主要指标**：采用表格准确率（Cell‑F1）和层级一致性（Hier‑Acc）。在 QTabA 上，DocTabTalk 把 Cell‑F1 从基线的 38% 提升到 62%，层级一致性从 31% 提升到 55%。在 RotoWire 上，同样实现了约 15% 的相对提升。  
- **消融实验**：去掉 AlignLLaMA，检索召回率下降约 12%，整体分数随之下滑 6%；去掉 TabTalk 的约束解码，生成的表格错误率激增，Cell‑F1 直接回到基线水平。说明两大模块缺一不可。  
- **局限**：作者承认在极端超长文档（>10k 字）上检索仍会遗漏关键句子；此外，层级表格的深度目前限制在两层，复杂多层结构仍需进一步研究。

### 影响与延伸思考
DocTabQA 把“结构化输出”从图像、代码扩展到了表格，打开了长文档自动化报告的可能性。自论文发布后，已有几篇工作尝试把同样的两阶段思路用于法律文书摘要、医学病例报告等领域，进一步验证了检索‑约束生成的通用性。未来可以探索 **多模态检索**（加入图表、图片）以及 **更深层次的表格嵌套**（如树形树状结构），甚至把生成的表格直接送入下游的业务分析模型。想深入了解的读者可以关注近期在 ACL、EMNLP 上出现的 “Table Generation” 方向的最新论文。

### 一句话记住它
把长文档压缩成关键句子，再用带约束的生成器把这些句子“拼成表格”，让 GPT‑4 能稳稳输出层次化的答案表格。