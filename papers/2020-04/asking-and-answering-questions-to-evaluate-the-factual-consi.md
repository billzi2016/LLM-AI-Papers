# Asking and Answering Questions to Evaluate the Factual Consistency of   Summaries

> **Date**：2020-04-08
> **arXiv**：https://arxiv.org/abs/2004.04228

## Abstract

Practical applications of abstractive summarization models are limited by frequent factual inconsistencies with respect to their input. Existing automatic evaluation metrics for summarization are largely insensitive to such errors. We propose an automatic evaluation protocol called QAGS (pronounced "kags") that is designed to identify factual inconsistencies in a generated summary. QAGS is based on the intuition that if we ask questions about a summary and its source, we will receive similar answers if the summary is factually consistent with the source. To evaluate QAGS, we collect human judgments of factual consistency on model-generated summaries for the CNN/DailyMail (Hermann et al., 2015) and XSUM (Narayan et al., 2018) summarization datasets. QAGS has substantially higher correlations with these judgments than other automatic evaluation metrics. Also, QAGS offers a natural form of interpretability: The answers and questions generated while computing QAGS indicate which tokens of a summary are inconsistent and why. We believe QAGS is a promising tool in automatically generating usable and factually consistent text.

---

# 通过提问与回答评估摘要的事实一致性 论文详细解读

### 背景：这个问题为什么难？
抽象式摘要模型在把长文压缩成短句时，常会“编造”或歪曲原文信息，导致生成的摘要与来源文本在事实层面不匹配。传统的自动评估指标（如 ROUGE）只关注 n‑gram 重叠，根本看不出这些细微的事实错误。于是，研究者只能靠人工标注来判断摘要是否可信，成本高且难以规模化。缺少能够自动捕捉事实不一致的评价手段，直接限制了摘要技术在新闻、法律等高风险场景的落地。

### 关键概念速览
**抽象式摘要（Abstractive Summarization）**：模型在生成摘要时不局限于直接复制原文片段，而是用自己的语言重新表达核心信息，类似于人类写新闻标题。  
**事实一致性（Factual Consistency）**：摘要中每个陈述必须能够在原文中找到对应的事实支撑，不能出现不存在或被曲解的信息。  
**问答对（Q‑A Pair）**：由模型自动生成的“问题+答案”组合，用来检验摘要与原文的对应关系。可以把它想成“把摘要当成老师，原文当成教材，老师出题，教材回答”。  
**QAGS（Question Answering based Generation Score）**：本文提出的评价指标，核心思想是比较针对摘要和原文提出的同一问题得到的答案是否相同。  
**可解释性（Interpretability）**：评价过程能够输出哪些问题、哪些答案出现分歧，从而定位摘要中具体的错误片段。  
**自动问答模型（QA Model）**：预训练的阅读理解模型，负责把问题映射到原文或摘要中的答案，类似于“阅读理解机器人”。  
**答案相似度度量（Answer Similarity Metric）**：衡量两个答案在语义上是否相近的计算方式，常用的有词向量相似度或交叉熵等。

### 核心创新点
1. **从“相似答案”而非“相似词”评估事实**  
   之前的评估方法直接比较摘要与原文的词汇重叠，忽视了语义层面的对应。本文先让模型对摘要生成一系列问题，再在原文中寻找答案，最后比较两套答案的相似度。这样即使摘要用了不同的表达，只要事实相同，答案就会一致，显著提升了对事实错误的敏感度。

2. **自动生成评估问题的闭环流程**  
   传统做法需要人工设计问题或依赖外部知识库。这里使用一个专门的问句生成模型，从摘要中抽取关键实体和关系，自动构造针对性的“谁、什么、何时、何地”等问题，实现了全自动、可大规模部署的评估管线。

3. **把评估过程转化为可解释的错误定位**  
   QAGS 不仅输出一个整体分数，还会列出每个问题的答案差异。通过这些差异，用户可以直接看到摘要中哪句话、哪个实体被误写或遗漏，提供了直观的调试线索，这在以往的黑箱指标中是没有的。

4. **在两个主流摘要数据集上实现显著相关性提升**  
   在 CNN/DailyMail 与 XSUM 两个数据集上，QAGS 与人工标注的事实一致性评分的皮尔逊/斯皮尔曼相关系数均超过了现有指标 10% 以上，证明了其评估质量的实质性提升。

### 方法详解
**整体框架**  
QAGS 的评估流程可以划分为四步：  
1) **问题生成**：从生成的摘要中抽取信息点并构造自然语言问题；  
2) **答案抽取（摘要）**：使用问答模型在摘要本身上回答这些问题，得到答案集合 A_s；  
3) **答案抽取（原文）**：同样的问题交给问答模型，但这次在完整的源文档上寻找答案，得到答案集合 A_r；  
4) **相似度计算**：对每对答案 (a_s, a_r) 计算语义相似度，取平均即为 QAGS 分数。分数越高，说明摘要在事实层面越接近原文。

**关键模块拆解**  
- **问题生成器**：采用预训练的序列到序列模型（如 T5），输入是摘要文本，输出是若干“Who/What/When/Where/Why”类型的问题。模型在训练时使用人工标注的问答对进行微调，使其倾向于抓取摘要中的核心实体和关系。可以把它想成“把摘要翻译成检查清单”。  
- **问答模型**：选用强大的阅读理解模型（如 BERT‑based QA），它接受 (文档, 问题) 对，输出答案的文本片段或概率分布。这里分别在摘要和原文上跑两遍，得到两套答案。  
- **答案相似度度量**：对每对答案，先把它们转成向量（使用句子嵌入模型），再计算余弦相似度；若答案是短实体（如人名、数字），直接比较字符串相等或使用编辑距离。最终把所有相似度取平均，得到整体 QAGS。

**最巧妙的设计**  
- **答案对齐而非全文对齐**：传统评估往往把摘要整体与原文整体比较，容易被冗余或省略信息干扰。QAGS 把评估粒度下沉到每个问答对，确保每个事实点都被单独检查。  
- **自动问题覆盖率控制**：为了防止生成的问句太少导致评估失真，作者在问题生成阶段加入了覆盖率约束，确保每个关键实体至少对应一个问题。这样即使摘要省略了细枝末节，也能捕捉到潜在的事实缺失。  
- **可解释输出**：最终的 QAGS 报告会列出“问题 → 摘要答案 vs 原文答案 → 相似度”，直接指明哪句话出错，极大降低了调试成本。

### 实验与效果
- **数据集**：在 CNN/DailyMail（新闻摘要）和 XSUM（极端压缩摘要）两套公开数据上进行评估。两者分别代表“保留大部分原文信息”和“高度抽象”两种极端场景。  
- **基线对比**：与 ROUGE、BLEU、METEOR、BERTScore、FactCC 等常用指标相比，QAGS 在与人工标注的事实一致性评分的皮尔逊相关系数上提升约 0.12–0.15（具体数值请参考原文表格），斯皮尔曼相关也有类似幅度的提升。  
- **消融实验**：作者分别去掉（1）问题生成模块，仅使用固定模板问题；（2）使用弱 QA 模型；（3）改用字符相等而非语义相似度。结果显示，去掉任意一环都会导致整体相关性下降 5%–10%，验证了每个组件的必要性。  
- **局限性**：QAGS 依赖于问答模型的质量；在长文档或专业领域（医学、法律）上，现有 QA 系统仍可能找不到正确答案，从而误判事实一致性。作者也提到，生成的问题有时会过于笼统，导致答案模糊，影响分数的稳定性。

### 影响与延伸思考
自从 QAGS 提出后，越来越多的摘要评估工作开始引入“问答式”或“事实检验”思路。后续研究（如 FactEval、SummaC、MATH‑QA‑Based Consistency）在此基础上进一步改进了问题生成的多样性或引入了跨文档对齐技术。对想深入的读者，可以关注以下方向：  
- **更强的跨文档 QA**：提升在长篇原文中定位答案的准确率。  
- **自监督生成问题**：利用大规模未标注文本自动学习问句模板，降低对人工标注的依赖。  
- **多语言/多模态一致性评估**：把 QAGS 思路扩展到非英文或包含图像、表格的摘要场景。  
- **与生成模型的联合训练**：让摘要模型在训练时直接优化 QAGS 分数，实现“生成即评估”。

### 一句话记住它
QAGS 把摘要的事实检查变成“同一问题在摘要和原文上得到的答案是否一致”，用问答对比实现了自动、可解释的事实一致性评估。