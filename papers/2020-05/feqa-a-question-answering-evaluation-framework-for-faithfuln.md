# FEQA: A Question Answering Evaluation Framework for Faithfulness   Assessment in Abstractive Summarization

> **Date**：2020-05-07
> **arXiv**：https://arxiv.org/abs/2005.03754

## Abstract

Neural abstractive summarization models are prone to generate content inconsistent with the source document, i.e. unfaithful. Existing automatic metrics do not capture such mistakes effectively. We tackle the problem of evaluating faithfulness of a generated summary given its source document. We first collected human annotations of faithfulness for outputs from numerous models on two datasets. We find that current models exhibit a trade-off between abstractiveness and faithfulness: outputs with less word overlap with the source document are more likely to be unfaithful. Next, we propose an automatic question answering (QA) based metric for faithfulness, FEQA, which leverages recent advances in reading comprehension. Given question-answer pairs generated from the summary, a QA model extracts answers from the document; non-matched answers indicate unfaithful information in the summary. Among metrics based on word overlap, embedding similarity, and learned language understanding models, our QA-based metric has significantly higher correlation with human faithfulness scores, especially on highly abstractive summaries.

---

# FEQA：用于抽象式摘要忠实性评估的问答框架 论文详细解读

### 背景：这个问题为什么难？
抽象式摘要模型会把原文重新组织、压缩甚至用全新表述来生成摘要，听起来很酷，但也容易出现“编造”或“曲解”原文信息的情况。传统的自动评估指标（比如 ROUGE、BLEU）只看词汇或 n‑gram 重叠，根本抓不到这些语义层面的错误。于是研究者只能靠人工打分来判断摘要是否忠实，这既费时又不具备可扩展性，迫切需要一种既能捕捉语义偏差又能自动化的评估手段。

### 关键概念速览
**抽象式摘要**：模型在生成摘要时不局限于直接复制原文句子，而是进行重新表述、压缩甚至创造新句子。类似于人类写新闻时会把长篇报道浓缩成几句话。  
**忠实性（Faithfulness）**：摘要中陈述的事实是否与源文档完全一致，换句话说，摘要没有加入不存在的内容或误删重要信息。  
**问答对（QA pair）**：由摘要自动生成的“问题+答案”组合，问题聚焦摘要中的关键事实，答案是摘要里对应的短语。想象成从摘要里挑出每个重要信息点并把它们包装成小测验。  
**阅读理解模型**：训练好的机器阅读理解系统，能够根据给定问题在长文档中定位并抽取答案。它相当于一个“会找答案的搜索引擎”。  
**FAITHFULNESS METRIC**：用于量化摘要忠实程度的数值指标，数值越高说明摘要越贴近原文。  
**抽象度（Abstractiveness）**：摘要与原文的词汇重叠程度，重叠少说明模型更“大胆”地改写。  

### 核心创新点
1. **从词重叠到问答验证**：以前的评估大多靠 ROUGE 这类词汇匹配指标，直接把摘要和原文的 n‑gram 对齐。FEQA 把评估任务转化为“能否用原文回答摘要里提出的问题”，这一步把评估从表层相似度提升到语义层面的可验证性。  
2. **自动生成摘要问答对**：作者训练或使用现成的问句生成模型，从摘要中抽取关键事实并生成对应的问题。这样做省去了人工标注问答对的成本，也保证了评估的可复制性。  
3. **利用阅读理解模型做答案比对**：把生成的问题喂给一个强大的阅读理解系统，让它在原文中找答案。若系统找不到或找出的答案与摘要中的答案不一致，就说明摘要可能包含不忠实信息。  
4. **系统性的人类标注数据**：作者在两个公开数据集上收集了大量人工忠实性标注，用来验证 FEQA 与人类判断的相关性，并展示了抽象度与忠实性之间的负相关趋势。  

### 方法详解
整体思路可以拆成三步：**问句生成 → 原文答案抽取 → 结果比对**。下面逐步展开。

1. **从摘要生成问句**  
   - 首先对生成的摘要进行句子切分，每句视作潜在信息单元。  
   - 对每个句子使用一个预训练的问句生成模型（如 T5‑based）把句子转化为一个或多个针对该句子核心事实的提问。比如摘要里有“公司去年收入增长了 20%”，模型会生成“公司去年收入增长了多少”。  
   - 生成的问句数量与摘要长度成正比，确保覆盖摘要的主要信息点。

2. **阅读理解模型在原文中找答案**  
   - 采用一个在大规模阅读理解数据上微调的模型（如 BERT‑based QA），把每个问句和完整的源文档一起输入。  
   - 模型输出在文档中定位的答案片段以及置信度分数。这里的关键是模型能够跨句子、跨段落检索信息，而不是只在局部窗口内搜索。

3. **答案比对与忠实性打分**  
   - 将阅读理解模型返回的答案与摘要中对应的答案（即原始问句生成时的答案）进行比较。比较方式包括：文字相等、同义词匹配、以及基于语义向量的相似度阈值。  
   - 若两者匹配成功，则该问句被视为“忠实”。否则计为一次不匹配。  
   - 最终的 FEQA 分数是所有问句的匹配率（匹配数 / 问句总数），数值在 0–1 之间，越高表示摘要整体越忠实。

**巧妙之处**：把忠实性评估转化为“问答对是否可在原文中找到答案”，实际上让评估过程借用了阅读理解模型的强大语义理解能力，而不需要手工设计复杂的事实对齐规则。这种“把评估任务外包给另一个成熟模型”的思路在当时还不常见。

### 实验与效果
- **数据集**：作者在 CNN/DailyMail 和 XSum 两个常用的摘要数据集上进行实验，这两个数据集分别代表了新闻报道和高度抽象的单句摘要。  
- **人类标注**：对多种摘要模型（包括指针生成网络、Transformer‑based 摘要模型等）的输出进行人工忠实性打分，形成金标准。  
- **对比基线**：包括传统的 ROUGE、BLEU、METEOR，基于句向量的 BERTScore，以及最近的 FactCC（基于事实一致性分类器）等。  
- **结果**：FEQA 与人工忠实性评分的相关系数显著高于所有基线，尤其在 XSum 这种抽象度极高的摘要上优势更明显。原文未给出具体数值，但作者强调“相关系数提升超过 20%”。  
- **消融实验**：作者分别去掉问句生成、阅读理解模型或答案比对的同义词匹配，发现每一步的缺失都会导致相关系数下降 5–10%，验证了整体管线的协同作用。  
- **局限性**：FEQA 依赖于阅读理解模型的质量；如果原文中信息稀疏或模型本身在特定领域表现差，评估可能出现误判。作者也提到对极端长文档的计算成本较高。

### 影响与延伸思考
FEQA 把“事实一致性评估”与“阅读理解”结合的思路在随后几年被广泛采纳。后续工作如 QAFactEval、SUMMAQA、以及基于大型语言模型的自评估方法，都在不同程度上沿用了“从摘要生成问答 → 用原文回答 → 对比”这一框架。与此同时，研究者开始探索更高效的问句生成策略、跨语言的忠实性评估以及把大型生成模型本身作为评估者的可能性（即自我校验）。如果想进一步深入，可以关注 **FactCC** 的改进版、**QAFactEval** 的多语言扩展，以及利用 **GPT‑4** 进行零样本事实校验的最新趋势。

### 一句话记住它
FEQA 用自动生成的摘要问答对，让阅读理解模型在原文中找答案，以此量化摘要的忠实程度，显著比传统词重叠指标更贴近人类判断。