# Long-form factuality in large language models

> **Date**：2024-03-27
> **arXiv**：https://arxiv.org/abs/2403.18802

## Abstract

Large language models (LLMs) often generate content that contains factual errors when responding to fact-seeking prompts on open-ended topics. To benchmark a model's long-form factuality in open domains, we first use GPT-4 to generate LongFact, a prompt set comprising thousands of questions spanning 38 topics. We then propose that LLM agents can be used as automated evaluators for long-form factuality through a method which we call Search-Augmented Factuality Evaluator (SAFE). SAFE utilizes an LLM to break down a long-form response into a set of individual facts and to evaluate the accuracy of each fact using a multi-step reasoning process comprising sending search queries to Google Search and determining whether a fact is supported by the search results. Furthermore, we propose extending F1 score as an aggregated metric for long-form factuality. To do so, we balance the percentage of supported facts in a response (precision) with the percentage of provided facts relative to a hyperparameter representing a user's preferred response length (recall).   Empirically, we demonstrate that LLM agents can outperform crowdsourced human annotators - on a set of ~16k individual facts, SAFE agrees with crowdsourced human annotators 72% of the time, and on a random subset of 100 disagreement cases, SAFE wins 76% of the time. At the same time, SAFE is more than 20 times cheaper than human annotators. We also benchmark thirteen language models on LongFact across four model families (Gemini, GPT, Claude, and PaLM-2), finding that larger language models generally achieve better long-form factuality. LongFact, SAFE, and all experimental code are available at https://github.com/google-deepmind/long-form-factuality.

---

# 大语言模型的长文本事实性 论文详细解读

### 背景：这个问题为什么难？
在开放式问答里，大语言模型（LLM）常常能写出流畅的长段落，却时不时掺杂不实信息。过去的评估大多聚焦于单句或短答案的准确性，缺少对整篇文章中每个事实的细粒度检查。人工标注虽然能捕捉错误，但成本高、速度慢，难以支撑大规模模型对比。于是，缺少一种既可靠又经济的方式来衡量 LLM 在长文本场景下的事实性，成为制约模型改进的瓶颈。

### 关键概念速览
**LongFact**：由 GPT‑4 自动生成的包含数千个跨 38 个主题的问题集合，用来测评模型的长文本事实性。相当于一套“长篇考试卷”。  
**SAFE（Search‑Augmented Factuality Evaluator）**：一种让 LLM 自己拆解答案、检索证据、判断真伪的评估管线。把模型当成“审稿人”，先把文章拆成小块，再去 Google 搜索验证。  
**事实拆解**：把一段生成的文字分割成独立的陈述句或事实单元，类似把一篇新闻稿拆成每条新闻点。  
**多步推理**：模型在判断一个事实是否成立时，会先生成检索关键词、发送搜索请求、读取搜索摘要、最后给出支持/反驳的结论。  
**扩展 F1**：在传统的精确率/召回率基础上，引入用户期望的回答长度（超参数）来平衡“提供多少事实”和“这些事实有多可靠”。  

### 核心创新点
1. **从人工标注转向 LLM‑agent 自动评估**：过去的长文本事实性评测依赖大量人工标注，费用高且难以扩展。本文让 LLM 本身充当评审员，利用搜索引擎做外部验证，实现了成本下降 20 倍以上的效果。  
2. **事实拆解 + 搜索验证的两层结构**：先把答案拆成细粒度事实，再对每个事实发起针对性的搜索查询。相比直接让模型判断整段文字的真实性，这种细化让错误更容易被定位，也让搜索更精准。  
3. **基于搜索结果的多步推理流程**：SAFE 让模型在每一步都显式输出思考过程（检索词 → 搜索摘要 → 支持度），类似人类查证的步骤，提升了判断的可靠性。  
4. **面向长文本的扩展 F1 指标**：传统 F1 只衡量已标注事实的精确率和召回率，忽略回答长度。作者引入用户偏好的长度阈值，使得指标能够同时奖励“多提供有价值信息”和“高比例真实信息”。  

### 方法详解
整体思路可以拆成四个阶段：**问题准备 → 生成答案 → 事实拆解 → 搜索‑推理评估**。

1. **问题准备（LongFact）**  
   - 使用 GPT‑4 按照预设的 38 个主题生成数千条开放式提问。每条提问都要求模型给出详细、结构化的长答案（如步骤、背景、例子），确保答案里包含足够多的可验证事实。

2. **答案生成**  
   - 将 LongFact 中的问题分别喂给待评测的 LLM（如 Gemini、Claude 等），得到完整的长文本回答。这里不做任何后处理，保持原始生成质量。

3. **事实拆解**  
   - SAFE 调用一个专门的 LLM（通常是同一系列的较小模型）来把长答案切分成独立的事实单元。拆解规则包括：每个完整的陈述句、数字统计、因果关系等都视为一个事实。拆解过程会输出类似 “Fact 1: …”，方便后续追踪。

4. **搜索‑推理评估**  
   - 对每个事实，SAFE 按以下步骤操作：  
     a. **生成检索词**：模型根据事实内容生成 1‑2 条简短的搜索查询，力求覆盖核心概念。  
     b. **调用 Google Search**：把检索词发送给搜索引擎，获取前几条摘要或网页标题。  
     c. **摘要阅读**：模型读取搜索结果的摘要，提取与事实相关的关键信息。  
     d. **支持判定**：模型基于检索到的证据给出 “支持” 或 “不支持” 的二元标签，并给出置信度分数。整个过程会在模型的输出中留下思考链，类似 “我搜索了…，结果显示…，因此该事实成立”。  
   - 最后，SAFE 统计所有事实的支持比例（精确率）和已提供事实占用户期望长度的比例（召回率），用扩展 F1 合并为整体分数。

**最巧妙的点**在于让模型自行完成检索词生成和证据阅读，而不是让评估者手动挑选关键词。这样既保持了评估的自动化，又利用了搜索引擎的实时知识，弥补了 LLM 训练数据的时效性限制。

### 实验与效果
- **数据集**：使用 LongFact 的全部约 16 000 条事实作为评估基准。  
- **基线**：传统人工标注（众包）以及直接让 LLM 通过内部知识判断（不使用搜索）。  
- **主要结果**：SAFE 与众包标注在 72% 的事实上达成一致；在随机抽取的 100 例标注冲突中，SAFE 获得 76% 的胜率，说明它在多数情况下比人类更可靠。成本方面，SAFE 的每条事实评估费用约为人工的 1/20。  
- **模型对比**：在四大模型家族（Gemini、GPT、Claude、PaLM‑2）中，模型规模越大，LongFact 上的扩展 F1 越高，验证了“更大模型更擅长保持事实性”。  
- **消融实验**：去掉搜索步骤或改用固定检索词会导致整体准确率下降约 10%–15%，说明搜索‑推理环节是提升评估质量的关键。  
- **局限**：SAFE 依赖于搜索引擎的可访问性和返回质量；对极其专业或最新的细节（搜索结果稀缺）仍可能误判。论文也提到对多语言或低资源领域的评估尚未充分验证。

### 影响与延伸思考
这篇工作打开了“让 LLM 自己审稿”的新思路，随后出现的研究纷纷在此基础上加入更强的检索器、跨模态证据（如图像、表格）或利用专用知识库来提升可靠性。对想进一步探索的读者，可以关注以下方向：  
- **检索增强生成（RAG）** 与评估的统一框架，尝试让模型在生成时就引用实时证据。  
- **多语言 SAFE**：将搜索‑推理流程迁移到非英文搜索引擎，评估跨语言事实性。  
- **证据可解释性**：把模型的检索‑推理链可视化，帮助用户理解为何某个事实被标记为错误。  

### 一句话记住它
让大语言模型自己拆事实、上网查证，再用搜索支持度算分——自动化、低成本、比人工更靠谱的长文本事实性评估。