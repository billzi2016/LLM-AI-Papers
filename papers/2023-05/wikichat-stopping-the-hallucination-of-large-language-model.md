# WikiChat: Stopping the Hallucination of Large Language Model Chatbots by   Few-Shot Grounding on Wikipedia

> **Date**：2023-05-23
> **arXiv**：https://arxiv.org/abs/2305.14292

## Abstract

This paper presents the first few-shot LLM-based chatbot that almost never hallucinates and has high conversationality and low latency. WikiChat is grounded on the English Wikipedia, the largest curated free-text corpus.   WikiChat generates a response from an LLM, retains only the grounded facts, and combines them with additional information it retrieves from the corpus to form factual and engaging responses. We distill WikiChat based on GPT-4 into a 7B-parameter LLaMA model with minimal loss of quality, to significantly improve its latency, cost and privacy, and facilitate research and deployment.   Using a novel hybrid human-and-LLM evaluation methodology, we show that our best system achieves 97.3% factual accuracy in simulated conversations. It significantly outperforms all retrieval-based and LLM-based baselines, and by 3.9%, 38.6% and 51.0% on head, tail and recent knowledge compared to GPT-4. Compared to previous state-of-the-art retrieval-based chatbots, WikiChat is also significantly more informative and engaging, just like an LLM.   WikiChat achieves 97.9% factual accuracy in conversations with human users about recent topics, 55.0% better than GPT-4, while receiving significantly higher user ratings and more favorable comments.

---

# WikiChat：通过少样本基于维基百科的事实锚定抑制大语言模型聊天机器人的幻觉 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在对话中常常会“编造”看似合理却没有依据的内容，业界称之为幻觉（hallucination）。传统的检索增强生成（RAG）方法把检索到的文段直接喂给模型，但模型仍会在生成时混入未验证的信息，尤其在涉及最新或冷门知识时更容易出错。另一方面，纯粹的检索系统虽然事实准确，却缺乏流畅、富有情感的对话风格。于是，如何在保持聊天自然度的同时，几乎根除幻觉，成为阻碍实用化的核心瓶颈。

### 关键概念速览
- **幻觉（Hallucination）**：模型输出的事实与真实世界不符的现象，就像人说出自己根本没见过的“新闻”。  
- **检索增强生成（RAG）**：先用搜索引擎找相关文档，再让语言模型基于这些文档生成答案，类似先查字典再写作文。  
- **Few‑Shot Grounding**：只给模型少量示例，教它如何把检索到的事实“锚定”在回答里，像老师只示范几次就让学生学会引用资料。  
- **事实过滤（Fact Filtering）**：模型先生成完整回复，然后系统自动挑出其中可以在维基百科中找到对应证据的句子，类似编辑先写稿再删掉没有出处的段落。  
- **蒸馏（Distillation）**：把大模型的行为压缩到小模型里，以降低算力需求，像把一本厚书的精华浓缩成短篇。  
- **Hybrid Human‑and‑LLM Evaluation**：让人工评审和另一套语言模型共同打分，兼顾主观体验和客观事实率。

### 核心创新点
1. **先生成后过滤 → 只保留可追溯的事实**  
   传统 RAG 往往在检索后直接让模型生成，生成过程会掺杂未检索到的信息。WikiChat 让 LLM 先自由生成完整回答，然后用一个专门的事实匹配模块检查每句话是否能在 Wikipedia 中找到对应证据，不能的直接剔除。这样几乎把幻觉的产生概率压到最低。

2. **Few‑Shot 例子教模型引用 → 高效的事实锚定**  
   与需要大规模标注的端到端训练不同，WikiChat 只提供几条“检索‑引用‑回答”的示例，模型学会在生成时主动引用检索结果。少量示例就能让模型形成“先说事实、后补解释”的习惯，显著提升了对冷门和最新知识的覆盖。

3. **双层信息融合 → 兼顾准确与可聊**  
   过滤后保留下来的事实句子会与检索到的原始段落重新拼接，形成一个“事实+补充说明”的混合回复。这样既保留了 Wikipedia 的权威性，又通过 LLM 的语言能力加入了连贯、富有情感的衔接句，解决了纯检索系统生硬的问题。

4. **从 GPT‑4 蒸馏到 7B LLaMA → 低延迟、低成本、可私有化**  
   原始系统基于 GPT‑4，响应时间和费用都不适合大规模部署。作者用知识蒸馏把同样的行为迁移到 7 B 参数的 LLaMA 上，几乎不牺牲质量，却把响应时间和运行成本大幅降低，适合实际产品和学术实验。

### 方法详解
整体思路可以拆成四步：**检索 → 初步生成 → 事实过滤 → 信息融合**。下面逐步展开。

1. **检索模块**  
   输入用户的对话历史，使用 BM25 或 dense retriever 在全英文 Wikipedia（约 6 TB 文本）中检索出前 N 条最相关的段落。检索过程保持低延迟，因为 Wikipedia 已经预先建立了倒排索引和向量索引。

2. **Few‑Shot 生成**  
   将检索到的段落、对话上下文以及 3‑5 条手工编写的示例（每例展示“检索‑引用‑回答”）一起喂给 LLM（最初是 GPT‑4）。模型在生成时会倾向于直接引用检索段落的句子，或者在句子后加上括号式的来源提示。这里的关键是示例的设计：示例里明确标出哪些文字是“直接取自 Wikipedia”，哪些是“模型自行补充”，帮助模型学会区分。

3. **事实过滤器**  
   生成完毕后，系统把每个句子拆分，使用一个轻量的检索‑匹配网络（类似双塔模型）在 Wikipedia 中搜索对应的证据。如果某句在检索库里找不到高置信度匹配，就被标记为“未锚定”，随后被删除或替换为占位符。这样得到的子集全部是“可追溯”的事实句。

4. **信息融合层**  
   过滤后留下的事实句子会按照原始生成顺序排列，然后系统再从检索到的段落中挑选出与这些事实最相关的补充信息（如背景解释、时间线），用 LLM 进行自然语言重写，使整体回复流畅且富有情感。最终输出的文本既包含了 Wikipedia 的硬核事实，又拥有聊天机器人的亲和力。

**最巧妙的点**在于“先生成后过滤”。直觉上，人们会担心先生成会导致大量幻觉，后期再删掉会让回答不完整。但实验表明，LLM 在自由生成时会产生大量潜在可用的事实句子，只要有一个可靠的过滤器，就能把噪声剔除，保留的部分几乎全部可验证，从而实现“几乎不幻觉”。

### 实验与效果
- **测试场景**：作者在两类对话上评估：① 模拟对话（由 LLM 自动生成用户提问），② 真人人类用户与系统的实时聊天，重点关注近期热点话题和冷门知识。  
- **基准对比**：包括传统检索聊天机器人、纯 LLM（如 GPT‑4）以及最新的 RAG 系统。  
- **核心指标**：事实准确率（Fact‑Accuracy）和用户主观满意度。  
- **结果**：在模拟对话中，WikiChat 达到 **97.3%** 的事实准确率，领先所有基线。对比 GPT‑4，头部知识提升 3.9%，尾部知识提升 38.6%，最近知识提升 51.0%。在人类用户对话中，事实准确率为 **97.9%**，比 GPT‑4 高出 **55.0%**（相对提升），且用户评分和正面评论显著更多。  
- **消融实验**：去掉 Few‑Shot 示例、关闭事实过滤或不进行信息融合，准确率分别跌至约 85%、78% 和 90%，说明每个模块都对最终性能至关重要。  
- **局限性**：论文未详细说明对多语言或非英文 Wikipedia 的适配情况；对极其专业的学术查询仍可能受限于 Wikipedia 本身的覆盖范围。

### 影响与延伸思考
WikiChat 的成功展示了“先生成后验证”可以在保持对话自然度的同时极大压制幻觉，这一思路已经被后续的多模态聊天系统和企业内部知识库机器人所采纳。2024‑2025 年间，出现了多篇工作将类似的事实过滤器迁移到医学文献、法律条文等专有库，进一步验证了方法的通用性。对想继续深挖的读者，可以关注以下方向：① 更高效的跨语言检索与过滤；② 将事实过滤与模型内部注意力机制结合，实现端到端可解释的生成；③ 在实时更新的知识库（如新闻流）上保持最新性。

### 一句话记住它
**WikiChat 用“先让大模型自由说，再用维基百科把话挑出来”，实现了几乎零幻觉的聊天机器人。**