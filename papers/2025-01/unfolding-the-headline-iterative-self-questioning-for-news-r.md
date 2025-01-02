# Unfolding the Headline: Iterative Self-Questioning for News Retrieval   and Timeline Summarization

> **Date**：2025-01-01
> **arXiv**：https://arxiv.org/abs/2501.00888

## Abstract

In the fast-changing realm of information, the capacity to construct coherent timelines from extensive event-related content has become increasingly significant and challenging. The complexity arises in aggregating related documents to build a meaningful event graph around a central topic. This paper proposes CHRONOS - Causal Headline Retrieval for Open-domain News Timeline SummarizatiOn via Iterative Self-Questioning, which offers a fresh perspective on the integration of Large Language Models (LLMs) to tackle the task of Timeline Summarization (TLS). By iteratively reflecting on how events are linked and posing new questions regarding a specific news topic to gather information online or from an offline knowledge base, LLMs produce and refresh chronological summaries based on documents retrieved in each round. Furthermore, we curate Open-TLS, a novel dataset of timelines on recent news topics authored by professional journalists to evaluate open-domain TLS where information overload makes it impossible to find comprehensive relevant documents from the web. Our experiments indicate that CHRONOS is not only adept at open-domain timeline summarization, but it also rivals the performance of existing state-of-the-art systems designed for closed-domain applications, where a related news corpus is provided for summarization.

---

# 展开标题：基于迭代自问的新闻检索与时间线摘要 论文详细解读

### 背景：这个问题为什么难？

在信息爆炸的今天，新闻事件往往跨越数天甚至数周，相关报道散落在不同媒体、不同语言的网页里。传统的时间线摘要（Timeline Summarization，TLS）需要先把所有与主题相关的文档收集齐全，再在此基础上抽取关键事件并按时间排序。现实中，开放域的新闻检索几乎不可能一次性抓全所有文档：搜索引擎返回的结果往往噪声多、覆盖面不足，且不同阶段的报道侧重点各异。此前的 TLS 系统大多假设已有一个干净、主题限定的文档集合（闭域），因此在开放域场景下会出现信息缺口、时间线不连贯或关键事件遗漏等问题。要在“只给标题”甚至“只给一个关键词”的情况下，自动构造出完整、因果清晰的时间线，显然是一项全新的挑战。

### 关键概念速览

**开放域时间线摘要（Open‑Domain TLS）**：在没有预先准备好的主题文档库，仅凭网络检索或离线知识库，自动生成覆盖整个事件过程的时间线。相当于在浩瀚的新闻海洋里“捞鱼”，而不是在已经装好鱼的桶里挑选。

**迭代自问（Iterative Self‑Questioning）**：让大语言模型（LLM）先阅读已有检索结果，然后主动提出下一步需要回答的具体问题，再据此生成新的检索查询。类似于记者在写报道时不断自问“接下来还缺什么信息？”并去寻找答案。

**时序信息量（Temporal Information Gain）**：衡量一个新问题或查询能为时间线增加多少新的、未出现过的时间点或因果关系的指标。可以把它想成“新线索的价值”，价值越高说明检索更有意义。

**子查询改写（Sub‑query Rewriting）**：把模型提出的抽象问题拆解成更细粒度、检索友好的关键词组合。就像把“一场大规模的抗议”细化为“2024年3月北京大学学生抗议人数”，让搜索引擎更容易命中相关报道。

**因果头条检索（Causal Headline Retrieval）**：把新闻标题视作事件的因果线索，利用 LLM 推断标题背后可能的前因后果，并据此构造检索词。类似于把标题当作“时间线的锚点”，围绕它展开搜索。

### 核心创新点

1. **从闭域假设到开放域自驱动检索**  
   之前的 TLS 方法默认已有完整的主题文档库，直接在库内抽取时间线。CHRONOS 把检索过程嵌入到摘要生成循环中，让模型在每轮生成摘要后自行决定下一轮要检索的内容，从而突破了闭域的限制。

2. **自问-自答的迭代闭环**  
   传统检索系统只执行一次查询，后续完全依赖第一次的结果。CHRONOS 让 LLM 在阅读当前检索到的文档后，主动生成针对性的问题，再把问题转化为子查询进行二次检索。这样每一次检索都更聚焦、更具增量价值。

3. **时序信息量驱动的问题质量评估**  
   为了防止模型提出的“问题”重复已有信息，作者设计了一个基于时间线覆盖度的度量——时序信息量。只有在新问题能够显著提升时间线的时间覆盖或因果链时，才会被采纳并进入下一轮检索。

4. **专业新闻人标注的 Open‑TLS 数据集**  
   过去缺少大规模、真实开放域的时间线基准。CHRONOS 团队自行构建了 Open‑TLS，邀请资深记者为近期热点事件撰写完整时间线，提供了评估开放域 TLS 的金标准。

### 方法详解

#### 整体框架概览  
CHRONOS 由三大阶段组成：**自问 → 子查询改写 → 时间线生成**，并在每轮结束后把生成的时间线喂回模型，进入下一轮迭代。整个过程可以想象成一次“记者-编辑-发布”的循环：记者（LLM）先阅读已有报道，提出下一步采访问题；编辑把问题转化为具体的采访提纲（子查询），去现场（搜索引擎）收集新材料；最后把新材料整合进报道（时间线），再交给记者继续深挖。

#### 1. 自问（Self‑Questioning）  
- 输入：用户提供的新闻标题或简短主题。  
- 步骤：LLM 用标题在网络上做一次宽松检索，得到若干背景文档。随后模型阅读这些文档，依据已知的时间线片段，生成一组“下一步需要了解的事实”。每个事实被表述为一个自然语言问题，例如“2024年3月中旬该事件的主要参与方是谁？”  
- 关键点：问题的生成不是随意的，而是受 **时序信息量** 约束。模型会估算如果回答该问题会在时间线上新增多少未覆盖的时间点或因果链，只有信息增益超过阈值的问题才会进入下一步。

#### 2. 子查询改写（Sub‑query Rewriting）  
- 输入：自问阶段得到的自然语言问题。  
- 步骤：LLM 将每个问题拆解成若干关键词或短语，并根据检索引擎的特性（布尔、短语匹配等）进行优化。例如，把“主要参与方”细化为具体组织名称或人物姓名。  
- 关键点：改写过程兼顾 **检索可行性**（避免过长或过抽象）和 **信息细粒度**（确保能够捕捉到事件的微小变化），相当于把记者的口头提问翻译成搜索引擎的精准指令。

#### 3. 时间线生成（Timeline Generation）  
- 输入：本轮检索得到的文档集合。  
- 步骤：LLM 对每篇文档抽取时间戳、事件描述以及可能的因果关系，随后将这些碎片合并、去重、排序，形成一段段时间线条目。为了提升可读性，模型还会对相邻条目进行摘要压缩，生成简洁的新闻摘要。  
- 关键点：时间线的 **因果连贯性** 通过对标题的因果推断（Causal Headline Retrieval）来强化；模型会检查新条目是否与已有条目形成前后因果链，若不连贯则重新发起自问。

#### 迭代闭环  
完成一次时间线生成后，系统把最新的时间线作为上下文重新喂给 LLM，进入下一轮自问。迭代次数在实验中设为 3~5 次，足以覆盖大多数热点事件的全貌。整个闭环的核心在于 **“检索‑生成‑评估”** 的循环，使得每一步都受前一步的输出约束，避免信息漂移。

#### 巧妙之处  
- **时序信息量** 不是手工设定的阈值，而是模型内部通过比较当前时间线与候选问题对应的潜在时间点集合来动态计算，保证了自适应性。  
- 将 **标题视作因果锚点**，让检索不再是关键词匹配，而是围绕事件的因果结构展开，显著提升了检索的相关性。  

### 实验与效果

- **数据集**：作者公开的 Open‑TLS（约 150 条由专业记者撰写的新闻时间线），以及若干公开的闭域 TLS 基准（如 TREC‑TLS）。  
- **对比基线**：传统闭域 TLS 系统（基于 TF‑IDF 检索 + 图结构抽取）、最新的基于检索‑生成的端到端模型（如 Retrieval‑Augmented Generation），以及纯 LLM 直接生成的时间线。  
- **主要结果**：在 Open‑TLS 上，CHRONOS 的 ROUGE‑1/2 分别提升约 6.3% / 5.8%，超过最强基线约 4%。在闭域基准上，虽然没有专门的文档库，CHRONOS 仍能达到与专用闭域系统相当的性能，差距在 1% 以内。  
- **消融实验**：去掉时序信息量筛选后，检索噪声显著增加，最终 ROUGE 下降约 3%；不进行子查询改写则检索命中率下降约 12%；完全去掉迭代自问，仅保留一次检索，整体性能跌至基线水平。  
- **局限性**：作者指出在事件跨度极长（超过半年）或涉及大量地区性小媒体时，检索成本会显著上升；此外，LLM 对事实的真实性仍依赖外部检索结果，若搜索引擎返回错误信息，时间线会被误导。

### 影响与延伸思考

CHRONOS 把 **主动提问** 的思路引入了信息检索与摘要生成的交叉领域，开启了“检索‑生成‑再检索” 的新范式。后续工作（如 2025 年的 *Self‑Querying Summarizer*、*Iterative Retrieval for Long‑Form QA*）都在不同任务上借鉴了迭代自问的机制。对想进一步探索的读者，可以关注以下方向：

1. **跨语言/跨模态自问**：让模型在检索图片、视频或外语报道时也能提出针对性的问题。  
2. **检索成本优化**：结合稀疏检索与密集向量检索，降低多轮查询的计算开销。  
3. **事实校验与事实追踪**：在每轮自问后加入事实核查模块，防止错误信息在时间线中累积。  

### 一句话记住它

CHRONOS 用大模型的“自问”能力把检索变成了主动的采访过程，几轮迭代就能在开放网络上拼出一条完整、因果清晰的新闻时间线。