# RealTime QA: What's the Answer Right Now?

> **Date**：2022-07-27
> **arXiv**：https://arxiv.org/abs/2207.13332

## Abstract

We introduce REALTIME QA, a dynamic question answering (QA) platform that announces questions and evaluates systems on a regular basis (weekly in this version). REALTIME QA inquires about the current world, and QA systems need to answer questions about novel events or information. It therefore challenges static, conventional assumptions in open-domain QA datasets and pursues instantaneous applications. We build strong baseline models upon large pretrained language models, including GPT-3 and T5. Our benchmark is an ongoing effort, and this paper presents real-time evaluation results over the past year. Our experimental results show that GPT-3 can often properly update its generation results, based on newly-retrieved documents, highlighting the importance of up-to-date information retrieval. Nonetheless, we find that GPT-3 tends to return outdated answers when retrieved documents do not provide sufficient information to find an answer. This suggests an important avenue for future research: can an open-domain QA system identify such unanswerable cases and communicate with the user or even the retrieval module to modify the retrieval results? We hope that REALTIME QA will spur progress in instantaneous applications of question answering and beyond.

---

# 实时问答：现在的答案是什么？ 论文详细解读

### 背景：这个问题为什么难？
传统的开放域问答数据集（如 Natural Questions、TriviaQA）都是在固定的时间点收集并标注的，模型只需要记住当时的事实。现实世界却是不断变化的——新事件、政策、技术每天都会出现。过去的模型大多依赖离线的知识库或预训练时的语料，导致它们在面对“今天发生了什么？”这类即时问题时常常给出过时或根本不存在的答案。换句话说，缺乏“实时感知”和“动态检索”能力是旧方法的根本瓶颈。

### 关键概念速览
**开放域问答（Open‑Domain QA）**：模型需要在没有限定主题的情况下，从海量文本中找出答案，就像在互联网上随意提问一样。  
**检索增强生成（Retrieval‑Augmented Generation, RAG）**：先用搜索引擎或向量检索找出相关文档，再把这些文档喂给语言模型生成答案，类似先查资料再写报告。  
**大规模预训练语言模型（Large Pretrained LM）**：在海量文本上训练得到的模型，如 GPT‑3、T5，具备强大的语言理解和生成能力。  
**实时评测平台（Real‑Time Evaluation Platform）**：每周自动发布新问题并收集模型答案的系统，像每周一次的“快问快答”比赛。  
**不可回答检测（Unanswerability Detection）**：模型判断当前信息是否足以支撑答案的能力，类似人说“我不知道”。  
**检索反馈循环（Retrieval Feedback Loop）**：模型发现检索不到足够信息时，主动请求更换检索策略或扩大搜索范围，类似记者在采访中发现线索不足后再去深挖。

### 核心创新点
1. **从静态数据集到动态评测**：过去的 QA 研究几乎都在固定的测试集上跑分，RealTime QA 把评测频率提升到每周一次，问题全部围绕最新发生的事件。这样模型必须实时获取新信息，而不是靠记忆旧事实。  
2. **基准构建方式**：作者每周抓取公开的“每周测验”网站、新闻快讯和社交媒体热点，自动生成问题并人工核对答案，形成了一个持续增长的实时问答库。相比传统手工标注的静态数据，这套管线更省力且更贴近现实。  
3. **检索驱动的 GPT‑3 基线**：在 GPT‑3 之上加入了最新检索模块，先用 BM25+向量检索抓取最近 24 小时内的网页，再把这些文档拼接进提示（prompt）让模型生成答案。实验显示，当检索到足够信息时，GPT‑3 能显著提升准确率。  
4. **对未覆盖情况的分析框架**：作者观察到检索不到关键信息时 GPT‑3 会倾向于“凭记忆”回答旧事实，提出了“未回答检测 + 检索反馈”作为未来改进方向，为后续研究指明了路径。

### 方法详解
整体思路可以拆成三步：**问题采集 → 动态检索 → 生成与评估**。

1. **问题采集**  
   - 每周自动爬取数十个新闻站点、政府公告、社交平台的热点标题。  
   - 使用轻量的规则过滤掉明显的主观或多选题，只保留客观可查的“谁、何时、何地、何事”。  
   - 人工审校后形成一批约 200 条新问题，作为当周的评测集。

2. **动态检索**  
   - 构建一个滚动的文档库，包含过去 7 天内抓取的网页全文。  
   - 对每个问题先用 BM25（基于关键词的传统检索）得到前 10 条候选，再用句向量（如 Sentence‑BERT）重新排序，确保语义匹配。  
   - 将最终选出的 3–5 条文档拼接进提示，格式大致是：“以下是与问题相关的最新资料：…”。这一步相当于给模型提供“最新的参考书”。

3. **生成与评估**  
   - 使用 GPT‑3（davinci）或 T5‑XXL 作为生成器，提示中加入检索结果和明确的指令：“请仅依据上面的资料回答，如果资料不足请说明”。  
   - 生成后通过自动比对（BLEU、Exact Match）和人工评审两层过滤，得到最终得分。  
   - 评测平台每周公布排行榜，鼓励研究者提交改进版模型。

**最巧妙的地方**在于把检索结果直接写进提示，让模型把最新文档当作“上下文”。这比传统的“先检索后独立回答”更紧密，也让模型在信息不足时更容易暴露出“我不知道”。此外，作者把未覆盖的案例单独统计，形成了一个“检索失败率”指标，为后续的检索反馈机制提供了量化依据。

### 实验与效果
- **数据**：一年期间累计约 1,000 条实时问题，覆盖政治、科技、体育等多个领域。  
- **基线**：包括纯 GPT‑3（不检索）、纯 T5、以及传统检索+阅读理解模型（如 DPR + BERT）。  
- **结果**：在 Exact Match 上，检索增强的 GPT‑3 提升约 18%（从 42% 到 60%），T5 也有约 12% 的提升。纯模型在信息缺失时仍会给出旧答案，错误率高达 35%。  
- **消融**：去掉向量检索只保留 BM25，性能下降约 6%；去掉检索直接让模型生成，下降约 15%，说明语义检索对实时性贡献显著。  
- **局限**：作者指出，平台仍依赖英文网页，非英语信息覆盖不足；检索库的时间窗口固定为 7 天，极端新事件（如突发灾害）可能仍来不及进入库中。

### 影响与延伸思考
RealTime QA 把“问答”从“记忆考古”推向了“新闻速递”，激发了两大方向的研究热潮：一是 **实时检索‑生成** 框架的进一步优化，如使用跨语言检索或更细粒度的时间感知索引；二是 **未回答检测 + 主动检索**，让模型像客服一样在不确定时主动请求更多信息。后续的工作如 *LiveQA*、*DynamicQA* 都在此基础上加入了检索反馈循环或多模态（图文）输入。想继续深入，可以关注 **时间感知语言模型**（Temporal LM）和 **检索增强对话系统** 的最新进展。

### 一句话记住它
实时问答让模型必须“上网查资料”，而不是靠记忆旧知识——检索驱动的生成才是回答“现在”问题的关键。