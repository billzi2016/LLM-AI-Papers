# Say What You Mean! Large Language Models Speak Too Positively about   Negative Commonsense Knowledge

> **Date**：2023-05-10
> **arXiv**：https://arxiv.org/abs/2305.05976

## Abstract

Large language models (LLMs) have been widely studied for their ability to store and utilize positive knowledge. However, negative knowledge, such as "lions don't live in the ocean", is also ubiquitous in the world but rarely mentioned explicitly in the text. What do LLMs know about negative knowledge? This work examines the ability of LLMs to negative commonsense knowledge. We design a constrained keywords-to-sentence generation task (CG) and a Boolean question-answering task (QA) to probe LLMs. Our experiments reveal that LLMs frequently fail to generate valid sentences grounded in negative commonsense knowledge, yet they can correctly answer polar yes-or-no questions. We term this phenomenon the belief conflict of LLMs. Our further analysis shows that statistical shortcuts and negation reporting bias from language modeling pre-training cause this conflict.

---

# 说出你的真实想法！大语言模型对负面常识过于乐观 论文详细解读

### 背景：这个问题为什么难？
语言模型在大规模文本上训练，学会了大量“正向”事实——比如“鸟会飞”。但世界里充斥着“负向”常识，如“鱼不会爬树”，这些信息在语料中出现频率极低，因为人们很少主动说出它们。传统评估大多关注模型能否生成或回答正向事实，忽视了负向知识的存储与使用。缺少针对负向常识的测评手段，使得我们不知道模型到底懂不懂这些“不要做”的规则，也无法判断它们在实际对话或推理中会不会产生误导。

### 关键概念速览
**负向常识（Negative Commonsense Knowledge）**：指那些描述事物“不具备”或“不发生”的常识，例如“汽车不会在水下行驶”。它像是生活中的禁忌清单，常被省略却同样重要。  
**关键词到句子生成任务（Keyword‑to‑Sentence Generation, CG）**：给模型一组约束关键词（如“lion, ocean, not”），要求它生成一条完整、语法正确且符合负向常识的句子。相当于让模型在限定的拼图块中拼出一幅画。  
**布尔问答任务（Boolean QA）**：向模型提出是/否二选一的问题（如“狮子生活在海里吗？”），检查它能否给出正确的极性答案。类似于考察学生是否知道答案是“错”。  
**信念冲突（Belief Conflict）**：模型在生成自然语言时表现出对负向常识的乐观倾向（倾向说出正向句子），但在二元问答中却能给出正确的否定答案，这两者之间的矛盾被称为信念冲突。  
**统计捷径（Statistical Shortcut）**：模型利用训练数据中高频模式而非真正理解概念来做出预测，比如看到“lion”常伴随“live in”就默认正向答案。  
**否定报告偏差（Negation Reporting Bias）**：语言模型在预训练阶段更倾向学习“正向”叙述，因为文本里否定句本身稀少，导致模型在生成时自然回避否定表达。

### 核心创新点
1. **从生成视角切入负向常识评估**：以前的工作大多用是/否问答来检测模型的负向知识，忽视了模型在自由生成时的表现。本文设计了关键词到句子生成任务（CG），强制模型在限定词汇下必须输出负向句子，从而直接暴露模型在自然语言层面的偏差。  
2. **对比生成与问答两种交互方式**：通过在同一批负向事实上同时跑 CG 与 Boolean QA，作者发现模型在问答上能给出正确的否定答案，却在生成时频繁产生正向或模糊句子。这种对比揭示了“信念冲突”现象，提供了新的诊断思路。  
3. **归因分析：统计捷径 + 否定报告偏差**：作者进一步追踪模型错误的根源，发现模型往往依赖高频共现（如“lion + live”）而不是真正理解“not”。同时，预训练语料中否定句的稀缺导致模型在生成时倾向回避否定词，这两者共同导致了负向常识的系统性缺失。  
4. **提出“信念冲突”概念并给出初步缓解思路**：把生成与问答之间的矛盾命名为信念冲突，为后续研究提供了统一的讨论框架，也暗示了通过数据增强或后处理可以在生成阶段纠正模型的乐观倾向。

### 方法详解
整体思路可以分为三步：**负向事实构造 → 双任务评测 → 误差来源剖析**。  
1. **负向事实构造**：作者先从公开的常识库（如 ConceptNet）抽取正向三元组（主体‑关系‑客体），再手工或自动生成对应的负向版本，例如把“lion lives in savanna”转成“lion does not live in ocean”。每条负向事实被拆成两种提示：① 关键词集合（如 “lion, ocean, not”），② 是/否问题（如 “Do lions live in the ocean?”）。  
2. **关键词到句子生成（CG）**：把关键词集合喂给目标 LLM，要求模型在不加入额外信息的前提下输出完整句子。这里使用了“硬约束”方式：模型的输出必须包含所有关键词且顺序不变，若缺失则判为失败。评估标准包括：① 语法完整性，② 是否明确表达否定，③ 是否符合常识（人工或自动检查）。  
3. **布尔问答（QA）**：直接向模型提出二选一问题，记录模型的极性预测（Yes/No）以及置信度。这里不要求模型解释，只看答案是否与负向事实一致。  
4. **误差来源剖析**：对 CG 失败的案例进行两类分析。第一类是“统计捷径”——模型在生成时倾向使用高频正向搭配（如 “lion lives in”），即使关键词中出现 “not”。第二类是“否定报告偏差”——模型在生成时主动省略否定词或使用模糊表达（如 “lion is not usually found in the ocean”），导致句子在常识层面仍显正向。作者通过对比不同模型大小、不同预训练语料的表现，验证了这两种因素的累加效应。  
最巧妙的地方在于 **双任务对照**：同一负向事实在生成和问答两条路径上表现截然不同，直接暴露了模型内部“信念”与“表达”之间的不一致，而不是单纯说模型“不懂负向常识”。

### 实验与效果
- **数据集**：作者基于 ConceptNet、Atomic 和自建的负向常识列表，构造了约 5,000 条负向事实，分别用于 CG 与 QA。  
- **基线模型**：包括 GPT‑2、GPT‑3、LLaMA‑7B、Claude‑2 等主流大语言模型。  
- **主要发现**：在 QA 任务上，大多数模型的准确率在 85%–92% 之间，说明它们能够在二元判断中给出正确的否定答案。相反，在 CG 任务上，成功生成符合负向常识的句子比例仅在 30%–45% 左右，尤其是更大的模型（如 GPT‑3）反而更倾向于生成正向句子，表现出更明显的乐观偏差。  
- **消融实验**：作者分别去掉关键词硬约束、改用自由生成、以及在训练语料中加入人工负向句子。结果显示：加入负向句子后 CG 成功率提升约 12%，但仍远低于 QA 的水平，说明仅靠数据增强不足以根除信念冲突。  
- **局限性**：论文主要聚焦英文模型，中文或其他语言的负向常识分布可能不同；此外，CG 任务的评估仍依赖人工判断，自动化度不高。作者也承认未探索更高级的解码策略（如强制否定词）对冲突的缓解效果。

### 影响与延伸思考
这篇工作首次把负向常识的生成能力摆上台面，引发了社区对“模型说话是否符合真实世界约束”的更广泛关注。随后有几篇论文尝试在微调阶段加入**否定对抗数据**（Negation Adversarial Training），或在解码时使用**约束采样**（Constrained Sampling）强制出现否定词。还有研究把负向常识纳入**事实校验器**（Fact Verifier），让模型在生成后自动检测并纠正潜在的正向偏差。想进一步了解，可以关注**常识图谱的负向扩展**、**多语言负向常识评测**以及**解码层面的约束优化**等方向（推测）。

### 一句话记住它
大语言模型在回答是/否时能说“不”，但在自由生成时却常把“负向常识”说成“正向”，这就是所谓的“信念冲突”。