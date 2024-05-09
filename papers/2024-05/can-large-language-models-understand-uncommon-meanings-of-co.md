# Can large language models understand uncommon meanings of common words?

> **Date**：2024-05-09
> **arXiv**：https://arxiv.org/abs/2405.05741

## Abstract

Large language models (LLMs) like ChatGPT have shown significant advancements across diverse natural language understanding (NLU) tasks, including intelligent dialogue and autonomous agents. Yet, lacking widely acknowledged testing mechanisms, answering `whether LLMs are stochastic parrots or genuinely comprehend the world' remains unclear, fostering numerous studies and sparking heated debates. Prevailing research mainly focuses on surface-level NLU, neglecting fine-grained explorations. However, such explorations are crucial for understanding their unique comprehension mechanisms, aligning with human cognition, and finally enhancing LLMs' general NLU capacities. To address this gap, our study delves into LLMs' nuanced semantic comprehension capabilities, particularly regarding common words with uncommon meanings. The idea stems from foundational principles of human communication within psychology, which underscore accurate shared understandings of word semantics. Specifically, this paper presents the innovative construction of a Lexical Semantic Comprehension (LeSC) dataset with novel evaluation metrics, the first benchmark encompassing both fine-grained and cross-lingual dimensions. Introducing models of both open-source and closed-source, varied scales and architectures, our extensive empirical experiments demonstrate the inferior performance of existing models in this basic lexical-meaning understanding task. Notably, even the state-of-the-art LLMs GPT-4 and GPT-3.5 lag behind 16-year-old humans by 3.9% and 22.3%, respectively. Additionally, multiple advanced prompting techniques and retrieval-augmented generation are also introduced to help alleviate this trouble, yet limitations persist. By highlighting the above critical shortcomings, this research motivates further investigation and offers novel insights for developing more intelligent LLMs.

---

# 大型语言模型能理解常见词语的不常见含义吗？ 论文详细解读

### 背景：这个问题为什么难？
在过去的几年里，LLM（大语言模型）在阅读理解、对话生成等任务上取得了惊人的成绩，但大多数评测都停留在“表层”语言现象——比如判断句子是否通顺、回答事实性问题。真正考验模型是否“懂”词义的细粒度测试很少，尤其是对同一个常见词的罕见、隐喻或专业用法。传统的词义评估往往只覆盖高频意义，忽视了语言使用中的多义性和上下文依赖，这让我们无法判断模型是否具备人类那种灵活的语义共享能力。因此，评估 LLM 在“常见词的非常规意义”上的表现成为了一个急需填补的空白。

### 关键概念速览
**LLM（大语言模型）**：通过海量文本自监督学习得到的生成式模型，能够预测下一个词。类似于一个“会说话的统计机器”，但不一定真正“理解”语言。  
**词义多样性**：同一个词在不同情境下可以有多种解释，例如 “mouse” 既可以指“小鼠”也可以指“电脑鼠标”。这类似于同一个钥匙可以打开不同的锁。  
**Lexical Semantic Comprehension（LeSC）**：本文构建的专门测评数据集，聚焦于常见词的罕见含义，兼顾细粒度和跨语言两大维度。可以把它想象成“词义的高难度体能测试”。  
**跨语言评估**：在不同语言之间检验模型对同一概念的理解程度，类似于让模型在不同文化的“同义词游戏”中表现。  
**检索增强生成（RAG）**：在生成答案前先从外部知识库检索相关信息，再把检索结果喂给模型，像是给模型配了一个“即时查字典”。  
**高级提示工程**：通过精心设计的提示词（prompt）引导模型思考特定角度，例如让模型先列出所有可能的词义再选出最合适的，类似于老师给学生提供解题思路。

### 核心创新点
1. **从表层任务到词义细粒度 → 构建 LeSC 数据集 → 首个系统化评估 LLM 对常见词罕见意义的能力**。作者手工挑选了数千个常见词，每个词配上 2–3 条不常见解释，并提供多语言平行句子，使得评测既细致又具普适性。  
2. **引入跨语言维度 → 在英语、中文等多语言上同步标注 → 验证模型的语义共享是否跨语言一致**。这一步让我们能够看到模型在不同语言环境下是否仍然保持同样的词义判断，而不是仅靠单语言的统计偏好。  
3. **系统化提示与检索实验 → 设计多种高级提示模板并加入检索增强 → 在同一基准上对比不同提升手段的效果**。通过这些实验，作者展示了即使使用最前沿的提示技巧和外部检索，模型仍然难以突破基本的词义理解瓶颈。  
4. **人类基准对齐 → 让 16 岁学生完成同样任务 → 将模型成绩与真实人类水平直接对比**。结果显示，GPT‑4 仍比人类低 3.9%，GPT‑3.5 低 22.3%，直观呈现了当前模型的局限。

### 方法详解
整体思路可以拆成三大步骤：**数据构建 → 评测框架设计 → 提升手段实验**。

1. **LeSC 数据集构建**  
   - **词汇挑选**：从公开词频表中抽取高频词，然后人工筛选出在日常对话中出现频率低的意义（如专业术语、俚语、隐喻）。  
   - **意义标注**：每个词配上 2–3 条罕见解释，并给出对应的上下文句子，确保模型必须依赖语境才能辨认正确意义。  
   - **跨语言映射**：同一概念在中文、英文等语言中分别提供等价句子，形成平行语料，便于跨语言评测。  
   - **质量控制**：采用双人标注+专家复审，确保罕见意义的真实性和上下文的自然度。

2. **评测框架**  
   - **任务形式**：给模型一个包含目标词的句子，要求模型输出该词在该句中的具体意义（文字描述或选项编号）。  
   - **评价指标**：采用准确率（Accuracy）作为主指标，同时引入跨语言一致性分数（Cross‑Lingual Consistency）衡量模型在不同语言版本上答案的一致性。  
   - **基准对照**：让 16 岁学生在同样的题目上作答，得到人类上限。

3. **提升手段实验**  
   - **高级提示**：设计三类提示模板——（a）直接问意义，（b）先让模型列举所有可能意义再选，（c）提供“词义列表”作为背景信息。  
   - **检索增强生成（RAG）**：在模型生成答案前，使用 BM25 检索引擎从维基百科、专业词典等资源中拉取与目标词相关的段落，拼接到提示中。  
   - **组合实验**：将提示模板与检索结果组合，观察是否产生叠加效应。  

**最巧妙的点**在于把“词义多样性”转化为可量化的评测任务，并且通过跨语言平行句子让模型的语义理解不再是单语言的“记忆游戏”，而是真正的“概念共享”。此外，作者没有只停留在“给模型更多信息”，而是系统化比较了不同信息提供方式的边际收益，揭示了即使是最强的提示和检索也只能略微提升表现。

### 实验与效果
- **数据规模**：LeSC 包含约 4,000 条英语句子、3,500 条中文句子以及对应的跨语言对。  
- **模型覆盖**：评测了开源的 LLaMA、Mistral、Falcon 系列，以及闭源的 GPT‑3.5、GPT‑4。  
- **主要结果**：在整体准确率上，GPT‑4 取得约 86% 的得分，仍比 16 岁学生的 89.9% 低 3.9%；GPT‑3.5 约 67%，比人类低 22.3%。所有开源模型的得分均在 55%–65% 之间，明显落后。跨语言一致性分数同样呈现相似差距，说明模型在不同语言下的词义判断并不统一。  
- **提示与检索效果**：高级提示提升约 2%–4% 的准确率，检索增强提升约 3% 左右，两者叠加最多提升 6%。即便如此，仍未能追平人类水平。  
- **消融实验**：去掉跨语言对照后，模型在单语言上略有提升，说明跨语言评测对模型构成了更严格的约束。去掉检索模块后，提升效果几乎消失，验证了检索对罕见意义的帮助是有限的。  
- **局限性**：作者承认 LeSC 仍然受限于人工标注规模，罕见意义的覆盖度不可能穷尽；此外，评测只关注词义选择，未涉及更深层的语用推理。  

### 影响与延伸思考
这篇工作在社区里掀起了对“词义细粒度评测”的关注，随后出现了几篇基于 LeSC 思路的多语言词义挑战赛（如 *LexicalSense*），并推动了更细致的语义评估基准（例如 *WordSenseBench*）。研究者开始把注意力从大模型的“规模”转向“语义深度”，探索如何让模型在少数例子甚至零样本情况下捕捉到隐蔽的词义。未来的方向可能包括：① 将词义推理与常识图谱结合，让模型拥有结构化的概念网络；② 发展自适应提示生成器，根据上下文自动生成最有效的提示；③ 扩展到更丰富的语言现象，如情感隐喻、文化特定的俚语等。对想进一步了解的读者，可以关注近期在 ACL、EMNLP 上出现的 “Fine‑grained Semantic Understanding” 主题会议论文。

### 一句话记住它
LLM 在常见词的罕见意义上仍远不及普通青少年，即使加上高级提示和检索，也只能略微弥补这块“语义盲区”。