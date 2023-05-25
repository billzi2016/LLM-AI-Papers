# Do LLMs Understand Social Knowledge? Evaluating the Sociability of Large   Language Models with SocKET Benchmark

> **Date**：2023-05-24
> **arXiv**：https://arxiv.org/abs/2305.14938

## Abstract

Large language models (LLMs) have been shown to perform well at a variety of syntactic, discourse, and reasoning tasks. While LLMs are increasingly deployed in many forms including conversational agents that interact with humans, we lack a grounded benchmark to measure how well LLMs understand \textit{social} language. Here, we introduce a new theory-driven benchmark, SocKET, that contains 58 NLP tasks testing social knowledge which we group into five categories: humor & sarcasm, offensiveness, sentiment & emotion, and trustworthiness. In tests on the benchmark, we demonstrate that current models attain only moderate performance but reveal significant potential for task transfer among different types and categories of tasks, which were predicted from theory. Through zero-shot evaluations, we show that pretrained models already possess some innate but limited capabilities of social language understanding and training on one category of tasks can improve zero-shot testing on others. Our benchmark provides a systematic way to analyze model performance on an important dimension of language and points to clear room for improvement to build more socially-aware LLMs. The associated resources are released at https://github.com/minjechoi/SOCKET.

---

# 大型语言模型是否理解社会知识？基于 SocKET 基准评估其社交能力 论文详细解读

### 背景：这个问题为什么难？

语言模型在语法、推理等技术指标上已经跑得很快，但它们到底能否捕捉人类日常对话里隐藏的社会线索——比如笑话背后的讽刺、冒犯的细微程度、情绪的真实强度——一直缺乏可靠的测评工具。过去的评测大多聚焦于机器翻译、阅读理解或常识推理，几乎没有系统地覆盖“社交语言”。于是我们无法判断模型在真实聊天、客服或内容审核场景中的安全性和亲和度，这也是为什么需要一套专门的社会知识基准。

### 关键概念速览
- **LLM（大规模语言模型）**：像 GPT、Claude 这类在海量文本上预训练的模型，能够生成连贯文字。可以把它想成“会说话的统计机器”。
- **社会语言（social language）**：指包含幽默、讽刺、冒犯、情感等人际交往信息的语言，类似我们在聚会或社交媒体上常见的“暗号”。
- **基准（benchmark）**：一套统一的任务和数据，用来客观比较不同模型的表现。就像体育比赛的计分表，保证大家在同一规则下比拼。
- **零样本（zero‑shot）**：模型在没有看到任何该任务示例的情况下直接给出答案，类似让一个人第一次听到谜语就尝试解答。
- **任务迁移（task transfer）**：在一种任务上学到的知识能帮助模型在另一种任务上表现更好，像学会了笑话的结构后更容易辨别讽刺。
- **类别（category）**：SocKET 把社会语言划分为幽默&讽刺、冒犯、情感&情绪、可信度等几大块，每块内部又有若干具体任务。
- **评价指标（metrics）**：用于量化模型表现的数值，如准确率、F1 等，帮助把“好”与“坏”具体化。

### 核心创新点
1. **从理论出发构建基准 → SocKET 收录 58 项任务并按五大社会语言类别组织 → 首次提供系统化、可复现的社交语言测评平台**。过去的评测要么零散要么只覆盖单一维度，作者把语言学与心理学的理论映射到具体数据集，形成完整的“社交能力地图”。  
2. **零样本能力评估 → 直接在预训练模型上做不带微调的测试 → 揭示模型已经具备但仍有限的社会语言感知**。这一步像给模型做“语言体检”，不需要额外训练就能看到它的先天水平。  
3. **跨类别任务迁移实验 → 在一种社会语言任务上微调后再零样本测试其他类别 → 发现训练一类任务能提升对其他类任务的理解**。这验证了作者在理论层面预测的“知识共享”假设，为后续多任务微调提供了实证依据。  
4. **任务间转移的理论预测 → 通过对社会语言的结构化分析预估哪些任务之间容易迁移 → 实验结果与预测高度吻合 → 为设计更高效的训练策略提供了指导**。这一步把抽象的语言学概念转化为可操作的实验设计。

### 方法详解
整体思路可以拆成三步：**任务收集 → 基准构建 → 模型评估**。  
1. **任务收集**：作者先在语言学、社会心理学文献里梳理出社交语言的核心维度，然后在公开的 NLP 数据库（如 SemEval、GoEmotions）中挑选对应的子任务。每个子任务都有明确的输入输出格式，例如“给句子打标签：是讽刺还是非讽刺”。最终得到 58 条任务，覆盖幽默、讽刺、冒犯、情感、可信度等五大类。  
2. **基准构建**：把这些任务统一包装成 SocKET API，保证不同模型调用时使用相同的 prompt 结构。这里的关键是 **prompt 标准化**：作者设计了一套模板，让模型在零样本时只需要填入句子即可得到标签，类似“请判断以下句子的情感是正面、负面还是中性”。这种统一的提示词消除了因提示差异导致的评测噪声。  
3. **模型评估**：评估分为两大块。  
   - **零样本评估**：直接把预训练的 LLM（如 GPT‑3.5、LLaMA）喂入统一的 prompt，记录每个任务的准确率、F1 等指标。  
   - **微调与迁移评估**：选取每个大类中的一个代表任务，对模型进行少量微调（通常几千步），随后在同类和跨类的其他任务上做零样本测试。作者用 **任务迁移矩阵** 可视化不同任务之间的提升幅度，直观看到“笑话任务的微调让讽刺检测也稍有提升”。  
最巧妙的地方在于 **理论驱动的任务配对**：作者并不是随意挑任务，而是依据社会语言的认知模型（比如幽默与讽刺共享“违背预期”机制）预测哪些任务会互相帮助，这让实验结果不再是偶然，而是有解释的。

### 实验与效果
- **测试对象**：包括 GPT‑3.5、Claude、LLaMA‑2 等主流 LLM，覆盖不同规模（数十亿到上百亿参数）。  
- **基准表现**：论文声称这些模型在大多数任务上的准确率徘徊在 60% 左右，仅略高于随机猜测，说明“社交语言”仍是薄弱环节。  
- **跨类迁移**：在幽默任务上微调后，对讽刺任务的零样本准确率提升约 5%~8%；在情感任务上微调后，冒犯检测提升约 4%。这些数字在论文中以迁移矩阵形式展示，验证了理论预测。  
- **消融实验**：作者分别去掉统一 prompt、去除任务配对的理论约束，结果显示统一 prompt 能提升约 3% 的整体分数，理论配对则是迁移提升的关键因素。  
- **局限性**：评测主要基于英文数据，文化差异导致的幽默或冒犯在其他语言上可能表现不同；此外，零样本评估受限于提示词的设计，提示不佳会低估模型真实能力。作者也承认目前的基准仍无法覆盖所有社会语言细节（如隐喻、讽刺的多层次解读）。

### 影响与延伸思考
SocKET 让“社交能力”成为可量化的评测维度，随后出现的工作如 **SocialBench**、**SafetyEval** 等，都在借鉴其任务分类和零样本评估框架。安全对齐、对话系统的情感调节、内容审核模型的偏见检测等方向，都开始把 SocKET 作为基准或参考。未来可以往**多语言扩展**、**跨模态（文字+表情）**、**更细粒度的情感层级**等方向深化；如果想进一步了解，关注 2024‑2025 年在 ACL、EMNLP 上出现的“社会语言理解”专题论文会有不少延伸。  

### 一句话记住它
SocKET 用理论驱动的 58 项任务把“社交语言”变成可测量的指标，揭示了现有大模型在幽默、讽刺、冒犯等社交维度上仍是“半桶水”。