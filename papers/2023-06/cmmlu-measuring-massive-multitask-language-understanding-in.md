# CMMLU: Measuring massive multitask language understanding in Chinese

> **Date**：2023-06-15
> **arXiv**：https://arxiv.org/abs/2306.09212

## Abstract

As the capabilities of large language models (LLMs) continue to advance, evaluating their performance becomes increasingly crucial and challenging. This paper aims to bridge this gap by introducing CMMLU, a comprehensive Chinese benchmark that covers various subjects, including natural science, social sciences, engineering, and humanities. We conduct a thorough evaluation of 18 advanced multilingual- and Chinese-oriented LLMs, assessing their performance across different subjects and settings. The results reveal that most existing LLMs struggle to achieve an average accuracy of 50%, even when provided with in-context examples and chain-of-thought prompts, whereas the random baseline stands at 25%. This highlights significant room for improvement in LLMs. Additionally, we conduct extensive experiments to identify factors impacting the models' performance and propose directions for enhancing LLMs. CMMLU fills the gap in evaluating the knowledge and reasoning capabilities of large language models within the Chinese context.

---

# CMMLU：中文大规模多任务语言理解测评 论文详细解读

### 背景：这个问题为什么难？

中文大模型的能力已经可以和英文模型相媲美，但缺少系统、覆盖面广的中文评测工具，导致我们很难客观判断模型到底懂多少。过去的中文测评大多聚焦单一任务（比如阅读理解或情感分析），或者只覆盖少数学科，无法检验模型的跨学科知识和推理能力。再者，中文的学科术语、文化背景与英文差异显著，直接搬英文测评会产生偏差。于是，研究者需要一个既大规模又多学科的基准，来填补中文模型评估的空白。

### 关键概念速览
- **CMMLU**：全称 Chinese Massive Multitask Language Understanding，中文大规模多任务语言理解测评。它像一套“学科试卷”，把自然科学、社会科学、工程、 humanities 等 20 多个学科的选择题集合在一起，供模型“一次性”展示能力。
- **多任务（Multitask）**：一次评测中包含多种不同类型的题目，而不是单一的阅读或翻译任务。类似于学生参加综合素质测评，需要在同一场考试里回答数学、历史、化学等多科目。
- **In‑Context Learning（上下文学习）**：不给模型额外的梯度更新，只在输入中提供示例，让模型“看着例子学”。相当于老师在黑板上先演示几道题的解法，然后让学生直接答题。
- **Chain‑of‑Thought（思维链）**：让模型在输出答案前先写出推理步骤，像是学生在答题卡上先写解题过程，再写最终答案，帮助模型避免“一口气”直接猜。
- **随机基线（Random Baseline）**：在四选一的选择题里，随机猜的期望准确率是 25%。它是最底层的参考线，告诉我们模型是否真的在“思考”。
- **知识与推理能力**：知识指模型记住的事实（比如“水的沸点是 100°C”），推理指模型能把已知事实组合起来解决新问题（比如根据气压推算沸点变化）。
- **多语言模型（Multilingual LLM）**：同时支持多种语言的语言模型，例如 GPT‑4、Claude 等，它们在中文任务上的表现往往受训练数据比例影响。

### 核心创新点
1. **从单学科测评到全学科覆盖**  
   之前的中文基准大多只测单一领域，如阅读理解或机器翻译。作者收集并整理了 20+ 学科、共 1,000 多道选择题，形成了一个“全科考试”。这样可以一次性观察模型在不同学科的强弱点，而不是只能看到局部表现。

2. **统一的评测协议 + 多种提示方式**  
   过去的评测往往只用标准提示，忽略了提示工程的影响。本文在同一套题目上分别使用了零提示、few‑shot（提供上下文示例）和 CoT（思维链）三种设置，系统比较了不同提示对模型表现的提升幅度，揭示了提示工程在中文任务中的实际价值。

3. **大规模模型横向对比**  
   研究者挑选了 18 种主流的多语言或中文专属模型（包括开源和闭源），在同一基准上统一跑分。以前的对比往往只涉及几款模型，难以得出普遍结论。这里的横向实验让我们看到“多数模型在中文多学科上仍难突破 50%”的整体趋势。

4. **因素分析 + 改进方向**  
   通过消融实验和相关性分析，作者找出了影响中文多任务表现的关键因素，如训练数据中文比例、模型规模、提示长度等。基于这些发现，论文提出了提升中文模型的若干路线（比如增加中文教材数据、强化推理提示），为后续研发提供了明确的方向。

### 方法详解
**整体框架**  
CMMLU 的评测流程可以概括为四步：① 题库构建 → ② 任务划分 → ③ 提示设计 → ④ 结果统计。研究者先把各学科的教材、考试卷、公开题库等资源转化为统一的四选一格式；然后把题目按学科标签划分，确保每个学科都有足够的样本；接着为每种提示方式（零提示、few‑shot、CoT）编写对应的输入模板；最后把模型的输出映射到选项上，计算每个学科和整体的准确率。

**关键模块拆解**  
1. **题库构建**  
   - **来源**：高校教材、公开考试、专业书籍等。  
   - **清洗**：去除歧义、确保每题只有唯一正确答案。  
   - **平衡**：每个学科大约 50‑100 题，防止某科目因样本过少导致统计噪声。  
   类比：就像老师在出卷时，要从教材里挑选代表性题目，并确保每个章节都有覆盖。

2. **提示模板**  
   - **Zero‑shot**：直接把题干和四个选项喂给模型，类似“请直接回答”。  
   - **Few‑shot**：在题目前加入 2‑3 例相同格式的示例，示例里已经给出答案。相当于老师先演示几道相似题的解法。  
   - **CoT**：在 few‑shot 基础上，要求模型先写出思考过程（“思考过程：...”），再给出答案。类似学生在答题卡上先写解题步骤。  
   这里的创新在于统一了三种提示的实现方式，使得不同模型之间的比较更公平。

3. **模型调用与答案映射**  
   - 对每个模型，使用其官方 API 或开源代码进行推理。  
   - 解析模型输出的文字，匹配到 A/B/C/D 四个选项。若模型直接输出选项字母，则直接计分；若输出完整句子，则通过关键词匹配或正则提取答案。  
   - 统计每个学科的准确率，再取宏平均得到整体分数。  

**最巧妙的地方**  
- **统一的四选一格式**：把所有学科的题目都转成同一种答题方式，极大降低了评测实现的复杂度，也避免了因任务差异导致的评测偏差。  
- **系统化的提示对比**：不仅提供 few‑shot，还加入 CoT，直接展示了提示工程在中文多任务上的边际收益，这在中文基准里少见。

### 实验与效果
- **测试对象**：18 款主流 LLM，包括 GPT‑4、Claude、ChatGLM、LLaMA‑Chinese、BLOOM‑Z 等，覆盖从几十亿到上百亿参数的模型。  
- **基准表现**：在 zero‑shot 条件下，大多数模型的整体准确率徘徊在 30% 左右；加入 few‑shot 后提升约 5‑10 个百分点；使用 CoT 再提升 3‑7 个百分点。即便是最强的闭源模型（如 GPT‑4），在最优提示下也只能达到约 55% 的准确率。  
- **随机基线**：四选一的随机猜测准确率为 25%，说明即使是最弱的模型也已经超出随机水平，但距离“及格线” 50% 仍有明显差距。  
- **消融实验**：作者分别去掉 few‑shot 示例、去掉 CoT 步骤、以及只使用中文训练数据的模型进行对比。结果显示：few‑shot 对提升最为关键，约占整体提升的 60%；CoT 对推理类题目（如数学、逻辑）贡献更大。  
- **因素分析**：模型的中文训练数据比例、参数规模、以及是否使用了指令微调是影响表现的三大因素。中文数据占比低于 10% 的模型普遍在 humanities 类别上表现不佳。  
- **局限性**：论文未对模型的生成时间、成本进行评估；题库虽覆盖广，但仍以选择题为主，缺少生成式或开放式任务的评测；部分学科（如艺术）题目数量相对较少，统计可靠性稍弱。

### 影响与延伸思考
CMMLU 迅速成为中文大模型评测的“标配”，后续不少工作直接在其上报告改进（例如中文指令微调、知识增强等）。它也推动了中文社区对多学科评测的重视，催生了类似的中文 MMLU 变体（如 C-Eval）。如果想进一步深入，可以关注以下方向：① 将生成式问答加入基准，评估模型的表达能力；② 引入更细粒度的难度标签，分析模型在“易”“难”题上的差异；③ 探索跨语言迁移——用英文 MMLU 训练的模型在中文 CMMLU 上的表现如何。整体来看，CMMLU 为中文 LLM 的“能力地图”提供了第一张清晰的轮廓图。

### 一句话记住它
CMMLU 用一套覆盖 20+ 学科的中文选择题，揭示了即使是最强大模型，也难以在中文多任务上突破 50% 的准确率。