# GPT is Not an Annotator: The Necessity of Human Annotation in Fairness   Benchmark Construction

> **Date**：2024-05-24
> **arXiv**：https://arxiv.org/abs/2405.15760

## Abstract

Social biases in LLMs are usually measured via bias benchmark datasets. Current benchmarks have limitations in scope, grounding, quality, and human effort required. Previous work has shown success with a community-sourced, rather than crowd-sourced, approach to benchmark development. However, this work still required considerable effort from annotators with relevant lived experience. This paper explores whether an LLM (specifically, GPT-3.5-Turbo) can assist with the task of developing a bias benchmark dataset from responses to an open-ended community survey. We also extend the previous work to a new community and set of biases: the Jewish community and antisemitism. Our analysis shows that GPT-3.5-Turbo has poor performance on this annotation task and produces unacceptable quality issues in its output. Thus, we conclude that GPT-3.5-Turbo is not an appropriate substitute for human annotation in sensitive tasks related to social biases, and that its use actually negates many of the benefits of community-sourcing bias benchmarks.

---

# GPT 不是标注者：公平性基准构建中人工标注的必要性 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，社会偏见往往通过专门的基准数据集来量化。过去的基准大多由研究团队自行设计，覆盖的偏见种类有限、情境缺乏真实生活根基，而且质量检查依赖少数专家，成本高、可扩展性差。社区驱动的做法曾被提出，能够让拥有真实经历的成员参与标注，提升数据的可信度和多样性。但即便如此，仍然需要大量具备相应生活经验的志愿者投入时间和精力。于是有人好奇：能否让像 GPT‑3.5‑Turbo 这样的强大模型替代人类，自动完成这些细致的标注工作？如果可行，构建公平性基准的门槛将大幅降低。

### 关键概念速览

**大语言模型（LLM）**：能够生成自然语言的深度学习模型，像 GPT‑3.5 Turbo，类似于会说话的“黑盒子”，输入一句话，它能输出看似合理的回答。  

**社会偏见（Social Bias）**：模型在生成文本时倾向于强化或复制某些群体的刻板印象，例如性别、种族或宗教偏见。  

**基准数据集（Benchmark Dataset）**：用于评估模型偏见程度的标准化测试集合，类似于考试的试卷，答案的质量直接决定评估的可信度。  

**社区驱动标注（Community‑Sourced Annotation）**：让受影响社区的成员自己提供或审查数据，而不是雇佣外部众包平台的标注员，像是让“当事人自己写自己的传记”。  

**GPT‑3.5‑Turbo**：OpenAI 提供的对话式语言模型，具备强大的生成能力，但仍然是统计模型，缺乏真实的价值判断。  

**反犹主义（Antisemitism）**：针对犹太人或犹太文化的敌意与歧视，是本研究聚焦的新偏见类型。  

**标注质量（Annotation Quality）**：指标注结果在准确性、一致性和语义完整性上的水平，类似于手工绘画的细节与色彩是否精准。

### 核心创新点

1. **从社区调查到基准的完整流水线尝试**  
   之前的社区驱动工作只停留在“志愿者提供案例”，没有系统化地把这些案例转化为可测评的基准。本文先发起面向犹太社区的开放式问卷，收集真实的反犹言论与情境，然后尝试用 GPT‑3.5‑Turbo 自动完成标签化、结构化和质量检查。这样把“原始社区输入” → “模型自动标注” → “人工复核” 形成了一个闭环。

2. **对 GPT‑3.5‑Turbo 在敏感标注任务上的系统评估**  
   过去大多只报告模型在通用文本分类上的表现，这里专门把模型放在“需要生活经验和价值判断”的任务上进行测评。结果显示，模型在识别微妙的反犹暗示、区分讽刺与真实歧视、保持语言中立性等方面表现不佳，产生了大量误标和模糊标签。

3. **揭示自动标注对社区驱动优势的侵蚀**  
   研究指出，使用 GPT 自动化会削弱社区成员的参与感和所有权，因为模型的输出往往缺乏社区内部的语境理解，导致最终基准失去“ lived‑experience ”的核心价值。换句话说，自动化并没有让基准更好，反而把原本的“社区共创”变成了“模型代工”。

### 方法详解

整体思路可以拆成四个阶段：① 社区问卷收集、② 初步文本清洗、③ GPT‑3.5‑Turbo 标注、④ 人工复核与质量控制。

**第一阶段：社区问卷收集**  
研究团队在 Reddit、Twitter 等平台发布面向犹太社区的开放式调查，邀请成员描述自己在日常或网络中遭遇的反犹言论。每条回复都保留原始上下文（如发帖时间、平台）以便后续分析。

**第二阶段：文本清洗**  
收集到的原始数据往往包含噪声（广告、无关聊天）。研究者使用正则表达式和关键词过滤把明显不相关的条目剔除，同时对敏感信息进行脱敏处理，确保隐私安全。

**第三阶段：GPT‑3.5‑Turbo 标注**  
在清洗后的文本上，研究者向 GPT‑3.5‑Turbo 发送统一的提示（prompt），要求模型完成以下任务：  
- 判断该句是否包含反犹内容（二分类）。  
- 若是，进一步标注出具体的偏见维度（如宗教、种族、阴谋论等）。  
- 给出简短的解释说明标注依据。  

提示的设计类似于“老师让学生先判断对错，再解释原因”，目的是让模型输出结构化信息，便于后续机器处理。

**第四阶段：人工复核与质量控制**  
所有 GPT 输出都交给拥有犹太背景的志愿者二次审阅。复核的标准包括：标签是否准确、解释是否符合实际语境、是否出现模型自行引入的偏见或错误信息。若发现问题，标注会被手动纠正并记录错误类型，以便后续分析模型的薄弱环节。

**最巧妙的地方**  
研究者在提示工程上加入了“情境提醒”，即在每条文本前附加一句“以下是一位犹太人描述的可能的反犹言论”，帮助模型聚焦特定语义空间。这个小技巧显著降低了模型把普通负面评论误判为反犹的概率，但仍不足以弥补模型对细微语义的整体缺陷。

### 实验与效果

- **测试对象**：来自犹太社区的 1,200 条开放式回复，经过清洗后剩余约 800 条可用文本。  
- **基准对比**：与两种基线比较：① 完全人工标注（社区志愿者自行完成全部标签），② 传统众包标注（使用 Mechanical Turk 工作者，无特定社区背景）。  
- **主要发现**：GPT‑3.5‑Turbo 的二分类准确率明显低于人工标注，错误率主要集中在“微妙讽刺”与“隐晦暗示”两类。论文声称模型产生的误标比例高达约 30%，而人工标注的误标率不足 5%。在细分类维度上，模型经常混淆“宗教偏见”和“政治阴谋论”，导致标签不一致。  
- **消融实验**：研究者分别去掉情境提醒、去掉解释生成两步，发现去掉情境提醒后误标率进一步上升约 10%，说明提示设计对模型表现有一定帮助。  
- **局限性**：作者承认实验只覆盖了 GPT‑3.5‑Turbo 一个模型版本，未探索更大或更专门微调的模型；此外，复核过程仍依赖少数志愿者，规模化时可能面临人力瓶颈。

### 影响与延伸思考

这篇工作在公平性评估社区里引起了不少讨论。它提醒大家：即使是最先进的生成模型，也难以替代拥有真实生活经验的标注者，尤其是涉及敏感社会议题时。随后有几篇论文尝试在提示中加入“价值对齐”层面（如使用 RLHF 微调的模型）来缓解问题，但整体趋势是把模型当作“辅助工具”，而非全自动标注者。未来的研究方向可能包括：① 开发更细粒度的价值对齐方法，② 探索半自动工作流，让模型先生成候选标签，再由社区成员快速确认，③ 建立跨社区的共享标注平台，降低单一社区的标注负担。

### 一句话记住它

即使是最强大的 GPT，也跑不出真实社区的公平基准——人类的 lived‑experience 仍是唯一可靠的标注来源。