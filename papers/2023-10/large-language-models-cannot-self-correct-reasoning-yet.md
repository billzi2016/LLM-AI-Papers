# Large Language Models Cannot Self-Correct Reasoning Yet

> **Date**：2023-10-03
> **arXiv**：https://arxiv.org/abs/2310.01798

## Abstract

Large Language Models (LLMs) have emerged as a groundbreaking technology with their unparalleled text generation capabilities across various applications. Nevertheless, concerns persist regarding the accuracy and appropriateness of their generated content. A contemporary methodology, self-correction, has been proposed as a remedy to these issues. Building upon this premise, this paper critically examines the role and efficacy of self-correction within LLMs, shedding light on its true potential and limitations. Central to our investigation is the notion of intrinsic self-correction, whereby an LLM attempts to correct its initial responses based solely on its inherent capabilities, without the crutch of external feedback. In the context of reasoning, our research indicates that LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction. Drawing from these insights, we offer suggestions for future research and practical applications in this field.

---

# 大型语言模型尚不能自行纠正推理错误 论文详细解读

### 背景：这个问题为什么难？
在 LLM（大型语言模型）被广泛用于写代码、答题、写作之前，大家已经发现它们经常会“自信地”给出错误答案。传统的做法是让模型在外部监督（比如人工标注或检索）下再生成一次答案，但这需要额外的资源。于是出现了“自纠正”概念：让模型自己检查并改正自己的输出。看似只动动嘴巴就能提升准确率，然而推理过程本身涉及多步逻辑、隐含假设和数学运算，这些环节极易产生细微错误。若没有外部提示，模型能否发现并修正这些错误，一直是个悬而未决的难题。

### 关键概念速览
**LLM（大型语言模型）**：通过海量文本训练得到的生成式模型，能够根据输入生成连贯的文字。可以把它想象成一个“会说话的百科全书”。  
**自纠正（Self‑Correction）**：模型在给出初始答案后，再次被要求审视并改进自己的输出，整个过程不依赖外部标注或检索。类似于学生做完题目后自行检查卷子。  
**内在自纠正（Intrinsic Self‑Correction）**：指模型仅凭自身已有的知识和推理能力进行纠正，而不借助任何外部信息。相当于让学生在不看答案的情况下自行找错。  
**外部反馈（External Feedback）**：来自人类、检索系统或奖励模型的指示，用来告诉模型哪儿错了。像老师在批改作业时给出的分数和评语。  
**推理（Reasoning）**：在回答过程中需要进行多步思考、演绎或计算的任务，例如数学题、逻辑谜题。可以类比为解一道需要多步计算的方程。  
**思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前先把每一步推理写出来，像在草稿纸上写演算过程。  
**提示工程（Prompt Engineering）**：通过设计输入文字的方式，引导模型产生期望的行为。相当于老师给学生的“解题指令”。  
**性能退化（Performance Degradation）**：自纠正后模型的答案比原始答案更差的现象，类似于学生在检查时把本来对的题目改错了。

### 核心创新点
1. **明确区分内在自纠正与外部反馈** → 论文把“模型自己纠正”和“借助外部信息纠正”严格分开，专门构建了只使用模型自身能力的实验流程 → 让研究者能够直接评估模型的自我纠错潜力，而不是混杂外部帮助的效果。  
2. **系统化评估框架** → 设计了“一次生成 + 自纠正”两阶段的实验管线：先让模型完成任务，再用同一模型在不提供额外信息的情况下生成“纠正稿”。 → 通过对比原始答案、纠正稿和人工参考答案，量化自纠正是否真的提升了正确率。  
3. **发现并量化性能退化** → 在多个推理基准上，作者观察到自纠正有时会让正确率下降，并给出具体的退化比例。 → 这揭示了自纠正并非万能的“后处理”，提醒社区在实际系统中慎用。  
4. **提出未来方向的思考** → 基于实验结果，作者指出需要更强的内部一致性检查、元认知能力或更好的提示设计，才能让模型真正做到自我纠正。 → 为后续研究提供了明确的路线图。

### 方法详解
**整体思路**  
论文的实验流程可以概括为三步：① 给定任务输入 → ② 让 LLM 直接生成答案（初始答案） → ③ 再次把同一个模型放进一个“自纠正提示”，让它在不看外部答案的情况下尝试改写或补充自己的答案。整个过程完全在模型内部完成，没有任何检索、人工标注或奖励模型的介入。

**关键模块拆解**  
1. **任务输入**：选取需要多步推理的公开基准（如数学文字题、逻辑推理题），每个样本只提供题干。  
2. **初始生成**：使用标准的直接提示（例如 “请解答以下问题：...”），得到模型的第一版答案。这里可以开启或关闭 CoT，如果开启，模型会先写出思维链再给出结论。  
3. **自纠正提示**：构造一个专门的提示，形式大致为：“你刚才的回答是……请检查并在必要时纠正它”。提示中不包含正确答案或任何外部信息，只让模型回顾自己的输出。  
4. **纠正生成**：模型在上述提示下产生第二版答案（纠正稿）。如果模型认为原答案已经完美，它可能直接复述；如果检测到矛盾或不确定，它会尝试修改。  
5. **评估**：把初始答案、纠正稿分别与人工参考答案比对，统计正确率、错误类型以及是否出现退化。

**背后的假设与技巧**  
- **自我审视能力**：作者假设 LLM 能够在内部形成“元认知”，即对自己的输出进行评估。提示的设计正是激活这种能力的关键。  
- **提示的层次化**：自纠正提示比直接生成提示更长、更具指令性，类似于老师让学生先“检查答案是否符合逻辑”。这种层次化的提示被认为能让模型进入检查模式。  
- **不使用外部检索**：所有信息都来源于模型内部的参数，这一点与大多数“自我纠错”实验（往往会加入检索或奖励模型）形成鲜明对比。

**最巧妙的地方**  
论文没有引入任何新模型或额外训练，只是通过精心设计的两轮提示，直接检验了模型的内在纠错潜力。这种“最小干预”思路让结论更具说服力：如果连在同一模型内部都做不到自纠正，那么再加外部模块的改进才是唯一的出路。

### 实验与效果
- **测试任务**：作者在多个需要推理的公开数据集上做实验，包括数学文字题（类似 GSM8K）、逻辑谜题和常识推理等。  
- **对比基线**：主要对比了“直接生成”与“自纠正后生成”。在一些基准上，直接生成的正确率大约在 60% 左右（具体数字请参考原文），而自纠正后正确率并没有显著提升，甚至在部分数据集上下降了约 5%~10%。  
- **消融实验**：论文尝试了不同的自纠正提示长度、是否开启 CoT、以及是否让模型先复述原答案再纠正。结果显示，提示的细节对性能影响有限，核心瓶颈仍在模型本身缺乏可靠的内部一致性检查。  
- **局限性**：作者承认实验只覆盖了几类推理任务，未涉及更大规模的开放域对话或代码生成；此外，模型规模仅限于公开的几种主流 LLM，未测试最新的超大模型。  

### 影响与延伸思考
这篇论文在社区里引发了对“自我审视”能力的重新审视。随后出现的工作，如 **Self‑Consistency**（让模型多次采样后取多数答案）和 **Tree‑of‑Thought**（在搜索树中进行多步推理），都在尝试通过内部多样性来弥补单轮自纠正的不足。还有一些研究把 **奖励模型（Reward Model）** 与自纠正结合，形成“内部反馈回路”。如果你想进一步了解，可以关注 **Meta‑Learning for Self‑Correction**、**Meta‑Prompting** 等方向，它们尝试让模型在训练阶段就学习如何检测并修正自己的错误——这正是本文指出的下一步关键。  

### 一句话记住它
LLM 在没有外部帮助的情况下，仍然难以自行发现并改正推理错误，甚至可能把好答案弄坏。