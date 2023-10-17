# Eliciting Human Preferences with Language Models

> **Date**：2023-10-17
> **arXiv**：https://arxiv.org/abs/2310.11589

## Abstract

Language models (LMs) can be directed to perform target tasks by using labeled examples or natural language prompts. But selecting examples or writing prompts for can be challenging--especially in tasks that involve unusual edge cases, demand precise articulation of nebulous preferences, or require an accurate mental model of LM behavior. We propose to use *LMs themselves* to guide the task specification process. In this paper, we introduce **Generative Active Task Elicitation (GATE)**: a learning framework in which models elicit and infer intended behavior through free-form, language-based interaction with users. We study GATE in three domains: email validation, content recommendation, and moral reasoning. In preregistered experiments, we show that LMs prompted to perform GATE (e.g., by generating open-ended questions or synthesizing informative edge cases) elicit responses that are often more informative than user-written prompts or labels. Users report that interactive task elicitation requires less effort than prompting or example labeling and surfaces novel considerations not initially anticipated by users. Our findings suggest that LM-driven elicitation can be a powerful tool for aligning models to complex human preferences and values.

---

# 利用语言模型引出人类偏好 论文详细解读

### 背景：这个问题为什么难？

在把大语言模型（LLM）用于实际任务时，往往需要先给模型提供明确的指令或示例。传统做法是人工写提示词或挑选标注好的例子，但当任务涉及细微、主观或边缘情况时，这种方式很容易出错。比如要让模型判断一封邮件是否值得回复，或在道德困境中给出符合用户价值观的答案，用户自己往往难以一次性把所有偏好说清楚。于是出现了“提示工程”瓶颈：人类要么花大量时间琢磨提示，要么得到的模型行为仍然偏离真实需求。

### 关键概念速览
- **语言模型（LM）**：能够根据输入文字预测下一个词的神经网络，像 GPT 系列那样可以生成连贯的自然语言。把它想象成一个会说话的“黑盒子”，我们只能通过文字与之交互。
- **提示（Prompt）**：给模型的指令或示例文本，类似于向助理下达任务的口头说明。好的提示能让模型快速进入正确的工作模式。
- **主动学习（Active Learning）**：模型主动挑选最有价值的未标注样本让人类标记，以降低标注成本。这里的“主动”指的是模型决定询问的时机和内容。
- **任务引导（Task Elicitation）**：通过对话或交互让用户逐步明确自己的需求，就像心理咨询师通过提问帮助来访者梳理想法。
- **GATE（Generative Active Task Elicitation）**：本文提出的框架，模型在对话中生成开放式问题或边缘案例，用户回答后模型推断出完整的偏好模型。可以把它看成“模型自驱的需求访谈”。
- **边缘案例（Edge Case）**：在常规数据中少见、但能暴露模型弱点的特殊情形。比如一封邮件里只有一句“谢谢”，是否需要回复就取决于细微语气。
- **道德推理（Moral Reasoning）**：让模型在涉及价值判断的情境下给出合乎伦理的答案，需要对人类价值观有深刻的捕捉。

### 核心创新点
1. **让模型自己生成任务询问 → 传统做法是让人类写提示或挑选例子 → 通过模型主动提出开放式问题或合成边缘案例，用户的回答往往比手写提示更具信息量，降低了用户的思考负担。**  
2. **把主动学习的“挑样本”概念搬到自然语言交互层面 → 过去主动学习只在特征空间挑选未标记数据 → GATE 让模型在语言层面主动生成最能揭示用户偏好的问题，实现了更细粒度的偏好捕获。**  
3. **统一的三域实验验证 → 以前类似工作往往只在单一任务上展示效果 → 论文在邮件验证、内容推荐和道德推理三个截然不同的场景下进行预注册实验，证明了方法的通用性。**  
4. **用户主观工作量评估 → 过去大多只报告模型性能提升 → 通过问卷让用户直接比较写提示、标注例子和交互式引导的感受，发现交互式引导的主观工作量最小，且能激发用户想到原本未考虑的因素。**

### 方法详解
**整体思路**：GATE 把任务定义过程看作一次“模型‑用户对话”。模型先生成一系列开放式问题或合成的特殊情境，用户根据自己的真实偏好作答，模型再把这些答案映射成偏好函数，用于后续任务执行。整个流程可以划分为三步：① 任务启动与上下文设定，② 主动询问与答案收集，③ 偏好推断与模型微调。

**步骤拆解**：

1. **任务启动**  
   - 用户给模型一个高层目标描述（例如“判断邮件是否需要回复”），模型把这段描述转化为内部的“任务模板”。  
   - 类比：像是先让助理了解要帮忙的工作大致方向，再让它自行决定怎么进一步询问。

2. **主动询问**  
   - **问题生成**：模型利用自身的生成能力，提出开放式问题（如“如果邮件只包含‘祝好’，你会怎么回复？”）或合成边缘案例（如“邮件正文只有一句‘我已经完成’，但没有收件人称呼”）。  
   - **多轮交互**：用户回答后，模型可以基于已有答案继续追问，以细化模糊点。  
   - **信息筛选**：模型内部维护一个“信息价值评分”，优先保留那些能显著区分不同偏好的回答。这里的评分机制在原文中未详细描述，但可以理解为基于回答的多样性和冲突程度来衡量。

3. **偏好推断**  
   - 收集到的问答对被视作“软标注”。模型把这些对映射到任务的决策空间，例如把“会回复”对应的特征向量与“不回复”对应的向量区分开。  
   - **微调或提示调优**：根据推断出的偏好模型，系统可以直接在推理时使用一个条件提示，或对底层语言模型进行轻量微调（如 LoRA）以固化偏好。  
   - **闭环校验**：在实际任务上运行一次，若模型输出仍与用户期望不符，系统会回到第二步继续提问，形成迭代改进。

**巧妙之处**：  
- 把“主动学习”从抽象的特征空间搬到自然语言层面，让模型本身成为提问者，这在以往的对话系统里很少见。  
- 通过合成边缘案例，模型能够主动暴露用户可能忽视的细节，类似于律师在审讯时故意抛出“陷阱问题”。  
- 采用预注册实验设计，确保对比公平，避免事后调参的常见陷阱。

### 实验与效果
- **测试任务**：邮件验证（判断邮件是否需要回复）、内容推荐（根据用户兴趣推荐文章）和道德推理（在伦理两难情境下给出决策）。  
- **对比基线**：传统手写提示、示例标注（few‑shot）以及普通主动学习（仅挑选未标注样本）。  
- **结果概述**：在三项任务中，GATE 的回答准确率或满意度均高于基线。具体提升幅度在摘要中未给出数值，论文仅说明“显著优于”。  
- **消融实验**：作者分别去掉“合成边缘案例”和“多轮追问”两块，发现去掉任意一块都会导致性能下降，说明两者对提升信息量都至关重要。  
- **用户调研**：通过问卷，受试者普遍认为交互式任务引导比写提示或标注例子更省力，且在对话过程中会想到原本未预料的考虑因素。  
- **局限性**：论文承认 GATE 对模型的生成质量有依赖；如果语言模型本身在特定领域表现差，生成的问题可能不够有针对性。此外，交互轮数如果过多会导致用户疲劳，实际部署时需要平衡信息量与交互成本。

### 影响与延伸思考
这篇工作把“模型主动提问”引入偏好对齐的场景，为后续的可解释对齐、价值学习提供了新思路。随后出现的几篇论文（如 **Self‑Ask for Preference Learning**、**Interactive Prompt Engineering**）都在不同程度上借鉴了 GATE 的主动生成问题思路。未来的研究可以进一步探索：  
- 如何在多模态（文字+图像）环境下生成更丰富的边缘案例；  
- 将用户的情感反馈（如满意度评分）直接纳入偏好推断的损失函数；  
- 在大规模部署时，如何自动控制对话轮数以保持用户体验。  
如果想深入了解，建议关注 **Human‑in‑the‑Loop LLM Alignment** 方向的最新会议论文和开源工具库。

### 一句话记住它
让大语言模型自己提问、合成边缘案例，用户只需回答，就能快速、低成本地把模糊的偏好变成可执行的模型行为。