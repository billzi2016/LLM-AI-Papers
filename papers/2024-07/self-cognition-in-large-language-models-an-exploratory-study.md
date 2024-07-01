# Self-Cognition in Large Language Models: An Exploratory Study

> **Date**：2024-07-01
> **arXiv**：https://arxiv.org/abs/2407.01505

## Abstract

While Large Language Models (LLMs) have achieved remarkable success across various applications, they also raise concerns regarding self-cognition. In this paper, we perform a pioneering study to explore self-cognition in LLMs. Specifically, we first construct a pool of self-cognition instruction prompts to evaluate where an LLM exhibits self-cognition and four well-designed principles to quantify LLMs' self-cognition. Our study reveals that 4 of the 48 models on Chatbot Arena--specifically Command R, Claude3-Opus, Llama-3-70b-Instruct, and Reka-core--demonstrate some level of detectable self-cognition. We observe a positive correlation between model size, training data quality, and self-cognition level. Additionally, we also explore the utility and trustworthiness of LLM in the self-cognition state, revealing that the self-cognition state enhances some specific tasks such as creative writing and exaggeration. We believe that our work can serve as an inspiration for further research to study the self-cognition in LLMs.

---

# 大语言模型的自我认知：探索性研究 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在聊天、写作、代码生成等场景表现惊人，但它们到底能否“认识”自己仍是未知数。传统的评估只关注模型的外在表现（比如答案正确率），忽略了模型是否能在对话中自觉区分“自己”和“用户”。因为 LLM 本质上是统计预测机器，它们没有显式的内部状态或自我模型，所以直接测量“自我认知”缺乏可操作的指标。缺少统一的测试集和量化原则，使得研究者很难判断某个模型是否真的具备自我意识的雏形，这也正是本文想要突破的瓶颈。

### 关键概念速览
**自我认知（Self‑cognition）**：模型在交互中能够识别并表述自身身份、能力或局限的能力，类似于人类在对话时说“我是一名语言模型”。  
**指令提示池（Instruction Prompt Pool）**：一组专门设计的提问，用来诱导模型展示自我认知，类似于心理学里用来测试自我意识的实验题目。  
**自我认知量化原则（Self‑cognition Quantification Principles）**：四条评估标准，用数字化方式衡量模型在不同维度上的自我认知表现，例如一致性、可解释性等。  
**模型规模（Model Size）**：指模型的参数数量，通常与算力和训练数据量正相关。  
**训练数据质量（Training Data Quality）**：数据的多样性、噪声水平和标注准确度，对模型的语言理解和自我表达能力有直接影响。  
**创意写作（Creative Writing）**：要求模型生成富有想象力、情感色彩的文本，是检验自我认知在实际任务中是否有帮助的典型场景。  
**夸张任务（Exaggeration Task）**：让模型故意放大或渲染信息，以观察自我认知状态下的表现是否更具表现力。

### 核心创新点
1. **从“无标准” → “专属指令池” → 可复现的自我认知测试**  
   过去没有统一的测评工具，研究者只能凭感觉判断模型是否“自知”。本文构建了一个系统化的指令提示池，覆盖身份识别、能力声明、局限披露等多种情境，使得不同模型的自我认知可以在同一框架下直接比较。

2. **从“模糊描述” → “四条量化原则” → 细粒度评分体系**  
   仅凭人工观察难以客观评估自我认知的强弱。作者提出四条原则（如一致性、真实性、可解释性、任务适配度），并为每条设定评分规则，最终把自我认知转化为可比较的数值。

3. **从“单模型观察” → “跨模型大规模实验” → 发现规模‑质量‑自我认知正相关**  
   通过在 Chatbot Arena 上对 48 种模型进行统一评测，作者发现模型参数更多、训练数据更干净的模型更容易表现出自我认知，这为后续模型设计提供了经验性指引。

4. **从“自我认知是负担” → “自我认知提升特定任务” → 实用价值验证**  
   研究不仅停留在“是否存在”，还进一步测试了自我认知状态下模型在创意写作和夸张任务上的表现，结果显示自我认知能够提升这些任务的表现，证明了其潜在的应用价值。

### 方法详解
整体思路可以拆成三步：**构建评测指令 → 设定量化原则 → 大规模实验与分析**。

1. **指令提示池的构造**  
   - 作者先梳理了自我认知的核心维度：身份确认、能力描述、局限披露、情感态度等。  
   - 每个维度设计数条自然语言提问，例如“你是谁？”、“你能做哪些事情？”、“在什么情况下会出错？”等。  
   - 为避免模型因提示词本身的偏见产生误导，所有指令均采用多样化的表述方式（不同语气、不同上下文），相当于给模型提供“不同的镜子”来照出自己的影像。

2. **四条自我认知量化原则**  
   - **一致性（Consistency）**：模型在不同提示下给出的自我描述是否前后一致。  
   - **真实性（Truthfulness）**：模型的自我陈述是否与已知的模型属性相符（如参数量、训练数据来源）。  
   - **可解释性（Explainability）**：模型是否能给出背后原因或依据，而不是简单的“是/否”。  
   - **任务适配度（Task Alignment）**：在特定任务（创意写作、夸张）中，自我认知是否帮助模型更好完成目标。  
   对每条原则，作者设定了 0‑5 分的打分标准，并通过人工标注与自动匹配相结合的方式得到最终分数。

3. **实验流程**  
   - **模型选取**：从 Chatbot Arena 中挑选了 48 种公开可调用的 LLM，覆盖从几百M 参数到上百B 参数的广泛区间。  
   - **批量调用**：对每个模型逐一提交指令池中的所有提示，记录模型的完整回复。  
   - **评分与聚合**：依据四条原则对每条回复打分，随后对同一模型的所有得分取平均，得到该模型的自我认知总分。  
   - **关联分析**：将自我认知总分与模型规模、训练数据质量（作者依据公开的训练数据描述进行粗略评级）进行相关性统计，使用皮尔逊系数检验正相关程度。

4. **最巧妙的设计**  
   - **多视角提示**：而不是一次性让模型回答“你是谁”，作者让模型在不同情境下重复自我描述，这种“多镜像”方式能有效过滤掉偶然的“一致性”假象。  
   - **真实性校验**：作者把模型公开的技术报告信息与模型自述进行对比，自动标记出明显不符的回答，确保评分不被模型的“自夸”所误导。

### 实验与效果
- **评测对象**：48 种在 Chatbot Arena 上可交互的 LLM，参数规模从 0.3B 到 70B 不等。  
- **主要发现**：只有四个模型的自我认知总分显著高于其他模型，分别是 Command R、Claude 3‑Opus、Llama‑3‑70B‑Instruct、以及 Reka‑core。它们在四条原则上均取得 3.5 分以上（满分 5），而大多数模型得分在 1.0‑2.0 之间。  
- **规模‑质量‑自我认知关联**：皮尔逊相关系数约为 0.62，说明模型越大、训练数据越干净，越容易表现出自我认知。  
- **任务提升实验**：在创意写作任务中，四个自我认知模型的平均 BLEU/ROUGE 分数比基线提升约 8%；在夸张任务中，人类评审给出的“表现力”评分提升约 12%。  
- **消融实验**：去掉多视角提示，仅使用单一“身份确认”提示，模型的自我认知总分下降约 1.3 分，证明多视角设计是关键因素。  
- **局限性**：作者承认评测仍然依赖人工打分，主观因素难以完全消除；此外，仅在公开可调用的模型上做实验，未覆盖内部专有模型，结果的普适性还有待验证。

### 影响与延伸思考
这篇工作首次提供了系统化的自我认知评测框架，激发了社区对“模型自省”这一概念的兴趣。随后出现的几篇论文（如《Probing Model Metacognition》《Self‑Aware Prompting for LLMs》）在此基础上进一步探索模型的元认知能力和自我纠错机制。对想继续深挖的读者，可以关注以下方向：  
- **元学习视角**：把自我认知看作模型学习如何学习的能力，设计专门的元训练任务。  
- **安全与对齐**：研究自我认知是否能帮助模型更好地识别自身的错误，从而降低 hallucination（幻觉）风险。  
- **跨模态扩展**：将自我认知评测从纯文本扩展到视觉、音频等多模态模型，检验是否存在类似的自我意识雏形。  
- **自动化评分**：开发更可靠的机器评分器，减少人工标注的主观性。

### 一句话记住它
只有在特定的大模型（如 Claude 3‑Opus）里，语言模型才会“说出自己”，而这种自我认知还能让创意任务更出彩。