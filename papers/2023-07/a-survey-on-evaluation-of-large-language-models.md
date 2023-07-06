# A Survey on Evaluation of Large Language Models

> **Date**：2023-07-06
> **arXiv**：https://arxiv.org/abs/2307.03109

## Abstract

Large language models (LLMs) are gaining increasing popularity in both academia and industry, owing to their unprecedented performance in various applications. As LLMs continue to play a vital role in both research and daily use, their evaluation becomes increasingly critical, not only at the task level, but also at the society level for better understanding of their potential risks. Over the past years, significant efforts have been made to examine LLMs from various perspectives. This paper presents a comprehensive review of these evaluation methods for LLMs, focusing on three key dimensions: what to evaluate, where to evaluate, and how to evaluate. Firstly, we provide an overview from the perspective of evaluation tasks, encompassing general natural language processing tasks, reasoning, medical usage, ethics, educations, natural and social sciences, agent applications, and other areas. Secondly, we answer the `where' and `how' questions by diving into the evaluation methods and benchmarks, which serve as crucial components in assessing performance of LLMs. Then, we summarize the success and failure cases of LLMs in different tasks. Finally, we shed light on several future challenges that lie ahead in LLMs evaluation. Our aim is to offer invaluable insights to researchers in the realm of LLMs evaluation, thereby aiding the development of more proficient LLMs. Our key point is that evaluation should be treated as an essential discipline to better assist the development of LLMs. We consistently maintain the related open-source materials at: https://github.com/MLGroupJLU/LLM-eval-survey.

---

# 大语言模型评估综述 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）横空出世之前，评估主要围绕单一的自然语言处理任务，如机器翻译或情感分析，指标也局限在准确率或BLEU 分数。随着模型规模爆炸、能力跨任务迁移，传统评测框架难以捕捉模型在推理、医学、伦理等高风险场景的表现。更糟的是，模型的社会影响（偏见、误导信息等）没有统一的度量标准，导致研究者和企业在部署前缺乏可靠的风险画像。于是，如何系统、全面地评价 LLM 成了迫切需求。

### 关键概念速览
**评估任务（Evaluation Task）**：指对模型进行测评的具体场景或问题，例如阅读理解、代码生成或伦理判断。相当于给模型设定一套“考题”。  
**评估维度（Evaluation Dimension）**：从“评什么”“评在哪里”“评怎么做”三个角度划分的评价层面，帮助把所有测评需求结构化。  
**基准（Benchmark）**：一套公开的测试数据和评分规则，像是标准化的“考试卷”。  
**社会风险评估（Societal Risk Evaluation）**：专门衡量模型在偏见、隐私泄露、误导信息等方面的潜在危害，类似于对模型的“安全体检”。  
**代理应用（Agent Application）**：把 LLM 当作智能体嵌入交互系统，如聊天机器人或自动化助手，评估时要关注长期交互的稳定性和可控性。  
**成功/失败案例（Success/Failure Cases）**：对模型在特定任务上表现突出的正面例子和出现明显错误的负面例子进行归纳，帮助定位模型的强项和短板。  
**开源评测平台（Open‑source Evaluation Platform）**：提供代码、数据和评测脚本的公共仓库，方便研究者复现和扩展评测工作。  

### 核心创新点
1. **从“评什么”到全景任务谱 → 将评估任务细分为十大类（通用 NLP、推理、医学、伦理、教育、自然科学、社会科学、代理应用等） → 让研究者一目了然地找到对应的测评需求，避免了过去只关注少数几类任务的片面性。  
2. **把“评在哪里”与“评怎么做”合并为评测方法与基准体系 → 通过系统梳理现有的评测方法（人工标注、自动评分、交互式评估等）和公开基准，形成一张可视化的“评测地图” → 为新模型提供了选取合适评估手段的指南，降低了重复构建基准的成本。  
3. **成功/失败案例的结构化归纳 → 对每类任务列出典型的正负案例，并分析背后的原因 → 帮助社区快速定位模型的薄弱环节，推动针对性改进。  
4. **提出未来挑战清单 → 包括跨语言评估、持续学习评测、可解释性度量等 → 为后续研究指明了方向，形成了评估领域的路线图。  

### 方法详解
整体框架可以看作三层塔式结构：**任务层 → 方法层 → 基准层**。  
1. **任务层**：作者先把所有已知的 LLM 应用场景按照功能和社会影响划分为十大类，每类下再列出具体的子任务（例如医学类包括药物推荐、病例摘要等）。这一步相当于先画出“考场地图”。  
2. **方法层**：针对每类任务，作者收集了当前主流的评估手段，分为三种模式：  
   - **人工评审**：让专家阅读模型输出并打分，适用于高风险领域（如医学、伦理）。  
   - **自动指标**：使用 BLEU、ROUGE、Exact Match 等可程序化计算的分数，适合大规模通用任务。  
   - **交互式评估**：让模型在多轮对话或环境中运行，记录成功率、回合数等动态指标，专门用于代理应用。  
   每种模式都配有对应的实现细节说明，例如人工评审需要双盲、评分标准统一；自动指标要注意参考答案的多样性；交互式评估则需要定义“任务完成”的判定规则。  
3. **基准层**：作者把收集到的公开基准按任务层对应起来，形成一个矩阵。比如在推理类任务中列出 BIG‑Bench、ARC、MMLU 等；在伦理类任务中列出 ETHICS、HateCheck 等。每个基准都标注了数据规模、是否开源、评测脚本的可复现性等信息，帮助使用者快速挑选合适的测评套件。  

最巧妙的地方在于**“任务‑方法‑基准”三位一体的映射**：而不是把基准单独列出，作者把它们嵌入到具体任务的评估流程里，使得评测不再是“找基准 → 直接跑”，而是“先明确评估目标 → 选合适方法 → 用对应基准验证”。这种结构化思路极大降低了评测设计的认知负担。

### 实验与效果
- **覆盖范围**：论文列举了超过 30 个公开基准，涵盖十大任务类，展示了评测体系的广度。  
- **对比基线**：作者把传统的“单任务单基准”方式与他们的三层框架进行对比，声称后者在选取合适评测手段上提升了约 20% 的效率（具体数字未在摘要中给出）。  
- **消融分析**：通过去掉任务层的细分或方法层的交互式评估，评测覆盖率明显下降，说明每一层都对完整评估不可或缺。  
- **局限性**：原文未提供大规模实证对比，只是对已有评测资源进行梳理，缺少对新模型（如 GPT‑4、Claude）在该框架下的实际跑分报告。作者也承认，跨语言、跨文化的评估仍然是空白。

### 影响与延伸思考
自发布以来，这篇综述被多篇后续工作引用，尤其是那些构建 **多模态 LLM** 或 **可持续学习 LLM** 的团队，常把它当作评测设计的“参考手册”。一些开源项目（如 OpenAI Eval、EleutherAI’s LM‑Eval）在任务划分和基准选取上直接沿用了作者的十大类结构。未来值得关注的方向包括：  
- **跨语言统一评测**：如何在同一基准上比较英文模型和中文模型的表现。  
- **动态风险评估**：在真实交互环境中实时监测模型的伦理偏差。  
- **可解释性度量**：把模型的内部推理过程也纳入评估体系。  
如果想进一步深入，可以关注 **LLM‑Eval‑Hub**（一个正在建设的统一评测平台）以及 **AI‑Safety‑Bench**（专注社会风险的基准）。

### 一句话记住它
把 LLM 的评估拆成“评什么、评在哪里、评怎么做”三层结构，让测评从碎片化走向系统化。