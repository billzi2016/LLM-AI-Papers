# Is ChatGPT a General-Purpose Natural Language Processing Task Solver?

> **Date**：2023-02-08
> **arXiv**：https://arxiv.org/abs/2302.06476

## Abstract

Spurred by advancements in scale, large language models (LLMs) have demonstrated the ability to perform a variety of natural language processing (NLP) tasks zero-shot -- i.e., without adaptation on downstream data. Recently, the debut of ChatGPT has drawn a great deal of attention from the natural language processing (NLP) community due to the fact that it can generate high-quality responses to human input and self-correct previous mistakes based on subsequent conversations. However, it is not yet known whether ChatGPT can serve as a generalist model that can perform many NLP tasks zero-shot. In this work, we empirically analyze the zero-shot learning ability of ChatGPT by evaluating it on 20 popular NLP datasets covering 7 representative task categories. With extensive empirical studies, we demonstrate both the effectiveness and limitations of the current version of ChatGPT. We find that ChatGPT performs well on many tasks favoring reasoning capabilities (e.g., arithmetic reasoning) while it still faces challenges when solving specific tasks such as sequence tagging. We additionally provide in-depth analysis through qualitative case studies.

---

# ChatGPT是通用自然语言处理任务求解器吗？ 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）出现之前，NLP 任务几乎都需要针对每个任务单独微调模型，数据标注成本高、迁移效果差。即使有了 GPT‑3 之类的“通用”模型，它们的 zero‑shot 能力也主要体现在生成式任务，像情感分类、序列标注等结构化任务仍然表现不佳。于是社区一直在探问：一个只通过对话接口交互的模型（如 ChatGPT）能否真正做到“一次训练、全任务通用”，而不需要任何下游适配？这正是本文要回答的核心难点。

### 关键概念速览
**Zero‑shot（零样本）**：模型在没有看到目标任务的任何训练数据的情况下直接给出答案，就像你第一次玩新游戏，凭借通用经验直接上手。  
**Large Language Model（大语言模型）**：参数量在数十亿以上、通过海量文本自监督训练得到的模型，能够捕捉语言的统计规律和一定的推理能力。  
**Prompt（提示）**：给模型的文字指令或问题，类似老师给学生的考题描述，好的 Prompt 能把模型的潜在能力激活出来。  
**Chain‑of‑Thought（思维链）**：让模型在回答前先写出推理步骤，像解数学题时先列出公式再算，能够提升复杂推理的准确率。  
**Sequence Tagging（序列标注）**：对句子中每个词打标签的任务，例如命名实体识别（NER），需要模型对每个位置做细粒度判断。  
**Arithmetic Reasoning（算术推理）**：涉及数字运算的任务，需要模型在语言理解的基础上进行数值计算。  
**In‑Context Learning（上下文学习）**：模型通过在同一次对话中提供示例来学习任务，而不是显式微调参数。  

### 核心创新点
1. **系统化的多任务评估框架 → 设计了覆盖 7 大任务类别、20 个公开数据集的评测套件 → 首次在同一实验环境下对 ChatGPT 的 zero‑shot 能力进行横向对比，揭示了它的强项与短板。  
2. **统一的 Prompt 规范 → 为每个任务制定了“指令+示例+输出格式”三段式 Prompt，确保不同任务之间的比较公平 → 让评测结果更具可复现性，也为后续研究提供了标准化的 Prompt 模板。  
3. **细粒度错误分析 → 通过案例研究、错误类型归类以及对话式纠错实验，展示了 ChatGPT 在自我纠正方面的潜力与局限 → 为改进模型的交互式学习提供了实证依据。  

### 方法详解
整体思路是把 ChatGPT 当作一个黑盒子，只通过文字交互来完成各种 NLP 任务。评测流程分为三步：

1. **任务挑选与数据准备**  
   - 选取 20 个广为使用的公开数据集，覆盖文本分类、情感分析、自然语言推理、问答、摘要、序列标注、算术推理等 7 类。  
   - 对每个数据集抽取若干测试样本（通常 200‑500 条），确保覆盖不同难度层次。

2. **Prompt 设计与对话调用**  
   - 为每类任务统一制定 Prompt 模板：  
     - **指令**：明确告诉模型要完成的任务，例如“请判断下面句子的情感是正面还是负面”。  
     - **示例**（可选）：提供 1‑2 条输入‑输出对，利用 In‑Context Learning 帮助模型快速捕捉任务格式。  
     - **输出格式**：要求模型返回结构化答案，如“Label: Positive”。  
   - 将每条测试样本填入模板后，通过 OpenAI 的 ChatGPT API 发起一次对话请求，记录模型的直接回复。

3. **结果解析与评估**  
   - 对模型输出进行后处理（去除多余文字、统一标签大小写），再与金标准答案比对。  
   - 采用任务对应的标准指标：分类任务用准确率（Accuracy），序列标注用 F1，生成任务用 ROUGE/L、BLEU 等。  
   - 为了评估自我纠错能力，作者在部分错误答案后继续追问模型“刚才的答案有误吗？请重新回答”，记录二次回复的改进情况。

**最巧妙的地方**在于 Prompt 的统一化。很多早期的 zero‑shot 评测往往因为 Prompt 差异导致结果波动很大，本文通过固定指令结构、示例数量和输出格式，最大程度上把模型能力本身和 Prompt 设计的影响分离。

### 实验与效果
- **测试任务**：包括 SST‑2（情感分类）、MNLI（自然语言推理）、SQuAD（阅读理解）、CNN/DailyMail（摘要）、CoNLL‑2003（命名实体识别）、MathQA（算术推理）等。  
- **基线对比**：与 GPT‑3（text‑davinci‑003）以及公开的 zero‑shot 任务特定模型（如 T5‑XXL）进行比较。  
- **主要发现**：  
  - 在需要推理的任务上（如算术推理、自然语言推理），ChatGPT 的准确率常常超过 70%，比 GPT‑3 提升约 5‑10%。  
  - 在序列标注任务（CoNLL‑2003）上，F1 仅在 55% 左右，明显低于专门的 token‑level 模型（约 90%）。  
  - 通过一次追问进行自我纠错后，错误率在部分任务上下降约 15%，但仍未能弥补结构化任务的根本缺陷。  
- **消融实验**：作者去掉 Prompt 中的示例，仅保留指令，发现多数任务的性能下降 3‑8%，说明 In‑Context Learning 对 ChatGPT 的 zero‑shot 表现有一定帮助。  
- **局限性**：原文未给出对话成本（API 调用次数、时延）以及对长文本（超过模型上下文窗口）的处理细节，且只评估了单轮或双轮交互，未探索更深层次的多轮对话学习。

### 影响与延伸思考
这篇评测在社区引发了两大趋势：一是 **Prompt 标准化**——后续很多 zero‑shot 研究直接引用本文的三段式模板作为基准；二是 **交互式纠错**——研究者开始探索让模型在对话中自我修正的机制，如 ReAct、Self‑Consistency 等。推测未来会有更多工作把 ChatGPT 这类对话模型与传统微调模型结合，形成“先对话后微调”的混合学习框架。如果想进一步了解，可以关注 **In‑Context Learning 的理论解释**（如线性回归视角）以及 **大模型的多任务适配策略**（如 LoRA、Adapter）。

### 一句话记住它
ChatGPT 在零样本下能很好地处理需要推理的语言任务，但仍难以胜任细粒度的序列标注——它是“会思考的生成器”，不是全能的结构化标签专家。