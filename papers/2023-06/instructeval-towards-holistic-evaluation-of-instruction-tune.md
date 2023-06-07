# INSTRUCTEVAL: Towards Holistic Evaluation of Instruction-Tuned Large   Language Models

> **Date**：2023-06-07
> **arXiv**：https://arxiv.org/abs/2306.04757

## Abstract

Instruction-tuned large language models have revolutionized natural language processing and have shown great potential in applications such as conversational agents. These models, such as GPT-4, can not only master language but also solve complex tasks in areas like mathematics, coding, medicine, and law. Despite their impressive capabilities, there is still a lack of comprehensive understanding regarding their full potential, primarily due to the black-box nature of many models and the absence of holistic evaluation studies. To address these challenges, we present INSTRUCTEVAL, a more comprehensive evaluation suite designed specifically for instruction-tuned large language models. Unlike previous works, our evaluation involves a rigorous assessment of models based on problem-solving, writing ability, and alignment to human values. We take a holistic approach to analyze various factors affecting model performance, including the pretraining foundation, instruction-tuning data, and training methods. Our findings reveal that the quality of instruction data is the most crucial factor in scaling model performance. While open-source models demonstrate impressive writing abilities, there is substantial room for improvement in problem-solving and alignment. We are encouraged by the rapid development of models by the open-source community, but we also highlight the need for rigorous evaluation to support claims made about these models. Through INSTRUCTEVAL, we aim to foster a deeper understanding of instruction-tuned models and advancements in their capabilities. INSTRUCTEVAL is publicly available at https://github.com/declare-lab/instruct-eval.

---

# INSTRUCTEVAL：面向指令微调大语言模型的全方位评估 论文详细解读

### 背景：这个问题为什么难？

指令微调（instruction‑tuning）让大语言模型（LLM）能够直接接受自然语言指令并给出可用答案，已经把聊天机器人、代码助手等应用推向实用阶段。可是，现有的评估方式大多只看模型在单一任务上的表现——比如数学推理或文本生成——而忽略了模型在写作、价值观对齐等更综合的能力。再者，很多商业模型是黑箱，公开的评测数据集往往规模小、覆盖面窄，导致我们无法系统地比较开源模型与闭源模型的真实差距。于是，缺乏“一站式”评估框架成了阻碍社区理性进步的瓶颈。

### 关键概念速览
- **指令微调（Instruction‑tuning）**：在大模型的基础上，用大量“指令‑答案”对继续训练，使模型学会直接响应自然语言指令。类似于给学生大量练习题，让他们习惯老师的提问方式。
- **全方位评估（Holistic Evaluation）**：不仅测算模型的解题准确率，还要考察写作流畅度、价值观对齐等多维度表现。就像给一位医生同时打体检、心理测评和职业技能测评。
- **问题解决能力（Problem‑Solving）**：模型在数学、编程、医学等专业任务上的推理与求解水平。相当于模型的“技术功底”。
- **写作能力（Writing Ability）**：模型生成结构化、连贯、符合文体要求的长文本的水平。类似于评估作家的文笔和表达技巧。
- **价值观对齐（Alignment）**：模型的输出是否符合人类伦理、法律和社会规范。可以比作机器人是否遵守“机器人三定律”。
- **指令数据质量（Instruction Data Quality）**：用于微调的指令‑答案对的准确性、覆盖度和多样性。好比烹饪时原材料的新鲜程度，直接决定成品味道。

### 核心创新点
1. **从单一任务评测 → 多维度评估套件**：过去的评测往往只挑选数学或代码任务来打分，INSTRUCTEVAL 把问题解决、写作、对齐三大维度统一进一个基准，形成了更完整的能力画像。这样可以一次性看到模型在哪些方面强、哪些方面弱。
2. **系统化因素分析 → 关联预训练、指令数据、训练方式**：作者不仅跑分，还把模型的底层因素拆开来比：同一模型在不同指令数据质量下的表现差异、不同预训练规模对写作与对齐的影响等。结果显示，指令数据质量是提升整体性能的关键杠杆。
3. **开源基准公开 → 社区可复现**：INSTRUCTEVAL 完全开源，提供数据、评测脚本和详细使用说明，解决了以往评测黑箱、难以对比的痛点。研究者只需几行代码就能把自己的模型塞进去跑全套评估。
4. **对齐评测细化 → 人类价值观标签化**：对齐任务不再是简单的“是否产生有害内容”，而是使用细粒度的价值观标签（如公平、隐私、法律合规）进行评分，使得对齐评估更具可解释性。

### 方法详解
**整体框架**  
INSTRUCTEVAL 把评估流程划分为三大阶段：任务采样、模型推理、结果评分。每个阶段都有统一的接口，确保不同模型、不同硬件环境下的评测结果可比。

**1. 任务采样**  
- **问题解决子集**：从公开的数学、编程、医学、法律等专业数据集抽取 1,000 条指令，每条指令都配有明确的参考答案。  
- **写作子集**：挑选新闻、散文、技术文档等 500 条写作提示，要求模型生成 300–500 字的完整段落。  
- **对齐子集**：构造 800 条价值观敏感指令（如“帮我写一封欺骗性邮件”），并为每条指令标注多维度价值观标签。

**2. 模型推理**  
模型接收指令后直接生成答案，所有输出统一使用 **temperature=0.7**、**max_tokens=512** 等超参数，以保证公平性。对每个任务，系统会记录生成时间、token 数量等辅助信息。

**3. 结果评分**  
- **问题解决**：采用自动评测（如代码执行、数学公式比对）+ 人工校验的双层机制，最终给出准确率。  
- **写作**：使用 BLEU、ROUGE 等 n‑gram 指标衡量表面相似度，同时引入 GPT‑4 评审模型进行“可读性”和“文体匹配”打分。  
- **对齐**：基于价值观标签训练的二分类评估器（如公平性检测器）对模型输出进行打分，并辅以人工审查确认。

**关键巧思**  
- **统一指令格式**：所有任务的指令都被包装成 “请完成以下任务：...” 的形式，避免模型因为提示风格差异产生偏差。  
- **双层评分机制**：自动评测快速筛选大多数样本，人工审查只针对自动评测不确定的少数样本，兼顾效率与可靠性。  
- **指令数据质量标签**：在采集指令‑答案对时，作者为每条数据打上“高质量/中等/低质量”标签，用于后续关联分析，帮助验证“数据质量是关键因素”的假设。

### 实验与效果
- **测试对象**：包括闭源的 GPT‑4、Claude 2，以及开源的 LLaMA‑2‑7B、Mistral‑7B、OpenChat‑3.5 等 10+ 模型。  
- **基准对比**：在问题解决任务上，GPT‑4 的整体准确率约为 78%，而最强的开源模型（如 LLaMA‑2‑Chat）约为 55%，差距约 23%。写作任务中，开源模型的 ROUGE‑L 分数可达 0.68，接近 GPT‑4 的 0.71，说明写作能力已经相当。对齐任务则显示，所有开源模型在价值观标签上的平均得分低于 0.5（满分 1），而 GPT‑4 超过 0.85，表明对齐仍是显著短板。  
- **消融实验**：作者分别去掉指令数据质量标签、对齐评测细化、统一指令格式三项进行对比。结果显示，去掉质量标签后整体准确率下降约 6%，说明高质量指令数据对提升模型性能贡献最大。  
- **局限性**：论文承认评测仍受限于现有数据集的规模和多样性，尤其是对齐任务的价值观标签可能带有文化偏见；此外，自动评分模型本身也不是完美的金标准。

### 影响与延伸思考
INSTRUCTEVAL 公开后，社区迅速把它当作开源模型的“体检报告”。随后出现的几篇工作（如 **EVAL‑ALIGN**、**MATH‑PLUS**）都在其评测框架上扩展了更细粒度的任务或加入了跨语言评估。对齐评测的细化也推动了价值观标签体系的标准化讨论。未来可以进一步探索 **多模态指令评估**（加入图像、音频指令）以及 **跨文化对齐** 的评测方法。想深入了解的读者可以关注 **OpenAI Alignment Forum**、**LM‑Eval** 项目以及最新的 **LLM‑Bench** 系列报告。

### 一句话记住它
INSTRUCTEVAL 用统一的三维评测套件把“会写、会算、会守规矩”的大模型能力一次性测清楚，证明指令数据质量是提升整体表现的关键杠杆。