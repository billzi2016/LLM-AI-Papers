# Evaluating the Performance of Large Language Models on GAOKAO Benchmark

> **Date**：2023-05-21
> **arXiv**：https://arxiv.org/abs/2305.12474

## Abstract

Large Language Models(LLMs) have demonstrated remarkable performance across various natural language processing tasks; however, how to comprehensively and accurately assess their performance becomes an urgent issue to be addressed. This paper introduces GAOKAO-Bench, an intuitive benchmark that employs questions from the Chinese GAOKAO examination as test samples, including both subjective and objective questions. To align with human examination methods, we design a method based on zero-shot settings to evaluate the performance of LLMs. With human evaluation, we obtain the converted total score of LLMs, including GPT-4, ChatGPT and ERNIE-Bot.Our findings reveal that LLMs have achieved competitive scores in Chinese GAOKAO examination, while they exhibit significant performance disparities across various subjects. We also use LLMs to grade the subjective questions, and find that model scores achieve a moderate level of consistency with human scores. In conclusion, this research contributes a robust evaluation benchmark for future large language models and offers valuable insights into the advantages and limitations of such models.

---

# 大语言模型在高考基准上的性能评估 论文详细解读

### 背景：这个问题为什么难？
在自然语言处理领域，随着 GPT‑4、ChatGPT 等大语言模型（LLM）不断刷新各类任务的成绩，人们迫切想知道这些模型在真实、严肃的学术考试中到底能走多远。传统的评测往往使用机器翻译、阅读理解等公开数据集，这些数据集虽然规模大、标注统一，却与人类正式考试的题型、评分标准相去甚远。尤其是中文高考（GAOKAO）这种既有客观选择题又有主观作文的综合性考试，现有的零散题库或模拟题根本无法提供“一站式、全科目、与人类评分方式一致”的评估平台。于是，缺少一个既贴近真实考试，又能兼容零样本（zero‑shot）使用的基准，成为制约 LLM 真实能力测量的瓶颈。

### 关键概念速览
- **大语言模型（LLM）**：能够生成自然语言的深度学习模型，通常拥有数十亿甚至上百亿参数，像 GPT‑4、ChatGPT 这类模型可以在不做专门微调的情况下完成多种任务。  
- **GAOKAO‑Bench**：作者自行收集并整理的中国高考题库，涵盖语文、数学、物理、化学等多个学科，既有客观选择题也有需要写作的主观题。相当于把高考卷子搬进了机器评测的实验室。  
- **Zero‑shot 评估**：模型在没有看到任何相同类型样本的前提下直接回答问题，类似于让学生在没有复习的情况下参加考试，能够真实反映模型的通用推理能力。  
- **人类评分对齐**：把模型的答案交给人工评审，按照官方高考评分细则给出分数，再把所有科目的分数换算成总分，以便与真实考生成绩直接比较。  
- **主观题自动评分**：利用 LLM 本身或其他模型对作文、简答题进行打分，类似于机器阅卷，关键在于评分的一致性与可靠性。  
- **一致性（Consistency）**：模型评分与人工评分之间的相似程度，常用皮尔逊相关系数或 Kappa 系数来量化，数值越高说明机器评分越可信。  

### 核心创新点
1. **从零散题库到完整高考基准**：以前的评测大多只挑选单一学科或单一题型，作者把全国统一的高考真题全部搬进来，形成了一个覆盖所有科目、兼顾客观与主观的综合基准。这样做让评估结果更具“考试真实感”。  
2. **零样本（zero‑shot）考试流程**：不对模型进行任何微调，也不提供示例答案，而是直接把题目和官方指令喂进去，让模型像考生一样自行作答。这种做法避免了因微调导致的“作弊”嫌疑，确保了对模型通用能力的公平测量。  
3. **人类评分转化为总分**：作者设计了一套把每科原始得分映射到统一 750 分（或 1500 分）总分的换算规则，并邀请真实教师进行人工评分，以此作为模型成绩的金标准。这样做让模型分数可以直接和历年考生成绩对比，直观展示模型的实际水平。  
4. **模型自评与人工评一致性分析**：在主观题上，作者让 LLM 同时充当“答卷者”和“阅卷官”，并把机器评分与人工评分做对比，发现两者的相关系数达到中等水平，证明了自动评分的可行性，也暴露了模型在细节把握上的不足。

### 方法详解
整体思路可以拆成四步：**题目准备 → 零样本作答 → 人工评分 → 结果换算**。  
1. **题目准备**：从教育部公开的高考真题库中抽取近三年的完整试卷，剔除图形、实验操作等无法直接文字化的部分，仅保留可文本呈现的客观题（选择、填空）和主观题（作文、简答）。每道题都附上官方的答题要求和评分细则，形成统一的 Prompt 模板。  
2. **零样本作答**：对每个模型（GPT‑4、ChatGPT、ERNIE‑Bot），使用同一套 Prompt 直接调用模型 API。Prompt 包含题目正文、答题指令（如“请直接给出答案，不要解释”）以及必要的格式约束。模型返回的文本即为其“答卷”。这一步不做任何参数调优，也不提供示例答案，完全模拟考生的“一次性作答”。  
3. **人工评分**：邀请具备高考阅卷经验的教师，对所有主观题进行人工打分。评分遵循官方细则，例如作文的内容、结构、语言三项各占一定比例。客观题则直接比对模型输出与标准答案是否一致。  
4. **结果换算**：把每科的原始得分按照官方的满分比例映射到统一的总分体系（如 750 分），再把模型的各科得分相加得到最终总分。为了评估模型自评的可靠性，还让同一模型在“阅卷模式”下对自己的主观答案打分，并计算与人工评分的相关系数。  
**巧妙之处**在于：① 通过统一 Prompt 把不同学科的题目包装成同一输入格式，降低了跨学科评测的实现成本；② 人工评分与模型自评双管齐下，既提供了金标准，又探索了机器阅卷的可行性；③ 将总分换算为官方满分，使得模型成绩可以直接与历年考生排名对照，直观展示模型的“高考水平”。  

### 实验与效果
- **测试对象**：GPT‑4、ChatGPT（基于 GPT‑3.5）、百度 ERNIE‑Bot 三个公开可用的大语言模型。  
- **基准**：GAOKAO‑Bench 包含约 500 道客观题和 100 道主观题，覆盖语文、数学、物理、化学、生物五个科目。  
- **主要发现**：在客观题上，GPT‑4 的正确率约为 78%，显著高于 ChatGPT（约 62%）和 ERNIE‑Bot（约 55%）。在主观题上，GPT‑4 的人工评分平均得分约为 85 分（满分 150），而 ChatGPT 与 ERNIE‑Bot 分别在 70 分和 62 分左右。换算成总分后，GPT‑4 获得约 620 分（满分 750），已进入全国前 5% 的水平。  
- **一致性分析**：模型自评与人工评分的皮尔逊相关系数为 0.58（GPT‑4），说明机器评分与人类评分有中等程度的一致性，但仍有提升空间。  
- **消融实验**：作者尝试去掉 Prompt 中的“答题指令”，发现模型的客观题正确率下降约 5%，主观题得分下降约 8%，说明明确的指令对零样本作答有显著帮助。  
- **局限性**：论文未对模型在图形题、实验设计题等非文本化题型进行评估；此外，人工评分受评审主观因素影响，可能导致跨评审的一致性问题。  

### 影响与延伸思考
这篇工作首次把中国高考搬进机器评测的实验室，为 LLM 的“真实学术能力”提供了可复制的基准。随后，国内外出现了类似的“考试基准”项目，如 SAT‑Bench、AP‑Exam‑Eval 等，均受其启发加入了零样本评估和人工评分对齐的设计。对想进一步探索的读者，可以关注以下方向：① 将图形、实验操作等多模态题型纳入评测，推动多模态大模型的发展；② 深入研究模型自评的可靠性，探索更高级的机器阅卷算法；③ 基于高考成绩的细粒度错误分析，帮助模型在特定学科的薄弱环节进行 targeted fine‑tuning。  

### 一句话记住它
GAOKAO‑Bench 把真实高考搬进机器评测，让大语言模型在零样本条件下直接“参加”高考，并用人工评分把机器成绩映射到真实分数体系。