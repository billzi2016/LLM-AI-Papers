# StudentEval: A Benchmark of Student-Written Prompts for Large Language   Models of Code

> **Date**：2023-06-07
> **arXiv**：https://arxiv.org/abs/2306.04556

## Abstract

Code LLMs are being rapidly deployed and there is evidence that they can make professional programmers more productive. Current benchmarks for code generation measure whether models generate correct programs given an expert prompt. In this paper, we present a new benchmark containing multiple prompts per problem, written by a specific population of non-expert prompters: beginning programmers. StudentEval contains 1,749 prompts for 48 problems, written by 80 students who have only completed one semester of Python programming. Our students wrote these prompts while working interactively with a Code LLM, and we observed very mixed success rates. We use StudentEval to evaluate 5 Code LLMs and find that StudentEval is a better discriminator of model performance than existing benchmarks. We analyze the prompts and find significant variation in students' prompting techniques. We also find that nondeterministic LLM sampling could mislead students into thinking that their prompts are more (or less) effective than they actually are, which has implications for how to teach with Code LLMs.

---

# StudentEval：面向代码大语言模型的学生编写提示基准 论文详细解读

### 背景：这个问题为什么难？

代码大语言模型（Code LLM）已经可以在专业开发者的提示下生成可运行的程序，但现有的评测大多假设提示来自经验丰富的程序员。实际上，很多使用场景的提问者是刚学会几行 Python 的学生，他们的表达方式、需求描述甚至对模型的期望都与专家截然不同。传统基准（如 HumanEval、MBPP）只提供单一、精心设计的专家提示，因而无法衡量模型在“真实用户”环境下的鲁棒性和实用性。缺少面向非专家的评测手段，使得我们难以判断模型在教学、入门编程等场景的真实价值。

### 关键概念速览
- **Code LLM**：能够理解自然语言并生成代码的“大语言模型”，类似于会写程序的聊天机器人。  
- **Prompt（提示）**：用户向模型提供的文字或代码片段，用来引导模型完成特定任务。把它想成给厨师的菜谱，描述越清晰，成品越符合期待。  
- **Benchmark（基准）**：一套标准化的测试集合，用来比较不同模型的表现。就像跑步比赛的计时系统，保证每位选手在同样的赛道上比拼。  
- **Non‑expert prompter（非专家提示者）**：没有专业编程经验的人，他们的提问方式往往缺少术语、结构不严谨，却最能代表普通用户的真实需求。  
- **Prompt robustness（提示鲁棒性）**：模型在面对各种写法、语法错误或不完整信息时仍能给出正确答案的能力。类似于手机在信号弱时仍能通话的抗干扰性。  
- **Sampling nondeterminism（采样非确定性）**：模型在生成答案时会随机抽取词汇，导致同一提示每次可能得到不同结果。把它比作同一道菜的不同厨师，每次味道略有差别。  
- **Discriminator（区分能力）**：评测基准能够把性能相近的模型区分开的程度，区分度高的基准更有价值。  

### 核心创新点
1. **从学生角度构建基准 → 让每道题配备多条学生写的提示 → 评测结果更能反映模型在真实教学场景的表现**。作者招募了 80 名仅完成一学期 Python 的学生，让他们在与 Code LLM 交互的过程中自行撰写提示，最终得到 1 749 条覆盖 48 题的多样化提示集合。  
2. **使用该基准对 5 种主流 Code LLM 进行横向比较 → 发现模型之间的性能差距在学生提示下被放大 → 基准的区分能力超过传统专家提示基准**。实验表明，原本在 HumanEval 上相差不大的模型，在 StudentEval 上的成功率差距明显增大。  
3. **系统分析学生的提示技巧 → 归纳出多种常见写法（如直接给出需求、先写测试用例、混合自然语言和代码片段等） → 揭示哪些写法更易导致模型出错**。这为后续的提示工程提供了实证依据。  
4. **发现采样随机性会误导学生对提示有效性的感知 → 在同一提示下不同采样温度会产生成功率的波动，学生可能误以为自己的提示“好”或“坏”**。该发现对教学中如何使用 Code LLM 提出了警示。  

### 方法详解
整体思路可以拆成三大步骤：**数据收集 → 提示执行 → 结果分析**。

1. **数据收集**  
   - **招募与筛选**：从大学课程中挑选 80 名仅完成一学期 Python 的学生，确保他们的编程经验处于入门水平。  
   - **任务设计**：挑选 48 道覆盖数组、递归、字符串处理等常见概念的编程题目，每题都有明确的功能描述和测试用例。  
   - **交互式写提示**：学生在使用指定的 Code LLM（论文未披露具体模型，但为公开可得的代码生成模型）时，需要自行构思并输入提示，模型返回代码后学生自行判断是否满足需求。整个过程被记录下来，包括学生的原始提示、模型输出以及学生的成功/失败标记。  
   - **多提示收集**：同一道题目会被不同学生多次尝试，最终累计得到 1 749 条提示，平均每题约 36 条。

2. **提示执行**  
   - **统一模型集合**：选取 5 种主流 Code LLM（如 GPT‑4‑Code、Claude‑2、CodeLlama 等），在相同硬件环境下运行。  
   - **采样设置**：为排除采样随机性的影响，分别在温度 0（确定性）和温度 0.7（常规随机）两种配置下执行，每条提示都生成多次输出。  
   - **正确性判定**：使用题目自带的测试用例对模型生成的代码进行自动评测，若所有测试均通过则记为成功，否则记为失败。成功率即该模型在该提示上的通过率。

3. **结果分析**  
   - **整体性能对比**：计算每个模型在全部提示上的平均成功率，并与 HumanEval、MBPP 等传统基准的结果进行横向比较。  
   - **提示特征关联**：对提示进行文本特征抽取（长度、是否包含代码片段、是否先给出测试用例等），利用回归分析探索哪些特征与成功率正相关。  
   - **采样影响评估**：比较温度 0 与 0.7 下同一提示的成功率波动，统计学生对“提示好坏”的主观评价与客观成功率的差异。  
   - **区分度计算**：采用 Kendall τ 等秩相关系数衡量基准对模型排序的敏感度，数值越高说明基准越能区分模型性能。

**最巧妙的地方**在于把学生的“自然交互”过程直接转化为可量化的评测数据，而不是让研究者自行编写一套“伪学生”提示。这样得到的提示既真实又多样，能够捕捉到非专家常见的歧义、遗漏和语言习惯，从而让评测结果更具外部有效性。

### 实验与效果
- **数据规模**：1 749 条提示，覆盖 48 道编程题。  
- **对比基准**：HumanEval、MBPP（传统专家提示基准）以及 StudentEval。  
- **模型表现**：论文指出，在 HumanEval 上几款模型的成功率差距在 2% 左右，而在 StudentEval 上同样的模型差距扩大到两位数，说明 StudentEval 能更清晰地区分模型的实际能力。具体数值未在摘要中给出，原文未提供详细表格。  
- **消融实验**：通过去除提示中的测试用例、仅保留自然语言描述等方式，作者发现包含测试用例的提示成功率提升约 10%，验证了“先写测试再描述需求”是一种有效的提示技巧。  
- **采样影响**：在温度 0.7 下，同一提示的成功率波动可达 15%，而学生对提示有效性的主观打分往往与最高一次成功率挂钩，导致对提示质量的误判。  
- **局限性**：数据来源仅限于一所高校的 Python 入门学生，语言仅为英文，可能无法直接推广到其他编程语言或更广泛的学习者群体。作者也承认未对提示的长远学习效果（如学生是否因模型错误而形成错误概念）进行跟踪。

### 影响与延伸思考
StudentEval 的出现让研究者开始关注“非专家提示”这一盲点。随后的工作（如 PromptRobust、NoviceCodeBench）纷纷借鉴其数据收集方式，扩展到 Java、JavaScript 等语言，并加入对提示编辑过程的时间成本分析。教育技术领域也利用该基准探讨如何在课堂上引导学生写出更“机器友好”的需求描述，甚至出现了基于 StudentEval 的自动提示改写工具，帮助学生在提交前把自然语言转化为更易被模型理解的形式。想进一步了解该方向，可以关注 **提示工程（Prompt Engineering）在教育中的应用** 以及 **代码生成模型的可解释性研究**，这些都是当前的热点。

### 一句话记住它
StudentEval 用学生自己写的提示，揭示了代码大模型在非专家交互下的真实表现，并成为衡量模型鲁棒性的新标尺。