# DS-1000: A Natural and Reliable Benchmark for Data Science Code   Generation

> **Date**：2022-11-18
> **arXiv**：https://arxiv.org/abs/2211.11501

## Abstract

We introduce DS-1000, a code generation benchmark with a thousand data science problems spanning seven Python libraries, such as NumPy and Pandas. Compared to prior works, DS-1000 incorporates three core features. First, our problems reflect diverse, realistic, and practical use cases since we collected them from StackOverflow. Second, our automatic evaluation is highly specific (reliable) -- across all Codex-002-predicted solutions that our evaluation accept, only 1.8% of them are incorrect; we achieve this with multi-criteria metrics, checking both functional correctness by running test cases and surface-form constraints by restricting API usages or keywords. Finally, we proactively defend against memorization by slightly modifying our problems to be different from the original StackOverflow source; consequently, models cannot answer them correctly by memorizing the solutions from pre-training. The current best public system (Codex-002) achieves 43.3% accuracy, leaving ample room for improvement. We release our benchmark at https://ds1000-code-gen.github.io.

---

# DS-1000：面向数据科学代码生成的自然且可靠基准 论文详细解读

### 背景：这个问题为什么难？

在代码生成领域，已有的基准大多聚焦于算法题或通用编程语言特性，缺少真实业务场景的代表性。数据科学工作常常涉及 NumPy、Pandas 等库的组合使用，代码的正确性不仅取决于语法，还依赖于对数据结构和统计函数的细致理解。过去的评测往往只跑简单的单元测试，或者直接比对答案字符串，这导致模型可能在表面上“对”，但在实际运行时会出错。再者，公开的题目大多来源于教材或人工编写，模型可以通过记忆训练数据轻松拿高分，难以衡量真正的推理能力。于是，需要一个既贴近真实业务、又能可靠评估模型输出的基准。

### 关键概念速览
- **代码生成基准**：用来衡量模型从自然语言描述生成可执行代码的能力，就像跑步比赛的计时表，提供统一的评价标准。  
- **多库数据科学任务**：指同时使用多个 Python 科学计算库（如 NumPy、Pandas、Matplotlib 等）完成的数据处理或分析任务，类似厨房里要同时用刀、锅、搅拌机完成一道菜。  
- **自动化评估**：通过脚本自动运行生成的代码并检查输出是否符合预期，而不是人工逐个审查，像自动化考试阅卷系统。  
- **多准则指标**：不仅检查功能正确性，还限制使用的 API 或关键字，确保答案在实现方式上符合要求，类似老师不仅看答案对不对，还要看解题思路是否符合教材。  
- **记忆防御（Memorization Defense）**：对原始题目做轻微改动，使模型不能直接靠记忆答案来得分，像把考试题目换成不同的数字让学生必须真正会做。  
- **功能正确性**：运行代码后得到的结果是否通过预设的测试用例，等同于检查程序是否“能跑”。  
- **表层约束**：对代码的结构、使用的函数或关键字设定限制，确保模型遵守特定的实现规范。  

### 核心创新点
1. **真实业务来源 → 从 StackOverflow 抽取千条数据科学问题**  
   过去的基准多是人工编写或教材题，缺乏真实用户需求。作者直接爬取 StackOverflow 上的实际提问，覆盖七大常用库，保证题目自然、实用。结果是评测更能反映模型在真实工作中的表现。

2. **单一评估 → 多准则自动评估体系**  
   传统评测只跑功能测试，容易出现“伪正确”。本文引入了功能正确性 + API 使用约束两层检查，只有同时满足两者的解答才算通过。这样把误判率压到 1.8%，大幅提升评估可靠性。

3. **记忆攻击 → 轻度改写题目防止模型直接复现**  
   为防止模型靠预训练时看到的答案作弊，作者对原始 StackOverflow 内容做了细微改动（如变量名、数据规模），使得直接记忆的解答不再适用。这样模型必须真正理解问题才能得分。

4. **规模与多样性 → 1000 题、七库覆盖**  
   之前的基准往往只有几百题且库单一，难以检验模型的跨库组合能力。DS-1000 的规模和库多样性让评测更具挑战性，也为后续模型提供了更丰富的训练/微调素材。

### 方法详解
整体思路可以拆成三步：**题目构建 → 防记忆改写 → 多准则评估**。

1. **题目构建**  
   - 从 StackOverflow 按标签筛选出涉及 NumPy、Pandas、Matplotlib、SciPy、Scikit‑learn、Seaborn、Statsmodels 的提问。  
   - 每条提问保留原始的自然语言描述、输入数据示例以及最佳答案代码。  
   - 为保证难度均衡，作者手动挑选并分类，使得每个库都有从基础到进阶的题目。

2. **防记忆改写**  
   - 对原始代码进行结构化改动：更换变量名、调换函数参数顺序（只要语义不变）、修改数据规模或随机种子。  
   - 对自然语言描述也做同义替换或重新组织句式，保持信息完整但避免与公开数据集完全相同。  
   - 这种改写足够轻微，不会改变解题思路，却能让模型的直接记忆失效。

3. **多准则自动评估**  
   - **功能正确性**：为每道题目准备一套隐藏的测试用例，运行模型生成的代码并捕获异常。只有输出完全匹配预期才算通过。  
   - **表层约束**：在评估脚本中解析生成代码的抽象语法树（AST），检查是否使用了指定的 API 或是否避免了禁用的关键字。例如，要求使用 `pandas.DataFrame.merge` 而不是手写循环。  
   - 两个条件必须同时满足，才算一次成功。通过统计所有通过的比例得到模型的 **准确率**。  

最巧妙的地方在于 **表层约束**：它把评估从“跑通了没”提升到“用了对的方法”，这在代码生成评测里相当少见，能够过滤掉那些只会“硬拼”答案的模型。

### 实验与效果
- **数据集**：DS-1000 包含 1000 条经改写的真实数据科学任务，覆盖七个常用库。  
- **基线模型**：使用 OpenAI 的 Codex‑002（即 Codex-002）作为公开可得的最强模型。  
- **结果**：Codex‑002 在 DS-1000 上的整体准确率为 **43.3%**，即约四成的题目能够同时满足功能和表层约束。  
- **误判率**：在所有被评估系统接受的解答中，只有 **1.8%** 实际上是错误的，说明多准则评估的可靠性很高。  
- **消融实验**：原文未提供详细的消融结果，但可以推测如果去掉表层约束，误判率会显著上升；若不进行记忆防御，模型的准确率可能会被记忆效应虚高。  
- **局限性**：评测仍然依赖于预设的测试用例，极端情况下模型可能通过巧妙的“作弊”代码（如捕获异常后返回默认值）逃过检测；此外，改写后的题目仍然可能与训练数据有一定重叠，完全消除记忆仍有难度。

### 影响与延伸思考
DS-1000 为代码生成领域提供了首个聚焦数据科学、兼顾真实业务和评估可靠性的基准。自发布后，多个团队把它作为微调数据或评测套件，推动了对库调用推理、跨库组合能力的研究。后续工作可能会在以下方向继续深化：  
- **更细粒度的约束**，比如限定时间复杂度或内存使用；  
- **交互式评估**，让模型在运行时根据错误提示进行调试，模拟真实开发流程；  
- **跨语言扩展**，把基准搬到 R、Julia 等数据科学语言，检验模型的多语言迁移能力。  
如果想进一步了解代码生成评测的趋势，可以关注近期在 *NeurIPS*、*ICLR* 上出现的 “CodeBench” 系列论文，它们在 DS-1000 的基础上加入了更丰富的交互式评测。

### 一句话记住它
DS-1000 用真实的 StackOverflow 题目、轻度改写防记忆、功能+表层双重自动评估，打造了首个可靠衡量数据科学代码生成能力的基准。