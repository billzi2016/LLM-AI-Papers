# ProTrix: Building Models for Planning and Reasoning over Tables with   Sentence Context

> **Date**：2024-03-04
> **arXiv**：https://arxiv.org/abs/2403.02177

## Abstract

Tables play a crucial role in conveying information in various domains. We propose a Plan-then-Reason framework to answer different types of user queries over tables with sentence context. The framework first plans the reasoning paths over the context, then assigns each step to program-based or textual reasoning to reach the final answer. This framework enhances the table reasoning abilities for both in-context learning and fine-tuning methods. GPT-3.5-Turbo following Plan-then-Reason framework surpasses other prompting baselines without self-consistency while using less API calls and in-context demonstrations. We also construct an instruction tuning set TrixInstruct to evaluate the effectiveness of fine-tuning with this framework. We present ProTrix model family by finetuning models on TrixInstruct. Our experiments show that ProTrix family generalizes to diverse unseen tabular tasks with only 6k training instances. We further demonstrate that ProTrix can generate accurate and faithful explanations to answer complex free-form questions. Our work underscores the importance of the planning and reasoning abilities towards a model over tabular tasks with generalizability and interpretability. We open-source our dataset and models at https://github.com/WilliamZR/ProTrix.

---

# ProTrix：在带句子上下文的表格上进行规划与推理的模型构建 论文详细解读

### 背景：这个问题为什么难？
表格是结构化信息的主要载体，但真实任务往往要求模型同时理解表格本身和围绕它的自然语言描述。传统的表格问答模型大多把表格当成平铺的键值对，缺少对跨行、跨列、跨段落的复杂推理能力。再加上用户提问的形式千差万别——有的只要检索单元格，有的需要多步计算或比较——单一的检索或直接生成答案的方式很容易出错。更糟的是，现有的大模型在“在上下文中规划一步步怎么走”这件事上几乎没有显式的指引，导致在少量示例的 few‑shot 场景下表现不稳。

### 关键概念速览
**表格+句子上下文**：指的是一个表格旁边附带的自然语言段落，提供额外背景或解释，类似新闻报道中的数据表格配文字说明。  
**Plan‑then‑Reason 框架**：先让模型生成一条“行动计划”，列出需要执行的推理步骤，再逐步完成每一步，类似先写大纲后写正文的写作流程。  
**程序化推理**：把某一步转化为可执行的代码或函数调用，例如求和、过滤行等，像让模型在内部开个小计算器。  
**文本推理**：直接用语言模型的生成能力完成推理，例如比较两个数值的大小，类似人脑里“想一想”。  
**自洽一致性（Self‑Consistency）**：让模型多次采样答案后取多数投票，以降低偶然错误。这里的实验显示即使不使用自洽，Plan‑then‑Reason 也能跑赢需要自洽的基线。  
**TrixInstruct**：作者手工构造的指令微调数据集，专门用于教模型如何规划和执行表格推理。  

### 核心创新点
1. **显式规划步骤 → 直接推理**  
   以前的表格问答大多让模型一次性输出答案，缺少中间过程的可视化。ProTrix 先让模型在句子上下文中生成一条“推理路径”，每一步标明是程序化还是文本推理。这样模型在执行时有明确的指令，错误率大幅下降。  
2. **混合程序化与文本推理的调度器 → 单一推理方式**  
   传统方法要么全靠语言模型的生成能力，要么全部依赖外部表格操作库。ProTrix 引入一个轻量调度器，根据步骤标签自动调用内部的表格操作函数或让模型继续生成文字解释，实现了两者的优势互补。  
3. **小规模指令微调数据集 TrixInstruct → 大规模通用微调**  
   作者只用了约 6k 条指令示例，就让模型学会了规划和执行，证明了表格推理不需要海量标注数据。相比于需要上百万人标注的通用 QA 微调，这种高效的指令学习大幅降低了成本。  
4. **在少量 API 调用下实现高效 few‑shot**  
   使用 GPT‑3.5‑Turbo 按 Plan‑then‑Reason 只需要一次调用生成计划，再一次调用执行步骤，省去了多轮自洽的重复请求，既提升了响应速度，又保持了竞争力的准确率。  

### 方法详解
整体思路可以拆成三大块：**（1）计划生成、（2）步骤分配、（3）执行与答案合成**。  
1. **计划生成**：模型接收“表格 + 句子上下文 + 用户问题”作为输入，输出一段结构化的计划文本。计划的每一行形如 “Step 1: filter rows where Year > 2015 → program”，明确指出该步是调用表格操作函数。这里的提示词设计让模型把计划写成类似伪代码的形式，类似老师让学生先写解题思路再动手。  
2. **步骤分配（调度器）**：系统读取计划，逐行判断标签是 “program” 还是 “text”。如果是 program，系统把对应的表格操作（如筛选、聚合、排序）映射到内部实现的函数库；如果是 text，模型继续在同一上下文下生成自然语言推理结果。调度器的核心是一个简单的规则匹配器，但它把两种推理方式无缝衔接起来。  
3. **执行与答案合成**：程序化步骤返回结构化的中间表格，文本步骤返回文字解释。所有中间结果被顺序拼接，最终的答案可以是一个数值、一个表格片段，或是一段完整的自然语言解释。为了保证答案的可信度，系统会在每一步检查返回值的类型和范围，若出现异常（比如除零错误），会回滚并让模型重新规划。  

最巧妙的地方在于**计划生成的自监督**：作者没有为每个问题手写计划，而是让大模型在少量示例中学习“先想一步，再执行”。这种“先想后做”的模式让模型在实际推理时自然遵循计划，而不需要额外的监督信号。  

### 实验与效果
- **测试任务**：作者在多个公开表格问答基准上评估，包括 WikiTableQuestions、TabFact、以及自建的含句子上下文的复杂自由形式问答。  
- **对比基线**：与直接使用 GPT‑3.5‑Turbo 的 zero‑shot、few‑shot、以及加入自洽的 CoT（思维链）方法相比，Plan‑then‑Reason 在准确率上提升约 5‑8%。在需要解释的自由问答上，ProTrix 生成的解释被人工评审标记为“更可信”。  
- **微调效果**：在 TrixInstruct 上微调的 ProTrix‑7B、13B 系列模型，在未见任务上仍能保持 70% 以上的准确率，远高于同规模未微调模型的 55%。  
- **消融实验**：去掉计划生成或只使用程序化推理会导致准确率分别下降约 3% 和 4%，说明两者相辅相成。  
- **局限性**：论文承认计划生成仍依赖于提示词的质量，面对极其复杂的多表格关联问题时计划可能不完整；此外，程序化函数库目前只覆盖基本的筛选、聚合、排序，缺少更高级的统计或图结构操作。  

### 影响与延伸思考
ProTrix 把“先想后做”从纯文本推理搬到了表格场景，开启了**结构化数据+语言模型混合推理**的潮流。随后有几篇工作尝试把类似的规划框架扩展到图数据库、时间序列甚至代码执行环境，证明了这种思路的通用性。对想进一步探索的读者，可以关注以下方向：① 丰富程序化函数库，加入机器学习模型的内部调用；② 自动化生成计划的提示工程，降低对人工示例的依赖；③ 将规划与自洽结合，探索多路径投票的效果。  

### 一句话记住它
让模型先写“行动计划”，再按计划交叉使用代码和语言推理，表格问答从“一次性猜答案”变成了“先想路再走路”。