# DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with   Self-Correction

> **Date**：2023-04-21
> **arXiv**：https://arxiv.org/abs/2304.11015

## Abstract

There is currently a significant gap between the performance of fine-tuned models and prompting approaches using Large Language Models (LLMs) on the challenging task of text-to-SQL, as evaluated on datasets such as Spider. To improve the performance of LLMs in the reasoning process, we study how decomposing the task into smaller sub-tasks can be effective. In particular, we show that breaking down the generation problem into sub-problems and feeding the solutions of those sub-problems into LLMs can be an effective approach for significantly improving their performance. Our experiments with three LLMs show that this approach consistently improves their simple few-shot performance by roughly 10%, pushing the accuracy of LLMs towards SOTA or surpassing it. On the holdout test set of Spider, the SOTA, in terms of execution accuracy, was 79.9 and the new SOTA at the time of this writing using our approach is 85.3. Our approach with in-context learning beats many heavily fine-tuned models by at least 5%. Additionally, when evaluated on the BIRD benchmark, our approach achieved an execution accuracy of 55.9%, setting a new SOTA on its holdout test set.

---

# DIN‑SQL：基于分解式上下文学习的文本到SQL生成与自纠正 论文详细解读

### 背景：这个问题为什么难？
把自然语言问题直接翻译成SQL查询是一项典型的跨模态推理任务。传统做法要么靠大规模标注数据微调模型，要么直接用Few‑Shot Prompt让大语言模型（LLM）一次性生成完整SQL。前者成本高，后者在复杂的多表联结、嵌套子查询等场景经常出现语法错误或逻辑漏洞。尤其在Spider这类需要跨表推理的基准上，纯提示的LLM与经过精细调参的专用模型之间仍有10%以上的性能差距，这说明单轮生成的推理深度和可靠性仍不足。

### 关键概念速览
**文本到SQL（Text‑to‑SQL）**：把用户的自然语言问句转成对应的结构化SQL语句，类似把口头指令翻译成数据库指令。  
**大语言模型（LLM）**：参数量在数十亿以上、通过海量文本预训练得到的通用生成模型，如GPT‑3、Claude等。  
**上下文学习（In‑Context Learning）**：模型在没有梯度更新的情况下，仅靠提示（prompt）中提供的示例来学习任务模式。相当于给模型“现场演示”。  
**Few‑Shot Prompt**：在提示里放入少量（通常1‑5个）输入‑输出示例，让模型模仿这些例子完成新任务。  
**任务分解（Decomposition）**：把一个大问题拆成若干小子问题，分别求解后再组合，类似把一道大题拆成若干小题逐个解答。  
**自纠正（Self‑Correction）**：模型在生成答案后自行检查并修正错误，像学生先写答案再自己批改。  
**执行准确率（Execution Accuracy）**：把生成的SQL在真实数据库上执行，若返回的结果与金标准一致则计为正确。  
**Spider 数据集**：业界最具挑战性的 Text‑to‑SQL 基准，包含上百个跨表查询，常用于评估模型的推理能力。  

### 核心创新点
1. **任务分解 → 子问题驱动的提示**  
   以前的Few‑Shot Prompt直接让LLM一次性输出完整SQL。DIN‑SQL先把自然语言问句拆成「模式识别」「列映射」「子查询结构」等子任务，每个子任务都有专属的示例提示。这样模型只需要在每一步解决一个局部问题，显著降低一次性推理的负担。实验表明，这种分解让同一模型的执行准确率提升约10%。  

2. **子任务答案回填 → 连续上下文**  
   在生成子任务答案后，DIN‑SQL把这些答案拼回提示中，作为后续子任务的额外上下文。相当于让模型“记住”前一步的决定，避免重复推理或信息丢失。与仅使用原始自然语言提示的基线相比，这一步进一步提升了SQL的结构完整性。  

3. **自纠正回路 → 结果检查与再生成**  
   完成全部子任务并拼装出完整SQL后，系统让LLM自行执行一次“错误检查”。模型会对SQL的语法、列名匹配、执行结果进行评估，若发现不一致则在同一提示框架下重新生成有问题的子句。这个闭环让模型在不依赖外部校验器的情况下自行纠错，提升了对细节错误的容忍度。  

4. **纯提示即达 SOTA**  
   通过上述三点组合，DIN‑SQL在Spider的holdout测试集上实现了85.3%的执行准确率，超过当时最佳的微调模型（79.9%）。同样的策略在BIRD基准上也刷新了记录，说明分解+自纠正的思路在不同数据库场景下具有普适性。

### 方法详解
**整体框架**  
DIN‑SQL的工作流程可以概括为四步：① 任务分解、② 子任务逐一提示生成、③ 子任务答案回填形成完整SQL、④ 自纠正检查并可能迭代。整个过程全程在LLM的上下文窗口内完成，无需梯度更新或外部模型。

**1. 任务分解**  
系统首先使用一个轻量的规则或小模型把用户问句拆成若干结构化子任务。常见的子任务包括：  
- **意图识别**：判断是查询、计数还是排序等。  
- **表/列映射**：把自然语言中的实体对应到数据库的表名或列名。  
- **子查询结构**：决定是否需要嵌套、JOIN 的方式以及过滤条件。  

**2. 子任务提示生成**  
每种子任务都有专门的Few‑Shot示例。例如，列映射任务的提示会列出几条自然语言‑列名对，让模型学习“‘订单日期’对应 column `order_date`”。这些示例放在同一提示里，后面紧跟当前子任务的输入，模型只需输出对应的答案。

**3. 回填与拼装**  
当模型输出子任务答案后，系统把答案直接写回提示的后部，形成“已知信息 + 待解子任务”的新上下文。比如在生成完列映射后，后续的子查询结构提示会看到已经确定好的列名，从而可以直接写出 `SELECT order_date FROM orders WHERE ...` 的雏形。所有子任务完成后，系统把拼装好的SQL交给模型进行一次整体校验。

**4. 自纠正回路**  
自纠正分两层：  
- **语法层**：模型检查SQL是否符合基本语法（如缺少 FROM、括号不匹配），若发现错误，提示模型重新生成对应子句。  
- **执行层**：模型在模拟的数据库上执行SQL（或使用一个轻量的执行器），比较返回结果与金标准的结构（如列数、数据类型）是否匹配。若不匹配，模型会被要求定位错误来源（列名、过滤条件等），并在相应子任务上重新生成。  

**巧妙之处**  
- **一次性上下文循环**：所有子任务和自纠正都在同一个提示窗口内完成，避免了跨模型的接口开销。  
- **无需外部校验器**：自纠正完全依赖LLM的内部推理能力，省去额外的规则引擎或专用评估器。  
- **模块化可扩展**：如果遇到更复杂的SQL特性（如窗口函数），只需新增对应的子任务模板即可，无需重新训练模型。

### 实验与效果
- **数据集**：主要在Spider的公开holdout测试集上评估，同时在BIRD基准的holdout上做二次验证。  
- **对比基线**：包括强微调模型（如 PICARD、SmBoP、T5‑3B）以及纯提示的GPT‑3.5、Claude 等。  
- **核心结果**：在Spider上，DIN‑SQL的执行准确率达到 **85.3%**，领先当时最佳微调模型的 **79.9%**，并且比多数微调模型至少高出 **5%**。在BIRD上取得 **55.9%** 的执行准确率，同样刷新记录。  
- **消融实验**：论文报告了去掉任务分解或自纠正任意一环后，性能下降约 **8‑10%**，说明两者对提升效果同等重要。  
- **局限性**：作者指出方法对LLM的上下文长度有依赖，极其复杂的查询可能导致提示超出窗口；此外，自纠正仍然受限于模型自身的“自我评估”能力，极端错误仍可能未被捕获。  

### 影响与延伸思考
DIN‑SQL的成功让业界重新审视“提示即模型”这一思路，尤其是**任务分解+自纠正**的组合被多篇后续工作引用，出现了如 “CoT‑SQL”、 “Self‑Refine‑SQL”等变体。它也推动了在其他结构化生成任务（如代码生成、表格填充）中尝试类似的分步提示策略。未来的研究方向可能包括：  
- 将分解过程交给专门的“规划模型”，进一步降低人工规则依赖。  
- 融合外部执行器的硬性校验，以提升自纠正的可靠性。  
- 探索在更大上下文窗口（如 GPT‑4‑32K）下的端到端分解深度。  

### 一句话记住它
**把一次性生成的难题拆成小步骤，让大模型在每一步自我检查，Prompt 也能跑到 SOTA。**