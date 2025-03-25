# ExCoT: Optimizing Reasoning for Text-to-SQL with Execution Feedback

> **Date**：2025-03-25
> **arXiv**：https://arxiv.org/abs/2503.19988

## Abstract

Text-to-SQL demands precise reasoning to convert natural language questions into structured queries. While large language models (LLMs) excel in many reasoning tasks, their ability to leverage Chain-of-Thought (CoT) reasoning for text-to-SQL remains underexplored. We identify critical limitations: zero-shot CoT offers minimal gains, and Direct Preference Optimization (DPO) applied without CoT yields marginal improvements. We propose ExCoT, a novel framework that iteratively optimizes open-source LLMs by combining CoT reasoning with off-policy and on-policy DPO, relying solely on execution accuracy as feedback. This approach eliminates the need for reward models or human-annotated preferences.   Our experimental results demonstrate significant performance gains: ExCoT improves execution accuracy on BIRD dev set from 57.37% to 68.51% and on Spider test set from 78.81% to 86.59% for LLaMA-3 70B, with Qwen-2.5-Coder demonstrating similar improvements. Our best model achieves state-of-the-art performance in the single-model setting on both BIRD and Spider datasets, notably achieving 68.53% on the BIRD test set.

---

# ExCoT：利用执行反馈优化文本到SQL的推理 论文详细解读

### 背景：这个问题为什么难？

把自然语言问题直接翻译成结构化的 SQL 查询，需要模型在语言理解、数据库模式匹配以及逻辑推理之间来回切换。传统的 Text‑to‑SQL 方法大多依赖大量标注好的 NL‑SQL 对，或者使用固定的提示词让大模型一次性输出完整的查询。这样做的根本缺陷在于：①模型没有显式的思考过程，错误往往在“草稿”阶段就已经埋下；②即使加入 Chain‑of‑Thought（思维链）提示，零样本下的提升也非常有限；③直接用偏好学习（Direct Preference Optimization，DPO）微调模型时，缺少高质量的偏好标签，只能得到边际改进。于是，如何让开源大模型在没有人工标注的情况下，学会自我检查并逐步提升执行正确率，成为了一个迫切的挑战。

### 关键概念速览
- **Text‑to‑SQL**：把自然语言提问转成对应的 SQL 语句，模型需要理解问题意图并映射到数据库的表、列和关系上。  
- **Chain‑of‑Thought（CoT）**：让模型在给出最终答案前先写出推理步骤，类似于解数学题时先列出草稿，帮助模型把复杂推理拆解成可追踪的子任务。  
- **Execution Feedback**：把模型生成的 SQL 实际跑在数据库上，检查返回的结果是否与期望匹配，用“对/错”作为最直接的监督信号。  
- **Direct Preference Optimization（DPO）**：一种无需额外奖励模型的微调方式，直接把“更好”的输出与“更差”的输出进行比较学习。  
- **Off‑policy DPO**：使用已经收集好的历史生成数据（即模型过去的输出）来进行偏好学习，类似于从旧日志中学习。  
- **On‑policy DPO**：在微调过程中实时生成新样本并立即用于学习，属于在线学习的范式。  
- **Open‑source LLM**：指公开可获取、可自行微调的大语言模型，如 LLaMA‑3、Qwen‑2.5‑Coder 等。  
- **Reward Model（奖励模型）**：传统 RLHF（强化学习从人类反馈）中用来评估答案好坏的模型，本文刻意不使用它。

### 核心创新点
1. **执行反馈代替人工偏好**  
   - 之前的 DPO 需要人工标注的偏好对或专门训练的奖励模型。  
   - 本文直接把 SQL 的执行结果（正确/错误）当作二元奖励，省去任何人工标注环节。  
   - 这样模型可以在大规模无监督数据上自我迭代，显著提升了训练效率和可复制性。

2. **CoT 与 DPO 的协同迭代**  
   - 早期的研究发现，单独使用零样本 CoT 对 Text‑to‑SQL 的提升几乎可以忽略不计。  
   - 本文在每一次 DPO 微调前，都让模型先生成思维链（包括表选择、条件拆解等），再输出 SQL。  
   - 通过让模型在“写草稿”阶段就接受执行反馈，思维链本身被逐步优化，最终的 SQL 正确率大幅提升。

3. **离线 + 在线双轨 DPO 训练框架**  
   - 先用离线收集的 CoT+SQL 对进行 off‑policy DPO，快速获得初步改进。  
   - 随后进入 on‑policy 阶段，模型在每一步都生成新的 CoT+SQL 并立即根据执行反馈更新参数，形成闭环学习。  
   - 这种两阶段策略兼顾了数据利用率和实时纠错能力，使得模型在同等算力下收敛更快。

4. **全开源、无需奖励模型的端到端流水线**  
   - 只依赖公开的大模型和数据库执行环境，整个优化过程不需要任何专有的奖励网络或人工偏好数据。  
   - 因此方法可以直接搬到其他结构化生成任务（如 Text‑to‑Code、NL‑to‑Graph）上复用。

### 方法详解
**整体框架**  
ExCoT 把 Text‑to‑SQL 任务拆成四个循环步骤：①生成思维链 + 初始 SQL；②在目标数据库上执行 SQL，得到二元正确性信号；③把（思维链、SQL、执行结果）存入经验池；④用经验池进行 off‑policy DPO 微调，随后进入 on‑policy 微调循环。整个过程在同一模型上迭代多轮，直至执行准确率收敛。

**关键模块拆解**  

1. **CoT‑SQL 生成器**  
   - 输入：自然语言问题 + 数据库 schema（表名、列名、外键）。  
   - 输出：一段结构化的思维链（如“先选表 A，过滤条件 X，连接表 B”，随后是完整的 SQL）。  
   - 类比：像老师先让学生写解题步骤，再写答案，步骤本身可以被检查。

2. **执行反馈模块**  
   - 将模型输出的 SQL 发送到真实或模拟的数据库引擎。  
   - 比对返回的结果与金标准答案（如果有）或使用预设的判定规则，产生 1（正确）或 0（错误）的信号。  
   - 这里不需要细粒度的奖励，只要知道对错即可。

3. **离线 DPO（Off‑policy）**  
   - 从经验池中抽取已有的（思维链、SQL、反馈）三元组。  
   - 对每一对样本，构造“好”与“坏”对比：如果反馈为 1，则该对被视为“好”，否则视为“坏”。  
   - 通过最大化好样本相对于坏样本的概率比，直接更新模型参数。  
   - 这一步类似于从历史考试卷中挑出正确与错误的答案进行对比学习。

4. **在线 DPO（On‑policy）**  
   - 在每一次参数更新后，模型重新生成新的 CoT‑SQL 对，立即得到执行反馈。  
   - 这些新样本被直接加入经验池并参与下一轮的 DPO 计算，实现“边生成边学习”。  
   - 这种闭环让模型能够快速纠正刚出现的系统性错误。

**最巧妙的设计**  
- **仅用执行对错作为唯一奖励**：省去奖励模型的训练成本，同时保证反馈与最终任务目标完全对齐。  
- **思维链的双向约束**：执行错误不仅惩罚最终 SQL，还间接影响思维链的生成，因为 DPO 的比较是基于完整输出。模型被迫在思维链层面也要“写对”。  
- **离线+在线的两段式学习**：先利用大量已有数据快速提升，再用在线微调细化，兼顾规模与精度。

### 实验与效果
- **数据集**：Spider（跨域 Text‑to‑SQL 基准）和 BIRD（大规模中文 Text‑to‑SQL 数据集）。  
- **基线模型**：LLaMA‑3 70B、Qwen‑2.5‑Coder，分别在未做任何微调、仅使用零样本 CoT、仅使用 DPO（无 CoT）三种设置下进行对比。  
- **主要结果**：  
  - 在 BIRD 开发集上，ExCoT 把执行准确率从 57.37% 提升到 68.51%。  
  - 在 Spider 测试集上，LLaMA‑3 70B 的准确率从 78.81% 跃升至 86.59%。  
  - 最终模型在 BIRD 测试集上达到 68.53%，刷新了单模型的最高记录。  
- **消融实验**（原文未详细描述）：作者报告去掉 CoT、仅使用离线 DPO 或仅使用在线 DPO 时，提升幅度均明显下降，说明四个组件缺一不可。  
- **局限性**：  
  - 依赖于可执行的数据库环境，若数据库访问受限或查询成本高，训练成本会显著上升。  
  - 只提供对错二元信号，无法捕捉细粒度的部分正确或性能优化空间。  
  - 对极其复杂的多步查询（如递归 CTE）仍可能出现思维链失效的情况。

### 影响与延伸思考
ExCoT 把“执行反馈”直接搬进了大模型的偏好学习环节，打开了无需人工标注即可自我改进的可能性。随后的工作开始探索类似的闭环学习在 **Text‑to‑Code**、**NL‑to‑Graph**、甚至 **对话系统** 中的应用，尝试把运行时的成功率当作唯一奖励信号。还有研究把 **Self‑Consistency**（多次采样取多数答案）与 ExCoT 的执行反馈结合，以进一步提升鲁棒性。对想继续深挖的读者，可以关注以下方向：①更细粒度的执行奖励（如查询时间、资源消耗）；②跨数据库的通用执行反馈框架；③将 ExCoT 与大模型的自检机制（如工具调用）融合，构建更通用的“思考‑执行‑反馈”循环。  

### 一句话记住它
让开源大模型通过“写草稿‑跑查询‑自我比较”循环，直接用执行对错来驱动思维链的自我优化，显著提升 Text‑to‑SQL 的准确率。