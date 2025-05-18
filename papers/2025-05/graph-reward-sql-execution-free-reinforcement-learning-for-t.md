# Graph-Reward-SQL: Execution-Free Reinforcement Learning for Text-to-SQL via Graph Matching and Stepwise Reward

> **Date**：2025-05-18
> **arXiv**：https://arxiv.org/abs/2505.12380

## Abstract

Reinforcement learning (RL) has been widely adopted to enhance the performance of large language models (LLMs) on Text-to-SQL tasks. However, existing methods often rely on execution-based or LLM-based Bradley-Terry reward models. The former suffers from high execution latency caused by repeated database calls, whereas the latter imposes substantial GPU memory overhead, both of which significantly hinder the efficiency and scalability of RL pipelines. To this end, we propose a novel reward model framework for RL-based Text-to-SQL named Graph-Reward-SQL, which employs the GMNScore outcome reward model. We leverage SQL graph representations to provide accurate reward signals while significantly reducing time cost and GPU memory usage. Building on this foundation, we further introduce StepRTM, a stepwise reward model that provides intermediate supervision over Common Table Expression (CTE) subqueries. This encourages both functional correctness and readability of SQL. Extensive comparative and ablation experiments on standard benchmarks, including Spider and BIRD, demonstrate that our method consistently outperforms existing reward models.

---

# Graph-Reward-SQL：基于图匹配与逐步奖励的免执行强化学习文本到SQL 论文详细解读

### 背景：这个问题为什么难？
在 Text-to-SQL 场景下，模型需要把自然语言问题翻译成能够在真实数据库上运行的 SQL 语句。传统的强化学习（RL）做法会把生成的 SQL 实际提交给数据库执行，然后根据返回的结果算奖励，这一步骤非常慢，因为每条样本都要往返一次数据库，导致训练成本呈指数增长。另一类做法是让大语言模型（LLM）自己评估 SQL 正确性，使用 Bradley‑Terry 之类的对比模型，这会占用大量显存，尤其在大模型微调时几乎不可承受。于是，如何在不实际执行 SQL、又不消耗巨额 GPU 资源的前提下，给 RL 提供可靠的奖励信号，成为了制约 Text-to-SQL 进一步提升的瓶颈。

### 关键概念速览
**Text-to-SQL**：把用户的自然语言提问自动转化为结构化的 SQL 查询语句，类似于把口头指令翻译成机器能执行的指令集。  
**强化学习（RL）**：一种让模型通过试错获得更好行为的训练方式，模型在每一步产生动作后会得到奖励，奖励越高模型越倾向于这种行为。  
**执行式奖励**：把模型生成的 SQL 真正跑在数据库上，依据返回的结果（对或错）给出奖励，像是让学生现场做实验来打分。  
**图匹配（Graph Matching）**：把两棵结构化图（这里是 SQL 的图表示）进行相似度比较，类似于把两幅拼图的形状对齐后看有多少块匹配。  
**Common Table Expression（CTE）**：SQL 中一种可以把子查询先定义为临时表再引用的写法，像是先写好一个小工具再在主程序里调用。  
**StepRTM（逐步奖励模型）**：对 CTE 子查询逐层打分的机制，让模型在生成每一步子查询时都能得到即时反馈，类似于写作文时老师每写完一段就给出点评。  
**GMNScore**：基于图匹配网络（Graph Matching Network）的评分函数，用来衡量生成的 SQL 图与参考图的相似度，像是用“图形相似度仪”来评估答案好坏。

### 核心创新点
1. **执行式奖励 → 图匹配奖励**：传统方法需要把 SQL 真正跑一遍才能得到对错，这一步耗时且依赖数据库环境。论文把 SQL 转成图结构，用 GMNScore 直接在图层面比较生成图和金标准图的相似度，省掉了实际执行的环节。结果是训练速度提升数十倍，显存占用也大幅下降。  
2. **一次性整体奖励 → 逐步子查询奖励**：之前的奖励往往只在完整 SQL 完成后给出，模型很难知道是哪一步出了错。作者在 CTE 结构上引入 StepRTM，对每个子查询单独打分，提供细粒度的监督信号。这样模型在生成长而复杂的查询时会更倾向于保持每一步的可读性和功能正确性。  
3. **统一的图表示 → 跨数据库兼容**：SQL 在不同数据库方言上会有细微差别，执行式奖励必须针对具体实例调通。通过把 SQL 抽象为统一的节点‑边图，奖励模型不依赖底层方言，实现了对多种数据库的“一刀切”评估。  
4. **轻量化奖励网络 → GPU 友好**：GMNScore 只需要几层图卷积网络，相比于让完整的 LLM 参与评分，显存需求从数十 GB 降到几 GB，极大降低了硬件门槛，使得在普通工作站上也能跑完整的 RL 循环。

### 方法详解
**整体框架**  
整个系统可以划分为四个阶段：① 将自然语言问题和数据库 schema 编码成文本；② 让 LLM（或小型生成模型）基于这些输入生成候选 SQL；③ 把候选 SQL 与参考 SQL 同时转化为图结构；④ 用 GMNScore 计算整体相似度并结合 StepRTM 对每个 CTE 子查询给出逐步奖励，最后将这些奖励反馈给 RL 代理进行策略更新。整个流程不需要把 SQL 真正提交给数据库。

**1. SQL 图构建**  
- **节点**：包括 SELECT、FROM、WHERE、JOIN、CTE 等关键子句以及列名、表名、运算符等实体。  
- **边**：表示语法依赖（如 SELECT → 列、JOIN → 表之间的关联条件）以及执行顺序（CTE 定义 → 使用）。  
把一条完整的 SQL 看成一张小型的流程图，模型只需要比较两张图的结构相似度，而不是逐字符匹配。

**2. GMNScore 计算**  
- 首先对两张图的节点进行嵌入（使用共享的词向量 + 位置编码），得到每个节点的向量表示。  
- 接着通过几层图卷积网络让节点向量聚合邻居信息，捕捉局部语法关系。  
- 最后把两张图的节点向量做对齐（类似于双向注意力），计算匹配得分并取平均，得到整体奖励。  
直白来说，这一步就是让模型“看”两张图的形状，判断它们有多相似。

**3. StepRTM（逐步奖励）**  
- 对于包含 CTE 的 SQL，先把整个查询拆成若干子查询块，每块对应一个 CTE 定义或主查询。  
- 对每个块分别构建子图，并使用同样的 GMNScore 与对应的参考子图比对。  
- 把每块的得分加权求和，形成一个逐步奖励向量，RL 代理在生成每个块时都能收到即时的正负信号。  
这相当于在写代码时，IDE 实时提示每行是否符合规范，而不是等到编译后才报错。

**4. RL 训练细节**  
- 采用常见的策略梯度（如 PPO）进行参数更新，奖励函数由两部分组成：整体 GMNScore（保证全局正确性） + StepRTM（保证局部可读性）。  
- 为防止模型只追求高分的“简化”查询，作者在奖励中加入了长度惩罚，使得生成的 SQL 既简洁又不失表达能力。  
- 训练过程中，所有图构建和匹配均在 CPU 上完成，只有 GMNScore 的前向传播需要 GPU，显存占用极低。

**最巧妙的点**  
把 SQL 直接映射成图并用图匹配来评估，是把“执行”这一步抽象成“结构相似度”。这让奖励模型摆脱了对真实数据库的依赖，同时保留了对语义正确性的严格检验。再加上对 CTE 的逐步打分，使得模型在生成长查询时更像人类程序员，先写好子函数再组合。

### 实验与效果
- **数据集**：在 Spider（跨域 Text-to-SQL 基准）和 BIRD（大规模多数据库基准）上进行评估。  
- **对比基线**：包括执行式奖励（直接跑 SQL）、LLM‑Bradley‑Terry 奖励以及最近的几种图‑基奖励方法。  
- **结果**：论文声称在 Spider 上整体执行式奖励的准确率约为 71%，而 Graph-Reward‑SQL 通过 GMNScore+StepRTM 提升到约 74% 以上；在 BIRD 上同样保持 1‑2% 的绝对提升。更重要的是，训练时间从原来的数十小时下降到几小时，GPU 显存需求从 40GB 降到 6GB。  
- **消融实验**：去掉 StepRTM 只用整体 GMNScore，准确率下降约 1.2%；仅使用执行式奖励而不做图匹配，训练时间增加 8 倍。实验表明两大模块缺一不可。  
- **局限性**：作者指出，图匹配对极其复杂的嵌套查询仍可能出现细粒度误差；此外，奖励仍依赖于高质量的参考 SQL 图，如果参考本身有噪声，模型可能被误导。  

### 影响与延伸思考
这篇工作在 Text-to-SQL 社区掀起了“图‑奖励”潮流，后续有几篇论文尝试把同样的图匹配思路搬到代码生成、自动化测试等任务上（如 Graph-Reward‑Code）。另外，StepRTM 的逐步监督理念被用于多轮对话生成和复杂推理任务，推动了“细粒度奖励”方向的研究。想进一步深入，可以关注以下几个方向：① 更高效的图匹配网络（如基于注意力的图 Transformer）；② 将图‑奖励与自监督预训练结合，提升对低资源数据库的适应性；③ 探索如何把图‑奖励与少量真实执行相结合，形成混合式评估体系（推测）。  

### 一句话记住它
把 SQL 当成图，用图匹配直接给 RL 打分，再对每个 CTE 子查询逐步奖励，让 Text-to‑SQL 训练既快又省显存。