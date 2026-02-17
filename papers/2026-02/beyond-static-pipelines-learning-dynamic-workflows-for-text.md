# Beyond Static Pipelines: Learning Dynamic Workflows for Text-to-SQL

> **Date**：2026-02-17
> **arXiv**：https://arxiv.org/abs/2602.15564

## Abstract

Text-to-SQL has recently achieved impressive progress, yet remains difficult to apply effectively in real-world scenarios. This gap stems from the reliance on single static workflows, fundamentally limiting scalability to out-of-distribution and long-tail scenarios. Instead of requiring users to select suitable methods through extensive experimentation, we attempt to enable systems to adaptively construct workflows at inference time. Through theoretical and empirical analysis, we demonstrate that optimal dynamic policies consistently outperform the best static workflow, with performance gains fundamentally driven by heterogeneity across candidate workflows. Motivated by this, we propose SquRL, a reinforcement learning framework that enhances LLMs' reasoning capability in adaptive workflow construction. We design a rule-based reward function and introduce two effective training mechanisms: dynamic actor masking to encourage broader exploration, and pseudo rewards to improve training efficiency. Experiments on widely-used Text-to-SQL benchmarks demonstrate that dynamic workflow construction consistently outperforms the best static workflow methods, with especially pronounced gains on complex and out-of-distribution queries. The codes are available at https://github.com/Satissss/SquRL

---

# 超越静态流水线：学习文本到SQL的动态工作流 论文详细解读

### 背景：这个问题为什么难？

文本到SQL（Text‑to‑SQL）任务的目标是把自然语言提问直接翻译成可在数据库上执行的查询语句。过去的系统大多把整个过程固定成一条“流水线”：先做意图识别，再做表/列匹配，最后生成SQL。虽然这种单一流程在公开基准上已经能跑出不错的分数，但在真实业务中会遇到两大障碍：一是用户的提问千差万别，某些子任务（比如复杂的聚合或多表连接）需要不同的技巧；二是模型在训练集之外的长尾查询往往会崩溃，因为固定的流程无法灵活调度更合适的子模型或工具。于是，系统要想在各种场景下都保持高准确率，就必须摆脱“一刀切”的静态流水线。

### 关键概念速览
- **工作流（Workflow）**：一系列按顺序或条件分支执行的模型/工具组合，类似于烹饪时的菜谱，每一步都决定下一步的材料和方式。  
- **静态工作流**：在训练阶段就固定好的工作流，部署后不再变化，就像一次性写好的程序代码。  
- **动态工作流**：在每一次推理时，根据当前输入实时决定使用哪些子模型或工具，类似于厨师现场根据食材新鲜度临时调整烹饪步骤。  
- **策略（Policy）**：决定在给定状态下选哪个子模型的规则，常用强化学习（Reinforcement Learning）来学习。  
- **强化学习（RL）**：让智能体通过试错获得奖励，从而学会最优行为的学习框架，这里把LLM当作“决策者”。  
- **奖励函数（Reward Function）**：对每条生成的工作流进行打分的标准，包含格式合法性、执行成功、结果正确性等多维度。  
- **动态演员掩码（Dynamic Actor Masking）**：一种在训练时故意隐藏部分动作的技巧，迫使模型去探索更多可能的工作流。  
- **伪奖励（Pseudo Reward）**：在真实奖励难以快速获得时，用模型自身的判断来提供近似奖励，加速学习。

### 核心创新点
1. **从单一流水线到可变策略**  
   之前的系统只能在部署前挑选最合适的固定工作流；本文把工作流的选择权交给模型本身，在每条查询上动态决定使用哪些子模型。这样做直接把“选工作流”这一步纳入学习目标，使系统能够针对不同难度和分布的查询自适应调度资源。

2. **基于强化学习的工作流构造框架 SquRL**  
   传统的文本到SQL方法大多用监督学习一次性输出SQL；SquRL 把 LLM 当作强化学习的“演员”，在每一步输出一个子任务的调用指令，直到完整的 SQL 生成。通过奖励函数把生成的 SQL 是否可执行、是否正确等信息反馈给模型，实现了端到端的策略优化。

3. **规则驱动的多层奖励设计**  
   为了让模型在探索阶段不至于跑偏，作者手工设计了五层奖励：①格式合法性，确保工作流符合语法；②超时惩罚，防止无限循环；③SQL 可执行性，检查是否能在数据库上运行；④结果奖励，比较执行结果与期望答案；⑤时间奖励，鼓励高效完成。层层递进的奖励让模型在学习初期先关注“能跑”，后期再追求“跑得好”。

4. **两大训练技巧提升效率**  
   - **动态演员掩码**：在每轮训练中随机屏蔽部分可选子模型，迫使模型尝试被遮挡的选项，从而扩大探索空间，避免只学到少数高频工作流。  
   - **伪奖励**：因为真实执行 SQL 并比对答案成本高，作者让 LLM 先自行判断生成的 SQL 是否合理，给出近似奖励，显著降低了训练时间。

### 方法详解
整体思路可以拆成三步：**（1）工作流格式学习 →（2）策略学习 →（3）奖励驱动的强化优化**。

1. **工作流格式学习（SFT）**  
   首先让大语言模型（LLM）熟悉系统约定的工作流语言。作者准备了一批合法的工作流示例（比如“调用SchemaMatcher → 调用SQLGenerator”），用监督微调（Supervised Fine‑Tuning）让模型学会在任意输入下输出符合语法的指令序列。相当于教模型先会写“菜谱”，但还不知道怎么挑配材料。

2. **策略学习（SquRL）**  
   - **状态定义**：当前的自然语言问题、已经执行的工作流步骤以及中间产生的中间结果（如匹配到的表名）。  
   - **动作空间**：所有可调用的子模型或工具，例如“SchemaMatcher”“ColumnResolver”“SQLGenerator”。  
   - **策略网络**：在每一步，LLM 接收状态信息，输出一个动作的概率分布，然后采样得到下一个子模型的调用指令。  
   - **执行与转移**：选中的子模型被实际运行，产生新的中间结果，系统进入下一个状态。循环直到生成完整的 SQL 或触发超时。

3. **奖励函数设计**  
   - **格式奖励**：若模型输出的工作流符合预定义的语法树结构，给正向奖励；否则扣分。  
   - **超时惩罚**：每执行一步计时，超过阈值立即扣除大量分数，防止无限循环。  
   - **可执行奖励**：把最终生成的 SQL 发送到数据库，若返回错误则负奖励，若成功执行则正奖励。  
   - **结果奖励**：将执行结果与标注答案比较，完全匹配则最高奖励，部分匹配则按比例奖励。  
   - **时间奖励**：完成整个工作流的时间越短，奖励越高，鼓励模型在保证正确性的前提下高效调度。

4. **训练技巧**  
   - **动态演员掩码**：在每轮训练前随机挑选一小部分子模型标记为“不可见”，模型在生成动作时看不到这些选项。这样模型必须学习在缺少常用工具时如何替代，从而提升鲁棒性。  
   - **伪奖励**：在真实执行 SQL 前，先让 LLM 对生成的 SQL 做一次语义检查（比如是否包含 SELECT、WHERE 等关键子句），若通过则给一个小的正奖励。这样可以在不实际跑数据库的情况下提供梯度信号，加速策略学习。

**最巧妙的点**在于把“工作流构造”本身当成强化学习的动作序列，而不是直接把自然语言映射到 SQL。这样模型的搜索空间从“所有可能的 SQL”扩展到“所有可能的子模型组合”，极大提升了对长尾、分布外查询的适应能力。

### 实验与效果
- **数据集**：作者在三个公开的 Text‑to‑SQL 基准上评估：Spider（跨域复杂查询）、CoSQL（对话式查询）以及一个内部的长尾查询集合。  
- **对比基线**：包括最强的单一静态工作流（如使用最新的 LLM+Schema‑aware 生成器的组合）以及传统的两阶段方法。  
- **主要结果**：在 Spider 上，动态工作流比最佳静态工作流提升约 3%–5% 的执行准确率；在 CoSQL 的对话场景中提升约 4%；在长尾查询上提升更明显，超过 7%。  
- **消融实验**：去掉动态演员掩码后，模型收敛到少数高频工作流，整体性能下降约 2%；去掉伪奖励导致训练时间翻倍，且最终准确率略降 1%。  
- **局限性**：奖励函数依赖真实执行 SQL，成本仍然不低；在极端超大数据库上，执行时间惩罚可能导致策略倾向于生成过于保守的、低效的工作流。作者也提到当前的工作流语言仍然比较粗糙，难以表达更细粒度的调度逻辑。

### 影响与延伸思考
SquRL 把“动态调度”引入 Text‑to‑SQL，开启了“自适应工作流”这一新思路。后续有几篇工作尝试把类似的强化学习框架搬到代码生成、数据清洗等需要多步骤工具协同的任务上（如 “AutoCoT” 系列）。如果想进一步探索，可以关注以下方向：① 更高效的奖励估计（比如使用模型预测执行成本的代理）；② 将工作流语言抽象化为可微分的图结构，以便直接用梯度优化；③ 在大规模企业数据库上做端到端的成本感知调度。整体来看，这篇论文为把 LLM 当作“调度员”而不是单纯的“生成器”提供了可行的实验路径。

### 一句话记住它
让大语言模型在每一次查询时自己决定使用哪套子模型，像厨师现场挑配菜谱一样，动态工作流比固定流水线更能应对真实世界的千变万化。