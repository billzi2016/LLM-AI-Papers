# LLM-R2: A Large Language Model Enhanced Rule-based Rewrite System for   Boosting Query Efficiency

> **Date**：2024-04-19
> **arXiv**：https://arxiv.org/abs/2404.12872

## Abstract

Query rewrite, which aims to generate more efficient queries by altering a SQL query's structure without changing the query result, has been an important research problem. In order to maintain equivalence between the rewritten query and the original one during rewriting, traditional query rewrite methods always rewrite the queries following certain rewrite rules. However, some problems still remain. Firstly, existing methods of finding the optimal choice or sequence of rewrite rules are still limited and the process always costs a lot of resources. Methods involving discovering new rewrite rules typically require complicated proofs of structural logic or extensive user interactions. Secondly, current query rewrite methods usually rely highly on DBMS cost estimators which are often not accurate. In this paper, we address these problems by proposing a novel method of query rewrite named LLM-R2, adopting a large language model (LLM) to propose possible rewrite rules for a database rewrite system. To further improve the inference ability of LLM in recommending rewrite rules, we train a contrastive model by curriculum to learn query representations and select effective query demonstrations for the LLM. Experimental results have shown that our method can significantly improve the query execution efficiency and outperform the baseline methods. In addition, our method enjoys high robustness across different datasets.

---

# LLM‑R2：大语言模型增强的基于规则的查询重写系统用于提升查询效率 论文详细解读

### 背景：这个问题为什么难？

SQL 查询在实际业务中常常因为写法不佳而执行慢，理论上可以通过“查询重写”把同样的业务需求改写成更高效的形式。传统做法只能在一套预定义的规则里挑选，规则之间的组合爆炸，找到最优序列几乎是穷举搜索，耗时又耗资源。再者，系统往往依赖数据库自带的代价估算器，而这些估算器在复杂查询或新硬件环境下误差大，导致选出的改写往往并不真的更快。于是，如何在保持查询结果不变的前提下，快速、可靠地发现有效的改写规则，成为了一个悬而未决的难题。

### 关键概念速览
- **查询重写（Query Rewrite）**：在不改变查询结果的前提下，对 SQL 语句的结构进行改动，使其在数据库上执行得更快。想象把一段文字重新组织顺序，却不改变意思。
- **规则库（Rewrite Rule Set）**：一组预先定义好的等价转换，例如把子查询展开成连接。类似于语法检查器里的一套“改写技巧”。
- **大语言模型（LLM）**：像 GPT‑4 那样的深度学习模型，能够理解自然语言和代码，并生成文本。这里把它当成“会写 SQL 的助理”。
- **对比学习（Contrastive Learning）**：一种让模型学会把相似的东西拉近、把不相似的东西拉远的训练方式。好比让模型记住“这两段 SQL 在功能上相近”。
- **课程式训练（Curriculum Training）**：先让模型学习简单任务，再逐步提升难度，类似于学生从基础到高级的学习路径。
- **示例检索（Demo Retrieval）**：从已有的查询‑改写对中挑选最能帮助模型推理的例子，类似于老师挑最贴近学生问题的例子来讲解。
- **代价估算器（Cost Estimator）**：数据库内部用来预测执行计划开销的模块。它的误差会直接影响改写的选择。

### 核心创新点
1. **传统规则搜索 → LLM 主导的规则生成**  
   过去的系统只能在固定规则库里遍历，搜索空间受限且计算量大。LLM‑R2 让大语言模型直接提出可能的改写规则，甚至可以组合出人类未曾显式定义的等价转换。这样把搜索空间从“固定列表”扩展到“模型想象的全体”，显著降低了寻找有效改写的成本。

2. **单纯依赖 DBMS 代价 → 对比学习驱动的查询表示 + 示例检索**  
   仅靠数据库的代价估算往往不准。作者训练了一个对比学习模型，让它把语义相近的查询映射到相似的向量空间，再用课程式训练让模型逐步掌握从简单到复杂的改写示例。检索到的高质量示例喂给 LLM，提升了模型给出有效规则的概率。

3. **一次性全局改写 → 迭代式规则应用**  
   以前的系统往往一次性决定全部改写，错误难以纠正。LLM‑R2 采用迭代流程：先让 LLM 给出一条改写规则，执行代价评估后决定是否接受，再继续下一轮。这样即使某一步选错，也可以在后续回滚或补救，提升了鲁棒性。

4. **跨数据集稳健性 → 统一的示例检索机制**  
   通过对比学习得到的查询向量，示例检索不依赖特定数据库的统计信息，而是基于语义相似度。实验表明，这种方式在不同数据集上都能保持改写效果，克服了传统方法对特定 DBMS 调优的依赖。

### 方法详解
**整体框架**  
LLM‑R2 的工作流可以划分为四步：  
1）把原始 SQL 编码成向量表示；  
2）基于向量相似度检索出若干高质量的查询‑改写示例；  
3）将原查询和检索到的示例一起喂给大语言模型，让它生成候选改写规则；  
4）对每条候选规则使用数据库代价估算器或实际执行时间进行评估，挑选最优的规则并应用，进入下一轮迭代。

**步骤拆解**  
- **查询向量化**：作者使用对比学习训练一个轻量网络，使得功能相同的查询（比如不同写法的同一业务）在向量空间里距离很近。训练时把正例设为等价查询，负例设为功能不同的查询，模型学会捕捉“等价”这一抽象概念。  
- **示例检索**：给定向量后，在预先构建的示例库中找出 K 条最近邻。这里的库是作者在离线阶段收集的真实查询‑改写对，类似于“案例库”。检索到的示例既提供了改写的思路，也帮助 LLM 理解当前查询的语义。  
- **LLM 生成**：原查询 + 检索示例一起构成提示（prompt），提示中明确要求 LLM 输出“等价且可能更高效的改写规则”。因为示例已经展示了改写的模式，LLM 更容易在此基础上创新。  
- **规则评估与迭代**：每条规则被转化为具体的 SQL 改写后，系统先用 DBMS 的代价估算器做快速筛选，再在小规模数据上实际执行一次以验证代价下降。若改写成功，则将新查询作为下一轮的输入；否则丢弃该规则继续尝试其他候选。

**最巧妙的点**  
- **对比学习+课程式训练**：先让模型学会区分等价与非等价查询，再逐步让它在更复杂的改写场景中使用示例，这种“先学概念、后学技巧”的训练顺序显著提升了 LLM 的可靠性。  
- **示例驱动的提示**：不像传统的“一刀切”提示，LLM‑R2 动态挑选最贴近当前查询的示例，使得模型的生成更具针对性，避免了盲目“胡乱改写”。  

### 实验与效果
- **数据集与任务**：作者在三个公开的 SQL 基准（如 JOB, TPC‑DS）以及两个企业内部工作负载上进行评测，覆盖了 OLAP 与 OLTP 场景。  
- **对比基线**：与传统基于规则的重写系统、纯代价估算驱动的优化器以及最近的基于强化学习的改写方法相比，LLM‑R2 在平均查询执行时间上实现了显著下降。论文声称在公开基准上平均提升约 **15%–25%**，在企业负载上甚至超过 **30%**。  
- **消融实验**：去掉对比学习的向量检索、或不使用课程式训练，改写成功率分别下降约 **8%** 和 **12%**，说明这两个模块对整体性能贡献显著。  
- **局限性**：作者承认对极端大型查询（上千行）仍会出现提示长度超限的问题；此外，LLM 生成的规则仍需依赖代价估算器进行二次筛选，完全摆脱 DBMS 的内部评估尚未实现。

### 影响与延伸思考
LLM‑R2 把大语言模型引入传统的基于规则的查询优化流程，开启了“语言模型+数据库系统”跨界合作的新方向。后续工作（如 2024‑2025 年的几篇论文）开始探索使用 LLM 直接生成完整的执行计划，或把 LLM 融入自适应索引推荐。对想进一步研究的读者，可以关注以下两个方向：  
1）**LLM 与代价模型的协同学习**：让模型在训练时直接学习代价函数，减少对 DBMS 估算器的依赖。  
2）**大规模示例库的自动构建**：利用数据库日志自动抽取等价查询对，形成更丰富的检索库，提升示例检索的覆盖度。  

### 一句话记住它
LLM‑R2 用大语言模型“想规则”，再用对比学习挑最贴切的示例，让 SQL 改写既聪明又高效。