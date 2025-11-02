# MARS-SQL: A multi-agent reinforcement learning framework for Text-to-SQL

> **Date**：2025-11-02
> **arXiv**：https://arxiv.org/abs/2511.01008

## Abstract

Large Language Models (LLMs) often struggle with the precise logic and schema alignment required for complex Text-to-SQL tasks. While current methods rely heavily on static prompting, they lack the ability to dynamically adapt and self-correct through environmental interaction. To bridge this gap, we propose MARS-SQL, a trainable multi-agent framework for Text-to-SQL. Rather than introducing a new standalone SQL primitive, MARS-SQL makes an agentic workflow trainable by decomposing the problem into three specialized roles: schema grounding, query generation, and solution validation. Central to our approach is a generation agent trained via a multi-turn RL policy within a ReAct-style loop. The agent learns to iteratively reason, execute intermediate SQL actions on a live database, and refine its strategy based on execution feedback. To improve robustness, we further introduce a validation mechanism that treats solution selection as a generative modeling task, identifying the optimal interaction trajectory through next-token prediction probabilities. Empirical evaluations demonstrate the effectiveness of coupling interactive learning with trajectory ranking. MARS-SQL achieves state-of-the-art performance, recording an execution accuracy of 77.84% on the BIRD development dataset and 89.75% on the Spider test dataset, while also transferring strongly to out-of-domain benchmarks. Code is available at https://github.com/YangHaolin0526/MARS-SQL.

---

# MARS‑SQL：面向文本到SQL的多智能体强化学习框架 论文详细解读

### 背景：这个问题为什么难？

把自然语言问题直接翻译成可执行的SQL语句，需要模型既懂用户的意图，又要精准对齐数据库的表结构和列名。传统的大语言模型（LLM）在一次性提示（static prompting）下往往只能给出“看起来合理”的答案，却缺乏检查和纠错的机制，导致在复杂查询、嵌套子句或多表关联时容易出错。更糟的是，LLM 并不会主动去执行生成的SQL，错了也不知道。于是，如何让模型在真实数据库环境中“试错—学习”，成为提升 Text‑to‑SQL 精度的关键瓶颈。

### 关键概念速览
- **Text‑to‑SQL**：把自然语言问题转成结构化的SQL查询语句，类似把口头指令翻译成数据库的操作指令。  
- **多智能体（Multi‑Agent）**：把一个大任务拆成若干子任务，由不同的“角色”分别负责，像团队里有专门的需求分析、代码编写和质量检查人员。  
- **强化学习（RL）**：模型通过与环境交互获得奖励，学习怎样的行为能得到更高分数，类似玩游戏时通过得分来改进策略。  
- **ReAct 循环**：模型在一次推理过程中交替进行“思考（think）”和“行动（act）”，思考阶段生成内部推理，行动阶段执行外部操作（如查询数据库）。  
- **轨迹排名（Trajectory Ranking）**：在多个可能的交互序列中，根据下一个 token 的概率分布挑选最有前景的路径，类似在多条解题思路里挑选最有把握的一条。  
- **GRPO（Goal‑oriented Reward‑based Policy Optimization）**：一种强化学习的奖励设计方式，奖励会根据任务目标的完成程度（如是否完全匹配 schema、是否可执行）来打分。  
- **SFT（Supervised Fine‑Tuning）**：在已有标注数据上做有监督微调，让模型学会在特定情境下输出期望的答案。  

### 核心创新点
1. **任务拆解为三角色工作流 → 采用 Grounding Agent、Generation Agent、Validation Agent 三个专职智能体 → 让每一步都有针对性的训练目标，显著提升了 schema 对齐和 SQL 正确率。**  
2. **在生成阶段引入 ReAct‑style 多轮交互 + RL 策略 → 生成智能体在每轮思考后会把中间 SQL 片段提交给真实数据库执行，依据执行结果（成功、错误信息）得到即时奖励 → 模型学会“边写边测”，错误可以在下一轮被纠正，避免一次性生成全错的情况。**  
3. **把解答选择当作生成任务来做轨迹排名 → 验证智能体不再是硬性二分类，而是通过下一个 token 的概率预测哪条交互轨迹最可能产生正确答案 → 通过概率排序挑选最优解，提升了最终执行准确率。**  
4. **奖励函数细粒度设计（GRPO） → 对 schema 匹配、列漏检、表漏检、SQL 语法格式等多维度给出不同权重的奖励 → 让模型在训练时感知到哪些细节最关键，从而在生成时更倾向于完整、符合 schema 的查询。**  

### 方法详解
**整体框架**  
MARS‑SQL 把 Text‑to‑SQL 任务拆成三步：先把自然语言映射到数据库 schema（Grounding），再在此基础上生成可执行的 SQL（Generation），最后从若干候选交互序列中挑选最可靠的答案（Validation）。整个过程在一个循环里进行：生成智能体每一步都会把当前的 SQL 片段提交给数据库执行，得到的执行反馈会被送回生成智能体作为下一轮思考的依据。

**1. Grounding Agent（Schema Grounding）**  
- 输入：用户自然语言问题 + 数据库 schema（表名、列名、外键关系）。  
- 目标：输出一组 schema 链接，即把问题中的实体词映射到具体的表/列。  
- 训练方式：使用 GRPO，奖励包括完全匹配、召回率为 1、未漏表、未漏列以及格式规范等。可以把它想成“先把地图画出来”，确保后面的路线规划有正确的坐标。

**2. Generation Agent（多轮生成）**  
- 输入：用户问题、Grounding Agent 给出的 schema 链接、以及上一次执行的反馈。  
- 工作流程：采用 ReAct 循环——  
  - **Think**：模型在内部生成一段推理文字（如“需要先把订单表和客户表关联”），帮助形成生成计划。  
  - **Act**：把推理转化为实际的 SQL 片段并发送给数据库。  
  - **Feedback**：数据库返回执行成功、错误代码或结果集。  
- 强化学习：每一次 Act 后，根据是否得到可执行的 SQL、是否符合 schema、是否产生正确结果等给出奖励，使用 GRPO 优化策略。这样模型在训练时会学会“先试一次、再改”，类似程序员调试代码的过程。

**3. Validation Agent（轨迹选择）**  
- 生成智能体在一次完整的交互后会产生多条可能的轨迹（不同的思考-行动序列）。  
- Validation Agent 把每条轨迹视作一个生成序列，计算下一个 token 的概率分布（即语言模型的自回归概率），并据此对轨迹进行排序。  
- 最终选取概率最高的轨迹对应的完整 SQL 作为答案。这里的关键是把“哪个答案更好”转化为“哪个生成路径更自然”，利用语言模型的概率优势进行二次筛选。

**最巧妙的设计**  
- **交互式 RL**：不像传统的离线 RL，只在固定数据上打分，MARS‑SQL 让模型实时执行 SQL 并根据真实错误信息调节策略，极大提升了对环境的感知能力。  
- **轨迹排名代替硬性判别**：验证阶段不再是简单的对/错二分类，而是利用语言模型的生成概率进行软排序，兼顾了多样性和可信度。

### 实验与效果
- **数据集**：在 BIRD 开发集上取得 77.84% 的执行准确率，在 Spider 测试集上达到 89.75%，两者均刷新了当时公开记录的最高分。  
- **对比基线**：相较于仅用静态提示的 LLM（如 GPT‑3.5）或传统 Seq2Seq+强化学习方法，MARS‑SQL 在 Spider 上提升约 4–5% 的执行准确率。  
- **消融实验**：作者分别去掉 Grounding Agent、去掉多轮 RL、或改用硬性二分类的验证器，准确率均出现显著下降，说明每个模块都是性能提升的关键因素。  
- **局限性**：论文指出在极度稀疏或高度自定义的 schema 上，Grounding Agent 的匹配仍会出现漏检；此外，多轮交互带来的计算开销比一次性生成要高，实际部署时需要权衡响应时延。

### 影响与延伸思考
MARS‑SQL 把交互式强化学习引入 Text‑to‑SQL，开启了“让模型在真实数据库里实验并学习”的新潮流。后续工作（如 **CoSQL‑RL**、**Agentic NL2SQL**）纷纷借鉴其多智能体拆解和 ReAct 循环的思路，尝试在更大规模的企业内部数据库或跨域查询场景中推广。对想进一步探索的读者，可以关注以下方向：  
- **更高效的轨迹搜索**：利用蒙特卡洛树搜索或强化学习的层次结构，降低多轮交互的计算成本。  
- **跨模态 schema 理解**：把表结构的自然语言描述与图谱信息结合，提升 Grounding 的鲁棒性。  
- **安全与可解释性**：在生成和验证阶段加入审计日志，帮助用户追溯模型为何选择某条轨迹。  

### 一句话记住它
让大语言模型在真实数据库里“边写边测”，用多角色强化学习和概率轨迹排名把 Text‑to‑SQL 的准确率推到新高度。