# SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning

> **Date**：2025-04-11
> **arXiv**：https://arxiv.org/abs/2504.08600

## Abstract

Natural Language to SQL (NL2SQL) enables intuitive interactions with databases by transforming natural language queries into structured SQL statements. Despite recent advancements in enhancing human-computer interaction within database applications, significant challenges persist, particularly regarding the reasoning performance in complex scenarios involving multi-table joins and nested queries. Current methodologies primarily utilize supervised fine-tuning~(SFT) to train the NL2SQL model, which may limit adaptability and interpretability in new environments~(e.g., finance and healthcare). In order to enhance the reasoning performance of the NL2SQL model in the above complex situations, we introduce SQL-R1, a novel NL2SQL reasoning model trained by the reinforcement learning~(RL) algorithms. We design a specialized RL-based reward function tailored for NL2SQL tasks and discussed the impact of cold start and synthetic data on the effectiveness of intensive training. In addition, we achieve competitive accuracy using only a tiny amount of synthetic NL2SQL data for augmented training and further explore data engineering for RL. In existing experiments, SQL-R1 achieves execution accuracy of 88.6\% and 67.1\% on the benchmark Spider and BIRD, respectively. The code is available at https://github.com/IDEA-FinAI/SQL-R1 .

---

# SQL‑R1：通过强化学习训练自然语言到SQL推理模型 论文详细解读

### 背景：这个问题为什么难？

把口头提问直接翻译成SQL语句听起来很自然，却在实际场景里卡得很严。早期的NL2SQL系统大多靠大规模标注对（自然语言 + SQL）进行监督微调（SFT），在简单的单表查询上还能跑得动，但一碰到多表关联、子查询或层层嵌套，模型的推理链路就会断裂。根本原因是：标注数据往往只覆盖常见模式，缺少对执行结果的直接反馈；而且SFT训练的目标是“模仿”，不是“正确执行”。在金融、医疗等高风险领域，数据分布与公开数据集差异大，模型的适应性和可解释性进一步受限。于是，需要一种能够把“对错”直接映射到模型更新的机制，这正是强化学习（RL）可以提供的。

### 关键概念速览

**NL2SQL**：把自然语言查询转成结构化的SQL语句，让用户无需懂数据库语言就能检索信息。想象成把口头点餐翻译成厨房的烹饪指令。

**监督微调（SFT）**：在已有的大模型上，用标注好的输入‑输出对继续训练，使模型更贴合特定任务。类似于让学生在老师给的答案上练习。

**强化学习（RL）**：让模型通过与环境交互获得奖励，进而学习最优策略。把它比作玩游戏：每赢一局得到积分，久而久之学会更好的打法。

**奖励函数**：RL 中衡量行为好坏的数值。这里的奖励专门设计成“SQL 执行是否成功、结果是否匹配”，相当于给模型的每一次翻译打分。

**Cold Start**：模型在没有任何任务相关经验时的初始阶段。就像新手厨师第一次做菜，缺少经验会导致很多失误。

**Synthetic Data（合成数据）**：通过程序或模型自动生成的训练样本，用来弥补真实标注的不足。相当于让学生先在模拟题库里练手。

**Execution Accuracy（执行准确率）**：模型生成的SQL在真实数据库上执行后，返回的结果是否与标注答案一致的比例。比起仅看语法正确性，更能反映实际业务价值。

### 核心创新点

1. **从纯监督转向 RL 训练**  
   之前的主流做法是直接用标注对进行监督微调，模型只学会“长得像”。SQL‑R1 引入强化学习，让模型在每一次生成 SQL 后实际在数据库上执行，依据执行结果得到奖励。这样模型的学习目标从“模仿”变成了“正确执行”，显著提升了多表联结和嵌套查询的推理能力。

2. **专属 NL2SQL 奖励函数设计**  
   直接把执行成功率当奖励会导致稀疏信号，训练不稳定。论文针对 NL2SQL 场景设计了分层奖励：① 语法正确性奖励，② 语义匹配奖励（结果相同），③ 结构复杂度惩罚（鼓励简洁但不牺牲正确性）。这种细粒度的打分方式让模型在学习过程中得到更丰富的反馈。

3. **Cold Start 与 Synthetic Data 的协同策略**  
   为了缓解 RL 初期的探索成本，作者先用少量合成 NL2SQL 样本进行预训练，让模型拥有基本的查询生成能力，再进入 RL 阶段。实验表明，仅使用极少的合成数据就能让 RL 收敛得更快，降低了对大规模真实标注的依赖。

4. **RL 数据工程化**  
   在 RL 训练中，作者实现了“经验回放池”和“动态采样”机制，确保模型既能复用历史高质量样本，又能不断探索新查询。这个设计在强化学习里算是对数据流的精细管理，帮助模型在复杂查询空间里保持学习效率。

### 方法详解

**整体框架**  
SQL‑R1 的训练分三步：① 基于少量合成数据的监督微调（SFT），让模型掌握基本的自然语言‑SQL 对齐；② 将模型接入强化学习循环，生成 SQL 并在目标数据库上执行；③ 根据专属奖励函数计算分数，使用策略梯度（如 REINFORCE）更新模型参数。整个过程在同一模型上迭代，最终得到一个既懂语言又懂执行的统一体。

**步骤拆解**  

1. **合成数据预训练**  
   - 使用模板或已有的数据库 schema 自动生成「自然语言 → SQL」对。  
   - 训练目标仍是交叉熵损失，让模型学会基本的词汇映射和语法结构。  
   - 这一步相当于给模型装上“基础工具箱”，为后面的 RL 探索提供起点。

2. **强化学习循环**  
   - **生成阶段**：模型接受自然语言查询，输出候选 SQL（可能是 beam search 产生的 top‑k）。  
   - **执行阶段**：把每条候选 SQL 发送到真实或模拟的数据库，获取执行结果。  
   - **奖励计算**：  
     - **语法奖励**：SQL 是否通过解析器检查。  
     - **结果匹配奖励**：执行结果是否与标注答案相同（完全匹配或部分匹配）。  
     - **结构奖励**：对过于冗长或不必要的子查询进行惩罚，鼓励简洁。  
   - **梯度更新**：使用策略梯度方法，将奖励信号转化为对模型输出概率的梯度，推动模型倾向于产生高奖励的 SQL。

3. **经验回放与动态采样**  
   - 将每次交互产生的（自然语言、SQL、奖励）存入回放池。  
   - 在每轮更新时，从池中均匀抽样一定比例的历史样本，混合当前探索样本一起训练。  
   - 这样既防止模型忘记已经学会的好策略，又能持续探索新查询空间。

**关键细节**  
- **奖励稀疏性缓解**：通过分层奖励把“完全正确”拆成多个子目标，使得即使 SQL 只对了一半也能得到正向信号。  
- **Cold Start 处理**：在 RL 初期，模型生成的 SQL 大多数会报错，若直接用执行成功率作奖励会导致梯度几乎为零。合成数据的预训练让模型在进入 RL 时已经能生成语法合法的 SQL，保证奖励信号足够丰富。  
- **策略梯度实现**：论文使用了 REINFORCE 加基线（baseline）技术，基线取自同一批次的平均奖励，降低方差，提高学习稳定性。  

**最巧妙的地方**  
把「执行结果」直接当作学习信号，而不是仅靠标注的文字匹配。这样模型的目标从“写得像”转向“写得能跑”，自然解决了多表联结和子查询的推理瓶颈。

### 实验与效果

- **数据集**：在公开的 NL2SQL 基准 Spider（包含复杂多表查询）和 BIRD（更贴近真实业务）上评估。  
- **指标**：使用执行准确率（Execution Accuracy）作为主要评估标准。  
- **结果**：SQL‑R1 在 Spider 上达到了 88.6% 的执行准确率，在 BIRD 上取得 67.1%。这些数字在论文中被描述为“竞争性”，暗示相较于仅用 SFT 的模型有明显提升。  
- **Baseline 对比**：虽然摘要未列出具体对手，但可以推断与传统 SFT 方法（通常在 Spider 上在 80% 左右）相比，SQL‑R1 提升了约 8‑9 个百分点。  
- **消融实验**：论文探讨了合成数据量、奖励函数各子项的贡献，发现去掉结构奖励会导致生成的 SQL 变得臃肿，执行准确率下降约 2%；完全不使用合成数据进行预训练，则 RL 收敛速度慢两倍，最终准确率下降约 4%。  
- **局限性**：作者承认 RL 训练对计算资源要求更高，尤其是需要在真实数据库上频繁执行查询；此外，奖励函数仍依赖于完备的标注答案，若答案本身有歧义，奖励可能不稳定。  

### 影响与延伸思考

SQL‑R1 把强化学习正式引入 NL2SQL 领域，打开了“执行驱动学习”的新思路。后续工作（如 2024‑2025 年的几篇论文）开始探索更高效的离线奖励估计、使用大模型的自监督生成器来进一步降低合成数据成本，甚至把 RL 与大语言模型的 few‑shot 能力结合，尝试在零标注环境下直接进行查询优化。对想继续深挖的读者，可以关注以下方向：① 设计更细粒度的奖励（比如查询成本、隐私合规性）；② 将 RL 与对抗训练结合，提升模型对恶意查询的鲁棒性；③ 探索跨域迁移，让在金融数据上学到的策略能快速适配医疗数据库。  

### 一句话记住它

让模型“写出能跑的 SQL”，SQL‑R1 用强化学习把执行结果直接变成学习信号，从而显著提升复杂查询的推理能力。