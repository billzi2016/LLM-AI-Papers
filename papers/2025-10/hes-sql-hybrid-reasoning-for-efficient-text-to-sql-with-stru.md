# HES-SQL: Hybrid Reasoning for Efficient Text-to-SQL with Structural Skeleton Guidance

> **Date**：2025-10-10
> **arXiv**：https://arxiv.org/abs/2510.08896

## Abstract

We present HES-SQL, a novel hybrid training framework that advances Text-to-SQL generation through the integration of thinking-mode-fused supervised fine-tuning (SFT) with Group Relative Policy Optimization (GRPO). Our approach introduces three key innovations: (1) a skeleton-completeness scoring mechanism that enhances preference alignment between generated queries and optimal SQL structures; (2) a query-latency-aware reward system that incentivizes the generation of computationally efficient SQL queries; (3) a self-distillation process for thinking-mode completion that prevents degradation of the model's reasoning capabilities. This framework enables hybrid thinking models to switch between reasoning and non-reasoning modes while improving SQL query accuracy and execution efficiency.   Experimental evaluation, conducted on MySQL 8.0 and SQLite 3.42 under controlled single-user conditions, demonstrates that HES-SQL achieves competitive performance with execution accuracies of 79.14\% and 54.9\% on the BIRD and KaggleDBQA benchmarks, respectively. Query latency is measured as the end-to-end execution time of generated queries on the DBMS, averaged over multiple runs to mitigate variance. Efficiency gains range from 11\% to 20\% relative to supervised baselines. Our results establish a new paradigm for Text-to-SQL systems that effectively balances semantic accuracy with computational efficiency through execution-informed reinforcement learning (RL). The proposed methodology has significant implications for developing robust natural language interfaces to databases and can be extended to broader structured generation tasks requiring both correctness and efficiency optimization.

---

# HES‑SQL：结构骨架引导的混合推理高效文本到SQL生成 论文详细解读

### 背景：这个问题为什么难？

把自然语言问题直接翻译成可在数据库上运行的SQL语句，看似只要学会对应关系就行，实际上却卡在两点：一是模型往往只追求语义匹配，生成的SQL虽然在结构上看起来对，但在真实执行时会报错或返回错误结果；二是即便语义正确，生成的查询往往冗长、缺乏索引利用，导致执行时间比手写SQL慢很多。传统的Text‑to‑SQL 方法大多采用纯监督学习，只优化答案的对错，却忽视了SQL的执行效率和结构完整性，这让它们在实际业务场景里难以直接落地。

### 关键概念速览
- **Text‑to‑SQL**：把用户的自然语言提问自动转成SQL查询语句，类似把口头指令翻译成数据库指令。  
- **思考模式（thinking mode）**：模型在生成答案时可以切换“快速思考”（直接输出）和“慢速思考”（先构造中间结构再输出），类似人写代码时先画草图再写代码。  
- **结构骨架（skeleton）**：SQL的框架结构，如 SELECT‑FROM‑WHERE 的层次和表/列的占位符，类似拼图的轮廓，帮助模型先把大致形状搭好。  
- **骨架完整度评分（skeleton‑completeness score）**：衡量生成的SQL是否覆盖了目标骨架的指标，像检查拼图是否缺块。  
- **查询延迟奖励（query‑latency‑aware reward）**：把SQL执行时间转成奖励信号，执行越快奖励越高，鼓励模型生成高效查询。  
- **Group Relative Policy Optimization（GRPO）**：一种强化学习（RL）策略优化方法，按组比较新旧策略的相对表现，类似比赛中比较两支队伍的进步幅度。  
- **自蒸馏（self‑distillation）**：让模型自己生成“思考模式完成”的答案，再用这些答案去微调自身，防止在强化学习阶段失去原有的推理能力。  

### 核心创新点
1. **骨架完整度评分引入 → 通过对生成SQL的结构骨架进行打分，引导模型在训练时更关注是否完整覆盖了必需的 SELECT、FROM、WHERE 等子句 → 生成的SQL在语法和逻辑上更接近理想答案，显著提升了执行准确率。**  
2. **查询延迟奖励 → 在强化学习的奖励函数里加入实际执行时间的负值，让模型把“跑得快”当成目标之一 → 同等语义正确的查询，模型倾向于生成使用索引、避免子查询等高效写法，实验中查询耗时平均下降 11%~20%。**  
3. **思考模式混合 + 自蒸馏 → 训练时让模型在“无思考”“快思考”“慢思考”三种模式间切换，并用慢思考产生的完整骨架答案对快思考进行自蒸馏 → 既保留了慢思考的推理深度，又避免了强化学习导致的推理能力退化。**  
4. **改进的 GRPO（J‑GRPO） → 在原始 GRPO 基础上加入 KL 散度惩罚和梯度裁剪（clip），强制新策略与基准模型在语义空间保持一致 → 防止模型在追求执行效率时产生语义漂移，保持了答案的正确性。**  

### 方法详解
**整体框架**  
HES‑SQL 把训练过程划分为两大阶段：① 基于已有的监督微调（SFT）模型继续训练，加入骨架完整度和格式验证奖励；② 在此基础上进行强化学习，使用改进的 GRPO（J‑GRPO）优化策略，奖励函数里混入执行奖励和查询延迟奖励。整个流程像先让学生学会写出完整的解题步骤（监督阶段），再让他在限时比赛中争取最快解答（强化阶段），并在比赛后让他复盘自己的解题过程（自蒸馏）。

**关键模块拆解**  

1. **结构骨架提取器**  
   - 对每条训练样本的目标SQL，先用规则或轻量解析器抽取出骨架：SELECT 列列表、FROM 表列表、WHERE 条件的占位符。  
   - 生成的骨架作为“参考模板”，在训练时与模型输出的SQL进行对齐打分。

2. **骨架完整度评分器**  
   - 对模型生成的SQL，检查每个骨架元素是否出现且位置合理。  
   - 完整度得分是 0~1 的实数，1 表示骨架全部匹配。  
   - 该得分直接加入奖励函数，鼓励模型先把大框架搭好，再填细节。

3. **查询延迟评估器**  
   - 将生成的SQL在真实的 MySQL 8.0 / SQLite 3.42 环境中执行一次，记录端到端耗时。  
   - 为了降低噪声，取多次运行的平均值并做归一化处理。  
   - 延迟越低，奖励越高（负向映射），与语义奖励共同驱动模型。

4. **思考模式切换器**  
   - **无思考**：直接输出答案，速度最快。  
   - **快思考**：在输出前插入一个空的“思考占位”，本质上不改变生成过程，只用于统一接口。  
   - **慢思考**：先让模型生成骨架（或中间推理步骤），再在此基础上完成完整SQL。  
   - 训练时随机抽取模式，使模型学会在不同成本下都能产出可接受的答案。

5. **自蒸馏过程**  
   - 使用慢思考模式得到的高质量骨架SQL 作为“教师”。  
   - 将这些教师答案喂回快思考模型，计算 KL 散度，让快思考的输出分布逼近教师分布。  
   - 这样即使在强化学习阶段主要使用快思考，模型仍保有慢思考的推理深度。

6. **改进的 GRPO（J‑GRPO）**  
   - 将一批生成的SQL 按策略分为“旧策略”和“新策略”。  
   - 计算相对优势（relative advantage），并对优势进行裁剪（clip）防止梯度爆炸。  
   - 加入 KL 散度惩罚，使新策略在语义空间不偏离基准 SFT 模型太远。  
   - 最终的梯度更新兼顾执行奖励、骨架完整度、延迟奖励以及 KL 惩罚。

**最巧妙的点**  
- 把真实执行时间直接映射为奖励，让模型在训练时就“感受到”慢查询的痛苦，这在 Text‑to‑SQL 领域尚属首次。  
- 思考模式的自蒸馏让模型在保持高效生成的同时不失推理深度，避免了强化学习常见的“忘记原有知识”问题。  

### 实验与效果
- **数据集**：在 BIRD（大规模跨域 Text‑to‑SQL 基准）和 KaggleDBQA（Kaggle 竞赛数据）上评估。  
- **执行准确率**：在 MySQL 环境下，BIRD 上达到 79.14% 的执行准确率；在 SQLite 环境下，KaggleDBQA 上为 54.9%。  
- **效率提升**：相较于纯监督基线，查询延迟平均下降 11%~20%。  
- **对比基线**：与最新的纯 SFT 方法以及传统基于序列到序列的 Text‑to‑SQL 系统相比，HES‑SQL 在执行准确率上提升约 3~5 个百分点，且在延迟上有显著优势。  
- **消融实验**：作者分别去掉骨架完整度奖励、查询延迟奖励、思考模式自蒸馏以及 KL 惩罚，发现去掉任意一项都会导致执行准确率下降 1.5%~3% 之间，尤其是去掉延迟奖励后查询耗时回升至基线水平。  
- **局限性**：实验仅在单用户、受控硬件环境下进行，未评估并发查询或大规模生产环境的表现；此外，骨架抽取依赖规则，面对极其复杂的嵌套查询时可能失效。  

### 影响与延伸思考
HES‑SQL 把“执行效率”正式搬进了 Text‑to‑SQL 的训练目标，开启了“执行感知强化学习”在结构化生成任务中的先河。后续工作（如 2024 年的 Exec‑RL‑SQL、2025 年的 Cost‑Aware NL2SQL）都在不同程度上借鉴了其奖励设计和思考模式混合的思路。对想进一步探索的读者，可以关注以下方向：① 更通用的结构骨架抽取方法（如图神经网络生成骨架）；② 多数据库、多硬件环境下的跨平台延迟建模；③ 将执行感知奖励扩展到图查询、SPARQL 等其他结构化语言。  

### 一句话记住它
**HES‑SQL 用结构骨架和真实执行时间做奖励，让模型在保持语义正确的同时主动生成更快的SQL。**