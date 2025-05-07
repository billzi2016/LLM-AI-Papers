# Reward-SQL: Boosting Text-to-SQL via Stepwise Reasoning and Process-Supervised Rewards

> **Date**：2025-05-07
> **arXiv**：https://arxiv.org/abs/2505.04671

## Abstract

Recent advances in large language models (LLMs) have significantly improved performance on the Text-to-SQL task by leveraging their powerful reasoning capabilities. To enhance accuracy during the reasoning process, external Process Reward Models (PRMs) can be introduced during training and inference to provide fine-grained supervision. However, if misused, PRMs may distort the reasoning trajectory and lead to suboptimal or incorrect SQL generation. To address this challenge, we propose Reward-SQL, a framework that systematically explores how to incorporate PRMs into the Text-to-SQL reasoning process effectively. Our approach follows a "cold start, then PRM supervision" paradigm. Specifically, we first train the model to decompose SQL queries into structured stepwise reasoning chains using common table expressions (Chain-of-CTEs), establishing a strong and interpretable reasoning baseline. Then, we investigate four strategies for integrating PRMs, and find that combining PRM as an online training signal (e.g.,GRPO) with PRM-guided inference (e.g., best-of-N sampling) yields the best results. Empirically, on the BIRD benchmark, Reward-SQL enables models supervised by PRM (7B) to achieve a 13.1% performance gain across various guidance strategies. Notably, our GRPO-aligned policy model based on Qwen2.5-Coder-7B-Instruct achieves 68.9% accuracy on the BIRD development set, outperforming all baseline methods under the same model size. These results demonstrate the effectiveness of Reward-SQL in leveraging reward-based supervision for Text-to-SQL reasoning.

---

# Reward‑SQL：通过逐步推理与过程监督奖励提升 Text‑to‑SQL 论文详细解读

### 背景：这个问题为什么难？

Text‑to‑SQL 要求模型把自然语言问题转换成可以在关系数据库上执行的 SQL 语句。传统方法往往直接让大模型一次性输出完整的 SQL，遇到多表关联、嵌套查询或复杂的过滤条件时容易走偏，因为模型缺少对中间推理步骤的可视化和校正。即使是已经很强的 LLM，也会因为“黑箱”式生成而在细节上出错，导致执行结果不符合用户意图。于是，如何让模型在生成 SQL 的过程中得到细粒度的监督，成为提升准确率的关键瓶颈。

### 关键概念速览
- **Text‑to‑SQL**：把自然语言问句翻译成结构化的 SQL 查询语句，类似把口头指令变成数据库指令。
- **Process Reward Model（PRM）**：一种专门评估模型生成过程质量的奖励模型，它会给每一步的中间产出打分，就像老师给学生写作的每段文字打分一样。
- **Chain‑of‑CTEs（CTE 链）**：把复杂的 SQL 拆成一系列公共表表达式（CTE），每一步都产生一个临时表，层层堆叠后得到最终结果，类似把一道大题拆成若干小题逐个解答。
- **GRPO（Gradient‑based Reward‑guided Policy Optimization）**：一种把 PRM 的奖励信号直接用于梯度更新的在线训练方式，让模型在学习时就“感受到”奖励的方向。
- **Best‑of‑N 采样**：在推理阶段生成 N 条候选答案，挑选奖励最高的那一条，就像让模型多写几篇草稿，挑最好的交给老师评分。
- **BIRD 基准**：一个覆盖多种数据库模式和查询难度的大规模 Text‑to‑SQL 测评集合，用来检验模型的通用性。

### 核心创新点
1. **先让模型学会 CTE 链 → 再引入 PRM**  
   过去的工作直接在原始自然语言上加奖励，容易让模型走偏。Reward‑SQL 先用监督学习让模型把 SQL 拆成结构化的 CTE 链，形成可解释的推理路径。这样模型在后续接受奖励时，有了明确的“步骤”可供评估，显著降低了奖励误导的风险。

2. **系统化四种 PRM 融合策略 → 找到最佳组合**  
   作者分别尝试了（1）仅在推理时用 PRM 选样本、（2）仅在训练时用 PRM 作为奖励、（3）两者混合、（4）把 PRM 当作约束过滤。实验表明，把 GRPO 作为在线训练信号再配合 Best‑of‑N 采样的推理策略，效果最突出，说明奖励既能指导学习，又能在生成时筛选。

3. **冷启动 + 过程监督的训练范式**  
   传统的奖励学习往往从一开始就加入奖励，导致模型在没有基本能力的情况下被噪声奖励牵着走。Reward‑SQL 采用“冷启动”——先让模型掌握基本的 CTE 分解，再逐步加入 PRM，确保奖励作用在已经具备基本推理能力的模型上，提升了收敛速度和最终准确率。

4. **在 7B 参数模型上实现显著提升**  
   在同等模型规模下，使用 Reward‑SQL 的 Qwen2.5‑Coder‑7B‑Instruct 在 BIRD 开发集上达到 68.9% 的准确率，超过所有同尺寸基线，并且相较于仅使用 PRM 的版本提升了 13.1%。这证明了过程奖励在中小模型上的实用价值。

### 方法详解
**整体框架**  
Reward‑SQL 的训练与推理分为两大阶段：  
1️⃣ **结构化推理预训练**：模型学习把自然语言问题映射为 CTE 链；  
2️⃣ **过程奖励微调**：在已有的 CTE 基础上，引入 PRM 进行在线奖励优化，并在推理时使用奖励驱动的采样策略。

**步骤拆解**  

1. **CTE 链构造**  
   - 输入：自然语言问句。  
   - 输出：一系列形如 `WITH step_i AS ( … )` 的 CTE，每个 CTE 对应一次子查询或中间表。  
   - 训练方式：使用公开的 Text‑to‑SQL 数据集的标注 SQL，先用规则或模板把它们拆解成 CTE 链，形成监督对齐。模型的目标是最小化生成的 CTE 链与标注链的差异。

2. **过程奖励模型（PRM）**  
   - PRM 本身是一个小型的判别模型，输入是当前生成的 CTE 步骤和数据库 schema，输出一个标量奖励，表示这一步是否合理。  
   - 奖励的设计遵循“局部正确性”：如果一步产生的临时表能够在后续步骤中被正确引用，则奖励高；否则低。

3. **GRPO 在线微调**  
   - 在每一次前向生成 CTE 链时，实时调用 PRM 计算每一步的奖励。  
   - 将这些奖励加权求和后，作为强化学习的回报信号，直接对生成模型的参数做梯度更新（类似 REINFORCE，但使用了基线来降低方差）。  
   - 关键点在于奖励被视作“软约束”，而不是硬性惩罚，避免模型因为单一步骤的低分而整体崩溃。

4. **Best‑of‑N 推理**  
   - 推理时让模型一次性生成 N 条完整的 CTE 链（N 通常取 5~10）。  
   - 对每条链调用 PRM 计算整体奖励，挑选奖励最高的那一条转化为最终 SQL。  
   - 这一步相当于“多草稿选优”，利用 PRM 的评估能力在生成阶段进一步过滤错误。

**最巧妙的设计**  
- **冷启动 + 过程监督**：先让模型学会结构化推理，再加奖励，避免了奖励信号在模型尚未具备基本能力时的噪声干扰。  
- **奖励的层次化使用**：GRPO 把奖励嵌入梯度更新，Best‑of‑N 把奖励用于后期筛选，两者相辅相成，形成了从学习到推理的全链路监督。

### 实验与效果
- **数据集**：主要在 BIRD 基准上评估，BIRD 包含多种数据库 schema 与不同难度的查询，能够全面检验模型的通用性。  
- **基线对比**：与同尺寸的 LLM（如未使用 PRM 的 Qwen2.5‑Coder‑7B、传统 CoT 方法等）相比，Reward‑SQL 的 GRPO‑aligned Qwen2.5‑Coder‑7B‑Instruct 在开发集上达到 **68.9%** 的准确率，领先所有同规模方法。  
- **增益幅度**：在加入 PRM 的情况下，整体性能提升 **13.1%**（相对增幅），说明过程奖励对模型的帮助是显著的。  
- **消融实验**：作者分别去掉 GRPO、去掉 Best‑of‑N、仅使用冷启动等配置，发现去掉任意一环都会导致准确率下降 3%~6%，验证了每个模块的必要性。  
- **局限性**：论文指出 PRM 的训练需要额外的标注或模拟奖励数据，且在极端长查询或非常稀疏的 schema 上奖励信号可能不够精准；此外，Best‑of‑N 采样会带来一定的推理时间开销。

### 影响与延伸思考
Reward‑SQL 为 Text‑to‑SQL 引入了“过程监督”这一新范式，激发了后续工作在其他结构化生成任务（如代码生成、数据可视化脚本）中尝试类似的 PRM 机制。近期有研究把过程奖励与自检（self‑verification）结合，进一步提升模型的自我纠错能力。想继续深挖的读者可以关注以下方向：  
- **跨任务的通用 PRM 设计**：如何让同一个奖励模型服务于多种生成任务。  
- **更高效的多样本采样**：在保持质量的前提下降低 Best‑of‑N 的计算成本。  
- **奖励信号的可解释性**：让 PRM 的评分过程对人类更透明，便于调试和安全审查。

### 一句话记住它
**Reward‑SQL 通过先让模型学会分步的 CTE 推理，再用过程奖励在训练和推理两端同步指路，实现了小模型在 Text‑to‑SQL 上的显著跃迁。**