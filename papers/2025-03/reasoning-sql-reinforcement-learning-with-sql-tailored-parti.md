# Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards   for Reasoning-Enhanced Text-to-SQL

> **Date**：2025-03-29
> **arXiv**：https://arxiv.org/abs/2503.23157

## Abstract

Text-to-SQL is a challenging task involving multiple reasoning-intensive subtasks, including natural language understanding, database schema comprehension, and precise SQL query formulation. Existing approaches often rely on handcrafted reasoning paths with inductive biases that can limit their overall effectiveness. Motivated by the recent success of reasoning-enhanced models such as DeepSeek R1 and OpenAI o1, which effectively leverage reward-driven self-exploration to enhance reasoning capabilities and generalization, we propose a novel set of partial rewards tailored specifically for the Text-to-SQL task. Our reward set includes schema-linking, AI feedback, n-gram similarity, and syntax check, explicitly designed to address the reward sparsity issue prevalent in reinforcement learning (RL). Leveraging group relative policy optimization (GRPO), our approach explicitly encourages large language models (LLMs) to develop intrinsic reasoning skills necessary for accurate SQL query generation. With models of different sizes, we demonstrate that RL-only training with our proposed rewards consistently achieves higher accuracy and superior generalization compared to supervised fine-tuning (SFT). Remarkably, our RL-trained 14B-parameter model significantly outperforms larger proprietary models, e.g. o3-mini by 4% and Gemini-1.5-Pro-002 by 3% on the BIRD benchmark. These highlight the efficacy of our proposed RL-training framework with partial rewards for enhancing both accuracy and reasoning capabilities in Text-to-SQL tasks.

---

# Reasoning‑SQL：面向推理增强的 Text‑to‑SQL 的 SQL 定制化部分奖励强化学习 论文详细解读

### 背景：这个问题为什么难？

Text‑to‑SQL 要把自然语言问题直接翻译成可以在数据库上运行的 SQL 语句，过程里模型必须先弄懂用户的意图、再匹配数据库的表结构，最后生成语法严谨、语义正确的查询。传统方法大多靠大规模监督学习，或者手工设计的推理路径（比如先找表再找列），这些“硬编码”的步骤把模型的自由度限制住，导致在新 schema 或复杂多表联结时容易卡壳。更关键的是，强化学习（RL）在这类任务上奖励极其稀疏——只有当整个 SQL 完全正确时才给正向信号，模型很难在海量搜索空间里找到有效的学习信号。于是，提升模型的自我推理能力、同时解决奖励稀疏问题，成为了迫切需要突破的瓶颈。

### 关键概念速览
- **Text‑to‑SQL**：把自然语言问句自动转成对应的 SQL 查询，类似把口头指令翻译成数据库指令。  
- **强化学习（RL）**：模型通过与环境交互获得奖励来学习策略，就像玩游戏时根据得分调整打法。  
- **奖励稀疏**：只有在极少数情况下（比如完整 SQL 正确）才会得到奖励，导致学习过程像在大海里找针。  
- **部分奖励（Partial Rewards）**：把整体奖励拆成若干小块，每完成一步就给分，类似做题时老师给的逐步提示。  
- **Schema‑linking**：把自然语言中的实体（如“订单金额”）映射到数据库的表/列名称的过程。  
- **AI 反馈（AI Feedback）**：让模型自行检查生成的 SQL 并给出错误提示，再据此调整生成策略。  
- **n‑gram 相似度**：比较模型输出的 SQL 与参考答案在词序列上的相似程度，类似拼写检查。  
- **语法检查（Syntax Check）**：验证生成的 SQL 是否符合语法规则，防止出现非法关键字或缺失括号。  
- **GRPO（Group Relative Policy Optimization）**：一种强化学习的策略优化算法，利用同一批次中不同模型的相对表现来平滑梯度，像是团队内部的相对排名来决定奖励分配。

### 核心创新点
1. **从整体奖励到多维部分奖励**  
   - 之前的 RL 方法往往只在整个查询完全正确时给奖励，导致学习信号极其稀疏。  
   - 本文设计了四类细粒度奖励：schema‑linking、AI 反馈、n‑gram 相似度和语法检查。每当模型在这些子任务上取得进展，就会得到对应的正向奖励。  
   - 这种拆分让模型在生成过程的每一步都有明确的学习目标，大幅提升了收敛速度和最终准确率。

2. **使用 GRPO 进行相对策略优化**  
   - 传统的 PPO（Proximal Policy Optimization）只看单个模型的奖励变化，容易受噪声影响。  
   - 作者引入 GRPO，先把同一批次的模型输出分成若干组，比较组内相对表现，再根据相对优势调整策略。  
   - 这种相对评估让梯度更稳健，尤其在奖励本身已经被细化的情况下，进一步降低了训练波动。

3. **纯 RL 训练即可超越监督微调**  
   - 过去的工作大多先用大规模标注数据做监督微调（SFT），再加一点 RL 微调。  
   - 本文直接用 RL（仅使用部分奖励）训练模型，省掉了监督阶段，却在多个模型规模上 consistently 超过了同等规模的 SFT 基线。  
   - 说明细粒度奖励和 GRPO 已经足够提供强大的学习信号，摆脱了对大规模标注数据的依赖。

4. **小模型也能击败更大的商业模型**  
   - 在 BIRD 基准上，14B 参数的模型在仅用 RL 训练后，比 OpenAI 的 o3‑mini（更大）高出约 4%，比 Gemini‑1.5‑Pro‑002（同样更大）高出约 3%。  
   - 这表明方法的通用性：只要奖励设计得当，即使是相对小的模型也能在推理密集的 Text‑to‑SQL 任务上实现领先。

### 方法详解
**整体框架**  
整个训练流程可以划分为三步：① 生成候选 SQL；② 对每个候选计算四类部分奖励；③ 用 GRPO 根据奖励更新模型策略。模型本身是一个标准的自回归大语言模型（LLM），输入是自然语言问句加上数据库 schema 描述，输出是逐 token 的 SQL 序列。

**步骤 1：候选生成**  
- 输入格式为：“Question: … Schema: Table1(col1, col2, …); Table2(…)”。  
- 模型使用采样（如 nucleus sampling）一次生成多个候选（通常 4‑8 条），相当于让模型在同一次前向传播中“思考”多条可能的解法。

**步骤 2：部分奖励计算**  
| 奖励类型 | 计算方式 | 直观意义 |
|---|---|---|
| Schema‑linking | 检查候选 SQL 中出现的表/列是否与问句中的实体对应，使用词向量相似度或规则匹配 | 确保模型把自然语言的概念正确映射到数据库结构 |
| AI 反馈 | 将候选 SQL 送回模型自身的评估子模块，让它输出错误类型（如列未匹配、聚合错误），依据错误数量给分 | 类似人类自检，鼓励模型主动发现并纠正错误 |
| n‑gram 相似度 | 计算候选与参考 SQL 的 n‑gram 重叠率（如 BLEU‑like），给出比例奖励 | 鼓励整体语言形式接近正确答案 |
| Syntax Check | 使用 SQL 解析器检查语法合法性，合法则奖励，否则扣分 | 防止模型生成根本无法执行的查询 |

每个奖励都有一个预设的权重，作者在实验中通过小规模搜索得到平衡系数，使得不同奖励的尺度相当。

**步骤 3：GRPO 更新**  
- 将同一批次的候选按照总奖励（四类加权和）分成若干组（例如高分组、低分组）。  
- 对每个组内部计算相对优势：高分组相对于低分组的奖励差距。  
- 依据优势对策略梯度进行加权，类似“在团队里，表现好的成员得到更多的学习信号”。  
- 这种相对更新方式可以抑制单个异常高奖励的噪声，因为它是基于组间比较而不是绝对值。

**最巧妙的设计**  
- **奖励拆解**：把原本稀疏的“全对奖励”拆成四个可独立评估的子任务，让模型在每一步都有明确的反馈。  
- **自我反馈回路**：AI 反馈模块本身也是同一个 LLM 的子模型，省去额外的评估器，形成闭环自检。  
- **相对优化**：GRPO 把奖励的相对差异放大，避免了单纯使用绝对奖励时的梯度方差问题。

### 实验与效果
- **数据集**：主要在 BIRD（大规模跨域 Text‑to‑SQL 基准）上评估，还包括少量公开的 Spider 子集用于验证跨域迁移。  
- **基线对比**：与同等规模的监督微调模型（SFT）、传统 PPO 强化学习以及最新的商业大模型（OpenAI o3‑mini、Gemini‑1.5‑Pro‑002）进行比较。  
- **核心结果**：14B 参数模型在 BIRD 上的执行准确率比 o3‑mini 高约 4%，比 Gemini‑1.5‑Pro‑002 高约 3%。在相同模型规模下，RL‑only 训练的模型普遍比 SFT 提升 2‑5% 的准确率。  
- **消融实验**：原文提供了对每种部分奖励的单独去除实验，发现去掉 **Schema‑linking** 会导致整体准确率下降约 1.8%，去掉 **AI 反馈** 降幅约 1.2%，而 **Syntax Check** 的贡献相对较小（约 0.6%），但对生成合法 SQL 至关重要。  
- **局限性**：作者指出当前奖励仍然依赖于参考 SQL（用于 n‑gram 计算），在完全无标注的零样本场景下效果未知；GRPO 的组划分在极端大模型上会增加显存开销；此外，奖励权重需要在新 schema 上重新调优，自动化程度还有提升空间。

### 影响与延伸思考
这篇工作在 Text‑to‑SQL 社区掀起了“细粒度奖励+相对优化”的潮流。随后有几篇论文尝试把类似的部分奖励迁移到代码生成、数学推理等任务上，证明了奖励拆解的通用性。还有研究把 **AI 反馈** 换成外部的检验器（如数据库执行器）形成更强的闭环。对想进一步探索的读者，可以关注以下方向：① 自动学习奖励权重的元学习方法；② 将 GRPO 与最新的 **PPO‑Clip** 或 **DPPO** 结合，提升大模型的训练稳定性；③ 在多模态（文本+表格）场景下扩展 schema‑linking 的表示能力。整体来看，细粒度奖励与相对策略优化为强化学习在高维结构化生成任务中的落地提供了可操作的路径。

### 一句话记住它
把稀疏的“全对奖励”拆成四个可即时评估的子奖励，并用相对策略优化让模型自我纠错，轻松让小模型在 Text‑to‑SQL 上跑赢更大的商业模型。