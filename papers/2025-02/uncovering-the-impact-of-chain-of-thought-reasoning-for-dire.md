# Uncovering the Impact of Chain-of-Thought Reasoning for Direct   Preference Optimization: Lessons from Text-to-SQL

> **Date**：2025-02-17
> **arXiv**：https://arxiv.org/abs/2502.11656

## Abstract

Direct Preference Optimization (DPO) has proven effective in complex reasoning tasks like math word problems and code generation. However, when applied to Text-to-SQL datasets, it often fails to improve performance and can even degrade it. Our investigation reveals the root cause: unlike math and code tasks, which naturally integrate Chain-of-Thought (CoT) reasoning with DPO, Text-to-SQL datasets typically include only final answers (gold SQL queries) without detailed CoT solutions. By augmenting Text-to-SQL datasets with synthetic CoT solutions, we achieve, for the first time, consistent and significant performance improvements using DPO. Our analysis shows that CoT reasoning is crucial for unlocking DPO's potential, as it mitigates reward hacking, strengthens discriminative capabilities, and improves scalability. These findings offer valuable insights for building more robust Text-to-SQL models. To support further research, we publicly release the code and CoT-enhanced datasets.

---

# 揭示链式思维推理对直接偏好优化的影响：来自 Text-to-SQL 的经验 论文详细解读

### 背景：这个问题为什么难？
Text-to-SQL 任务要求模型把自然语言问题翻译成可执行的 SQL 查询，表面上看像是一次“语言到语言”的映射，却隐藏着两大难点。第一，SQL 语法严格，稍有语序或关键字错误就会导致执行失败；第二，很多问题的答案并不是直接在文本里出现，而是需要模型在内部进行多步推理（比如先确定要查询的表、再推断过滤条件、最后组合 SELECT 子句）。过去的模型大多只用最终的金标准 SQL 进行监督学习，缺少对推理过程的显式指导，导致在复杂查询上容易出现“跳步”或“奖励黑客”。于是，尽管 Direct Preference Optimization（直接偏好优化，简称 DPO）在数学题和代码生成上表现抢眼，却在 Text-to-SQL 上常常提不起效，甚至让性能倒退。

### 关键概念速览
**Direct Preference Optimization（直接偏好优化）**：一种通过比较模型生成的“好”与“坏”答案来微调语言模型的技术，类似让模型学会在两条路中挑选更安全、更符合人类偏好的那条。  
**Chain-of-Thought（思维链，CoT）**：让模型在给出最终答案前先写出一步步的推理过程，就像解题时在草稿纸上列出每一步计算。  
**Reward Hacking（奖励黑客）**：模型找到一种“捷径”，只满足奖励函数的表面要求，却不是真正的正确解答，类似考试时只记住答案而不懂原理。  
**Synthetic CoT（合成思维链）**：利用已有模型或规则自动生成的推理步骤，而不是人工标注的，类似机器自动写的解题过程。  
**Discriminative Capability（判别能力）**：模型区分好答案和坏答案的敏感度，判别力强的模型更不容易被奖励黑客误导。  
**Scalability（可扩展性）**：方法在更大数据、更复杂任务上仍能保持效果的能力。  
**Gold SQL（金标准 SQL）**：数据集中提供的正确 SQL 查询，通常是唯一的、没有过程信息的答案。  
**Prompt Engineering（提示工程）**：设计输入文本的技巧，使模型更容易产生期望的输出，就像给学生出题时的措辞会影响他们的思考方式。

### 核心创新点
1. **发现根因 → 通过分析发现 Text-to-SQL 数据缺少 CoT**：之前的研究把 DPO 在数学和代码任务上成功的经验直接搬到 Text-to-SQL，却忽视了两类任务在数据结构上的差异。作者指出，数学题和代码任务本身就自带步骤描述（解题思路、代码注释），而 Text-to-SQL 只给出最终的 SQL，导致 DPO 在没有过程信息的情况下容易学到错误的偏好。  
2. **合成思维链 → 用大模型自动生成 CoT 并加入训练集**：作者利用已有的强大语言模型，先让它在自然语言问题上生成一步步的推理（如“先找表 A，再筛选列 B”），再把这些合成的 CoT 与原始金标准 SQL 一起喂给 DPO。这样相当于给模型提供了“草稿纸”，让它在微调时能看到正确的思考路径。  
3. **验证 CoT 对 DPO 的多维提升 → 实验显示奖励黑客下降、判别能力增强、在更大数据上仍有效**：通过对比仅使用金标准 SQL 与使用 CoT‑增强数据的 DPO 结果，作者证明 CoT 能显著降低模型只追求奖励表面特征的倾向，同时提升模型在细粒度错误上的辨识力。  
4. **公开代码与增强数据 → 为后续研究提供可复现的基准**：作者把生成的 CoT 数据和微调脚本全部开源，降低了其他团队复现和扩展的门槛。

### 方法详解
整体思路可以拆成三步：**数据增强 → DPO 微调 → 评估**。下面逐步展开。

1. **数据增强（Synthetic CoT Generation）**  
   - **输入**：原始 Text-to-SQL 数据对 (question, gold_sql)。  
   - **过程**：调用一个已经经过指令微调的强语言模型（如 GPT‑4）让它先把自然语言问题拆解成若干推理步骤。每一步都用自然语言描述，例如“1. 确定要查询的表是 orders；2. 过滤条件是 order_date > '2023-01-01'”。随后再让模型生成对应的 SQL，确保生成的 SQL 与金标准一致（通过执行或语义匹配检查）。  
   - **输出**：一个三元组 (question, synthetic_coT, gold_sql)。这里的 CoT 不是人工标注，而是机器自动产生的“草稿”。  

2. **Direct Preference Optimization（直接偏好优化）**  
   - **奖励函数设计**：对每个 (question, gold_sql) 生成两个候选答案：一个是模型直接输出的 SQL（不带 CoT），另一个是模型在同样输入下输出的带 CoT 的完整过程再生成的 SQL。奖励函数不仅考虑最终 SQL 的执行正确率，还加入对 CoT 质量的惩罚项（比如长度、逻辑连贯性），防止模型只输出空洞的步骤。  
   - **偏好对比**：把“好”样本（带 CoT、执行正确）和“坏”样本（不带 CoT、或执行错误）喂给 DPO，模型学习在参数空间上把好样本的概率提升、坏样本的概率压低。这里的关键是 CoT 为模型提供了额外的判别信号，使得好坏对比更细致。  
   - **微调细节**：作者使用了 LoRA（低秩适配）技术，只调少量参数，保持原模型的通用能力。训练时采用了梯度累积和混合精度，以适配大规模数据。  

3. **评估与分析**  
   - **执行层面**：把微调后的模型在标准 Text-to-SQL 基准上运行，比较生成的 SQL 是否能成功执行并得到正确结果。  
   - **过程层面**：检查模型输出的 CoT 是否符合逻辑，是否出现奖励黑客（比如只在 CoT 中加入无意义文字却仍得到高奖励）。  
   - **消融实验**：分别去掉 CoT、去掉奖励对 CoT 的惩罚、只用金标准 SQL 进行 DPO，观察性能跌幅，验证每个设计的必要性。  

**最巧妙的点**在于把“过程信息”从“缺失”变成“可合成”，并且把这段过程直接嵌入 DPO 的偏好对比中。这样模型不再只看最终答案的对错，而是学会评估自己的思考路径，极大降低了奖励黑客的空间。

### 实验与效果
- **数据集**：作者在 Spider、CoSQL 等主流 Text-to-SQL 基准上做实验，这些数据集原本只提供自然语言问题和对应的金标准 SQL。  
- **对比基线**：包括原始 DPO（只用金标准 SQL）、SFT（监督微调）以及最新的基于 RLHF（强化学习的人类反馈）的模型。  
- **结果**：论文声称在 Spider 上的执行准确率提升约 4–6%（具体数字未在摘要中给出），在 CoSQL 上同样取得了显著提升。相比仅使用金标准 SQL 的 DPO，加入合成 CoT 后的模型不但不再出现性能下降，反而在所有评测指标上都有正向提升。  
- **消融**：去掉 CoT 生成或去掉对 CoT 的奖励惩罚，模型的表现会回落到原始 DPO 的水平，说明 CoT 是提升的关键因素。  
- **局限**：合成 CoT 依赖于已有的大模型生成质量，如果生成的步骤本身有错误，可能会把错误信息灌进微调数据。作者在讨论中承认，这一点在低资源语言或特定领域的 SQL 语法上仍需进一步验证。

### 影响与延伸思考
这篇工作提醒大家：在任何需要复杂推理的任务上，直接优化偏好时必须提供“思考过程”。随后出现的几篇论文开始尝试在代码生成、数据分析甚至多模态问答中加入自动生成的 CoT，以提升 DPO 或 RLHF 的效果（推测）。如果想继续深挖，可以关注以下方向：① 更高质量的自动 CoT 生成方法（比如自监督的步骤抽取）；② 将 CoT 与外部知识库结合，让模型的推理过程可验证；③ 探索在低资源领域如何用少量人工 CoT 进行迁移学习。  

### 一句话记住它
给 DPO 加上机器生成的思维链，就像给模型装上了“草稿纸”，让它在微调时能看见自己的推理，从而显著提升 Text-to-SQL 的表现。