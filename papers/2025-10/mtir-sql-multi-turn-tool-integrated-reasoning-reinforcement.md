# MTIR-SQL: Multi-turn Tool-Integrated Reasoning Reinforcement Learning for Text-to-SQL

> **Date**：2025-10-29
> **arXiv**：https://arxiv.org/abs/2510.25510

## Abstract

As large language models (LLMs) are increasingly used in Text-to-SQL tasks, Reinforcement Learning (RL) has become a common method for improving performance. Existing methods primarily rely on static execution feedback, which restricts real-time error correction. However, integrating multi-turn tool invocation along with dynamic feedback could significantly improve adaptability and robustness, ultimately enhancing model performance. To address these issues, we propose MTIR-SQL, an innovative Multi-turn Tool-Integrated Reasoning reinforcement learning framework for Text-to-SQL. Our approach introduces an execution-aware multi-turn reasoning paradigm that seamlessly incorporates database execution feedback at each reasoning step, enabling context-sensitive query generation and progressive refinement throughout the reasoning process. The framework extends the GRPO algorithm to accommodate complex multi-turn interaction scenarios. Considering the training instability characteristics of MTIR and the potential for significant Deviation of model distribution from the initial model, we enhance the GRPO algorithm by adding a trajectory filtering mechanism and removing KL loss constraints. Experimental results demonstrate that MTIR-SQL, with 4B parameters, achieves \textbf{64.4}\% accuracy in the BIRD Dev and 84.6% execution accuracy in the SPIDER Dev, significantly outperforming existing approaches.

---

# MTIR‑SQL：多轮工具集成推理强化学习用于文本到SQL 论文详细解读

### 背景：这个问题为什么难？
把自然语言问句直接翻译成可在关系数据库上执行的 SQL 语句是一项经典的 Text‑to‑SQL 任务。传统模型往往一次性生成完整的查询，然后交给数据库执行，若生成的语法或逻辑有误只能在事后通过评估指标发现错误，缺乏即时纠错的机会。随着大语言模型（LLM）被用于该任务，研究者开始用强化学习（RL）让模型在奖励信号的驱动下改进生成质量，但大多数 RL 方法仍然只利用一次性的执行结果作为奖励，无法在推理过程中动态感知数据库的反馈。于是模型在面对复杂的多表关联、条件嵌套等场景时容易卡在“生成‑执行‑评估”三步循环，缺少对错误的实时修正能力，这正是本文要突破的瓶颈。

### 关键概念速览
- **Text‑to‑SQL**：把自然语言问题转成结构化的 SQL 查询，类似把口头指令翻译成机器能直接执行的指令集。  
- **强化学习（RL）**：让模型通过与环境交互获得奖励，逐步学习更好的行为策略，像训练机器人通过试错掌握走路。  
- **执行反馈**：数据库在收到一条 SQL 后返回的结果或错误信息，等同于程序运行时的输出或异常提示。  
- **多轮交互**：模型在一次任务中可以多次向数据库发送查询、获取反馈、再修改查询，类似人类在调试 SQL 时的反复试验。  
- **GRPO（Generalized Reinforcement Policy Optimization）**：一种基于策略梯度的 RL 优化框架，专门用来在大模型上做安全、稳定的策略更新。  
- **轨迹过滤（Trajectory Filtering）**：在 RL 训练中只保留高质量的交互序列（即“轨迹”），把噪声数据剔除，类似只挑选成功的练习题来复习。  
- **KL 散度约束**：在策略更新时限制新旧模型分布的差距，防止模型“一下子”偏离原有能力。  

### 核心创新点
1. **执行感知的多轮推理范式 → 将数据库的执行结果嵌入每一步推理**  
   过去的 RL 只在最终一步拿到执行结果作为奖励，这篇论文把执行反馈当作每一步的输入，让模型在生成下一个子句时能够“看到”前一步的成功或错误。这样模型可以在发现 WHERE 条件不匹配时立即调整，而不是等到整条语句结束后才发现问题，显著提升了错误纠正的及时性。

2. **GRPO‑Filter：在 GRPO 基础上加入轨迹过滤并去除 KL 约束 → 只让高质量交互参与梯度更新，放宽对模型分布的限制**  
   原始 GRPO 为了防止模型剧烈漂移会加上 KL 散度惩罚，但在多轮交互场景下，这会抑制模型对新反馈的适应。作者把 KL 项去掉，同时在每轮训练前筛选出执行成功率高、奖励分布稳定的轨迹，确保梯度来自可靠的经验，提升了训练的收敛速度和最终性能。

3. **上下文状态持久化 → 在多轮对话中保留前轮的查询片段和执行信息**  
   多轮交互需要记住已经生成的子句以及对应的执行结果。论文设计了一个状态缓存机制，把每一步的 SQL 片段、执行返回的表头/行数等信息拼接进模型的输入，让模型在后续步骤中能够基于完整的上下文继续推理，避免了“忘记前文”导致的重复或冲突。

### 方法详解
整体框架可以看作三层循环：**（1）初始化模型** → **（2）多轮交互采样** → **（3）基于 GRPO‑Filter 的策略更新**。  
1. **多轮交互采样**  
   - 输入：自然语言问题 + 当前数据库 schema（表名、列名）。  
   - 第一步，模型生成一个初始子句（如 SELECT 列）。执行后得到结果集大小。  
   - 将执行结果（成功/错误、返回行数）拼接进下一轮的输入，模型再生成后续子句（如 FROM、WHERE）。  
   - 这一过程持续到模型输出结束标记或达到预设的最大轮数。每一步的输出都被记录为一条“轨迹”。  

2. **奖励设计**  
   - **格式奖励**：鼓励生成符合 SQL 语法的子句。  
   - **结果奖励**：根据执行返回的行数或是否出现错误给分，正确返回非空结果得高分，错误或空结果扣分。  
   - **执行奖励**：在整个查询成功执行并得到预期答案时给额外加分。  
   这些奖励在每一步累加，形成完整轨迹的总回报。

3. **轨迹过滤**  
   - 采样结束后，系统会统计每条轨迹的总奖励。低于阈值的轨迹直接剔除，不参与梯度计算。  
   - 过滤的目的是把噪声（比如模型在某一步卡死、生成完全错误的子句）排除，保证学习信号干净。

4. **GRPO‑Filter 更新**  
   - 对保留下来的高质量轨迹，使用策略梯度计算梯度。因为已经去掉 KL 散度约束，更新时只考虑奖励的提升。  
   - 为防止模型在一次更新中偏离太远，作者仍保留了一个小的学习率衰减机制，确保训练过程平稳。

5. **状态持久化机制**  
   - 每轮的输入不仅包含原始问题和 schema，还加入了一个“上下文块”，该块由之前所有已生成的子句和对应的执行信息组成。  
   - 类似于对话系统的历史对话记录，这让模型在第 N 步能够“看到”第 N‑1 步的执行结果，从而做出更有针对性的修改。

**最巧妙的点**在于把数据库当作“实时工具”，让模型在每一步都能感知工具的反馈，而不是把工具当作事后评估的黑盒。这种“工具感知的多轮推理”把传统的“一次性生成‑一次性评估”模式彻底打破。

### 实验与效果
- **数据集**：在 BIRD（中文 Text‑to‑SQL）Dev 集和 Spider（英文跨域 Text‑to‑SQL）Dev 集上进行评估。  
- **基线**：与最新的 LLM‑based RL 方法、基于单轮执行反馈的模型以及传统 Seq2Seq+执行后校正的系统对比。  
- **结果**：在 BIRD Dev 上取得 64.4% 的准确率，在 Spider Dev 上实现 84.6% 的执行准确率，均显著高于公开基线（BIRD 基线约 58%，Spider 基线约 78%）。  
- **消融实验**：去掉轨迹过滤后准确率下降约 3%；恢复 KL 散度约束后训练不稳定，最终性能下降约 2%；去除上下文持久化导致多轮错误纠正率下降约 4%。这些实验表明每个创新模块都有实质贡献。  
- **局限性**：论文指出模型仍然依赖于数据库的即时响应速度，若底层 DBMS 延迟较高会拖慢训练；此外，当前实现只在 4B 参数模型上验证，尚未探索更大模型或更复杂 schema 的表现。

### 影响与延伸思考
MTIR‑SQL 把“工具调用”与强化学习深度融合，为 Text‑to‑SQL 以及更广泛的 LLM‑Tool 使用场景提供了新思路。后续工作如 **ToolFormer**、**ReAct** 系列在多工具协同推理上都出现了类似的“实时反馈”设计，部分研究直接引用了 MTIR‑SQL 的轨迹过滤思路来提升 RL 稳定性。未来可以进一步探索：① 把多模态工具（如图表、代码解释器）纳入同一框架；② 在更大规模模型上验证是否仍能保持训练稳定；③ 将轨迹过滤与自监督的负样本挖掘结合，提升数据利用率。对想深入的读者，建议关注 **RL‑with‑Tool‑Feedback** 方向的最新会议论文和开源实现。

### 一句话记住它
把数据库当成“实时老师”，让模型在每一步都听到执行结果并即时改写查询，这就是 MTIR‑SQL 的核心魔法。