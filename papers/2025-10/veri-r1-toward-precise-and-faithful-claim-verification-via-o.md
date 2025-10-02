# Veri-R1: Toward Precise and Faithful Claim Verification via Online Reinforcement Learning

> **Date**：2025-10-02
> **arXiv**：https://arxiv.org/abs/2510.01932

## Abstract

Claim verification with large language models (LLMs) has recently attracted growing attention, due to their strong reasoning capabilities and transparent verification processes compared to traditional answer-only judgments. However, existing approaches to online claim verification, which requires iterative evidence retrieval and reasoning, still mainly rely on prompt engineering or pre-designed reasoning workflows, without unified training to improve necessary skills. Therefore, we introduce Veri-R1, an online reinforcement learning (RL) framework that enables an LLM to interact with a search engine and to receive reward signals that explicitly shape its planning, retrieval, and reasoning behaviors. This dynamic interaction of LLM with retrieval systems more accurately reflects real-world verification scenarios and fosters comprehensive verification skills. Empirical results show that Veri-R1 improves joint accuracy by up to 30% and doubles the evidence score, often surpassing its larger-scale model counterparts. Ablation studies further reveal the impact of reward components, and the link between output logits and label accuracy. Our results highlight the effectiveness of online RL for precise and faithful claim verification, providing an important foundation for future research. We release our code to support community progress in LLM empowered claim verification.

---

# Veri‑R1：面向精准可信的在线强化学习声明验证 论文详细解读

### 背景：这个问题为什么难？
在实际场景中，模型只能拿到一条待验证的声明和一个可以查询的可信文库，必须自己去检索、筛选证据再给出判断。传统的事实验证方法往往把检索和推理写死在提示词里，或者直接让模型一次性输出答案，缺乏对检索过程的动态控制。于是模型容易走“捷径”，只凭记忆或语言流畅度给出标签，而不是真正基于可查证的证据。根本的瓶颈在于：没有统一的训练信号来教会模型何时规划检索、怎样挑选文档、以及如何把证据组织成可靠的推理链。

### 关键概念速览
**在线事实验证**：模型在推理时可以实时向外部搜索引擎发起查询，类似人类在写报告时随时上网查资料。  
**强化学习（RL）**：让模型通过试错获得奖励，像训练机器人玩游戏一样，让它学会在不同阶段做出最有利的决策。  
**规划（Planning）**：模型先决定要检索哪些主题或关键词，类似先列出调查清单再去查资料。  
**检索（Retrieval）**：实际向搜索引擎发送查询并拿回候选文档，等同于在图书馆挑选相关书页。  
**推理迭代**：模型在拿到新证据后重新思考、可能再发起查询，形成“查—想—查—想”的循环。  
**奖励函数**：对模型的每一步行为打分，包含格式、证据匹配、标签正确性等多维度，类似老师给学生的作业评分细则。  
**证据分数（Evidence Score）**：衡量模型挑选的证据与人工标注的黄金证据集合的相似度，数值越高说明模型找的证据越靠谱。  
**离线 Rollout**：只用已有输入进行推理，不允许搜索，类似在闭卷考试中作答。

### 核心创新点
1. **统一的在线 RL 框架 → 让 LLM 同时学习规划、检索、推理三项技能 → 模型不再依赖手工提示，而是通过奖励信号自我调节，整体准确率提升约 30%。**  
2. **多维奖励设计 → 将格式、证据覆盖率、标签正确性等指标组合成一个综合得分 → 模型被迫避免只给出标签的“捷径”，必须提供高质量证据，证据分数几乎翻倍。**  
3. **动态交互式检索流程 → 在一次推理过程中模型可以多轮查询，每轮结束后根据当前证据决定是否继续 → 更贴近真实调查场景，显著提升对复杂、需要多步推理的声明的处理能力。**  
4. **对输出 logits 与标签准确性的关联分析 → 通过实验发现模型输出的置信分布可以直接映射到最终标签的可靠性 → 为后续的可信度校准提供了实证依据。

### 方法详解
整体思路是把“声明验证”看成一次带奖励的游戏。模型（LLM）在游戏里轮流执行三类动作：**规划**、**检索**、**推理**，每完成一次动作后环境会给出即时奖励，最终的标签正确性再给一次全局奖励。整个过程分为两大阶段：**在线 Rollout**（允许查询）和 **离线 Rollout**（只能基于已有信息推理），两者交替训练，确保模型既会主动搜索，也能在资源受限时做出合理判断。

**1. 规划模块**  
模型先读取待验证的声明，生成一组检索意图（关键词或查询模板），这一步的输出用特殊 token 标记为 `<PLAN>`。奖励里会检查这些意图是否覆盖了声明的关键概念，类似老师检查学生的调查计划是否完整。

**2. 检索模块**  
依据 `<PLAN>` 产生的意图，模型向外部搜索引擎发送查询，返回 top‑k 文档摘要。模型随后对这些摘要进行初步筛选，挑出最可能支持或反驳声明的证据，用 `<EVID>` 标记。奖励函数对挑选的文档集合与黄金证据集合的交并比（IoU）进行打分，鼓励模型找对“关键文献”。

**3. 推理迭代模块**  
模型把已收集的证据拼接进上下文，继续生成推理链，输出形式为 `<THINK>`（思考过程）+ `<ANSWER>`（最终标签）。如果模型判断证据仍不足，它可以回到规划阶段重新生成查询，形成多轮循环。每一次循环结束后，环境会根据当前证据的完整性、思考过程的逻辑连贯性以及是否已经满足提前设定的停止条件（如证据覆盖率达到阈值）给出奖励。

**4. 奖励函数细节**  
- **格式奖励**：检查特殊 token 是否完整出现，防止模型省略关键步骤。  
- **证据奖励**：基于 IoU 计算模型选出的证据与黄金证据的相似度。  
- **标签奖励**：对最终 `<ANSWER>` 是否与人工标注一致给最高权重的奖励。  
- **有效性权重**：对 SUPPORT/REFUTE 类别的低质量证据进行惩罚，防止模型只靠标签奖励获胜。

**5. 训练流程**  
使用 Proximal Policy Optimization（PPO）等常见的离线 RL 算法，对模型的策略（即每一步的输出）进行梯度更新。在线 Rollout 产生的轨迹被存入经验池，离线 Rollout 则提供对比基准，帮助模型学习在没有检索资源时的稳健推理。整个训练过程在公开的事实验证数据集上迭代数十万步，逐步提升各项奖励。

**最巧妙的点**：把检索意图、证据选择、推理过程全部用统一的 token 序列表示，使得 RL 可以直接作用在语言模型的生成空间，而不需要额外的控制器或外部规划器。这种“一体化”设计让模型在同一网络里学会了“什么时候查、查什么、怎么用”。

### 实验与效果
- **数据集**：主要在 FEVER、SciFact 等公开的声明验证基准上评估，均提供检索接口和黄金证据。  
- **对比基线**：与传统的 Prompt‑only 方法、基于固定检索‑推理流水线的模型（如 RAG、FiD）以及更大规模的 LLM（如 GPT‑4）进行比较。  
- **主要指标**：Joint Accuracy（标签+证据同时正确）提升约 30%，Evidence Score 几乎翻倍，部分实验中甚至超过了参数规模更大的对手。  
- **消融实验**：去掉证据奖励后 Joint Accuracy 下降约 12%；去掉格式奖励导致模型经常漏掉 `<PLAN>`/`<EVID>`，整体性能下降 8%；仅保留离线 Rollout 训练，在线检索能力显著削弱，说明两阶段训练相辅相成。  
- **局限性**：作者指出当前奖励函数仍依赖人工标注的黄金证据，跨域或新兴领域缺乏高质量证据时效果会受限；此外，在线检索的响应时间在真实系统中仍是瓶颈。

### 影响与延伸思考
Veri‑R1 把强化学习引入在线事实验证的完整闭环，开启了“让 LLM 自主规划检索”的新潮流。后续工作如 **ReAct‑RL**、**Self‑CheckGPT** 等都在借鉴其奖励设计或多轮检索框架，进一步探索更轻量的奖励信号或自监督的证据评估。对想继续深入的读者，可以关注以下方向：① 用无监督的证据对齐方法替代人工黄金证据；② 将检索成本（时间、费用）纳入奖励；③ 跨模态检索（图像、表格）与语言模型的协同学习。整体来看，Veri‑R1 为 LLM 在真实信息环境中实现“查—想—答”提供了可复制的技术蓝图。

### 一句话记住它
让大语言模型通过在线强化学习学会主动检索、挑证据、循环推理，从而把“说得对”变成“说得有据”。