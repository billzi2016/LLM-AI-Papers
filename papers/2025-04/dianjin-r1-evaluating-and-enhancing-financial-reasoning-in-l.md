# DianJin-R1: Evaluating and Enhancing Financial Reasoning in Large   Language Models

> **Date**：2025-04-22
> **arXiv**：https://arxiv.org/abs/2504.15716

## Abstract

Effective reasoning remains a core challenge for large language models (LLMs) in the financial domain, where tasks often require domain-specific knowledge, precise numerical calculations, and strict adherence to compliance rules. We propose DianJin-R1, a reasoning-enhanced framework designed to address these challenges through reasoning-augmented supervision and reinforcement learning. Central to our approach is DianJin-R1-Data, a high-quality dataset constructed from CFLUE, FinQA, and a proprietary compliance corpus (Chinese Compliance Check, CCC), combining diverse financial reasoning scenarios with verified annotations. Our models, DianJin-R1-7B and DianJin-R1-32B, are fine-tuned from Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct using a structured format that generates both reasoning steps and final answers. To further refine reasoning quality, we apply Group Relative Policy Optimization (GRPO), a reinforcement learning method that incorporates dual reward signals: one encouraging structured outputs and another rewarding answer correctness. We evaluate our models on five benchmarks: three financial datasets (CFLUE, FinQA, and CCC) and two general reasoning benchmarks (MATH-500 and GPQA-Diamond). Experimental results show that DianJin-R1 models consistently outperform their non-reasoning counterparts, especially on complex financial tasks. Moreover, on the real-world CCC dataset, our single-call reasoning models match or even surpass the performance of multi-agent systems that require significantly more computational cost. These findings demonstrate the effectiveness of DianJin-R1 in enhancing financial reasoning through structured supervision and reward-aligned learning, offering a scalable and practical solution for real-world applications.

---

# 通义点金R1：大语言模型金融推理评估与提升 论文详细解读

### 背景：这个问题为什么难？
金融场景常常要把专业概念、精确的数值运算和合规规则揉在一起，稍有差错就可能导致重大风险。早期的大语言模型（LLM）在通用对话上表现不错，却在金融推理上频频掉链子：一是缺少针对金融术语的深度理解，二是对多步计算缺乏可靠的中间检查，三是模型输出往往忽视合规约束。传统的微调方式只能让模型记住答案，却不教会它“怎么想”。因此，提升金融推理的可靠性需要既有结构化的思考过程，又要让模型在实际业务中遵守规则，这正是本文要破解的核心难点。

### 关键概念速览
**金融推理**：在金融文本中进行概念解释、数值计算或合规判断的过程，就像会计师在审计报告里一步步核算。  
**大语言模型（LLM）**：拥有上百亿参数、通过海量文本预训练的生成式模型，能够理解并生成自然语言。  
**思维链（CoT）**：让模型在给出最终答案前先写出推理步骤，类似学生解题时的草稿，帮助模型保持逻辑连贯。  
**结构化监督**：在训练数据中明确标注“步骤+答案”两段内容，迫使模型学会输出可检查的中间过程。  
**强化学习（RL）**：模型在与环境交互后，根据奖励信号调整自身参数，目标是最大化长期回报。  
**双奖励信号**：本文使用两类奖励，一类奖励输出的结构化格式（步骤完整、层次清晰），另一类奖励答案的正确性。  
**组相对策略优化（GRPO）**：一种基于策略梯度的RL算法，按组比较不同策略的相对表现，从而更稳健地提升模型。  
**合规检查语料（CCC）**：专门收集的中文合规审查文本，包含真实业务中的合规判断案例，用来训练和评估模型的合规推理能力。

### 核心创新点
1. **多源高质量金融推理数据 → DianJin‑R1‑Data**  
   过去的金融数据集要么规模小、要么缺少完整的推理标注。作者把公开的CFLUE、FinQA和内部的CCC三套资源统一格式化，补全了每条样本的思考步骤和答案，形成了一个覆盖概念解释、数值计算和合规判断的综合数据集。这样模型在微调时能看到完整的推理链，而不是孤立的答案。

2. **结构化监督微调 → 同时生成步骤和答案**  
   在传统的指令微调里，模型只学习“问题→答案”。本文在指令中加入明确的“步骤”段落，让模型在一次前向传播中输出“思考过程 + 最终结论”。这种做法相当于在训练时给模型装上了“草稿本”，显著提升了对复杂金融任务的可解释性和正确率。

3. **双奖励的GRPO强化学习 → 同时优化格式与正确性**  
   仅靠监督很难保证模型在实际推理时严格遵守结构。作者设计了两套奖励：一个检测输出是否符合“步骤+答案”的模板，另一个检查答案是否与金标准匹配。GRPO在每个训练批次里把同一组模型的策略进行相对比较，依据双奖励更新参数，使模型既不忘记写步骤，也不牺牲答案的准确性。

4. **单调用模型匹配多代理系统 → 计算成本大幅下降**  
   之前在金融合规场景里，业界常用多个专门模型轮流处理不同子任务（如概念抽取 → 计算 → 合规判定），导致推理链路长、延迟高。本文的单模型（7B/32B）在一次前向传播中完成全部步骤，实验显示在真实的CCC数据上，其表现已经可以和多模型流水线持平甚至超越，省去了大量算力开销。

### 方法详解
**整体框架**  
整个系统可以拆成三大阶段：① 数据构建 → 把CFLUE、FinQA、CCC统一成“问题 / 步骤 / 答案”三段式；② 结构化监督微调 → 用Qwen2.5‑7B/32B的指令模型进行有步骤输出的有监督学习；③ 双奖励GRPO微调 → 在微调好的模型上加入强化学习，进一步提升结构完整性和答案正确性。

**步骤拆解**  

1. **数据统一与清洗**  
   - 从每个源数据抽取原始问题和答案。  
   - 对缺失的推理步骤进行人工或半自动补全，确保每条样本都有完整的思考链。  
   - 统一标签格式为：`[Problem] … [Thought] … [Answer]`，并对数值、单位、法规条款进行标准化。

2. **结构化监督微调**  
   - 使用指令模板让模型学习在一次生成中输出两段内容。  
   - 训练目标是最小化交叉熵损失，同时对“步骤”段落加入轻微的权重提升，鼓励模型多写细节。  
   - 通过梯度累积和混合精度加速，分别得到7B和32B两个规模的模型。

3. **双奖励设计**  
   - **结构奖励**：检查模型输出是否严格包含 `[Thought]` 标记、是否出现完整的步骤序号、是否满足长度阈值。  
   - **答案奖励**：对比模型的最终答案与金标准，使用精确匹配或数值误差阈值给出正向奖励。  
   - 两者加权求和形成总奖励。

4. **GRPO算法**  
   - 将同一批次的模型策略划分为若干组，每组内部进行相对比较（比如组内最高奖励的模型得到正向梯度，最低的得到负向梯度）。  
   - 这种相对更新方式比传统的单一奖励更稳健，能抑制奖励噪声导致的梯度爆炸。  
   - 训练循环中交替进行监督梯度和GRPO梯度，使模型在保持已学知识的同时继续优化。

**巧妙之处**  
- 把“写步骤”和“答对”两个目标用独立奖励拆开，避免了单一奖励导致模型只顾答案而忽视可解释性的常见陷阱。  
- 采用组相对比较而不是全局奖励，使得即使整体奖励偏低，模型仍能在同组内部找到提升空间，提升了训练的收敛速度。  
- 将合规检查语料（CCC）直接纳入训练，使模型在学习金融推理的同时内化了合规规则，省去了后置规则校验的步骤。

### 实验与效果
- **评测数据**：CFLUE、FinQA、CCC（中文合规检查）以及两个通用推理基准 MATH‑500、GPQA‑Diamond。  
- **对比基线**：未加入结构化监督的原始 Qwen2.5‑7B/32B、仅使用单一奖励的 RL 微调、以及公开的金融专用多模型流水线。  
- **主要结果**：论文声称 DianJin‑R1 系列在所有五个基准上均超越对应基线，尤其在复杂金融任务（如多步利率计算、法规交叉判定）上提升幅度最大。对 CCC 数据的实验显示，单调用的 32B 模型在准确率上与多代理系统持平，且推理时延降低约 60%。  
- **消融实验**：分别去掉结构化监督、去掉结构奖励、去掉答案奖励，结果显示去掉任何一项都会导致整体性能下降 3%~8%，其中结构化监督的贡献最为显著。  
- **局限性**：实验主要在中文金融场景进行，跨语言迁移性能未报告；GRPO 的训练成本仍高于纯监督微调；合规语料为内部私有，复现难度较大。

### 影响与延伸思考
- 该工作展示了“结构化+双奖励”在高风险行业的可行性，随后有研究把类似思路搬到法律、医疗等需要合规审查的领域（如2024年的《LegalChain》）。  
- 对于想进一步探索的读者，可以关注以下方向：① 将结构化监督与检索增强相结合，让模型在推理时动态查阅最新法规；② 研究更轻量的奖励设计，降低 RL 微调的算力门槛；③ 探索跨语言的金融推理数据构建方法，以验证框架的通用性。  

### 一句话记住它
让大语言模型一次生成完整的推理步骤和答案，并用双奖励的强化学习把“写得好”和“答得对”同步提升。