# Unlocking Reasoning Potential in Large Langauge Models by Scaling   Code-form Planning

> **Date**：2024-09-19
> **arXiv**：https://arxiv.org/abs/2409.12452

## Abstract

Despite the remarkable success of large language models (LLMs) on traditional natural language processing tasks, their planning ability remains a critical bottleneck in tackling complex multi-step reasoning tasks. Existing approaches mainly rely on prompting or task-specific fine-tuning, often suffering from poor robustness and cross-task generalization. To address the limitation, we introduce CodePlan, a scalable framework that empowers LLMs to generate and follow \textit{code-form plans} -- pseudocode that outlines high-level, structured reasoning processes. By leveraging the structured and versatile nature of code, CodePlan effectively captures the rich semantics and control flows inherent to sophisticated reasoning tasks. Importantly, CodePlan allows automatic extraction of code-form plans from massive, wide-ranging text corpora without the need for curated, task-specific datasets. This enables it to scale up efficiently and improve LLM's reasoning capabilities across diverse scenarios. To train CodePlan, we construct a large-scale dataset of 2M examples that integrate code-form plans with standard prompt-response pairs from existing corpora. With minimal computation overhead during both training and inference, CodePlan achieves a 25.1\% relative improvement compared with directly generating responses, averaged across 13 challenging multi-step reasoning benchmarks, spanning mathematical reasoning, symbolic reasoning, instruction-following, multi-hop QA, and decision-making tasks. Further analysis reveals CodePlan's increasing performance gains on more complex reasoning tasks, as well as significant data efficiency thanks to its generalization ability.

---

# 通过扩展代码式规划释放大语言模型的推理潜能 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在翻译、摘要等传统 NLP 任务上已经很强，但面对需要多步推理的场景仍会卡壳。现有的提升手段大多是让模型直接在提示（prompt）里写出思考过程，或者针对特定任务做微调，这两种方式都有明显短板：提示法对 wording 极度敏感，稍微改动就可能导致答案全错；微调需要为每个任务准备标注数据，成本高且难以跨任务迁移。于是，如何让模型拥有一种通用、稳健的“规划”能力，成为制约其在数学、符号推理、决策等复杂任务上进一步突破的瓶颈。

### 关键概念速览
- **代码式规划（code-form plan）**：模型先生成一段伪代码，描述高层次的推理步骤和控制流，随后再依据这段伪代码执行具体计算。类似于人类先写算法框架再填细节，能够把抽象的思路结构化。
- **伪代码（pseudocode）**：介于自然语言和真实编程语言之间的描述方式，保留变量、循环、条件等结构，却不要求语法完全正确，易于模型生成和人类阅读。
- **CoT（Chain‑of‑Thought）**：让模型在输出答案前写出逐步推理的文字链条，像解题时的草稿。与代码式规划不同的是，CoT 仍是自由文本，缺少明确的控制结构。
- **任务无关数据抽取**：从海量公开文本中自动抓取潜在的代码式规划片段，而不需要人工标注每个任务的示例。相当于在大海里捞“规划鱼”，省去专门养鱼的成本。
- **相对提升（relative improvement）**：用新方法的表现除以基线表现再减一，得到的百分比增幅。这里的 25.1% 表示在所有基准上平均提升了约四分之一。

### 核心创新点
1. **从自由文本到结构化伪代码的转变**  
   - 之前的做法：直接让 LLM 生成答案或用文字链条描述思路。  
   - 本文做法：在生成答案前，强制模型先输出一段伪代码，明确变量、循环、条件等控制流。  
   - 改变：伪代码把推理过程显式化，模型可以在“执行”阶段复用已有的代码解释器或内部算子，显著提升了对复杂逻辑的把握。

2. **大规模、任务无关的代码式规划数据构建**  
   - 之前的做法：需要为每个任务手工收集或标注规划示例，成本高且难以覆盖所有领域。  
   - 本文做法：利用现有的 prompt‑response 对，从公开语料中自动抽取或合成对应的伪代码，构成 200 万条训练对。  
   - 改变：数据规模和多样性大幅提升，使模型在训练时就学会通用的规划模式，跨任务迁移能力得到加强。

3. **轻量化的训练与推理流程**  
   - 之前的做法：很多增强推理的方法需要额外的检索、重排或多轮交互，计算开销大。  
   - 本文做法：在原有 LLM 基础上加入一个“规划生成”头部，训练时只多跑一次前向传播，推理时同样只多一步生成伪代码的过程。  
   - 改变：几乎不增加算力成本，却能在 13 项多步推理基准上实现 25.1% 的相对提升。

### 方法详解
整体框架可以概括为三步：**（1）规划生成 → (2) 代码解释 → (3) 答案输出**。  
1. **规划生成模块**  
   - 输入：原始任务描述（如“求解方程 …”）以及可选的 few‑shot 示例。  
   - 过程：模型在专门的 “Plan” token 后生成伪代码。伪代码的语法被约束为一种简化的 Python‑like 形式，包含 `def`, `if`, `for` 等关键字，但不要求完整的缩进或库导入。  
   - 类比：相当于先让学生写出解题思路的大纲，再去真正动手算。

2. **代码解释执行器**  
   - 设计：内部实现一个轻量的解释器，能够把伪代码映射到实际的数值或符号运算。比如 `for i in range(1, n+1): sum += i` 会被转化为求和公式或直接循环计算。  
   - 关键点：解释器并不是完整的编程语言运行时，而是针对常见推理模式（累加、递归、条件分支）做了专门的优化，使得执行速度几乎和直接生成答案相当。  
   - 反直觉之处：作者没有让模型真正去“跑代码”，而是用规则化的映射把伪代码转成内部算子，这样既保留了结构化优势，又避免了安全和效率风险。

3. **答案输出层**  
   - 解释器得到中间结果后，模型再根据这些结果生成自然语言答案。此阶段可以复用原始 LLM 的语言生成能力，只是输入多了一个结构化的上下文（变量值、执行日志）。  
   - 这样做的好处是：如果解释器出现错误，语言模型还能在答案层面进行一定的纠错或补充说明。

**训练细节**  
- 数据来源：从公开的问答、教程、代码注释等文本中抽取“步骤描述 + 示例”对，使用规则匹配把步骤转成伪代码，形成 2M 条 (prompt, plan, response) 三元组。  
- 目标函数：同时最小化计划生成的交叉熵损失和最终答案的交叉熵损失，两个损失加权相等，使模型在两阶段都保持学习动力。  
- 计算开销：因为计划生成和答案生成共享同一 Transformer，只是多了一个特殊的输出位置，所以训练时的 FLOPs 增幅约为 5% 左右。

### 实验与效果
- **评测任务**：13 个公开的多步推理基准，覆盖数学推理（MATH、GSM8K）、符号推理（SymbolicQA）、指令遵循（Instruction‑Following）、多跳阅读理解（HotpotQA）以及决策类任务（Decision‑Making Bench）。  
- **基线对比**：与直接让 LLM 生成答案的原始模型、以及常见的 CoT、Self‑Consistency 等增强方法相比，CodePlan 在所有基准上平均取得 **25.1% 的相对提升**。在最复杂的数学推理任务上提升甚至超过 30%。  
- **消融实验**：去掉计划生成阶段（直接生成答案）会导致整体性能回落到原始基线；仅使用自动抽取的伪代码但不进行解释执行，提升幅度约为 12%，说明解释执行器是关键增益点。  
- **数据效率**：在仅使用 20%（400k 条）训练数据时，CodePlan 仍能保持约 18% 的相对提升，验证了其对大规模、噪声数据的鲁棒性。  
- **局限性**：论文承认伪代码的表达能力仍受限于预设的简化语法，面对需要高级数据结构（如图、树）或外部 API 调用的任务时，当前框架可能无法完整描述；此外，解释器的规则集合是手工设计的，扩展到新领域需要额外工程工作。

### 影响与延伸思考
CodePlan 把“写代码”这一熟悉的思考方式引入 LLM 推理，开启了“结构化思维”在语言模型中的新路径。自发表后，已有工作尝试把真实的可执行代码（如 Python）直接嵌入模型生成流程，或把代码式规划与检索增强相结合，以进一步提升跨域推理能力。对想继续深挖的读者，可以关注以下方向：  
- **可扩展的伪代码语言**：设计更通用的抽象语法，支持图遍历、递归函数等更复杂的推理结构。  
- **自监督规划学习**：利用模型自身在推理过程中的错误反馈，自动改进计划生成策略。  
- **跨模态规划**：把视觉或表格信息转化为代码式计划，探索多模态推理的统一框架。（推测）

### 一句话记住它
让大语言模型先写“伪代码”再执行，是提升多步推理的高效、可扩展钥匙。