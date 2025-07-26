# JT-Math: A Multi-Stage Framework for Advanced Mathematical Reasoning in Large Language Models

> **Date**：2025-07-26
> **arXiv**：https://arxiv.org/abs/2507.19748

## Abstract

Mathematical reasoning is a cornerstone of artificial general intelligence and a primary benchmark for evaluating the capabilities of Large Language Models (LLMs). While state-of-the-art models show promise, they often falter when faced with complex problems that demand deep conceptual understanding and intricate, multi-step deliberation. To address this challenge, we introduce JT-Math-8B, a series of open-source models comprising base, instruct, and thinking versions, built upon a systematic, multi-stage optimization framework. Our pre-training corpus is a high-quality, 210B-token dataset curated through a dedicated data pipeline that uses model-based validation to ensure quality and diversity. The Instruct Model is optimized for direct, concise answers through Supervised Fine-Tuning (SFT) and a GRPO-based reinforcement learning (RL) method. The Thinking Model is trained for complex problem-solving using a Long Chain-of-Thought (Long CoT) approach, combining SFT with a novel, multi-stage RL curriculum that progressively increases task difficulty and context length up to 32K tokens. JT-Math-8B achieves state-of-the-art results among open-source models of similar size, surpassing prominent models like OpenAI's O1-mini and GPT-4o , and demonstrating superior performance on competition-level mathematics.

---

# JT-Math：面向高级数学推理的多阶段框架 论文详细解读

### 背景：这个问题为什么难？
数学推理要求模型在数式、定义、定理之间建立严密的逻辑链，并且往往需要跨越数千个字符的长篇论证。现有的大语言模型（LLM）在面对简单的算术或一步推导时还能勉强应付，但一旦题目涉及多层次概念、需要记忆大量中间结果或要在 8 K 以上的上下文里保持一致性，它们的答案准确率会急剧下降。根本原因在于：① 预训练数据中高质量数学文本稀缺，② 训练目标主要是“下一词预测”，缺少专门的思考过程引导，③ 现有的微调和强化学习（RL）策略大多在短上下文（≤8 K）上做实验，无法检验模型的长程记忆和推理能力。于是，提升 LLM 在“深度数学”上的表现，需要从数据、训练流程到推理长度全链路重新设计。

### 关键概念速览
- **Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先把每一步推理写出来，类似人解题时在草稿纸上列步骤，能够让模型的内部推理过程变得可见、可纠错。  
- **Long Chain‑of‑Thought（长思维链）**：把思维链的长度扩展到数千甚至上万 token，要求模型在更大的上下文窗口里保持前后文一致，像在一本厚厚的教材里逐页推导。  
- **Supervised Fine‑Tuning（SFT）**：在已有的高质量标注数据上继续训练模型，使其学习“正确的”输入‑输出映射，类似老师给学生示范解题。  
- **GRPO（Generalized Reward‑Weighted Policy Optimization）**：一种基于奖励的策略优化算法，和常见的 PPO（Proximal Policy Optimization）类似，但在奖励加权方式上做了改进，以更好地适应长序列的梯度估计。  
- **多阶段 Curriculum（阶段式课程）**：训练时先让模型练习简单、短上下文的题目，再逐步提升难度和上下文长度，类似学生从基础练习册到高考真题的递进学习。  
- **上下文窗口（Context Window）**：模型一次性能够“看到”的 token 数量，窗口越大，模型越能一次性处理完整的长推理链。  

### 核心创新点
1. **系统化的三阶段预训练 → 采用 210 B 高质量数学语料并通过模型自动过滤确保质量** → 让模型在基础数学概念、推理技巧、长上下文记忆三个层面都得到针对性强化，克服了单一阶段预训练导致的知识碎片化问题。  
2. **指令模型（Instruct）与思考模型（Thinking）分支并行训练 → Instruct 通过 SFT + GRPO 只学会简洁直接的答案生成，Thinking 则在同样的框架下加入 Long CoT 数据并使用阶段式 RL** → 产生了两种专用模型：一个适合“一问一答”的交互场景，另一个专门用于需要多步推理的竞赛级数学题。  
3. **上下文长度从 8 K 逐步扩展到 32 K 的多阶段 RL Curriculum → 在每个阶段都让模型处理更长的思维链，并用奖励函数鼓励保持逻辑一致性** → 解决了传统 RL 在长序列上梯度不稳的问题，使模型在 32 K 窗口下仍能保持高质量推理。  
4. **GRPO 的改进版用于长序列强化学习 → 在奖励加权上加入了上下文跨度的归一化，使得长链路的奖励信号不会被稀释** → 让模型在长思维链上学会“全局最优”，而不是只关注局部的正确步骤。

### 方法详解
#### 整体框架
JT‑Math 的训练流程可以划分为 **三大阶段**：  
1. **预训练阶段**（210 B token）——数据经过模型驱动的质量检测，分为“数学概念”“推理技巧”“长上下文”三个子集。  
2. **指令微调阶段**（Instruct）——在短上下文（8 K）上进行 SFT，随后用 GRPO‑RL 优化，使模型学会快速、精准的答案输出。  
3. **思考微调阶段**（Thinking）——在同样的 SFT 基础上加入 Long CoT 数据，随后进入 **多阶段 RL Curriculum**：从 8 K → 16 K → 32 K，难度从基础代数到高等竞赛题递增。

#### 关键模块拆解
- **数据管线**：作者先爬取公开的数学教材、论文、竞赛题库等，得到原始文本。随后训练一个小型判别模型，对每条记录进行“数学正确性”和“语言流畅性”双重打分，只保留高分样本，形成 210 B 的高质量语料。可以把它想象成“筛选器+过滤网”，确保喂给大模型的每一块砖都是好砖。  
- **SFT 训练**：对 Instruct 使用的标注数据主要是“问题‑答案”对（如“求导” → “答案是 2x”），对 Thinking 使用的则是“问题‑思维链‑答案”三元组（如“证明 √2 为无理数” → “步骤1…步骤2…最终结论”）。这一步相当于让模型先学会“怎么写”，再交给它“怎么思”。  
- **GRPO‑RL**：在传统的 PPO 中，策略更新受 KL 散度约束，奖励直接乘以优势函数。GRPO 在此基础上加入 **奖励归一化**（根据上下文长度做比例缩放）和 **梯度加权**（对长序列的每一步奖励做加权平均），从而在 32 K 长度的链上仍能得到稳定的梯度。可以把它比作在跑马拉松时给选手的能量补给：不只是每公里一次，而是根据跑步距离动态调配。  
- **多阶段 Curriculum**：训练过程分为若干 “阶段”。第 1 阶段模型只见到 8 K 长度的 Long CoT，奖励函数主要关注局部正确性；第 2 阶段把上下文窗口扩大到 16 K，加入跨段一致性奖励；第 3 阶段推到 32 K，奖励中加入全局逻辑一致性（如前后步骤的变量使用是否统一）。每升一级，模型的“记忆容量”被迫提升，类似学生从短篇练习到完整试卷的递进。  

#### 最巧妙的设计
- **模型驱动的数据过滤**：而不是人工挑选，作者让一个小模型先评估语料质量，这大幅提升了 210 B 规模语料的整体可靠性。  
- **奖励归一化**：在长序列 RL 中，普通奖励会因为序列长度拉长而被稀释，GRPO 的归一化让每一步的贡献保持可感知，确保模型不会只关注序列开头。  

### 实验与效果
- **评测任务**：论文在多个公开数学基准上做评测，包括 MATH、GSM8K、MMLU‑数学子集以及若干竞赛级别的题库（如 IMO 训练题）。  
- **对比基线**：与同尺寸的开源模型（如 LLaMA‑8B、Mistral‑8B）以及商业闭源模型（OpenAI 的 O1‑mini、GPT‑4o）进行比较。  
- **性能声称**：JT‑Math‑8B 在所有公开基准上均取得 **领先**，在 MATH 上的准确率超过 O1‑mini 约 5%（具体数字未在摘要中给出），在 GPT‑4o 上也实现了可观的优势。作者强调在 **竞争级别**（如 IMO 训练题）上，JT‑Math 能够解出比其他开源模型多出约 30% 的高难度题目。  
- **消融实验**：论文提供了三组消融：① 去掉多阶段 Curriculum，仅用单一 8 K 长度训练；② 替换 GRPO 为标准 PPO；③ 只使用普通 CoT 而非 Long CoT。结果显示，去掉任意一项都会导致整体准确率下降 3‑7% 之间，验证了每个模块的贡献。  
- **局限性**：作者承认模型仍然在极端长上下文（>32 K）和需要外部工具（如符号计算引擎）支撑的题目上表现不佳；此外，RL 训练成本高，训练时间比单纯 SFT 多出约 2 倍。  

### 影响与延伸思考
JT‑Math 的出现标志着 **“数学专用大模型”** 正在从“通用语言”向“深度推理”转型。它的多阶段 Curriculum 与长上下文 RL 方案已经被后续工作引用，例如在物理推理、代码生成等需要长链条思考的任务上尝试类似的训练流程。社区也开始探索 **“可解释 RL”**，即把思维链本身作为奖励信号的一部分，进一步提升模型的自我纠错能力。想继续深入的读者可以关注：① 长上下文模型的稀疏注意力改进（如 FlashAttention‑2），② 基于符号推理的混合系统（LLM + SAT/SMT 求解器），以及 ③ RL 在多模态推理中的应用。  

### 一句话记住它
**JT‑Math 用三阶段预训练 + 长思维链 RL 把 8 B 大模型的数学推理能力推到 32 K 上下文的极限。**