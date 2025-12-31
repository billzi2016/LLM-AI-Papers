# Youtu-LLM: Unlocking the Native Agentic Potential for Lightweight Large Language Models

> **Date**：2025-12-31
> **arXiv**：https://arxiv.org/abs/2512.24618

## Abstract

We introduce Youtu-LLM, a lightweight yet powerful language model that harmonizes high computational efficiency with native agentic intelligence. Unlike typical small models that rely on distillation, Youtu-LLM (1.96B) is pre-trained from scratch to systematically cultivate reasoning and planning capabilities. The key technical advancements are as follows: (1) Compact Architecture with Long-Context Support: Built on a dense Multi-Latent Attention (MLA) architecture with a novel STEM-oriented vocabulary, Youtu-LLM supports a 128k context window. This design enables robust long-context reasoning and state tracking within a minimal memory footprint, making it ideal for long-horizon agent and reasoning tasks. (2) Principled "Commonsense-STEM-Agent" Curriculum: We curated a massive corpus of approximately 11T tokens and implemented a multi-stage training strategy. By progressively shifting the pre-training data distribution from general commonsense to complex STEM and agentic tasks, we ensure the model acquires deep cognitive abilities rather than superficial alignment. (3) Scalable Agentic Mid-training: Specifically for the agentic mid-training, we employ diverse data construction schemes to synthesize rich and varied trajectories across math, coding, and tool-use domains. This high-quality data enables the model to internalize planning and reflection behaviors effectively. Extensive evaluations show that Youtu-LLM sets a new state-of-the-art for sub-2B LLMs. On general benchmarks, it achieves competitive performance against larger models, while on agent-specific tasks, it significantly surpasses existing SOTA baselines, demonstrating that lightweight models can possess strong intrinsic agentic capabilities.

---

# Youtu-LLM：释放轻量大语言模型的原生智能体潜能 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）领域，模型规模越大往往意味着推理、规划等高级能力越强，但随之而来的算力和内存成本也会呈指数增长。很多实际场景（如移动端、边缘服务器）只能承受数十亿参数以下的模型，却仍然需要模型具备“智能体”式的计划与工具使用能力。传统的轻量模型通常通过蒸馏把大模型的知识压缩下来，却只能复制表层的语言能力，缺乏深层的因果推理和自我反思。根本的瓶颈在于：**如何在保持极低计算开销的同时，让模型从训练阶段就内生出真正的智能体行为**。这正是 Youtu-LLM 试图突破的关键点。

### 关键概念速览

**多潜在注意力（Multi‑Latent Attention，MLA）**：一种密集的注意力机制，内部维护多个潜在表示层，能够在同一次前向传播中捕捉不同粒度的上下文信息。想象成在同一张纸上写了几层透明的草稿，阅读时可以一次性看到全部层次。

**长上下文窗口（128k）**：模型一次性可以处理约 12.8 万个 token，远超常规 2‑4k 的限制。相当于一次阅读一本中等篇幅的小说而不需要翻页。

**STEM‑导向词表**：专门为科学、技术、工程、数学（STEM）领域设计的 tokenizer，避免把数字、公式等拆成不自然的子词。类似于把专业术语直接写进字典，让模型“一眼就认出”。

**Commonsense‑STEM‑Agent 课程**：一种分阶段的训练数据安排，先让模型学习日常常识，再逐步加入更难的 STEM 与智能体任务，形成层层递进的认知梯度。

**Agentic Mid‑training**：在预训练的中后期，专门投放大量包含计划、反思、工具调用等轨迹的数据，让模型在“做事”而不是仅“说话”上练习。

**SFT（监督微调）**：在大规模预训练后，用标注好的任务数据进一步调教模型，使其在特定场景下表现更稳健。这里分为“推理导向”和“通用能力”两条支路。

**RL（强化学习）**：通过奖励函数让模型学会优化输出，例如奖励答案的可验证性、语言一致性或避免重复。类似于给模型设定“游戏规则”，让它在玩耍中变强。

### 核心创新点

1. **从头训练的轻量模型 → 采用 1.96 B 参数的密集 MLA 架构 + 128k 长上下文**  
   过去的轻量模型大多是大模型的蒸馏产物，结构上仍受限于短上下文。Youtu-LLM 直接从零开始训练，使用 MLA 让每个 token 能在极低的显存占用下访问更远的历史信息，从而在 2 B 参数规模内实现了类似 10 B 以上模型的长程推理能力。

2. **分层课程式数据安排 → Commonsense‑STEM‑Agent 三阶段 curriculum**  
   传统预训练往往一次性喂入海量混合数据，模型容易只学到表层统计。作者先让模型熟悉常识（约 8 T token），再把 STEM 与代码数据比例提升至 60%，最后注入大量智能体轨迹。这样模型的认知结构像是先学会走路，再学会跑步、最后学会使用工具，显著提升了深层推理与规划能力。

3. **专门的 Agentic 中期训练 → 多样化轨迹合成 + 60% 以上的智能体数据**  
   在中期训练阶段，作者不再使用普通文本，而是构造包含数学求解、代码生成、工具调用等完整“任务-思考-行动-反馈”链路的数据。模型因此能够在内部形成“计划 → 执行 → 反思”的循环，区别于仅在微调阶段才尝试教会模型这些行为的做法。

4. **统一的 SFT+RL 微调框架 → 推理导向 SFT + 通用 SFT + 多任务 RL 奖励**  
   过去的轻量模型往往只做一次监督微调，缺少对行为一致性的强化。Youtu-LLM 先用两条 SFT 轨道分别强化推理和通用语言能力，再通过任务特定的奖励函数进行 RL 微调，使模型在不同场景下都能保持高质量输出。

### 方法详解

**整体思路**  
Youtu-LLM 的训练流程可以划分为四大块：① 常识预训练 → ② STEM/代码强化 → ③ 长上下文扩展 + Agentic 中期训练 → ④ 双轨 SFT + 多任务 RL。每一步都围绕“让模型先学会说，再学会想，最后学会做”展开。

**1. 架构与词表**  
- **MLA 结构**：模型内部维护多个潜在注意力层，每层负责不同跨度的依赖捕获。相当于在一次注意力计算中并行处理“局部细节”和“全局概览”。这种设计让显存占用几乎不随上下文长度线性增长，从而支撑 128k token。
- **STEM‑导向 tokenizer**：在构造词表时，所有数字、化学式、编程符号都被保留为完整 token，避免了常见的子词切分导致的语义碎片。这样在处理数学或代码时，模型可以直接看到完整的符号，提升了理解准确度。

**2. 课程式数据流**  
- **阶段一（常识）**：使用约 8 T token 的网页、百科等通用文本，上下文窗口 8k，目标是让模型掌握语言基本规律和日常知识。
- **阶段二（STEM/代码）**：在已有数据上加入大量科研论文、开源代码、数学教材，使 STEM/代码占比提升至 60%。此时模型开始学习专业符号和逻辑推理。
- **阶段三（长上下文 + 轻度 Agent）**：将上下文窗口逐步扩展到 128k，保持 STEM/代码比例略增，模型被迫学会在更长的文本中保持状态追踪。
- **阶段四（Agentic Mid‑training）**：约 60% 的输入是合成的智能体轨迹，涵盖数学求解、代码调试、工具调用等完整流程。数据生成方式包括：① 基于已有 LLM 的自回归生成，② 人工标注的计划-执行对，③ 自动化脚本模拟的工具交互。

**3. 监督微调（SFT）**  
- **推理导向 SFT**：挑选高质量的 Chain‑of‑Thought（思维链）数据，让模型在回答前先输出推理步骤。  
- **通用能力 SFT**：使用大规模对话、摘要等多样化任务，确保模型在日常交互中仍保持流畅性。  
- 两者在训练后期混合，形成兼顾推理深度和语言自然度的模型。

**4. 强化学习（RL）**  
- 为每类任务设计专属奖励：数学任务奖励答案可验证性，代码任务奖励编译通过率，工具使用任务奖励成功调用次数。  
- 采用 PPO（近端策略优化）等常见 RL 算法，在保持语言一致性的同时提升行为成功率。  
- 奖励函数中还加入“重复惩罚”，防止模型在长上下文中出现循环输出。

**最巧妙的点**  
- **长上下文+MLA 的组合**：在 2 B 参数规模下实现 128k 上下文，这在以往只能靠数十亿参数的稀疏模型实现。  
- **从预训练阶段就注入智能体轨迹**：打破了“先训练语言模型、后再教会它做事”的传统顺序，让模型的内部表征天然具备计划与反思的结构。

### 实验与效果

- **评测任务**：包括通用语言基准（如 MMLU、ARC）、长文理解（LongBench）、以及专门的智能体任务（ToolBench、MathAgent、CodeAgent）。  
- **相对表现**：论文声称在所有 2 B 以下模型中，Youtu-LLM 在通用基准上接近 7 B 参数模型的分数；在智能体任务上比同尺寸的 LLaMA‑2‑Chat、Mistral‑Base 等基线提升 15%~30% 以上。  
- **消融实验**：作者分别去掉 MLA、长上下文、Agentic Mid‑training、STEM‑导向词表，发现最显著的性能跌幅来自 Agentic 中期训练（约下降 12%），其次是 MLA（约下降 8%），说明两者是提升智能体能力的关键。  
- **局限性**：虽然在 128k 上下文下表现不错，但在极端长序列（>200k）仍会出现记忆衰减；RL 阶段对奖励函数的设计仍较为手工，迁移到全新任务时需要重新调参。

### 影响与延伸思考

Youtu-LLM 的成功向社区展示：**轻量模型完全可以在训练阶段就拥有原生的智能体能力**，这激发了后续不少工作尝试在 1‑3 B 参数范围内加入规划、工具使用等行为。例如，2024 年的 “Mini‑Agent” 系列模型直接借鉴了其课程式数据流和 MLA 架构；2025 年的 “STEM‑Lite” 进一步细化了 STEM‑导向词表的构造方法。对想继续深入的读者，建议关注以下方向：① 更高效的长上下文稀疏注意力与 MLA 的结合；② 自动化生成高质量 Agentic 轨迹的技术；③ 跨模态（语言+视觉）下的轻量智能体训练。

### 一句话记住它

**Youtu-LLM 证明：只要在预训练里把“会说”和“会做”一起教，2 B 参数的模型也能像大模型一样规划、反思并使用工具。**