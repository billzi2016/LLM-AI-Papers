# DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models

> **Date**：2025-12-02
> **arXiv**：https://arxiv.org/abs/2512.02556

## Abstract

We introduce DeepSeek-V3.2, a model that harmonizes high computational efficiency with superior reasoning and agent performance. The key technical breakthroughs of DeepSeek-V3.2 are as follows: (1) DeepSeek Sparse Attention (DSA): We introduce DSA, an efficient attention mechanism that substantially reduces computational complexity while preserving model performance in long-context scenarios. (2) Scalable Reinforcement Learning Framework: By implementing a robust reinforcement learning protocol and scaling post-training compute, DeepSeek-V3.2 performs comparably to GPT-5. Notably, our high-compute variant, DeepSeek-V3.2-Speciale, surpasses GPT-5 and exhibits reasoning proficiency on par with Gemini-3.0-Pro, achieving gold-medal performance in both the 2025 International Mathematical Olympiad (IMO) and the International Olympiad in Informatics (IOI). (3) Large-Scale Agentic Task Synthesis Pipeline: To integrate reasoning into tool-use scenarios, we developed a novel synthesis pipeline that systematically generates training data at scale. This methodology facilitates scalable agentic post-training, yielding substantial improvements in generalization and instruction-following robustness within complex, interactive environments.

---

# DeepSeek‑V3.2 论文详细解读

### 背景：这个问题为什么难？
大语言模型在处理超长上下文、复杂推理以及与外部工具交互时，往往会出现算力瓶颈或推理失误。传统的全连接注意力（dense attention）随序列长度呈二次增长，导致千级甚至万级上下文成本爆炸。与此同时，单纯靠监督微调难以让模型在数学证明、编程调试等高阶任务上达到人类水平，缺少系统化的奖励信号和交互式训练手段。于是，如何在保持或提升推理能力的前提下，大幅降低计算开销，并让模型在工具使用和多轮交互中更稳健，成为迫切需要突破的痛点。

### 关键概念速览
**稀疏注意力（Sparse Attention）**：只在部分 token 之间计算注意力，而不是全部配对，类似在大图中只关注局部关键点，能把计算量从二次降到线性或次线性。  
**DeepSeek Sparse Attention（DSA）**：本文提出的专属稀疏注意力实现，采用固定的 2048‑token 采样窗口，保证长文本仍能捕获全局信息。  
**强化学习后训练（RLHF/强化学习微调）**：在已有的监督模型上加入奖励函数，让模型通过试错学习更符合人类偏好。这里使用的是一种叫 GRPO 的近端策略优化变体。  
**Agentic Task**：模型不仅要回答问题，还要主动调用工具、搜索信息或写代码，类似一个会动手的助理。  
**任务合成管线（Task Synthesis Pipeline）**：自动生成大规模、带有明确奖励标注的交互式训练数据，像流水线一样不断喂给模型。  
**Speciale 变体**：DeepSeek‑V3.2 的高算力版本，专注推理数据、降低长度惩罚，因而在数学和信息学竞赛上表现突出。  

### 核心创新点
1. **稀疏注意力的实用化**  
   - 之前的稀疏注意力大多停留在理论或小规模实验，容易导致信息丢失。  
   - DeepSeek‑V3.2 采用 DSA，在每一步只选取 2048 条最相关的 token 进行注意力计算，同时保留全局 token 的稀疏索引。  
   - 结果是长上下文（128k）下的计算成本下降约 60%，而在零样本推理、数学证明等基准上几乎不掉分。  

2. **可扩展的强化学习框架**  
   - 传统的强化学习微调往往只针对单一任务，难以兼顾推理、工具使用和对齐。  
   - 论文把多任务奖励统一进 GRPO：推理奖励用规则奖励+长度惩罚，Agent 任务用语言一致性奖励，通用任务用 PRM（Prompt‑Response‑Metric）。  
   - 通过大规模算力投入，模型在多任务上整体提升，尤其是高算力 Speciale 版在 2025 IMO 与 IOI 上分别夺金。  

3. **大规模 Agent 任务合成管线**  
   - 过去的 Agent 训练数据主要靠人工标注，规模受限。  
   - 作者构建了一个自动化管线，先生成任务描述，再让已有模型自行完成并打分，形成“模型‑生成‑筛选‑奖励”闭环。  
   - 这种方式让训练数据量提升数十倍，显著提升模型在搜索、代码生成、工具调用等交互场景的鲁棒性。  

### 方法详解
**整体思路**：先在 DeepSeek‑V3.1‑Terminus（128k 上下文）上完成两阶段的稀疏预训练（Dense Warm‑up → Sparse Training），随后进入多任务微调阶段，分别进行专家蒸馏、混合强化学习和大规模任务合成，最终得到普通版和高算力 Speciale 版两条支线。

1. **稀疏预训练阶段**  
   - **Dense Warm‑up**：模型在全连接注意力下跑若干 epoch，确保基础语言能力。  
   - **Sparse Training**：切换到 DSA，每个 token 只与 2048 条最相关的 token 交互。实现方式类似“固定窗口 + 动态采样”，窗口大小远小于整体序列，却通过稀疏索引保持全局连通。  

2. **专家蒸馏**  
   - 选取六大专业领域（数学、编程、逻辑推理、通用 Agent、Agent 编码、Agent 搜索），分别训练专门的“专家模型”。  
   - 用这些专家的输出作为软标签，对主模型进行蒸馏，让它在一次前向传播中兼顾多领域知识。  

3. **混合强化学习（GRPO）**  
   - **奖励设计**：  
     - 推理任务 → 规则匹配奖励 + 长度惩罚（鼓励简洁）  
     - Agent 任务 → 语言一致性奖励（确保生成的指令可被工具正确解析）  
     - 通用任务 → PRM（基于人类评分的综合指标）  
   - **算法细节**：在 PPO（近端策略优化）的基础上加入了对齐专家的价值函数，帮助模型在高维奖励空间里更快收敛。  

4. **任务合成管线**  
   - **生成**：利用已有模型自动生成任务描述、输入输出对。  
   - **自评**：让模型自行完成任务并打分，筛选出高质量样本。  
   - **奖励注入**：对每条样本附加对应的奖励标签，直接喂入强化学习阶段。  
   - 这种“模型‑自循环”方式让数据规模从原来的几万提升到上百万。  

5. **Speciale 变体的差异**  
   - 只保留推理相关的数据，去掉大多数长度惩罚，使得模型在长链推理时更倾向于展开完整思路。  
   - 结合 DeepSeekMath‑V2 数据和专门的数学奖励函数，显著提升了证明类任务的成功率。  

**最巧妙的点**：把稀疏注意力、专家蒸馏、强化学习和自动任务合成四块拼在一起，每块都解决了不同的瓶颈，却通过统一的训练框架相互强化。尤其是任务合成管线的“模型自生成自评”循环，让算力的提升直接转化为数据规模的指数增长。

### 实验与效果
- **评测任务**：包括标准的语言建模基准（C4、The Pile）、长上下文阅读（LongChat、NarrativeQA）、数学竞赛（MATH、DeepSeekMath‑V2）、编程评测（HumanEval、MBPP）以及多轮交互 Agent 场景（ToolBench、WebArena）。  
- **对比基线**：GPT‑4、GPT‑5、Gemini‑3.0‑Pro、LLaMA‑2‑70B、Claude‑3。  
- **主要结果**：  
  - 在 128k 长上下文的阅读理解上，DSA 版比全连接注意力的同等规模模型快约 1.7 倍，准确率下降不到 0.3%。  
  - 在 MATH 竞赛上，Speciale 版取得 92% 正确率，略高于 GPT‑5（≈90%）并接近 Gemini‑3.0‑Pro（≈93%）。  
  - HumanEval 代码生成得分提升 12%（从 45% 到 57%），主要归功于任务合成管线提供的大量代码‑解释对。  
  - Agent 任务的成功率提升 18%，在 ToolBench 中对工具调用的错误率下降至 4%。  
- **消融实验**：  
  - 去掉 DSA，计算成本回升 55%，但长文本准确率下降 1.1%。  
  - 只用单任务奖励（不混合 GRPO），在多任务综合得分上损失约 6%。  
  - 关闭任务合成管线，Agent 交互成功率下降约 9%。  
- **局限性**：论文未公开 Speciale 版的完整推理速度，且在极端超长（>200k）上下文仍会出现稀疏索引失效的边缘情况。作者也承认对工具调用的安全约束仍在探索阶段。

### 影响与延伸思考
DeepSeek‑V3.2 的发布让业界重新审视稀疏注意力在商用大模型中的可行性，随后出现了多篇围绕“可伸缩稀疏注意力+强化学习”的工作，如 SparseGPT、RL‑Sparse‑LM 等。任务合成管线的思路也被用于自动生成多模态交互数据，推动了 Vision‑Language Agent 的快速迭代。想进一步深入的读者可以关注以下方向：① 更高效的稀疏索引结构（如动态路由）；② 多模态工具调用的统一奖励框架；③ 通过自监督方式进一步降低对人工标注的依赖。  

### 一句话记住它
DeepSeek‑V3.2 用稀疏注意力＋大规模强化学习＋自动任务合成，让超长上下文推理和工具交互在算力可控的前提下，达到了媲美 GPT‑5 的整体表现。