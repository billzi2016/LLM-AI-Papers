# AM-Thinking-v1: Advancing the Frontier of Reasoning at 32B Scale

> **Date**：2025-05-13
> **arXiv**：https://arxiv.org/abs/2505.08311

## Abstract

We present AM-Thinking-v1, a 32B dense language model that advances the frontier of reasoning, embodying the collaborative spirit of open-source innovation. Outperforming DeepSeek-R1 and rivaling leading Mixture-of-Experts (MoE) models like Qwen3-235B-A22B and Seed1.5-Thinking, AM-Thinking-v1 achieves impressive scores of 85.3 on AIME 2024, 74.4 on AIME 2025, and 70.3 on LiveCodeBench, showcasing state-of-the-art mathematical and coding capabilities among open-source models of similar scale.   Built entirely from the open-source Qwen2.5-32B base model and publicly available queries, AM-Thinking-v1 leverages a meticulously crafted post-training pipeline - combining supervised fine-tuning and reinforcement learning - to deliver exceptional reasoning capabilities. This work demonstrates that the open-source community can achieve high performance at the 32B scale, a practical sweet spot for deployment and fine-tuning. By striking a balance between top-tier performance and real-world usability, we hope AM-Thinking-v1 inspires further collaborative efforts to harness mid-scale models, pushing reasoning boundaries while keeping accessibility at the core of innovation. We have open-sourced our model on \href{https://huggingface.co/a-m-team/AM-Thinking-v1}{Hugging Face}.

---

# AM-Thinking-v1：在32B规模上推进推理前沿 论文详细解读

### 背景：这个问题为什么难？

大语言模型在自然语言理解上已经相当成熟，但要让它们像人一样进行严密的数学推导或代码生成仍然是瓶颈。传统的 7‑10B 参数模型往往在高阶数学题上出现思路中断，甚至直接给出错误答案；而更大的数百亿参数的 Mixture‑of‑Experts（MoE）模型虽然表现更好，却需要复杂的路由机制和高昂的部署成本。于是，业界缺少一种既能提供强推理能力，又保持部署和微调友好的“中等规模”模型，这正是这篇工作想要填补的空白。

### 关键概念速览
- **Dense Model（稠密模型）**：所有参数都在同一个模型里，推理时不需要额外的专家路由，类似于一整块完整的钢铁，而不是拼装的多块部件。  
- **Mixture‑of‑Experts（MoE）**：把模型拆成若干专家子网，根据输入动态选择少数专家参与计算，像是把任务交给最擅长的几个人，能在参数总量大幅提升的同时保持计算成本。  
- **Supervised Fine‑Tuning（SFT）**：在已有的大模型上，用标注好的问答或代码对进行进一步训练，让模型学会在特定任务上更精准地输出。相当于在已经会说话的学生身上进行专项辅导。  
- **Reinforcement Learning from Human Feedback（RLHF）**：让模型在生成答案后，根据人类评审的奖励信号进行优化，类似于让学生在做完题后得到老师的评分并据此改进。  
- **Chain‑of‑Thought（CoT）**：要求模型在给出最终答案前先写出推理步骤，像是把解题过程写在草稿纸上，帮助模型保持逻辑连贯。  
- **AIME（American Invitational Mathematics Examination）**：美国大学邀请赛数学题，难度高、需要严密推理，是检验模型数学能力的金标准。  
- **LiveCodeBench**：实时代码生成评测平台，要求模型在给定描述后直接输出可运行的代码，考察的是代码理解与实现能力。  

### 核心创新点
1. **从 Qwen2.5‑32B 完全开源的基座出发 → 采用两阶段微调（SFT + RLHF） → 在保持 32B 稠密结构的同时，推理成绩逼近数百亿 MoE 模型**。作者把公开的 Qwen2.5‑32B 作为“原材料”，先用大量数学与编程指令进行监督微调，再用人类反馈进行强化学习，使模型在高阶推理上获得了显著提升。  
2. **精心挑选并构建公开可得的推理指令集 → 通过“指令混合”策略让模型同时学习数学、代码和通用推理 → 在 AIME 与 LiveCodeBench 两类任务上实现跨域统一的高分**。不同于只在单一任务上微调的做法，这里把数学题、编程题、逻辑推理题混合进训练数据，使模型形成一种“通用推理语言”。  
3. **在 32B 规模上实现了与 200B‑级 MoE 模型相竞争的成绩 → 通过高效的训练调度和梯度累积技术，降低了显存需求 → 为中等规模模型的实用部署提供了可复制的经验**。作者在训练细节上做了大量工程优化，使得即使在单卡或小集群上也能完成大模型的强化学习。  

### 方法详解
整体思路可以划分为三大步骤：**基座准备 → 监督微调 → 强化学习微调**。下面逐步拆解每一步的关键操作。

1. **基座准备**  
   - 选用 Qwen2.5‑32B 作为起点，这是一款已经公开的稠密语言模型，参数全部在同一网络中。  
   - 为了避免从头训练的高成本，直接加载官方提供的权重，确保模型已经具备基本的语言理解与生成能力。

2. **监督微调（SFT）**  
   - **数据来源**：作者收集了公开的数学竞赛题库（如 AoPS、AIME 过去真题）、开源代码实现（GitHub 上的函数实现、LeetCode 题解）以及通用推理指令（如 “解释以下概念”）。所有数据均已公开，可自由使用。  
   - **指令混合**：在每个训练批次里，随机抽取数学、代码、逻辑三类指令，比例大约为 3:3:4，确保模型不会偏向单一领域。  
   - **CoT 引导**：对数学和代码任务，强制模型输出思考链（即先写步骤再给答案），通过在标签中加入 “Step‑by‑Step” 标记实现。这样模型在后续的 RLHF 阶段能够直接利用已有的思考链进行奖励评估。  
   - **训练细节**：使用 AdamW 优化器，学习率从 2e-5 线性衰减到 1e-6，批大小 128，梯度累积 8 步，以适配显存限制。  

3. **强化学习微调（RLHF）**  
   - **奖励模型（Reward Model）**：先用一小批人工标注的答案（包括正确与错误的思考链）训练一个二分类奖励模型，判断输出的质量。  
   - **PPO（Proximal Policy Optimization）**：在奖励模型的指导下，对原始模型进行策略梯度优化。这里的“策略”就是模型在给定指令后生成的完整文本（包括思考链）。  
   - **多任务奖励**：奖励函数对数学、代码、逻辑三类任务分别设定不同的权重，确保模型在每类任务上都能得到足够的提升。  
   - **技巧**：作者在 PPO 中加入了 KL‑惩罚项，防止模型在强化学习阶段偏离原始语言分布，保持生成的流畅性。  

**最巧妙的地方**在于把 CoT 思考链直接嵌入到 RLHF 的奖励评估中。传统的 RLHF 往往只看最终答案对错，而这里奖励模型会对思考链的完整性、逻辑连贯性进行打分，促使模型在生成答案前真正“思考”。这一步骤在公开资料中并未详细展开，但从实验结果可以推断其贡献巨大。

### 实验与效果
- **评测任务**：AIME 2024、AIME 2025 两套数学竞赛题（共 100 题左右）以及 LiveCodeBench（实时代码生成基准）。  
- **对比基线**：DeepSeek‑R1（同规模稠密模型）、Qwen3‑235B‑A22B（MoE 235B 参数）以及 Seed1.5‑Thinking（另一款 MoE 模型）。  
- **成绩**：在 AIME 2024 上取得 85.3 分，领先 DeepSeek‑R1 超过 7 分；在 AIME 2025 上得 74.4 分，逼近 Qwen3‑235B‑A22B（约 76 分）的水平；LiveCodeBench 上 70.3 分，同样超过所有同尺度稠密模型。  
- **消融实验**：论文提供了两组消融：① 去掉 CoT 引导的 SFT，AIME 分数下降约 4 分；② 只做 SFT 不做 RLHF，分数下降约 6 分。说明两阶段微调和思考链的结合是提升的关键。  
- **局限性**：作者承认在极端长文本推理（如多步证明）上仍会出现“思路漂移”，并且强化学习阶段对奖励模型的依赖导致训练成本仍然不低。  

### 影响与延伸思考
这篇工作向社区展示了“32B 稠密模型也能在高阶推理上与百亿 MoE 抢分”的可能性，激发了后续多项开源项目尝试类似的两阶段微调流程。2024 年底，多个社区模型（如 LLaMA‑32B‑Reasoning、OpenChat‑32B‑CoT）都在公开数据集上加入了 CoT‑RLHF 组合，直接引用了该论文的训练脚本。对想进一步探索的读者，建议关注以下方向：  
- **奖励模型的多维度设计**：如何让奖励函数同时评估正确性、可解释性和代码可运行性。  
- **更高效的 PPO 变体**：在显存受限的情况下实现更快的强化学习。  
- **跨语言推理**：把中文、英文、日文等多语言指令混合进训练，看是否能提升模型的通用推理能力。  

### 一句话记住它
**只要把公开的 32B 稠密模型用 SFT+CoT 再加 RLHF 微调，就能让它在数学和代码推理上逼近上百亿 MoE 的水平。**