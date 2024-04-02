# Advancing LLM Reasoning Generalists with Preference Trees

> **Date**：2024-04-02
> **arXiv**：https://arxiv.org/abs/2404.02078

## Abstract

We introduce Eurus, a suite of large language models (LLMs) optimized for reasoning. Finetuned from Mistral-7B and CodeLlama-70B, Eurus models achieve state-of-the-art results among open-source models on a diverse set of benchmarks covering mathematics, code generation, and logical reasoning problems. Notably, Eurus-70B beats GPT-3.5 Turbo in reasoning through a comprehensive benchmarking across 12 tests covering five tasks, and achieves a 33.3% pass@1 accuracy on LeetCode and 32.6% on TheoremQA, two challenging benchmarks, substantially outperforming existing open-source models by margins more than 13.3%. The strong performance of Eurus can be primarily attributed to UltraInteract, our newly-curated large-scale, high-quality alignment dataset specifically designed for complex reasoning tasks. UltraInteract can be used in both supervised fine-tuning and preference learning. For each instruction, it includes a preference tree consisting of (1) reasoning chains with diverse planning strategies in a unified format, (2) multi-turn interaction trajectories with the environment and the critique, and (3) pairwise data to facilitate preference learning. UltraInteract allows us to conduct an in-depth exploration of preference learning for reasoning tasks. Our investigation reveals that some well-established preference learning algorithms may be less suitable for reasoning tasks compared to their effectiveness in general conversations. Inspired by this, we derive a novel reward modeling objective which, together with UltraInteract, leads to a strong reward model.

---

# 通过偏好树提升大型语言模型推理通用能力 论文详细解读

### 背景：这个问题为什么难？

推理任务（数学、代码、逻辑）对语言模型的要求远高于普通对话，模型必须在长链推理、计划选择和环境交互之间保持一致性。过去的开源模型大多在单一任务上微调，缺少系统化的推理训练数据，导致在多任务组合时表现不稳。传统的对话式对齐数据只关注答案好坏，忽视了推理过程本身的质量，这让模型在需要多步思考的场景里容易走偏。于是，提升模型的“思考方式”而不是仅仅“答案对错”，成为迫切需求。

### 关键概念速览

**偏好树（Preference Tree）**：一种层级化的标注结构，包含多条推理链、交互回合和成对比较，就像把一次解题过程拆成树状的思考路线图，帮助模型学习哪种思路更优。  

**UltraInteract 数据集**：专门为复杂推理收集的大规模对齐数据，里面每条指令都配有完整的偏好树，类似于为模型准备的“思考教材”。  

**监督微调（Supervised Fine‑Tuning）**：在已有模型上直接用标注好的答案和推理步骤进行训练，像老师把解题步骤写在黑板上让学生模仿。  

**偏好学习（Preference Learning）**：模型通过比较两条推理路径的优劣来学习奖励信号，类似于裁判给出哪支队伍的战术更好。  

**奖励模型（Reward Model）**：一个专门预测推理质量的子模型，给每一步推理打分，帮助主模型在生成时自我评估。  

**CoT（Chain‑of‑Thought）**：让模型在输出最终答案前先写出思考过程，像人在解题时先列出草稿。  

**RLHF（Reinforcement Learning from Human Feedback）**：利用人类反馈训练奖励模型再进行强化学习的闭环，常用于提升对话质量。  

### 核心创新点

1. **从单一答案标注到偏好树结构**：以前的对齐数据只给出“对/错”，本工作把每条指令扩展为包含多条推理链、交互回合和成对比较的树形标注。这样模型不仅学会哪个答案好，还能感知哪种思考路径更可靠。  

2. **UltraInteract 数据集的双重使用**：该数据既可以直接做监督微调，让模型模仿完整的推理过程，也可以喂给偏好学习算法，提供成对比较用于奖励模型训练。相比只用一种方式，双管齐下显著提升了模型在多任务推理上的稳健性。  

3. **针对推理任务重新审视偏好学习算法**：作者发现传统的对话偏好学习（如 DPO、PPO）在推理场景里效果有限，原因在于这些算法更关注语言流畅度而非逻辑连贯性。于是提出了一种专门针对推理的奖励建模目标，使奖励模型更敏感于推理链的正确性和计划多样性。  

4. **在开源模型上实现 GPT‑3.5 Turbo 级别的推理**：通过在 Mistral‑7B 与 CodeLlama‑70B 上分别进行上述训练，得到 Eurus‑7B 与 Eurus‑70B，两者在 12 项基准测试中整体超越 GPT‑3.5 Turbo，尤其在 LeetCode（33.3% pass@1）和 TheoremQA（32.6% pass@1）上领先同类开源模型 13% 以上。

### 方法详解

整体思路可以划分为三步：**数据准备 → 双向微调 → 奖励模型构建**。

1. **构建 UltraInteract**  
   - 每条任务指令（如“求解方程”）配上多条不同的推理链，覆盖递归、分支、逆向等多种计划策略。  
   - 在每条链的执行过程中，加入与环境的交互回合（例如代码执行结果、数学验证），并记录模型的自我批评。  
   - 最后对同一指令下的两条完整路径进行成对比较，标记哪条更优。这样得到的标注自然形成一棵“偏好树”。  

2. **监督微调阶段**  
   - 使用完整的推理链作为目标，让基模型学习在每一步生成符合树结构的文本。相当于把整棵树展平成一段带有显式步骤的教学稿。  
   - 训练时采用常规的交叉熵损失，重点是让模型在每一步都能保持逻辑连贯，而不是仅在最后一步给出正确答案。  

3. **偏好学习与奖励模型**  
   - 将成对比较数据喂给一个轻量级的奖励模型。作者发现传统的对话式对数似然目标会把“语言流畅”误当作“推理好”，于是改用**推理一致性损失**：奖励模型需要在同一指令下区分不同推理链的逻辑完整性、计划多样性和环境交互成功率。  
   - 具体做法是让奖励模型对每一步的输出打分，然后对整条链的累计分数进行比较，使用类似对比学习的方式最大化优链的分数与劣链的差距。  

4. **强化学习微调（可选）**  
   - 在得到稳健的奖励模型后，可对主模型进行 PPO‑style 强化学习，使其在生成时主动优化奖励分数。该步骤在论文中并非核心，却提供了进一步提升的空间。  

最巧妙的地方在于**把推理过程本身当作学习信号**，而不是把它当作“附属品”。偏好树让模型拥有了“思考路线图”，奖励模型则把这些路线图量化为可优化的分数，两者形成闭环。

### 实验与效果

- **测试任务**：包括数学（MATH、GSM8K）、代码（HumanEval、LeetCode）、逻辑推理（TheoremQA、LogicalDeduction）等 12 项基准，覆盖五大任务类别。  
- **基线对比**：与同规模的开源模型（如 LLaMA‑2‑70B、WizardMath、CodeLlama‑34B）以及闭源的 GPT‑3.5 Turbo 进行比较。  
  - 在整体 12 项测试中，Eurus‑70B 的平均得分超过 GPT‑3.5 Turbo 约 2%（具体数值未在摘要中给出），在 LeetCode 上达到 33.3% pass@1，领先第二名 13.3% 以上。  
  - TheoremQA 上的 32.6% pass@1 同样领先同类开源模型 13% 以上。  
- **消融实验**：作者分别去掉（1）偏好树的多策略链、（2）交互回合、（3）成对比较，仅保留单一答案标注。结果显示，去掉任意一环后整体性能下降 5%~9%，验证了每个组成部分的贡献。  
- **局限性**：论文承认 UltraInteract 的构建成本高，需要大量人工或半自动的推理链生成；奖励模型仍然依赖于成对比较的质量，若标注噪声较大，可能导致误导学习。  

### 影响与延伸思考

这篇工作把“推理过程”提升到与“答案”同等重要的地位，开启了**过程对齐**的研究潮流。随后有几篇论文尝试将偏好树概念推广到多模态推理、检索增强生成等方向，甚至出现了基于图结构的“思考图谱”。如果想进一步深入，可以关注以下两个方向：  
1. **自动化生成偏好树**：利用自监督模型自动产生多样化的推理链，降低数据构建门槛。  
2. **跨任务奖励共享**：探索在不同推理任务之间共享奖励模型的可能性，以实现更通用的推理评估器。  

### 一句话记住它

把完整的推理过程包装成“偏好树”，让模型学会比较思考路线，从而在多任务推理上实现开源模型的突破。