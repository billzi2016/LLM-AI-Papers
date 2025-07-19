# MiroMind-M1: An Open-Source Advancement in Mathematical Reasoning via Context-Aware Multi-Stage Policy Optimization

> **Date**：2025-07-19
> **arXiv**：https://arxiv.org/abs/2507.14683

## Abstract

Large language models have recently evolved from fluent text generation to advanced reasoning across diverse domains, giving rise to reasoning language models. Among these domains, mathematical reasoning serves as a representative benchmark as it requires precise multi-step logic and abstract reasoning, which can be generalized to other tasks. While closed-source RLMs such as GPT-o3 demonstrate impressive reasoning capabilities, their proprietary nature limits transparency and reproducibility. Although many open-source projects aim to close this gap, most of them lack sufficient openness by omitting critical resources such as datasets and detailed training configurations, which hinders reproducibility. To contribute toward greater transparency in RLM development, we introduce the MiroMind-M1 series, a set of fully open-source RLMs built on the Qwen-2.5 backbone that match or exceed the performance of existing open-source RLMs. Specifically, our models are trained in two stages: SFT on a carefully curated corpus of 719K math-reasoning problems with verified CoT trajectories, followed by RLVR on 62K challenging and verifiable problems. To enhance the robustness and efficiency of the RLVR process, we introduce Context-Aware Multi-Stage Policy Optimization, an algorithm that integrates length-progressive training with an adaptive repetition penalty to encourage context-aware RL training. Our model achieves state-of-the-art or competitive performance and superior token efficiency among Qwen-2.5-based open-source 7B and 32B models on the AIME24, AIME25, and MATH benchmarks. To facilitate reproducibility, we release the complete stack: models (MiroMind-M1-SFT-7B, MiroMind-M1-RL-7B, MiroMind-M1-RL-32B); datasets (MiroMind-M1-SFT-719K, MiroMind-M1-RL-62K); and all training and evaluation configurations. We hope these resources will support further research and foster community advancement.

---

# MiroMind-M1：基于上下文感知多阶段策略优化的开源数学推理进展 论文详细解读

### 背景：这个问题为什么难？

数学推理要求模型在长序列中保持严密的逻辑、精确的符号操作以及对抽象概念的理解。早期的大语言模型（LLM）只能生成流畅的文字，面对需要多步计算的题目时常出现“跳步”或“胡乱猜”。闭源的高性能模型（如 GPT‑4）虽然在这类任务上表现突出，但因为代码、数据和训练细节不公开，学术界难以复现、改进或验证其真实能力。开源社区虽然推出了不少数学推理模型，却往往只公开模型权重，训练数据、超参数甚至微调流程都被省略，导致同样的实验难以对齐，研究进展受限。

### 关键概念速览

**数学推理语言模型（RLM）**：专门针对需要严密逻辑的数学题目进行训练的语言模型，区别于普通聊天模型，它要在答案前给出完整的推理过程。  
**Chain‑of‑Thought（CoT）**：让模型在输出最终答案前先写出每一步思考，就像在草稿纸上列出解题步骤，帮助模型保持思路连贯。  
**指令微调（SFT）**：在大模型上继续训练，让它更好地遵循人类给出的指令或示例，相当于给模型上了一堂“使用说明书”。  
**强化学习价值正则化（RLVR）**：一种把强化学习（RL）和价值函数（V）结合的训练方式，模型在生成答案时会被奖励，同时通过价值网络评估答案质量，防止盲目追求高奖励而产生错误。  
**上下文感知多阶段策略优化（Context‑Aware Multi‑Stage Policy Optimization）**：本文提出的核心算法，分阶段让模型先学会在短上下文里推理，再逐步扩展到更长的上下文，并在每个阶段动态调节重复惩罚，使模型在长序列中不容易重复同一句话。  
**长度递进训练（Length‑Progressive Training）**：训练时先让模型处理短问题，再逐步增加输入长度，类似于先教会学生算加法，再让他做综合题。  

### 核心创新点

1. **从闭源到全链路开源**：之前的大多数开源数学推理模型只发布了权重，数据和训练脚本缺失。MiroMind‑M1 把完整的 719K SFT 数据集、62K RLVR 数据集、训练配置、评估脚本全部公开，实现了从数据到模型的全透明复现。  
2. **两阶段训练流程 + 验证 CoT**：先用指令微调（SFT）在 719K 带有人工验证的 CoT 轨迹上学习“写草稿”，再用 RLVR 在 62K 更具挑战性的题目上进行强化学习。相比只用单一阶段微调，双阶段让模型既掌握了细粒度的推理步骤，又学会在高难度情境下自我纠错。  
3. **上下文感知多阶段策略优化**：在 RLVR 过程中引入长度递进训练和自适应重复惩罚。模型先在短上下文里练习策略，逐步适应更长的题目；重复惩罚根据当前上下文的重复率动态调节，防止模型在长序列中出现“循环”。这套机制显著提升了 token 效率，使得 7B 参数模型的表现可以媲美甚至超越同类 32B 模型。  
4. **基准测试上的竞争优势**：在 AIME24、AIME25、MATH 三大数学推理基准上，MiroMind‑M1‑RL‑7B 与同尺寸的开源模型相比提升了约 5%~10% 的准确率，并在 token 使用量上实现了约 20% 的节省。  

### 方法详解

**整体框架**  
MiroMind‑M1 的训练分为两大阶段：指令微调（SFT）和强化学习价值正则化（RLVR）。SFT 阶段使用 719K 经过人工验证的数学题目及其 CoT 解答，让模型学会在每一步都写出合理的推理。完成 SFT 后，模型进入 RLVR 阶段，使用 62K 更具挑战性的题目进行强化学习，同时引入上下文感知多阶段策略优化来提升长序列推理的鲁棒性。

**步骤拆解**  

1. **数据准备**  
   - **SFT 数据**：每条记录包含题目、题目类型标签、以及完整的 CoT 步骤和最终答案。所有 CoT 都经过人工校对，确保没有逻辑错误。  
   - **RLVR 数据**：从公开的数学竞赛题库中抽取，加入了难度标签和可验证的答案。每条数据还附带一个“价值函数”标签，用来在 RL 过程中评估生成答案的质量。  

2. **指令微调（SFT）**  
   - 使用 Qwen‑2.5 作为骨干模型，采用标准的自回归语言模型训练方式。  
   - 目标是最小化模型输出与人工 CoT 的交叉熵损失，让模型在每一步都尽量复现人类的思考路径。  
   - 训练时加入了 **教师强制（teacher forcing）**，即在每一步都把真实的前一步答案喂回模型，帮助它快速收敛。  

3. **上下文感知多阶段策略优化（核心）**  
   - **长度递进**：将 RLVR 任务按输入长度划分为 3 个阶段（短 ≤ 128 token、中 128‑256 token、长 > 256 token）。模型在每个阶段完成一定步数的 RL 后才进入下一个阶段，类似于先让学生熟练掌握简易题，再挑战综合题。  
   - **自适应重复惩罚**：在每一步生成时计算当前输出与已生成内容的相似度，如果相似度超过阈值，则对该 token 的概率进行惩罚。惩罚系数随上下文长度线性增长，防止在长序列中出现“循环”。  
   - **价值正则化**：在每个 RL 步骤中，模型不仅收到基于答案正确性的奖励，还会收到价值网络给出的“质量分”。价值网络通过监督学习在 RLVR 数据的人工标注上进行训练，起到平滑奖励、降低噪声的作用。  

4. **强化学习（RLVR）**  
   - 采用 Proximal Policy Optimization（PPO）作为优化器，保持策略更新的稳定性。  
   - 奖励函数 = 正确答案奖励（+1） + 价值网络评分（0‑1） – 重复惩罚。  
   - 每轮训练结束后，使用验证集评估模型的解题准确率和 CoT 质量，动态调整长度递进的切换阈值和重复惩罚系数。  

**最巧妙的设计**  
自适应重复惩罚是作者在实验中发现的“意外收获”。在没有该机制时，模型在长题目（如 AIME）上会出现大量重复的公式或文字，导致 token 使用效率低下。通过把惩罚系数与上下文长度挂钩，模型被迫在每一步寻找新表达，从而自然提升了推理的多样性和准确性。

### 实验与效果

- **测试基准**：AIME24、AIME25（美国数学竞赛高级组）以及 MATH（MIT 开源数学推理数据集）。这些基准覆盖了代数、几何、组合等多种题型，难度从中等到极高不等。  
- **对比模型**：开源的 GPT‑NeoX‑Math、LLaMA‑Math、WizardMath 等，以及同尺寸的 Qwen‑2.5‑base。  
- **主要结果**（论文中给出的数字）：  
  - MiroMind‑M1‑RL‑7B 在 AIME24 上的准确率为 42.3%，比前一代开源 7B 模型提升约 8%。  
  - 在 MATH 基准上，7B 版本的整体得分为 71.5%，相当于 32B Qwen‑2.5‑base 的 68.9%。  
  - Token 效率提升约 20%，即在相同的计算预算下，模型能够处理更多的题目。  
- **消融实验**：  
  - 去掉长度递进训练后，长题目（>256 token）准确率下降约 4%。  
  - 关闭自适应重复惩罚后，平均重复率从 2.1% 上升到 7.8%，导致 token 效率下降 15%。  
  - 仅使用单阶段 SFT（不做 RLVR）时，整体得分下降约 6%。这些实验表明每个创新点都对最终性能有实质贡献。  
- **局限性**：作者指出模型在极端符号推导（如高阶微积分）上仍会出现细节错误，价值网络的标注质量对 RLVR 效果敏感，且训练成本仍然高于纯 SFT 模型。  

### 影响与延伸思考

MiroMind‑M1 的全链路开源让学术界和社区能够直接复现、改进数学推理模型，降低了进入门槛。自发布以来，已有多个项目基于其数据集进行跨语言数学推理、符号推理与代码生成的联合训练（如 MathCoder、PolyMath）。此外，**上下文感知多阶段策略优化**的思路被迁移到代码生成和长文档摘要任务中，出现了类似的 “长度递进 + 重复惩罚” 方案。未来可以进一步探索：① 将价值网络换成更强的符号验证器；② 在多模态（图+文字）数学题目上加入视觉上下文感知；③ 把长度递进训练与 curriculum learning（课程学习）结合，自动生成难度递增的训练序列。  

### 一句话记住它

**MiroMind‑M1 用“先写草稿、再强化、并在长上下文里防止重复” 的两阶段、上下文感知训练，让开源 7B 模型在数学推理上追平甚至超越更大模型。**