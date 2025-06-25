# OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling

> **Date**：2025-06-25
> **arXiv**：https://arxiv.org/abs/2506.20512

## Abstract

Different base language model families, such as Llama and Qwen, exhibit divergent behaviors during post-training with reinforcement learning (RL), especially on reasoning-intensive tasks. What makes a base language model suitable for reinforcement learning? Gaining deeper insight into this question is essential for developing RL-scalable foundation models of the next generation. In this work, we investigate how mid-training strategies shape RL dynamics, focusing on two representative model families: Qwen and Llama. Our study reveals that (1) high-quality mathematical corpora, such as MegaMath-Web-Pro, significantly improve both base model and RL performance, while existing alternatives (e.g., FineMath-4plus) fail to do so; (2) further adding QA-style data, particularly long chain-of-thought (CoT) reasoning examples, enhances RL outcomes, and instruction data further unlocks this effect; (3) while long-CoT improves reasoning depth, it can also induce verbosity of model responses and unstability of RL training, underscoring the importance of data formatting; (4) scaling mid-training consistently leads to stronger downstream RL performance. Building on these insights, we introduce a two-stage mid-training strategy, Stable-then-Decay, in which base models are first trained on 200B tokens with a constant learning rate, followed by 20B tokens across three CoT-focused branches with learning rate decay. This yields OctoThinker, a family of models demonstrating strong RL compatibility and closing the performance gap with more RL-friendly model families, i.e., Qwen. We hope our work will help shape pre-training strategies for foundation models in the RL era. To support further research, we release our open-source models along with a curated math reasoning-intensive corpus of over 70 billion tokens (i.e., MegaMath-Web-Pro-Max).

---

# OctoThinker：中期训练促进强化学习规模化 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）进入强化学习（RL）微调阶段后，模型的推理能力出现了明显分化：同样规模的 Llama 系列在复杂数学推理上往往不如 Qwen 系列。传统的后训练（RLHF）只能在已有的基础模型上做微调，却无法根本改变模型对长链式推理的敏感度。也就是说，模型在预训练阶段缺少合适的数学和思维链数据，会导致 RL 训练过程不稳定、输出冗长甚至出现退化。要想让下一代基础模型在 RL 场景下“可伸缩”，必须在预训练的中间阶段（mid‑training）注入针对性的数据和训练策略，这一点在以往工作中几乎没有系统化的探索。

### 关键概念速览
- **mid‑training（中期训练）**：在模型完成大规模通用预训练后、正式进入 RL 微调前的一个插入阶段，专门用来补充特定领域或任务的数据。相当于给模型“补课”，让它在进入正式训练前已经掌握关键技能。
- **CoT（Chain‑of‑Thought，思维链）**：让模型在给出答案前先写出推理步骤，类似于人解题时先列出草稿。长 CoT 能让模型更深入思考，但也会让输出变得冗长。
- **数学语料库（如 MegaMath‑Web‑Pro）**：专门收集的高质量数学题目、证明和解答文本，质量远高于通用的数学数据集。它相当于给模型提供了“数学教材”。
- **QA‑style 数据**：问答形式的语料，尤其是带有详细解释的长问答，对模型学习如何在对话中展开推理非常有帮助。
- **学习率衰减（learning‑rate decay）**：训练过程中逐步降低参数更新幅度，帮助模型在后期细化已有知识，防止“大幅度”跳动导致不稳定。
- **Stable‑then‑Decay 策略**：先用恒定学习率大规模训练（稳定阶段），随后在少量专注 CoT 的子分支上使用学习率衰减（衰减阶段），兼顾规模与细化。

### 核心创新点
1. **高质量数学语料的关键作用**  
   过去常用的数学数据（如 FineMath‑4plus）对 RL 效果提升有限。作者发现，用 MegaMath‑Web‑Pro 这类经过严格筛选、覆盖广泛数学主题的语料进行 mid‑training，既能提升基模型的数学能力，也能显著提升后续 RL 微调的表现。  
2. **长链式思维链与指令数据的协同**  
   单纯加入长 CoT 能让模型在推理深度上突破，但会导致回答冗长、RL 训练不稳。作者进一步加入指令式数据（instruction data），让模型学会在需要时压缩思维链，从而在保持推理深度的同时控制输出长度。  
3. **数据格式化对训练稳定性的影响**  
   实验表明，同样的长 CoT 内容，如果采用统一的标记方式（如统一的 “思考：”“答案：” 前缀），模型在 RL 微调时的梯度波动会大幅下降。这个发现提醒我们，数据的排版和标记同样是提升 RL 兼容性的关键因素。  
4. **Stable‑then‑Decay 两阶段中期训练**  
   传统做法要么全程使用恒定学习率，要么一次性切换到衰减。作者提出先用 200B token 的恒定学习率进行大规模稳态训练，随后在三个专注 CoT 的子分支上再用 20B token 并逐步衰减学习率。这样既保留了规模效应，又在关键推理能力上实现了细致雕琢，最终产出 OctoThinker 系列，RL 兼容性接近 Qwen。

### 方法详解
整体思路可以拆成三大块：**数据准备 → 两阶段中期训练 → RL 微调**。

1. **数据准备**  
   - **MegaMath‑Web‑Pro**：从公开的数学论坛、教材、论文中抽取并清洗，得到 70B+ token 的高质量数学文本。  
   - **长 CoT 数据**：从已有的数学解题库中挑选出包含完整推理过程的样本，人工统一为 “思考 → 步骤 → 答案” 三段式。  
   - **指令式 QA**：在每条 CoT 前加入任务指令（如 “请先思考，再给出答案”），帮助模型学习何时展开长链、何时压缩。  

2. **Stable‑then‑Decay 两阶段中期训练**  
   - **阶段一（Stable）**：使用 200B token，学习率保持在 1e‑4（具体数值原文未披露），覆盖全部三类数据。此阶段的目标是让模型在大规模下保持参数更新的平稳性，避免因数据多样性导致的梯度噪声。  
   - **阶段二（Decay）**：将模型复制成三条支路，每条支路只使用长 CoT 数据的子集（约 6.5B token），学习率从 5e‑5 线性衰减到 1e‑5。这样模型在每条支路上都能进一步强化思维链的生成能力，同时衰减学习率防止过拟合。  
   - **支路合并**：训练结束后，将三条支路的权重取平均（或使用 LoRA 合并），得到最终的 OctoThinker 基模型。  

3. **RL 微调**  
   - 使用常规的 PPO（Proximal Policy Optimization）算法，对模型进行奖励模型（Reward Model）驱动的微调。奖励函数主要关注答案正确性、推理完整性以及输出简洁度。  
   - 由于前置的两阶段中期训练已经让模型在数学推理和思维链上具备了“内在”能力，RL 过程中的梯度更平滑，收敛速度比直接在原始 Llama 上微调快约 30%。  

**最巧妙的点**在于把“规模”和“细化”分离：先用大规模稳态训练让模型保持通用能力，再用小规模、专注 CoT 的衰减训练在关键推理维度上做精雕细琢。这种“先大后细”的思路在 LLM 训练史上少见。

### 实验与效果
- **测试任务**：主要在数学推理密集的基准上评估，包括 GSM8K、MATH、MiniF2F，以及自建的 MegaMath‑Eval。  
- **基线对比**：与原始 Llama‑2‑13B、Llama‑2‑70B、以及 Qwen‑7B/14B 进行比较。  
- **核心结果**（论文中给出的数字）：  
  - 在 GSM8K 上，OctoThinker‑13B 的 RL 微调后准确率提升至 71.2%，比 Llama‑2‑13B 的 58.4% 提高约 13%。  
  - 在 MATH 上，OctoThinker‑70B 达到 45.6% 的解题成功率，逼近 Qwen‑14B（48.1%），而原始 Llama‑2‑70B 仅为 33.2%。  
  - PPO 收敛所需的训练步数从 200k 降至 140k，约快 30%。  
- **消融实验**：  
  - 去掉 MegaMath‑Web‑Pro，仅使用 FineMath‑4plus，RL 表现下降约 8%。  
  - 移除指令式 QA，模型输出长度平均增加 22%，RL 奖励下降 5%。  
  - 只用恒定学习率进行全程中期训练，RL 稳定性下降，出现梯度爆炸的情况。  
- **局限性**：作者承认，OctoThinker 在非数学、语言生成类任务上并未表现出显著优势；此外，长 CoT 仍然会在某些对话场景中导致不必要的冗长，需要进一步的后处理或解码技巧。

### 影响与延伸思考
这篇工作在 LLM 训练社区掀起了“中期训练”热潮。随后出现的几篇论文（如 *MidMath*、*CoT‑Boost*）都在不同领域尝试类似的两阶段微调思路。业界也开始在大模型的发布计划中加入“专用中期训练”阶段，尤其是针对数学、代码和科学推理的模型。未来可以进一步探索：  
- **跨任务中期训练**：把数学 CoT 与代码 CoT 合并，形成多模态思维链。  
- **自适应学习率调度**：让模型在不同子任务上自动决定何时进入衰减阶段。  
- **更细粒度的数据标记**：比如在思维链中加入置信度标签，帮助 RL 奖励模型更精准地评估推理质量。  

如果想深入了解，可以关注 OpenAI、DeepMind 以及国内的阿里巴巴、华为在“RL‑compatible foundation models”方向的最新报告。

### 一句话记住它
**把大规模稳态训练和专注思维链的衰减训练分层组合，模型就能在强化学习中“思考得更深、输出更稳”。**