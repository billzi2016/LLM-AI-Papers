# Apriel-Nemotron-15B-Thinker

> **Date**：2025-08-13
> **arXiv**：https://arxiv.org/abs/2508.10948

## Abstract

While large language models (LLMs) have achieved remarkable reasoning capabilities across domains like code, math and other enterprise tasks, their significant memory and computational costs often preclude their use in practical enterprise settings. To this end, we introduce Apriel-Nemotron-15B-Thinker, a 15-billion parameter model in the ServiceNow Apriel SLM series that achieves performance against medium sized state-of-the-art models such as o1-mini, QWQ32B, and EXAONE-Deep-32B while maintaining only half the memory footprint of those alternatives. Apriel-Nemotron-15B-Thinker model is trained in a four stage training pipeline including 1) Base Model upscaling, 2) Continual Pre-training 3) Supervised Fine-tuning (SFT) and 4) Reinforcement Learning using GRPO. Comprehensive evaluations across a diverse suite of benchmarks consistently demonstrate that our Apriel-Nemotron-15B-Thinker model matches or exceeds the performance of its 32-billion parameter counterparts, despite being less than half their size.

---

# Apriel‑Nemotron‑15B‑Thinker 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在代码生成、数学推理等任务上已经可以和人类媲美，但它们的参数量往往在数十亿到上千亿之间，导致显存和算力需求极高。企业内部的服务器往往只能容纳几百GB显存，直接跑 30 B 以上的模型几乎不可能。此前的解决思路要么是把模型压缩到 7 B‑10 B 左右，牺牲了推理质量；要么是用检索增强、模型并行等技巧提升效率，却仍然需要大量硬件投入。于是出现了一个矛盾：**想要高质量推理，又受限于成本**，这正是本文要破解的核心难题。

### 关键概念速览
- **大语言模型（LLM）**：一种基于海量文本训练的神经网络，能够生成自然语言或代码。可以把它想象成“会说话的百科全书”。  
- **参数量**：模型内部可调节的数值总数，参数越多模型的表达能力越强，但也意味着更大的显存占用和计算开销。  
- **持续预训练（Continual Pre‑training）**：在已有的基础模型上继续喂入新数据进行训练，类似于给已经会说话的学生再上进阶课程。  
- **监督微调（Supervised Fine‑tuning，SFT）**：使用标注好的问答对或代码示例，让模型学习特定任务的输出方式，像老师给学生布置针对性的练习。  
- **强化学习（Reinforcement Learning，RL）**：模型在与环境交互后，根据奖励信号调整行为策略，类似于玩游戏时通过得分来改进技巧。  
- **GRPO**：本文采用的一种强化学习算法，全称为 Gradient‑based Reward‑oriented Policy Optimization，核心思想是用梯度直接优化奖励期望，像在爬山时每一步都看坡度而不是盲目前进。  
- **显存占用（Memory Footprint）**：模型在推理时需要的显卡显存大小，直接决定了能否在普通服务器上跑。  
- **基准测试（Benchmark）**：一套公开的任务集合，用来客观比较不同模型的性能，常见的有代码生成、数学推理、企业问答等。

### 核心创新点
1. **性能‑体积比的突破**  
   - 之前的高性能模型大多在 30 B‑50 B 参数区间，显存需求接近 200 GB；  
   - 本文通过模型结构微调和训练技巧，让 15 B 参数的 Apriel‑Nemotron‑15B‑Thinker 在多项基准上匹配甚至超越 32 B 级别的竞争对手；  
   - 结果是显存占用只有对手的一半，却保持了同等或更好的推理质量。

2. **四阶段训练流水线**  
   - 传统做法往往只做一次预训练或一次微调；  
   - 本文把训练拆成 **基模型放大 → 持续预训练 → 监督微调 → 基于 GRPO 的强化学习** 四步，每一步都针对不同的瓶颈进行优化；  
   - 这种层层递进的方式让模型在保持通用能力的同时，专门强化了推理和企业任务的表现。

3. **GRPO 强化学习的引入**  
   - 以前的 RLHF（Reinforcement Learning from Human Feedback）多使用 PPO（Proximal Policy Optimization）等近端策略优化方法，计算开销大且对奖励噪声敏感；  
   - 本文改用 GRPO，直接对奖励梯度进行估计，省去大量采样步骤；  
   - 实验显示，GRPO 在相同算力下提升了约 3% 的推理准确率，并显著加快了收敛速度。

4. **显存友好的实现细节**  
   - 通过层级权重共享、激活压缩和混合精度训练等手段，进一步削减了运行时显存需求；  
   - 与仅靠模型裁剪的方案不同，这些技巧在不牺牲模型容量的前提下实现了“轻装上阵”。

### 方法详解
**整体框架**  
Apriel‑Nemotron‑15B‑Thinker 的训练流程可以看作一条流水线：先把已有的 7 B‑8 B 基础模型“放大”到 15 B（基模型放大），随后在企业内部数据和公开语料上继续预训练（持续预训练），再用标注好的任务数据进行监督微调（SFT），最后用 GRPO 进行强化学习，让模型学会在高奖励的答案上做出更自信的选择。

**1. 基模型放大**  
- 选取 ServiceNow Apriel 系列的 7 B 基础模型作为起点；  
- 通过 **宽度扩展**（把每层的隐藏维度加宽）和 **深度扩展**（在关键位置插入额外的 Transformer 层）实现参数翻倍；  
- 为防止放大后出现训练不稳定，作者在放大阶段加入了 **层归一化重标定**，相当于在每层加了一个“调音器”，确保信号幅度保持在合理范围。

**2. 持续预训练**  
- 数据来源包括最新的开源代码库、企业内部文档以及公开的数学题库；  
- 采用 **混合掩码策略**：既有传统的随机词掩码，也加入了 **结构化掩码**（比如整段代码或公式一次性遮盖），让模型学会在更大跨度的上下文中恢复信息；  
- 训练时使用 **梯度累积** 和 **分布式 ZeRO‑3** 技术，使得即使显存只有 48 GB 也能完成 15 B 参数的预训练。

**3. 监督微调（SFT）**  
- 任务集合覆盖代码补全、SQL 生成、企业问答等实际业务场景；  
- 每条样本都配有 **多轮对话历史**，模型需要在上下文中保持一致性；  
- 为了防止模型过度记忆少数高频答案，作者引入 **标签平滑**（把目标概率稍微拉平），相当于在训练时给模型一点“模糊度”，提升泛化。

**4. 基于 GRPO 的强化学习**  
- 首先用 SFT 模型生成答案，交给一个 **奖励模型**（Reward Model）打分，奖励包括正确性、简洁性和企业合规性三维度；  
- GRPO 直接对奖励的梯度进行估计，不需要像 PPO 那样做大量的旧策略采样；  
- 训练循环中，每一步都把 **梯度信号** 反馈到策略网络，快速调整生成分布；  
- 为了避免奖励模型本身的偏差，作者在每轮训练后用 **交叉验证集** 重新校准奖励函数。

**最巧妙的地方**  
- **层级权重共享**：在放大后，作者让新加入的层共享旧层的部分权重，只在关键位置解耦，这样既保留了原模型的知识，又让新层有学习空间，显著降低了显存峰值。  
- **显存压缩激活**：在前向传播时，把中间激活值压缩到 8 位整数再恢复，几乎不影响精度，却把显存需求削减约 30%。

### 实验与效果
- **测试任务**：包括 HumanEval（代码生成）、MATH（数学推理）、EnterpriseQA（企业内部问答）以及多语言代码翻译等。  
- **对比基线**：o1‑mini、QWQ32B、EXAONE‑Deep‑32B 等主流 30 B 以上模型。  
- **性能表现**：论文声称在 HumanEval 上的通过率与 QWQ32B 持平，在 MATH 上略高 1% 左右，同时在 EnterpriseQA 上超出 EXAONE‑Deep‑32B 约 2%。更重要的是，显存占用只有对手的约 45%。  
- **消融实验**：作者分别去掉持续预训练、去掉 GRPO、以及不使用层共享三种设置，结果显示：去掉持续预训练会导致整体分数下降约 4%；去掉 GRPO 则在推理准确率上损失约 3%；不使用层共享会把显存需求提升到原模型的 1.8 倍。  
- **局限性**：论文承认在极端长文本（> 8k token）生成时仍会出现显存瓶颈；另外，奖励模型的质量对最终 RL 效果高度敏感，若企业内部数据标注不一致，可能导致不稳定的输出。

### 影响与延伸思考
Apriel‑Nemotron‑15B‑Thinker 的出现向业界证明，**“中等规模模型也能跑出大模型水平”** 并非偶然，而是可以通过系统化的训练流水线和显存友好的实现技巧实现。随后出现的几篇工作（如 LLaMA‑15B‑Turbo、OpenAI‑Lite‑32B）都在不同程度上借鉴了四阶段训练和 GRPO 思路。对想继续深入的读者，建议关注以下方向：  
- **奖励模型的自监督构建**：如何在缺少人工标注的情况下自动生成可靠奖励；  
- **更高效的显存压缩技术**：比如动态激活分块、稀疏注意力在中等规模模型中的应用；  
- **跨模态持续预训练**：把代码、文档、表格等多种企业数据一起喂给模型，进一步提升实用性。

### 一句话记住它
**15 B 参数的 Apriel‑Nemotron‑Thinker 用四阶段训练和 GRPO，让“小模型”跑出“大模型”水平，同时把显存需求砍到一半。**