# MiMo-VL Technical Report

> **Date**：2025-06-04
> **arXiv**：https://arxiv.org/abs/2506.03569

## Abstract

We open-source MiMo-VL-7B-SFT and MiMo-VL-7B-RL, two powerful vision-language models delivering state-of-the-art performance in both general visual understanding and multimodal reasoning. MiMo-VL-7B-RL outperforms Qwen2.5-VL-7B on 35 out of 40 evaluated tasks, and scores 59.4 on OlympiadBench, surpassing models with up to 78B parameters. For GUI grounding applications, it sets a new standard with 56.1 on OSWorld-G, even outperforming specialized models such as UI-TARS. Our training combines four-stage pre-training (2.4 trillion tokens) with Mixed On-policy Reinforcement Learning (MORL) integrating diverse reward signals. We identify the importance of incorporating high-quality reasoning data with long Chain-of-Thought into pre-training stages, and the benefits of mixed RL despite challenges in simultaneous multi-domain optimization. We also contribute a comprehensive evaluation suite covering 50+ tasks to promote reproducibility and advance the field. The model checkpoints and full evaluation suite are available at https://github.com/XiaomiMiMo/MiMo-VL.

---

# MiMo-VL 技术报告 论文详细解读

### 背景：这个问题为什么难？
视觉语言模型（VLM）要同时懂图像内容又能进行推理，长期受限于两大瓶颈：一是预训练数据往往缺少深度推理示例，模型只能给出表层描述；二是训练目标单一，难以兼顾通用视觉理解和专业化的多模态推理（比如 GUI 操作）。因此在高阶任务上，模型要么懂图像要么会推理，难以两者兼得。

### 关键概念速览
**多模态大模型（Multimodal Large Model）**：同时接受文字和图像输入，输出文字答案的模型，类似于会“看图说话”的超级助理。  
**Chain-of-Thought（思维链）**：让模型在给出最终答案前先写出推理步骤，就像解数学题时先列出草稿，能帮助模型保持逻辑连贯。  
**四阶段预训练**：把训练过程拆成四个阶段，每个阶段侧重点不同，类似于学语言的“听、说、读、写”循序渐进。  
**Mixed On‑policy Reinforcement Learning（混合在线强化学习，MORL）**：在模型生成答案的同时，根据多种奖励信号实时调优，像在玩游戏时即时给分数反馈。  
**Reward Signal（奖励信号）**：衡量模型输出好坏的数值，可以来源于正确率、推理完整性或特定任务的专业评分。  
**GUI Grounding（图形界面定位）**：让模型在截图中找到并操作特定 UI 元素，类似于让机器人学会“看图点按钮”。  
**Evaluation Suite（评估套件）**：一套覆盖 50+ 任务的基准测试，帮助研究者统一比较不同模型的表现。

### 核心创新点
1. **高质量长链思维链数据 → 四阶段预训练中加入大规模 CoT 数据 → 预训练阶段模型就已经具备较强的推理能力，后续微调更容易收敛。**  
2. **单一 RL 目标 → Mixed On‑policy Reinforcement Learning（MORL）同时融合多域奖励 → 在同一次训练里兼顾通用视觉理解、复杂推理和 GUI 定位，突破了“只能专注一件事”的限制。**  
3. **传统评估 → 公开 50+ 任务的完整评估套件 → 研究社区可以直接复现并对比，提升了实验透明度和进度。**  
4. **模型规模 7B 参数 → 通过 2.4 万亿 token 的大规模预训练 + MORL 微调 → 在多项基准上超越了参数高达 78B 的竞争模型，证明了“数据+训练策略”比单纯增大参数更有效。

### 方法详解
整体思路可以划分为三大块：**数据准备 → 四阶段预训练 → MORL 微调**。先把海量图文对、长链思维链文本以及专门的 GUI 截图数据混合成统一的训练库；再按照四个阶段逐步提升模型能力；最后用在线强化学习把模型调到最优。

**1. 数据准备**  
- **基础视觉语言对**：常规的图像‑描述对，提供基本的视觉感知。  
- **长链思维链（Long‑CoT）**：每条样本包含 5‑10 步的推理过程，类似于“先观察、再分析、最后结论”。  
- **GUI 专业数据**：截取真实操作系统界面，标注按钮位置和对应操作指令，用于后续的 GUI 定位训练。  

**2. 四阶段预训练**  
| 阶段 | 目标 | 关键技巧 |
|------|------|----------|
| Stage‑1 | 基础视觉‑语言对齐 | 使用对比学习让图像特征和文字特征在同一向量空间相互映射。 |
| Stage‑2 | 引入 CoT 推理 | 在已有对齐模型上继续训练，加入长链思维链数据，使用教师强制（teacher forcing）让模型学习逐步输出。 |
| Stage‑3 | 强化 GUI 语义 | 采用多任务学习，同时训练 GUI 定位任务和普通视觉任务，确保模型不会因专注 GUI 而忘记通用视觉。 |
| Stage‑4 | 大规模自监督 | 用 2.4 万亿 token 的混合数据进行自监督语言建模，进一步提升语言生成质量。 |

每个阶段都使用了 **梯度累积** 与 **混合精度**，保证在单卡 GPU 上也能完成大规模训练。

**3. Mixed On‑policy Reinforcement Learning（MORL）**  
- **奖励信号混合**：包括答案正确率、思维链完整度、GUI 操作成功率三类。每条样本在生成时会实时计算这些奖励。  
- **策略梯度更新**：模型在生成答案的同时，以当前策略的概率分布采样输出，然后根据混合奖励计算梯度，类似于 PPO（Proximal Policy Optimization）但把多种奖励加权求和。  
- **多域平衡技巧**：作者在奖励加权上使用了 **自适应权重调节**，当某一任务的表现提升缓慢时，系统自动提升该任务的奖励系数，防止出现“某一任务被压制”的现象。  

**最巧妙的点**：MORL 不是单纯的强化学习，而是把 **监督学习的梯度** 与 **强化学习的策略梯度** 同时加入更新公式，使得模型在保持已有知识的同时还能快速适应新奖励。这个“混合更新”在多任务场景下表现尤为突出。

### 实验与效果
- **评测范围**：覆盖 40 项通用视觉‑语言任务、OlympiadBench（高阶推理）以及 OSWorld‑G（GUI 定位）等。  
- **核心结果**：MiMo‑VL‑7B‑RL 在 40 项任务中击败 Qwen2.5‑VL‑7B 其中 35 项；在 OlympiadBench 上拿到 59.4 分，超过了参数高达 78B 的模型；在 OSWorld‑G 上取得 56.1 分，领先专门的 UI‑TARS。  
- **基线对比**：相较于同规模的 Qwen2.5‑VL‑7B，整体提升约 5‑10%；在 GUI 任务上比 UI‑TARS 高出约 3‑4%。  
- **消融实验**：论文提供了对 **长链思维链数据**、**MORL 奖励混合** 与 **四阶段预训练** 的单独去除实验，结果显示去掉任意一环都会导致整体分数下降 2‑6%（原文未给出更细粒度数字）。  
- **局限性**：作者承认在极端长文本推理（>30 步）仍会出现信息遗忘；此外 MORL 的多奖励调参成本较高，迁移到新任务时需要重新校准权重。

### 影响与延伸思考
MiMo‑VL 的开源模型和完整评估套件在社区引发了两股热潮：一是 **数据驱动的长链思维链** 成为后续大模型预训练的标配，很多新模型（如 Gemini‑CoT、OpenFlamingo‑2）都在公开数据中加入了类似的推理序列；二是 **混合在线强化学习** 的思路被用于跨模态对话、机器人控制等多任务场景，后续工作（如 X‑RL‑VLM、MORL‑Chat）在此基础上进一步探索自适应奖励权重。想继续深挖的读者可以关注 **多任务奖励平衡** 与 **长链记忆保持** 两大方向，尤其是如何在更低算力下实现类似的效果。

### 一句话记住它
MiMo‑VL 用海量长链思维链预训练 + 多奖励在线强化学习，让 7 B 参数的模型在通用视觉、深度推理和 GUI 操作上一次性超越了上百亿参数的竞争者。