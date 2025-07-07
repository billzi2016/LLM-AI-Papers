# Pre-Trained Policy Discriminators are General Reward Models

> **Date**：2025-07-07
> **arXiv**：https://arxiv.org/abs/2507.05197

## Abstract

We offer a novel perspective on reward modeling by formulating it as a policy discriminator, which quantifies the difference between two policies to generate a reward signal, guiding the training policy towards a target policy with desired behaviors. Based on this conceptual insight, we propose a scalable pre-training method named Policy Discriminative Learning (POLAR), which trains a reward model (RM) to discern identical policies and discriminate different ones. Unlike traditional reward modeling methods relying on absolute preferences, POLAR captures the relative difference between one policy and an arbitrary target policy, which is a scalable, high-level optimization objective suitable for modeling generic ranking relationships. Leveraging the POLAR pre-training paradigm, we present a series of RMs with parameter scales from 1.8B to 7B. Empirical results show that POLAR substantially outperforms traditional non-pre-trained methods, significantly enhancing RM performance. For instance, POLAR-7B could improve preference accuracy from 54.8% to 81.0% on STEM tasks and from 57.9% to 85.5% on creative writing tasks compared to SOTA baselines. POLAR also shows robust generalization capabilities in RLHF using Reinforcement Fine-tuning (RFT), providing reliable reward signals and markedly enhancing policy performance--improving LLaMa3.1-8B from an average of 47.36% to 56.33% and Qwen2.5-32B from 64.49% to 70.47% on 20 benchmarks. Moreover, scaling experiments reveal a clear power-law relationship between computation and performance, supported by linear correlation coefficients approaching 0.99. The impressive performance, strong generalization, and scaling properties suggest that POLAR is a promising direction for developing general and strong reward models.

---

# 预训练策略判别器是通用奖励模型 论文详细解读

### 背景：这个问题为什么难？
在强化学习从人类反馈（RLHF）中训练大语言模型时，奖励模型（Reward Model，RM）是关键的桥梁。传统做法需要大量标注的“绝对偏好”，即让标注者直接比较两段输出并选出更好的一段。这种方式成本高、标注噪声大，而且只能捕捉到局部的偏好，难以推广到全新任务或未见的语言风格。更根本的限制在于，现有 RM 只学会“这段更好”，而不懂“相对于某个目标策略，这段差多少”，导致在多任务、多模态的真实场景里，奖励信号往往不够稳健、可扩展。

### 关键概念速览
**奖励模型（Reward Model）**：用来给语言模型的生成结果打分的网络，类似于裁判给选手打分，帮助强化学习找到更好的策略。  
**策略（Policy）**：模型在给定输入下产生输出的行为方式，换句话说，就是模型的“决策函数”。  
**策略判别器（Policy Discriminator）**：一种二分类网络，判断两段输出是否来自同一个策略，类似于“这两段话是不是同一个人写的”。  
**相对奖励（Relative Reward）**：不是给出绝对好坏，而是衡量当前策略与目标策略之间的差距，像是比较两辆车的速度差而不是直接说谁快。  
**POLAR（Policy Discriminative Learning）**：本文提出的预训练框架，训练策略判别器来学习通用的奖励信号。  
**RLHF（Reinforcement Learning from Human Feedback）**：利用人类反馈训练强化学习模型的整体流程，奖励模型是其中的关键环节。  
**RFT（Reinforcement Fine‑tuning）**：在已有模型上进一步用强化学习微调的步骤，本文用它来验证奖励模型的实用性。  

### 核心创新点
1. **从绝对偏好到策略判别**：过去的奖励模型依赖标注者直接给出“更好”或“更差”的标签。本文把奖励模型重新定义为“策略判别器”，让模型学习判断两段输出是否来自同一策略。这样一来，奖励信号不再局限于单一比较，而是捕捉到更丰富的相对信息。  
2. **可扩展的预训练任务**：作者设计了一个大规模的自监督任务——让判别器在海量生成数据中辨认“相同策略”和“不同策略”。与需要人工标注的偏好数据不同，这种任务可以用几乎无限的合成数据进行训练，极大提升了数据规模和多样性。  
3. **从单一任务到通用奖励**：通过在多种任务（STEM、创意写作等）上统一训练，POLAR 学到的判别能力能够直接迁移到未见任务上。实验显示，同一模型在不同领域的偏好准确率提升显著，说明奖励模型已经具备了跨任务的通用性。  
4. **规模化的经验规律**：作者在 1.8 B、3 B、7 B 参数模型上做了系统的计算量 vs. 性能实验，发现两者呈现近乎线性的幂律关系（相关系数≈0.99），为以后构建更大奖励模型提供了经验依据。

### 方法详解
整体思路可以拆成三步：**数据生成 → 判别器预训练 → 迁移使用**。  
1. **数据生成**：先用若干已有语言模型（如 LLaMA、Qwen）在同一输入上生成多条候选答案。每个模型的输出视为一种“策略”。随后随机配对这些答案，标记为“同策略”（来自同一模型）或“异策略”（来自不同模型）。这一步不需要人工标注，只要有足够多的基模型即可。  
2. **判别器预训练（POLAR）**：构建一个二分类网络，输入是一对文本（或它们的嵌入），输出是“同策略”概率。训练目标是最大化正确标记的概率。这里的关键技巧是**对比学习的强化**：在同一批次里加入大量负样本，使判别器必须学会捕捉细微的风格、结构差异，而不是仅凭长度或词频区分。  
3. **转化为奖励模型**：训练好的判别器可以直接用作奖励函数。给定一个候选答案和一个“目标策略”（可以是人类偏好数据中最好的答案对应的策略），判别器输出的同策略概率越高，说明该答案与目标策略越接近，奖励也随之提升。于是，在 RLHF 的 RFT 阶段，模型会被引导去生成更接近目标策略的文本。  
**最巧妙的地方**在于把“相同策略概率”当作奖励，而不是直接学习“好坏”。这样做的好处是奖励信号天然具备相对性：即使目标策略换成别的，只要有对应的参考输出，判别器仍能给出有意义的分数。  

### 实验与效果
- **测试任务**：论文在两大类任务上评估：STEM（数学、物理等专业题目）和创意写作（故事、诗歌）。每类任务都使用公开的偏好数据集来测量奖励模型的偏好准确率。  
- **基线对比**：与未预训练的奖励模型以及最新的 SOTA 预训练方法相比，POLAR‑7B 在 STEM 上把偏好准确率从 54.8% 提升到 81.0%，在创意写作上从 57.9% 提升到 85.5%。这表明相对奖励的学习显著提升了模型对人类偏好的捕捉能力。  
- **RLHF 效果**：在实际的强化学习微调（RFT）环节，使用 POLAR‑7B 作为奖励模型，使 LLaMA‑3.1‑8B 的综合得分从 47.36% 提升到 56.33%，Qwen2.5‑32B 从 64.49% 提升到 70.47%，覆盖 20 项基准测试。  
- **消融实验**：作者分别去掉对比负样本、只使用单一基模型生成的数据、以及不使用目标策略参考进行奖励计算。结果显示，负样本对比是性能提升的主要驱动力，去掉后准确率跌回 60% 左右。  
- **局限性**：论文指出，判别器仍然依赖于生成数据的多样性；如果基模型的风格过于单一，判别器学习到的差异可能不足。此外，奖励信号的标度仍需在实际 RLHF 中手动调节，未实现完全自适应。  

### 影响与延伸思考
这篇工作打开了“奖励模型即策略判别器”的新视角，随后有几篇后续研究尝试把对比学习、元学习甚至多模态信息引入奖励模型的预训练阶段（如 **Contrastive Reward Pre‑training**、**Multimodal Policy Discriminator**），进一步提升跨语言、跨模态的通用性。对想深入的读者，可以关注以下方向：① 如何在更大规模的基模型上生成更丰富的策略集合；② 将人类偏好直接映射为目标策略的自动化方法；③ 将 POLAR 思路与人类价值对齐（AI Alignment）结合，探索更安全的奖励信号。  

### 一句话记住它
把奖励模型训练成“判断两段文字是否出自同一策略”，就能用相对差距生成通用、可扩展的奖励信号。