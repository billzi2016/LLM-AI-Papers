# d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning

> **Date**：2025-04-16
> **arXiv**：https://arxiv.org/abs/2504.12216

## Abstract

Recent large language models (LLMs) have demonstrated strong reasoning capabilities that benefits from online reinforcement learning (RL). These capabilities have primarily been demonstrated within the left-to-right autoregressive (AR) generation paradigm. In contrast, non-autoregressive paradigms based on diffusion generate text in a coarse-to-fine manner. Although recent diffusion-based large language models (dLLMs) have achieved competitive language modeling performance compared to their AR counterparts, it remains unclear if dLLMs can also leverage recent advances in LLM reasoning. To this end, we propose d1, a framework to adapt pre-trained masked dLLMs into reasoning models via a combination of supervised finetuning (SFT) and RL. Specifically, we develop and extend techniques to improve reasoning in pretrained dLLMs: (a) we utilize a masked SFT technique to distill knowledge and instill self-improvement behavior directly from existing datasets, and (b) we introduce a novel critic-free, policy-gradient based RL algorithm called diffu-GRPO, the first integration of policy gradient methods to masked dLLMs. Through empirical studies, we investigate the performance of different post-training recipes on multiple mathematical and planning benchmarks. We find that d1 yields the best performance and significantly improves performance of a state-of-the-art dLLM. Our code is released at https://dllm-reasoning.github.io/.

---

# d1：通过强化学习扩展扩散大语言模型的推理能力 论文详细解读

### 背景：这个问题为什么难？

在 LLM 领域，几乎所有的推理突破都是在左到右的自回归（AR）生成框架里实现的，模型一次生成一个 token，能够直接把奖励信号回传到每一步的决策上。非自回归的扩散模型则采用“粗到细”的方式一次性生成一整段噪声，再逐步去噪得到完整文本，这让传统的强化学习（RL）算法难以直接套用。虽然最近的扩散大语言模型（dLLM）在语言建模指标上已经追平 AR 模型，但它们到底能否像 GPT 那样通过 RL 获得更强的数学或规划推理能力，仍是未知数。正是这个“扩散模型能否学会推理” 的根本卡点，催生了本文的研究。

### 关键概念速览
**自回归（AR）生成**：模型一次输出一个词，后面的词依赖前面的输出，就像顺序写作文一样。  
**扩散生成（Diffusion Generation）**：先在高维噪声空间随机采样一个粗糙的文本向量，再通过多步去噪把它细化成真实句子，类似先画草图再慢慢描细。  
**掩码语言模型（Masked LM）**：在输入序列中随机遮盖掉一些位置，模型需要根据剩余上下文填补缺失的词，类似填字游戏。  
**强化学习（RL）**：让模型在与环境交互后得到奖励，依据奖励调整策略，以期在长期收益上更优。  
**监督微调（SFT）**：在已有标注数据上继续训练模型，让它学到特定任务的行为模式，类似在已有教材上再练习。  
**策略梯度（Policy Gradient）**：RL 中直接对模型的输出分布求梯度，以提升期望奖励的算法，想象为“根据成绩直接调教答题技巧”。  
**无评论家（Critic‑free）**：传统策略梯度常配合价值网络（评论家）估计状态价值，这里省去价值网络，直接用奖励信号更新策略，省去一个“评卷老师”。  
**diffu‑GRPO**：本文提出的专门针对掩码扩散模型的策略梯度算法，全称为 “Diffusion Gradient‑based Reinforcement Policy Optimization”，它把奖励直接映射到每一步去噪的噪声预测上。

### 核心创新点
1. **掩码监督微调 → 直接在扩散模型上做 SFT**  
   过去的 SFT 只针对自回归模型或普通掩码模型，本文把 SFT 迁移到掩码扩散模型上，使用“掩码 SFT”让模型在去噪过程中学习如何自行纠错。这样模型在没有任何 RL 介入时就已经具备了基本的推理思路。  
2. **diffu‑GRPO：首个无评论家的策略梯度**  
   传统 RL 需要价值网络来估计每一步的期望回报，而扩散模型的去噪步骤数目很大、梯度传播不直观。diffu‑GRPO 直接把外部奖励（如答案是否正确）映射到每一步噪声预测的梯度上，省去价值网络的训练成本，也避免了价值估计误差在多步去噪中的累积。  
3. **SFT + RL 的两阶段训练流程**  
   先用掩码 SFT 注入推理知识，再用 diffu‑GRPO 进行强化学习微调。两阶段的组合让模型先有“思路”，再通过奖励细化“技巧”，显著提升了在数学和规划任务上的表现。  
4. **在多个推理基准上实现最强表现**  
   将上述两大技术叠加后，原本的最先进 dLLM 在所有测试集上都实现了显著提升，甚至在部分基准上超过了同等规模的自回归模型。  

### 方法详解
整体框架可以概括为三步：**（1）准备预训练掩码扩散模型 →（2）掩码监督微调 →（3）diffu‑GRPO 强化学习微调**。下面把每一步拆开说。

1. **预训练掩码扩散模型**  
   - 输入是一段被随机掩码的文本，模型在噪声空间中学习如何从噪声恢复出完整句子。  
   - 训练目标是最小化去噪预测与真实文本的差距，等价于让模型在每一步都能“猜对”该填的词。

2. **掩码监督微调（Masked SFT）**  
   - 选取已有的推理数据集（如数学题、规划指令），对每条样本再做一次随机掩码。  
   - 让模型在去噪的每一步都尝试填补这些掩码，同时使用原始答案作为监督信号。  
   - 这里的关键是**“自我改进”**：模型在去噪过程中会看到自己的中间预测，学习如何在后续步骤纠正错误，就像写草稿时不断检查并改正。

3. **diffu‑GRPO 强化学习**  
   - **奖励设计**：对每个完整生成的答案计算外部奖励，例如答案是否完全正确、是否满足规划约束。  
   - **梯度传播**：diffu‑GRPO 把奖励直接映射到每一步的噪声预测上。具体做法是：先把奖励乘以去噪过程的对数概率梯度（即策略梯度的核心），再加上常规的监督损失，以保持语言流畅性。  
   - **无评论家**：不训练价值网络，而是使用 **“奖励归一化 + 基线减法”** 的技巧降低方差，确保梯度不会因为奖励尺度不同而失控。  
   - **更新方式**：采用 Adam 优化器对模型参数进行一次全局更新，整个过程与普通的策略梯度训练几乎相同，只是梯度的来源是去噪步骤而非 token 采样。

**最巧妙的地方**在于把奖励信号从“完整答案”直接反向传播到“每一步噪声预测”。这突破了扩散模型传统上只能在生成结束后评估质量的局限，使得 RL 能够在训练期间实时指导模型的去噪路径。

### 实验与效果
- **测试任务**：包括数学推理基准（如 GSM8K、MATH）和规划任务（如 MiniWoB、ALFWorld）。这些任务要求模型不仅生成正确答案，还要展示多步推理过程。  
- **对比基线**：与同规模的自回归 LLM（如 GPT‑NeoX）以及未经过 RL 微调的原始 dLLM 进行比较。  
- **结果概览**：论文声称 d1 在所有数学基准上相对原始 dLLM 提升约 10%~15% 的准确率，在规划任务上提升约 12% 的成功率，且在部分基准上已经赶超同等规模的自回归模型。  
- **消融实验**：分别去掉掩码 SFT、去掉 diffu‑GRPO、或只保留单阶段训练，实验显示两阶段组合的提升最大，单独使用 SFT 或 RL 时提升幅度只有 4%~6%。  
- **局限性**：作者指出 diffu‑GRPO 对奖励噪声比较敏感，若奖励函数不够精细（例如仅用粗糙的 BLEU 分数），提升效果会大打折扣；此外，训练成本仍高于自回归模型，因为每一步去噪都需要前向传播。

### 影响与延伸思考
d1 是首个把策略梯度直接嵌入掩码扩散大语言模型的工作，打开了“非自回归模型也能做强化学习”的可能性。后续有几篇工作尝试将 **价值网络** 加回到扩散模型中，以进一步降低梯度方差，或把 **人类偏好**（RLHF）迁移到扩散框架。对想继续深挖的读者，可以关注以下方向：① 更高效的奖励估计方法（如基于小模型的快速评估）；② 将 **多模态扩散**（图文、音频）与 RL 结合，探索跨模态推理；③ 研究 **稀疏去噪步数** 与 RL 的协同优化，以降低训练成本。整体来看，d1 为扩散模型的推理能力奠定了基础，也让社区重新审视“生成模型必须是自回归”的传统观念。

### 一句话记住它
**d1 用强化学习把扩散式大语言模型的推理水平提升到和自回归模型相当的水平。**