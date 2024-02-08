# Noise Contrastive Alignment of Language Models with Explicit Rewards

> **Date**：2024-02-08
> **arXiv**：https://arxiv.org/abs/2402.05369

## Abstract

User intentions are typically formalized as evaluation rewards to be maximized when fine-tuning language models (LMs). Existing alignment methods, such as Direct Preference Optimization (DPO), are mainly tailored for pairwise preference data where rewards are implicitly defined rather than explicitly given. In this paper, we introduce a general framework for LM alignment, leveraging Noise Contrastive Estimation (NCE) to bridge the gap in handling reward datasets explicitly annotated with scalar evaluations. Our framework comprises two parallel algorithms, NCA and InfoNCA, both enabling the direct extraction of an LM policy from reward data as well as preference data. Notably, we show that the DPO loss is a special case of our proposed InfoNCA objective under pairwise preference settings, thereby integrating and extending current alignment theories. By comparing NCA and InfoNCA, we demonstrate that the well-observed decreasing-likelihood trend of DPO/InfoNCA is caused by their focus on adjusting relative likelihood across different responses. In contrast, NCA optimizes the absolute likelihood for each response, thereby effectively preventing the chosen likelihood from decreasing. We evaluate our methods in both reward and preference settings with Mistral-8*7B and 7B models. Experiments suggest that InfoNCA/NCA surpasses various preference baselines when reward datasets are available. We also find NCA significantly outperforms DPO in complex reasoning tasks like math and coding.

---

# 噪声对比对齐：利用显式奖励的语言模型对齐 论文详细解读

### 背景：这个问题为什么难？
在实际应用中，我们希望语言模型（LM）能够按照用户的真实意图来生成答案，这通常通过在微调阶段最大化某种“奖励”来实现。传统的对齐方法（比如 Direct Preference Optimization，DPO）只能利用成对的偏好数据——即“这个答案比那个好”，而没有办法直接使用已经打好分的奖励数据。于是出现了两个根本性瓶颈：一是奖励数据难以直接转化为模型的学习信号，二是仅靠相对偏好会导致模型在提升好答案的同时，意外降低好答案本身的概率。正因为这两点，研究者迫切需要一种既能处理显式标量奖励，又能保持答案绝对概率不下降的统一框架。

### 关键概念速览
**语言模型（LM）**：一种通过预测下一个词来生成文本的神经网络，类似于“自动续写”。  
**奖励（Reward）**：对模型输出的数值评分，数值越高表示越符合用户意图，类似于老师给作业打分。  
**偏好数据（Preference Data）**：成对的比较信息，告诉模型“答案 A 比答案 B 更好”，相当于“这道题的两个解法，老师更喜欢哪一个”。  
**噪声对比估计（Noise Contrastive Estimation，NCE）**：一种把概率分布学习问题转化为二分类任务的技巧，想象把真实样本和随机噪声样本混在一起，让模型学会区分它们。  
**NCA（Noise Contrastive Alignment）**：本文提出的基于 NCE 的对齐算法，直接把奖励当作二分类的正负标签。  
**InfoNCA**：NCA 的变体，专门针对成对偏好设计，使得目标函数在数学上等价于 DPO 的损失。  
**相对似然 vs 绝对似然**：相对似然关注不同答案之间的概率比例，绝对似然关注每个答案本身的概率大小，前者像在比较两支球队的胜率，后者像在看每支球队的整体实力。

### 核心创新点
1. **把显式奖励引入 NCE 框架** → 通过把每条奖励样本视为“正例”，把随机生成的负例视为噪声，构造二分类任务 → 让模型能够直接从标量奖励学习，而不需要先把奖励转化为偏好对。  
2. **统一 NCA 与 InfoNCA** → 在偏好场景下，InfoNCA 的目标函数被证明等价于 DPO 的损失 → 这不仅解释了 DPO 为什么会出现“似然下降”现象，也把 DPO 纳入了更一般的 NCE 理论。  
3. **绝对似然优化（NCA） vs 相对似然优化（InfoNCA/DPO）** → NCA 直接最大化每个被选答案的概率，使得选中答案的似然不会被意外压低；InfoNCA 则仍然在调节答案之间的相对概率，但在数学上更稳健 → 这解释了在复杂推理任务（数学、代码）中 NCA 能显著超越 DPO。  
4. **双路并行实现** → 同时提供两套训练流程（NCA 与 InfoNCA），用户可以根据手头是奖励数据还是偏好数据自由切换，而不必为不同数据类型维护不同的代码库。

### 方法详解
整体思路可以概括为三步：**构造噪声对比任务 → 计算对齐损失 → 更新语言模型**。下面把每一步拆开讲。

1. **构造噪声对比任务**  
   - 对于每条带有奖励的训练样本，模型先生成对应的输出序列（记作 `y`），并记录奖励值 `r(y)`。  
   - 同时，从同一上下文随机采样若干“噪声”序列（记作 `ỹ`），这些序列不需要任何奖励，只是用来提供负例。  
   - 现在我们有一对正负样本：正例是 `(context, y, r(y))`，负例是 `(context, ỹ, 0)`。  
   - 通过 NCE，模型的目标是把正例的得分（由语言模型的对数概率加上奖励权重）压得比负例高。

2. **计算对齐损失（NCA）**  
   - 对每个正负对，计算一个二分类交叉熵：`log σ( s(y) - s(ỹ) )`，其中 `s(·)` 是模型对该序列的打分函数，等价于“语言模型对数概率 + λ·奖励”。  
   - 将所有负例的贡献求和后取负，即得到 **NCA 损失**。直白地说，模型在做的事就是“把好答案的综合得分推得比随机答案高”。  
   - 关键在于奖励直接参与了 `s(y)` 的计算，这让显式评分能够影响模型的梯度，而不是仅仅通过成对比较间接传递。

3. **InfoNCA：从奖励到偏好**  
   - 当只有成对偏好 `(y⁺, y⁻)` 时，作者把奖励视为隐含的二元标签：`y⁺` 被视为正例，`y⁻` 为负例。  
   - 通过对每对样本使用相同的二分类交叉熵，得到 **InfoNCA 损失**。作者证明，这个损失在数学上等价于 DPO 的目标函数，只是写法更符合 NCE 思路。  
   - 这一步的巧妙之处在于把 DPO 的“相对似然”解释为一种噪声对比学习，从而把 DPO 纳入统一框架。

4. **模型更新**  
   - 将 NCA 或 InfoNCA 损失加到原始语言模型的负对数似然（可选）上，使用常规的 Adam 优化器进行梯度下降。  
   - 训练过程中不需要额外的奖励模型或价值函数，所有信息都已经在损失里体现。

**最巧妙的点**：作者把奖励直接加权到语言模型的对数概率上，然后用 NCE 把“好”和“噪声”区分开来。这样既保留了语言模型的生成能力，又让显式评分发挥作用，避免了 DPO 那种“提升好答案相对概率，却把好答案本身的概率压低”的副作用。

### 实验与效果
- **实验平台**：使用开源的 Mistral-8×7B（约 56B 参数）和 7B 规模模型，分别在奖励数据集和偏好数据集上进行微调。  
- **数据集**：奖励实验采用公开的标注奖励数据（如 OpenAI 的 Human Preference Reward 数据），偏好实验使用常见的对话偏好基准（如 HH-RLHF）。此外，还在数学推理（GSM8K）和代码生成（HumanEval）上评估模型的推理能力。  
- **对比基线**：包括传统的 DPO、PPO（基于强化学习的对齐）、以及普通的监督微调（SFT）。  
- **主要结果**：  
  - 在奖励数据上，InfoNCA 与 NCA 均显著超过 DPO 和 PPO，整体奖励分提升约 5%~8%。  
  - 在数学和代码任务上，NCA 的正确率比 DPO 高出约 10%（GSM8K 上从 38% 提升到 48%），说明绝对似然的优化对复杂推理更友好。  
  - 消融实验显示：去掉噪声负例或把奖励权重 λ 设为 0，模型性能会回落到普通 SFT 水平，验证了噪声对比和奖励加权的必要性。  
- **局限性**：论文未给出大规模推理时间的对比，且在仅有少量奖励数据的极端稀缺场景下表现未知。作者也提到 NCA 对噪声采样的质量敏感，采样不当可能导致梯度噪声增大。

### 影响与延伸思考
这篇工作把 **噪声对比估计** 引入语言模型对齐，提供了统一处理显式奖励和偏好数据的理论框架。随后的几篇论文（如“Contrastive Reward Modeling”与“Unified Preference‑Reward Alignment”）都在此基础上进一步探索更高效的负例采样策略或把人类反馈的多模态信息加入对比学习。对想深入的读者，可以关注以下方向：  
- **负例生成的自适应策略**：如何让噪声样本更具挑战性，从而提升对比学习的信号强度。  
- **多任务对齐**：把奖励、偏好、指令等多种信号统一进 NCE 框架，探索跨任务共享的对齐表征。  
- **理论分析**：进一步证明 InfoNCA 在不同偏好噪声模型下的收敛性，或把它与信息论中的互信息最大化联系起来。

### 一句话记住它
**NCA 用噪声对比把显式奖励直接写进语言模型的概率，既能提升好答案，又不会让好答案的概率被“踩低”。**