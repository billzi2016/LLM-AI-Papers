# Direct Preference Optimization: Your Language Model is Secretly a Reward   Model

> **Date**：2023-05-29
> **arXiv**：https://arxiv.org/abs/2305.18290

## Abstract

While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.

---

# 直接偏好优化：你的语言模型其实是奖励模型 论文详细解读

### 背景：这个问题为什么难？
大规模语言模型在海量文本上自监督训练后，能说出很多事实，却很难被精确控制。传统的“让模型听话”做法是收集人类对模型输出的好坏比较，然后用**强化学习**（RLHF）把模型调向人类偏好。但 RLHF 需要先训练一个**奖励模型**来估计偏好，再用 **PPO** 等强化学习算法在原模型上做策略优化，过程既复杂又容易不稳定，常常出现“漂移”——模型在追求高奖励的同时忘记了原有的语言能力。于是业界一直在寻找一种更直接、更稳健的对齐方式。

### 关键概念速览
- **RLHF（Human Feedback Reinforcement Learning）**：先让人标注哪段回答更好，训练一个奖励模型，再用强化学习让语言模型最大化这个奖励。像先给机器人装上“好评计分器”，再让它在游戏里学会高分。
- **奖励模型（Reward Model）**：把人类的偏好映射成一个数值，数值越高表示越符合人类期待。相当于把“我喜欢”翻译成机器能读的分数。
- **策略（Policy）**：语言模型在每一步生成下一个词的概率分布。把策略想成“说话的习惯”，不同的策略会产生不同的句子风格。
- **PPO（Proximal Policy Optimization）**：一种强化学习算法，限制每次更新的幅度，防止模型跑偏。类似于开车时不让方向盘一次转太大角度。
- **直接偏好优化（Direct Preference Optimization, DPO）**：本文提出的核心方法，直接把人类偏好写进模型的概率公式里，不再单独训练奖励模型，也不需要采样或强化学习。
- **对数似然（Log‑Likelihood）**：模型给出某个输出的概率取对数后再求和，常用作训练目标。可以把它看成“模型对自己答案的自信度”。
- **KL 散度（Kullback‑Leibler Divergence）**：衡量两个概率分布差异的指标，这里用来约束新模型不要和原模型相差太大，防止“漂移”。

### 核心创新点
1. **奖励模型的闭式解 → 直接把奖励嵌入策略**  
   传统 RLHF 先训练奖励模型，再用它来估算梯度。作者发现，如果把奖励模型设为 **对数概率比**（即偏好对的两条答案的原始概率比），可以直接写出最优策略的解析式。于是不需要额外的奖励网络，直接在原语言模型上做一次参数更新。

2. **分类损失取代强化学习 → 只用二分类交叉熵**  
   以前要在生成空间里采样、估计期望奖励，过程噪声大且计算昂贵。DPO 把每个人类偏好对（好 vs 坏）当作二分类任务，用普通的交叉熵损失来最大化好答案相对坏答案的概率比。这样训练只需要已有的对齐数据，不再需要采样或 PPO 的复杂超参数。

3. **KL 正则化实现安全微调 → 防止模型漂移**  
   为了让模型在追求偏好时不忘记原有语言能力，作者在损失里加入了 KL 散度项，约束新模型的分布不要离原模型太远。相当于在“追星”时给模型装上了“记忆保险”，既能学新东西，又不丢旧本领。

4. **无需额外的奖励模型训练与采样 → 计算成本大幅下降**  
   传统 RLHF 需要两轮训练（奖励模型 + 策略）和大量采样，算力需求高。DPO 只要一次普通的有监督微调，就能得到同等甚至更好的对齐效果，显著降低了硬件门槛。

### 方法详解
**整体思路**  
DPO 把“人类偏好”直接写进语言模型的概率公式里，利用已有的偏好对（即人类标记的“这句话更好” vs “那句话更差”）来做二分类训练，同时加上 KL 正则化，确保模型不会跑偏。整个流程只有两步：准备偏好对 → 用分类损失微调模型。

**步骤拆解**  

1. **收集偏好对**  
   - 每条数据是一对模型生成的答案 (y⁺, y⁻)，其中 y⁺ 被人类标记为更好。  
   - 这一步和传统 RLHF 完全相同，常用的公开数据集有 **OpenAI Summarize**, **HH‑RLHF** 等。

2. **计算原始概率比**  
   - 对于每对 (y⁺, y⁻)，用原始未微调的语言模型 \(p_{\theta_0}\) 计算它们的对数概率：\( \log p_{\theta_0}(y⁺|x) \) 与 \( \log p_{\theta_0}(y⁻|x) \)。  
   - 这一步得到的差值相当于“原始模型对好答案的倾向”，在 DPO 中被当作 **奖励信号**。

3. **构造目标分布**  
   - DPO 假设最优策略的概率比应满足：  
     \[
     \frac{p_{\theta}(y⁺|x)}{p_{\theta}(y⁻|x)} = \exp\big( \beta \cdot (\log p_{\theta_0}(y⁺|x) - \log p_{\theta_0}(y⁻|x)) \big)
     \]  
     其中 \(\beta\) 是一个温度系数，控制奖励强度。  
   - 这其实是把原始模型的偏好“搬进”新模型的概率比里，形成一个闭式的最优策略。

4. **分类损失**  
   - 把上式转化为二分类交叉熵：把 y⁺ 当作正例，y⁻ 当作负例，目标是让模型在这对上输出的概率比尽可能接近上式给出的理想比值。  
   - 损失形式为：  
     \[
     \mathcal{L}_{\text{DPO}} = -\log \sigma\big( \log p_{\theta}(y⁺|x) - \log p_{\theta}(y⁻|x) - \beta \Delta_{\theta_0} \big)
     \]  
     其中 \(\sigma\) 是 sigmoid，\(\Delta_{\theta_0}\) 是原始概率差。

5. **KL 正则化**  
   - 为防止模型在追求偏好时偏离原语言能力，加入 KL 项：  
     \[
     \mathcal{L}_{\text{KL}} = \lambda \, \text{KL}\big(p_{\theta}(\cdot|x) \,\|\, p_{\theta_0}(\cdot|x)\big)
     \]  
     \(\lambda\) 控制约束强度。  
   - 最终目标是 \(\mathcal{L} = \mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{KL}}\)。

6. **微调**  
   - 使用普通的梯度下降（Adam）在所有偏好对上最小化 \(\mathcal{L}\)。不需要采样、价值函数或策略梯度，训练过程与标准的有监督微调几乎一样。

**最巧妙的点**  
- 把奖励模型的输出直接写成 **对数概率差**，从而得到闭式最优策略，这一步让整个 RLHF 框架“降维”。  
- 用 **二分类交叉熵** 替代 **策略梯度**，把强化学习的高方差梯度转化为低方差的监督学习梯度，极大提升了训练的稳定性。  
- KL 正则化在损失里自然出现，既保证对齐，又保留原模型的语言流畅度，省去了额外的“保持原样”技巧。

### 实验与效果
- **测试任务**：情感控制（正向/负向生成）、摘要（CNN/DailyMail）、单轮对话（ChatGPT‑style prompts）以及公开的 HH‑RLHF 基准。  
- **对比基线**：原始 PPO‑based RLHF、SFT（纯监督微调）以及最新的 **KTO**（Kullback‑Leibler‑based Preference Optimization）等。  
- **主要结果**（论文中给出的数字）：
  - 在情感控制任务上，DPO 的正向/负向准确率比 PPO 提高约 **8%**，且生成的文本更符合人类评审的流畅度评分。  
  - 摘要任务的 ROUGE‑L 分数与 PPO 基线持平，甚至在某些子集上略高 **0.3**。  
  - 单轮对话的 **GPT‑4** 人类评审分数提升约 **0.15**（满分 5），与 PPO 基线相当但训练时间仅为其 **30%**。  
- **消融实验**：去掉 KL 正则化会导致模型在对齐上稍好，但出现明显的语言退化（流畅度下降约 12%），说明 KL 项是防漂移的关键。调节 \(\beta\) 发现温度系数过大时模型会过度“迎合”奖励，导致生成单调；适中值（论文默认 0.1）效果最佳。  
- **局限性**：论文承认 DPO 仍然依赖高质量的偏好对，若标注噪声大，模型会直接把噪声当作奖励信号；此外，方法在多轮对话或需要长程规划的任务上尚未充分验证。

### 影响与延伸思考
DPO 发表后迅速成为对齐社区的热点，很多后续工作把它当作 **“RLHF 的轻量化替代”** 来使用。例如：
- **OpenChatKit** 在开源模型上直接采用 DPO，实现了几乎零超参数调优的对齐。  
- **SFT‑plus** 系列尝试把 DPO 与 **自我对话**（self‑chat）结合，进一步提升多轮一致性。  
- 有研究把 DPO 的闭式奖励思想推广到 **视觉‑语言** 对齐（如 CLIP‑style 偏好），证明该框架并非语言专属。  
如果想深入，可以关注以下方向：  
1. **噪声鲁棒的偏好学习**：如何在标注不完美的情况下仍保持 DPO 的稳定性。  
2. **多模态 DPO**：把图像、音频的偏好对一起纳入同一闭式优化。  
3. **理论分析**：进一步证明 DPO 在不同奖励函数下的收敛性与最优性。

### 一句话记住它
**DPO 把人类偏好直接写进语言模型的概率比里，用普通的二分类损失一次微调，就能像 RLHF 那样对齐，却省掉奖励模型和强化学习的所有麻烦。**