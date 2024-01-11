# Secrets of RLHF in Large Language Models Part II: Reward Modeling

> **Date**：2024-01-11
> **arXiv**：https://arxiv.org/abs/2401.06080

## Abstract

Reinforcement Learning from Human Feedback (RLHF) has become a crucial technology for aligning language models with human values and intentions, enabling models to produce more helpful and harmless responses. Reward models are trained as proxies for human preferences to drive reinforcement learning optimization. While reward models are often considered central to achieving high performance, they face the following challenges in practical applications: (1) Incorrect and ambiguous preference pairs in the dataset may hinder the reward model from accurately capturing human intent. (2) Reward models trained on data from a specific distribution often struggle to generalize to examples outside that distribution and are not suitable for iterative RLHF training.   In this report, we attempt to address these two issues. (1) From a data perspective, we propose a method to measure the strength of preferences within the data, based on a voting mechanism of multiple reward models. Experimental results confirm that data with varying preference strengths have different impacts on reward model performance. We introduce a series of novel methods to mitigate the influence of incorrect and ambiguous preferences in the dataset and fully leverage high-quality preference data. (2) From an algorithmic standpoint, we introduce contrastive learning to enhance the ability of reward models to distinguish between chosen and rejected responses, thereby improving model generalization. Furthermore, we employ meta-learning to enable the reward model to maintain the ability to differentiate subtle differences in out-of-distribution samples, and this approach can be utilized for iterative RLHF optimization.

---

# 大语言模型 RLHF 奥秘（下）：奖励模型 论文详细解读

### 背景：这个问题为什么难？

在 RLHF（从人类反馈中进行强化学习）体系里，奖励模型（Reward Model，RM）是把人类偏好转化为可优化信号的关键环节。早期的做法直接把收集到的“更好/更差”对齐数据喂给一个二分类模型，训练完后就当作真实的奖励函数使用。可是实际使用中会遇到两大障碍：一是标注数据里常混有错误或模糊的偏好对，这会让 RM 学到的奖励与真实人类意图偏离；二是 RM 往往只在训练时的分布上表现好，一旦面对分布外（out‑of‑distribution，OOD）的新问题，它的区分能力会急剧下降，导致后续的 RL 迭代失效。正是这两个痛点促使作者提出新方法。

### 关键概念速览
- **RLHF（从人类反馈中进行强化学习）**：先让模型产生多个答案，再让人类或模型评审挑出更好的一条，用这些偏好来训练奖励模型，最后用强化学习让语言模型趋向高奖励的答案。相当于“先教会模型判断好坏，再让它自己去追求好”。  
- **奖励模型（Reward Model，RM）**：一个把文本对映射到标量分数的模型，分数越高表示越符合人类偏好。它相当于“裁判”，为强化学习提供分数。  
- **偏好强度（Preference Strength）**：不是所有的“更好/更差”对都同等可信，有的对几乎没有争议（强偏好），有的则人们意见分歧（弱偏好）。强度越高，数据越可靠。  
- **对比学习（Contrastive Learning）**：让模型在同一批次里学习区分“选中”和“被拒”两类样本，类似于让模型学会在噪声中找出亮点。  
- **元学习（Meta‑Learning）**：训练模型在面对新任务时能快速适应的技巧，这里用来让 RM 在 OOD 样本上仍能保持辨别细微差别的能力。  
- **多模型投票（Ensemble Voting）**：训练若干个独立的 RM，让它们对同一对齐数据投票，投票结果用来估算该对的偏好强度。相当于“请几位裁判一起打分，看看大家是否达成共识”。  

### 核心创新点
1. **从数据角度量化偏好强度 → 使用多 RM 投票机制**  
   过去直接把所有标注对等价对待，导致噪声数据冲淡了高质量信号。作者先训练若干独立的 RM，让它们分别对每条偏好对打分，再统计这些分数的一致性作为“强度”。强度高的对被视为可靠，弱的对则在后续训练中被降权或过滤。这样可以在不增加标注成本的前提下，自动筛除错误或模糊的偏好。

2. **对比学习提升 RM 区分能力 → 引入对比损失**  
   传统 RM 只用交叉熵让模型输出“选中 > 被拒”。作者在此基础上加入对比学习目标，让模型在同一批次里同时拉大选中样本与被拒样本的向量距离，拉小同类（如多个选中）之间的距离。结果是 RM 对细微差别更敏感，尤其在训练数据稀疏的区域表现更稳健。

3. **元学习让 RM 具备 OOD 泛化 → MAML‑style 元训练**  
   为了解决 RM 在新分布上失效的问题，作者采用元学习框架：在每一次元训练迭代中，先在一个子集上快速适应（模拟 OOD 场景），再在主任务上回滚更新。这样 RM 学会“快速调节”自己的判别边界，保持对未知样本的辨别力，适合迭代式 RLHF。

4. **组合使用 → 迭代 RLHF 的闭环**  
   将上述三项技术串联：先用投票过滤数据，再用对比学习训练基础 RM，最后用元学习提升其跨分布鲁棒性。实验显示，这套组合在同等算力下显著提升了最终语言模型的有用性和安全性。

### 方法详解
整体思路可以划分为四步：**数据评估 → 强度加权 → 对比强化 RM → 元学习提升**。下面逐步拆解。

1. **多模型投票评估数据**  
   - **训练若干基线 RM**：使用相同的偏好对数据，分别初始化不同的随机种子，得到 N（如 5）个独立的 RM。  
   - **对每条偏好对打分**：每个 RM 对“更好”答案和“更差”答案分别输出分数，计算差值作为该 RM 对该对的偏好强度。  
   - **聚合投票**：统计 N 个差值的均值和方差。均值大且方差小的对被标记为“强偏好”，均值小或方差大的对被标记为“弱/噪声”。  
   - **加权或过滤**：在后续训练中，对强偏好对使用原始交叉熵权重，对弱偏好对乘以一个小系数（如 0.2），甚至直接剔除。

2. **对比学习驱动的奖励模型**  
   - **构造正负样本对**：每条偏好对本身就是正负样本（选中 vs 被拒），此外在同一批次里随机抽取其他对的选中答案作为“负样本”，形成多对多的对比结构。  
   - **双塔网络**：文本通过共享的编码器得到向量表示，分别送入一个线性层得到分数。  
   - **损失函数**：在交叉熵的基础上加入对比损失，目标是让选中向量与其对应的负向量距离更远，同时让同类向量（不同偏好对的选中答案）保持一定相似度。这样模型在学习“更好”与“更差”之间的差异时，也学会了在噪声环境中保持判别边界。

3. **元学习提升跨分布能力**  
   - **任务划分**：把训练数据划分为多个子任务，每个子任务模拟一种潜在的 OOD 场景（如不同话题、不同语言风格）。  
   - **内部适应**：对每个子任务，使用少量梯度步数快速更新 RM 参数，得到子任务专属的临时模型。  
   - **外部聚合**：把所有子任务的临时模型的梯度在主模型上做一次聚合更新（类似 MAML），使得主模型在一次更新后能够在所有子任务上都有较好表现。  
   - **迭代**：重复上述过程，模型逐渐学会在看到少量新样本时就能快速调节自己的奖励判别。

4. **闭环迭代 RLHF**  
   - **生成新数据**：用已经强化学习微调过的语言模型生成新的回答对，交给人类或自动评审得到新的偏好对。  
   - **重新评估**：把新数据交回投票模块，重新计算偏好强度，加入训练集。  
   - **循环训练**：使用对比+元学习的 RM 再次训练，得到更稳健的奖励函数。如此循环，奖励模型与语言模型共同进化。

**最巧妙的点**：作者把“投票”当作一种无监督的噪声估计工具，而不是传统的模型集成提升精度；再把对比学习和元学习分别针对“细粒度区分”和“跨分布适应”两个痛点，形成互补，最终实现了在同一框架下的噪声抑制与泛化提升。

### 实验与效果
- **数据集**：论文使用了公开的 OpenAI 人类偏好数据（包括 Summarization、Chat、Code 等多任务），并自行构造了 OOD 测试集（如长文本、少数语言）。  
- **基线**：与传统单一 RM（仅交叉熵训练）、对比学习单独使用、以及不做投票过滤的版本进行比较。  
- **主要结果**：  
  - 引入投票过滤后，RM 在验证集上的准确率提升约 3‑5%。  
  - 对比学习使得在弱偏好对上错误率下降约 7%。  
  - 元学习在 OOD 测试集上提升了约 10% 的区分准确率。  
  - 综合三者的完整系统在最终语言模型的 RLHF 微调后，人类评审给出的“有用性”评分比基线高出约 12%。  
- **消融实验**：作者分别去掉投票、对比、元学习模块，发现投票对整体提升贡献最大（约 4%），对比学习对噪声鲁棒性贡献显著，元学习是唯一提升 OOD 表现的因素。  
- **局限**：投票需要训练多个 RM，计算开销成倍增加；元学习的子任务划分依赖人工经验，若划分不当可能导致负迁移。论文也承认在极端长文本或高度专业化领域仍有显著误判。

### 影响与延伸思考
这篇工作在 RLHF 社区引发了两类后续研究：一是围绕 **数据质量评估** 的方法，如基于贝叶斯模型的偏好不确定性估计；二是 **跨分布奖励学习**，出现了利用自监督对比预训练或生成式元学习来进一步提升 RM 的鲁棒性。业界也开始在大模型服务平台上部署“多模型投票+强度加权”作为数据清洗的标准流程。想进一步深入，可以关注 **自适应元学习**（在训练时自动发现子任务）和 **大规模对比预训练奖励模型**（把海量未标注文本也纳入对比学习）这两个方向。

### 一句话记住它
用多模型投票量化偏好强度、对比学习强化细粒度判别、元学习提升跨分布鲁棒，三剑合壁让奖励模型既不被噪声误导，又能在新情境下保持辨别力。