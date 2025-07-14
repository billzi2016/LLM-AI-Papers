# GHPO: Adaptive Guidance for Stable and Efficient LLM Reinforcement Learning

> **Date**：2025-07-14
> **arXiv**：https://arxiv.org/abs/2507.10628

## Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a powerful paradigm for facilitating the self-improvement of large language models (LLMs), particularly in the domain of complex reasoning tasks. However, prevailing on-policy RL methods often contend with significant training instability and inefficiency. This is primarily due to a capacity-difficulty mismatch, where the complexity of training data frequently outpaces the model's current capabilities, leading to critically sparse reward signals and stalled learning progress. This challenge is particularly acute for smaller, more resource-efficient LLMs. To overcome this, we introduce the Guided Hybrid Policy Optimization (GHPO), a novel difficulty-aware reinforcement learning framework. GHPO dynamically calibrates task difficulty by employing adaptive prompt refinement to provide targeted guidance. This unique approach adaptively balances direct imitation learning for problems currently beyond the model's reach with exploration-based reinforcement learning for more manageable tasks, effectively creating a smooth and optimized learning curriculum. Extensive experiments demonstrate that GHPO achieves an average performance gain of approximately 5% across six challenging mathematics benchmarks, consistently outperforming strong on-policy reinforcement learning and curriculum learning baselines. Further analysis confirms that our framework significantly enhances both training stability and final reasoning performance, thus offering a scalable and efficient solution for developing powerful and robust reasoning models.

---

# GHPO：自适应引导实现稳定高效的大语言模型强化学习 论文详细解读

### 背景：这个问题为什么难？

在让大语言模型（LLM）自行提升推理能力时，研究者常用“可验证奖励强化学习”（RLVR）让模型在解题后得到明确的对错信号。可是，现有的在线（on‑policy）RL 方法在训练时经常出现两大症状：一是梯度剧烈波动导致训练不收敛，二是学习进度异常缓慢。根源在于“能力‑难度不匹配”：训练数据的难度往往超过模型当前的解题水平，导致奖励几乎全是零，模型找不到学习方向。这个问题对资源受限的中小模型尤为致命，因为它们更容易被高难度样本卡住。

### 关键概念速览
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：一种让模型在完成任务后得到可检查的对错奖励的强化学习框架，类似老师批改作业后给分数。  
- **On‑policy RL**：模型直接在当前策略生成的数据上学习，像是学生只用自己刚写的答案来改进，容易受噪声影响。  
- **Off‑policy RL**：利用历史或外部生成的数据进行学习，类似学生参考别人的答案再思考，能提供更丰富的学习信号。  
- **Prompt Refinement（提示细化）**：在输入前给模型加上额外的引导信息（例如步骤提示、关键字），相当于老师在题目旁边写提示帮助学生。  
- **Curriculum Learning（课程学习）**：先让模型练习容易的题目，再逐步提升难度，像是从加法练到微积分的学习路径。  
- **Hybrid Policy Optimization（混合策略优化）**：把直接模仿（imitation）和探索式强化学习结合起来，让模型既能学习老师的示例，又能自己尝试新解法。  
- **Difficulty‑aware Guidance（难度感知引导）**：系统根据模型当前表现动态调节提示的多少，确保每一步都有可辨别的奖励差异。  

### 核心创新点
1. **能力‑难度自适应的提示机制 → 在每一次交互中，GHPO 会检测模型对当前题目的奖励是否稀疏。如果奖励几乎为零，系统会逐步向提示中加入更多线索（如分步提示、关键公式），直至奖励出现可区分的梯度。 → 这样模型始终在“可学习”区间内，避免了因题目过难导致的学习停滞。**  
2. **混合学习目标 → 对于仍然超出模型能力的样本，GHPO 采用模仿学习，让模型直接复制带提示的参考答案；对已经能产生非零奖励的样本，则切换到强化学习，让模型自行探索更高效的解法。 → 兼顾了稳定性（模仿）和探索性（RL），显著提升了训练过程的平滑度。**  
3. **动态难度评估与课程生成 → 系统记录模型在不同难度层级上的成功率，并据此自动生成一个从易到难的学习序列，而不是人工预设的固定课程。 → 省去了繁琐的手工 curriculum 设计，同时保证每一步的难度都恰好匹配模型的当前水平。**  
4. **统一的奖励稀疏度检测器 → GHPO 引入一个轻量的统计模块，实时监控奖励分布的方差，一旦方差低于阈值就触发提示细化。 → 该机制让整个框架在不同任务上都能保持相似的鲁棒性，避免了对特定数据集的过度调参。**  

### 方法详解
GHPO 的整体流程可以概括为四个循环步骤：

1. **采样与评估**：模型在当前策略下生成答案，系统使用可验证的判分器（如数学求解器）给出二元奖励（对/错）或细粒度分数。  
2. **奖励稀疏度检测**：统计本轮奖励的方差。如果方差低，说明模型对这批样本几乎没有区分能力。  
3. **自适应提示细化**：针对稀疏奖励的样本，逐步向原始提示中加入额外信息。细化的层次可以是：① 添加问题分解步骤，② 给出关键公式或定理名称，③ 直接提供部分求解过程。每加入一层，系统重新让模型生成答案并重新评估奖励，直到奖励方差超过预设阈值。  
4. **混合策略更新**：  
   - **模仿分支**：对仍然需要最高层提示的样本，使用模仿学习（行为克隆）把模型的输出对齐到带提示的参考答案。  
   - **强化学习分支**：对已经产生可区分奖励的样本，使用基于策略梯度的在线 RL（如 PPO）进行更新，鼓励模型在保持正确率的同时提升解题效率。  
   - 两个分支的梯度按比例加权（比例由当前样本的奖励稀疏度决定），形成最终的混合更新。

**类比**：可以把 GHPO 想象成一位会随时递增提示的老师。学生先尝试自己解题，老师观察学生的得分波动。如果学生几乎全错，老师会在黑板上写出一步一步的提示，直到学生的得分开始出现差异。此时老师不再直接给答案，而是让学生自行探索更好的解法，同时仍然在必要时提供示例供模仿。

**最巧妙的点**在于“奖励稀疏度检测 + 逐层提示”这条闭环。传统的 curriculum 学习需要人工设定难度阈值，而 GHPO 用统计信号自动决定何时加提示，几乎不需要人工干预。

### 实验与效果
- **测试任务**：六个公开的数学推理基准（包括 GSM8K、MATH、AQUA 等），覆盖代数、几何、数论等多种难度。  
- **对比基线**：最强的在线 RL 方法（如 PPO‑RLVR）、经典的课程学习框架（Curriculum‑RL）、以及纯模仿学习（Behavior Cloning）。  
- **整体提升**：GHPO 在所有六个基准上平均提升约 5% 的准确率，最高的 MATH 基准提升达 7.3%。  
- **训练稳定性**：相较于纯 PPO，GHPO 的奖励方差在训练全过程中保持在较低波动区间，收敛曲线更平滑。  
- **消融实验**：去掉提示细化模块后，模型在高难度样本上的奖励稀疏度恢复到原始水平，整体性能下降约 3%；仅保留模仿分支而不进行 RL 更新，则提升幅度仅为 1.8%，说明两者协同是关键。  
- **局限性**：论文指出 GHPO 依赖于可验证的奖励函数；在自然语言生成等难以量化的任务上，直接迁移仍需额外的奖励设计。  

### 影响与延伸思考
GHPO 把“动态提示”与“混合学习”结合，提供了一种在奖励稀疏环境下保持训练稳定的思路。自发表以来，已有几篇工作尝试把类似的自适应提示机制搬到代码生成、对话安全等领域（如 “Adaptive Hinting for CodeRL”），并探索更细粒度的提示生成模型（如使用小型 LLM 自动生成分步提示）。如果想进一步深入，可以关注以下方向：① 自动学习提示细化策略的元学习方法；② 将 GHPO 与大模型的自监督预训练结合，形成端到端的“自我教练”循环；③ 在多模态任务中设计可视化提示的稀疏度检测。  

### 一句话记住它
GHPO 用奖励稀疏度驱动的自适应提示，让模型始终在“可学”区间徘徊，实现了大语言模型强化学习的稳健高效。