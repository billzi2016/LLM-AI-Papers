# Learning to Reason without External Rewards

> **Date**：2025-05-26
> **arXiv**：https://arxiv.org/abs/2505.19590

## Abstract

Training large language models (LLMs) for complex reasoning via Reinforcement Learning with Verifiable Rewards (RLVR) is effective but limited by reliance on costly, domain-specific supervision. We explore Reinforcement Learning from Internal Feedback (RLIF), a framework that enables LLMs to learn from intrinsic signals without external rewards or labeled data. We propose Intuitor, an RLIF method that uses a model's own confidence-termed self-certainty-as its sole reward signal. Intuitor replaces external rewards in Group Relative Policy Optimization (GRPO) with self-certainty scores, enabling fully unsupervised learning. Experiments demonstrate that Intuitor matches GRPO's performance on mathematical benchmarks while achieving better generalization to out-of-domain tasks like code generation, without requiring gold solutions or test cases. Our findings show that intrinsic model signals can drive effective learning across domains, offering a scalable alternative to RLVR for autonomous AI systems where verifiable rewards are unavailable. Code is available at https://github.com/sunblaze-ucb/Intuitor

---

# 无需外部奖励的推理学习 论文详细解读

### 背景：这个问题为什么难？

在让大语言模型（LLM）进行复杂推理时，业界常用“可验证奖励的强化学习”（RLVR）来把模型的输出和人工标注的正确答案对齐。可验证奖励需要高质量的金标准答案或测试用例，这在数学、代码、法律等专业领域成本极高，而且每新增一个任务几乎都要重新标注。于是模型的推理能力受限于标注的可得性和费用，难以实现真正的自我提升。

### 关键概念速览
- **大语言模型（LLM）**：能够生成自然语言的深度神经网络，像 GPT、Claude 那样通过海量文本学习语言规律。  
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：用外部、可验证的奖励信号（如答案是否正确）来指导模型的策略更新。  
- **RLIF（Reinforcement Learning from Internal Feedback）**：不依赖外部奖励，而是让模型自己产生学习信号的强化学习框架。  
- **自信度（self‑certainty）**：模型对自己输出的确信程度，用输出分布相对均匀分布的 KL 散度来量化，类似于人对答案的“把握感”。  
- **GRPO（Group Relative Policy Optimization）**：一种强化学习算法，比较同一批次中不同策略的相对表现来更新模型，避免单纯的绝对奖励波动。  
- **内在奖励（intrinsic reward）**：来源于模型内部状态或信号的奖励，而非外部标注。  
- **跨域泛化（out‑of‑domain generalization）**：模型在未见过的任务或数据上仍能保持高性能的能力。

### 核心创新点
1. **外部奖励 → 自信度奖励**  
   过去的 RLVR 需要人工提供的正确答案作为奖励；Intuitor 把模型自己的自信度直接当作奖励信号。这样模型在没有任何标注的情况下也能得到正向反馈，实现了完全无监督的推理学习。

2. **GRPO 中的奖励函数替换**  
   原始 GRPO 通过比较不同策略在外部奖励上的相对得分来更新策略。Intuitor 将这些外部得分全部换成自信度得分，使得整个优化过程只依赖模型内部信息，却仍保持 GRPO 的相对比较机制，避免了奖励尺度不一致的问题。

3. **自信度的计算方式**  
   论文提出用输出分布与均匀分布的 KL 散度来衡量自信度，这相当于测量模型“到底有多确定”。这种度量既易于计算，又能在没有标签的情况下捕捉到模型对答案的信念强弱。

4. **跨任务的学习迁移**  
   实验显示，使用自信度奖励训练的模型在数学基准上能追平 GRPO，同时在代码生成等全新任务上表现更好。说明内部信号能够帮助模型学到更通用的推理技巧，而不是仅仅记忆特定任务的奖励模式。

### 方法详解
Intuitor 的整体思路可以拆成三步：**信号提取 → 奖励构造 → 策略更新**。

1. **信号提取**  
   - 给定一个输入（如数学题），模型先生成一个概率分布（每个 token 的概率）。  
   - 计算该分布与均匀分布之间的 KL 散度，数值越大表示模型越“自信”。这一步相当于让模型自己给自己的答案打分。

2. **奖励构造**  
   - 将每一次生成的自信度视作该样本的奖励 r。  
   - 为了保留 GRPO 的相对比较特性，Intuitor 把同一批次（group）内的所有 r 按均值归一化，得到相对奖励 δr。这样即使整体自信度偏高或偏低，模型仍然只关注相对优势。

3. **策略更新（GRPO 机制）**  
   - 采用原始 GRPO 的核心公式：对每个策略（即当前模型的参数）计算优势估计 A = δr - baseline。  
   - 使用策略梯度方法对模型参数进行微调，使得产生更高自信度的输出概率被放大。  
   - 训练循环中，模型每完成一次完整的推理（例如解完一道题），就把这次的自信度加入奖励池，继续迭代。

**最巧妙的点**在于：自信度本身是模型的“内部评估”，但通过 KL 散度的方式把它量化为可比较的数值；再把它嵌入 GRPO 的相对比较框架，使得学习过程既保持了强化学习的探索性，又不需要任何外部标签。

### 实验与效果
- **测试任务**：论文在数学推理基准（如 GSM‑8K）以及代码生成任务（如 HumanEval）上做评估。  
- **对比基线**：与使用外部奖励的 GRPO、传统的监督微调以及零-shot LLM 进行比较。  
- **结果**：论文声称 Intuitor 在数学基准上达到了与 GRPO 相当的准确率，同时在代码生成任务上超出 GRPO 大约 5% 的成功率，展示了更好的跨域泛化。  
- **消融实验**：作者分别去掉 KL 散度的归一化、或直接使用原始概率最大值作为奖励，发现性能明显下降，说明相对奖励和 KL 散度的设计是关键。  
- **局限性**：自信度依赖模型的概率校准；如果模型在错误答案上过度自信，可能会强化错误的推理路径。论文也提到在极大规模模型（数百亿参数）上的实验尚未完成。

### 影响与延伸思考
Intuitor 打开了“内部奖励驱动的推理学习”这一新方向，后续有多篇工作尝试用模型的置信度、信息增益或自我纠错信号来替代人工标注，例如 Self‑Reward、Confidence‑Guided RL 等。对想进一步探索的读者，可以关注以下两个方向：  
1. **自信度校准技术**：如何让模型的概率分布更真实地反映不确定性，从而提升内部奖励的可靠性。  
2. **多模态内部奖励**：把视觉、音频等模态的自我评估融合进统一的强化学习框架，推动跨模态推理的自监督学习。

### 一句话记住它
Intuitor 证明，模型只要能量化自己的“确信感”，就能在没有任何外部标注的情况下学会可靠推理。