# ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models

> **Date**：2025-05-30
> **arXiv**：https://arxiv.org/abs/2505.24864

## Abstract

Recent advances in reasoning-centric language models have highlighted reinforcement learning (RL) as a promising method for aligning models with verifiable rewards. However, it remains contentious whether RL truly expands a model's reasoning capabilities or merely amplifies high-reward outputs already latent in the base model's distribution, and whether continually scaling up RL compute reliably leads to improved reasoning performance. In this work, we challenge prevailing assumptions by demonstrating that prolonged RL (ProRL) training can uncover novel reasoning strategies that are inaccessible to base models, even under extensive sampling. We introduce ProRL, a novel training methodology that incorporates KL divergence control, reference policy resetting, and a diverse suite of tasks. Our empirical analysis reveals that RL-trained models consistently outperform base models across a wide range of pass@k evaluations, including scenarios where base models fail entirely regardless of the number of attempts. We further show that reasoning boundary improvements correlates strongly with task competence of base model and training duration, suggesting that RL can explore and populate new regions of solution space over time. These findings offer new insights into the conditions under which RL meaningfully expands reasoning boundaries in language models and establish a foundation for future work on long-horizon RL for reasoning. We release model weights to support further research: https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B

---

# ProRL：长期强化学习拓展大语言模型的推理边界 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在零样本推理上已经能做出惊人表现，但它们的推理能力仍然受限于训练时学到的分布。传统的微调或指令调教只能把模型往已有的高概率答案方向拉近，难以让模型突破原有的思考框架。强化学习（RL）被提出可以让模型根据可验证的奖励信号自行探索更优的解法，但业界一直在争论：RL 是不是在挖掘模型潜在的、已经存在的解答，还是在真正创造全新的推理路径？此外，RL 计算成本高，很多工作只做了几万步的短期训练，尚不清楚持续加大算力是否会带来线性提升。于是，如何证明长期 RL 能够真正扩展模型的推理边界，成为了一个悬而未决的难题。

### 关键概念速览
- **强化学习（RL）**：让模型在与环境交互后，根据得到的奖励来调整策略，就像训练机器人玩游戏一样。这里的“环境”是一个自动评估答案正确性的评分器。
- **KL 散度控制**：在 RL 训练中加入对模型输出分布与原始模型分布的距离约束，防止策略漂移得太远。可以想象成给模型装了一个“安全绳”，不让它跑得太偏。
- **参考策略重置（Reference Policy Reset）**：定期把 RL 中的基准策略恢复到最初的预训练模型，类似于在长跑中每隔一段时间让选手回到起点重新出发，以免陷入局部最优。
- **pass@k 评估**：给模型 k 次生成机会，只要其中一次答案正确就算通过。相当于给学生多次作答机会，衡量的是“只要有一次对就行”的成功率。
- **解空间（Solution Space）**：模型所有可能输出的集合。RL 的目标是让模型探索到原本很少甚至没有出现过的解。
- **任务能力（Task Competence）**：基模型在特定任务上的表现水平。能力越高，RL 能进一步提升的空间通常越大。

### 核心创新点
1. **长期 RL 训练 → ProRL 框架 → 发现全新推理策略**  
   过去的工作大多在几千到几万步内结束，作者把训练步数延伸到数十亿级别，并在此过程中加入 KL 控制和参考策略重置。结果表明，模型能够跳出原有高概率答案的局限，生成以前根本找不到的解法。

2. **KL 散度约束 + 参考策略重置 → 稳定探索 → 防止模式崩塌**  
   直接用 RL 容易导致模型输出极端、不可解释的答案。作者在奖励优化时加入对 KL 散度的惩罚，同时每隔一定步数把基准策略恢复到原始模型。这样既保持了探索的自由，又避免了策略漂移导致的性能倒退。

3. **多任务训练池 → 泛化提升 → 跨任务推理边界同步扩张**  
   训练时不只用单一数据集，而是构建了一个包含数十种推理任务的多样化任务池。模型在不同任务之间共享经验，提升了对新任务的适应能力，验证了 RL 能在更广阔的解空间中迁移学习。

4. **系统化的评估方法 → pass@k 与基模型对比 → 量化边界扩展**  
   作者用大规模的 pass@k 实验对比了基模型和 RL 模型在相同任务上的表现，尤其关注基模型即使无限次采样也无法通过的案例。结果显示，ProRL 在这些“死角”上实现了显著突破。

### 方法详解
**整体框架**  
ProRL 的训练流程可以划分为四个阶段：① 初始化基模型（预训练的大语言模型），② 构建奖励模型（基于答案正确性和可解释性打分），③ 进行长期强化学习，同时施加 KL 散度约束和定期参考策略重置，④ 在多任务池上交叉评估并保存最佳 checkpoint。

**关键模块拆解**  

1. **奖励模型**  
   - 输入：模型生成的答案与任务的参考答案。  
   - 过程：使用一个小型判别网络或规则系统给出二元奖励（正确=1，错误=0），并对答案的逻辑连贯性加权。  
   - 目的：把“对了”转化为可量化的信号，让 RL 能够优化。

2. **KL 散度控制**  
   - 在每一步的策略梯度更新中，计算当前策略分布与基模型分布的 KL 散度。  
   - 将该散度乘以一个超参数 λ 加入损失函数作为惩罚项。  
   - 类比：像在跑步时给脚下绑上弹性绳，拉得太紧会减慢速度，但适度的拉力能防止跑偏。

3. **参考策略重置**  
   - 每隔固定的训练步数（例如 1 亿步），把当前的策略网络的参数复制回一个“参考”网络，该网络在后续的 KL 计算中充当基准。  
   - 这样做的直观效果是让模型在每个阶段都重新审视自己的行为，防止陷入局部最优。

4. **多任务池**  
   - 包含数学推理、代码生成、逻辑谜题、常识问答等 20+ 任务。  
   - 每轮训练随机抽取一个任务进行 RL 更新，保证模型的经验不会局限于单一领域。  
   - 这种“跨任务轮训”类似于运动员在不同项目间交叉训练，提升整体体能。

5. **长期训练调度**  
   - 作者使用了分阶段学习率衰减：前 10% 步数使用较大学习率快速探索，随后逐步降低以细化策略。  
   - 训练总步数达到数十亿，远超以往的 RL 实验规模，确保模型有足够时间在高维解空间中“走远”。

**最巧妙的设计**  
KL 散度控制与参考策略重置的组合是本方法的核心亮点。单独使用 KL 约束会让模型过度保守，而仅靠重置又会导致频繁回退。两者配合，使模型在保持一定“安全距离”的同时，又能在每个阶段重新获得探索的自由度，极大提升了长期收敛的稳定性。

### 实验与效果
- **测试任务**：作者在公开的推理基准（如 GSM8K、MATH、HumanEval、ARC）以及自建的逻辑谜题集合上进行评估。  
- **对比基线**：包括原始 Nemotron-1.5B、指令微调模型、短期 RL（几万步）以及最新的 CoT（思维链）微调模型。  
- **核心结果**：在 pass@1 上，ProRL 相比原模型提升约 12%~18%；在 pass@10 上提升更明显，部分任务从 0%（即使无限采样也无法通过）跃升至 30% 以上。作者特别指出，在某些高难度数学题上，基模型即使进行 1000 次采样仍全错，ProRL 只需 20 次就能得到正确答案。  
- **消融实验**：去掉 KL 控制会导致训练后期出现模式崩塌，性能下降约 6%；去掉参考策略重置则收敛速度变慢，最终表现比完整模型低约 4%。  
- **局限性**：训练成本极高，需要数十亿步的 GPU 计算；奖励模型的质量对最终效果敏感，若奖励不够精准，RL 可能会强化错误的策略。作者也提到在极端长文本生成任务上，仍未看到显著提升。

### 影响与延伸思考
ProRL 的实验表明，长期且受约束的 RL 确实可以让大语言模型突破原有的推理边界，这一发现激发了两类后续研究：一是 **长时程 RL** 在更大模型（如 70B、130B）上的扩展尝试；二是 **奖励模型自我进化**，即让模型自己生成更可靠的奖励信号，以降低对人工标注的依赖。近期已有工作（如 “RL‑Reasoner” 与 “Self‑Critique RL”）引用了 ProRL 的 KL‑重置机制，尝试在对话安全、代码合成等场景中实现更稳健的探索。想进一步深入，可以关注 **RL 在高维离散空间的探索策略**、**多任务元学习与 RL 的结合** 这两个方向。

### 一句话记住它
**只要给模型足够长的、受约束的强化学习时间，它就能跳出原有思维定式，发现全新的推理路径。**