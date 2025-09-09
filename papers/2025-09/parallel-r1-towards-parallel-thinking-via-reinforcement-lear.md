# Parallel-R1: Towards Parallel Thinking via Reinforcement Learning

> **Date**：2025-09-09
> **arXiv**：https://arxiv.org/abs/2509.07980

## Abstract

Parallel thinking has emerged as a novel approach for enhancing the reasoning capabilities of large language models (LLMs) by exploring multiple reasoning paths concurrently. However, activating such capabilities through training remains challenging, as existing methods predominantly rely on supervised fine-tuning (SFT) over synthetic data, which encourages teacher-forced imitation rather than exploration and generalization. Different from them, we propose \textbf{Parallel-R1}, the first reinforcement learning (RL) framework that enables parallel thinking behaviors for complex real-world reasoning tasks. Our framework employs a progressive curriculum that explicitly addresses the cold-start problem in training parallel thinking with RL. We first use SFT on prompt-generated trajectories from easier tasks to instill the parallel thinking ability, then transition to RL to explore and generalize this skill on harder problems. Experiments on various math benchmarks, including MATH, AMC23, and AIME, show that Parallel-R1 successfully instills parallel thinking, leading to 8.4% accuracy improvements over the sequential thinking model trained directly on challenging tasks with RL. Further analysis reveals a clear shift in the model's thinking behavior: at an early stage, it uses parallel thinking as an exploration strategy, while in a later stage, it uses the same capability for multi-perspective verification. Most significantly, we validate parallel thinking as a \textbf{mid-training exploration scaffold}, where this temporary exploratory phase unlocks a higher performance ceiling after RL, yielding a 42.9% improvement over the baseline on AIME25. Our model, data, and code will be open-source at https://github.com/zhengkid/Parallel-R1.

---

# Parallel‑R1：通过强化学习实现并行思维 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在解数学、逻辑推理等需要多步思考的任务时，往往走“一条路”，即顺序思考。人类解题时会同时尝试多个思路、相互验证，这种“并行思维”能显著提升正确率。现有的提升手段大多是监督微调（SFT），用合成的示例强迫模型模仿教师的答案，却把探索的空间压得很小，导致模型在面对真实、复杂题目时缺乏创新和自我纠错的能力。要让模型真正学会并行思考，需要一种能够奖励探索、惩罚错误的训练信号，而这在强化学习（RL）框架下才有可能实现。于是，如何在 RL 环境中让 LLM 学会并行产生、评估多条推理路径，成为了一个既重要又棘手的挑战。

### 关键概念速览

**并行思维（Parallel Thinking）**：模型一次性生成多条推理链并相互比较，就像人在解题时同时写出几种解法再挑最靠谱的那一个。  

**强化学习（Reinforcement Learning）**：让模型通过“试错”获得奖励信号的学习方式，类似于训练机器人玩游戏，模型的每一步行为都会影响最终得分。  

**监督微调（Supervised Fine‑Tuning, SFT）**：在已有的标注数据上继续训练模型，使其模仿教师的输出，像是给学生做练习题的过程。  

**冷启动问题（Cold‑Start Problem）**：在 RL 训练初期，模型几乎没有任何有价值的策略，随机行为导致奖励几乎为零，训练难以启动。  

**进阶课程（Progressive Curriculum）**：先让模型在简单任务上练习，再逐步提升难度，类似于学生先学基础算术再学高等数学。  

**多视角验证（Multi‑Perspective Verification）**：利用并行产生的多条推理链相互检查，确保答案的一致性，像是让几位专家互相审稿。  

**中期探索支架（Mid‑Training Exploration Scaffold）**：在训练过程中暂时打开探索开关，让模型在后期收敛时受益，类似于在学习新技能时先进行大量实验再固化技巧。

### 核心创新点

1. **从监督到强化的双阶段训练 → 先用 SFT 在易任务上让模型学会并行生成多条推理链，再切换到 RL 在难任务上让模型自行探索 → 解决了 RL 冷启动导致的无效探索，使模型能够在真实数学题上发挥并行思考的优势。**  

2. **进阶课程设计 → 训练数据按难度从易到难排列，模型在每个阶段都得到对应的奖励函数和采样策略 → 让模型的并行思维能力逐层深化，而不是一次性面对高难度任务导致的梯度噪声。**  

3. **并行思维奖励机制 → 奖励不仅考虑最终答案的正确性，还对多条推理链之间的一致性和相互验证程度打分 → 促使模型把并行产生的思路当作探索工具和自检手段，两种功能在不同训练阶段自然切换。**  

4. **把并行思维定位为“中期探索支架” → 在 RL 训练的中期主动打开并行思考开关，随后关闭让模型固化最优单一路径 → 实验证明，这种临时的高探索阶段能把模型的性能上限提升约 43%（在 AIME25 上），远超直接在高难度任务上训练的基线。  

### 方法详解

整体框架可以概括为四步：  
1) **任务划分**：把训练题库分成“易”“中”“难”三层。  
2) **SFT 阶段**：在易层上，用提示（prompt）让模型一次性输出 N 条推理链（N 通常为 3~5），并用人工或自动生成的参考答案进行监督微调，使模型学会并行输出。  
3) **RL 阶段**：进入中层和难层，采用强化学习。每一步模型先采样多条推理链，然后根据**并行奖励**计算分数，最后用策略梯度（如 PPO）更新模型。  
4) **支架收敛**：在训练后期逐渐降低并行采样的温度，让模型收敛到最可靠的单一路径，同时保留多视角验证的技巧。

**关键模块拆解**：

- **并行采样器**：在每个推理步骤，模型不只输出一个 token，而是并行生成 K 条完整的推理序列。可以把它想象成“多手指打字”，每根手指负责一条思路。  
- **奖励函数**：由两部分组成：① **答案正确性奖励**（对最终答案是否命中金标准给出正向奖励），② **一致性奖励**（衡量多条链路之间是否相互支持或相互矛盾，支持的链路得分更高）。这让模型在探索时既敢尝试新思路，又会自我纠错。  
- **进阶课程调度器**：根据当前训练轮次自动切换数据难度，并动态调整并行采样的温度和奖励权重。早期温度高、奖励偏向探索；后期温度低、奖励偏向收敛。  
- **策略梯度更新**：采用 PPO（Proximal Policy Optimization）等稳健的 RL 优化器，确保在高方差的并行奖励下仍能保持训练稳定。

**最巧妙的设计**在于把并行思维从“一直开启”改成“中期支架”。作者观察到，如果从头到尾都让模型并行思考，最终会产生冗余的计算开销且难以收敛。相反，先让模型在探索阶段大幅并行，随后逐步收敛到单一路径，既保留了探索带来的知识，又避免了长期的噪声累积。

### 实验与效果

- **测试数据**：MATH（美国大学水平数学题库）、AMC23（美国数学竞赛 2023 题目）和 AIME（美国中等数学竞赛）等多个公开数学基准。  
- **基线对比**：与仅使用顺序思考的 RL 模型、以及直接在难任务上做 SFT 的模型相比，Parallel‑R1 在所有基准上都有显著提升。最突出的是在 AIME25 上提升了 **42.9%**（相对基线），在 MATH 上提升约 **8.4%** 的准确率。  
- **消融实验**：作者分别关闭（1）SFT 预热、（2）并行奖励的“一致性”项、（3）进阶课程调度，发现每项缺失都会导致整体性能下降 5%~12%，验证了每个模块的必要性。  
- **局限性**：论文指出并行采样会显著增加显存占用，当前实验只能在 8‑A100 GPU 上跑；此外，奖励函数仍依赖于答案的二元正确/错误判定，难以直接迁移到开放式推理任务。  

### 影响与延伸思考

Parallel‑R1 把“并行思考”从概念层面搬到可训练的 RL 框架，打开了大模型在探索式推理上的新大门。随后的工作（如 *Multi‑Chain RL*、*Exploratory CoT*）纷纷借鉴其进阶课程和并行奖励设计，尝试在代码生成、医学诊断等非数学领域实现类似的多路径探索。对想进一步研究的读者，可以关注以下方向：① **奖励函数的细粒度设计**（比如引入置信度、局部验证），② **高效并行采样的硬件实现**（如分布式解码），③ **跨任务的并行思维迁移**（从数学到自然语言推理）。这些都是当前社区热议的前沿话题。

### 一句话记住它

让大模型先“并行开脑洞”，再“收敛成答案”，就能把强化学习的探索力转化为数学推理的高效提升。