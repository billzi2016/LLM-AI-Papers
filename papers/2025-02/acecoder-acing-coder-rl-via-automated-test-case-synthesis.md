# ACECODER: Acing Coder RL via Automated Test-Case Synthesis

> **Date**：2025-02-03
> **arXiv**：https://arxiv.org/abs/2502.01718

## Abstract

Most progress in recent coder models has been driven by supervised fine-tuning (SFT), while the potential of reinforcement learning (RL) remains largely unexplored, primarily due to the lack of reliable reward data/model in the code domain. In this paper, we address this challenge by leveraging automated large-scale test-case synthesis to enhance code model training. Specifically, we design a pipeline that generates extensive (question, test-cases) pairs from existing code data. Using these test cases, we construct preference pairs based on pass rates over sampled programs to train reward models with Bradley-Terry loss. It shows an average of 10-point improvement for Llama-3.1-8B-Ins and 5-point improvement for Qwen2.5-Coder-7B-Ins through best-of-32 sampling, making the 7B model on par with 236B DeepSeek-V2.5. Furthermore, we conduct reinforcement learning with both reward models and test-case pass rewards, leading to consistent improvements across HumanEval, MBPP, BigCodeBench, and LiveCodeBench (V4). Notably, we follow the R1-style training to start from Qwen2.5-Coder-base directly and show that our RL training can improve model on HumanEval-plus by over 25\% and MBPP-plus by 6\% for merely 80 optimization steps. We believe our results highlight the huge potential of reinforcement learning in coder models.

---

# ACEcoder：通过自动化测试用例合成提升代码生成强化学习 论文详细解读

### 背景：这个问题为什么难？
代码生成模型的性能大多靠大规模的监督微调（SFT）提升，而强化学习（RL）在这块儿几乎是空白。RL 需要一个可靠的奖励信号来告诉模型“这段代码好不好”，但在代码领域，手工标注的奖励稀缺且成本高。现有的奖励往往是基于语言模型的对齐评分，或者是简单的通过率统计，容易受噪声干扰，导致 RL 训练不稳定。于是，缺少高质量、可规模化的奖励数据成为阻碍代码模型使用 RL 的根本瓶颈。

### 关键概念速览
**监督微调（SFT）**：在大模型上继续用标注好的（问题‑答案）对进行训练，让模型更贴近特定任务的输出风格。  
**强化学习（RL）**：模型在生成代码时会收到一个数值奖励，依据奖励来调整策略，类似于玩游戏时的得分系统。  
**测试用例（Test‑case）**：一段输入数据和对应的期望输出，用来检验代码是否按预期工作，就像考试的答案键。  
**偏好对（Preference Pair）**：把两个模型生成的代码放在一起比较，谁的测试通过率更高就被视为更好，用来训练奖励模型。  
**Bradley‑Terry 损失**：一种把“谁更好”这种二元比较转化为概率的方式，类似于投票选出更受欢迎的选手。  
**Best‑of‑k 采样**：让模型一次生成 k 份候选代码，挑选通过率最高的那一个作为最终答案，类似于多次尝试后挑最好的。  
**HumanEval / MBPP / BigCodeBench / LiveCodeBench**：公开的代码生成评测套件，提供题目描述和隐藏测试用例，用来衡量模型的真实编程能力。  

### 核心创新点
1. **从已有代码自动生成（问题‑测试用例）对** → 通过大规模爬取公开代码库，利用自动化工具为每段函数生成对应的输入‑输出对，形成海量的（question, test‑cases）数据集。 → 解决了奖励数据稀缺的问题，使得后续的奖励模型训练和 RL 训练都有真实的可验证信号。  
2. **基于通过率构造偏好对并训练奖励模型** → 对同一道题目，随机采样多个模型生成的代码，统计它们在自动生成的测试用例上的通过率，形成“代码 A 更好”或“代码 B 更好”的比较对。使用 Bradley‑Terry 损失训练一个专门评估代码质量的奖励模型。 → 让奖励模型不再依赖语言模型的打分，而是直接反映代码的功能正确性。  
3. **双重奖励：奖励模型 + 测试通过率** → 在 RL 训练阶段，模型同时收到奖励模型给出的分数和实际测试用例的通过率，两者相加形成最终奖励。 → 兼顾了代码的语义合理性（奖励模型）和功能正确性（通过率），提升了训练的稳健性。  
4. **极少步数的高效 RL 微调** → 直接在 Qwen2.5‑Coder‑base 上进行 R1‑style（单轮）RL 微调，仅 80 步就在 HumanEval‑plus 上提升 25% 以上。 → 展示了在大模型上进行轻量级 RL 调优的可行性，降低了算力门槛。

### 方法详解
整体框架可以划分为三大阶段：**数据合成 → 奖励模型训练 → 强化学习微调**。

1. **自动化测试用例合成**  
   - 从公开的代码仓库（如 GitHub）抽取函数实现和对应的注释/文档。  
   - 使用静态分析和符号执行工具（如 PySym、Jazzer）自动生成输入空间，并执行函数得到输出，形成 (input, output) 对。  
   - 对每个函数生成若干测试用例，确保覆盖不同分支和异常路径。这样得到的 (question, test‑cases) 对可以直接喂给后续模型。  
   - 类比：把一套数学题的答案键全部自动写出来，省去老师手工批改的工作。

2. **偏好对构造与奖励模型**  
   - 选取一个基线代码生成模型，针对每个问题采样 N（如 32）个候选代码。  
   - 将这些代码在合成的测试用例上运行，记录每个候选的通过率（通过的测试用例数 / 总数）。  
   - 两两比较候选代码的通过率，生成偏好对（A 更好，B 更差）。  
   - 用 Bradley‑Terry 损失训练一个小型的奖励网络，它接受 (question, code) 作为输入，输出一个标量分数，目标是让模型的分数与偏好对的排序一致。  
   - 关键点在于把“功能对错”直接映射为“谁更好”，而不是让语言模型去猜测代码的质量。

3. **强化学习微调**  
   - 采用 Proximal Policy Optimization（PPO）等常见的 RL 算法。  
   - 每一步生成代码后，先用奖励模型得到一个分数，再在合成的测试用例上实际运行一次，得到通过率。两者加权求和构成最终奖励。  
   - 为了降低计算开销，作者使用 **Best‑of‑k** 采样：在每个训练步骤里先生成 k=32 份候选，挑选通过率最高的那一份计算奖励，这相当于在训练时已经做了一层“筛选”。  
   - 训练只进行极少的优化步数（如 80 步），因为奖励信号已经非常强，模型很快收敛。  

**最巧妙的地方**：把自动生成的测试用例既当作奖励模型的训练标签，又直接参与 RL 的即时奖励，使得两层信号相互校准，避免了单纯语言模型评分的偏差。

### 实验与效果
- **评测套件**：HumanEval、MBPP、BigCodeBench、LiveCodeBench（V4）以及作者自行扩展的 HumanEval‑plus、MBPP‑plus。  
- **基线模型**：Llama‑3.1‑8B‑Ins、Qwen2.5‑Coder‑7B‑Ins、DeepSeek‑V2.5（236B）等。  
- **主要结果**：在 Best‑of‑32 采样下，Llama‑3.1‑8B‑Ins 的分数提升约 10 分，Qwen2.5‑Coder‑7B‑Ins 提升约 5 分，使得 7B 规模的模型在 HumanEval 上的表现接近 236B 的 DeepSeek‑V2.5。  
- **RL 微调效果**：在仅 80 步的 R1‑style 微调后，Qwen2.5‑Coder‑base 在 HumanEval‑plus 上提升超过 25%，在 MBPP‑plus 上提升 6%。所有公开基准均出现一致的正向提升。  
- **消融实验**：作者分别去掉奖励模型、去掉测试通过率奖励、以及不使用 Best‑of‑k 采样，发现去掉任意一项都会导致分数下降 3‑7 分，说明三者协同是关键。  
- **局限性**：自动生成的测试用例质量受代码库和符号执行工具的限制，复杂的 I/O（如文件、网络）仍难以覆盖；RL 训练仍需要大量 GPU 计算，虽然步数少但每步的多样本采样成本不低。  

### 影响与延伸思考
ACEcoder 的思路打开了代码生成领域使用强化学习的大门。随后出现的工作开始探索 **更高效的测试用例生成**（如基于大模型的自我调试）以及 **多模态奖励**（把代码可读性、执行效率等因素加入奖励函数）。还有研究尝试把人类提交的 PR（Pull Request）作为高质量的偏好对，进一步提升奖励模型的真实性。想继续深入，可以关注以下方向：  
- **自动化测试用例的鲁棒性提升**：如何让生成的测试覆盖更广的边界条件。  
- **低算力 RL 微调**：利用离线 RL、经验回放等技术进一步压缩计算成本。  
- **跨语言奖励模型**：把同一套测试用例映射到不同编程语言，训练通用的代码质量评估器。  

### 一句话记住它
用大规模自动生成的测试用例直接造奖励，让代码模型的强化学习像“跑单元测试”一样真实且高效。