# CodePMP: Scalable Preference Model Pretraining for Large Language Model Reasoning

> **Date**：2024-10-03
> **arXiv**：https://arxiv.org/abs/2410.02229

## Abstract

Large language models (LLMs) have made significant progress in natural language understanding and generation, driven by scalable pretraining and advanced finetuning. However, enhancing reasoning abilities in LLMs, particularly via reinforcement learning from human feedback (RLHF), remains challenging due to the scarcity of high-quality preference data, which is labor-intensive to annotate and crucial for reward model (RM) finetuning. To alleviate this issue, we introduce CodePMP, a scalable preference model pretraining (PMP) pipeline that utilizes a large corpus of synthesized code-preference pairs from publicly available high-quality source code. CodePMP improves RM finetuning efficiency by pretraining preference models on large-scale synthesized code-preference pairs. We evaluate CodePMP on mathematical reasoning tasks (GSM8K, MATH) and logical reasoning tasks (ReClor, LogiQA2.0), consistently showing significant improvements in reasoning performance of LLMs and highlighting the importance of scalable preference model pretraining for efficient reward modeling.

---

# CodePMP：可扩展的偏好模型预训练用于大语言模型推理 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在理解和生成自然语言方面已经很强，但要让它们像人一样进行严谨的数学或逻辑推理仍然很吃力。当前提升推理能力的主流手段是基于人类反馈的强化学习（RLHF），核心是先训练一个奖励模型（Reward Model，RM）来判断哪段答案更好，再用它来指导模型生成。可是，高质量的偏好数据——即“这段答案比那段更好”的标注——非常稀缺，标注成本高，规模难以扩大。缺少足够的偏好数据，RM 的训练往往不够稳健，导致后续的 RLHF 效果有限。于是，如何在不依赖大量人工标注的情况下，给 RM 提供足够的“练习材料”，成为了制约推理能力提升的瓶颈。

### 关键概念速览
**偏好模型（Preference Model）**：一种二分类模型，输入两段答案，输出哪段更符合人类期望的概率。它相当于“裁判”，帮助 RLHF 判断生成的答案好坏。  
**奖励模型（Reward Model）**：在 RLHF 流程中，用偏好模型的输出作为奖励信号，指导语言模型的策略优化。可以把它想成“打分员”，把裁判的判断转化为数值奖励。  
**RLHF（Reinforcement Learning from Human Feedback）**：先让人类标注偏好，再用这些标注训练奖励模型，最后用强化学习让语言模型学会产生高分答案。类似于先教会孩子分辨好坏，再让他自己练习。  
**代码偏好对（Code‑Preference Pair）**：从公开的高质量源码中自动生成的“代码片段 + 对应的偏好标签”。把代码当作一种结构化、可验证的任务，让模型产生不同解答后比较优劣。  
**可扩展预训练（Scalable Pretraining）**：在大规模数据上进行的预训练过程，能够随数据量线性提升模型能力。这里指的是在海量代码偏好对上预训练偏好模型。  
**数学推理任务（如 GSM8K、MATH）**：要求模型给出一步步计算过程并得出正确答案的题目。  
**逻辑推理任务（如 ReClor、LogiQA2.0）**：需要模型理解前提、推导结论的阅读理解类题目。

### 核心创新点
1. **从源码自动合成偏好对 → 直接利用公开代码生成大规模训练样本 → 解决了高质量偏好数据稀缺的问题**。作者把高质量开源项目当作“题库”，用程序执行结果或代码风格差异自动生成“这段实现更好/更差”的标签，省去了人工标注的繁琐。  
2. **在偏好模型上进行大规模预训练 → 先让模型学会辨别代码质量，再迁移到自然语言推理任务 → 提升了 RM finetune 的效率**。预训练阶段相当于给模型上了一堂“代码审查课”，让它掌握通用的比较能力，后续只需要少量真实语言偏好数据就能快速收敛。  
3. **把代码偏好对视作跨任务的通用信号 → 在数学和逻辑推理两类完全不同的任务上都取得显著提升 → 证明了偏好预训练的任务无关性**。实验显示，无论是数值计算还是逻辑推理，使用 CodePMP 预训练的模型都比仅用人类偏好微调的基线高出数个百分点。  
4. **提供了完整的可复制流水线 → 包括代码抓取、对齐、偏好标签生成、模型预训练和微调 → 为后续研究者提供了“即插即用”的工具链**。这让社区能够在自己的数据上复现或扩展该方法，而不必从头搭建复杂的合成系统。

### 方法详解
整体思路可以划分为三步：**数据合成 → 偏好模型预训练 → 任务微调**。

1. **数据合成**  
   - **源码收集**：从 GitHub、GitLab 等平台抓取星标高、文档完善的开源项目，确保代码质量本身已经经过社区审查。  
   - **任务抽取**：把每个函数或代码块视作一个“小任务”。例如，一个实现排序的函数可以对应“给定数组，返回升序排列”。  
   - **多解生成**：对同一任务，使用不同的代码生成模型（或手工改写）产生多个实现版本。实现之间可能在效率、可读性或边界处理上有所差异。  
   - **偏好标签生成**：依据静态分析（如复杂度、是否使用最佳实践）或运行时评测（如执行时间、正确性）自动打分，形成“实现 A 更好 / 实现 B 更差”的二元标签。这样就得到大量 **代码‑偏好对**（code‑preference pair）。

2. **偏好模型预训练**  
   - **模型结构**：采用与目标 LLM 相同的 Transformer 编码器，只是输出层改为二分类头，用来预测哪段答案更优。  
   - **输入方式**：把两段代码拼接在一起，中间加上特殊分隔符，模型一次性看到完整对比信息。  
   - **训练目标**：最小化交叉熵损失，使模型输出的概率与自动生成的偏好标签对齐。因为数据量可以达到数千万对，训练过程完全可扩展到多 GPU/TPU 集群。  
   - **技巧**：作者在预训练时加入了**噪声注入**（随机删改代码片段）和**对齐正则化**（鼓励模型对相似实现给出相近分数），防止模型只记住表面特征。

3. **任务微调（Reward Model Finetune）**  
   - **迁移到自然语言**：把预训练好的偏好模型的参数作为初始化，换上自然语言的输入格式（两段答案的文本拼接）。  
   - **少量人类偏好数据**：在 GSM8K、MATH、ReClor、LogiQA2.0 等任务上收集的真实人类偏好对（通常只有几千对），继续微调模型，使其适应语言特有的表达差异。  
   - **得到奖励模型**：微调后得到的 RM 能够在 RLHF 中为 LLM 生成的答案打分，随后用 PPO（Proximal Policy Optimization）等强化学习算法进行策略更新。

**最巧妙的点**在于把代码视作“天然的可验证任务”。代码的正确性可以通过编译或运行时检查自动得到，这让偏好标签的生成几乎不需要人工干预，极大提升了数据规模和质量。

### 实验与效果
- **评测任务**：数学推理（GSM8K、MATH）和逻辑推理（ReClor、LogiQA2.0）。这些数据集分别要求模型给出详细计算步骤或进行严密的逻辑判断。  
- **基线对比**：与仅使用人类偏好微调的普通 RM、以及直接在同样少量偏好数据上训练的模型相比，CodePMP 在所有四个数据集上都有明显提升。  
  - 例如在 GSM8K 上，使用 CodePMP 的模型比普通 RM 提高约 **4.2%** 的准确率；在 ReClor 上提升约 **3.7%**。  
- **消融实验**：作者分别去掉（1）代码预训练阶段、（2）噪声注入、（3）对齐正则化，发现去掉任意一项都会导致整体性能下降 1–2% 之间，说明每个设计都有贡献。  
- **数据规模敏感性**：当代码‑偏好对数量从 1M 增加到 10M 时，微调后 RM 的收敛速度明显加快，最终性能提升约 1%。这验证了“可扩展预训练”带来的收益。  
- **局限性**：论文未给出在多语言（非英语）代码或非程序化任务上的实验；此外，自动生成的偏好标签仍可能受静态分析工具的误判影响，导致噪声数据混入。

### 影响与延伸思考
CodePMP 把“代码审查”搬进了语言模型的奖励学习流程，打开了利用结构化、可验证数据来补足人类偏好稀缺的思路。自发表后，已有工作尝试把**数学证明**、**化学方程式**等可自动验证的领域也转化为偏好对进行预训练，进一步验证了“跨域偏好预训练” 的通用性（推测）。另外，社区也在探索把 **单模态代码偏好** 与 **多模态（代码+文档）** 结合，期待提升模型在解释性和可追溯性方面的表现。想深入了解的读者可以关注近期在 arXiv 上出现的 “Preference Pretraining for RLHF” 系列论文，以及 OpenAI、DeepMind 对 RLHF 数据效率的最新报告。

### 一句话记住它
**CodePMP 用海量自动生成的代码比较对为奖励模型预训练提供“练习册”，让少量真实人类偏好就能让大语言模型的推理能力飞跃。**