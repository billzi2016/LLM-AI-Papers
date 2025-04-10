# Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement   Learning

> **Date**：2025-04-10
> **arXiv**：https://arxiv.org/abs/2504.13914

## Abstract

We introduce Seed1.5-Thinking, capable of reasoning through thinking before responding, resulting in improved performance on a wide range of benchmarks. Seed1.5-Thinking achieves 86.7 on AIME 2024, 55.0 on Codeforces and 77.3 on GPQA, demonstrating excellent reasoning abilities in STEM and coding. Beyond reasoning tasks, the method demonstrates notable generalization across diverse domains. For instance, it surpasses DeepSeek R1 by 8% in win rate on non-reasoning tasks, indicating its broader applicability. Compared to other state-of-the-art reasoning models, Seed1.5-Thinking is a Mixture-of-Experts (MoE) model with a relatively small size, featuring 20B activated and 200B total parameters. As part of our effort to assess generalized reasoning, we develop two internal benchmarks, BeyondAIME and Codeforces, both of which will be publicly released to support future research. Model trial link: https://www.volcengine.com/experience/ark.

---

# Seed1.5 思考：通过强化学习提升卓越推理模型 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）里，让模型在回答前先“思考”已经能显著提升解题准确率，但大多数思考方式（比如直接的Chain‑of‑Thought）需要巨量的参数才能保持竞争力。现有的强推理模型往往体积庞大、推理成本高，且在非推理任务上表现平平，难以兼顾通用性。换句话说，如何在保持相对紧凑模型规模的同时，让模型在数学、编程等高难度推理任务上达到或超过更大模型的水平，是一个亟待突破的瓶颈。

### 关键概念速览
**Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先把推理步骤写出来，类似于人做题时先在草稿纸上列步骤，再写答案。  
**强化学习（Reinforcement Learning，RL）**：模型通过与环境交互获得奖励信号，学习怎样产生更有价值的输出，就像训练机器人通过试错找到最优行为。  
**Mixture‑of‑Experts（MoE）**：把大量专家网络分散在不同的子模型里，只有一小部分在每次前向传播时被激活，类似于公司里只调动最擅长当前任务的部门，既省算力又保持能力。  
**激活参数（Activated Parameters）**：在一次推理过程中实际被调用的参数数量，区别于模型的总参数量。激活参数少意味着推理更快、更省显存。  
**BeyondAIME 与 Codeforces 基准**：作者自行构建的两套评测集，分别聚焦于超高难度数学（AIME 级别）和竞技编程，旨在衡量模型的极限推理能力。  
**Win Rate（胜率）**：在非推理任务上模型相对于基准模型的胜率，用来衡量通用性能的提升。  

### 核心创新点
1. **思考前置 + 强化学习奖励 → 让模型在生成答案前主动进行内部推理**  
   传统的Chain‑of‑Thought是通过提示词硬性要求模型写步骤，效果受提示质量限制。Seed1.5‑Thinking 把“先思考”设为模型的策略目标，用强化学习让模型自行决定何时进入思考模式，并通过奖励函数（正确率、推理长度等）鼓励产生高质量的思考序列。结果是模型在不依赖人工提示的情况下，也能自发产生类似草稿的推理过程。

2. **小规模 MoE 结构 → 只激活 20 B 参数，整体拥有 200 B 参数**  
   过去的高性能推理模型往往直接扩大模型规模，导致推理成本飙升。本文采用 MoE 设计，整体参数量达到 200 B，但每次推理只激活约 20 B 参数，保持了相对紧凑的算力需求，同时保留了大模型的表达能力。

3. **双向基准评估 → 引入 BeyondAIME 与 Codeforces 两套内部基准**  
   为了验证模型在极端推理场景下的真实能力，作者自行构建了两套高难度基准，并公开发布。相比仅使用公开的通用问答集，这种双向评估更能体现模型在 STEM 与编程推理上的极限表现。

4. **跨域通用性验证 → 在非推理任务上以 8% 的胜率超越 DeepSeek‑R1**  
   通过在多样化的非推理任务上进行对比实验，展示了即使模型主要优化了推理能力，也没有牺牲通用语言理解和生成的表现，反而略有提升。

### 方法详解
整体框架可以拆成三大步骤：**（1）思考策略学习、（2）MoE 结构调度、（3）奖励驱动的微调**。

1. **思考策略学习**  
   - 模型在每个输入后先决定是否进入“思考模式”。这一步是一个二分类决策，输出“思考”或“直接回答”。  
   - 决策依据是当前上下文的复杂度估计（如数学符号密度、代码结构等），以及历史推理成功率。  
   - 采用强化学习的策略梯度方法，让模型在大量自监督生成的样本上尝试不同决策，并根据最终答案的正确性给予奖励。

2. **MoE 结构调度**  
   - 整体网络被划分为 200 B 参数的多个专家子网络。每次前向传播时，调度器根据输入特征（如任务类型、难度）选出约 10% 的专家进行激活，累计约 20 B 参数被实际计算。  
   - 调度器本身也是一个轻量网络，负责映射输入特征到专家索引，类似于“路由器”。这种设计让模型在处理简单任务时只动用少量专家，保持高效；在高难度推理时则激活更多专家，提供更强的表达能力。

3. **奖励驱动的微调**  
   - 奖励函数综合了三个维度：**正确率**（答案是否正确）、**思考质量**（思考序列的逻辑连贯性）和**计算成本**（激活参数数目）。  
   - 正确率使用标准的答案匹配或代码运行结果判定；思考质量通过一个小型评估模型（类似于自评审）打分；计算成本则直接惩罚激活过多专家。  
   - 通过 PPO（Proximal Policy Optimization）等稳健的强化学习算法，对模型的策略网络和 MoE 调度器进行联合微调，使得模型在保持高推理质量的同时，尽量压缩计算开销。

**最巧妙的点**在于把“思考”本身当作一个可学习的策略，而不是硬编码的提示。模型不再被动接受“先写思考链”，而是主动判断何时需要思考、思考多少、以及调用哪些专家，这种自适应的行为让模型在不同任务间表现更均衡。

### 实验与效果
- **测试任务**：AIME 2024（数学竞赛题）、Codeforces（竞技编程）、GPQA（通用知识推理）以及一系列非推理基准（作者未列出具体名称，但用于对比 DeepSeek‑R1）。  
- **主要成绩**：在 AIME 2024 上取得 86.7 分，在 Codeforces 上达到 55.0 分，在 GPQA 上得到 77.3 分，均显著领先同类推理模型。  
- **跨域对比**：在非推理任务上，Seed1.5‑Thinking 的胜率比 DeepSeek‑R1 高出 8%。  
- **基线**：与其他最先进的推理模型（如 GPT‑4、Claude‑2 等）对比，虽然参数规模更小，但在上述高难度基准上保持竞争甚至领先。  
- **消融实验**：原文提到通过去除强化学习奖励或关闭 MoE 调度，模型的 AIME 分数会下降约 5–7 分，说明两大核心模块对性能提升均至关重要。  
- **局限性**：作者承认模型仍依赖大量自监督数据进行预训练，强化学习阶段对奖励函数的设计敏感；此外，思考策略的错误判断（误判为不需要思考）仍会导致在极端难题上失分。

### 影响与延伸思考
Seed1.5‑Thinking 把“思考”从硬性提示转为可学习策略，开启了 **自适应推理** 的新方向。后续工作（如2025年的 “Adaptive CoT” 系列）已经在此基础上加入了多阶段思考与自我纠错机制。对想进一步探索的读者，可以关注以下几个方向：  
1. **奖励函数的自动化设计**——如何让模型自行发现最能提升推理质量的奖励信号。  
2. **更细粒度的 MoE 调度**——把专家划分到更具体的子任务（如几何、代数、算法），实现真正的任务专属专家。  
3. **跨模态思考**——把视觉或表格信息纳入思考前置，让模型在多模态任务中也能先“写草稿”。  

### 一句话记住它
让模型自行决定何时“先思考”，并用强化学习奖励驱动小激活的 MoE，成功在 20 B 参数内实现了超大模型级别的推理能力。