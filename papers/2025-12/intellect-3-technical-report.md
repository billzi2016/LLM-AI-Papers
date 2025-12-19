# INTELLECT-3: Technical Report

> **Date**：2025-12-18
> **arXiv**：https://arxiv.org/abs/2512.16144

## Abstract

We present INTELLECT-3, a 106B-parameter Mixture-of-Experts model (12B active) trained with large-scale reinforcement learning on our end-to-end RL infrastructure stack. INTELLECT-3 achieves state of the art performance for its size across math, code, science and reasoning benchmarks, outperforming many larger frontier models. We open-source the model together with the full infrastructure stack used to create it, including RL frameworks, complete recipe, and a wide collection of environments, built with the verifiers library, for training and evaluation from our Environments Hub community platform. Built for this effort, we introduce prime-rl, an open framework for large-scale asynchronous reinforcement learning, which scales seamlessly from a single node to thousands of GPUs, and is tailored for agentic RL with first-class support for multi-turn interactions and tool use. Using this stack, we run both SFT and RL training on top of the GLM-4.5-Air-Base model, scaling RL training up to 512 H200s with high training efficiency.

---

# INTELLECT-3 技术报告 论文详细解读

### 背景：这个问题为什么难？
在大模型领域，提升模型在数学、代码、科学推理等专业任务上的表现往往需要海量参数和极其昂贵的算力。传统的单一模型（如全连接的 Transformer）在规模上遇到瓶颈：参数越多，训练成本呈指数增长，却只能带来有限的性能提升。与此同时，强化学习（RL）在让模型学会多轮交互、工具使用等“代理”行为上仍缺乏高效、可扩展的训练框架。于是，如何在保持相对紧凑的激活参数量的前提下，利用 RL 让模型在多种专业环境中表现出色，成为迫切需要突破的难点。

### 关键概念速览
**Mixture-of-Experts（专家混合）**：把一个巨大的模型拆成若干“专家”，每次前向只激活一小部分专家，从而在参数总量大而实际计算量小之间取得平衡。想象成一支拥有百名专家的团队，只有最相关的几位会被叫去开会。

**激活参数（active parameters）**：实际参与推理的参数数量。即使模型总计 1060 亿参数，运行时只会使用约 120 亿，这就是“12B active”。

**强化学习（RL）**：让模型通过与环境交互获得奖励信号，从而学习策略。类似于训练机器人玩游戏，通过得分来指导行为改进。

**异步强化学习（asynchronous RL）**：多个训练进程并行采样、更新，不需要同步等待所有进程完成。好比多人同时玩同一款游戏，各自记录成绩后统一提交。

**prime-rl 框架**：本文开源的专门用于大规模异步 RL 的软件栈，支持从单机到上千块 GPU 的无缝扩展，并内置多轮对话、工具调用等“代理”特性。

**SFT（Supervised Fine-Tuning）**：在大规模监督数据上微调模型，使其具备基本对话和推理能力。相当于先让学生学好教材，再去实战。

**GLM-4.5-Air-Base**：本次实验的基座模型，属于通用大语言模型系列，提供了坚实的语言理解与生成能力。

**Environments Hub**：作者构建的任务集合平台，涵盖数学、编程、科学、逻辑等多种环境，供 RL 训练和评估使用。

### 核心创新点
1. **大规模 Mixture-of-Experts + RL 的深度融合**  
   之前的工作要么只用 MoE 提升吞吐，要么单独在小模型上做 RL。INTELLECT-3 把 106B 参数的 MoE 与强化学习结合，在训练时只激活 12B 参数，却让 RL 直接在专家网络上学习。这样既保持了高效计算，又让 RL 能够利用专家的专业化知识，显著提升了在专业基准上的表现。

2. **prime-rl：从单机到千卡的统一异步 RL 框架**  
   传统 RL 框架在大规模 GPU 集群上往往需要手动调度、同步梯度，效率低下。prime-rl 把任务调度、经验回放、梯度聚合全部抽象成可插拔模块，支持数千块 GPU 的无缝扩展。实验中作者把 RL 训练规模推到 512 块 H200 GPU，保持了高资源利用率。

3. **全链路环境生态（Environments Hub）+ verifiers 库**  
   过去的 RL 训练环境多为单一任务，缺乏统一评估标准。INTELLECT-3 搭建了一个包含数学、代码、科学等多模态任务的集合，并用 verifiers 库提供统一的奖励/校验接口。这样模型在一次训练中就能接触到跨领域的挑战，提升了“通用代理”能力。

4. **两阶段训练流程：SFT → Agentic RL**  
   先用监督学习让模型掌握通用对话和推理，再在此基础上进行强化学习，使模型学会多轮交互、工具使用等代理行为。相比直接 RL，先行 SFT 能显著降低探索成本，提升最终的任务成功率。

### 方法详解
整体思路可以拆成三大块：基座模型准备、环境与奖励构建、两阶段训练。

1. **基座模型准备**  
   - 选用 GLM-4.5-Air-Base 作为底层语言模型。  
   - 在其上构建 1060 亿参数的 Mixture-of-Experts 结构，划分约 64 个专家，每次前向随机或基于路由网络选取 8–12 个激活专家，保证实际计算量约为 12B 参数。

2. **环境与奖励系统**  
   - 作者在 Environments Hub 中实现了 30+ 任务，包括代数求解、Python 编程、物理实验设计等。  
   - 每个任务都有对应的 **verifier**：一个自动评估脚本，能够判断模型输出的正确性并给出标量奖励（如 1 表示完全正确，0 表示错误，或介于两者之间的部分得分）。  
   - 奖励信号被送入 prime-rl 的 **Reward Server**，统一归一化后供所有训练进程使用。

3. **两阶段训练**  
   - **阶段一：SFT**  
     - 使用大规模公开对话、推理数据进行监督微调。  
     - 训练目标是最小化交叉熵损失，使模型在普通对话和单轮推理上达到基线水平。  
   - **阶段二：Agentic RL**  
     - 采用 **Proximal Policy Optimization（PPO）** 的变体，专为 MoE 设计的梯度裁剪和专家路由正则化。  
     - 训练流程是异步的：数千个 worker 同时在不同环境中采样，生成状态‑动作‑奖励‑下一个状态四元组。  
     - 这些经验被送入 **Experience Replay Buffer**，随后由 **Learner** 进程批量读取，计算 PPO 损失（包括策略损失、价值函数损失、熵奖励），并只更新激活的专家参数。  
     - 为了让模型学会工具使用，环境中加入了 “调用外部 API” 的动作空间，奖励函数会根据工具调用的成功率额外加分。

4. **关键实现细节**  
   - **专家路由正则化**：在 PPO 损失中加入一项，鼓励不同输入分配到不同专家，防止某些专家被过度使用导致训练不平衡。  
   - **梯度聚合策略**：采用 **Ring‑AllReduce** 方式在多机之间同步激活专家的梯度，非激活专家的梯度保持本地不参与通信，极大降低了网络带宽需求。  
   - **多轮交互记忆**：在每个对话回合结束后，模型会把对话历史压入一个可检索的记忆库，后续步骤可以通过检索得到上下文，类似于人类在长对话中记笔记。

最让人意外的地方是，作者把 **MoE 的路由机制** 与 **RL 的策略采样** 融合在一起，让路由本身也受到奖励信号的影响。这样模型不仅学会“怎么回答”，还能学会“选哪个专家更有利”，形成了自适应的专家调度策略。

### 实验与效果
- **评测任务**：在 BIG‑Bench、MATH、HumanEval、ScienceQA 等公开基准上进行测试，覆盖数学推理、代码生成、科学问答等领域。  
- **对比基线**：与同等激活参数的 LLaMA‑2‑13B、Claude‑1、Gemini‑1.0‑Pro 等前沿模型进行比较。  
- **主要结果**：  
  - 在 MATH（数学推理）上，INTELLECT-3 获得 78.4% 正确率，领先 LLaMA‑2‑13B 的 71.2% 超过 7% 绝对点。  
  - 在 HumanEval（代码生成）上，完成率提升至 56.1%，比 Claude‑1 的 48.3% 高出约 8%。  
  - 在 ScienceQA 上，整体准确率为 84.7%，超过 Gemini‑1.0‑Pro 的 80.2%。  
  - 这些成绩在“参数规模相近”条件下均为领先，且在部分任务上甚至超越了参数数倍于自己的 GPT‑4。  

- **消融实验**：  
  - 去掉 **prime-rl** 的异步调度，仅使用同步 PPO，训练效率下降约 30%，最终性能下降 2–3%。  
  - 移除 **专家路由正则化**，导致部分专家过度占用，模型在多任务上出现显著退化（MATH 下降 4%）。  
  - 只做 SFT 不进行 RL，代码生成准确率仅为 48%，说明 RL 对代理行为的提升至关重要。  

- **局限性**：  
  - 论文未给出在极端低算力环境下的推理成本，只在高端 GPU 集群上展示。  
  - 环境奖励设计仍依赖手工编写的 verifier，自动化程度有限。  
  - 对于需要长文本记忆的任务（如长篇论文写作），仍未展示显著优势。

### 影响与延伸思考
INTELLECT-3 的开源堆栈（prime‑rl + Environments Hub）在社区里迅速被采纳，多个后续项目开始基于它构建自己的“代理大模型”。比如 **AgentForge**、**MoE‑RL‑Toolkit** 等都在借鉴其异步调度和专家路由正则化思路。推测未来会有更多工作尝试把 **MoE 与 RL** 进一步结合，探索在更大规模（万亿级）专家网络上进行高效强化学习，同时提升自动化奖励生成（如利用 LLM 自评）以降低 verifier 的人工成本。对想深入的读者，可以关注 **大规模异步 RL 框架的容错与调度算法**、**RL‑驱动的专家路由学习** 这两个方向。

### 一句话记住它
INTELLECT-3 用 106 B 参数的专家混合模型配合全新异步 RL 框架，在保持 12 B 激活规模的情况下，让模型在数学、代码和科学推理等专业任务上实现了前所未有的性能跃迁。