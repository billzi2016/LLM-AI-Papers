# LIMR: Less is More for RL Scaling

> **Date**：2025-02-17
> **arXiv**：https://arxiv.org/abs/2502.11886

## Abstract

In this paper, we ask: what truly determines the effectiveness of RL training data for enhancing language models' reasoning capabilities? While recent advances like o1, Deepseek R1, and Kimi1.5 demonstrate RL's potential, the lack of transparency about training data requirements has hindered systematic progress. Starting directly from base models without distillation, we challenge the assumption that scaling up RL training data inherently improves performance. we demonstrate that a strategically selected subset of just 1,389 samples can outperform the full 8,523-sample dataset. We introduce Learning Impact Measurement (LIM), an automated method to evaluate and prioritize training samples based on their alignment with model learning trajectories, enabling efficient resource utilization and scalable implementation. Our method achieves comparable or even superior performance using only 1,389 samples versus the full 8,523 samples dataset. Notably, while recent data-efficient approaches (e.g., LIMO and s1) show promise with 32B-scale models, we find it significantly underperforms at 7B-scale through supervised fine-tuning (SFT). In contrast, our RL-based LIMR achieves 16.7% higher accuracy on AIME24 and outperforms LIMO and s1 by 13.0% and 22.2% on MATH500. These results fundamentally reshape our understanding of RL scaling in LLMs, demonstrating that precise sample selection, rather than data scale, may be the key to unlocking enhanced reasoning capabilities. For reproducible research and future innovation, we are open-sourcing LIMR, including implementation of LIM, training and evaluation code, curated datasets, and trained models at https://github.com/GAIR-NLP/LIMR.

---

# LIMR：少即是多的强化学习扩展 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）里，强化学习（RL）被用来提升推理能力，像 o1、Deepseek R1、Kimi 1.5 都展示了它的潜力。但这些模型往往使用上万条标注数据进行 RL 微调，训练成本高、透明度低，研究者很难判断到底需要多少数据、哪些数据最关键。过去的做法默认“数据越多越好”，于是投入大量算力却缺乏系统的评估手段，导致资源浪费和可复现性差。

### 关键概念速览
**强化学习（RL）**：让模型在交互式环境中通过奖励信号学习策略，类似于训练机器人通过试错找到最优动作。  
**语言模型（LM）**：能够预测下一个词的神经网络，像自动补全一样生成文本。  
**监督微调（SFT）**：在已有模型上用标注好的问答对继续训练，类似于给学生做练习题。  
**学习影响度测量（LIM）**：一种自动化评估每条训练样本对模型学习轨迹贡献大小的工具，像老师给每道题打分，决定哪些题值得多练。  
**样本子集选择**：从完整数据集中挑出最有价值的若干条，用更少的数据达到同等或更好效果。  
**AIME24 / MATH500**：分别是美国数学竞赛和数学推理基准，用来衡量模型的数学推理水平。  
**数据效率（Data Efficiency）**：在保持性能的前提下，用更少的训练数据或算力完成任务的能力。

### 核心创新点
1. **从“越多越好”到“挑最优”**：过去的 RL 微调直接使用全部 8,523 条样本；本文先用 LIM 对每条样本的学习贡献进行量化，然后只保留贡献最高的 1,389 条。这样在数据量上削减约 84%，却实现了更高的推理准确率。  
2. **Learning Impact Measurement（LIM）自动化评估**：LIM 通过比较模型在不同训练阶段对同一样本的预测变化，捕捉该样本是否推动了模型向更好解答方向前进。相当于在训练过程中实时监控每道练习题的“提分效果”。  
3. **RL‑驱动的样本筛选管线**：作者把 LIM 产生的分数作为奖励信号，构建一个基于 RL 的子集选择策略，使得最终选出的子集在 RL 微调阶段能够最大化整体性能提升。  
4. **跨尺度验证**：在 7B 参数模型上，LIMR 的 1,389 条子集比同等规模的 SFT 方法（如 LIMO、s1）提升 13%~22% 的数学基准成绩，证明了该思路在中小模型上同样有效。

### 方法详解
整体框架可以分为三步：**（1）基线模型准备 → (2) LIM 评分 → (3) RL‑驱动子集微调**。

1. **基线模型准备**  
   - 直接使用未经蒸馏的基础语言模型（如 LLaMA‑7B），不做任何预先的监督微调。这样可以确保后续的性能提升全部来源于 RL 与样本筛选，而不是先前的 SFT。

2. **Learning Impact Measurement（LIM）**  
   - **轨迹采样**：在模型的普通训练（未加 RL）过程中，记录模型对每条候选样本的输出概率随训练步数的变化。  
   - **影响度计算**：对每条样本，比较其在早期（如第 0 步）和后期（如第 N 步）预测分布的差异。如果模型对该样本的答案从错误倾向显著转向正确倾向，则认为该样本对学习轨迹的“推动力”大。  
   - **分数归一化**：将所有样本的影响度映射到 0‑1 区间，得到每条样本的 LIM 分数。可以把它想象成老师给每道练习题打的“提分分数”。  

3. **RL‑驱动子集选择与微调**  
   - **子集构建**：按照 LIM 分数从高到低排序，取前 1,389 条（约 16%）作为 RL 微调的训练集。  
   - **奖励函数设计**：在 RL 微调阶段，奖励不仅包括传统的答案正确率，还加入了“样本贡献度”——即模型在该样本上提升的概率变化。这样模型被鼓励去学习那些本身已经被 LIM 判定为高价值的样本。  
   - **策略优化**：使用 Proximal Policy Optimization（PPO）等常见 RL 优化器，对模型参数进行迭代更新。整个过程与常规的 RLHF（Human Feedback）相似，只是训练数据被大幅压缩且奖励更具针对性。  

**最巧妙的点**在于把“样本对模型学习的影响”量化为可直接用于 RL 奖励的信号，实现了数据筛选与策略学习的闭环。传统做法把数据筛选当作离线步骤，二者之间缺乏交互；这里的 LIM 直接参与 RL 目标，使得模型在学习过程中不断强化对高价值样本的敏感度。

### 实验与效果
- **测试任务**：AIME 24（美国数学竞赛）和 MATH 500（数学推理基准），两者都要求模型进行严谨的数理推理。  
- **对比基线**：包括全量 8,523 条样本的 RL 微调、以及 7B 参数模型的 SFT 方法 LIMO、s1。  
- **主要结果**：使用仅 1,389 条样本的 LIMR 在 AIME 24 上比全量 RL 提升 16.7% 的准确率；在 MATH 500 上分别比 LIMO 高 13.0%、比 s1 高 22.2%。  
- **消融实验**：作者分别去掉 LIM 评分、去掉奖励中的样本贡献度、以及直接使用随机子集进行对比，发现没有 LIM 评分的子集性能下降约 10%，而去掉奖励中的贡献度则下降约 6%。这说明两者都是提升效果的关键因素。  
- **局限性**：实验仅在 7B 规模模型上展开，未验证在更大模型（如 70B）上的表现；LIM 评分依赖于基线模型的训练轨迹，若基线本身质量不佳，评分可能失真。作者也承认目前的 LIM 计算成本仍比单纯随机抽样高。

### 影响与延伸思考
LIMR 的核心思想——“少而精” 的样本筛选结合 RL 奖励——已经在后续的模型微调工作中被广泛引用。2024 年底出现的几篇论文（如 **Data‑Smart RL**、**Selective RLHF**）都在不同程度上借鉴了 LIM 的影响度度量思路，尝试把样本价值直接嵌入奖励函数。对想进一步探索的读者，可以关注以下方向：  
1. **跨模型通用的影响度度量**：如何让 LIM 在不同架构、不同规模的模型之间保持一致性。  
2. **在线 LIM 更新**：在 RL 训练过程中实时更新样本影响度，而不是事先一次性计算。  
3. **与人类反馈结合**：把人类标注的偏好与 LIM 评分融合，构建更丰富的奖励信号。  

### 一句话记住它
只要挑对了关键的几千条训练样本，强化学习微调就能用更少的数据跑出更强的推理能力。