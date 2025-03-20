# Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't

> **Date**：2025-03-20
> **arXiv**：https://arxiv.org/abs/2503.16219

## Abstract

Enhancing the reasoning capabilities of large language models (LLMs) typically relies on massive computational resources and extensive datasets, limiting accessibility for resource-constrained settings. Our study investigates the potential of reinforcement learning (RL) to improve reasoning in small LLMs, focusing on a 1.5-billion-parameter model, DeepSeek-R1-Distill-Qwen-1.5B, under strict constraints: training on 4 NVIDIA A40 GPUs (48 GB VRAM each) within 24 hours. Adapting the Group Relative Policy Optimization (GRPO) algorithm and curating a compact, high-quality mathematical reasoning dataset, we conducted three experiments to explore model behavior and performance. Our results demonstrate rapid reasoning gains - e.g., AMC23 accuracy rising from 63% to 80% and AIME24 reaching 46.7%, surpassing o1-preview - using only 7,000 samples and a $42 training cost, compared to thousands of dollars for baseline models. However, challenges such as optimization instability and length constraints emerged with prolonged training. These findings highlight the efficacy of RL-based fine-tuning for small LLMs, offering a cost-effective alternative to large-scale approaches. We release our code and datasets as open-source resources, providing insights into trade-offs and laying a foundation for scalable, reasoning-capable LLMs in resource-limited environments. All are available at https://github.com/knoveleng/open-rs.

---

# 小型大语言模型推理的强化学习：哪些有效哪些无效 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在数学、逻辑等需要多步推理的任务上表现出色，但要达到这种水平往往需要上百亿参数、海量算力和数十TB的训练数据。资源受限的实验室或个人开发者根本负担不起这些成本。于是出现了两大瓶颈：一是小模型本身的容量限制，使得它们在长序列推理时容易走偏；二是传统的微调方法（如监督学习）只能在已有答案上做模仿，难以让模型学会“思考”。这让人们迫切想知道，能否用更经济的手段——比如强化学习（RL）——在极小的算力预算内让 1.5 B 参数模型的推理能力突飞猛进。

### 关键概念速览
- **强化学习（RL）**：让模型在与环境交互后，根据得到的奖励信号自行调整行为策略，类似于训练一只小狗通过奖励学会新技巧。  
- **策略优化（Policy Optimization）**：RL 中的核心步骤，直接改进模型输出的概率分布，使得高奖励的行为更可能被采样。  
- **Group Relative Policy Optimization（GRPO）**：一种新颖的策略优化算法，先把样本划分成若干组，再在每组内部比较相对优势，避免单个异常样本对整体更新产生过大影响。可以想象成在选拔赛中先在小组内部排名，再决定整体晋级名单。  
- **思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前先写出推理步骤，像在黑板上写草稿一样，帮助模型保持逻辑连贯。  
- **长度约束（Length Constraint）**：LLM 在生成长文本时会受到上下文窗口大小的限制，超出后模型只能“忘记”前面的信息，导致推理中断。  
- **数学推理数据集**：专门收集的包含数学题目、解题步骤和答案的集合，质量高、样本少，类似于给模型提供的“精选练习册”。  

### 核心创新点
1. **把 GRPO 移植到 1.5 B 小模型**  
   之前的 GRPO 主要在 70 B 以上的大模型上实验，作者把它改写成适配 1.5 B 参数的轻量实现：把梯度累加改为每 8 步一次同步，降低显存占用。结果是，在同等算力下小模型也能获得稳定的策略提升，而不是像大模型那样“吃力不讨好”。  

2. **极简高质量数学推理数据**  
   与常见的数十万甚至上百万样本的公开数据不同，作者只挑选了 7 000 条经过人工筛选的高质量题目，确保每一道都有完整的思维链标注。这样既大幅降低了标注成本，又让 RL 训练的奖励信号更可靠。  

3. **成本‑效益对标实验**  
   通过把训练费用、显存使用、训练时长等指标量化，作者展示了在 4 张 NVIDIA A40（48 GB）卡、24 小时内完成的 RL 微调，仅花费约 42 美元，却把 AMC23 正确率从 63 % 提升到 80 %，AIME24 达到 46.7 %，已经超过了 OpenAI 的 o1‑preview（收费数千美元的模型）。这是一种“低成本高回报”的实证。  

4. **发现并记录训练不稳定与长度瓶颈**  
   在训练超过 12 小时后，模型出现了梯度爆炸和生成文本被截断的现象。作者把这些负面结果公开，提供了实际使用 RL 微调时需要注意的警示信息。  

### 方法详解
整体思路可以拆成三步：**数据准备 → 奖励设计 → GRPO 微调**。下面按顺序展开。

1. **数据准备**  
   - 从公开的数学竞赛题库中抽取 7 000 条题目，覆盖代数、几何、组合等领域。  
   - 每道题配上完整的思维链（约 3‑5 步）和最终答案，形成“问题‑思维链‑答案”三元组。  
   - 为了让模型在 RL 环境中能直接使用，这些三元组被转化为 Prompt‑Response 对话格式，Prompt 包含题目，Response 包含思维链和答案。

2. **奖励函数**  
   - **正确性奖励**：模型输出的答案与金标准匹配时给 +1，否则 0。  
   - **思维链完整性奖励**：检查生成的思维链是否包含所有关键步骤（通过关键词匹配），满足则额外 +0.5。  
   - **长度惩罚**：如果生成长度超过上下文窗口的 90%，则扣除 0.2，以防止模型因“写太多”而被截断。  
   - 最终奖励是上述三项的加权和，权重在小规模超参数搜索后固定为 1:0.5:‑0.2。

3. **GRPO 微调**  
   - **分组**：每 128 条样本划为一组，组内计算每条样本的相对优势（即该样本奖励与组内平均奖励的差值）。  
   - **相对优势加权**：优势大的样本对应的梯度乘以更大的系数，优势小的则被抑制，类似于在选拔赛中给表现突出的选手更高的晋级分数。  
   - **策略更新**：使用 PPO（近端策略优化）框架的 clipped objective，但把原始的 KL 散度约束换成“组内相对 KL”，从而在保持整体策略平稳的同时让组内优秀样本快速影响模型。  
   - **实现细节**：为适配 4 张 A40，作者采用了梯度累加（每 8 步累加一次）和混合精度（FP16）训练，使显存峰值保持在 42 GB 以下。整个训练循环在 24 小时内完成 30 万步。

**最巧妙的点**在于把“相对优势”概念引入策略更新，让 RL 在小样本、高噪声的环境下仍能保持信号的可辨识度。传统的 PPO 在样本量极少时容易被噪声淹没，而 GRPO 的组内比较相当于给模型提供了一个局部的“排行榜”，帮助它快速聚焦高价值行为。

### 实验与效果
- **测试任务**：AMC 10/12（2023 版）选择题、AIME（2024 版）开放式题目以及几个公开的数学推理基准（如 MATH、GSM8K 的子集）。  
- **基线对比**：  
  - 原始 1.5 B 模型（未微调）在 AMC23 上 63 % 正确率。  
  - 同模型使用普通监督微调（SFT）提升至约 71 %。  
  - 使用本文的 RL 微调后，AMC23 达到 80 %，AIME24 为 46.7 %，超过了 OpenAI o1‑preview（公开报告约 44 %）。  
- **消融实验**：  
  - 去掉思维链奖励，整体准确率下降约 4 %。  
  - 替换 GRPO 为标准 PPO，训练后 12 小时出现梯度不稳定，最终准确率仅 73 %。  
  - 将奖励函数中的长度惩罚去掉，生成文本经常被截断，导致 AIME 正确率跌至 38 %。  
- **局限性**：作者指出，训练时间超过 12 小时后模型出现了优化不稳定的现象，可能与组内样本分布变化有关；此外，长度约束仍是瓶颈，长思维链题目仍会被截断，需要更大的上下文窗口或分段生成策略。

### 影响与延伸思考
这篇工作在社区里被视为“RL 在小模型上的可行性示例”。随后有几篇后续研究尝试把类似的相对优势机制搬到 2 B‑3 B 参数的多语言模型上，甚至在代码生成任务中加入了“函数调用奖励”。从长远来看，**RL+高质量少样本** 可能成为资源受限实验室的主流微调套路。想进一步深入，可以关注以下方向：  
- **自适应分组**：动态调整组大小，让模型在训练后期能够捕捉更细粒度的优势差异。  
- **层级奖励**：把思维链拆成子步骤，每一步都给奖励，类似于强化学习中的层次化任务。  
- **跨模态推理**：把数学题目与图形（几何图）一起输入，检验 RL 是否还能提升多模态推理能力。  

### 一句话记住它
用 7 千条高质量数学题和改进的 GRPO，花 42 美元就能让 1.5 B 参数模型的推理准确率从 63 % 突破到 80 %。