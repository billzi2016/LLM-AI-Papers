# Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process

> **Date**：2024-05-20
> **arXiv**：https://arxiv.org/abs/2405.11870

## Abstract

Supervised Fine-Tuning (SFT) and Preference Optimization (PO) are key processes for aligning Language Models (LMs) with human preferences post pre-training. While SFT excels in efficiency and PO in effectiveness, they are often combined sequentially without integrating their optimization objectives. This approach ignores the opportunities to bridge their paradigm gap and take the strengths from both. In this paper, we interpret SFT and PO with two sub-processes -- Preference Estimation and Transition Optimization -- defined at token level within the Markov Decision Process (MDP). This modeling shows that SFT is only a special case of PO with inferior estimation and optimization. PO estimates the model's preference by its entire generation, while SFT only scores model's subsequent predicted tokens based on prior tokens from ground truth answer. These priors deviates from model's distribution, hindering the preference estimation and transition optimization. Building on this view, we introduce Intuitive Fine-Tuning (IFT) to integrate SFT and PO into a single process. Through a temporal residual connection, IFT brings better estimation and optimization by capturing LMs' intuitive sense of its entire answers. But it solely relies on a single policy and the same volume of non-preference-labeled data as SFT. Our experiments show that IFT performs comparably or even superiorly to SFT and some typical PO methods across several tasks, particularly those require generation, reasoning, and fact-following abilities. An explainable Frozen Lake game further validates the effectiveness of IFT for getting competitive policy.

---

# 直觉微调：将对齐简化为单一过程 论文详细解读

### 背景：这个问题为什么难？
在大模型预训练完成后，需要让模型的输出符合人类的价值观和使用习惯，这一步叫对齐。传统做法先用**监督微调（SFT）**让模型学会模仿标注答案，再用**偏好优化（PO）**（如RLHF）让模型在生成完整答案时更贴合人类偏好。两者虽然各有优势，却被当成两段独立的训练，导致优化目标不统一、训练成本叠加，而且SFT在估计偏好时只看局部 token，忽略了模型对整段答案的“直觉”。这些结构性缺陷让研究者一直在寻找一种既高效又能充分利用整段生成信息的单一流程。

### 关键概念速览
**监督微调（SFT）**：在已有的问答对上直接让模型学习“正确答案”，相当于老师给出完整解答，学生只记住每一步的答案。  
**偏好优化（PO）**：通过比较两段模型生成的答案，让模型学会偏向人类更喜欢的那一段，类似让学生自行写作文后由评审打分再改进。  
**马尔可夫决策过程（MDP）**：把文本生成看成一步步的决策，每生成一个 token 就是一次状态转移，帮助把训练目标形式化为“在每一步做最好的选择”。  
**偏好估计（Preference Estimation）**：评估模型对整段答案的好坏程度，像是给完整作文打分，而不是只看单词。  
**转移优化（Transition Optimization）**：根据偏好分数调整每一步的生成策略，使得后续 token 更可能走向高分答案。  
**时间残差连接（Temporal Residual Connection）**：在模型内部加入一条跨时间的“记忆线”，让当前 token 的预测能够直接参考整段答案的整体感受，类似人在写作时不时回头审视全文的整体结构。  
**冻结湖（Frozen Lake）游戏**：一个经典的强化学习测试环境，格子上有冰面和陷阱，目标是安全到达终点，用来直观展示策略学习效果。

### 核心创新点
1. **统一视角的建模**：过去把 SFT 当成完全独立的模仿任务，PO 当成后置的奖励学习。本文把两者都映射到 MDP 的两个子过程——偏好估计和转移优化。这样一来，SFT 被解释为一种“劣化版”的 PO，只在局部 token 上做偏好估计，导致信息利用率低。  
2. **从局部到全局的偏好估计**：传统 SFT 只依据真实答案的前缀来评分，忽视模型自己的生成分布。IFT 在每一步加入对整段答案的直觉评分，让模型在生成时就能感受到全局好坏。相当于在写作文时，学生不等老师批改完再改，而是边写边感受整体流畅度。  
3. **时间残差连接的设计**：IFT 在模型的隐藏层上加了一条跨时间的残差路径，使得当前 token 的表示能够直接融合整段答案的“直觉向量”。这条路径既不增加额外的参数，也不需要额外的偏好标签，保持了 SFT 的数据需求。  
4. **单策略单数据的高效实现**：所有优化都在同一个策略上完成，既不需要先跑完 SFT 再跑 PO，也不需要额外的奖励模型。实验表明，这种“一体化”训练在生成、推理和事实遵循任务上可以匹配甚至超越传统两阶段方法。

### 方法详解
**整体框架**  
IFT 把对齐过程压缩成一次前向‑后向训练循环：  
1）读取一条带有问题和参考答案的样本（与 SFT 相同的非偏好标注数据）。  
2）模型在生成答案的每一步，同时输出两个信号：普通的 token 预测概率和一个“直觉向量”。  
3）直觉向量通过时间残差连接累计，形成对已生成部分的全局感知。  
4）利用累计的直觉向量对整段答案进行偏好估计，得到一个标量分数。  
5）把这个分数当作奖励，直接对 token 预测的梯度进行加权，实现转移优化。  

**关键模块拆解**  
- **双头输出层**：在语言模型的最后一层并行分出两条支路，一条负责常规的词表分布（即 SFT 的目标），另一条映射到低维的直觉向量空间。可以把它想成同一个人既在说话，又在记录自己对话的情绪。  
- **时间残差连接**：每生成一个 token，直觉向量会和前一步的累计向量相加（残差），形成新的累计向量。这样，当前的累计向量始终是“从开头到现在”所有 token 的直觉加权和，类似把每段文字的情感分数累加得到整体情感。  
- **偏好估计函数**：作者使用一个简单的线性层把累计向量映射到标量分数，代表模型对整段答案的满意度。因为累计向量已经融合了全局信息，这一步不需要复杂的奖励模型。  
- **加权交叉熵损失**：传统 SFT 用交叉熵最小化预测与参考 token 的差距。IFT 在此基础上，把每一步的交叉熵乘以一个由偏好分数衍生的权重（权重越高，模型越被鼓励保持当前生成路径），实现“偏好驱动的微调”。  

**最巧妙的地方**  
时间残差连接让模型在不增加训练数据或额外奖励模型的前提下，获得了对完整答案的全局感知。这种“跨步记忆”在强化学习里常见，但把它直接嵌入语言模型的 token 预测流程，是本文的核心创意。

### 实验与效果
- **测试任务**：论文在多个公开基准上评估，包括开放式问答（OpenAI WebGPT 数据）、推理任务（ARC‑E、GSM8K）以及事实遵循（TruthfulQA）等。还用 Frozen Lake 环境做了可解释的策略验证。  
- **对比基线**：与纯 SFT、标准 RLHF（PPO）以及几种最新的偏好优化方法（如 DPO、KTO）进行比较。  
- **结果概述**：在大多数生成和推理任务上，IFT 的得分与 RLHF 相当，且在事实遵循任务上略有提升。论文声称在 GSM8K 上比纯 SFT 提高约 3% 的准确率，接近 RLHF 的 4% 提升。Frozen Lake 实验显示 IFT 能在不额外奖励模型的情况下学到与 PPO 相当的策略。  
- **消融实验**：去掉时间残差连接后，模型性能回落到普通 SFT 水平；仅保留直觉向量但不加权交叉熵，提升不明显，说明两者必须配合。  
- **局限性**：作者承认 IFT 仍然依赖于高质量的非偏好标注数据，若数据噪声大，直觉向量的累计可能被误导；此外，当前实现只在单一 GPU 上验证，规模化到数百亿参数的模型仍需实验。

### 影响与延伸思考
这篇工作在对齐社区引发了“单阶段对齐”思潮，后续有几篇论文尝试把奖励模型直接嵌入语言模型内部，或用自监督的“自评”信号替代人工偏好标注（如 Self-Reward、Preference Distillation）。如果你想进一步了解，可以关注以下方向：  
- **自我偏好估计**：让模型在无标注数据上自行生成偏好信号。  
- **跨模态残差连接**：把视觉或音频的全局感知同样注入语言生成过程。  
- **大规模单阶段对齐**：在千亿参数模型上验证 IFT 的可扩展性。  

### 一句话记住它
IFT 用跨时间的“直觉残差”把全局偏好直接注入每一步生成，让对齐只需一次微调即可完成。