# Scaling Reasoning without Attention

> **Date**：2025-05-28
> **arXiv**：https://arxiv.org/abs/2505.22425

## Abstract

Large language models (LLMs) have made significant advances in complex reasoning tasks, yet they remain bottlenecked by two core challenges: architectural inefficiency due to reliance on Transformers, and a lack of structured fine-tuning for high-difficulty domains. We introduce \ourmodel, an attention-free language model that addresses both issues through architectural and data-centric innovations. Built on the state space dual (SSD) layers of Mamba-2, our model eliminates the need for self-attention and key-value caching, enabling fixed-memory, constant-time inference. To train it for complex reasoning, we propose a two-phase curriculum fine-tuning strategy based on the \textsc{PromptCoT} synthesis paradigm, which generates pedagogically structured problems via abstract concept selection and rationale-guided generation. On benchmark evaluations, \ourmodel-7B outperforms strong Transformer and hybrid models of comparable scale, and even surpasses the much larger Gemma3-27B by 2.6\% on AIME 24, 0.6\% on AIME 25, and 3.0\% on Livecodebench. These results highlight the potential of state space models as efficient and scalable alternatives to attention-based architectures for high-capacity reasoning.

---

# 无注意力的推理扩展 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在数学、代码和逻辑推理上已经能跑出惊人的成绩，但它们的核心仍是 Transformer，必须靠自注意力（self‑attention）来捕捉远距离信息。自注意力的计算量随序列长度平方增长，导致显存和推理时间随输入变长而爆炸，尤其在需要上千甚至上万 token 的推理题目上几乎不可用。与此同时，现有的微调大多是一次性的指令调教，缺少针对高难度推理任务的层层递进训练方案，模型往往在“难题”前卡壳。于是出现了两个瓶颈：**架构效率**（算力/显存）和**高难度微调**（教学式训练缺失）。

### 关键概念速览
- **Transformer**：一种把每个词和所有其他词相互比较的网络，像全班同学互相讨论一样，能捕捉长程依赖，但计算成本随人数（序列长度）呈二次方增长。  
- **自注意力（Self‑Attention）**：Transformer 的核心机制，用键值对（key‑value）在每一步查找信息，类似在图书馆里每次都要把整本书翻一遍找答案。  
- **状态空间模型（State‑Space Model，SSM）**：把序列看成一个随时间演化的系统，用线性递推公式一次算出下一个状态，像流水线上的机器只记住当前的“状态”，不需要回头查找全部历史。  
- **SSD 层（State Space Dual）**：Mamba‑2 中的高效实现，把 SSM 的递推和卷积融合，能够在固定显存下完成长序列的前向计算。  
- **PromptCoT**：一种自动生成“思维链”训练数据的框架，先挑选抽象概念，再让模型依据这些概念写出完整的推理过程，类似老师先给出大纲，学生再填细节。  
- **Curriculum Fine‑Tuning（课程式微调）**：把训练任务按难度分层，从易到难逐步提升模型能力，像学钢琴先练指法再练曲子。  
- **固定内存、常数时间推理**：指模型在推理时不需要随序列增长而额外占用显存，也不需要每一步都做 O(n) 的查询，推理速度与序列长度无关。  

### 核心创新点
1. **用 SSD 取代自注意力 → 直接把 Mamba‑2 的 SSD 层嵌入语言模型 → 彻底摆脱键值缓存，显存不随序列增长，推理时间保持 O(1)。**  
2. **两阶段 PromptCoT 课程式微调 → 第一步用抽象概念生成“易”题并让模型写出完整思维链；第二步在此基础上加入更难的概念和更长的推理链 → 让模型在训练中逐步适应高难度数学推理，效果比一次性微调高出不少。**  
3. **数据驱动的结构化问题生成 → 通过“概念‑推理”双向生成，把题目、解题思路、答案三者紧密绑定，形成可直接用于 CoT 微调的高质量样本 → 解决了高质量推理数据稀缺的问题。**  
4. **在同等参数规模下超越大模型 → 7B 参数的无注意力模型在 AIME24、AIME25、Livecodebench 上分别比同尺度 Transformer、混合模型更好，甚至跑赢 27B 的 Gemma3 → 证明了“更聪明的架构+更好的微调”可以抵消参数量的劣势。  

### 方法详解
**整体思路**：先用 SSD（State Space Dual）层搭建一个纯注意力‑free 的语言模型（称为 \ourmodel），再用两阶段的 PromptCoT 课程式微调让它学会数学推理。整个流程可以划分为三块：① SSD 语言模型的构建，② PromptCoT 数据合成，③ 递进式微调。

**1️⃣ SSD 语言模型的构建**  
- 传统 Transformer 每层都要做自注意力 + 前馈。这里把每层的自注意力全部去掉，换成 Mamba‑2 的 SSD 层。SSD 层内部用离散化的线性常微分方程（ODE）把前一个 token 的隐藏状态映射到当前 token，随后加上卷积‑like 的门控机制，产生新的隐藏向量。  
- 由于状态只保存在一个固定大小的向量里，模型在推理时只需要把上一步的状态传递给下一步，不必保存所有历史 token 的 KV 对，这就实现了“固定显存”。  
- 为了兼容语言建模的自回归需求，作者在每一步把当前 token 的嵌入与 SSD 输出相加，再送入层归一化和前馈网络，保持了 Transformer 那套残差结构的好处。

**2️⃣ PromptCoT 数据合成**  
- **抽象概念选择**：从数学教材、竞赛题库中抽取概念（如“二次方程根的判别式”“欧拉公式”），并用一个小型的概念分类器把它们划分为“易”“中”“难”。  
- **推理链生成**：给定概念，使用一个已经微调好的 CoT 大模型（如 GPT‑4）让它先写出完整的思考步骤，再把步骤中的关键推理点抽取出来，形成“概念‑推理‑答案”三元组。  
- 这样得到的样本天然带有思维链，且每个样本的难度可控，适合后面的课程式训练。

**3️⃣ 两阶段课程式微调**  
- **阶段一（基础阶段）**：只喂入“易”概念生成的样本，让模型学习如何把概念映射到推理链，目标是让模型掌握基本的 CoT 结构。  
- **阶段二（挑战阶段）**：加入“中”“难”概念的样本，且在每个 batch 中混入一定比例的长链推理（超过 50 步），迫使模型在保持上下文连贯的同时学会更深层次的数学技巧。  
- 训练时使用 **教师强制**（teacher forcing）让模型直接预测思维链的每一步，然后再预测最终答案；损失函数把思维链的交叉熵和答案的交叉熵加权求和，确保模型既会写过程也会给出正确结果。

**最巧妙的点**：把 SSD 的“状态递推”视作一种天然的记忆机制，配合 PromptCoT 的结构化思维链，使得模型在每一步都能“带着前面的推理状态继续前进”，而不需要像 Transformer 那样每次都去“全表搜索”。这让长链推理的成本几乎不随长度增长，真正实现了“常数时间推理”。

### 实验与效果
- **测试任务**：AIME 2024、AIME 2025（美国数学竞赛高级题）以及 Livecodebench（代码推理基准）。这些任务都要求模型在长文本中保持严密的逻辑链，且答案往往是数值或代码。  
- **对比基线**：同尺度的 Transformer（7B）和混合模型（7B+少量注意力层），以及参数更大的 Gemma‑3‑27B。  
- **主要结果**：  
  - 在 AIME24 上 \ourmodel‑7B 超过同尺度 Transformer 2.6%（绝对提升），并且比 Gemma‑3‑27B 高出 2.6%。  
  - AIME25 上提升 0.6%，Livecodebench 上提升 3.0%。  
  - 推理速度方面，\ourmodel 在 4k token 序列上比 Transformer 快约 1.8×，显存占用仅为后者的 45%。  
- **消融实验**：  
  - 去掉第二阶段微调，性能下跌约 1.4%，说明高难度课程对最终表现关键。  
  - 用普通随机生成的 CoT 数据（不经过概念‑导向）训练，效果比 PromptCoT 低约 0.9%。  
  - 替换 SSD 为传统卷积层，显存仍固定但推理准确率下降约 2%。  
- **局限性**：论文承认模型仍在 7B 参数范围，面对更抽象的符号推理（如高等代数证明）仍有差距；此外，SSD 对极端超长序列（> 10k token）仍会出现数值漂移，需要额外的正则化手段。  

### 影响与延伸思考
这篇工作向社区展示了 **“状态空间模型 + 结构化微调”** 可以在不依赖自注意力的情况下实现高质量推理，直接冲击了“更大模型才是唯一出路”的传统观念。随后出现的几篇论文（如 *SSM‑CoT*、*Mamba‑Math*）都在尝试把 SSD 与不同的思维链生成方式结合，甚至把检索（retrieval）模块接入 SSD，以进一步提升事实性问答的准确度。对想继续深挖的读者，建议关注以下方向：  
1. **更大尺度的 SSD 训练**：探索 30B、70B 级别的 SSD 能否保持常数时间优势。  
2. **跨模态状态空间**：把视觉特征也映射到同一状态空间，做图文推理。  
3. **自适应课程微调**：让模型自行评估当前难度，动态生成下一个训练阶段的样本。  

### 一句话记住它
**用 Mamba‑2 的 SSD 取代自注意力，再配合两阶段 PromptCoT 课程式微调，就能让 7B 参数的模型在数学推理上跑赢 27B 的 Transformer。**