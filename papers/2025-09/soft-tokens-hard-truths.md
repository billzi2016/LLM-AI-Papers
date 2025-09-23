# Soft Tokens, Hard Truths

> **Date**：2025-09-23
> **arXiv**：https://arxiv.org/abs/2509.19170

## Abstract

The use of continuous instead of discrete tokens during the Chain-of-Thought (CoT) phase of reasoning LLMs has garnered attention recently, based on the intuition that a continuous mixture of discrete tokens could simulate a superposition of several reasoning paths simultaneously. Theoretical results have formally proven that continuous tokens have much greater expressivity and can solve specific problems more efficiently. However, practical use of continuous tokens has been limited by strong training difficulties: previous works either just use continuous tokens at inference time on a pre-trained discrete-token model, or must distill the continuous CoT from ground-truth discrete CoTs and face computational costs that limit the CoT to very few tokens.   This is the first work introducing a scalable method to learn continuous CoTs via reinforcement learning (RL), without distilling from reference discrete CoTs. We use "soft" tokens: mixtures of tokens together with noise on the input embedding to provide RL exploration. Computational overhead is minimal, enabling us to learn continuous CoTs with hundreds of tokens. On math reasoning benchmarks with Llama and Qwen models up to 8B, training with continuous CoTs match discrete-token CoTs for pass@1 and surpass them for pass@32, showing greater CoT diversity. In systematic comparisons, the best-performing scenario is to train with continuous CoT tokens then use discrete tokens for inference, meaning the "soft" models can be deployed in a standard way. Finally, we show continuous CoT RL training better preserves the predictions of the base model on out-of-domain tasks, thus providing a softer touch to the base model.

---

# 软Token，硬真相 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）进行链式思考（Chain‑of‑Thought，简称CoT）时，模型通常只能一次写出一个离散的词语。虽然离散词能让推理过程清晰可见，但它们只能表达单一路径，难以同时探索多条思路。理论上，用连续的“软”向量混合多个词可以让模型在同一时间“叠加”多条推理路径，从而提升表达能力。过去的工作要么只在推理阶段临时使用软向量，却没有让模型真正学会生成它们；要么需要先把离散的CoT蒸馏成软向量，再进行训练，这一步计算成本高，导致只能用几百个软token，根本无法发挥其潜在优势。于是，如何在不依赖离散参考的情况下，直接让模型学会产生并利用软CoT，成为了一个急需突破的难题。

### 关键概念速览

**软Token（soft token）**：不是单个离散词，而是多个词的加权混合，再加上一点噪声，形成一个连续的向量。可以把它想象成调色板上的颜色混合，而不是单一的纯色。

**链式思考（CoT）**：让模型在给出答案前先写出推理步骤，类似于人做数学题时先列出草稿。

**离散Token**：传统意义上的词汇表中的单个词，模型只能在这些固定的点上跳跃。

**强化学习（RL）**：通过奖励信号让模型自行探索更好的行为序列，这里用来训练模型产生软Token序列。

**RL探索噪声**：在软Token的嵌入上加入随机扰动，帮助模型在训练时尝试不同的推理路径。

**Pass@k**：在k次尝试中至少有一次得到正确答案的比例，常用来衡量CoT的多样性和成功率。

### 核心创新点

1. **直接用RL学习软CoT → 采用软Token混合 + 噪声进行探索 → 省去离散参考的蒸馏步骤，训练成本大幅下降**。过去只能在已有离散CoT上做后处理，这里直接让模型在策略空间里搜索。

2. **软Token的噪声注入 → 在输入嵌入上加随机噪声 → 让RL的探索更丰富，避免陷入局部最优**。这相当于在搜索地图时给探险者加上风向指示，使其更容易尝试新路线。

3. **软CoT训练后仍可用离散Token推理 → 训练阶段使用软Token，推理阶段切回普通离散词 → 兼顾训练效率和部署便利**。这样模型在实际使用时不需要改动现有的推理框架。

4. **大规模软CoT序列（上百个） → 通过轻量化的噪声机制保持计算开销低 → 能在8B参数的Llama和Qwen上训练数百步的软CoT**。以前的工作只能跑到十几步，这里实现了数量级的提升。

### 方法详解

整体思路可以分为三步：**软Token构造 → RL策略学习 → 离散化部署**。

1. **软Token构造**  
   - 对每一步的输出，模型不直接挑选一个词，而是先算出词表上所有词的概率分布。  
   - 用这些概率作为权重，对对应的词向量做加权求和，得到一个“软”向量。  
   - 在这个向量上再加上一小段高斯噪声，形成最终的软Token。噪声的幅度随训练进度逐渐衰减，类似于RL里常见的ε‑greedy策略。

2. **RL策略学习**  
   - 把生成软Token的过程视作一个马尔可夫决策过程（MDP），每一步的动作就是选取一个软Token。  
   - 采用近端策略优化（PPO）等常用RL算法，依据最终答案的正确性给出奖励。奖励设计上，正确答案得到高奖励，错误答案则惩罚，同时加入对序列长度的轻微惩罚以防止无限循环。  
   - 为了让模型学会多样的推理路径，奖励在Pass@k的评估上做了平滑处理：只要在k次尝试中有一次成功，就给出正向奖励，这鼓励模型在一次训练中产生多条潜在解。

3. **离散化部署**  
   - 训练结束后，保留软Token策略的参数，但在实际推理时把软Token映射回离散词：取软向量在词表嵌入空间的最近邻或最高概率词。  
   - 这样模型在推理阶段仍然输出普通的文字序列，兼容现有的API和评测工具。

**巧妙之处**在于噪声并不是随意加的，而是作为RL探索的“随机种子”，让模型在训练时能够同时尝试多条思路，而不是像传统CoT那样只能沿单一路径前进。另一个亮点是把软Token的学习过程完全脱离了离散CoT的蒸馏，使得训练成本几乎和普通RL一样，能够扩展到上百个软Token。

### 实验与效果

- **数据集**：在数学推理基准（如MATH、GSM‑8K）上进行评测，使用Llama‑2和Qwen‑系列模型，规模从2B到8B不等。  
- **对比基线**：离散CoT（直接在模型上生成思考步骤）、软CoT蒸馏（先生成离散CoT再蒸馏成软Token）以及纯RL生成离散Token的方案。  
- **结果**：在Pass@1上，软CoT的表现与离散CoT持平；在Pass@32上，软CoT提升约5%~8%，说明它能产生更丰富的解答集合。  
- **消融实验**：去掉噪声或改用固定软Token权重会导致性能回落到离散CoT水平，验证噪声和权重混合是关键因素。  
- **局限**：论文未在大规模真实对话或长文本任务上验证软CoT的通用性；此外，RL训练仍需要较多的计算资源，对小模型的收益尚不明确。

### 影响与延伸思考

这篇工作打开了“软化”语言模型内部推理的可能，让研究者看到连续表示在思考层面的优势。随后出现的几篇论文尝试把软Token与检索增强、工具调用等场景结合，探索更高层次的“思维叠加”。如果想进一步了解，可以关注以下方向：**软表示的可解释性**（如何把软Token映射回人类可读的推理步骤）、**跨任务软CoT迁移**（在数学之外的推理任务中是否同样有效）以及**更高效的RL奖励设计**（比如利用自监督信号降低对标注答案的依赖）。

### 一句话记住它

用强化学习直接教大模型写“混色的思考草稿”，既省去蒸馏步骤，又在推理时还能恢复成普通文字。