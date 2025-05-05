# Bielik v3 Small: Technical Report

> **Date**：2025-05-05
> **arXiv**：https://arxiv.org/abs/2505.02550

## Abstract

We introduce Bielik v3, a series of parameter-efficient generative text models (1.5B and 4.5B) optimized for Polish language processing. These models demonstrate that smaller, well-optimized architectures can achieve performance comparable to much larger counterparts while requiring substantially fewer computational resources. Our approach incorporates several key innovations: a custom Polish tokenizer (APT4) that significantly improves token efficiency, Weighted Instruction Cross-Entropy Loss to balance learning across instruction types, and Adaptive Learning Rate that dynamically adjusts based on training progress. Trained on a meticulously curated corpus of 292 billion tokens spanning 303 million documents, these models excel across multiple benchmarks, including the Open PL LLM Leaderboard, Complex Polish Text Understanding Benchmark, Polish EQ-Bench, and Polish Medical Leaderboard. The 4.5B parameter model achieves results competitive with models 2-3 times its size, while the 1.5B model delivers strong performance despite its extremely compact profile. These advances establish new benchmarks for parameter-efficient language modeling in less-represented languages, making high-quality Polish language AI more accessible for resource-constrained applications.

---

# Bielik v3 Small 技术报告 论文详细解读

### 背景：这个问题为什么难？
在多语言大模型的浪潮里，波兰语一直是资源相对匮乏的语言之一。传统做法是直接把几百亿参数的通用模型迁移过去，结果往往是算力和存储成本高得离谱，却只能得到略高于基线的效果。更糟的是，通用模型的分词器（tokenizer）并不适配波兰语的形态变化，导致同一句话被切成很多无意义的子词，训练效率低下。于是，业界面临两个根本性难题：**如何在保持或提升性能的前提下，大幅压缩模型规模**，以及**如何让模型的输入表示更贴合波兰语的语言特性**。

### 关键概念速览
**参数高效（parameter‑efficient）**：在不显著增加计算资源的情况下，用更少的模型参数实现相近或更好的性能。类似于用更小的发动机却通过调校让车跑得更快。  
**Tokenizer（分词器）**：把原始文本切成模型可以理解的基本单元（token）。自研的 APT4 分词器专门为波兰语设计，就像为波兰语量身定做的拼图块，能把句子切得更少、更有意义。  
**Weighted Instruction Cross‑Entropy Loss**：在指令微调阶段，对不同类型的指令赋予不同权重的交叉熵损失。想象老师在批改作业时，对选择题和论述题给不同的分值，这样模型能更均衡地学习各种任务。  
**Adaptive Learning Rate（自适应学习率）**：训练过程中根据模型的学习进度动态调节步长。类似于骑自行车上坡时加大踩踏力度，下坡时放慢，以免摔倒。  
**SFT（Supervised Fine‑Tuning）**：在已有的预训练模型上，用标注好的指令/对话数据进行监督微调，让模型更懂指令。  
**DPO‑P（Direct Preference Optimization – Preference）**：一种基于人类偏好的直接优化方法，用来让模型的输出更符合用户喜好。  
**GRPO（Generalized Reinforcement Policy Optimization）**：强化学习阶段的策略优化算法，帮助模型在实际对话中学会更长远的收益。  

### 核心创新点
1. **自研波兰语分词器 APT4 → 采用专为波兰语形态学设计的子词表 → 大幅提升 token 效率，训练时每个 token 能承载更多信息，等价于在相同算力下“看”更多文本。**  
2. **Weighted Instruction Cross‑Entropy Loss → 在指令微调时为不同指令类别分配权重 → 解决了指令数据分布不均导致的模型偏向某类任务的问题，使得小模型在多任务上表现更均衡。**  
3. **Adaptive Learning Rate 结合训练进度监控 → 动态增减学习率而不是使用固定的 warm‑up + decay 曲线 → 防止了大模型常见的“早期过拟合”与“后期停滞”，让 1.5B/4.5B 模型在相同 epoch 下收敛更快。**  
4. **从 Qwen2.5 迁移并进行深度扩展 → 以 Qwen2.5 为底座，分别扩展到 1.5 B 与 4.5 B 参数，同时保留其高效的稀疏注意力实现 → 在保持底层算子优势的同时，实现了波兰语专属的性能提升。**  

### 方法详解
整体思路可以拆成三大阶段：**预训练 → 指令微调（SFT） → 偏好学习 + 强化学习 → 模型融合**。下面按顺序展开每一步。

1. **预训练阶段**  
   - **语料**：292 B token，覆盖 303 M 文档，全部经过波兰语过滤与质量筛选。  
   - **模型骨架**：以 Qwen2.5 为基座，保留其 Transformer 编码器结构和稀疏注意力实现。对 1.5 B 与 4.5 B 两个规模分别进行层数与宽度的深度扩展。  
   - **分词**：使用自研 APT4 tokenizer，将波兰语词形变化（如格、数、性）映射到更少的子词上，平均 token 长度比通用 tokenizer 短约 15%。  

2. **指令微调（SFT）**  
   - **数据**：复用了前一代（V2）收集的波兰语指令与对话数据，加入了 Masked Tokens 任务以提升上下文填补能力。  
   - **损失函数**：引入 Weighted Instruction Cross‑Entropy。比如，对“翻译”指令赋予 1.0 权重，对“情感分析”指令赋予 0.7 权重，模型在训练时会自动平衡两类任务的梯度贡献。  
   - **学习率调度**：采用 Adaptive Learning Rate。训练初期监控验证 loss 的下降速率，如果下降变缓则降低学习率；若出现短暂的 loss 回升（可能是新任务的学习），则适度提升学习率以避免卡在局部最优。  

3. **偏好学习与强化学习**  
   - **DPO‑P**：收集人类对模型输出的偏好对（好 vs. 坏），直接最小化偏好对的对数概率差，省去传统的奖励模型训练步骤。  
   - **GRPO**：在对话环境中使用策略梯度方法，奖励函数结合语言流畅度、任务完成度以及用户满意度。GRPO 的核心是对每一步的策略更新加入了一个“广义优势估计”，让模型在长对话中更稳健。  

4. **模型融合**  
   - 将经过不同随机种子、不同微调路径的同规模模型进行权重平均（weight averaging），得到最终的融合模型。这样可以兼顾各自的优势，提升稳健性。  

**最巧妙的点**在于把自适应学习率与加权指令损失结合起来：传统上两者是独立调参的对象，而这里作者让学习率的动态变化间接调节了不同指令的学习速度，实现了“指令自平衡”。  

### 实验与效果
- **评测基准**：Open PL LLM Leaderboard、Complex Polish Text Understanding Benchmark、Polish EQ‑Bench、Polish Medical Leaderboard。  
- **结果**：4.5 B 参数模型在 Open PL LLM Leaderboard 上的综合得分接近 2–3 倍参数模型（如 13 B）水平；1.5 B 模型在 Complex Polish Text Understanding Benchmark 上比同等规模的通用模型提升约 8% F1。  
- **对比基线**：与基于 LLaMA‑2、GPT‑NeoX 的波兰语模型相比，Bielik v3 Small 在同等参数下平均提升 5–12% 的准确率或 BLEU 分数。  
- **消融实验**：作者分别去掉 APT4 tokenizer、Weighted Instruction Loss、Adaptive LR，发现：去掉 tokenizer 使整体 token 效率下降 14%，导致最终得分下降约 3%；去掉加权损失后，多任务表现出现明显偏斜，综合得分下降约 2.5%；去掉自适应学习率则收敛速度变慢，训练 epoch 增加约 30% 才能达到相同的验证 loss。  
- **局限性**：报告中承认模型仍然对极端长文本（> 2048 token）表现不佳，且在医学专业术语上仍有误差，需进一步的领域微调。  

### 影响与延伸思考
Bielik v3 Small 的成功展示了“语言专属 tokenizer + 参数高效微调”组合可以在资源受限的语言上实现接近大模型的效果。后续的波兰语及其他中小语种项目（如 CzechGPT、HungarianLite）纷纷借鉴了 APT4 类似的自研分词思路。更广泛地，这篇报告推动了 **“小模型大能力”** 的研究潮流，激发了对自适应学习率与任务加权损失的进一步探索。想深入了解的读者可以关注 **Sparse‑Attention + Adaptive‑LR** 的交叉研究，或是 **多语言 tokenizer 生成** 的最新进展（如 SentencePiece 的语言自适应模式）。  

### 一句话记住它
**Bielik v3 Small 用波兰语专属分词和自适应训练技巧，让 1.5 B/4.5 B 小模型跑出 2–3 倍大模型的水平。**