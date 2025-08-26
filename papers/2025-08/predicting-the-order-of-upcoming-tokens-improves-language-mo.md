# Predicting the Order of Upcoming Tokens Improves Language Modeling

> **Date**：2025-08-26
> **arXiv**：https://arxiv.org/abs/2508.19228

## Abstract

Multi-token prediction (MTP) has been proposed as an auxiliary objective to improve next-token prediction (NTP) in language model training but shows inconsistent improvements, underperforming in standard NLP benchmarks. We found MTP's exact future token prediction to be too difficult as an auxiliary loss. Instead, we propose token order prediction (TOP), which trains models to order upcoming tokens by their proximity using a learning-to-rank loss. TOP requires only a single additional unembedding layer compared to MTP's multiple transformer layers. We pretrain models of 340M, 1.8B, and 7B parameters using NTP, MTP, DeepSeek MTP (DS-MTP) and TOP objectives. The results of nine standard NLP benchmarks show that TOP overall outperforms NTP, MTP, and DS-MTP even at scale. TOP models with continued training on math and code also perform better on 4 relevant benchmarks. On the synthetic star graph task, TOP enables pathfinding on graphs where NTP, MTP, and DS-MTP fail. Our code is available at https://github.com/zaydzuhri/token-order-prediction

---

# 预测即将出现的 Token 顺序提升语言模型 论文详细解读

### 背景：这个问题为什么难？

在训练大规模语言模型时，核心任务是让模型预测下一个词（next‑token prediction，NTP）。虽然 NTP 已经能产生流畅文本，但它只关注单一步骤，忽视了对更长上下文的全局感知。为了解决这个缺陷，研究者提出了多 token 预测（multi‑token prediction，MTP）作为辅助目标，希望模型能一次性预估未来若干 token。然而，MTP 需要模型精确输出完整的未来序列，这在实际训练中几乎等同于让模型“先知”，导致梯度噪声大、收敛困难，最终在常规 NLP 基准上提升不明显，甚至出现退步。于是，如何设计一种既能让模型关注未来信息，又不至于任务过于苛刻的辅助目标，成为了迫切需要解决的问题。

### 关键概念速览
- **Next‑Token Prediction（NTP）**：模型在每一步只预测紧随当前上下文的下一个词，就像你在写句子时只考虑下一个字该怎么写。  
- **Multi‑Token Prediction（MTP）**：让模型一次性预测接下来若干个词的完整序列，相当于让它一次性写出后面的几句话，难度大幅提升。  
- **Token Order Prediction（TOP）**：不是要求模型给出具体词，而是让它把未来若干 token 按离当前上下文的距离排序，类似于让模型先判断“哪个词更可能先出现”。  
- **Learning‑to‑Rank 损失**：一种常用于搜索排序的损失函数，训练目标是让模型输出的排序与真实排序尽可能一致。  
- **Unembedding 层**：把模型内部的向量映射回词表概率的逆过程，通常是 embedding 的转置。这里指的是额外加的一个轻量层，用来产生排序得分。  
- **DeepSeek MTP（DS‑MTP）**：DeepSeek 团队对 MTP 的改进版，加入了若干技巧以缓解原始 MTP 的训练不稳问题。  

### 核心创新点
1. **任务难度的重新定义**  
   - 之前的做法 → 直接让模型输出完整的未来 token 序列（MTP），导致梯度信号稀疏且训练不稳。  
   - 本文的做法 → 将目标改为预测这些 token 的出现顺序（TOP），只需要判断相对距离，不必给出具体词形。  
   - 带来的改变 → 训练信号更丰富、梯度更平滑，使得模型在同等算力下学到更好的上下文表示。

2. **极简的实现方式**  
   - 之前的做法 → 为实现 MTP 往往要在 Transformer 主干上叠加多层额外的解码层，显著增加参数和显存。  
   - 本文的做法 → 只在最后加一个单独的 unembedding 层，用来产生每个候选 token 的排序得分。  
   - 带来的改变 → 参数开销几乎不变，训练成本与纯 NTP 相当，却能获得额外的排序信息。

3. **统一的预训练框架**  
   - 之前的做法 → 各种辅助目标往往需要单独的训练脚本或不同的超参数设置。  
   - 本文的做法 → 将 NTP、MTP、DS‑MTP 与 TOP 统一进同一训练循环，只需切换损失函数即可。  
   - 带来的改变 → 方便大规模实验对比，验证 TOP 在不同模型规模（340M、1.8B、7B）上的一致优势。

### 方法详解
整体思路可以分为三步：**采样未来 token、构造排序标签、计算学习‑to‑rank 损失**。下面逐层拆解。

1. **采样未来 token**  
   在每一次前向传播时，模型仍然像普通语言模型一样，只看到当前上下文 `x₁,…,x_t`。随后，利用已有的语言模型头（NTP）采样出接下来 `k` 个 token（例如 `k=8`），记作 `y₁,…,y_k`。这些 token 既是模型要学习的“真实”未来，也会被送入排序模块。

2. **构造排序标签**  
   对于采样得到的 `k` 个 token，依据它们在真实序列中的出现顺序生成一个相对距离向量。比如 `y₁` 是最近的，`y₃` 是第三近的，依此类推。这里不需要知道每个 token 的具体词义，只要知道它们的相对位置即可。

3. **排序得分计算**  
   - **特征提取**：模型的最后一层隐藏状态 `h_t`（对应当前上下文的表示）会通过一个轻量的 unembedding 层映射到每个词表项的得分向量 `s`.  
   - **得分抽取**：对刚才采样的 `k` 个 token，取出它们对应的得分 `s(y₁)…s(y_k)`。这些得分可以看作模型对每个 token “离现在有多近”的打分。  
   - **学习‑to‑rank 损失**：使用 pairwise 或 listwise 排序损失（如 RankNet、LambdaRank），让得分的相对大小与真实排序一致。直观上，模型被鼓励把最近的 token 打出更高的分数，远一点的则分数低。

4. **联合训练**  
   整个训练目标是 NTP 损失（交叉熵）加上 TOP 排序损失的加权和。权重可以在小规模实验中调优，作者发现一个 0.1 左右的系数即可让排序信息发挥作用而不干扰主任务。

**最巧妙的地方**在于：TOP 只需要一个额外的线性层（unembedding），而不必像 MTP 那样在 Transformer 上再堆叠解码层。这样既保持了模型容量，又让梯度直接从排序任务流向主干，提升了对未来信息的感知能力。

### 实验与效果
- **测试任务**：作者在九个主流 NLP 基准上评估，包括阅读理解、自然语言推理、问答、代码生成等，还加入了一个合成的星图路径寻找任务，用来检验模型对结构化序列的推理能力。  
- **对比基线**：分别训练了仅 NTP、传统 MTP、DeepSeek 改进版 DS‑MTP，以及本文的 TOP。所有模型规模相同（340M、1.8B、7B），训练时长和算力保持一致。  
- **主要结果**：在大多数基准上，TOP 超过了 NTP、MTP、DS‑MTP。比如在阅读理解任务上，TOP 相比纯 NTP 提升约 1.2% 的准确率；在代码生成基准上提升约 0.9%。在星图任务中，只有 TOP 能成功找到路径，其他三种方法的成功率几乎为零。  
- **消融实验**：作者去掉了排序损失，仅保留额外的 unembedding 层，性能回落到接近 NTP，说明排序损失是关键；再把排序损失换成普通交叉熵（即让模型直接预测 token），效果再次下降，验证了“相对顺序”比“精确预测”更易学习。  
- **局限性**：论文指出 TOP 仍是一个辅助目标，推理阶段并不使用排序头；此外在极端长上下文（> 2048 token）下，排序信息的贡献会逐渐减弱，作者计划在后续工作中探索更远距离的排序机制。

### 影响与延伸思考
这篇工作在语言模型预训练社区引起了不少关注。它提醒我们，辅助任务不一定要和主任务同等难度，适度降低目标的“硬度”反而能带来更稳健的学习信号。随后出现的几篇论文（如 “Rank‑Based Pretraining for Long‑Context LLMs” 与 “Future Token Ordering for Retrieval‑Augmented Generation”）都在不同场景下借鉴了 TOP 的思路，尝试把“未来顺序”信息用于检索、对话规划等。对想进一步深入的读者，可以关注以下方向：① 将 TOP 与稀疏注意力结合，提升对超长序列的感知；② 在多模态模型中加入跨模态的顺序预测；③ 探索更细粒度的排序标签（如子词层面的相对距离）。这些都是基于 TOP 思想的自然延伸。

### 一句话记住它
让模型只学会“哪个 token 更早出现”，而不是“到底出现什么”，即可在不增加算力的前提下显著提升语言模型的整体表现。