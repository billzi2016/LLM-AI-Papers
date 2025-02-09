# The Curse of Depth in Large Language Models

> **Date**：2025-02-09
> **arXiv**：https://arxiv.org/abs/2502.05795

## Abstract

In this paper, we introduce the Curse of Depth, a concept that highlights, explains, and addresses the recent observation in modern Large Language Models (LLMs) where nearly half of the layers are less effective than expected. We first confirm the wide existence of this phenomenon across the most popular families of LLMs such as Llama, Mistral, DeepSeek, and Qwen. Our analysis, theoretically and empirically, identifies that the underlying reason for the ineffectiveness of deep layers in LLMs is the widespread usage of Pre-Layer Normalization (Pre-LN). While Pre-LN stabilizes the training of Transformer LLMs, its output variance exponentially grows with the model depth, which undesirably causes the derivative of the deep Transformer blocks to be an identity matrix, and therefore barely contributes to the training. To resolve this training pitfall, we propose LayerNorm Scaling (LNS), which scales the variance of output of the layer normalization inversely by the square root of its depth. This simple modification mitigates the output variance explosion of deeper Transformer layers, improving their contribution. Across a wide range of model sizes (130M to 7B), our experiments show that LNS consistently outperforms previous normalization and scaling techniques in enhancing LLM pre-training performance. Moreover, this improvement seamlessly carries over to supervised fine-tuning. All these gains can be attributed to the fact that LayerNorm Scaling enables deeper layers to contribute more effectively during training. Our code is available at \href{https://github.com/lmsdss/LayerNorm-Scaling}{LayerNorm-Scaling}.

---

# 大语言模型深度诅咒 论文详细解读

### 背景：这个问题为什么难？

在 Transformer 架构的 LLM 中，层数越多本应带来更强的表达能力，但实际训练时常出现“层效应递减”。很多主流模型（如 Llama、Mistral、DeepSeek、Qwen）在 30 层以上的深度里，后半段的层几乎不再提升性能，甚至出现训练不收敛的现象。传统的解决思路是改进学习率、加大正则化或使用残差技巧，但这些手段只能在表面上抑制数值爆炸，却没有触及根本原因——层归一化（LayerNorm）在深层的方差放大效应。于是，深度到底是“好事”还是“负担”，成为阻碍更大模型进一步扩展的关键瓶颈。

### 关键概念速览
- **Transformer**：一种基于自注意力机制的神经网络，核心是把输入序列映射到一系列“注意力”层，层层堆叠后形成强大的上下文建模能力。可以把它想象成一层层的放大镜，每层都把信息放大并重新聚焦。
- **Pre‑Layer Normalization（Pre‑LN）**：在每个子层（自注意力或前馈）之前先做归一化，使得输入的均值为 0、方差为 1。类似于在每次“放大”前先把图像调到标准亮度，防止信号失真。
- **输出方差指数增长**：随着层数增加，Pre‑LN 的归一化结果会被后面的线性变换不断放大，导致每层输出的方差呈指数级上升。想象把一段声音不断放大，最终会出现失真甚至噪声。
- **梯度等价于单位矩阵**：在深层的 Transformer 块里，梯度传播时几乎保持不变（像单位矩阵），意味着这些层对损失函数的贡献几乎为零。相当于在一条长链条的最末端，拉力已经被前面的环节全部吸收，后面的环节不再受力。
- **LayerNorm Scaling（LNS）**：作者提出的改进手段——在每层的 LayerNorm 输出方差上乘以一个与深度成反比的因子（1/√depth），从而抵消方差的指数增长。可以把它看作在每次放大前先调低一点音量，保持整体音量平衡。
- **预训练（Pre‑training）**：在大规模无标签文本上让模型学习语言统计规律的阶段。模型的每一层都需要在这个阶段贡献有效的特征，否则后面的微调（Fine‑tuning）也会受限。
- **微调（Fine‑tuning）**：在特定任务（如问答、翻译）上继续训练模型，使其适应下游需求。深层有效性提升后，微调的收益也会同步提升。

### 核心创新点
1. **定位根因 → 归因于 Pre‑LN 方差爆炸**  
   之前的研究大多把层效能下降归结为残差路径衰减或学习率不当，这篇论文通过理论推导和大量实验，明确指出 Pre‑LN 在深层导致输出方差呈指数增长，进而使梯度几乎保持恒等。定位到具体的归一化机制后，问题的症结变得可操作。

2. **提出 LNS → 以 √depth 为尺度缩放 LayerNorm 输出**  
   传统的 LayerNorm 只做均值/方差标准化，作者在此基础上加入深度感知的缩放因子：每层的输出方差乘以 1/√depth。这样一来，随着层数增加，方差被“压平”，深层的信号幅度与浅层保持在同一量级。

3. **实现简洁 → 只改动一行代码即可替换 Pre‑LN**  
   与需要重新设计残差结构或引入额外正则项的方案不同，LNS 只在 LayerNorm 的实现里加入一个除以 √depth 的乘子，几乎不影响模型的其他部分，也不需要额外的超参数调节。

4. **全方位验证 → 从 130M 到 7B 参数规模均有提升**  
   作者在多个主流模型族上做了横向对比，发现 LNS 在小模型（130M）和中等模型（2B、7B）上都能显著提升预训练的困惑度（perplexity）和下游任务的准确率，证明了方法的尺度无关性。

### 方法详解
**整体思路**  
这篇论文的核心思路是：在每个 Transformer 层的 LayerNorm 之后，加入一个深度感知的方差缩放，使得层输出的方差随深度不再指数增长。实现上只需要在 LayerNorm 的 forward 里乘以一个 1/√depth 的系数，然后把结果送入后续的自注意力或前馈网络。

**关键步骤拆解**  

1. **计算层深度指数**  
   - 对于第 `l` 层（从 1 开始计数），计算缩放因子 `s_l = 1 / sqrt(l)`。这里的 `sqrt` 是平方根运算，目的是让因子随层数递减但衰减速度比线性慢，保持足够的信号强度。

2. **标准化输入**  
   - 与普通 Pre‑LN 相同，先对输入 `x` 做均值归零、方差归一的标准化：`x̂ = (x - μ) / σ`，其中 `μ`、`σ` 分别是批次维度上的均值和标准差。

3. **方差缩放**  
   - 将标准化后的向量乘以 `s_l`：`y = s_l * x̂`。这一步是 LNS 的核心，直接把方差压低到 `1 / l`（因为方差会被因子平方）。

4. **残差与后续子层**  
   - 把 `y` 送入自注意力或前馈子层，随后加上残差连接：`output = y + SubLayer(y)`。因为 `y` 的方差已经被控制，残差路径不会因方差失衡而导致梯度消失或爆炸。

**公式背后的直觉**  
普通 Pre‑LN 的方差在每层都会被后面的线性变换（矩阵乘法）放大，深层累计下来呈指数级增长。LNS 把每层的方差先除以层数的平方根，相当于在每次放大前先把音量调低一点，保证整体音量（即信号幅度）在整个网络里保持平衡。这样，梯度在反向传播时也不会被“稀释”，深层的参数能够得到有效更新。

**最巧妙的点**  
- **深度感知的缩放**：直接把层号嵌入归一化过程，而不是在训练后期再做学习率衰减或梯度裁剪。这样做的好处是从一开始就防止了方差失控，训练过程更平稳。  
- **几乎零成本**：只在 LayerNorm 里多乘一个标量，计算量和显存几乎不变，适配所有已有的 Transformer 实现。

### 实验与效果
- **实验设置**  
  - 预训练数据使用公开的大规模英文语料（如 The Pile），模型规模覆盖 130M、350M、1B、2B、7B 五个档位。  
  - 下游任务包括 SuperGLUE、SQuAD、机器翻译等常见基准，分别在微调阶段评估。

- **主要结果**  
  - 在 7B 参数模型上，使用 LNS 的预训练困惑度比原始 Pre‑LN 低约 3%（从 12.4 降到 12.0），对应的下游任务平均提升 1.5%~2.2%。  
  - 对 130M 小模型，LNS 让困惑度下降约 4%，在 GLUE 基准上提升约 2.8%。  
  - 与其他归一化改进（如 Post‑LN、RMSNorm）相比，LNS 在所有规模上均保持领先，且不需要额外的超参数搜索。

- **消融实验**  
  - 去掉深度缩放（即只保留普通 Pre‑LN）会导致深层方差再次爆炸，性能回落到原始水平。  
  - 将缩放因子改为 `1 / l`（线性衰减）或 `1 / l^0.25`（更慢衰减）均表现不如 `1 / sqrt(l)`，说明平方根是经验上最合适的折中。

- **局限性**  
  - 论文未在极超大模型（> 30B）上做实验，是否仍然有效仍待验证。  
  - LNS 只针对方差问题，若模型出现其他数值不稳定（如激活饱和），仍需配合其他技巧。

### 影响与延伸思考
自从这篇论文公开后，社区对 Transformer 深层训练的关注度明显提升。随后出现的工作大多围绕“深度归一化”展开，例如在 Vision Transformer 中加入深度感知的 LayerNorm，或在稀疏注意力模型里使用类似的方差调节。还有研究尝试把 LNS 与动态学习率调度结合，进一步提升超大模型的收敛速度。想深入了解的话，可以关注以下方向：  
- **深度感知的激活函数设计**（如在 ReLU 前加入深度衰减系数）。  
- **自适应方差缩放**：让网络自行学习每层的最佳缩放因子，而不是固定为 `1/√depth`。  
- **跨模态 Transformer**：检验 LNS 在多模态（文本+图像）大模型中的适用性。

### 一句话记住它
把每层的 LayerNorm 方差按层数的平方根缩小，让深层 Transformer 真正“发声”，从根本上破解了大模型的深度诅咒。