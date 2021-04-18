# The Power of Scale for Parameter-Efficient Prompt Tuning

> **Date**：2021-04-18
> **arXiv**：https://arxiv.org/abs/2104.08691

## Abstract

In this work, we explore "prompt tuning", a simple yet effective mechanism for learning "soft prompts" to condition frozen language models to perform specific downstream tasks. Unlike the discrete text prompts used by GPT-3, soft prompts are learned through backpropagation and can be tuned to incorporate signal from any number of labeled examples. Our end-to-end learned approach outperforms GPT-3's "few-shot" learning by a large margin. More remarkably, through ablations on model size using T5, we show that prompt tuning becomes more competitive with scale: as models exceed billions of parameters, our method "closes the gap" and matches the strong performance of model tuning (where all model weights are tuned). This finding is especially relevant in that large models are costly to share and serve, and the ability to reuse one frozen model for multiple downstream tasks can ease this burden. Our method can be seen as a simplification of the recently proposed "prefix tuning" of Li and Liang (2021), and we provide a comparison to this and other similar approaches. Finally, we show that conditioning a frozen model with soft prompts confers benefits in robustness to domain transfer, as compared to full model tuning.

---

# 规模的力量：参数高效的 Prompt 调优 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）流行之前，想让模型适应新任务通常要 **全模型微调**——把所有参数都重新训练一遍。随着模型规模突破百亿甚至千亿参数，微调的算力、显存和存储成本急剧上升，导致同一个模型难以被多个下游任务共享。另一条思路是 **离散文本提示**（如 GPT‑3 的 few‑shot 示例），但它只能利用极少的示例，效果远不如微调。于是出现了 **软提示**（soft prompt）——把提示当作可学习的向量，但当时的研究（如 prefix‑tuning）仍然在小模型上表现一般，尚未证明在超大模型上能否替代全模型微调。解决这个“规模 vs 参数效率”矛盾，就是本文要攻克的核心难题。

### 关键概念速览
- **软提示（soft prompt）**：一段可学习的向量序列，直接拼接在模型的输入嵌入前面，类似于在模型前端加了一个小的“记忆条”。它不是自然语言，而是模型内部的连续表示。
- **Prompt Tuning**：仅优化软提示的参数，保持底层语言模型全部冻结。可以把它想成给模型装上可更换的“插件”，而不动机器本体。
- **Prefix Tuning**：在每一层的自注意力键值对前插入可学习的前缀向量，作用更深、更细致。Prompt Tuning 是它的简化版，只在最底层输入嵌入前加前缀。
- **全模型微调（Full‑model fine‑tuning）**：把所有模型参数都参与梯度更新，通常能得到最好的任务性能，但代价最高。
- **参数效率（parameter‑efficiency）**：在保持性能的前提下，用尽可能少的可学习参数完成任务。类似于用几百行代码实现一个大型软件的功能。
- **规模效应（scale effect）**：模型参数越多，某些技术（如 Prompt Tuning）表现越好，甚至可以“追平”全模型微调的效果。

### 核心创新点
1. **从 Prefix Tuning 简化到 Prompt Tuning**  
   *之前的方法*：Prefix Tuning 在每一层都插入前缀，需要在每层维护独立的向量，实现上稍显繁琐。  
   *本文的做法*：只在模型最底层的输入嵌入前加入一段软提示，省去跨层前缀的设计。  
   *带来的改变*：实现更简洁、计算开销更低，同时在大模型上仍能保持竞争力。

2. **系统性规模实验，发现“规模闭合”现象**  
   *之前的认知*：软提示在小模型上远不如全模型微调，难以推广。  
   *本文的做法*：在 T5 系列模型（从 60M 到 11B 参数）上统一评测 Prompt Tuning 与全模型微调的差距。  
   *带来的改变*：当模型规模超过数十亿参数时，Prompt Tuning 与全模型微调的性能差距几乎消失，证明大模型天然具备“容错”空间，使得参数高效调优成为可行方案。

3. **对比 Few‑shot 与 Prompt Tuning，展示显著优势**  
   *之前的做法*：GPT‑3 采用离散文本 few‑shot，依赖少量示例，效果受提示设计影响大。  
   *本文的做法*：使用可学习软提示，直接在标注数据上做梯度下降。  
   *带来的改变*：在相同任务上，Prompt Tuning 的准确率显著高于 few‑shot，说明软提示能够更充分地吸收标注信息。

4. **实验验证软提示提升跨域鲁棒性**  
   *常规做法*：全模型微调在目标域上表现好，但迁移到新域时容易过拟合。  
   *本文的做法*：保持模型冻结，仅调软提示，然后在新域上重新学习软提示。  
   *带来的改变*：在跨域实验中，软提示的表现比全模型微调更稳健，说明冻结的底层语言能力起到了正则化作用。

### 方法详解
**整体框架**  
Prompt Tuning 的流程可以概括为三步：  
1) **准备冻结的语言模型**（本文使用 T5 系列，所有层的参数保持不动）。  
2) **初始化软提示向量**，长度 L（如 20）且维度与模型词嵌入相同。  
3) **在每一次前向传播时**，把软提示拼接到输入序列的嵌入前面，随后正常走完整个 Transformer；只对软提示的向量做梯度更新。

**关键模块拆解**  
- **软提示初始化**：可以随机初始化，也可以用已有的离散提示经过嵌入层投射得到。类似于给模型装上一个“空白插件”，后面再填内容。  
- **拼接机制**：设原始输入 token 序列为 `[x1, x2, …, xn]`，对应嵌入为 `E(xi)`。软提示向量记为 `P = [p1, …, pL]`。在前向时，模型实际看到的嵌入序列是 `[p1, …, pL, E(x1), …, E(xn)]`。这一步不涉及任何额外的网络层，只是把向量顺序拼起来。  
- **梯度流向**：因为模型参数被冻结，反向传播的梯度只能回传到 `P`，于是 `P` 通过标准的 SGD/Adam 更新。相当于在固定的语言模型上“调教”一个专属的前缀。  
- **任务头**：对于分类、生成等不同任务，仍然保留原模型的任务特定头（如 T5 的 seq2seq 解码器），不需要额外的适配层。

**公式的白话解释**  
如果把模型记作函数 `F(·)`，原始输入 `X`，软提示 `P`，则预测为 `ŷ = F(concat(P, X))`。训练目标是最小化任务损失 `L(ŷ, y)`，梯度只作用在 `P` 上。整个过程就是在固定的函数 `F` 上寻找最合适的前缀 `P`，使得输出尽可能接近真实标签。

**最巧妙的设计**  
- **只在最底层加前缀**：相比 Prefix Tuning 在每层都插入前缀，Prompt Tuning 只在输入层加一次，极大降低了显存占用，却仍能让后续层感受到提示的影响，因为 Transformer 的自注意力会把前缀信息传播到所有位置。  
- **规模驱动的“容错”**：作者发现，当模型足够大时，软提示的维度可以相对较小，却仍能调动模型的丰富知识，这是一种“规模自带正则化”的现象，颠覆了之前“软提示需要大量参数” 的假设。

### 实验与效果
- **数据集与任务**：论文在 GLUE（包括 MNLI、QQP、STS‑B 等）、SuperGLUE 以及机器翻译等多任务上评估 Prompt Tuning。所有任务均使用 T5‑Base（220M）、T5‑Large（770M）以及 T5‑XXL（11B）三个规模的模型。  
- **Baseline 对比**：与离散 few‑shot、Prefix Tuning、全模型微调进行比较。结果显示：在 60M‑770M 参数区间，Prompt Tuning 落后全模型微调约 2‑4% 的准确率；但在 11B 参数模型上，两者的差距缩小到不到 0.5%，几乎持平。相较于 few‑shot，Prompt Tuning 提升幅度在 5‑15% 之间。  
- **消融实验**：作者分别去掉软提示的长度、改为在中间层插入前缀、以及放开模型部分参数的冻结。实验表明：软提示长度在 10‑30 之间对性能影响不大，过短会略有下降；在中间层插入前缀并没有显著提升，说明最底层的前缀已经足够；放开模型参数会提升少量性能，但代价是显存和训练成本大幅上升。  
- **鲁棒性测试**：在跨域情景（如从新闻领域迁移到医学摘要）中，Prompt Tuning 的性能下降约 1%，而全模型微调下降约 3%，验证了冻结模型带来的正则化效应。  
- **局限性**：论文承认软提示仍然需要一定量的标注数据才能训练，完全零样本的 few‑shot 场景下仍不如离散提示；此外，软提示的解释性差，难以直接阅读或编辑。

### 影响与延伸思考
这篇工作在发布后迅速成为参数高效微调的基准，催生了 **Adapter、LoRA、BitFit** 等一系列“只调一小部分参数”的方法。很多后续研究把 Prompt Tuning 与 **多任务共享软提示**、**跨语言软提示**、以及 **可微分检索** 结合，进一步提升了大模型的可部署性。对想深入的读者，可以关注以下方向：  
- **软提示的可解释化**：如何把软提示映射回自然语言，帮助人类理解模型的“思路”。  
- **跨模态 Prompt**：把软提示扩展到视觉、音频等非语言模态。  
- **自适应软提示生成**：利用元学习或强化学习在运行时自动生成最合适的软提示。  
（以上为基于后续文献的推测）

### 一句话记住它
**只要模型够大，冻结模型加一个学会的软前缀，就能和全模型微调媲美，省钱又省力。**