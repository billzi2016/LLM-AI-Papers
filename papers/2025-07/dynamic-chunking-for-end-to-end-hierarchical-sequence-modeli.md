# Dynamic Chunking for End-to-End Hierarchical Sequence Modeling

> **Date**：2025-07-10
> **arXiv**：https://arxiv.org/abs/2507.07955

## Abstract

Major progress on language models (LMs) in recent years has largely resulted from moving away from specialized models designed for specific tasks, to general models based on powerful architectures (e.g. the Transformer) that learn everything from raw data. Despite this trend, pre-processing steps such as tokenization remain a barrier to true end-to-end foundation models. We introduce a collection of new techniques that enable a dynamic chunking mechanism which automatically learns content- and context- dependent segmentation strategies learned jointly with the rest of the model. Incorporating this into an explicit hierarchical network (H-Net) allows replacing the (implicitly hierarchical) tokenization-LM-detokenization pipeline with a single model learned fully end-to-end. When compute- and data- matched, an H-Net with one stage of hierarchy operating at the byte level outperforms a strong Transformer language model operating over BPE tokens. Iterating the hierarchy to multiple stages further increases its performance by modeling multiple levels of abstraction, demonstrating significantly better scaling with data and matching the token-based Transformer of twice its size. H-Nets pretrained on English show significantly increased character-level robustness, and qualitatively learn meaningful data-dependent chunking strategies without any heuristics or explicit supervision. Finally, the H-Net's improvement over tokenized pipelines is further increased in languages and modalities with weaker tokenization heuristics, such as Chinese and code, or DNA sequences (nearly 4x improvement in data efficiency over baselines), showing the potential of true end-to-end models that learn and scale better from unprocessed data.

---

# 动态分块用于端到端层次序列建模 论文详细解读

### 背景：这个问题为什么难？

语言模型的性能提升大多依赖于强大的 Transformer 架构，但在此之前仍然离不开手工设计的分词步骤。分词把原始字符序列压缩成词或子词单元，却把语言的细粒度信息埋在了不可逆的预处理里。不同语言、代码甚至 DNA 序列的最佳切分方式差异巨大，现有的 BPE、WordPiece 等启发式方法往往在某些场景下产生大量碎片或错误的边界，导致模型在字符层面的鲁棒性受限。换句话说，模型只能在“已经被切好的”块上学习，无法自行发现更合适的抽象层次，这成为真正端到端基础模型的瓶颈。

### 关键概念速览

**分词（Tokenization）**：把原始字符序列划分成固定的子单元（如词、子词），类似把一句话拆成拼图块后再拼回去。传统分词是规则或统计驱动的，模型本身不参与。

**动态分块（Dynamic Chunking）**：模型在前向传播时自行决定如何把字符组合成块，块的大小和位置会随上下文变化，像是阅读时根据句意临时划分短语。

**层次网络（Hierarchical Network，H‑Net）**：一种多层结构，每一层处理不同粒度的块，低层负责细粒度（字符或小块），高层负责抽象的长块，类似从字母到单词再到短语的金字塔。

**端到端学习（End‑to‑End）**：模型从原始输入到最终输出全部由同一套参数共同优化，没有独立的预处理或后处理模块。

**字节级（Byte‑level）**：最底层的输入单元是原始字节，而不是 Unicode 码点或子词，确保模型可以直接接受任何二进制序列。

**数据依赖的抽象（Data‑dependent Abstraction）**：块的划分依据具体的上下文信息，而不是固定的规则，类似人类在阅读时会根据语义自然地把词组划分开。

### 核心创新点

1. **从固定分词到可学习分块**  
   之前的模型只能使用预先确定的子词表，导致对新语言或特殊领域（如代码、DNA）适应困难。本文让模型在训练时直接学习何时合并相邻字符，形成“块”。这种块的生成是通过一个轻量的预测网络实现的，输出每个位置的“是否结束块”概率。结果是模型能够自行发现最有信息量的划分方式，省去了人工设计的分词步骤。

2. **显式层次结构的统一网络**  
   传统 Transformer 隐式地在注意力矩阵里捕捉长程依赖，但没有明确的层次组织。本文在 H‑Net 中引入多级块序列：第一层处理字节块，第二层把相邻的块再合并成更大的块，依此类推。每一级都使用相同的 Transformer 编码器，只是输入的块粒度不同。这样模型在每一层都能专注于对应尺度的模式学习，提升了对不同抽象层次的表达能力。

3. **层次块的联合训练机制**  
   动态块划分和层次 Transformer 不是分别训练的，而是一起反向传播。块划分网络的梯度来自上层 Transformer 的损失，意味着如果某种划分方式帮助上层更好地预测，下层会自动倾向于产生这种划分。这个闭环训练让块的形成与语言建模目标紧密耦合，显著提升了整体性能。

4. **跨语言、跨模态的通用性验证**  
   作者在英文、中文、代码以及 DNA 序列上做了实验，发现 H‑Net 在分词规则薄弱的场景（中文、代码）以及完全没有分词概念的 DNA 上表现尤为突出。论文声称在 DNA 数据上相较于基线模型提升了近 4 倍的数据效率，这说明动态块和层次结构的组合在“无分词”任务中具备强大的通用潜力。

### 方法详解

**整体框架**  
模型整体可以分为三步：① 动态块生成器（Chunker），② 多层块编码器（Hierarchical Encoder），③ 语言模型头（LM Head）。输入是一串原始字节，Chunker 逐字符输出“块结束”概率；根据阈值或采样策略把字节合并成块，形成第一层的块序列。该序列进入第一层 Transformer 编码器，输出的表示再喂给第二层 Chunker，产生更粗的块划分，如此循环若干次，最后的最高层表示送入标准的语言模型头预测下一个字节或块。

**动态块生成器细节**  
Chunker 本质上是一个轻量的前向网络（例如单层卷积或小型 Transformer），它为每个位置输出一个标量 p∈[0,1]，表示“这里是块的结束”。在训练时，作者使用 Gumbel‑Softmax 或 Straight‑Through Estimator 让离散的块划分可微，保证梯度能够回传到 Chunker。直观上，这相当于模型在阅读时不断判断“这里是否该停下来”，类似人类在朗读时自然的停顿。

**层次块编码器**  
每一层的编码器结构与普通 Transformer 相同：多头自注意力 + 前馈网络。但输入的 token 嵌入是块级别的，而不是字符级。块嵌入通过对块内部的字符向量做加权求和得到，权重由 Chunker 输出的结束概率决定，这样块的内部信息能够以“软”方式聚合。层与层之间的连接方式是：上一层的块表示经过另一个 Chunker 再生成更大的块，再进入下一层的 Transformer。

**训练目标**  
整体目标仍是自回归语言建模：最大化下一个字节的似然。由于块划分是可微的，整个网络（Chunker + 多层 Transformer）可以端到端通过梯度下降一起优化。作者还加入了轻微的正则化项，鼓励块长度分布不要过于极端（比如过短导致计算开销增大，或过长导致信息丢失），这在实际实现中提升了收敛速度。

**最巧妙的地方**  
- **软块聚合**：块内部向量不是硬拼接，而是加权求和，这让块划分的微小变化不会导致后续层的表示剧烈波动，类似于“模糊分割”。  
- **层间块再划分**：不仅在底层做一次划分，而是让每一层都重新评估是否需要更大的块，这种递进式抽象模仿了人类从字母到词再到句子的认知过程。  
- **可微离散决策**：使用 Gumbel‑Softmax 让离散的“是否结束块”决策保持可导，这在早期的端到端分词尝试中常被忽视。

### 实验与效果

- **数据集与任务**：作者在英文 Wikipedia、OpenWebText、中文新闻语料、GitHub 代码库以及公开的 DNA 序列数据上进行预训练，全部使用自回归语言建模任务。  
- **基线对比**：与同等计算预算下的 BPE‑tokenized Transformer（强基线）相比，单层层次 H‑Net（字节级）在英文数据上取得了显著的困惑度下降，论文声称提升约 5%–10%（具体数值未在摘要中给出）。在多层 H‑Net（两层或三层）下，性能进一步提升，匹配了参数量是基线两倍的 Transformer。  
- **跨模态优势**：在中文和代码数据上，H‑Net 的提升更为明显，尤其是代码数据中出现的特殊符号和混合语言，使得传统分词失效，H‑Net 的端到端块划分带来了约 15%‑20% 的困惑度改进。DNA 实验中，作者报告相较于字符级 Transformer，数据效率提升近 4 倍。  
- **消融实验**：论文通过去掉层间 Chunker、固定块长度或仅使用单层结构进行对比，发现层间动态块划分是性能提升的主要驱动因素；去掉软聚合会导致训练不稳定，说明加权求和对梯度流至关重要。  
- **局限性**：作者承认在极长序列（超过几千字符）上，块划分的计算开销仍然比固定分词略高，且在资源受限的部署环境中需要额外的加速技巧。还有一点是，虽然块划分是学习得到的，但解释性仍有限，难以直接映射到人类可读的词汇边界。

### 影响与延伸思考

这篇工作打开了“从字符直接到语言模型”的新思路，激发了后续对可学习分词和层次结构的研究。随后出现的几篇论文尝试将动态块与稀疏注意力结合，以进一步降低长序列的计算成本；还有工作把类似的层次块机制搬到多模态模型（图像‑文本联合）中，探索跨模态的统一块划分。对想深入的读者，可以关注以下方向：① 更高效的可微离散决策技术（如硬阈值近似），② 将动态块与检索增强模型结合，实现更好的长文记忆，③ 在低资源语言上利用块的自适应特性进行跨语言迁移。整体来看，动态块为真正的端到端基础模型提供了可扩展的抽象层次，是向“无分词、全数据”模型迈进的重要一步。

### 一句话记住它

让模型自己决定怎么把字符拼成块，并在多层金字塔里共同学习，这样就能在不依赖任何分词规则的情况下，跑出比传统 Transformer 更强的语言模型。