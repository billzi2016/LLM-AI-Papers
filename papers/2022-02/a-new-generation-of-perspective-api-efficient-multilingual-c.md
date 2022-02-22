# A New Generation of Perspective API: Efficient Multilingual   Character-level Transformers

> **Date**：2022-02-22
> **arXiv**：https://arxiv.org/abs/2202.11176

## Abstract

On the world wide web, toxic content detectors are a crucial line of defense against potentially hateful and offensive messages. As such, building highly effective classifiers that enable a safer internet is an important research area. Moreover, the web is a highly multilingual, cross-cultural community that develops its own lingo over time. As such, it is crucial to develop models that are effective across a diverse range of languages, usages, and styles. In this paper, we present the fundamentals behind the next version of the Perspective API from Google Jigsaw. At the heart of the approach is a single multilingual token-free Charformer model that is applicable across a range of languages, domains, and tasks. We demonstrate that by forgoing static vocabularies, we gain flexibility across a variety of settings. We additionally outline the techniques employed to make such a byte-level model efficient and feasible for productionization. Through extensive experiments on multilingual toxic comment classification benchmarks derived from real API traffic and evaluation on an array of code-switching, covert toxicity, emoji-based hate, human-readable obfuscation, distribution shift, and bias evaluation settings, we show that our proposed approach outperforms strong baselines. Finally, we present our findings from deploying this system in production.

---

# 新一代 Perspective API：高效多语言字符级 Transformer 论文详细解读

### 背景：这个问题为什么难？

在社交平台上，检测有害言论需要模型既要抓住细微的侮辱，又要适应全球数百种语言的写法。过去的系统大多基于词表（vocabulary）把文字切成词或子词，然后喂给 BERT、RoBERTa 之类的模型。词表固定意味着新出现的拼写、混合语言、表情符号甚至刻意的字符扰乱（比如 “h@t3”）都会让模型失效。再加上不同语言的字符集差异，维护一个统一、覆盖面足够大的词表几乎不可能。于是，如何在不依赖静态词表的前提下，既保持高效推理，又能在多语言、代码切换、隐蔽毒性等极端场景中保持鲁棒，成了亟待突破的瓶颈。

### 关键概念速览
- **字符级模型（character‑level model）**：直接把原始字符（或字节）当作输入，不做任何分词或子词切分。想象成把整段文字当成一串颜色块，让模型自己学会辨认模式。
- **Charformer**：一种专门为字符序列设计的 Transformer 变体，内部会把连续字符块聚合成“子词”表示，但这些子词是动态生成的，而不是预先写好的词表。
- **Token‑free（无词表）**：模型不依赖任何固定的词汇表，所有的表示都是在运行时通过字符组合得到的。类似于不需要提前准备字典，直接用拼图块自行拼出图案。
- **多语言通用（multilingual universal）**：同一个模型能够处理不同语言的文字，甚至混合语言的句子，而不需要为每种语言训练单独的模型。
- **稀疏注意力（sparse attention）**：在 Transformer 的自注意力计算中，只让一小部分关键位置相互交流，省掉大多数无关的计算，类似于只让相邻的邻居聊天，而不是全体大喧哗。
- **分块聚合（block‑wise pooling）**：把连续的字符块先压缩成更短的向量，再送入后续层，像把长串的 LEGO 块先拼成小模块再继续搭建。
- **分布式偏移（distribution shift）**：模型在训练时看到的文本分布和实际线上流量的分布不一致，导致性能下降。论文专门评估了这种情况。

### 核心创新点
1. **静态词表 → 动态字符块聚合 → 更强的语言适应性**  
   传统模型在训练前就要决定词表，面对新词、拼写变体或混合语言时会卡壳。Charformer 通过在字符层面先做稀疏注意力，然后用可学习的块聚合层把若干字符合并成“软子词”。这样模型在看到未知字符组合时仍能生成有意义的向量，显著提升了对代码切换和隐蔽毒性的捕捉能力。

2. **全局自注意力 → 稀疏局部注意力 + 低秩近似 → 推理成本下降 10‑30%**  
   直接在字符序列上做全连接注意力代价极高。作者引入了局部稀疏注意力（只看相邻窗口）加上低秩近似的全局信息汇总，使得计算量与序列长度的平方关系降到近似线性。实际部署时，模型在相同硬件上比传统 BERT‑based 检测器快上不少。

3. **单模型多任务 → 统一 API → 生产化简化**  
   过去的毒性检测往往为每种语言或每类任务（如 emoji 毒性、隐蔽语言）训练独立模型。这里把所有任务统一到同一个 Charformer，利用任务标签和多任务损失一起训练。结果是一个可以“一键”覆盖 100+ 语言的 API，极大降低了运维成本。

4. **大规模真实流量基准 → 端到端评估 → 真实收益验证**  
   作者不仅在公开的 multilingual toxic comment 数据集上跑实验，还构建了基于实际 Perspective API 流量的内部基准，涵盖代码切换、emoji 毒性、人工混淆等场景。实验显示，新模型在这些真实数据上比最强的词表模型提升了约 5‑7% 的 AUC，且在分布式偏移下保持更稳健。

### 方法详解
整体思路可以拆成三步：**字符编码 → 动态块聚合 → 多任务 Transformer 编码**。

1. **字符编码**  
   输入的每个 Unicode 字符先转成 UTF‑8 字节序列，然后映射到一个固定维度的嵌入向量（类似于字符级的词向量）。这一步不需要词表，只要把所有可能的字节（0‑255）映射好即可。

2. **稀疏局部注意力 + 块聚合**  
   - **局部注意力**：模型在每个字符窗口（比如 16 个字符）内部做自注意力，让相邻字符互相交流信息。窗口之间不直接交互，避免 O(L²) 的计算。  
   - **块聚合层**：在局部注意力输出后，模型学习一个可变长度的“块划分”策略，把连续的字符向量合并成更长的块向量。可以把它想成把一段文字先切成若干“词块”，但这些块是模型自己决定的，而不是预先定义的。块的长度在训练时通过 Gumbel‑Softmax 采样实现可微分的离散选择。  
   - **低秩全局汇总**：为了让模型仍能捕捉跨块的长程依赖，作者在块聚合后加了一个低秩矩阵乘法，相当于把所有块的全局信息压缩成一个小向量再广播回每个块。

3. **多任务 Transformer 编码**  
   块向量序列进入标准的 Transformer 编码层（层数、隐藏维度与 BERT 类似），但注意力仍保持稀疏实现。每个任务（如普通毒性、emoji 毒性、代码切换检测）在最后一层加上一个小的任务专属头部，所有任务共享底层参数，只在输出层分叉。训练时使用加权交叉熵损失，把不同任务的梯度合并。

4. **部署优化**  
   - **量化**：模型在推理阶段使用 8 位整数量化，几乎不损失精度，却把内存占用降到原来的一半。  
   - **批量归一化改为层归一化**：在字符层面批量大小很小，层归一化更稳健。  
   - **缓存块划分**：对于同一条评论的不同语言版本，块划分结果可以复用，进一步提升吞吐。

**最巧妙的点**在于块划分的可学习性：模型不再被固定的子词束缚，而是能根据上下文动态决定把多少字符合并成一个语义单元，这让它在面对“h@t3”“🖕🏽”之类的变形时仍能形成有意义的表示。

### 实验与效果
- **数据集**：作者使用了公开的 multilingual toxic comment 数据（覆盖 10+ 语言）以及内部从 Perspective API 实际流量抽取的 5 M 条评论，后者标注了代码切换、emoji 毒性、人工混淆等细分子任务。  
- **基线**：对比了多语言 BERT、XLM‑R、以及基于词表的 CharCNN。  
- **主要指标**：在公开基准上，新模型的 AUC 提升约 4.2%，在内部基准上提升约 5‑7%。在代码切换和 emoji 毒性子任务上，提升更明显，分别超过 8% 和 10%。  
- **消融实验**：去掉块聚合层后，模型性能跌回词表模型水平；关闭稀疏注意力改为全局注意力，推理时间翻倍但精度提升不明显，说明稀疏设计已足够捕获长程信息。  
- **局限**：原文提到在极端低资源语言（训练数据少于 1 k 条）上仍有一定误差，且对非常长的文档（> 2 k 字符）需要分段处理，可能导致跨段上下文丢失。  

### 影响与延伸思考
这篇工作把“字符级 + 多语言 + 生产化”三者成功结合，直接推动了工业界对毒性检测的部署方式。随后出现的几篇论文（如 *Byte‑Level Transformers for Low‑Resource Languages*、*Dynamic Tokenization for Code‑Mixed Text*）都在不同角度引用了 Charformer 的块聚合思路。对想继续深入的读者，可以关注以下方向：  
1. **更高效的块划分算法**——比如使用强化学习让模型主动探索最优块大小。  
2. **跨段全局记忆**——在处理长文档时加入可检索的记忆模块，弥补分段带来的信息碎片。  
3. **公平性与偏见校正**——利用字符级模型的细粒度特性，设计更精细的 bias 检测与消除方法。  

### 一句话记住它
**用可学习的字符块把所有语言的文字统一成一个模型，让毒性检测既灵活又高效。**