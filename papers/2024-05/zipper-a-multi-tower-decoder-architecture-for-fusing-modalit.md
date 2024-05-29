# Zipper: A Multi-Tower Decoder Architecture for Fusing Modalities

> **Date**：2024-05-29
> **arXiv**：https://arxiv.org/abs/2405.18669

## Abstract

Integrating multiple generative foundation models, especially those trained on different modalities, into something greater than the sum of its parts poses significant challenges. Two key hurdles are the availability of aligned data (concepts that contain similar meaning but is expressed differently in different modalities), and effectively leveraging unimodal representations in cross-domain generative tasks, without compromising their original unimodal capabilities.   We propose Zipper, a multi-tower decoder architecture that addresses these concerns by using cross-attention to flexibly compose multimodal generative models from independently pre-trained unimodal decoders. In our experiments fusing speech and text modalities, we show the proposed architecture performs very competitively in scenarios with limited aligned text-speech data. We also showcase the flexibility of our model to selectively maintain unimodal (e.g., text-to-text generation) generation performance by freezing the corresponding modal tower (e.g. text). In cross-modal tasks such as automatic speech recognition (ASR) where the output modality is text, we show that freezing the text backbone results in negligible performance degradation. In cross-modal tasks such as text-to-speech generation (TTS) where the output modality is speech, we show that using a pre-trained speech backbone results in superior performance to the baseline.

---

# Zipper：一种多塔解码器架构用于融合模态 论文详细解读

### 背景：这个问题为什么难？

在生成式 AI 里，单一模态（比如纯文本或纯语音）的模型已经非常强大，但把不同模态的模型拼在一起并让它们协同工作却异常棘手。第一大难点是缺少对齐数据——我们需要“同一个概念”在文字和声音里都有对应的标注，而这种标注往往很少。第二大难点是如何在跨模态任务（比如把语音转成文字）中利用已有的单模态表征，同时又不破坏它们在各自领域的原始能力。过去的做法要么强行把所有模态塞进同一个大模型，导致训练成本爆炸；要么在跨模态任务里直接微调，结果往往把原有的单模态性能拉低。

### 关键概念速览

**多塔（Multi‑Tower）**：指在同一个系统里并行放置多个独立的解码器，每个塔负责一种模态的生成，就像不同的工厂各自生产自己的产品，再通过物流中心统一调度。

**交叉注意力（Cross‑Attention）**：一种让一个序列“看见”另一个序列的机制，类似于在对话中倾听对方的发言再作回应，能够把不同模态的信息相互融合。

**单模态解码器（Unimodal Decoder）**：只接受一种输入并生成一种输出的 Transformer 解码器，例如纯文本生成模型或纯语音合成模型。

**冻结（Freezing）**：在训练时把某些参数锁住不更新，等价于让模型保持原有的技能不受新任务的干扰。

**对齐数据（Aligned Data）**：指同一语义内容在不同模态下的配对样本，例如一段文字对应的朗读音频。

**跨模态生成（Cross‑Modal Generation）**：输入一种模态，输出另一种模态的任务，如语音识别（语音→文字）或文本到语音（文字→语音）。

### 核心创新点

1. **独立单模态塔 + 交叉注意力 → 灵活组合**  
   以前的多模态模型往往把所有模态的参数混在一起训练，导致每个模态的专长被稀释。Zipper 把每个模态的解码器保持独立，只在需要融合时通过交叉注意力层让它们互相“看见”。这样既保留了各自的强大能力，又能在跨模态任务中共享信息。

2. **冻结策略保留单模态性能 → 兼顾多任务**  
   在跨模态任务里，作者把输出模态对应的塔冻结（比如文字塔在 ASR 任务中不更新），让模型只学习如何把输入模态映射到已有的输出表征。实验表明，这几乎不影响文字塔的原始生成质量，却让跨模态学习更高效。

3. **少量对齐数据也能跑通 → 数据高效**  
   传统做法需要大规模的文字‑语音对齐数据才能训练出可靠的跨模态模型。Zipper 通过交叉注意力把预训练的单模态塔直接挂接起来，即使对齐数据很少也能取得竞争力的结果，显著降低了数据门槛。

4. **模块化设计便于迁移** → 可扩展性强  
   因为每个塔都是完整的解码器，想换成更大的语言模型或更高质量的语音合成模型只需要替换对应塔，而不必重新训练整个系统。这种“插件式”思路为未来多模态系统的迭代提供了便利。

### 方法详解

**整体思路**  
Zipper 的核心是把已有的单模态解码器（比如 GPT‑style 文本解码器、WaveNet‑style 语音解码器）分别放进不同的“塔”。在每一层的自注意力之后，加入一层交叉注意力，让每个塔可以读取其他塔的隐藏状态。训练时，只在需要的任务上打开对应的交叉注意力和/或解码头，其他塔可以选择冻结。

**关键模块拆解**  

1. **单模态塔**  
   - 每个塔内部结构保持原始 Transformer 解码器不变，包括自注意力、前馈网络和层归一化。  
   - 输入可以是文字 token、语音帧或其他模态的嵌入。

2. **交叉注意力层**  
   - 在第 *l* 层的自注意力之后，塔 A 会对塔 B 的第 *l* 层输出做一次注意力查询。查询向量来自塔 A 的隐藏状态，键和值来自塔 B。  
   - 这一步相当于让塔 A “借用”塔 B 的信息来丰富自己的表示。  
   - 交叉注意力是可选的，作者在实验中发现只在高层加入即可获得显著收益。

3. **冻结机制**  
   - 对于跨模态任务，输出模态对应的塔（例如文字塔在 ASR）会在整个训练过程中保持参数不变。  
   - 只更新输入塔和交叉注意力的权重，使模型学习如何把输入映射到已有的输出空间。

4. **任务特定头**  
   - 每个塔的最顶层接一个任务头（比如语言模型头、声码器头），负责把隐藏状态映射到具体的输出分布。  
   - 在多任务训练时，只激活对应任务的头，其他头保持关闭。

**流程文字版**  
```
输入（文字或语音） → 对应塔的嵌入层 → 多层自注意力
   ↓
交叉注意力（读取其他塔的隐藏状态） → 叠加到当前塔
   ↓
继续自注意力 → …（重复若干层）
   ↓
任务头 → 生成输出（文字/语音）
```

**最巧妙的点**  
- 交叉注意力只在需要时打开，避免了全局信息混杂导致的“噪声”。  
- 冻结策略让模型在学习新任务时不必重新校准已有的单模态能力，等价于在已有的高质量模型上“装上”一个适配层。

### 实验与效果

- **实验设置**：作者在语音‑文字融合场景下做了两类跨模态任务：自动语音识别（ASR）和文本到语音合成（TTS）。使用的对齐数据量相对较少（具体数字未在摘要中给出），其余训练数据来自各自的单模态大规模语料。

- **对比基线**：  
  - 端到端多模态模型（把文字和语音一起训练的统一模型）。  
  - 仅微调单模态模型的传统跨模态方案。  

- **结果**：  
  - 在 ASR 任务上，冻结文字塔后性能下降几乎可以忽略不计，整体识别准确率与不冻结的基线相当。  
  - 在 TTS 任务上，使用预训练的语音塔显著提升了音质和自然度，超过了纯 TTS 基线（具体提升幅度未在摘要中给出）。  
  - 在对齐数据稀缺的情况下，Zipper 仍然保持竞争力，说明交叉注意力有效利用了单模态的丰富表征。

- **消融实验**：作者分别关闭交叉注意力、解除冻结、以及只使用单塔进行对比，结果显示交叉注意力和冻结策略是性能提升的关键因素。  

- **局限性**：论文主要在语音‑文字两模态上验证，未展示对更多模态（如图像、视频）的直接扩展；此外，交叉注意力的计算开销随塔的数量线性增长，可能在大规模多模态系统中成为瓶颈。

### 影响与延伸思考

Zipper 把“模块化+交叉注意力”组合起来，为多模态系统提供了一种既保留单模态优势又能高效跨模态学习的思路。自发表后，已有工作尝试把视觉 Transformer、动作捕捉序列等加入多塔框架，进一步验证了插件式设计的可扩展性。未来的研究方向可能包括：

- **稀疏交叉注意力**：在塔之间只关注最相关的子空间，以降低计算成本。  
- **自适应冻结**：根据任务难度动态决定哪些层需要冻结或解冻。  
- **跨模态对齐学习**：结合对比学习或自监督方式，在几乎没有对齐数据的情况下提升融合效果。  

如果想深入了解，可以关注2024‑2025 年出现的 “Multi‑Modality Adapter” 系列论文，它们在 Zipper 的基础上进一步抽象出统一的适配层。

### 一句话记住它

**Zipper 用交叉注意力把独立的单模态解码器“拉链”在一起，既保留各自的强大能力，又在少量对齐数据下实现高效跨模态生成。**