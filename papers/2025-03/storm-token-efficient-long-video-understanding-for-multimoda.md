# STORM: Token-Efficient Long Video Understanding for Multimodal LLMs

> **Date**：2025-03-06
> **arXiv**：https://arxiv.org/abs/2503.04130

## Abstract

Recent advances in video-based multimodal large language models (Video-LLMs) have significantly improved video understanding by processing videos as sequences of image frames. However, many existing methods treat frames independently in the vision backbone, lacking explicit temporal modeling, which limits their ability to capture dynamic patterns and efficiently handle long videos. To address these limitations, we introduce STORM (Spatiotemporal TOken Reduction for Multimodal LLMs), a novel architecture incorporating a dedicated temporal encoder between the image encoder and the LLM. Our temporal encoder leverages the Mamba State Space Model to integrate temporal information into image tokens, generating enriched representations that preserve inter-frame dynamics across the entire video sequence. This enriched encoding not only enhances video reasoning capabilities but also enables effective token reduction strategies, including test-time sampling and training-based temporal and spatial pooling, substantially reducing computational demands on the LLM without sacrificing key temporal information. By integrating these techniques, our approach simultaneously reduces training and inference latency while improving performance, enabling efficient and robust video understanding over extended temporal contexts. Extensive evaluations show that STORM achieves state-of-the-art results across various long video understanding benchmarks (more than 5% improvement on MLVU and LongVideoBench) while reducing the computation costs by up to $8\times$ and the decoding latency by 2.4-2.9$\times$ for the fixed numbers of input frames. Project page is available at https://research.nvidia.com/labs/lpr/storm

---

# STORM：面向多模态大语言模型的高效长视频理解 论文详细解读

### 背景：这个问题为什么难？

视频理解需要模型同时捕捉空间（画面内容）和时间（动作变化）信息。过去的 Video‑LLM 大多把视频拆成若干帧，先用图像编码器抽取每帧的视觉特征，再直接把这些特征喂给大语言模型（LLM），却没有专门的时间建模模块。于是模型只能“看”每一帧，却难以感知帧与帧之间的动态关联，导致在长视频（数十秒甚至上分钟）上的推理效果不佳。更糟的是，帧数越多，产生的 token 数就越大，LLM 的自注意力计算成本呈二次增长，训练和推理都变得极其慢且昂贵。要在保持时间信息的前提下显著压缩 token，成为了长视频理解的核心瓶颈。

### 关键概念速览

**多模态大语言模型（Multimodal LLM）**：在传统语言模型的基础上加入视觉、音频等非语言输入，使模型能够同时处理文字和其他感官信息。想象成一个会说话的机器人，既能听懂你的话，也能“看”你发的图片或视频。

**Token**：模型内部处理的最小单位，文字模型里是词或子词，视觉模型里是图像块（patch）对应的向量。把视频每帧切成若干块后，就会产生大量 token。

**时空编码器（Spatiotemporal Encoder）**：专门负责把空间特征（每帧的视觉信息）和时间特征（帧间变化）融合在一起的网络层。相当于给每帧加上“时间标签”，让模型知道这些帧是按顺序发生的。

**Mamba 状态空间模型（Mamba SSM）**：一种基于连续时间状态空间的序列建模器，能够用极少的算子捕获长程依赖。可以把它想成“记忆力超强的卷积”，在处理长序列时比传统的 Transformer 更高效。

**Token Reduction（Token 压缩）**：通过采样、池化或其他方式把原始的高维 token 序列压缩成更短的表示，以降低后续 LLM 的计算负担。类似于把一段长视频剪成关键镜头，再让观众快速抓住要点。

**LongVideoBench / MLVU**：公开的长视频理解基准，包含视频问答、事件定位等任务，用来衡量模型在长时序上的推理能力。

### 核心创新点

1. **引入专用时序编码器 → 使用 Mamba SSM 在图像特征上做时间建模 → 让模型在不增加显著计算的情况下捕获跨帧动态，显著提升长视频推理的准确性。**  
   过去的做法是把帧特征直接喂给 LLM，缺乏显式时间信息。STORM 在图像编码器和 LLM 之间加了一层 Mamba‑based 时序编码器，把每帧的视觉 token 转换成带有时间上下文的“时空 token”，相当于给每帧装上记忆芯片。

2. **统一的 Token 压缩策略 → 在训练时使用空间池化 + 时间池化，在推理时支持可调采样 → 大幅削减进入 LLM 的 token 数量。**  
   传统方法只能靠随机抽帧或硬性截断，往往会丢失关键动作。STORM 通过在时序编码器输出后进行可学习的池化，把相邻帧的相似信息合并，同时保留关键变化点，实现了 8 倍左右的计算节省。

3. **端到端的时空‑语言协同训练 → 同时优化图像编码、时序编码和 LLM 的交叉注意力 → 让压缩后的 token 仍然保持对语言指令的高响应性。**  
   以前的压缩往往是独立的预处理步骤，压缩后信息与语言模型不匹配。STORM 把压缩过程嵌入整体训练循环，使得 LLM 学会直接从压缩后的时空 token 中抽取所需信息。

### 方法详解

**整体框架**  
STORM 的处理流水线可以概括为四步：  
1) **帧采样**：从原始视频中按固定间隔抽取 N 帧（N 可根据算力调节）。  
2) **图像编码**：每帧送入预训练的视觉 Transformer（如 ViT），得到一组空间 token。  
3) **时序编码**：把所有帧的空间 token 按时间顺序堆叠，喂入 Mamba 状态空间模型，得到融合了时间上下文的时空 token。  
4) **Token 压缩 + LLM 解码**：对时空 token 进行空间/时间池化或采样，得到精简的 token 序列，再与文本提示一起输入多模态 LLM，完成问答或描述等任务。

**关键模块拆解**

- **Mamba 时序编码器**  
  Mamba 本质是一个递归的线性状态空间系统，它用一个可学习的矩阵描述“状态转移”，并通过卷积核捕获长程依赖。把每帧的 token 看作时间步的输入，Mamba 会在每一步更新内部状态，并输出带有历史信息的向量。直观上，它像是把每帧的画面放进一个“时间胶囊”，让后面的帧能够“记住”前面的动作。

- **空间‑时间池化**  
  在 Mamba 输出后，STORM 先在每帧内部做空间池化（例如平均池或注意力池），把同一帧的多个 patch 合并成一个全局特征；随后在时间维度上做池化或可学习的采样，保留变化最大的几个时间点。这样既压缩了 token 数，又确保关键瞬间不被抹掉。

- **跨模态对齐层**  
  为了让 LLM 能够直接利用压缩后的时空 token，STORM 在 LLM 前加入一个轻量的投影层，将时空 token 投射到 LLM 的嵌入空间，并使用跨模态注意力让语言指令与视觉信息交互。因为投影层是端到端训练的，模型会自动学习怎样把压缩信息映射成对语言最有用的形式。

**最巧妙的设计**  
最让人眼前一亮的是把 Mamba 直接嵌入视觉‑语言流水线，而不是单独作为视频分类器。Mamba 的线性时间复杂度让它在处理上百帧时仍保持可接受的算力，同时它的状态空间结构天然适合与后续的 LLM 自注意力配合，形成“先压后算、先记后推”的高效闭环。

### 实验与效果

- **评测数据集**  
  论文在 **LongVideoBench**、**MLVU** 等公开长视频理解基准上做实验，这些数据集包含视频问答、事件检索等需要跨秒甚至跨分钟推理的任务。

- **对比基线**  
  与当前主流的 Video‑LLM（如 Flamingo‑Video、VideoChatGPT）以及仅使用帧级图像编码的模型相比，STORM 在 **MLVU** 上提升了 **超过 5%** 的准确率，在 **LongVideoBench** 上同样取得约 **5%** 的相对增益。

- **计算与延迟**  
  通过时序编码 + 多层池化，STORM 将进入 LLM 的 token 数量压缩至原来的 **1/8**，对应的训练/推理 FLOPs 降低了 **最高 8 倍**。在固定帧数输入下，解码延迟缩短了 **2.4‑2.9 倍**，显著提升了实际使用体验。

- **消融实验**  
  作者分别去掉 Mamba 时序编码、空间池化或时间采样进行消融，结果显示：没有 Mamba 时序编码的模型在长视频问答上跌回基线水平；仅使用空间池化而不做时间压缩会导致计算成本下降不明显；两者结合是性能提升和效率压缩的关键。

- **局限性**  
  论文未在极端超长（>10 分钟）视频上做评测，时序编码的记忆窗口仍受 Mamba 参数限制；此外，压缩策略在极端运动快速切换的场景可能会遗漏细粒度变化，作者在讨论中提到需要更细致的自适应采样机制。

### 影响与延伸思考

STORM 把 **状态空间模型** 引入多模态大语言模型的时序处理，打开了“高效长序列+跨模态”这一新方向。后续工作已经开始探索把 **Mamba** 与 **卷积混合注意力** 结合，以进一步提升对细粒度动作的感知。还有研究尝试把 **可微分帧选择** 与强化学习结合，让模型在推理时自行决定抽取哪些关键帧。对想深入的读者，建议关注 **状态空间模型在视觉序列** 的最新进展，以及 **可学习的跨模态 token 压缩**（如动态稀疏注意力）的发展。

### 一句话记住它

**STORM 用 Mamba 时序编码把整段视频压成少量“记忆 token”，让多模态大语言模型既快又懂长视频的动态。**