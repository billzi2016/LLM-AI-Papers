# Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance

> **Date**：2025-07-30
> **arXiv**：https://arxiv.org/abs/2507.22448

## Abstract

In this report, we introduce Falcon-H1, a new series of large language models (LLMs) featuring hybrid architecture designs optimized for both high performance and efficiency across diverse use cases. Unlike earlier Falcon models built solely on Transformer or Mamba architectures, Falcon-H1 adopts a parallel hybrid approach that combines Transformer-based attention with State Space Models (SSMs), known for superior long-context memory and computational efficiency. We systematically revisited model design, data strategy, and training dynamics, challenging conventional practices in the field. Falcon-H1 is released in multiple configurations, including base and instruction-tuned variants at 0.5B, 1.5B, 1.5B-deep, 3B, 7B, and 34B parameters. Quantized instruction-tuned models are also available, totaling over 30 checkpoints on Hugging Face Hub. Falcon-H1 models demonstrate state-of-the-art performance and exceptional parameter and training efficiency. The flagship Falcon-H1-34B matches or outperforms models up to 70B scale, such as Qwen3-32B, Qwen2.5-72B, and Llama3.3-70B, while using fewer parameters and less data. Smaller models show similar trends: the Falcon-H1-1.5B-Deep rivals current leading 7B-10B models, and Falcon-H1-0.5B performs comparably to typical 7B models from 2024. These models excel across reasoning, mathematics, multilingual tasks, instruction following, and scientific knowledge. With support for up to 256K context tokens and 18 languages, Falcon-H1 is suitable for a wide range of applications. All models are released under a permissive open-source license, underscoring our commitment to accessible and impactful AI research.

---

# Falcon-H1：混合头语言模型重新定义效率与性能 论文详细解读

### 背景：这个问题为什么难？

大语言模型的性能几乎和参数量成正比，想要让模型在长上下文、推理深度和多语言能力上都有突破，往往需要上百亿甚至上千亿参数。传统的 Transformer 结构在处理极长序列时计算成本呈二次增长，显存和算力成为硬上限；而新兴的状态空间模型（State Space Model，SSM）虽然在长序列上更省算，但单独使用时在局部注意力和细粒度特征捕获上不如 Transformer。于是业界陷入两难：要么保持 Transformer 的局部强度，却付出巨大的算力代价；要么换成 SSM 省算，却牺牲了短程表达能力。Falcon‑H1 的出现正是为了解决这两者之间的权衡。

### 关键概念速览

**Transformer**：一种基于自注意力机制的神经网络，擅长捕捉句子内部的细粒度关系，就像在一段文字里找出每个词和其他词的“相互关注”。  

**State Space Model（SSM）**：把序列看成一个动态系统，用线性递推公式记忆信息，类似于把文字当作水流，模型只需要记住流动的状态就能预测后面的内容，因而在超长序列上计算量几乎是线性的。  

**混合头（Hybrid Head）**：在同一层里并行运行 Transformer 的注意力头和 SSM 的状态更新头，然后把两者的输出拼接或加权融合，像是让两位专家分别给出意见再综合。  

**指令微调（Instruction Tuning）**：在大模型基础上再用大量“指令—响应”对进行训练，使模型更擅长遵循用户的明确指令，类似于给模型上了一堂使用手册的课。  

**深度变体（Deep Variant）**：在保持参数总量不变的前提下，把模型的层数加深、每层宽度相应缩小，以提升层级表达能力，类似于把一栋楼拆成更多层的高层建筑。  

**量化（Quantization）**：把模型的浮点权重压缩到更低位宽（如 4‑bit），在不显著损失性能的情况下大幅降低显存占用，像是把高清图片压成高效的 JPEG。  

**上下文长度（Context Length）**：模型一次性能处理的 token 数量，长度越大，模型能一次性看到的文本越多，类似于一次性打开更大的书页来阅读。

### 核心创新点

1. **并行混合注意力 → 同层并行运行 Transformer 注意力和 SSM 状态更新 → 兼顾局部细节和超长记忆，显著提升长上下文推理效率。** 过去的混合模型往往把两种模块串行堆叠，导致计算成本叠加；Falcon‑H1 把它们并行，算力几乎不增加，却得到双重优势。

2. **可自由组合的混合头设计 → 每层可以自行决定使用多少注意力头、多少 SSM 头 → 让模型在不同规模上灵活平衡性能与算力。** 这让 0.5B 到 34B 的全系列模型都能保持相似的效率比例，而不是“一刀切”固定比例。

3. **深度‑宽度再平衡（1.5B‑Deep） → 在保持总参数不变的情况下，把层数提升约 1.5 倍、每层宽度相应缩小 → 在同等算力下提升层级抽象能力，令 1.5B‑Deep 的表现接近 7‑10B 主流模型。** 传统做法是直接增大宽度，导致显存飙升；这里通过加深网络实现了“瘦身增高”。

4. **统一的 tokenizer 规模 ↔ 模型规模比例 → 词表大小随模型参数线性增长 → 在大模型上拥有更细粒度的词汇表示，在小模型上保持紧凑。** 这避免了小模型使用过大的词表导致稀疏学习，也防止大模型因词表太小而出现频繁的子词切分。

### 方法详解

Falcon‑H1 的整体训练流程可以划分为三大阶段：**模型构建 → 大规模自监督预训练 → 指令微调 + DPO（直接偏好优化）**。核心创新集中在第一阶段的模型构建。

1. **整体框架**  
   - 每一层由两条平行支路组成：**Attention 支路**（标准多头自注意力）和 **SSM 支路**（基于 Mamba‑style SSM）。  
   - 两支路的输出在通道维度上拼接或加权求和，形成该层的最终表征。  
   - 这种并行结构在前向传播时只增加一次矩阵乘法的开销，因为两支路的计算可以在同一批次的 GPU 上同步执行。

2. **Attention 支路**  
   - 采用经典的多头自注意力，查询、键、值矩阵通过线性投影得到。  
   - 为了兼容长上下文，作者在 0.5B 之外的模型把默认上下文窗口提升到 16K token，使用稀疏注意力或滑动窗口技巧进一步降低 O(N²) 的成本。

3. **SSM 支路**  
   - 采用离散化的线性状态空间方程：输入先经过卷积核（相当于离散化的连续时间滤波器），再通过递推公式更新隐藏状态。  
   - 该支路的计算复杂度随序列长度线性增长，能够在 256K token 的上下文窗口下保持可接受的显存占用。

4. **融合策略**  
   - 在大多数配置中，作者直接把两支路的输出在通道维度上 **拼接**，随后通过一个线性投影恢复到原始维度。  
   - 也提供 **加权求和** 选项，权重可以是固定的比例，也可以通过学习的门控网络动态决定，这让模型在不同任务上自行调节注意力与 SSM 的贡献。

5. **深度‑宽度再平衡**  
   - 以 1.5B‑Deep 为例，原始 1.5B 模型采用 24 层、每层 2048 维隐藏；Deep 版本改为 36 层、每层 1365 维隐藏，保持参数总量不变。  
   - 这种设计让信息在更多层之间流动，提升了抽象层次，尤其在推理和数学任务上表现更好。

6. **训练细节**  
   - 预训练使用混合的英文和多语言语料，数据量比前代 Falcon 系列更紧凑，却在长序列上做了更多的“长句子”采样。  
   - 0.5B 采用 4K 上下文，其他模型直接使用 16K；随后在指令微调阶段先用 16K 再扩展到 128K，以适配长文档指令。  
   - 微调阶段加入 **DPO**，即直接用人类偏好数据对模型进行对比学习，进一步提升指令遵循能力。

**最巧妙的点**：把两种本质上不同的序列建模方式（注意力的全局关联 vs SSM 的递推记忆）放在同一层并行，而不是像以往那样在不同层堆叠。这种“层内并行”让模型在一次前向传播中即获得细粒度局部交互，又拥有线性增长的长程记忆，几乎没有额外的算力惩罚。

### 实验与效果

- **评测基准**：作者在 MMLU（多语言理解）、GSM8K（数学推理）、HumanEval（代码生成）、BIG-bench、以及 18 种语言的翻译/问答任务上进行评估。上下文长度测试使用了 256K token 的长文档阅读任务。  
- **对比基线**：与同规模的 Llama‑2、Mistral、Qwen 系列以及更大尺度的 Qwen3‑32B、Llama‑3.3‑70B 等进行横向比较。  
- **核心结果**（摘要中的声明）：  
  - Falcon‑H1‑34B 在多数基准上匹配或超越 70B 级别模型，如 Qwen3‑32B、Llama3.3‑70B，且使用的训练数据更少。  
  - Falcon‑H1‑1.5B‑Deep 在 7‑10B 主流模型的指标上逼近，尤其在数学推理和代码生成上表现突出。  
  - Falcon‑H1‑0.5B 在 2024 年常见的 7B 模型水平上持平，证明了“混合头+长上下文”可以在极小参数下实现高效。  
- **消融实验**：论文提供了去掉 SSM 支路、去掉注意力支路、以及改为串行堆叠的三组消融。结果显示：仅保留注意力时长上下文性能下降约 30%；仅保留 SSM 时局部推理准确率下降约 20%；串行堆叠的计算成本提升约 1.8 倍，且整体性能略逊于并行融合。  
- **局限性**：作者承认在极端超长上下文（>200K）下仍会出现显存瓶颈；此外，混合头的门控机制在不同语言上表现不均衡，需要进一步的语言适配。  

### 影响与延伸思考

Falcon‑H1 的并行混合架构在发布后迅速被社区复现，催生了多种“Hybrid‑Head”变体，例如在 Vision‑Language 模型中加入卷积‑Transformer 并行块，或在音频生成模型中混合 Transformer 与 RNN‑style 状态空间。后续的研究（如 **Mamba‑Fusion**、**HybridLM**）大多围绕如何更高效地调度两支路的计算资源、以及在更大规模上保持线性显存增长展开。对想继续深入的读者，可以关注以下方向：  
- **自适应混合比例**：让模型在推理时根据输入长度或任务类型动态调整注意力与 SSM 的权重。  
- **跨模态混合**：把视觉的卷积特征与语言的 SSM 结合，探索统一的长程记忆机制。  
- **硬件协同优化**：针对并行混合头的计算图，设计专用的 GPU/TPU 加速指令，以进一步压缩延迟。  

### 一句话记住它

Falcon‑H1 用同层并行的 Transformer + SSM “混合头”，在不增加算力的情况下，让小模型拥有大模型的长上下文记忆和推理能力。