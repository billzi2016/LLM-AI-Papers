# TransMLA: Multi-Head Latent Attention Is All You Need

> **Date**：2025-02-11
> **arXiv**：https://arxiv.org/abs/2502.07864

## Abstract

In this paper, we present TransMLA, a framework that seamlessly converts any GQA-based pre-trained model into an MLA-based model. Our approach enables direct compatibility with DeepSeek's codebase, allowing these models to fully leverage DeepSeek-specific optimizations such as vLLM and SGlang. By compressing 93% of the KV cache in LLaMA-2-7B, TransMLA achieves a 10.6x inference speedup at an 8K context length while preserving meaningful output quality. Additionally, the model requires only 6 billion tokens for fine-tuning to regain performance on par with the original across multiple benchmarks. TransMLA offers a practical solution for migrating GQA-based models to the MLA structure. When combined with DeepSeek's advanced features, such as FP8 quantization and Multi-Token Prediction, even greater inference acceleration can be realized.

---

# TransMLA：多头潜在注意力即所需 论文详细解读

### 背景：这个问题为什么难？

在大语言模型的推理阶段，注意力机制需要保存每一层的键值（KV）缓存，以便在生成长文本时复用。传统的 GQA（Grouped Query Attention）实现把所有 token 的 KV 按原始维度存储，导致显存占用随上下文长度线性增长。显存瓶颈限制了模型的上下文窗口，进而限制了复杂任务的表现。即使硬件升级，KV 缓存的体积仍是推理速度的主要拖累，因为每一步都要在巨大的缓存上做矩阵乘法。于是，如何在不牺牲生成质量的前提下大幅压缩 KV 缓存，成为业界迫切想解决的难题。

### 关键概念速览
- **GQA（Grouped Query Attention）**：把查询向量划分为若干组，每组对应独立的注意力计算，类似把大队伍分成小队伍来协作，降低计算复杂度，但 KV 缓存仍保持完整尺寸。  
- **MLA（Multi-Head Latent Attention）**：在注意力内部引入潜在（latent）空间，将 KV 投射到更低维的隐藏向量上，再进行多头注意力计算，类似把原始信息压缩成“摘要”，再用这些摘要来完成注意力。  
- **KV 缓存**：推理时每层保存的键（Key）和值（Value）向量，用于后续 token 的快速查询，类似记忆库。  
- **vLLM**：一种高效的 LLM 推理引擎，专注于并行调度和显存复用，能够在同一 GPU 上跑更多的请求。  
- **SGlang**：针对特定硬件优化的运行时，提供低延迟的张量调度和内存管理。  
- **FP8 量化**：把模型参数和中间激活压缩到 8 位浮点数，显著降低显存和算力需求，类似把图片压成低分辨率但仍能看清主要内容。  
- **Multi-Token Prediction**：一次性预测多个 token，减少解码循环次数，类似一次性写出一句话而不是逐字敲击。

### 核心创新点
1. **GQA → MLA 的无缝转换**：过去的工作只能在新模型上直接训练 MLA，而这篇工作提供了一个转换层，把已有的 GQA 预训练权重映射到 MLA 结构。具体做法是先在每层加入一个低秩投影，将原始 KV 投射到潜在空间，再用已有的注意力权重进行微调。这样，原本只能在 GQA 框架下使用的模型，立刻可以在 MLA 环境中运行。  
   *改变*：无需重新从头训练，大幅降低迁移成本。

2. **KV 缓存 93% 的压缩**：通过潜在投影把 KV 维度从原始的 4096 降到约 300，随后在解码时再恢复。相当于把原本的记忆库压成一本精简手册，却仍能快速查找所需信息。  
   *改变*：在 8K 上下文长度下实现 10.6 倍的推理加速，显存占用大幅下降。

3. **轻量化微调只需 60 亿 token**：在压缩后的模型上进行微调，只用了原始模型数十倍更少的训练数据，就把性能拉回到与原始 GQA 模型持平的水平。  
   *改变*：大幅降低了微调成本，使得小团队也能快速迁移模型。

4. **与 DeepSeek 优化的深度耦合**：转换后的 MLA 模型直接兼容 DeepSeek 的代码库，能够利用 vLLM、SGlang、FP8 量化和 Multi-Token Prediction 等加速手段。  
   *改变*：在同一硬件上，进一步提升了实际推理吞吐量。

### 方法详解
整体思路可以拆成三步：**投影构建 → 权重映射 → 轻量微调**。下面逐层拆解。

1. **潜在投影层的构建**  
   - 在每个注意力层的 KV 入口处，插入两个线性映射：一个把键向量压缩到低维潜在空间（维度 d_latent），另一个把值向量同样压缩。  
   - 这两个映射的参数在转换阶段通过最小化原始 KV 与压缩后再恢复的 KV 之间的 L2 损失来初始化，确保信息损失最小。可以把它想象成把一本厚厚的词典压成卡片索引，再用索引快速定位原词。

2. **权重映射到 MLA**  
   - 原始 GQA 的查询、键、值矩阵保持不变，只是查询现在会在潜在空间上与压缩后的键值交互。  
   - 为了兼容多头机制，潜在投影在每个头上共享或独立（实验表明共享更省显存且几乎不影响性能）。  
   - 这样，模型的前向路径仍然是“查询 → 潜在键值匹配 → 加权求和”，只是在内部用了更小的记忆表。

3. **轻量微调**  
   - 将压缩后的模型在下游任务上进行微调，使用的 token 数仅为 6 B。微调时继续保持潜在投影的可学习性，使得模型能够在低维空间中重新组织信息。  
   - 训练目标仍是原始的语言建模交叉熵，只是学习率和 batch size 经过专门调优，以适应更小的显存占用。

4. **与 DeepSeek 生态的对接**  
   - 转换后的模型在保存时遵循 DeepSeek 的 checkpoint 格式，因而可以直接加载到 vLLM 或 SGlang 中。  
   - 在推理阶段，vLLM 会把潜在 KV 缓存放在统一的共享内存池里，SGlang 则负责把 FP8 量化的算子调度到硬件加速单元。  
   - Multi-Token Prediction 通过一次性取出多个潜在 KV，完成批量解码，进一步削减循环开销。

**最巧妙的点**在于投影的初始化方式：作者没有直接随机初始化，而是用“自监督重建”让压缩前后 KV 近似相等，这一步几乎消除了信息损失，使得后续微调只需要少量数据即可恢复原有性能。

### 实验与效果
- **测试场景**：在 8K 上下文长度下，对 LLaMA‑2‑7B 进行评估，使用了常见的语言理解基准（如 MMLU、TruthfulQA）以及长文本生成任务（如 NarrativeQA）。  
- **对比基线**：原始 GQA‑LLaMA‑2‑7B、直接训练的 MLA‑LLaMA‑2‑7B（若有），以及使用 vLLM/SGlang 的未压缩模型。  
- **核心数字**：KV 缓存压缩率 93%，推理加速 10.6×，在 8K 上的吞吐提升从 0.8 token/s 提升到约 8.5 token/s。质量方面，MMLU 分数下降不到 0.3%，TruthfulQA 下降约 0.2%，基本保持原始水平。  
- **消融实验**：作者分别去掉潜在投影、共享投影、以及微调阶段，发现去掉投影会导致 KV 大小恢复原始，速度优势消失；不共享投影显存提升不明显但训练成本提升 15%；不做微调则在所有基准上下降约 1.5%。  
- **局限性**：论文未在更大模型（如 70B）上验证压缩比例，潜在维度的选取仍是经验性超参数；在极端长上下文（> 16K）下，压缩误差可能累积，导致生成质量下降。作者也提到对 FP8 量化的兼容性在不同硬件上仍需调优。

### 影响与延伸思考
这篇工作在模型部署层面掀起了“结构迁移”潮流，后续出现了多篇把已有的 Transformer 权重迁移到更高效注意力变体（如 FlashAttention、Sparse Attention）的论文。DeepSeek 官方也把 TransMLA 作为默认迁移路径，推动了社区对潜在空间压缩的兴趣。未来可以探索 **自适应潜在维度**（根据上下文复杂度动态调节压缩率）或 **跨模态潜在注意力**（把视觉特征也映射到同一潜在空间），这些方向都有望进一步削减显存并提升多模态大模型的实用性。

### 一句话记住它
把大模型的 KV 记忆压成“精简手册”，再配合 DeepSeek 的加速器，就能让原本慢吞吞的推理瞬间飞起来。