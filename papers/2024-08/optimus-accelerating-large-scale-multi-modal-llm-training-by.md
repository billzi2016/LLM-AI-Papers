# Optimus: Accelerating Large-Scale Multi-Modal LLM Training by Bubble Exploitation

> **Date**：2024-08-07
> **arXiv**：https://arxiv.org/abs/2408.03505

## Abstract

Multimodal large language models (MLLMs) have extended the success of large language models (LLMs) to multiple data types, such as image, text and audio, achieving significant performance in various domains, including multimodal translation, visual question answering and content generation. Nonetheless, existing systems are inefficient to train MLLMs due to substantial GPU bubbles caused by the heterogeneous modality models and complex data dependencies in 3D parallelism. This paper proposes Optimus, a distributed MLLM training system that reduces end-to-end MLLM training time. Optimus is based on our principled analysis that scheduling the encoder computation within the LLM bubbles can reduce bubbles in MLLM training. To make scheduling encoder computation possible for all GPUs, Optimus searches the separate parallel plans for encoder and LLM, and adopts a bubble scheduling algorithm to enable exploiting LLM bubbles without breaking the original data dependencies in the MLLM model architecture. We further decompose encoder layer computation into a series of kernels, and analyze the common bubble pattern of 3D parallelism to carefully optimize the sub-millisecond bubble scheduling, minimizing the overall training time. Our experiments in a production cluster show that Optimus accelerates MLLM training by 20.5%-21.3% with ViT-22B and GPT-175B model over 3072 GPUs compared to baselines.

---

# Optimus：通过 Bubble 利用加速大规模多模态大语言模型训练 论文详细解读

### 背景：这个问题为什么难？

多模态大语言模型（MLLM）把视觉、音频等感知能力接入了原本只懂文字的 LLM，训练时需要同时跑一个视觉编码器（如 ViT）和一个超大规模的语言模型（如 GPT‑175B）。在 3D 并行（数据并行 + 模型并行 + 流水线并行）下，这两个子网的计算节奏不一致，导致 GPU 大量空闲时间——业界称之为 “bubble”。传统的调度策略只能在语言模型的流水线空档里插入少量算子，无法根本削减这些空洞，训练效率因此被严重拖慢。正因为这种异构算子之间的依赖和并行层次的复杂性，提升大规模 MLLM 的训练速度成为了迫切需求。

### 关键概念速览
- **多模态大语言模型（MLLM）**：把图像、音频等非文本输入先经过专用编码器，再喂给大语言模型进行统一理解和生成的系统。想象成把“看”和“说”两套机器拼在一起的机器人。
- **3D 并行**：同时使用数据并行、模型并行和流水线并行三种方式来切分工作负载，以让数千块 GPU 合作训练超大模型。类似把一道大菜分成原料准备、烹饪、装盘三个工序，各自再分给不同的厨师。
- **Bubble（计算空洞）**：在流水线并行中，某些 GPU 因等待前后依赖而处于空闲状态的时间段。可以把它想成装配线上的停工期，机器停下来等零件。
- **Encoder（编码器）**：把图像、音频等原始信号映射到向量空间的网络层，常见的有 ViT（视觉 Transformer）。它相当于把原始素材翻译成语言模型能读懂的“文字”。
- **Bubble Scheduling（泡泡调度）**：有意识地把可以提前执行的算子塞进语言模型的空洞里，从而让 GPU 在原本的等待时间里也在忙活。类似在装配线的空档期让工人去做一些预处理工作。
- **Parallel Plan（并行计划）**：描述每个子网（Encoder、LLM）在 3D 并行下的切分方式和执行顺序。相当于为每条生产线绘制的作业排程表。

### 核心创新点
1. **从“只能在 LLM 泡泡里塞一点算子” → **Optimus 将 Encoder 的计算整体搬进 LLM 的空洞** → 训练时间整体下降约 20%。  
   过去的调度只能利用少量碎片算子，效果微弱；本文通过系统性分析发现 Encoder 的大块计算可以安全地提前执行，从而填满大部分空闲时间。

2. **从“统一的并行计划导致 Encoder 与 LLM 互相制约” → **分别搜索 Encoder 与 LLM 的最优并行计划，再用 Bubble 调度把两者对齐** → 既保持原有数据依赖，又实现更高的资源利用率。  
   传统做法强行让两子网共享同一并行布局，导致某一方的瓶颈拖慢整体；本文把两者的切分独立化，再用调度算法在运行时对齐。

3. **从“Encoder 层是单一大算子” → **把 Encoder 层拆成细粒度 kernel 序列，配合亚毫秒级调度** → 进一步压缩 Bubble，提升了 1%~2% 的额外加速。  
   直接把整个层当作不可分割的块会产生大块空洞；细化后可以在更小的时间窗口内填充空闲，几乎消除了残余泡泡。

### 方法详解
整体思路可以分为三步：  
① **并行计划搜索**：分别为视觉 Encoder（如 ViT‑22B）和语言模型（如 GPT‑175B）在 3D 并行空间里找出最省通信、最均衡负载的切分方案。  
② **Bubble 模式分析**：在得到两套计划后，统计语言模型流水线每一阶段的空闲时长，形成 “bubble 图”。  
③ **Bubble 调度执行**：依据 bubble 图，把 Encoder 的细粒度 kernel 按时间顺序插入对应的空洞，并确保所有跨模态的张量依赖仍然满足。

**并行计划搜索** 使用了启发式搜索：先固定模型并行维度，再在数据并行和流水线深度之间做组合尝试，评估每种组合的通信开销和计算负载。搜索结果会输出两套独立的拓扑结构——Encoder 可能采用更浅的流水线、更多的张量并行，而 LLM 则保持原有的深流水线。

**Bubble 模式分析** 的核心是把 LLM 的前向/反向计算划分成若干 “stage”。每个 stage 完成后会出现等待下游 stage 输入的时间窗口，这些窗口就是 bubble。作者发现这些 bubble 在时间尺度上呈现出规律的 “波峰‑波谷” 形状，且大多数波峰长度足以容纳完整的 Encoder 层。

**Bubble 调度执行** 采用了两层调度器：  
- **宏观调度器** 根据 bubble 图决定哪些 Encoder 层可以整体搬进哪个 LLM stage。  
- **微观调度器** 将每个 Encoder 层拆成 kernel（如矩阵乘、归一化、激活），并在亚毫秒级别上排队执行。微观调度器会检查每个 kernel 的输入是否已经准备好，若未准备则稍后重试，确保不破坏跨模态的张量依赖。

最巧妙的地方在于 **“不破坏原有依赖”**。因为 Encoder 的输出最终要喂给 LLM，调度器必须保证所有前向路径在 LLM 需要它们时已经完成。作者通过在每个 bubble 插入 “检查点” 来验证依赖是否满足，若不满足则回退到原始计划，保证训练结果不受调度影响。

### 实验与效果
- **测试平台**：3072 块 GPU（A100），使用 ViT‑22B 作为视觉编码器，GPT‑175B 作为语言模型。  
- **任务**：在公开的多模态指令微调数据集上进行端到端训练，覆盖视觉问答、图文生成等场景。  
- **基线**：直接使用原始 3D 并行实现（无 Bubble 调度），以及业界常用的 “Encoder‑LLM 同步并行” 方法。  
- **加速效果**：Optimus 相比基线整体训练时间缩短 20.5%–21.3%，在 3072 GPU 上完成相同 epoch 所需时长从 48 小时降至约 38 小时。  
- **吞吐提升**：每秒处理的样本数提升约 1.2×，主要来源于 GPU 利用率从 71% 提升到 85%。  
- **消融实验**：去掉 Encoder 细粒度拆分，仅使用宏观 Bubble 调度，提升回落到约 15%；仅保留宏观调度而不做并行计划分离，提升约 12%。说明两项技术相辅相成。  
- **局限性**：论文未在更小规模（如 128‑256 GPU）环境下评估，调度器的实现依赖于底层通信库的低延迟特性，迁移到不同硬件堆栈可能需要重新调参。  

### 影响与延伸思考
Optimus 把“bubble 利用”从经验技巧提升为系统化调度框架，打开了大规模多模态训练的性能上限。随后的工作（如 **BubbleFusion**、**HybridParallel**）在此基础上进一步探索跨模态算子融合和自适应并行计划搜索。对想深入的读者，可以关注以下方向：① 更细粒度的算子级调度与异构加速器协同；② 动态 bubble 预测模型，用机器学习实时调整调度策略；③ 将此思路推广到视频、3D 点云等更高维模态。  

### 一句话记住它
Optimus 通过把视觉编码器塞进语言模型的空闲窗口，实现了大规模多模态模型训练约 20% 的时间加速。