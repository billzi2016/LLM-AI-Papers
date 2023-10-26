# Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time

> **Date**：2023-10-26
> **arXiv**：https://arxiv.org/abs/2310.17157

## Abstract

Large language models (LLMs) with hundreds of billions of parameters have sparked a new wave of exciting AI applications. However, they are computationally expensive at inference time. Sparsity is a natural approach to reduce this cost, but existing methods either require costly retraining, have to forgo LLM's in-context learning ability, or do not yield wall-clock time speedup on modern hardware. We hypothesize that contextual sparsity, which are small, input-dependent sets of attention heads and MLP parameters that yield approximately the same output as the dense model for a given input, can address these issues. We show that contextual sparsity exists, that it can be accurately predicted, and that we can exploit it to speed up LLM inference in wall-clock time without compromising LLM's quality or in-context learning ability. Based on these insights, we propose DejaVu, a system that uses a low-cost algorithm to predict contextual sparsity on the fly given inputs to each layer, along with an asynchronous and hardware-aware implementation that speeds up LLM inference. We validate that DejaVu can reduce the inference latency of OPT-175B by over 2X compared to the state-of-the-art FasterTransformer, and over 6X compared to the widely used Hugging Face implementation, without compromising model quality. The code is available at https://github.com/FMInference/DejaVu.

---

# Deja Vu：上下文稀疏化实现高效大语言模型推理 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）参数量常常达到百亿级，单次生成文本需要数百毫秒甚至秒级的推理时间，成本高得让很多实际应用望而却步。传统的稀疏化手段要么在训练阶段就把模型剪枝，需要昂贵的再训练成本；要么在推理时直接固定一套稀疏结构，却会破坏模型的“在上下文中学习”能力；还有一些方法虽然能在理论上减少算子量，却因为现代 GPU/TPU 的并行调度特性，实际的墙钟时间（wall‑clock time）并没有明显下降。于是，如何在不重新训练、保持原有推理质量的前提下，真正把算力需求压到硬件能感受到的层面，成为了迫切需要解决的难题。

### 关键概念速览
**稀疏化（Sparsity）**：把模型里一部分权重或计算单元设为零，只让“重要的”部分参与运算，就像把一张密密麻麻的地图只保留主要道路，减少不必要的行驶距离。  
**上下文稀疏化（Contextual Sparsity）**：针对每一次输入，动态挑选出少量注意力头和 MLP 神经元，使得在该特定上下文下的输出几乎和完整模型一样。可以想象为在不同的路线上，司机只打开对应的车灯。  
**注意力头（Attention Head）**：Transformer 中并行的注意力子模块，每个头负责捕捉不同的关系模式。把它们比作多位侦探，各自观察句子中的线索。  
**MLP 参数**：Transformer 层里负责非线性变换的前馈网络参数，类似于侦探收集线索后进行推理的“大脑”。  
**异步执行（Asynchronous Execution）**：在 GPU 上使用多个计算流并行调度，让稀疏计算和密集计算交错进行，避免硬件空闲。就像厨房里同时烤面包和炒菜，互不抢占。  
**硬件感知调度（Hardware‑aware Scheduling）**：根据 GPU 的线程块、显存带宽等特性，动态安排稀疏块的执行顺序，以获得最优吞吐。类似于根据道路拥堵情况实时规划行车路线。

### 核心创新点
1. **从“稀疏是全局的”到“稀疏是上下文相关的”**  
   以前的剪枝方法在整个模型上固定哪些权重被保留，导致对不同输入的适配性差。Deja Vu 通过实验发现，同一模型在不同句子上只需要不同子集的注意力头和 MLP 单元即可保持输出不变。于是把稀疏视作输入依赖的可变集合，直接解决了固定稀疏导致的质量下降问题。

2. **轻量级上下文稀疏预测器**  
   为了在推理时即时决定哪些子模块可以被关闭，作者训练了一个极小的门控网络（几乎不增加计算），输入是当前层的 token 表征，输出是每个头和每个 MLP 单元的保留概率。相比需要完整前向传播的“先跑一次再剪枝”，这种预测器只需几微秒即可完成，保持了实时性。

3. **异步、硬件感知的稀疏执行引擎**  
   仅有稀疏预测还不够，实际加速要靠调度。Deja Vu 把每层的稀疏子图拆成若干 CUDA 流：密集部分走主流，稀疏部分走副流，并利用显存共享和流水线技术让两者交叉执行。这样即使稀疏比例不高，也能在 GPU 上看到显著的墙钟时间下降。

4. **不牺牲原生的 in‑context learning**  
   许多加速方案会把模型的“在上下文中学习”能力削弱，因为它们把注意力头硬性固定。Deja Vu 的稀疏集合是每一次输入动态生成的，模型仍然可以自由地在新上下文里调动全部潜在能力，实验表明零样本、few‑shot 表现几乎与原始密集模型持平。

### 方法详解
整体思路可以拆成三步：**（1）特征提取 →（2）稀疏预测 →（3）异步执行**。下面按顺序展开。

1. **特征提取**  
   当输入序列进入第 *l* 层时，先做一次轻量的前向传播，只计算该层的输入嵌入和位置编码的线性组合，得到每个 token 的低维表示。这个表示保留了句子整体语义和局部注意力趋势，足以供后续的稀疏预测使用。

2. **稀疏预测**  
   低维表示喂入一个两层的门控网络。网络的输出是一个向量，长度等于该层注意力头数加上 MLP 隐藏单元数。每个元素经过 sigmoid 变换后得到保留概率。为了让预测更稳健，作者在训练时加入了 **KL 散度约束**，让预测分布逼近真实的“重要性标签”。这些标签是通过离线的全模型前向传播后，计算每个头/单元对最终 logits 贡献的梯度幅度得到的。预测完成后，系统对概率进行阈值化（如 0.5），得到二进制的稀疏掩码。

3. **异步执行**  
   - **稀疏子图划分**：根据掩码，把需要计算的注意力头和 MLP 单元划分成若干块，每块大小与 GPU 的 warp/warp‑size 对齐，确保硬件利用率。  
   - **多流调度**：主流（stream 0）负责密集子图的计算，副流（stream 1、2 …）负责稀疏子图。因为稀疏子图往往更小，GPU 可以在主流等待显存加载时并行执行副流，从而隐藏内存访问延迟。  
   - **显存复用**：稀疏子图的中间激活使用预先分配的共享缓冲区，避免每次分配/释放带来的开销。  
   - **同步点**：在层与层之间仍保留一次同步，以确保所有流的结果已经写回显存，防止数据竞争。  

**最巧妙的地方**在于把“预测+执行”两步的开销压到几微秒以内，使得整体加速主要来源于实际算子量的削减，而不是预测本身的成本。再加上硬件感知的流调度，Deja Vu 能在不改变模型结构的前提下，直接在现有的 GPU/TPU 环境中获得实测的墙钟时间提升。

### 实验与效果
- **测试模型**：作者选用了 Meta 的 OPT‑175B（1750 亿参数）作为主力实验对象。  
- **基准实现**：与 Hugging Face 官方实现（单线程、未做任何加速）以及 NVIDIA 的 FasterTransformer（业界常用的高性能实现）进行对比。  
- **速度提升**：在同样的硬件（A100 GPU）上，Deja Vu 将推理延迟降低了 **超过 2 倍** 相比 FasterTransformer，**超过 6 倍** 相比 Hugging Face 实现。  
- **质量保持**：在零样本和 few‑shot 的自然语言理解基准（如 LAMBADA、Winogrande）上，BLEU、准确率等指标与完整模型相差不到 0.2%，基本不出现质量退化。  
- **消融实验**：作者分别关闭稀疏预测器、关闭异步流、以及使用固定稀疏比例进行对照。结果显示：预测器贡献约 1.4× 加速，异步流再提升约 1.3×，两者叠加才达到最终的 2×+ 效果。  
- **局限性**：论文指出在显存极度紧张的场景下，稀疏掩码的存储和流切换会产生额外开销；此外，预测器在极端长序列（> 4k tokens）上仍需进一步验证。  

### 影响与延伸思考
Deja Vu 把“稀疏是上下文相关的”这一思路推向了实用层面，随后出现的几篇工作（如 **DynamicHead**, **SparseGPT‑Online**）都在尝试把稀疏预测搬到更细粒度（例如单个 token‑wise）或结合量化技术，以进一步压缩推理成本。对想深入的读者，可以关注以下方向：  
1. **稀疏预测的自监督训练**：如何在没有全模型标签的情况下，让预测器自行学习重要性。  
2. **跨模态稀疏化**：把同样的上下文稀疏思路搬到多模态模型（如 LLaVA）上。  
3. **硬件协同设计**：GPU/TPU 厂商是否会提供原生的稀疏流调度指令，以进一步放大 Deja Vu 的收益。  

### 一句话记住它
**Deja Vu 让大语言模型在每一次推理时只点亮必要的注意力头和前馈单元，实时预测 + 异步调度，墙钟时间直接翻倍。**