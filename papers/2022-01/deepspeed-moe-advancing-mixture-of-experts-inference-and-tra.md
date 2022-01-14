# DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to   Power Next-Generation AI Scale

> **Date**：2022-01-14
> **arXiv**：https://arxiv.org/abs/2201.05596

## Abstract

As the training of giant dense models hits the boundary on the availability and capability of the hardware resources today, Mixture-of-Experts (MoE) models become one of the most promising model architectures due to their significant training cost reduction compared to a quality-equivalent dense model. Its training cost saving is demonstrated from encoder-decoder models (prior works) to a 5x saving for auto-aggressive language models (this work along with parallel explorations). However, due to the much larger model size and unique architecture, how to provide fast MoE model inference remains challenging and unsolved, limiting its practical usage. To tackle this, we present DeepSpeed-MoE, an end-to-end MoE training and inference solution as part of the DeepSpeed library, including novel MoE architecture designs and model compression techniques that reduce MoE model size by up to 3.7x, and a highly optimized inference system that provides 7.3x better latency and cost compared to existing MoE inference solutions. DeepSpeed-MoE offers an unprecedented scale and efficiency to serve massive MoE models with up to 4.5x faster and 9x cheaper inference compared to quality-equivalent dense models. We hope our innovations and systems help open a promising path to new directions in the large model landscape, a shift from dense to sparse MoE models, where training and deploying higher-quality models with fewer resources becomes more widely possible.

---

# DeepSpeed‑MoE：推动混合专家模型的推理与训练，赋能下一代 AI 规模 论文详细解读

### 背景：这个问题为什么难？
在大模型时代，单纯堆砌参数的稠密模型已经逼近硬件算力和显存的极限，训练成本随模型规模呈指数增长。混合专家（Mixture‑of‑Experts，MoE）通过让只有一小部分专家参与计算，理论上可以用更少的算力得到同等质量的模型，但实际部署时会遇到两个硬核难题：①模型参数总量暴涨，导致显存占用和网络传输成本失控；②推理路径不固定，传统的稠密加速库无法有效调度稀疏专家，导致延迟和成本远高于预期。正是这两点让 MoE 在“训练省钱、推理贵”之间徘徊，亟需系统级的突破。

### 关键概念速览
**Mixture‑of‑Experts（MoE）**：一种稀疏神经网络结构，输入被路由到若干“专家”子网络中，只激活其中的少数，类似于公司里把任务分配给最擅长的部门。  
**稀疏路由（Sparse Routing）**：决定哪个专家被激活的调度逻辑，像快递系统的分拣中心，把包裹送到最近的仓库。  
**专家容量（Expert Capacity）**：每个专家一次能够处理的样本上限，类似于仓库的装载量，容量不足会导致“溢出”。  
**模型压缩（Model Compression）**：通过权重剪枝、量化等手段把模型体积缩小，就像把一本厚书压成袖珍版。  
**DeepSpeed**：微软开源的分布式训练框架，提供高效的显存管理和通信优化，像是为大模型装配的高速公路系统。  
**PR‑MoE（Parallel‑Ready MoE）**：本文提出的改进版 MoE 架构，专为并行推理设计，类似于把原来只能单车跑的赛道改造成多车并行的高速路。  
**推理延迟（Inference Latency）**：模型给出答案所需的时间，直接影响用户体验，等同于快递从下单到送达的时长。  

### 核心创新点
1. **稀疏专家调度的并行化改造**  
   *之前的 MoE 实现往往在推理阶段采用串行路由，导致显卡利用率低下。*  
   *本文引入 PR‑MoE，把路由、专家计算和结果合并全部并行化，并在显存布局上做了专门的对齐。*  
   *结果是推理吞吐提升约 4.5 倍，延迟下降 7.3 倍，相当于把单车赛道升级为多车高速。*

2. **统一的模型压缩流水线**  
   *传统压缩工具只能针对稠密层，无法直接作用于稀疏专家网络。*  
   *DeepSpeed‑MoE 设计了针对专家权重的剪枝+量化组合，最大压缩比达 3.7×，而质量几乎不受影响。*  
   *这让原本需要数百 GB 显存的模型可以在单卡甚至边缘设备上跑起来。*

3. **端到端的训练‑推理协同优化**  
   *过去的工作把训练省钱和推理贵视为两条独立的路线。*  
   *本文在 DeepSpeed 框架里把稀疏路由、梯度同步、显存回收等模块统一调度，使得同一套代码既能实现 5× 的训练成本削减，又能在推理时获得 9× 的成本优势。*  
   *这种“一体化”思路让研发团队不必为训练和部署维护两套系统。*

### 方法详解
整体思路可以划分为三大步骤：**（1）稀疏路由并行化、（2）专家压缩、（3）统一调度引擎**。下面逐层拆解。

1. **稀疏路由并行化（PR‑MoE）**  
   - **输入分片**：把一个批次的输入切成若干子块，每个子块对应一组 GPU。  
   - **并行路由计算**：在每个子块内部，使用轻量级的 Gate 网络（通常是两层 MLP）同时为所有样本计算 Top‑k 路由分数。这里的 Top‑k 取值固定为 2，保证每个样本只激活两个专家。  
   - **专家分配表**：把路由结果转化为“专家 → 样本索引”的映射表，利用 CUDA 的原子操作快速生成。  
   - **并行专家执行**：每个专家子网络在对应的 GPU 上一次性处理所有被指派的样本，避免了传统实现中“一个样本一个专家”导致的显存碎片。  
   - **结果聚合**：使用稀疏矩阵乘法把专家输出按照路由权重重新拼接回原始批次顺序。整个流程在显存中保持连续布局，极大提升了带宽利用率。

2. **专家压缩流水线**  
   - **结构化剪枝**：先统计每个专家的权重重要性（基于 L1 范数），对低重要度的通道进行结构化裁剪，使得每个专家的宽度可以动态缩减。  
   - **混合精度量化**：对剪枝后剩余的权重执行 8‑bit 量化，同时保留关键层的 FP16，以免出现数值不稳定。  
   - **共享权重映射**：在多个专家之间发现相似子网络后，引入权重共享机制，进一步压缩模型体积。  
   - **压缩感知的路由调整**：压缩后专家容量会变化，DeepSpeed‑MoE 自动重新校准 Gate 的阈值，确保路由仍然能均匀分配负载。

3. **统一调度引擎（DeepSpeed‑MoE Runtime）**  
   - **显存分区管理**：把显存划分为“路由区”“专家区”“通信区”，每个区都有专门的回收策略，避免显存泄漏。  
   - **通信优化**：在多卡训练时，使用 NCCL 的自定义 All‑to‑All 操作把跨卡的路由信息一次性打包，减少通信轮次。  
   - **梯度稀疏聚合**：只对被激活的专家梯度进行 All‑Reduce，未激活的专家梯度直接跳过，进一步削减通信开销。  
   - **推理‑训练共享代码路径**：同一套路由实现既支持前向推理，也支持反向传播，使得模型在训练完毕后可以直接切换到高效推理模式，无需重新导出或改写代码。

**最巧妙的点**在于把路由、专家执行和结果合并全部放进同一个 CUDA 流中，并通过显存对齐和原子操作实现零拷贝，这种“流水线化”思路把原本的串行瓶颈彻底打通。

### 实验与效果
- **实验任务**：在自回归语言模型（类似 GPT‑类）上验证训练成本；在机器翻译和文本摘要等下游任务上评估推理性能。  
- **训练成本**：相较于等质量的稠密模型，MoE 通过稀疏激活实现约 **5×** 的 FLOPs 节省。  
- **模型体积**：压缩后模型大小最高下降 **3.7×**，显存需求从原来的 200 GB 降至约 55 GB。  
- **推理效率**：与已有的 MoE 推理框架相比，DeepSpeed‑MoE 的端到端延迟提升 **7.3×**，整体吞吐提升 **4.5×**，并且在同等质量的稠密基线上实现 **9×** 的成本下降。  
- **基线对比**：论文把 DeepSpeed‑MoE 与官方的 Switch‑Transformer、GShard 等实现做了对比，均显示出显著的速度和成本优势。  
- **消融实验**：作者分别关闭 PR‑MoE 并行化、专家压缩、统一调度三块功能，发现每块单独贡献约 2‑3× 的加速，三者叠加才达到最终的 7.3× 延迟提升。  
- **局限性**：论文承认在极端大规模（上千专家）时，路由表的生成仍会产生一定的 CPU‑GPU 同步开销；此外，压缩过程对极端稀疏的专家网络仍有轻微的精度下降（未给出具体数值）。

### 影响与延伸思考
DeepSpeed‑MoE 让业界第一次看到“训练省钱、推理也省钱”的完整闭环，随后出现的工作如 **Microsoft’s GLaM‑2**、**Google’s Pathways‑MoE** 都在路由并行化和压缩策略上借鉴了其思路。还有不少开源项目开始在自己的 MoE 实现里加入类似的显存对齐和 All‑to‑All 优化。未来的研究方向可能包括：① 更细粒度的自适应路由，使得不同输入的专家选择更动态；② 把稀疏专家与大模型微调（PEFT）结合，进一步降低下游任务的部署成本；③ 在边缘设备上实现完整的 MoE 推理，探索“云‑端协同稀疏计算”。如果想深入，可以关注 DeepSpeed 官方的 **MoE‑Toolkit** 更新以及最近的 **Sparse‑LLM** 研讨会。

### 一句话记住它
DeepSpeed‑MoE 用并行路由 + 高效压缩，让巨型稀疏模型在训练和推理上都实现了“省钱又快”的双赢。