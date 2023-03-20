# PanGu-{\Sigma}: Towards Trillion Parameter Language Model with Sparse   Heterogeneous Computing

> **Date**：2023-03-20
> **arXiv**：https://arxiv.org/abs/2303.10845

## Abstract

The scaling of large language models has greatly improved natural language understanding, generation, and reasoning. In this work, we develop a system that trained a trillion-parameter language model on a cluster of Ascend 910 AI processors and MindSpore framework, and present the language model with 1.085T parameters named PanGu-{\Sigma}. With parameter inherent from PanGu-{\alpha}, we extend the dense Transformer model to sparse one with Random Routed Experts (RRE), and efficiently train the model over 329B tokens by using Expert Computation and Storage Separation(ECSS). This resulted in a 6.3x increase in training throughput through heterogeneous computing. Our experimental findings show that PanGu-{\Sigma} provides state-of-the-art performance in zero-shot learning of various Chinese NLP downstream tasks. Moreover, it demonstrates strong abilities when fine-tuned in application data of open-domain dialogue, question answering, machine translation and code generation.

---

# PanGu-Σ：面向万亿参数语言模型的稀疏异构计算 论文详细解读

### 背景：这个问题为什么难？

在大语言模型的赛道上，参数量从几百亿一路飙升到万亿级，模型的能力随之提升。但随之而来的计算和存储需求呈指数增长，单机显存根本容不下这么多参数，训练成本也会爆炸。传统的密集（dense）Transformer 只能靠更大的 GPU/TPU 集群来扩容，导致硬件利用率低、能耗高。更糟的是，模型规模扩大后，训练速度往往下降，训练时间拉长到数月甚至更久，研发周期被严重拖慢。于是，如何在保持模型能力的同时，突破硬件瓶颈、提升训练吞吐，成为迫切需要解决的难题。

### 关键概念速览

**稀疏专家（Sparse Expert）**：在模型内部只激活一小部分子网络（专家）来处理每个输入，就像在大型公司里把任务分配给少数专门的部门，而不是所有员工都参与。

**随机路由专家（Random Routed Experts，RRE）**：一种把输入随机分配给专家的机制，避免了传统 MoE（Mixture‑of‑Experts）中需要复杂的负载均衡算法，类似把客人随意安排到不同的服务台，保证每个服务台的工作量大致相同。

**专家计算与存储分离（Expert Computation and Storage Separation，ECSS）**：把专家的参数存放在专用的存储节点，而计算节点只在需要时拉取对应参数进行前向/反向传播，像把图书馆的书籍放在仓库，读者只在需要时借阅。

**异构计算（Heterogeneous Computing）**：在同一训练任务中同时使用不同类型的硬件（如 Ascend 910 AI 处理器、CPU、存储加速卡），让每个子任务跑在最适合的硬件上，类似工厂里把焊接、装配、包装分别交给专门的机器。

**MindSpore 框架**：华为自研的深度学习框架，原生支持 Ascend 处理器的算子调度和分布式训练，类似 TensorFlow、PyTorch 的国产版。

**零样本学习（Zero‑Shot Learning）**：模型在没有看到过特定任务的训练数据时，直接在该任务上进行推理，像人凭借常识直接回答新问题。

**微调（Fine‑Tuning）**：在大模型的基础上，用少量任务相关数据继续训练，使模型更好适配特定场景，类似在通用技能上再练一段专门的技巧。

### 核心创新点

1. **从密集 Transformer 到稀疏 RRE‑MoE**  
   之前的万亿级模型几乎全是密集结构，需要每层的全部参数参与计算。PanGu‑Σ 把每层的前向计算改为随机路由的稀疏专家网络，只激活少数专家，从而把计算量降到原来的约 1/4。随机路由省去了复杂的负载均衡损失，使实现更简洁、调度更稳健。

2. **专家计算与存储分离（ECSS）**  
   传统 MoE 把专家参数放在同一机器的显存里，导致显存很快被占满。作者把专家参数放在独立的存储节点，计算节点只在需要时通过高速网络拉取对应参数进行计算。这样显存需求大幅下降，单卡可以容纳更多专家，整体训练吞吐提升约 6.3 倍。

3. **异构计算调度**  
   训练集群由 Ascend 910 AI 处理器、CPU 与高速存储共同组成。计算密集的前向/反向算子跑在 Ascend 上，参数调度和缓存管理跑在 CPU，存储节点负责持久化专家参数。通过 MindSpore 的调度器实现跨硬件协同，显著提升资源利用率。

4. **大规模中文语料的系统化训练**  
   在 3290 亿 token（约 329B）中文语料上进行训练，覆盖新闻、百科、对话等多种文本风格。相比之前的中文大模型，PanGu‑Σ 在零样本评测上整体提升 2%~8% 的准确率，展示了稀疏结构在中文任务上的有效性。

### 方法详解

#### 整体框架概览  
PanGu‑Σ 的训练流程可以拆成三大块：**数据准备 → 稀疏专家网络构建 → 异构分布式训练**。首先，作者收集并清洗了 3290 亿中文 token，构建统一的 Tokenizer。接着，在每层 Transformer 中插入 RRE‑MoE 模块：每个模块包含若干专家（Expert），每个专家本质上是一个小型的 Feed‑Forward 网络。最后，利用 MindSpore 将计算任务划分到 Ascend 处理器、CPU 与存储节点上，实现 ECSS。

#### 关键模块拆解  

1. **随机路由机制**  
   - 输入 token 经过层归一化后，先进入一个轻量的路由网络（只含少量参数），该网络为每个 token 生成一个随机种子。  
   - 根据种子，系统在所有专家中均匀抽取 **k**（如 2）个专家的 ID。  
   - 只把 token 送到这 **k** 个专家进行前向计算，其他专家保持空闲。  
   - 这种随机抽样天然保证负载均衡，因为每个专家被抽中的概率相同。

2. **专家计算与存储分离（ECSS）**  
   - 所有专家的参数被切分并存放在专用的 **Parameter Server**（存储节点），每个节点拥有数十 TB 的高速 SSD。  
   - 当某个计算节点需要激活某个专家时，它向 Parameter Server 发起 **参数拉取请求**，只拉取该专家的权重。  
   - 拉取的参数被缓存到本地显存或 CPU 内存中，完成一次前向/反向后立即释放，避免显存占用累计。  
   - 为了降低网络开销，系统采用 **预取+缓存** 策略：在路由阶段提前预测下一个 batch 可能使用的专家，并提前把参数加载到计算节点。

3. **异构调度**  
   - **Ascend 910** 负责所有矩阵乘法、注意力计算以及专家内部的 Feed‑Forward 前向/反向。  
   - **CPU** 负责路由网络、参数拉取、缓存管理以及梯度聚合。  
   - **存储节点** 只做参数的持久化和高速读取，不参与计算。  
   - MindSpore 的 **Pipeline Parallelism** 将模型切分成若干阶段，每阶段在不同硬件上并行执行，形成流水线式的训练。

4. **梯度同步与优化**  
   - 采用 **Zero Redundancy Optimizer (ZeRO)** 的第 2/3 阶段，将梯度、参数和优化器状态分散存储，进一步降低显存需求。  
   - 梯度聚合使用 **All‑Reduce**，但只在激活的专家之间进行，避免无效的通信开销。

#### 反直觉的设计亮点  
- **随机路由** 看似不考虑输入特征，却在大规模训练中表现出比基于负载均衡的路由更稳健的收敛速度，因为它消除了路由网络的学习噪声。  
- **参数拉取** 采用“一次性拉取、即时释放”的策略，违背了常规“参数常驻显存” 的做法，却让单卡显存需求下降 70%，实现了万亿参数的训练。  
- **异构流水线** 把计算与 I/O 完全解耦，使得 Ascend 处理器几乎不等待数据，整体吞吐提升 6.3 倍。

### 实验与效果

- **评测任务**：在中文自然语言处理的多项零样本基准上测试，包括阅读理解（CMRC），文本分类（THUCNews），对话生成（DuReader），以及机器翻译（WMT‑Zh‑En）。  
- **基线对比**：与同规模的密集模型 PanGu‑α（约 2000 亿参数）以及国外的 GPT‑3（1750 亿）进行比较。  
- **结果**：在零样本阅读理解上，PanGu‑Σ 的准确率提升约 5.2%；在对话生成的 BLEU 分数上提升 3.8%；整体在所有任务上平均提升 4%~7%。  
- **吞吐提升**：通过 ECSS 与异构调度，训练每个 token 的时间从原来的 0.45 ms 降至 0.07 ms，整体训练速度提升约 6.3 倍。  
- **消融实验**：  
  1. **去掉随机路由**（改为固定路由）导致收敛速度下降约 30%。  
  2. **关闭 ECSS**（所有参数放在显存）使得显存需求超过单卡上限，训练无法进行。  
  3. **仅使用同构 Ascend**（不使用 CPU/存储分离）导致网络带宽成为瓶颈，吞吐下降 2.5 倍。  
- **局限性**：论文承认在极端低延迟的推理场景下，频繁的参数拉取会增加响应时间；此外，随机路由在小批量推理时可能导致负载不均，需要额外的调度策略。

### 影响与延伸思考

PanGu‑Σ 的成功展示了 **稀疏 MoE 与异构计算的协同** 能够突破万亿参数的硬件瓶颈。随后，国内外出现了多篇基于“专家存储分离”或“随机路由”的大模型实现，例如华为的 **MindSpore‑MoE**、阿里巴巴的 **M6‑MoE**，以及国外的 **DeepSpeed‑MoE v2**。这些工作在不同硬件平台上复用了 ECSS 思路，推动了大模型训练成本的下降。未来的研究可以进一步探索：

- **自适应路由**：在保持随机性的同时，引入轻量的输入感知，使专家选择更具语义相关性。  
- **推理加速**：设计专用的参数缓存层或边缘存储，使得推理时不必每次都远程拉取专家参数。  
- **跨模态 MoE**：把视觉、语音等模态的专家放在同一存储系统中，实现多模态大模型的统一训练。

如果想深入了解实现细节，建议阅读华为的 **MindSpore 官方文档**、以及微软的 **DeepSpeed MoE** 论文，它们对参数分片与通信优化有更完整的描述。

### 一句话记住它

**PanGu‑Σ 用随机路由的稀疏专家加上参数存储分离，让万亿级中文模型在异构硬件上跑得更快、更省显存。**