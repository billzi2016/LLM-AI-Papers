# Mixture of Lookup Experts

> **Date**：2025-03-20
> **arXiv**：https://arxiv.org/abs/2503.15798

## Abstract

Mixture-of-Experts (MoE) activates only a subset of experts during inference, allowing the model to maintain low inference FLOPs and latency even as the parameter count scales up. However, since MoE dynamically selects the experts, all the experts need to be loaded into VRAM. Their large parameter size still limits deployment, and offloading, which load experts into VRAM only when needed, significantly increase inference latency. To address this, we propose Mixture of Lookup Experts (MoLE), a new MoE architecture that is efficient in both communication and VRAM usage. In MoLE, the experts are Feed-Forward Networks (FFNs) during training, taking the output of the embedding layer as input. Before inference, these experts can be re-parameterized as lookup tables (LUTs) that retrieves expert outputs based on input ids, and offloaded to storage devices. Therefore, we do not need to perform expert computations during inference. Instead, we directly retrieve the expert's computation results based on input ids and load them into VRAM, and thus the resulting communication overhead is negligible. Experiments show that, with the same FLOPs and VRAM usage, MoLE achieves inference speeds comparable to dense models and significantly faster than MoE with experts offloading, while maintaining performance on par with MoE.

---

# Mixture of Lookup Experts 论文详细解读

### 背景：这个问题为什么难？

在大模型里，Mixture‑of‑Experts（MoE）通过只激活一小部分专家来保持低推理 FLOPs，参数量却可以随意增长。可惜，MoE 的专家是按需挑选的，所有专家的权重必须一次性装进显存，否则无法完成路由决策。显存是 GPU 上最紧缺的资源，导致即使 FLOPs 很低，模型仍然难以部署。把专家权重放到磁盘上再按需加载（offloading）可以省显存，却会把磁盘 I/O 变成瓶颈，推理延迟飙升。于是出现了“参数多、显存占用大、推理慢”三难局面，迫切需要一种既能保持 MoE 规模优势，又不把显存和延迟拖垮的方案。

### 关键概念速览

**Mixture‑of‑Experts（MoE）**：一种模型结构，包含大量子网络（专家），每次前向只让少数几个专家工作，类似于公司里只让最擅长的员工处理特定任务，从而在计算上保持轻量。

**路由（Routing）**：决定输入属于哪几个专家的过程，通常由一个小网络输出概率或硬选择，类似于客服系统把来电分配给合适的坐席。

**显存（VRAM）**：GPU 用来存放模型参数和中间激活的高速内存，容量有限，像是厨房的工作台面，太多材料会让操作变慢。

**离线重参数化（Offline Re‑parameterization）**：在训练结束后，把一个可训练的网络层转换成等价的固定结构（如查找表），相当于把手工烹饪的配方写成成品菜谱，直接取用。

**查找表（Lookup Table，LUT）**：用键（key）直接索引预先计算好的结果的结构，像是字典里查词一样，查询时间几乎为常数。

**离线专家离线化（Expert Offlineization）**：把训练时的专家网络转成查找表的过程，使得推理时不再需要算专家的前向，只需要一次键值检索。

### 核心创新点

1. **训练时仍使用普通 FFN 专家 → 推理前把每个专家转成查找表**  
   传统 MoE 在训练和推理阶段都使用全连接前馈网络（FFN）作为专家，推理时仍要算这些网络。本文把训练好的 FFN 用输入的离散 ID（来自嵌入层的词表索引）离线计算所有可能的输出，存成查找表。这样推理时不再做矩阵乘法，只做一次键值检索，显著降低计算量。

2. **显存占用从“所有专家权重必须常驻” → “只需加载查询得到的查找表块”**  
   由于查找表可以分块存放在磁盘或 SSD，推理时只把当前 batch 所需的表块装进显存。相比传统 MoE 必须一次性把所有专家权重装进显存，这种按需加载方式把显存需求降到了和稠密模型相当的水平。

3. **通信开销从“跨 GPU 传输完整专家权重” → “几乎为零的键值查询”**  
   在多卡部署时，MoE 需要把路由信息和专家权重在卡间同步，带来显著的网络流量。MoLE 把专家输出预先写进查找表，推理时只需要把输入 ID 发给对应的存储设备，查询成本极低，几乎不产生额外通信。

4. **保持 MoE 性能的同时实现稠密模型级别的推理速度**  
   实验表明，在相同 FLOPs 与显存预算下，MoLE 的推理速度与普通稠密模型相当，却比使用 offloading 的 MoE 快上数倍，且在任务精度上几乎不逊色。换句话说，它把 MoE 的“参数规模优势”与稠密模型的“部署友好性”合二为一。

### 方法详解

**整体思路**  
MoLE 的工作流程可以划分为三步：① 训练阶段使用标准 MoE（专家为 FFN），② 训练结束后对每个专家进行离线重参数化，生成基于词表 ID 的查找表，③ 推理时把输入的 ID 直接映射到查找表，检索对应的专家输出并送入后续层。整个过程的核心是把“计算”搬到离线阶段，把“查询”留给推理。

**关键模块拆解**

1. **Embedding + 路由**  
   输入序列先经过词向量嵌入层，得到离散的 token ID 与对应的向量。路由网络（通常是一个小的线性层加 softmax）根据这些向量决定每个 token 要走哪些专家。这里的路由输出仍是离散的专家索引，因为后续查找表只能接受离散键。

2. **训练时的 FFN 专家**  
   每个专家是一个两层前馈网络：先做线性变换 + 激活，再做第二层线性变换。训练期间，这些网络和路由一起端到端优化，保持 MoE 的表达能力。

3. **离线重参数化**  
   - **枚举所有可能的输入 ID**：因为嵌入层的输入是离散的词表索引，理论上可以遍历整个词表（或子词表）。  
   - **前向计算专家输出**：把每个 ID 送进对应的 FFN，记录下专家的输出向量。  
   - **构建查找表**：把 ID 作为键，专家输出向量作为值，保存为二进制块。若模型有多个专家，则为每个专家生成独立的表，或者把所有专家的输出拼接成一个大表，键中携带专家 ID 信息。

4. **推理时的查询路径**  
   - **加载必要的表块**：根据当前 batch 中出现的 token ID，挑选对应的查找表块并把它们映射到显存。  
   - **键值检索**：对每个 token，用其 ID 直接在表中查找预计算好的向量。检索过程几乎是 O(1) 的内存访问。  
   - **后续层处理**：检索到的向量直接进入模型的下游层（如 Transformer 的残差块），与稠密模型的前向路径相同。

**最巧妙的点**  
把原本需要在 GPU 上做的大量矩阵乘法，提前搬到磁盘上做一次性离线计算，然后在推理时只做一次键值查找。这个“计算-存储”权衡在大模型部署场景里极具冲击力，因为显存瓶颈往往比算力更难突破。

### 实验与效果

- **测试任务**：论文在语言模型基准上评估了 MoLE，包括常见的语言建模和下游微调任务（如文本分类、问答）。  
- **对比基线**：稠密 Transformer、传统 MoE（所有专家常驻显存）以及使用 offloading 的 MoE。  
- **性能表现**：在相同 FLOPs 与显存预算下，MoLE 的推理速度与稠密模型持平，明显快于 offloading MoE（论文声称快数倍），而在准确率或困惑度上与 MoE 基本持平，几乎没有性能损失。  
- **消融实验**：作者分别去掉离线查找表、仅对部分专家做离线化等设置，结果显示完整的离线查表是加速的关键因素。  
- **局限性**：离线查表的大小受词表规模限制，若词表极大或输入是连续值（如图像像素），直接枚举所有可能输入不可行。论文也提到查表生成过程需要额外的离线计算资源。

### 影响与延伸思考

MoLE 把 MoE 的规模优势和稠密模型的部署友好性结合起来，为大模型在资源受限的边缘设备或云端多租户环境提供了新思路。后续工作可能会探索：

- **对连续输入的离线化**：比如把视觉 Transformer 的 patch 编码离线成查表，或使用量化+哈希技术近似连续空间。  
- **自适应表块管理**：在推理时动态预测哪些表块最可能被访问，提前预取到显存，进一步压缩延迟。  
- **跨模态 MoLE**：把语言、视觉、音频等不同模态的专家都离线化，构建统一的多模态查表系统。  

如果想深入了解，可以关注近期在“离线模型压缩”“专家网络存储优化”等方向的会议论文，尤其是 NeurIPS、ICLR、ACL 上的相关投稿。

### 一句话记住它

把 MoE 的专家在推理前变成查找表，只查不算，让大模型像字典一样快速、低显存地上线。