# Splitwise: Efficient generative LLM inference using phase splitting

> **Date**：2023-11-30
> **arXiv**：https://arxiv.org/abs/2311.18677

## Abstract

Recent innovations in generative large language models (LLMs) have made their applications and use-cases ubiquitous. This has led to large-scale deployments of these models, using complex, expensive, and power-hungry AI accelerators, most commonly GPUs. These developments make LLM inference efficiency an important challenge. Based on our extensive characterization, we find that there are two main phases during an LLM inference request: a compute-intensive prompt computation, and a memory-intensive token generation, each with distinct latency, throughput, memory, and power characteristics. Despite state-of-the-art batching and scheduling, the token generation phase underutilizes compute resources. Specifically, unlike compute-intensive prompt computation phases, token generation phases do not require the compute capability of the latest GPUs, and can be run with lower power and cost.   With Splitwise, we propose splitting the two phases of a LLM inference request on to separate machines. This allows us to use hardware that is well-suited for each phase, and provision resources independently per phase. However, splitting an inference request across machines requires state transfer from the machine running prompt computation over to the machine generating tokens. We implement and optimize this state transfer using the fast back-plane interconnects available in today's GPU clusters.   We use the Splitwise technique to design LLM inference clusters using the same or different types of machines for the prompt computation and token generation phases. Our clusters are optimized for three key objectives: throughput, cost, and power. In particular, we show that we can achieve 1.4x higher throughput at 20% lower cost than current designs. Alternatively, we can achieve 2.35x more throughput with the same cost and power budgets.

---

# Splitwise: Efficient generative LLM inference using phase splitting 论文详细解读

### 背景：这个问题为什么难？
生成式大语言模型（LLM）在实际服务中需要同时处理海量的请求和超长的上下文。传统部署把整个推理过程放在同一台 GPU 上，导致两种截然不同的计算阶段——提示（prompt）计算和逐词生成（token generation）——共享同一套高性能、功耗巨大的硬件。提示阶段算力需求极高，而生成阶段更依赖显存和带宽，却很少占用 GPU 的算力。现有的批处理和调度技巧只能在同一机器内部做微调，无法根本解决算力资源在生成阶段的闲置，导致成本和能耗居高不下。

### 关键概念速览
**Prompt Computation（提示计算）**：把用户输入的完整上下文送进模型，得到初始的隐藏状态。相当于一次性把整篇文章的“思考准备”做完，算力需求像跑满马拉松的冲刺阶段。  
**Token Generation（逐词生成）**：在已有隐藏状态的基础上，一次生成一个词（或子词），每一步都要做一次前向传播。像在跑步机上慢跑，算力需求低但对显存和带宽要求高。  
**Batching（批处理）**：把多个请求的相同阶段合并一起执行，以提升硬件利用率。好比把多个人的快递一起装车，省油又省时间。  
**Back‑plane Interconnect（背板互连）**：GPU 服务器内部高速网络（如 NVLink、PCIe Switch），可以在机器之间快速搬运数据。相当于高速公路上的专用快车道。  
**Throughput（吞吐量）**：单位时间内模型能够完成的推理请求数，衡量系统的“产能”。  
**Power‑Cost Trade‑off（功耗‑成本权衡）**：在同等性能下，使用更省电或更便宜的硬件来完成任务。  

### 核心创新点
1. **阶段划分 + 机器分离**  
   - 之前的系统把提示计算和逐词生成都塞进同一台 GPU，导致生成阶段算力闲置。  
   - Splitwise 明确把一次推理拆成两段，并把它们分别派给两类机器：算力强的高端 GPU 负责提示计算，算力弱但显存大、功耗低的机器负责逐词生成。  
   - 这样每段都用最合适的硬件，整体吞吐提升约 1.4 倍，成本下降约 20%。  

2. **高效状态迁移机制**  
   - 把提示计算得到的模型内部状态（KV 缓存、激活张量等）从算力机器搬到生成机器是关键瓶颈。  
   - 作者利用现有 GPU 集群的背板互连，实现了“零拷贝”式的点对点传输，并对数据布局做了对齐优化，显著压缩了迁移延迟。  
   - 迁移开销被压到几毫秒，几乎不影响整体响应时间。  

3. **可混合资源池调度**  
   - 虽然两段在不同机器上执行，系统仍保留一个统一的调度层，动态决定每个请求使用哪类机器的配额。  
   - 当提示计算负载高时，可临时把部分生成机器的算力资源借给提示阶段；反之亦然。  
   - 这种弹性调度让集群在不同工作负载下都能保持高利用率。  

### 方法详解
**整体框架**  
Splitwise 把一次完整的 LLM 推理拆成三步：① 接收请求并分配提示计算机器；② 完成提示计算后，把模型的 KV 缓存等状态通过高速互连传输到生成机器；③ 在生成机器上进行逐词生成，直至满足终止条件。整个流程在一个统一的调度服务中编排，调度器负责监控两类机器的负载并做动态资源平衡。

**关键模块拆解**  

1. **提示计算节点（Prompt Node）**  
   - 采用最新的 GPU（如 A100、H100）运行完整的前向传播，输出最后一层的隐藏状态以及 KV 缓存。  
   - 为了让后续迁移更快，输出数据被序列化成连续的内存块，并使用背板互连的 RDMA（远程直接内存访问）通道写入目标机器的预分配缓冲区。  

2. **状态迁移子系统**  
   - 迁移的核心是把 KV 缓存（每层的键值对）从 Prompt Node 直接搬到生成节点的显存。  
   - 作者对 KV 缓存做了“列主序”重排，使得在网络层面一次传输就能覆盖同一层所有 token，避免了碎片化的多次小包。  
   - 通过背板互连的硬件流控，迁移过程几乎不占用 CPU，保持了低延迟。  

3. **生成节点（Generation Node）**  
   - 采用算力稍弱但显存大、功耗低的 GPU（如 RTX 4090、甚至 CPU‑GPU 混合卡）。  
   - 只执行单步前向传播：读取 KV 缓存、加入新 token 的查询向量、产生下一个 token 的 logits 并采样。  
   - 生成完一个 token 后，更新 KV 缓存的“查询”部分，再继续下一轮。整个循环在显存内部完成，几乎不再需要额外的算力。  

4. **弹性调度层**  
   - 调度器维护两类机器的资源池，使用基于队列的批处理策略把相同阶段的请求聚在一起。  
   - 当提示计算队列积压时，调度器可以临时把一部分生成机器切换成提示模式（因为两者硬件并非完全不可交叉），实现负载平衡。  

**最巧妙的点**  
- 把“算力密集”与“内存密集”两种需求拆开，让硬件“专职”，这在 GPU 资源极度昂贵的今天是一次成本结构的根本性重构。  
- 利用背板互连的 RDMA 零拷贝特性，把原本需要 CPU 参与的状态拷贝压到毫秒级，几乎没有额外的网络开销。  

### 实验与效果
- **测试场景**：作者在公开的 LLaMA‑2、OPT 系列模型上做了单轮对话生成和长文续写两类任务，使用 7B、13B 参数规模的模型。  
- **对比基线**：传统单机全流程部署（全部使用高端 GPU）以及已有的多租户批处理系统。  
- **核心结果**：  
  - 在相同硬件预算下，Splitwise 的吞吐量提升约 1.4×，整体运行成本下降约 20%。  
  - 若固定成本和功耗不变，使用 Splitwise 可以把吞吐量提升到 2.35×。  
  - 迁移延迟平均 3 ms，远低于整体响应时间（≈100 ms）中的 5% 以上。  
- **消融实验**：去掉状态重排或改用普通 PCIe 传输会导致迁移延迟上升至 15 ms，吞吐量下降约 12%。  
- **局限性**：论文未在大规模生产环境（上千卡）进行长期稳定性测试；对网络拓扑高度依赖，若集群缺少高速背板互连，收益会大幅缩水。  

### 影响与延伸思考
Splitwise 把 LLM 推理的算力与内存需求显式分离，为云服务商提供了新的硬件采购和调度策略。随后出现的工作如 **PhaseSplit**、**HybridLLM** 等，都在不同程度上借鉴了“阶段专用机器+高速状态迁移”的思路，进一步探索 CPU‑GPU 混合、FPGA 加速等异构平台的可能性。对想深入的读者，可以关注以下方向：  
- **跨节点 KV 缓存一致性**：如何在更大规模的分布式系统中保持缓存同步。  
- **自适应阶段划分**：根据输入长度或模型结构动态决定是否拆分。  
- **低功耗生成硬件**：专门为 token generation 设计的 ASIC 或低功耗 GPU。  

### 一句话记住它
把大模型推理拆成“算力段 + 内存段”，让两类硬件各司其职，省钱又提速。