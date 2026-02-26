# DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference

> **Date**：2026-02-25
> **arXiv**：https://arxiv.org/abs/2602.21548

## Abstract

The performance of multi-turn, agentic LLM inference is increasingly dominated by KV-Cache storage I/O rather than computation. In prevalent disaggregated architectures, loading the massive KV-Cache from external storage creates a fundamental imbalance: storage NICs on prefill engines become bandwidth-saturated, while those on decoding engines remain idle. This asymmetry severely constrains overall system throughput.   We present DualPath, an inference system that breaks this bottleneck by introducing dual-path KV-Cache loading. Beyond the traditional storage-to-prefill path, DualPath enables a novel storage-to-decode path, in which the KV-Cache is loaded into decoding engines and then efficiently transferred to prefill engines via RDMA over the compute network. DualPath combines this optimized data path -- which inherently avoids network congestion and avoids interference with latency-critical model execution communications -- with a global scheduler that dynamically balances load across prefill and decode engines.   Our evaluation on three models with production agentic workloads demonstrates that DualPath improves offline inference throughput by up to 1.87$\times$ on our in-house inference system. It can also improve online serving throughput by an average factor of 1.96$\times$ without violating SLO.

---

# DualPath：突破代理式大语言模型推理中的存储带宽瓶颈 论文详细解读

### 背景：这个问题为什么难？
在多轮对话或自动化代理任务里，LLM（大语言模型）需要在每一步保存和读取 KV‑Cache（键值缓存）以实现快速增量推理。随着模型规模和对话长度增长，KV‑Cache 体积变得巨量，必须从外部存储读取。传统的“预填充‑解码”分离架构把 KV‑Cache 只从存储送到预填充（prefill）节点，而解码（decode）节点几乎不参与 I/O。结果是预填充节点的存储网卡（NIC）被带宽压垮，而解码节点的 NIC 则闲置，整体吞吐受限。换句话说，系统的瓶颈从算力转向了存储带宽，单一路径已经无法满足高并发的代理工作负载。

### 关键概念速览
**KV‑Cache**：模型在每一步生成时保存的注意力键和值，类似于记忆本，后续只需要增量读取即可加速推理。  
**Prefill 引擎**：负责一次性处理长上下文的阶段，需要把完整 KV‑Cache 从存储加载进来，类似于一次性把整本书搬进图书馆。  
**Decode 引擎**：负责逐步生成新 token 的阶段，只需要读取最新的 KV‑Cache 条目，像是图书馆里只取出最新的几页。  
**存储‑到‑Prefill 路径**：传统的数据流，从外部磁盘经 NIC 直接送到预填充节点。  
**存储‑到‑Decode 路径**：本文提出的第二条数据流，从外部磁盘先送到解码节点，再经 RDMA（远程直接内存访问）转给预填充节点。  
**RDMA**：一种在计算网络上直接读写远程内存的技术，省去 CPU 介入，类似于两座仓库之间的高速传送带。  
**全局调度器**：监控所有预填充和解码节点的负载，动态决定每个请求走哪条路径，像是交通指挥中心根据路况分配车流。  

### 核心创新点
**传统单路径 I/O → 双路径 KV‑Cache 加载 → 带宽利用率提升**  
过去只把 KV‑Cache 送到预填充节点，导致存储 NIC 在预填充阶段饱和。DualPath 让解码节点也参与读取 KV‑Cache，形成第二条“存储‑到‑Decode”通路。这样存储带宽被分摊到两套 NIC，整体 I/O 吞吐几乎翻倍。

**仅存储‑到‑Prefill → 存储‑到‑Decode + RDMA 迁移 → 网络拥塞降低**  
在双路径中，KV‑Cache 从存储进入解码节点后，通过 RDMA 直接搬到预填充节点，避免了再次走存储网络。因为 RDMA 使用的是计算网络而非存储网络，这条通路不与模型推理的控制消息竞争，等于是把“搬家”任务搬到不拥挤的高速公路上。

**静态负载划分 → 基于 Compute‑NIC 的全局调度器 → 动态平衡**  
双路径带来了更复杂的资源调度需求。作者围绕 Compute‑NIC（计算节点的网卡）构建流量管理层，调度器实时监控每条路径的带宽占用和节点空闲情况，动态把新请求分配到最空闲的预填充或解码节点。这样既防止了某一路径过载，又保证了延迟关键的解码阶段不被干扰。

### 方法详解
整体思路可以分为三步：① 双路径数据入口，② RDMA 跨节点迁移，③ 全局调度与负载平衡。

1. **双路径数据入口**  
   - 当一个新对话请求到达系统时，调度器检查当前预填充和解码节点的 I/O 使用率。  
   - 若预填充节点的存储 NIC 已接近饱和，调度器会指示请求走“存储‑到‑Decode”路径：先把 KV‑Cache 读取到解码节点的本地内存。  
   - 反之，如果解码节点的 Compute‑NIC 负载低，系统仍可以选择传统路径。这样两套 NIC 能并行工作，避免单点瓶颈。

2. **RDMA 跨节点迁移**  
   - KV‑Cache 到达解码节点后，解码节点不直接使用它，而是通过 RDMA 将缓存块写入预填充节点的内存。  
   - RDMA 的零拷贝特性让数据在网络层面直接搬运，省去 CPU 复制和额外的网络协议开销。  
   - 迁移完成后，预填充节点即可像普通预填充一样继续推理，而存储‑到‑Decode 的读取过程已经在后台完成，不会阻塞模型的前向计算。

3. **全局调度与负载平衡**  
   - 调度器以 Compute‑NIC 为中心，维护两类资源的实时使用统计：存储 NIC 带宽占用和 Compute‑NIC 带宽占用。  
   - 每当有新请求或已有请求结束时，调度器重新评估哪条路径更空闲，并可能在运行时把已经在解码节点的 KV‑Cache 迁移回预填充节点，或相反。  
   - 这种自适应调度类似于“负载均衡器”，但它不仅平衡计算任务，还平衡 I/O 流量，确保两套网络资源都能高效利用。

**最巧妙的点**在于把原本只在预填充阶段才需要的大块 KV‑Cache 读取任务，拆分成两段：先小块读取到解码节点，再通过高速 RDMA 合并到预填充节点。这样既利用了闲置的解码 NIC，又把存储网络的流量转移到计算网络，彻底打破了原有的带宽不平衡。

### 实验与效果
- **测试场景**：作者在自研的生产级代理工作负载上评估了三种不同规模的 LLM（具体模型未在摘要中披露），包括离线批处理和在线实时服务两类场景。  
- **对比基线**：传统单路径存储‑到‑Prefill 系统，以及已有的 PD（prefill‑decode）分离实现。  
- **离线吞吐提升**：最高达到 1.87 倍，相当于在相同硬件下完成近两倍的请求量。  
- **在线 SLO（服务等级目标）保持**：在不违反延迟约束的前提下，平均提升 1.96 倍的吞吐。  
- **消融实验**：论文分别关闭“双路径读取”和“RDMA 迁移”两项功能，发现仅开启双路径但不使用 RDMA 时提升约 1.3 倍，说明 RDMA 的跨网络搬运是关键增益来源。  
- **局限性**：实验基于作者自建的内部系统，未在公开基准上复现；对极端大模型（KV‑Cache 超过数百 GB）时仍可能受限于存储节点的总带宽。原文未给出对不同网络拓扑（如多机房）下的适配情况。

### 影响与延伸思考
DualPath 把存储 I/O 与计算网络深度耦合，打开了“把闲置计算节点当作 I/O 加速器”的思路。后续的工作开始探索在异构集群中让 GPU 直接参与 KV‑Cache 读取，或把 NVMe-over-Fabrics 与 RDMA 进一步叠加，以进一步压缩存储‑到‑计算的时延。对想继续深入的读者，可以关注以下方向：① RDMA 在大模型训练/推理中的调度策略；② KV‑Cache 压缩与分层存储（如冷热分层）结合的系统设计；③ 多租户云环境下的存储‑计算协同调度。整体来看，DualPath 为解决代理式 LLM 推理的存储瓶颈提供了可落地的系统范式。

### 一句话记住它
DualPath 让解码节点也去“搬砖”，通过双路径读取加 RDMA 直传，把存储带宽瓶颈彻底拆掉。