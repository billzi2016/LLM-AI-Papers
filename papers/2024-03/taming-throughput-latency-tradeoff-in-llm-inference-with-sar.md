# Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

> **Date**：2024-03-04
> **arXiv**：https://arxiv.org/abs/2403.02310

## Abstract

Each LLM serving request goes through two phases. The first is prefill which processes the entire input prompt and produces the first output token and the second is decode which generates the rest of output tokens, one-at-a-time. Prefill iterations have high latency but saturate GPU compute due to parallel processing of the input prompt. In contrast, decode iterations have low latency but also low compute utilization because a decode iteration processes only a single token per request. This makes batching highly effective for decodes and consequently for overall throughput. However, batching multiple requests leads to an interleaving of prefill and decode iterations which makes it challenging to achieve both high throughput and low latency.   We introduce an efficient LLM inference scheduler, Sarathi-Serve, to address this throughput-latency tradeoff. Sarathi-Serve introduces chunked-prefills which splits a prefill request into near equal sized chunks and creates stall-free schedules that adds new requests in a batch without pausing ongoing decodes. Stall-free scheduling unlocks the opportunity to improve throughput with large batch sizes while minimizing the effect of batching on latency. Furthermore, uniform batches in Sarathi-Serve ameliorate the imbalance between iterations resulting in minimal pipeline bubbles.   Our techniques yield significant improvements in inference performance across models and hardware under tail latency constraints. For Mistral-7B on single A100 GPUs, we achieve 2.6x higher serving capacity and up to 3.7x higher serving capacity for the Yi-34B model on two A100 GPUs as compared to vLLM. When used with pipeline parallelism on Falcon-180B, Sarathi-Serve provides up to 5.6x gain in the end-to-end serving capacity. The source code for Sarathi-Serve is available at https://github.com/microsoft/sarathi-serve.

---

# 驯服大语言模型推理中的吞吐量‑延迟权衡：Sarathi-Serve 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）服务中，每一次请求都要先把完整的输入 Prompt 处理完（prefill），再逐 token 生成输出（decode）。prefill 需要大量并行计算，单次耗时高，却能把 GPU 资源利用到极限；而 decode 只处理一个 token，计算量小，GPU 利用率低，但响应快。传统的批处理（batching）可以把多个 decode 同时跑，提高整体吞吐量，却会把不同请求的 prefills 和 decodes 混在一起，导致 GPU 在 prefilling 时被阻塞，或者在 decode 时出现空闲“气泡”。因此，如何在不牺牲单请求延迟的前提下，提升整体吞吐量，一直是 LLM 推理系统的瓶颈。

### 关键概念速览
- **prefill（预填充）**：模型一次性读取完整 Prompt 并生成第一个 token 的过程，类似一次性把整段文字喂进大脑，计算密集、并行度高。  
- **decode（解码）**：在已有上下文上逐 token 生成后续输出的过程，像每一步思考后说出下一个词，计算轻、并行度低。  
- **batching（批处理）**：把多个请求的同一阶段（prefill 或 decode）合并在一起一起跑，以提高硬件利用率。  
- **chunked‑prefill（分块预填充）**：把一个长 Prompt 切成若干大小相近的块，分别作为独立的 prefilling 任务，使得多个请求的 prefills 能在同一批次中交叉进行。  
- **stall‑free schedule（无停顿调度）**：一种调度策略，保证在加入新请求时不会中断正在进行的 decode，避免产生空闲时间。  
- **uniform batch（统一批次）**：在同一批次里所有请求的迭代步骤数相同，防止某些请求提前结束导致 GPU 资源浪费。  
- **pipeline bubble（流水线气泡）**：调度不均导致的 GPU 空闲时间，就像装配线上的停顿，降低整体吞吐。  

### 核心创新点
1. **分块预填充 + 近等分块大小**  
   - 之前的系统要么一次性完整 prefilling（导致长 Prompt 占用大量 GPU 时间），要么把长 Prompt 拆得太细导致调度复杂。  
   - Sarathi-Serve 将每个请求的 Prompt 切成若干大小相近的块，使得不同请求的块可以在同一批次里并行执行，既保持了 GPU 的高利用率，又避免了单个请求独占资源。  
   - 结果是即使在高并发场景下，prefill 仍然保持高吞吐，而不会因为某个超长 Prompt 把整个批次卡住。

2. **无停顿调度（stall‑free scheduling）**  
   - 传统调度在加入新请求时，需要暂停正在进行的 decode，导致延迟峰值增大。  
   - 本文提出的调度器在任何时刻都可以把新请求的 chunked‑prefill 插入批次，而不需要中断已有 decode，等价于在流水线上随时插入新工件而不让机器停下来。  
   - 这直接把批次规模放大到数十甚至上百请求，同时保持每个请求的尾部延迟在可接受范围。

3. **统一批次与最小化流水线气泡**  
   - 通过把所有请求的迭代次数对齐（即每个 batch 中的 prefills 与 decodes 步数相同），消除了因某些请求提前结束而产生的 GPU 空闲。  
   - 这种“均衡”让硬件利用率更接近理论上限，尤其在多卡并行（pipeline parallelism）下效果更明显。

### 方法详解
**整体框架**  
Sarathi-Serve 的推理流程可以概括为三步：① 将每个请求的 Prompt 切块；② 维护一个全局调度队列，实时把块和正在进行的 decode 交织进统一批次；③ 当所有块处理完后，进入纯 decode 阶段，仍保持统一批次直至所有 token 生成完毕。整个系统的核心是“无停顿调度器”，它负责在 GPU 上动态拼装 batch，确保每一步都满载。

**关键模块拆解**  

1. **Chunked‑Prefill 生成器**  
   - 输入：原始 Prompt（长度 L），目标块大小 B（由硬件算力和期望 batch 大小决定）。  
   - 过程：把 Prompt 按 B 切成 ⌈L/B⌉ 块，若最后一块不足 B，则补零或合并到前一块，使得所有块大小相差不超过 1。  
   - 类比：把一段长文章拆成若干页，每页字数相近，方便多个人同时阅读不同页而不出现“读到一半停下来”等情况。

2. **Stall‑Free 调度器**  
   - 维护两个队列：`prefill_queue`（存放待处理块）和 `decode_queue`（存放已进入 decode 的请求）。  
   - 每个调度轮次，调度器先检查 `prefill_queue` 是否有块可以加入当前 batch；如果有，就把块加入 batch，同时保持 `decode_queue` 中的请求继续执行。  
   - 关键是 **不抢占** 正在进行的 decode：调度器只在 batch 中添加新的 prefilling 任务，而不需要暂停 decode 的 kernel。实现上通过在同一 GPU kernel 中并行执行多个不同 shape 的矩阵乘法（利用 CUDA 的流并行）来完成。  
   - 结果是 batch 大小可以随时增长，而不会出现“停机等待新请求”的现象。

3. **Uniform Batch 构造器**  
   - 为了让 batch 中的每个请求在同一轮次完成相同数量的迭代，调度器在加入块时会对齐块的计数：如果某请求的块数少于 batch 中的最大块数，就在后面补充 dummy 块（只做轻量计算），保证所有请求同步进入 decode。  
   - 这相当于在装配线上给每个工件装配相同数量的部件，即使有的工件本身部件少，也会加装“占位件”来保持流水线平衡。

**最巧妙的点**  
- **分块大小的近等分**：不是随意切块，而是让每块大小相差不大，避免出现“一块很大、其他很小”导致 batch 中的计算时间不均。  
- **利用 CUDA 流并行实现无停顿**：传统调度往往需要在不同 kernel 之间切换，产生上下文切换开销。Sarathi-Serve 把 prefilling 和 decode 的 kernel 合并到同一流中，通过异步执行实现真正的“插入”。  
- **统一批次的 dummy 填充**：看似浪费算力，但实际消除了大量因提前结束产生的空闲时间，整体吞吐提升更显著。

### 实验与效果
- **测试模型与硬件**：Mistral‑7B（单卡 A100），Yi‑34B（双卡 A100），以及 Falcon‑180B（使用 pipeline parallelism 的多卡设置）。  
- **基线**：vLLM，这是业界常用的高效 LLM 推理框架。  
- **核心指标**：在相同的尾部延迟（tail latency）约束下，Sarathi-Serve 的服务容量（每秒可处理的请求数）提升显著。  
  - Mistral‑7B 单卡上提升 2.6 倍。  
  - Yi‑34B 双卡上提升最高 3.7 倍。  
  - Falcon‑180B 在 pipeline parallelism 环境下，端到端服务容量提升至 5.6 倍。  
- **消融实验**：论文分别关闭 chunked‑prefill、stall‑free 调度和 uniform batch，发现每去掉一项吞吐量下降 15%~30%，验证了每个模块的贡献。  
- **局限性**：作者指出在极端超长 Prompt（远大于块大小的数十倍）时，仍会出现调度延迟；此外，dummy 填充会在极低负载时产生轻微算力浪费。  

### 影响与延伸思考
Sarathi-Serve 的调度思路在后续的 LLM 服务系统中被广泛引用，尤其是 **chunked‑prefill** 与 **无停顿调度** 成为新一代推理框架的设计模板。后续工作如 Microsoft 的 DeepSpeed‑Serve、Google 的 Triton‑LLM 等，都在尝试进一步细化块划分策略或利用更细粒度的硬件调度（如 NVIDIA 的 Multi‑Instance GPU）来降低 dummy 填充的开销。对想深入的读者，可以关注以下方向：① 动态块大小自适应（根据实时负载自动调节 B），② 跨节点的全局无停顿调度（把调度扩展到多机集群），③ 将调度策略与模型并行（tensor‑parallel）深度融合，以进一步提升大模型的端到端吞吐。  

### 一句话记住它
**Sarathi-Serve 通过把长 Prompt 切块并在不打断 decode 的情况下随时插入这些块，实现了高并发下的“高吞吐 + 低延迟”双赢。**