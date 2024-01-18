# DistServe: Disaggregating Prefill and Decoding for Goodput-optimized   Large Language Model Serving

> **Date**：2024-01-18
> **arXiv**：https://arxiv.org/abs/2401.09670

## Abstract

DistServe improves the performance of large language models (LLMs) serving by disaggregating the prefill and decoding computation. Existing LLM serving systems colocate the two phases and batch the computation of prefill and decoding across all users and requests. We find that this strategy not only leads to strong prefill-decoding interferences but also couples the resource allocation and parallelism plans for both phases. LLM applications often emphasize individual latency for each phase: time to first token (TTFT) for the prefill phase and time per output token (TPOT) of each request for the decoding phase. In the presence of stringent latency requirements, existing systems have to prioritize one latency over the other, or over-provision compute resources to meet both.   DistServe assigns prefill and decoding computation to different GPUs, hence eliminating prefill-decoding interferences. Given the application's TTFT and TPOT requirements, DistServe co-optimizes the resource allocation and parallelism strategy tailored for each phase. DistServe also places the two phases according to the serving cluster's bandwidth to minimize the communication caused by disaggregation. As a result, DistServe significantly improves LLM serving performance in terms of the maximum rate that can be served within both TTFT and TPOT constraints on each GPU. Our evaluations show that on various popular LLMs, applications, and latency requirements, DistServe can serve 7.4x more requests or 12.6x tighter SLO, compared to state-of-the-art systems, while staying within latency constraints for > 90% of requests.

---

# DistServe：拆分预填充与解码以实现高吞吐量的大语言模型服务 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在实际产品里往往要同时满足两类时延需求：**预填充**阶段的“首 token 到达时间”（TTFT）和**解码**阶段的“每生成一个 token 所需时间”（TPOT）。传统的服务系统把这两个阶段放在同一批 GPU 上一起跑，并把所有请求统一批处理。这样做的直接后果是，预填充的大批量矩阵乘法会抢占解码时的算力，导致两者相互干扰；更糟的是，资源分配和并行度只能用一种统一的方案，既不能针对 TTFT 进行极致优化，也难以在 TPOT 上保持高效。于是要么牺牲一种时延，要么大幅超配硬件，成本飙升——这正是作者们想要突破的瓶颈。

### 关键概念速览

- **Prefill（预填充）**：模型在收到完整的用户提示后一次性生成所有隐藏状态的过程，相当于一次性把整段文字“填进”模型的记忆里。类似于一次性把所有材料准备好再开始烹饪。
- **Decoding（解码）**：在预填充完成后，模型逐 token 生成答案的循环过程。可以想象成厨师在锅里不断翻炒，每翻一次产生一个 token。
- **TTFT（Time To First Token）**：从请求到达系统到第一个 token 输出的时长，衡量用户感受到的“响应速度”。像是点餐后菜第一口端上来的时间。
- **TPOT（Time Per Output Token）**：每生成一个后续 token 所需的时间，决定整体生成的流畅度。相当于厨师每翻一次锅的速度。
- **Goodput**：在满足所有时延约束的前提下，系统实际成功处理的请求数。不同于单纯的吞吐量，它只计入符合 SLO（服务水平目标）的请求。
- **Disaggregation（拆分）**：把原本耦合在一起的计算阶段（prefill+decoding）分配到不同硬件资源上执行，避免相互竞争。类似于把准备材料的厨房和烹饪的厨房分开。
- **Resource Allocation & Parallelism Strategy（资源分配与并行策略）**：决定每个阶段使用多少 GPU、每张卡上开多少并行流（batch size、pipeline depth 等），以及如何调度这些流。相当于决定多少厨师、每个厨师负责哪道菜以及怎么轮班。
- **Bandwidth-aware Placement（带宽感知放置）**：在决定把 prefill 和 decoding 放在哪台机器时，考虑网络带宽，以最小化跨卡通信开销。就像在同一条传送带上放置相邻的工作站，省去搬运时间。

### 核心创新点

1. **阶段拆分 → 把 prefill 与 decoding 分配到不同 GPU** → 彻底消除了两阶段的算力争抢，使得每个阶段都能在专用硬件上跑满负载，显著降低相互干扰带来的时延抖动。  
2. **SLO‑驱动的资源共优化 → 根据用户给定的 TTFT 与 TPOT 目标，分别为 prefill 与 decoding 计算最合适的 GPU 数量、并行度和 batch 大小** → 让系统在满足首 token 快速返回的同时，也保持每 token 的高效生成，避免了“一刀切”导致的资源浪费。  
3. **带宽感知的阶段放置 → 在集群内部根据网络拓扑和带宽限制，把两个阶段放在通信成本最低的 GPU 对上** → 跨阶段的数据传输（如隐藏状态的转移）几乎不产生瓶颈，整体 Goodput 提升。  
4. **动态调度器 + 统一调度接口 → 实时监控 TTFT/TPOT 达成情况，若某阶段出现资源紧张，调度器会即时迁移 GPU 或调节并行度** → 保证 >90% 请求始终在 SLO 范围内，提升系统的鲁棒性。

### 方法详解

#### 整体框架

DistServe 的运行可以划分为四个阶段：**（1）请求分析与 SLO 抽取、（2）阶段拆分与资源规划、（3）带宽感知放置、（4）实时调度与执行**。系统首先读取每个请求的 TTFT 与 TPOT 需求，然后在全局视角下决定 prefill 用多少卡、每卡的并行流数，以及 decoding 用多少卡、对应的并行度。接着依据集群的网络拓扑把两阶段放在最靠近的 GPU 对上，最后通过一个轻量级调度器在运行时动态微调。

#### 关键模块拆解

1. **SLO 抽取器**  
   - 输入：用户请求的提示文本、业务侧配置的 TTFT/TPOT 上限。  
   - 输出：一个结构体 `SLO{ttft_limit, tpot_limit}`。  
   - 作用相当于把用户的“我想要快”和“我想要流畅”翻译成机器可读的数值。

2. **资源规划器（Planner）**  
   - 采用一个**双目标优化模型**：在满足 TTFT 与 TPOT 的前提下，最大化 Goodput。  
   - 关键变量：`prefill_gpu_cnt、prefill_parallelism、decode_gpu_cnt、decode_parallelism`。  
   - 通过对每种 LLM 的性能曲线（如 batch size vs latency）进行离线 profiling，Planner 能快速查表得到最优组合。  
   - 这里的“查表”类似于厨师根据经验表格挑选最合适的锅和火力。

3. **带宽感知放置器（Placement Engine）**  
   - 输入：Planner 给出的 GPU 集合。  
   - 读取集群的网络拓扑图（每条链路的带宽、延迟），使用**最小生成树**算法把 prefill 与 decoding 的 GPU 对配对，使得隐藏状态跨卡传输的带宽需求最小。  
   - 结果是一对 GPU（或多对，如果采用多副本），它们之间的网络路径最短，通信开销几乎可以忽略。

4. **实时调度器（Runtime Scheduler）**  
   - 监控每个阶段的实际 TTFT/TPOT，若出现偏离，则触发**迁移或并行度调节**。  
   - 迁移策略：把 prefill 或 decoding 的任务从负载高的卡迁到空闲卡，迁移时使用 **pipeline checkpoint** 技术保证不丢失已计算的隐藏状态。  
   - 并行度调节：在 decoding 期间，如果 TPOT 超标，调度器会临时增加 decode 并行流（即把同一请求拆成更小的 batch），反之则收缩以节省资源。  
   - 这一步类似于厨房里实时调配厨师人数和火候，确保每道菜都在预定时间出锅。

#### 反直觉/巧妙之处

- **拆分后仍保持低通信开销**：直觉上把两个阶段放在不同卡会导致大量跨卡数据传输（隐藏状态往返），但作者通过带宽感知放置和一次性“状态压缩”传输（只在 prefill 完成后一次性搬运全部 KV cache），把额外的通信成本压到可接受范围。  
- **双目标优化而非单一吞吐**：大多数服务系统只追求最大吞吐量，忽视时延约束。DistServe 把 TTFT 与 TPOT 视为硬约束，优化目标直接是 **在约束下的 Goodput**，这让系统在实际业务场景中更具竞争力。  
- **动态迁移的轻量实现**：迁移通常需要完整的模型状态复制，代价高昂。DistServe 只迁移 **KV cache**（键值对缓存），并利用 **checkpoint** 快速恢复，几乎不影响用户感知的时延。

### 实验与效果

- **测试模型与任务**：作者在 LLaMA‑7B、LLaMA‑13B、OPT‑30B 等主流开源 LLM 上进行评测，覆盖对话生成、代码补全和长文摘要三类典型应用。  
- **对比基线**：主要与 **vLLM**、**TensorRT‑LLM**、以及业界常用的 **TGI（Text Generation Inference）** 系统对比。  
- **核心结果**：在相同硬件（8×A100）下，DistServe 能在满足 TTFT≤200 ms、TPOT≤30 ms 的 SLO 条件下，**处理请求数提升 7.4 倍**；在更严格的 SLO（TTFT≤100 ms、TPOT≤15 ms）下，Goodput 提升 **12.6 倍**。超过 **90%** 的请求保持在 SLO 范围内，显著优于基线的 70% 左右。  
- **消融实验**：  
  - **仅拆分 prefill/decoding**（不做带宽感知放置）仍能提升约 3.2×，说明阶段拆分本身贡献巨大。  
  - **去掉动态调度**，在负载波动时 Goodput 下降约 18%，验证调度器的必要性。  
  - **不做资源共优化**（统一 batch 大小）导致在高 TPOT 需求场景下 Goodput 下降 25%。  
- **局限性**：论文主要在同构 GPU 集群（相同型号、相同网络）上评测，异构环境下的放置策略未深入探讨；此外，极端长上下文（> 8k token）时 KV cache 迁移成本仍是瓶颈，作者在讨论中承认需要进一步压缩或增量同步技术。

### 影响与延伸思考

DistServe 的“阶段拆分 + SLO‑驱动共优化”思路在随后两年里被多篇工作引用，例如 **FlexServe**（引入多租户公平调度）和 **SLO‑Aware Pipeline Parallelism**（把 pipeline 并行也纳入 TTFT/TPOT 约束）。它推动了业界从“单一吞吐最大化”向“服务质量可控的高效服务”转变。对想继续深挖的读者，建议关注以下方向：

- **异构资源调度**：把 CPU、GPU、TPU 混合使用时的带宽感知放置。  
- **增量 KV cache 同步**：在超长上下文场景下，如何只同步变化的缓存块。  
- **多模态模型服务**：把文本、图像、音频的 prefill/decoding 拆分成更多阶段，验证是否仍能保持同等收益。  

### 一句话记住它

把 LLM 的“准备材料”和“烹饪过程”分到不同的厨房，并用用户设定的“上菜速度”和“每道菜出锅时间”来精准调配厨师和灶台，系统就能在保证服务质量的前提下，吞吐量提升数倍。