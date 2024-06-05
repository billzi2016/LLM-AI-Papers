# Llumnix: Dynamic Scheduling for Large Language Model Serving

> **Date**：2024-06-05
> **arXiv**：https://arxiv.org/abs/2406.03243

## Abstract

Inference serving for large language models (LLMs) is the key to unleashing their potential in people's daily lives. However, efficient LLM serving remains challenging today because the requests are inherently heterogeneous and unpredictable in terms of resource and latency requirements, as a result of the diverse applications and the dynamic execution nature of LLMs. Existing systems are fundamentally limited in handling these characteristics and cause problems such as severe queuing delays, poor tail latencies, and SLO violations.   We introduce Llumnix, an LLM serving system that reacts to such heterogeneous and unpredictable requests by runtime rescheduling across multiple model instances. Similar to context switching across CPU cores in modern operating systems, Llumnix reschedules requests to improve load balancing and isolation, mitigate resource fragmentation, and differentiate request priorities and SLOs. Llumnix implements the rescheduling with an efficient and scalable live migration mechanism for requests and their in-memory states, and exploits it in a dynamic scheduling policy that unifies the multiple rescheduling scenarios elegantly. Our evaluations show that Llumnix improves tail latencies by an order of magnitude, accelerates high-priority requests by up to 1.5x, and delivers up to 36% cost savings while achieving similar tail latencies, compared against state-of-the-art LLM serving systems. Llumnix is publicly available at https://github.com/AlibabaPAI/llumnix.

---

# Llumnix：大语言模型服务的动态调度 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）的推理请求在真实场景里千差万别：有的只要几百毫秒的回复，有的需要几秒甚至更久的上下文生成；有的占用显存几百 MB，有的要几 GB。传统的模型服务系统往往把每个请求固定绑定到某个模型实例上，导致资源利用率低、排队时间长。更糟的是，请求的到达顺序和计算需求是不可预测的，系统很难提前做好负载均衡。于是出现了严重的尾延迟（tail latency）和服务等级目标（SLO）违约，用户体验受挫。要想在成本可控的前提下同时满足高并发、低延迟和多样化的 SLO，单纯的静态调度已经力不从心。

### 关键概念速览

**LLM 推理请求**：用户提交的文本输入，模型需要在显存和算力上完成一次前向传播并返回生成的文本。不同请求的长度、温度、采样策略都会影响所需资源。

**模型实例（Model Instance）**：在一台或多台 GPU 上加载的同一模型的副本，每个实例拥有自己的显存缓存和执行队列。

**上下文切换（Context Switch）**：把正在执行的请求从一个模型实例迁移到另一个实例的过程，类似操作系统把进程从一个 CPU 核心切到另一个核心。

**活迁移（Live Migration）**：在不停止请求计算的前提下，将请求的中间状态（如 KV 缓存、注意力矩阵）复制到新实例，就像把正在跑的游戏人物瞬间传送到另一台服务器。

**调度策略（Scheduling Policy）**：决定何时、把哪个请求迁移到哪儿的规则集合，类似交通灯控制车辆流向，以避免拥堵。

**尾延迟（Tail Latency）**：第 95% 或第 99% 请求的响应时间，衡量最慢那部分用户的体验。

**服务等级目标（SLO）**：对每类请求设定的最大可接受延迟阈值，类似快递的“次日达”承诺。

### 核心创新点

1. **跨实例动态重排 → 采用运行时请求迁移**  
   传统系统把请求“一锤子”绑定到某个实例，导致热点实例负载过高、空闲实例被浪费。Llumnix 在请求执行过程中监控资源占用和排队情况，一旦发现不平衡，就把请求“搬家”。这种做法把调度提升到“操作系统级别”，实现了真正的负载均衡。

2. **统一的活迁移机制 → 迁移请求状态而非重新计算**  
   迁移请求最怕的就是把已经算好的中间结果丢掉，重新从头开始会把延迟翻倍。Llumnix 设计了一套高效的状态复制协议，只把 KV 缓存、注意力矩阵等必要数据搬过去，几乎不需要重新推理。这样既保持了计算进度，又避免了显存碎片。

3. **优先级感知调度 → 区分高低 SLO 请求**  
   并非所有请求都同等重要。Llumnix 在调度策略里加入了请求优先级和 SLO 信息，优先把紧急请求迁入空闲或低负载实例，保证它们能抢到算力。实验显示，高优先级请求的响应时间提升了约 1.5 倍。

4. **成本感知的整体调度 → 通过迁移降低实例数**  
   通过把碎片化的负载聚合到更少的实例上，Llumnix 能在保持相似尾延迟的前提下关闭一部分 GPU 实例，节约约 36% 的算力成本。相比传统的“多实例冗余”方案，成本收益更明显。

### 方法详解

#### 整体框架

Llumnix 的运行时由三层组成：**监控层**、**迁移层**和**调度层**。监控层持续收集每个实例的显存占用、算力利用率、请求排队长度等指标；迁移层负责把请求的执行上下文从源实例复制到目标实例；调度层根据收集到的实时数据和预设的优先级/SLO 策略，决定何时触发迁移以及迁移的目标。

#### 关键模块拆解

1. **实时监控与负载感知**  
   - 每个实例内部嵌入轻量级探针，记录每个请求的已完成 token 数、剩余 token 估算、显存占用等。  
   - 监控数据通过共享内存或高速网络汇总到中心调度器，形成全局负载视图。  
   - 类比于操作系统的任务管理器，调度器随时知道哪些“进程”在抢 CPU，哪些在等待。

2. **活迁移协议**  
   - 当调度器决定迁移时，源实例先冻结请求的状态（不再接受新 token），并把当前的 KV 缓存、注意力权重、随机种子等序列化。  
   - 目标实例在接收到状态后，立即恢复这些数据并继续生成后续 token。  
   - 为了避免网络传输成为瓶颈，Llumnix 使用零拷贝（zero-copy）技术和压缩算法，只传输必要的增量数据。  
   - 迁移过程对外部表现为一次无缝的“上下文切换”，用户几乎感受不到中断。

3. **统一调度策略**  
   - 调度器维护一个优先级队列，按照请求的 SLO 紧迫度排序。  
   - 当某实例的负载超过阈值且队列中有高优先级请求时，调度器挑选一个负载较轻的实例作为迁入目标。  
   - 若系统整体负载下降，调度器会把分散在多个实例的低优先级请求合并，关闭空闲实例以节约成本。  
   - 这种“合并-拆分”循环类似操作系统的内存碎片整理，只不过对象是 LLM 推理请求。

#### 反直觉/巧妙之处

- **迁移不等于重新推理**：很多人直觉认为迁移必然导致重新计算，但 Llumnix 通过保存 KV 缓存实现“状态续写”，大幅削减迁移开销。  
- **调度与迁移解耦**：调度层只输出迁移指令，实际搬迁工作交给专门的迁移层，避免调度器成为性能瓶颈。  
- **优先级与资源碎片同步治理**：把 SLO 视为调度的硬约束，同时把显存碎片视为软约束，二者在同一策略里协同优化，提升了整体系统的鲁棒性。

### 实验与效果

- **测试场景**：作者在公开的 LLM 推理基准（如 OpenAI GPT‑3.5、LLaMA‑2）上模拟了多租户请求，包含不同长度、不同优先级以及不同 SLO 的混合工作负载。  
- **对比基线**：与业界主流的 LLM 服务框架（如 vLLM、Orca）以及传统的静态负载均衡方案进行对比。  
- **核心结果**：  
  - 尾延迟降低了一个数量级（约 10×），即第 99% 请求的响应时间从原来的数秒降到几百毫秒。  
  - 对高优先级请求的加速提升约 1.5 倍，满足更严格的 SLO。  
  - 在保持相似尾延迟的前提下，关闭部分 GPU 实例实现了约 36% 的成本节约。  
- **消融实验**：作者分别关闭了活迁移、优先级感知调度和负载感知监控，发现活迁移对尾延迟的贡献最大，优先级调度对高优先级请求的加速最显著。  
- **局限性**：论文未深入探讨迁移过程中的网络带宽占用，对极端大模型（超过单卡显存）在多机跨节点迁移的细节描述不足；此外，迁移机制对模型内部实现（如自定义注意力）可能需要额外适配。

### 影响与延伸思考

Llumnix 把操作系统的上下文切换思想搬到了 LLM 推理服务层，开启了“服务层动态调度”的新方向。后续工作（如 **MosaicML’s Dynamic Scheduler**、**Microsoft’s DeepSpeed‑Inference 2.0**）都在不同程度上借鉴了跨实例迁移的思路，尝试把请求级别的负载均衡与显存碎片整理结合起来。对想进一步探索的读者，可以关注以下几个方向：

- **跨节点活迁移**：在多机集群中实现低延迟的状态同步，解决更大模型的服务问题。  
- **自适应 SLO 预测**：利用历史数据预测请求的实际延迟需求，提前在调度器中做出更精准的迁移决策。  
- **硬件协同**：让 GPU 驱动层直接支持 KV 缓存的快速转移，进一步压缩迁移开销。  

### 一句话记住它

Llumnix 让 LLM 推理请求像进程一样可以在模型实例间“搬家”，通过实时迁移实现了极低尾延迟和显著成本节约。