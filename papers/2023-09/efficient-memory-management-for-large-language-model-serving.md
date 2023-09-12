# Efficient Memory Management for Large Language Model Serving with   PagedAttention

> **Date**：2023-09-12
> **arXiv**：https://arxiv.org/abs/2309.06180

## Abstract

High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size. To address this problem, we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems. On top of it, we build vLLM, an LLM serving system that achieves (1) near-zero waste in KV cache memory and (2) flexible sharing of KV cache within and across requests to further reduce memory usage. Our evaluations show that vLLM improves the throughput of popular LLMs by 2-4$\times$ with the same level of latency compared to the state-of-the-art systems, such as FasterTransformer and Orca. The improvement is more pronounced with longer sequences, larger models, and more complex decoding algorithms. vLLM's source code is publicly available at https://github.com/vllm-project/vllm

---

# 使用PagedAttention进行大语言模型服务的高效内存管理 论文详细解读

### 背景：这个问题为什么难？
在实际部署大语言模型（LLM）时，服务端需要把多个用户请求一起批处理，以充分利用 GPU 的并行算力。每个请求在生成新 token 时都会在显存里维护一个键值缓存（KV cache），它记录了所有已经算过的注意力键和值。随着对话长度增长，这块缓存会不断膨胀、收缩，且每个请求的缓存大小不一致。传统的显存分配方式是为每个请求预留一块连续的内存，结果导致两类浪费：一是碎片化——空闲的显存被切成很多小块，难以再次利用；二是冗余复制——不同请求的前缀相同，却各自保存一份相同的 KV 数据。显存是 GPU 上最稀缺的资源，这种浪费直接限制了批大小，进而拖慢吞吐量。于是，如何在不牺牲延迟的前提下，做到“用更少的显存跑更多的请求”，成为迫切需要解决的技术难题。

### 关键概念速览
**KV cache（键值缓存）**：在自回归生成时，模型会把每一步的注意力键（K）和值（V）存下来，以便后续 token 复用，类似于人类记笔记后再查阅的过程。  

**Batch（批处理）**：一次性把多个请求的计算合并在同一张显卡上并行执行，像是把几个人的作业一起批改，提高效率。  

**Virtual Memory（虚拟内存）**：操作系统给每个进程提供的“看不见的”连续地址空间，实际物理内存可能是离散的块。  

**Paging（分页）**：把虚拟内存划分成固定大小的页（page），需要时才映射到真实的物理页，未使用的页可以回收或共享。  

**PagedAttention**：本文提出的注意力实现方式，借鉴分页思想，把 KV cache 按页管理，动态映射、共享、回收，避免碎片。  

**Decoding algorithm（解码算法）**：生成文本时决定下一个 token 的策略，如贪心、采样、束搜索等，不同算法对 KV cache 的访问模式不同。  

**Sharing KV cache（KV 缓存共享）**：当多个请求的已生成序列前缀相同，直接让它们指向同一块 KV 页，省掉重复存储。

### 核心创新点
1. **传统的连续显存分配 → PagedAttention 的页式管理 → 显存碎片几乎消失**。作者把每个请求的 KV cache 切成固定大小的页，使用页表记录逻辑位置到物理页的映射，新增 token 时只分配/映射所需的页，释放时直接回收页而不是整块内存。  

2. **每个请求独占 KV cache → 跨请求 KV 页共享 → 冗余存储大幅削减**。当两个请求的已生成序列在某段完全相同，系统检测到相同的 KV 内容后，让它们共享同一页，后续的更新只在共享页上进行，避免了多份相同数据占显存。  

3. **显存使用率随序列长度线性增长 → 动态页回收 + 按需扩容 → 内存占用随实际需求波动**。在生成长文本时，旧的 KV 页会被标记为可回收，新的 token 只占用新增的页，显存占用随实际活跃 token 数而非最大序列长度膨胀。  

4. **单纯提升吞吐的硬件优化 → 在 vLLM 框架中集成 PagedAttention + 高效调度 → 同等延迟下吞吐提升 2‑4 倍**。作者把分页注意力嵌入到完整的服务系统 vLLM，配合请求调度器把空闲页重新分配给新请求，实现了显存利用率的整体提升，实验显示对主流模型（如 LLaMA、Falcon）在长序列、复杂解码场景下都有显著收益。

### 方法详解
**整体思路**：vLLM 把每一次 token 生成视为“页面访问”。系统维护一个全局的页池（page pool），每页大小固定（如 16 KB）。每个请求拥有自己的逻辑 KV 表（类似虚拟地址空间），通过页表映射到实际的物理页。当请求产生新 token 时，系统检查对应的逻辑位置是否已有映射；若没有，则从页池中取出空闲页并建立映射；若已有映射且该页已被其他请求共享，则直接复用。生成完毕后，旧的页如果不再被任何请求引用，就被标记为可回收，返回页池。

**关键模块拆解**：

1. **页池（Page Pool）**  
   - 类似操作系统的空闲页链表，预先在显存中划分若干等大小的块。  
   - 提供“分配页”和“回收页”两个原子操作，保证并发请求安全。

2. **逻辑 KV 表 & 页表（KV Table & Page Table）**  
   - 每个请求维护一个一维数组，记录每个 token 对应的 KV 页编号。  
   - 页表把逻辑索引（如第 57 个 token）映射到具体的显存页地址。

3. **共享检测（Sharing Detection）**  
   - 当新请求到来或已有请求的前缀与其他请求相同，系统会比较对应的 KV 哈希值。  
   - 哈希相同且引用计数为 1 时，直接把页表指向已有页，并把引用计数加一。

4. **注意力计算（PagedAttention）**  
   - 在执行注意力时，先根据页表把需要的 KV 页拉入寄存器，随后进行标准的矩阵乘法。  
   - 因为页大小固定，内存访问模式保持连续，硬件加速器仍能高效利用。

**最巧妙的点**：把 KV cache 的“逻辑连续性”与“物理离散性”解耦，让显存像操作系统的虚拟内存一样被动态调度。这样既保留了注意力计算的高效矩阵操作，又避免了显存碎片和重复存储——两者在传统实现里几乎不可兼得。

### 实验与效果
- **测试模型与任务**：作者在 LLaMA‑7B、LLaMA‑13B、Falcon‑40B 等主流大模型上，使用长文本生成（序列长度 1k‑4k）以及复杂解码算法（束搜索、Top‑p 采样）进行评估。  
- **对比基线**：与 FasterTransformer、Orca 这两套业界常用的高性能推理库进行对比。  
- **吞吐提升**：在相同硬件（如 A100 40GB）和相同延迟目标下，vLLM 的吞吐量提升了约 2‑4 倍。尤其在序列长度超过 2k、模型参数超过 10 B 时，提升幅度更接近上限。  
- **显存利用率**：KV cache 的碎片率降至几乎为零，实际显存占用比传统实现少 30‑50%。  
- **消融实验**：作者分别关闭“跨请求共享”和“动态页回收”两项功能，发现共享机制贡献约 1.5× 的吞吐提升，页回收则主要降低显存峰值。  
- **局限性**：论文未详细说明在极端高并发、请求长度分布极不均匀的场景下页表维护的开销；此外，页大小固定可能在某些模型的 KV 结构上产生轻微对齐损失。作者承认这些细节在实际部署中仍需调参。

### 影响与延伸思考
vLLM 作为开源项目迅速在业界获得关注，很多云服务商和企业内部部署的 LLM 推理平台已经采用了其分页注意力实现。随后出现的工作如 **FlashAttention‑2**、**DeepSpeed‑Inference** 等，都在显存管理上加入了类似的“按需分页”或“共享缓存”机制，进一步验证了本文思路的通用价值。未来的研究方向可能包括：自适应页大小（根据模型层特性动态调整）、跨节点的分布式 KV 缓存共享、以及把分页概念扩展到全模型参数的加载（模型并行的虚拟内存化）。对想深入的读者，建议直接阅读 vLLM 的源码实现以及后续的 “PagedAttention for Multi‑GPU” 预印本。

### 一句话记住它
PagedAttention 把操作系统的分页机制搬进 LLM 推理，让 KV 缓存几乎不浪费，从而把同等硬件的吞吐提升数倍。