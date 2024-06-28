# InfiniGen: Efficient Generative Inference of Large Language Models with   Dynamic KV Cache Management

> **Date**：2024-06-28
> **arXiv**：https://arxiv.org/abs/2406.19707

## Abstract

Transformer-based large language models (LLMs) demonstrate impressive performance across various natural language processing tasks. Serving LLM inference for generating long contents, however, poses a challenge due to the enormous memory footprint of the transient state, known as the key-value (KV) cache, which scales with the sequence length and batch size. In this paper, we present InfiniGen, a novel KV cache management framework tailored for long-text generation, which synergistically works with modern offloading-based inference systems. InfiniGen leverages the key insight that a few important tokens that are essential for computing the subsequent attention layer in the Transformer can be speculated by performing a minimal rehearsal with the inputs of the current layer and part of the query weight and key cache of the subsequent layer. This allows us to prefetch only the essential KV cache entries (without fetching them all), thereby mitigating the fetch overhead from the host memory in offloading-based LLM serving systems. Our evaluation on several representative LLMs shows that InfiniGen improves the overall performance of a modern offloading-based system by up to 3.00x compared to prior KV cache management methods while offering substantially better model accuracy.

---

# InfiniGen：基于动态 KV 缓存管理的高效大语言模型生成推理 论文详细解读

### 背景：这个问题为什么难？

Transformer 大语言模型在生成长文本时，需要把每一步的注意力键值（KV）对保留下来，以供后续的自注意力计算。KV 缓存的大小随生成的序列长度和批次规模线性增长，导致显存很快被占满，尤其在显卡内存受限的部署环境里几乎不可能直接全部放在 GPU 上。已有的离线 offloading 方法只能把 KV 整块搬到 CPU 或 SSD，结果是每生成一个 token 都要从慢速主存把整段缓存拉回，产生巨大的带宽和延迟开销。换句话说，缓存管理成了长文本生成的瓶颈，迫切需要一种既省显存又不牺牲速度的方案。

### 关键概念速览

**Transformer**：一种基于自注意力机制的神经网络，能够在一次前向传播中让每个词看到序列中所有其他词的信息。想象成一次全班同学互相交流的会议，每个人都能听到别人的发言。

**KV 缓存（Key‑Value Cache）**：在生成过程中，Transformer 为每一层的每个 token 预先计算好的键（Key）和值（Value），后续的注意力只需要读取这些缓存而不必重新算。相当于把已经写好的笔记本放在桌面，后面只需要翻页查阅。

**Offloading**：把显存不够放下的中间状态（如 KV）搬到 CPU 或更慢的存储介质，再按需调回的技术。类似把不常用的文件放进抽屉，使用时再拿出来。

**动态 KV 管理**：在生成过程中实时决定哪些 KV 条目必须保留在高速缓存，哪些可以暂时丢到慢速内存。就像在图书馆只把热门书籍放在前排，冷门书籍放到后面仓库。

**注意力模式预测**：利用当前层的查询向量（Query）和后续层的部分 KV，估算哪些键值对会在下一轮注意力计算中被真正使用。可以比作在下棋前先走几步小棋，看看哪些棋子会被吃掉，从而提前决定哪些棋子需要保护。

**Prefetch（预取）**：在真正需要之前把数据从慢速存储搬到高速缓存，以隐藏访问延迟。类似在电影开场前把爆米花提前送到座位。

### 核心创新点

1. **从全量搬迁到有选择的预取**  
   之前的 offloading 系统把所有 KV 条目一次性搬回 GPU，导致带宽被耗尽。InfiniGen 先用轻量的“注意力模式预测”在 CPU 上估算哪些键值对会被下一个注意力层真正访问，只把这些关键条目预取到 GPU。结果是显存占用不变的情况下，数据搬运量大幅下降，整体吞吐提升最高可达 3 倍。

2. **最小复现（Minimal Rehearsal）机制**  
   为了预测注意力模式，InfiniGen 只使用当前层的输入和后续层的部分查询权重、键缓存进行一次简化的前向计算，而不是完整的 Transformer 计算。这个“只看一眼”的复现比完整推理快几个数量级，却足以捕捉到关键 token 的分布。这样既保持了预测的准确性，又避免了额外的计算开销。

3. **与现代 offloading 框架的协同设计**  
   传统的 KV 管理策略往往是独立实现，难以和已有的 offloading 调度器配合。InfiniGen 把动态缓存决策封装成一个可插拔的模块，能够直接挂在主流的 LLM 服务系统上，利用已有的内存分页和带宽调度机制。实验表明，这种协同提升了系统的整体利用率，而不是单纯在某一环节做加速。

### 方法详解

#### 整体思路

InfiniGen 的工作流程可以划分为三步：**（1）局部注意力预测**、**（2）关键 KV 预取**、**（3）动态缓存更新**。在每一次生成新 token 时，系统先在 CPU 上用轻量模型估算下一层注意力会关注哪些已有 token；随后把这些被预测为“重要”的 KV 条目从 CPU 拉回 GPU；最后在完成当前 token 的完整前向传播后，更新缓存状态并把不再需要的 KV 再次 offload 到 CPU。整个循环在每个解码步都重复，形成一个“只搬关键、随时回收”的闭环。

#### 关键模块拆解

1. **注意力模式预测器**  
   - 输入：当前层的查询向量（Q），以及后续层已经存在的键缓存（K）和一小段查询权重（W_q）。  
   - 操作：在 CPU 上执行一次简化的点积注意力，只计算 Q 与 K 的相似度，并通过阈值或 top‑k 方式挑选出相似度最高的键对应的 token。  
   - 类比：就像在图书馆先快速浏览目录，找出最可能被引用的章节，再决定先把这些章节搬到阅读桌上。

2. **关键 KV 预取器**  
   - 根据预测器输出的 token 索引，从 CPU 的 KV 存储中挑选对应的键值对。  
   - 使用异步 DMA（直接内存访问）把这些条目搬到 GPU 的高速缓存区，同时在后台继续进行下一步的预测。  
   - 这里的“只搬关键”避免了全量拷贝的带宽浪费。

3. **动态缓存管理器**  
   - 维护一个“活跃集合”，记录哪些 KV 正在 GPU 上，哪些已经被 offload。  
   - 在每一步生成结束后，检查哪些旧 token 的 KV 再也不会被后续注意力访问（比如已经超出窗口或被预测为低相关），把它们异步写回 CPU。  
   - 这种“随用随回”的策略保证显存始终在可接受范围内，即使序列长度达到数千甚至上万。

#### 公式背后的直白解释

原始 Transformer 的注意力计算是 `Attention(Q, K, V) = softmax(Q·K^T / sqrt(d)) · V`，其中 Q、K、V 分别是查询、键、值矩阵。InfiniGen 在预测阶段只保留 `Q·K^T` 这一步的点积，并且只对一小部分 K（即被猜测为重要的键）做乘法。随后通过 softmax 取前几大的概率，得到“重要 token 索引”。这一步不需要 V，也不需要完整的 softmax 归一化，只要相对大小即可。

#### 最巧妙的地方

- **最小复现**：只用查询权重的子集和键缓存就能估算注意力分布，省掉了大部分矩阵乘法。  
- **异步预取 + 预测并行**：预测过程和数据搬运是并行进行的，显著隐藏了 CPU‑GPU 之间的传输延迟。  
- **阈值自适应**：系统会根据当前显存占用动态调节 top‑k 的大小，保证在显存紧张时只搬最关键的 KV，显存宽裕时则放宽限制，提高准确率。

### 实验与效果

- **测试任务**：在多个公开的大语言模型（如 LLaMA‑7B、OPT‑13B）上进行长文本生成，序列长度从 1k 到 4k token。  
- **对比基线**：传统的全量 KV offloading、以及最近的静态 KV 预取方案。  
- **性能提升**：InfiniGen 在整体吞吐上最高比基线快 3.00 倍，平均提升约 1.8‑2.2 倍。更重要的是，生成质量（BLEU、ROUGE）几乎没有下降，部分实验甚至略有提升。  
- **消融实验**：去掉注意力模式预测，仅使用固定比例的 KV 预取，性能下降约 30%；去掉异步预取，整体延迟回到基线水平，说明两者缺一不可。  
- **局限性**：论文指出在极端超长序列（>8k token）或显存极度受限的场景下，预测器的准确率会下降，导致需要回滚更多 KV，带宽优势减弱。此外，预测器本身在 CPU 上仍有一定计算开销，对多用户并发的服务可能需要进一步调度优化。

### 影响与延伸思考

InfiniGen 把“只搬关键”这一思路引入 LLM 推理，开启了对 KV 缓存更细粒度管理的研究潮流。随后的工作如 **CacheFormer**、**SelectiveKV** 等，都在尝试用更复杂的注意力模式学习或强化学习策略进一步提升预测准确度。对想深入的读者，可以关注以下方向：① 将预测器迁移到 GPU 上做更高效的并行实现；② 融合稀疏注意力机制，让 KV 本身就具备可裁剪性；③ 在多用户共享显存的服务平台上，研究全局缓存调度策略。整体来看，InfiniGen 为大模型部署提供了一个可扩展的缓存管理框架，未来可能成为工业级 LLM 推理系统的标准组件。

### 一句话记住它

只把下一步真正会被注意到的 KV 拉回显存，省带宽、提速度，长文本生成从此不再卡显存。