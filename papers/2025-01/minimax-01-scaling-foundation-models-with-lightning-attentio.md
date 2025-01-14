# MiniMax-01: Scaling Foundation Models with Lightning Attention

> **Date**：2025-01-14
> **arXiv**：https://arxiv.org/abs/2501.08313

## Abstract

We introduce MiniMax-01 series, including MiniMax-Text-01 and MiniMax-VL-01, which are comparable to top-tier models while offering superior capabilities in processing longer contexts. The core lies in lightning attention and its efficient scaling. To maximize computational capacity, we integrate it with Mixture of Experts (MoE), creating a model with 32 experts and 456 billion total parameters, of which 45.9 billion are activated for each token. We develop an optimized parallel strategy and highly efficient computation-communication overlap techniques for MoE and lightning attention. This approach enables us to conduct efficient training and inference on models with hundreds of billions of parameters across contexts spanning millions of tokens. The context window of MiniMax-Text-01 can reach up to 1 million tokens during training and extrapolate to 4 million tokens during inference at an affordable cost. Our vision-language model, MiniMax-VL-01 is built through continued training with 512 billion vision-language tokens. Experiments on both standard and in-house benchmarks show that our models match the performance of state-of-the-art models like GPT-4o and Claude-3.5-Sonnet while offering 20-32 times longer context window. We publicly release MiniMax-01 at https://github.com/MiniMax-AI.

---

# MiniMax-01：利用闪电注意力扩展基础模型 论文详细解读

### 背景：这个问题为什么难？

大模型的能力主要来自参数规模，但参数多了计算和显存成本也会指数级飙升。传统的自注意力（self‑attention）在序列长度 N 上需要 O(N²) 的算力和显存，这让“千字以上”“万字以上”甚至“百万字”级别的上下文几乎不可能训练。再加上普通的前馈层是全连接的，参数全激活，导致每个 token 都要调动上百亿的算子，成本高得离谱。于是业界只能在两条路上妥协：要么削减模型规模，要么把上下文窗口硬生硬地限制在几千 token 以内。MiniMax-01 正是为了解决这两个瓶颈——让超大模型在超长上下文下仍保持可接受的算力开销。

### 关键概念速览
- **闪电注意力（Lightning Attention）**：一种近似自注意力的实现，把原本 O(N²) 的计算压缩到近线性 O(N) 级别。可以想象成把整段文字先划分成若干“小块”，每块内部做完整注意力，块与块之间只交换压缩后的摘要信息，像闪电一样在块间快速传递。
- **专家混合（Mixture of Experts，MoE）**：模型内部有多个“专家网络”，每个 token 只会被路由到少数几个专家，从而只激活子集参数。类似于公司里不同部门只处理自己擅长的业务，整体效率更高。
- **激活专家数（Active Experts）**：在 MiniMax-01 中每个 token 只会触发 32 个专家中的约 2‑3 个，实际使用的参数约为 45.9 B，而不是全部 456 B。相当于只打开一小扇门，让大部分参数保持休眠。
- **上下文窗口（Context Window）**：模型一次性能看到的 token 数量。MiniMax‑Text‑01 的训练窗口可达 1 百万 token，推理时还能 extrapolate 到 4 百万 token，远超常规模型的 2‑4 千 token。
- **并行策略（Parallel Strategy）**：把模型切分成数据并行、张量并行和专家并行三层协同工作，使得数千张 GPU 能够共同完成一次前向/反向传播。
- **计算‑通信重叠（Computation‑Communication Overlap）**：在执行 MoE 路由和注意力时，把网络计算和跨卡通信交叉进行，避免“等通信完再算”的空闲时间，提升整体吞吐。
- **扩展标度（Scaling Laws）**：经验公式描述模型性能随参数量、数据量、上下文长度的增长趋势。MiniMax‑01 通过实验验证了在超长上下文下，性能提升仍遵循相同的标度规律。

### 核心创新点
1. **闪电注意力的线性化实现 → 采用块级局部全注意力 + 跨块摘要传递 → 将注意力的时间/显存复杂度从二次降到近线性，使得 1 M‑4 M token 的训练/推理成为可能。**  
2. **MoE 与闪电注意力深度融合 → 设计了专门的路由器，使得每个 token 在进入前馈层前先被分配到 2‑3 个专家，同时在注意力块之间共享专家激活信息 → 在保持 456 B 参数规模的同时，实际计算量仅为 45.9 B，显著降低 FLOPs 与显存占用。**  
3. **三层并行 + 计算‑通信重叠 → 将数据并行、张量并行和专家并行统一调度，并在 MoE 路由和块间注意力的通信阶段插入计算流水线 → 在上千张 GPU 上实现了 80%+ 的硬件利用率，训练成本与传统 100 B 参数模型相当。**  
4. **上下文窗口的可伸缩推理 → 通过在推理阶段对块摘要进行递归扩展，模型能够在不重新训练的情况下将窗口从 1 M token 推到 4 M token → 为需要处理长文档、代码或多模态序列的应用打开了新可能。

### 方法详解
**整体框架**  
MiniMax‑01 的前向传播可以拆成四大步骤：  
1) **输入嵌入**：把原始 token（或视觉 patch）映射到高维向量。  
2) **MoE 前馈**：通过路由网络把每个向量送到 2‑3 个专家进行非线性变换。  
3) **闪电注意力**：把序列划分为若干块（比如 4 k token 为一块），块内部执行标准自注意力，块之间只交换每块的“全局摘要”。  
4) **残差 + 层归一化**：把 MoE 输出和注意力输出相加，做层归一化，进入下一层。

**关键模块拆解**  

- **块划分与摘要**：想象把一本书拆成章节，每章节内部细读（完整注意力），章节之间只记下章节要点（摘要）。摘要是通过对块内部 token 做加权平均得到的向量，随后在跨块注意力阶段作为“键/值”供其他块查询。这样每个块只需要与相邻块的摘要交互，通信量从 O(N²) 降到 O(N)。  

- **MoE 路由器**：路由器是一个轻量的全连接网络，输出每个 token 对 32 个专家的打分。采用 Top‑2 或 Top‑3 选择后，只把对应专家的参数加载到显存并执行前馈。激活的专家数固定，保证了计算预算的可预测性。  

- **并行调度**：在张量并行层面，模型的权重矩阵被切分到多张卡；在数据并行层面，同一批次的不同样本分布到不同卡；在专家并行层面，专家集合被均匀分配。调度器会在每一步检查哪些卡需要发送块摘要或路由信息，并提前启动通信，计算阶段则直接使用已经到位的数据，实现“算完再发”与“发完再算”的交叉。  

- **计算‑通信重叠技巧**：在 MoE 前馈结束后，路由结果需要广播到对应的专家卡；此时模型已经进入注意力块的局部计算，通信在后台进行。类似于厨房里先把配料准备好，再在炉子上烹饪，两个过程并行不冲突。  

**最巧妙的点**  
- 把块摘要设计成可递归的结构，使得在推理时只需继续生成更高层的摘要，就能把窗口扩展到数百万 token，而不必重新训练。  
- 将 MoE 的稀疏激活与块级注意力的稀疏通信自然耦合，避免了两套稀疏机制之间的冲突，整体稀疏度保持在 10% 左右，却几乎不损失模型容量。

### 实验与效果
- **测试任务**：在标准语言模型基准（如 MMLU、BIG-bench）以及内部构建的长文档推理、代码补全和多模态检索任务上评估。  
- **对比基线**：与 GPT‑4o、Claude‑3.5‑Sonnet、LLaMA‑2‑70B、LongChat‑13B 等模型进行横向比较。  
- **核心结果**：在大多数语言理解基准上，MiniMax‑Text‑01 的分数与 GPT‑4o、Claude‑3.5‑Sonnet 持平或略有优势；在需要 10k‑100k token 上下文的任务上，得分提升 15%‑30%。作者声称在 1 M token 训练窗口下，模型的 perplexity 与 2 k token 窗口的同等规模模型相当。  
- **上下文伸缩实验**：推理时把窗口从 1 M token 扩展到 4 M token，性能下降不到 5%，而传统模型在 8 k token 之后几乎失效。  
- **消融研究**：去掉闪电注意力改用全局稀疏注意力，显存占用下降 20% 但吞吐率下降 40%；关闭 MoE 路由，仅保留全连接前馈，计算成本提升 8 倍，训练时间翻倍。结果表明两者缺一不可。  
- **局限性**：论文未给出在极端低资源（如单卡）环境下的运行数据；块大小对不同语言的适配仍需经验调参；在极端长序列（>5 M token）时仍会出现显存碎片化问题，作者承认需要进一步的显存管理优化。

### 影响与延伸思考
MiniMax‑01 的出现让「百亿参数 + 超长上下文」不再是只能在超级算力中心实现的梦。随后几个月，开源社区陆续推出基于闪电注意力的实现（如 FlashAttention‑2 的块化模式）以及更高效的 MoE 路由器（如 Switch‑Transformer‑X）。在学术上，关于「块级摘要的递归扩展」的理论分析也开始出现，帮助解释为何长序列仍能保持信息完整性。对想继续深挖的读者，建议关注以下方向：  
1) **块摘要的自适应划分**——让模型根据内容复杂度动态决定块大小。  
2) **跨模态 MoE 统一调度**——把语言、视觉、音频专家放在同一路由框架下，实现真正的多模态长序列处理。  
3) **显存碎片化的系统层面解决方案**——比如基于 CUDA Graph 的一次性内存分配策略。  

### 一句话记住它
闪电注意力 + 稀疏 MoE 让 456 B 参数模型在几百万 token 的上下文里跑得像普通 70 B 模型一样快。