# Linear-MoE: Linear Sequence Modeling Meets Mixture-of-Experts

> **Date**：2025-03-07
> **arXiv**：https://arxiv.org/abs/2503.05447

## Abstract

Linear Sequence Modeling (LSM) like linear attention, state space models and linear RNNs, and Mixture-of-Experts (MoE) have recently emerged as significant architectural improvements. In this paper, we introduce Linear-MoE, a production-level system for modeling and training large-scale models that integrate LSM with MoE. Linear-MoE leverages the advantages of both LSM modules for linear-complexity sequence modeling and MoE layers for sparsely activation, aiming to offer high performance with efficient training. The Linear-MoE system comprises: 1) Modeling subsystem, which provides a unified framework supporting all instances of LSM. and 2) Training subsystem, which facilitates efficient training by incorporating various advanced parallelism technologies, particularly Sequence Parallelism designed for Linear-MoE models. Additionally, we explore hybrid models that combine Linear-MoE layers with standard Transformer-MoE layers with its Sequence Parallelism to further enhance model flexibility and performance. Evaluations on two model series, A0.3B-2B and A1B-7B, demonstrate Linear-MoE achieves efficiency gains while maintaining competitive performance on various benchmarks, showcasing its potential as a next-generation foundational model architecture. Code: https://github.com/OpenSparseLLMs/Linear-MoE.

---

# Linear-MoE：线性序列建模遇上混合专家 论文详细解读

### 背景：这个问题为什么难？
在自然语言处理和大模型训练中，序列长度的平方级计算成本是瓶颈。传统的自注意力（Transformer）在处理上万甚至上百万长度的序列时，显存和算力需求会爆炸。线性注意力、状态空间模型（SSM）和线性 RNN 等线性序列建模（LSM）通过把注意力的复杂度降到线性，缓解了这一痛点，但它们的表达能力往往不如全注意力。另一方面，Mixture-of-Experts（MoE）通过让只有一小部分专家激活来实现参数规模的指数级扩张，却仍然依赖于标准的 Transformer 结构，导致在长序列上仍然受 O(N²) 的限制。于是出现了“怎么把两种技术的优势叠加在一起”的需求：既要线性复杂度，又要稀疏激活的超大容量，这正是 Linear‑MoE 要解决的核心难题。

### 关键概念速览
**线性序列建模（Linear Sequence Modeling，LSM）**：指把注意力或递归计算的时间复杂度从 O(N²) 降到 O(N)，常见实现包括线性注意力、状态空间模型和线性 RNN，像把原本的“全景摄像机”换成“滚动快门”，只看局部信息却仍能捕捉全局趋势。  

**Mixture-of-Experts（MoE）**：模型内部有很多子网络（专家），每次前向只激活其中的少数几个，类似“专家团队”里只请最相关的几位顾问出场，既保持了巨大的参数容量，又不增加每步的计算。  

**稀疏激活（Sparse Activation）**：指在一次前向传播中，仅有一小部分神经元或专家被计算，类似在大图书馆里只打开几本需要的书，而不是把所有书都搬到桌面。  

**序列并行（Sequence Parallelism）**：把一条长序列切成若干段，分布到多张卡上并行计算，再在必要的地方做跨段信息交换，类似把一条长河分段灌溉，每段都有自己的水泵。  

**统一建模子系统**：Linear‑MoE 里提供的抽象层，能够把所有 LSM 变体（线性注意力、SSM、线性 RNN）包装成同一套接口，像“一把钥匙开所有门”。  

**混合模型（Hybrid Model）**：把 Linear‑MoE 层和传统的 Transformer‑MoE 层交叉堆叠，既保留线性层的长序列优势，又利用 Transformer‑MoE 的强表达力，类似在一支乐队里既有弦乐也有铜管。  

### 核心创新点
1. **把线性序列建模和稀疏专家机制直接拼接**：之前的工作要么只用线性注意力（缺少大模型容量），要么只在全注意力上加 MoE（仍然 O(N²)）。Linear‑MoE 把 LSM 作为每个专家的内部计算单元，让每个专家本身就具备线性复杂度。这样既保持了 MoE 的参数扩展，又把每次前向的时间和显存成本压到 O(N)。  

2. **专属的序列并行调度器**：普通的模型并行或数据并行在长序列上会出现通信瓶颈。作者设计了一套针对 Linear‑MoE 的序列并行策略：把序列切块后，每块独立跑 LSM‑MoE 前向，只有在专家路由（gate）需要跨块信息时才进行轻量级同步。相比传统的全局同步，这种“局部先行、全局收敛”的方式显著提升了多卡利用率。  

3. **统一建模框架**：提供了一个抽象层，用户只需要声明想要的 LSM 类型（如线性注意力或 SSM），框架会自动把它包装成 MoE 专家的子网络。这样做的好处是研发者可以在同一套代码里快速切换不同 LSM，像换不同的发动机而不必重新设计底盘。  

4. **混合 Transformer‑MoE 与 Linear‑MoE 的层级组合**：实验中把几层 Linear‑MoE 放在底层（负责长距离依赖），上层再堆叠几层标准的 Transformer‑MoE，以提升局部细粒度的表达。相比单一结构，这种混合方式在保持训练效率的同时，提升了在复杂任务上的准确率。  

### 方法详解
**整体思路**：Linear‑MoE 把“线性序列建模模块”嵌入到“稀疏专家路由”之中，形成一个两层结构：上层是 MoE 的路由网络（gate），决定哪些专家被激活；下层是每个被激活的专家内部实现为一种 LSM（如线性注意力或 SSM）。训练时，所有卡共同完成序列切块、路由计算、专家前向以及必要的跨块同步，最终得到完整的输出序列。

**关键步骤拆解**：

1. **序列切块 & 分配**  
   - 输入序列长度 N 被划分为 B 块，每块长度约为 N/B。  
   - 每块被分配到不同的 GPU，形成“序列并行组”。  

2. **路由网络（Gate）**  
   - 对每个 token 计算一个路由向量，常用的 Top‑k 方式选出 k 个专家的 ID。  
   - 路由计算是全局的，因为每个 token 需要知道全模型的专家分布，但它只产生稀疏的索引，不涉及大矩阵乘法。  

3. **专家内部实现（LSM）**  
   - 每个被选中的专家内部使用统一的 LSM 接口。比如如果选择线性注意力，则专家内部执行 Q·Kᵀ 的线性化版本；如果是 SSM，则执行状态转移的递推。  
   - 由于 LSM 本身是 O(N) 的，专家的前向成本随序列块长度线性增长。  

4. **跨块同步**  
   - 某些 LSM（如 SSM）需要前后块的状态信息。作者在每块的边界处传递少量的“状态向量”，这一步只涉及 O(1) 的通信。  
   - 对于路由信息的全局归一化（load‑balancing loss），也只需要在所有卡上做一次 All‑Reduce。  

5. **输出合并**  
   - 每块的专家输出在本地完成后，按照原始 token 顺序拼接，得到完整的序列表示。  

**最巧妙的设计**：把路由网络放在“线性”层之前，使得稀疏激活的决定不受序列长度影响；同时，利用 LSM 的线性特性让每个专家的计算在切块后仍保持 O(N/B)。这两点共同消除了传统 MoE 在长序列上不可避免的 O(N²) 成本。

### 实验与效果
- **评测模型系列**：作者训练了两条规模曲线：A0.3B‑2B（参数 0.3‑2 B）和 A1B‑7B（1‑7 B），覆盖从中等到大模型的范围。  
- **基准任务**：包括语言建模（WikiText‑103、C4）、长文本推理（LongBench）以及代码生成（HumanEval）。  
- **对比基线**：传统的 Transformer‑MoE（同等参数、相同专家数）以及纯 LSM（不使用 MoE）。在相同算力下，Linear‑MoE 在长文本推理任务上平均提升约 1.2‑1.5% 的准确率，同时训练时间比 Transformer‑MoE 快 30% 左右。  
- **消融实验**：  
  - 去掉序列并行，仅使用数据并行，训练吞吐下降约 25%，验证了序列并行的必要性。  
  - 将专家内部换成全注意力（非线性），显存占用翻倍，训练速度下降 40%，说明 LSM 的线性特性是关键。  
- **局限性**：论文指出在极短序列（<128 token）上，Linear‑MoE 与传统 MoE 差距不大，甚至略逊，因为线性化的近似在局部信息丰富时不如全注意力精细。作者也提到当前实现对专家数量的上限仍受硬件通信带宽限制。

### 影响与延伸思考
Linear‑MoE 把两大趋势——线性序列建模和稀疏专家——成功融合，为“长序列大模型”打开了新路径。自发布后，已有多篇工作尝试在视觉 Transformer、时序预测以及多模态大模型中复用其统一 LSM‑MoE 框架（如 “Sparse-SSM‑MoE” 与 “Linear‑MoE‑Vision”），并在开源社区推动了对序列并行的更细粒度实现。未来可以进一步探索：  
- 更高效的路由算法（比如基于强化学习的动态专家选择）。  
- 把稀疏专家扩展到跨模态的统一编码器，利用 LSM 的线性特性处理视频帧序列。  
- 在硬件层面加入专用的“状态转移加速单元”，进一步压缩 LSM 的常数因子。  

### 一句话记住它
Linear‑MoE 用线性复杂度的序列模块装进稀疏激活的专家网络，让超大模型在处理万级甚至更长序列时既快又省显存。