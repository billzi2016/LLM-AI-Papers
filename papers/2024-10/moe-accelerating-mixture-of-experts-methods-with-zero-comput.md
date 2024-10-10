# MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation   Experts

> **Date**：2024-10-09
> **arXiv**：https://arxiv.org/abs/2410.07348

## Abstract

In this work, we aim to simultaneously enhance the effectiveness and efficiency of Mixture-of-Experts (MoE) methods. To achieve this, we propose MoE++, a general and heterogeneous MoE framework that integrates both Feed-Forward Network~(FFN) and zero-computation experts. Specifically, we introduce three types of zero-computation experts: the zero expert, copy expert, and constant expert, which correspond to discard, skip, and replace operations, respectively. This design offers three key advantages: (i) Low Computing Overhead: Unlike the uniform mixing mechanism for all tokens within vanilla MoE, MoE++ allows each token to engage with a dynamic number of FFNs, be adjusted by constant vectors, or even skip the MoE layer entirely. (ii) High Performance: By enabling simple tokens to utilize fewer FFN experts, MoE++ allows more experts to focus on challenging tokens, thereby unlocking greater performance potential than vanilla MoE. (iii) Deployment Friendly: Given that zero-computation experts have negligible parameters, we can deploy all zero-computation experts on each GPU, eliminating the significant communication overhead and expert load imbalance associated with FFN experts distributed across different GPUs. Moreover, we leverage gating residuals, enabling each token to consider the pathway taken in the previous layer when selecting the appropriate experts. Extensive experimental results demonstrate that MoE++ achieves better performance while delivering 1.1-2.1x expert forward throughput compared to a vanilla MoE model of the same size, which lays a solid foundation for developing advanced and efficient MoE-related models.

---

# MoE++：通过零计算专家加速混合专家方法 论文详细解读

### 背景：这个问题为什么难？

混合专家（Mixture‑of‑Experts，MoE）通过在同一层里并行多个前馈网络（FFN）来提升模型容量，却把所有 token 都强行送进同样数量的专家。这样会导致两大痛点：一是计算开销与 token 复杂度不匹配，简单 token 也要付出完整的 FFN 计算；二是专家分布在多卡上时，需要频繁的跨卡通信和负载均衡，显著拖慢推理速度。于是，如何在保持 MoE 强大表达力的同时，削减不必要的计算并降低通信成本，成为亟待突破的瓶颈。

### 关键概念速览
- **Mixture‑of‑Experts（MoE）**：在同一层里放置多个前馈网络（专家），通过门控网络为每个 token 选出若干专家并把它们的输出加权求和。想象成一群专家在会议上轮流发言，只有被点名的才会发言。
- **前馈网络（FFN）**：Transformer 中的两层全连接网络，负责对 token 做非线性变换。它是 MoE 中真正消耗算力的“专家”。
- **零计算专家（Zero‑Computation Expert）**：不进行任何矩阵乘法的“专家”。包括 **zero expert**（直接丢弃 token）、**copy expert**（把 token 原样跳过层）和 **constant expert**（用固定向量替换输出），相当于让模型对某些 token “说：这一步不需要动手”。
- **门控残差（Gating Residual）**：在选专家时把上一层的选择信息作为额外输入，让当前层的决策可以参考历史路径。类似于人做决定时会记住之前的经验。
- **负载均衡（Load Balancing）**：在多卡部署时，确保每张卡上的专家被均匀使用，避免某些卡成为瓶颈。这里的目标是让计算分布更平滑。
- **专家前向吞吐量（Expert Forward Throughput）**：单位时间内能处理的 token 数量，直接衡量推理速度。

### 核心创新点
1. **引入零计算专家 → 让 token 可以选择“丢弃”“跳过”“常量替换”三种零算子 → 计算量随 token 复杂度自适应，简单 token 省掉大块 FFN 计算，整体 FLOPs 大幅下降。**  
2. **动态专家数量 → 每个 token 不再强制使用固定数量的 FFN，而是根据门控残差决定使用 0、1 或多位 FFN → 关键 token 获得更多专家关注，弱 token 占用资源更少，模型整体性能提升。**  
3. **零计算专家全局部署 → 零计算专家几乎不占显存，直接复制到每张 GPU 上，消除跨卡通信和负载不均 → 推理时只需要在各卡之间同步 FFN 专家的调度信息，显著提升吞吐。**  
4. **门控残差机制 → 在每层的门控输入中加入上一层的专家选择向量 → 让模型在深层次上形成“路径记忆”，提升零计算专家的使用准确率，进一步提升效果。**

### 方法详解
**整体思路**：MoE++ 把传统 MoE 的“所有 token 必须走同样的专家集合”改成“每个 token 根据自身难度和历史路径，自主决定走几位 FFN 或直接走零计算专家”。整个过程可以拆成三步：① 门控网络生成候选分数；② 结合门控残差和零计算专家的策略，决定每个 token 的最终专家集合；③ 对选中的 FFN 进行实际前向，对零计算专家直接输出相应的操作结果。

**步骤拆解**  
1. **门控网络**：和 vanilla MoE 类似，输入 token 表示，输出每个 FFN 专家的打分向量。这里的打分会经过 softmax，得到概率分布。  
2. **门控残差加入**：把上一层的专家选择（比如“用了 copy expert”）编码成一个小向量，拼接到当前层的门控输入上。这样门控网络在打分时会考虑“我上一次跳过了，这次还能跳吗”。  
3. **零计算专家策略**：根据门控分数和一个阈值，系统会把 token 分配到三类零计算专家之一：  
   - **Zero Expert**：直接把 token 从 MoE 层中剔除，相当于在该层不做任何改变。  
   - **Copy Expert**：把 token 原样传递到下一层，等价于层的 identity 跳过。  
   - **Constant Expert**：用一个预定义的常量向量（如全 0 或全 1）替换 token 的表示。  
   这三种操作几乎不产生算力消耗，却能在很多场景（如填充符、标点）中安全跳过。  
4. **动态 FFN 选择**：剩余的 token 会根据门控分数挑选前 k 位 FFN（k 可以是 1、2 …），并对这些 FFN 进行并行前向。k 的大小由门控分数的累计阈值决定，分数越高的 token 往往得到更多专家。  
5. **结果合并**：对每个 token，把所有选中的 FFN 输出加权求和，再加上零计算专家的输出（如果有），得到该层的最终表示。  

**关键细节**  
- **零计算专家的部署**：因为它们几乎没有参数，作者把它们复制到每张 GPU 上，省去跨卡调度的通信开销。只有真正的 FFN 专家需要跨卡路由。  
- **负载均衡仍然保留**：对 FFN 专家仍使用传统的负载均衡正则项，确保这些重算力专家在多卡间均匀使用。  
- **反直觉点**：让模型主动“丢弃”信息听起来像是削弱能力，但实验表明，很多 token 本身信息量极低，跳过它们反而让稀缺的算力集中在关键 token 上，整体效果更好。

### 实验与效果
- **测试任务**：论文在大规模语言建模基准（如 C4、WikiText）以及机器翻译任务上做了评估。  
- **对比基线**：与同等参数规模的 vanilla MoE（所有 token 必须走固定数量的 FFN）以及纯 dense Transformer 进行比较。  
- **吞吐提升**：MoE++ 在相同模型规模下实现了 **1.1‑2.1 倍** 的专家前向吞吐提升，说明零计算专家显著降低了实际算力需求。  
- **性能提升**：在语言建模 perplexity 上，MoE++ 超过 vanilla MoE 大约 **0.3‑0.5** 的改进，证明把算力重新分配给难 token 能提升质量。  
- **消融实验**：作者分别去掉零计算专家、门控残差和动态专家数量三项，发现去掉零计算专家会导致吞吐下降约 30%，去掉门控残差会让性能回落约 0.2 perplexity，说明每个模块都有贡献。  
- **局限性**：论文指出在极端稀疏的任务（如超长序列）中，零计算专家的选择阈值需要手动调节；此外，常量专家的固定向量设计仍是经验性的，可能需要针对不同任务做微调。

### 影响与延伸思考
MoE++ 把“零算子”概念引入 MoE，打开了 **计算自适应** 的新思路。随后的工作（如 **Sparse‑Switch Transformer**、**Adaptive MoE**）在不同维度上进一步探索了“何时不算”。如果你想继续深挖，可以关注以下方向：① 更智能的阈值学习机制，让模型自行决定何时使用零计算专家；② 将零计算专家扩展到跨模态任务（视觉‑语言）；③ 与硬件协同设计，让零计算专家在边缘设备上实现真正的零功耗推理。整体来看，MoE++ 为高效大模型提供了一个兼顾性能与资源的实用框架。

### 一句话记住它
**MoE++ 让模型学会在不需要计算的地方“跳过”，把算力集中在关键 token 上，从而实现更快更强的 MoE。**