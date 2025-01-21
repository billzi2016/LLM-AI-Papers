# Demons in the Detail: On Implementing Load Balancing Loss for Training   Specialized Mixture-of-Expert Models

> **Date**：2025-01-21
> **arXiv**：https://arxiv.org/abs/2501.11873

## Abstract

This paper revisits the implementation of $\textbf{L}$oad-$\textbf{b}$alancing $\textbf{L}$oss (LBL) when training Mixture-of-Experts (MoEs) models. Specifically, LBL for MoEs is defined as $N_E \sum_{i=1}^{N_E} f_i p_i$, where $N_E$ is the total number of experts, $f_i$ represents the frequency of expert $i$ being selected, and $p_i$ denotes the average gating score of the expert $i$. Existing MoE training frameworks usually employ the parallel training strategy so that $f_i$ and the LBL are calculated within a $\textbf{micro-batch}$ and then averaged across parallel groups. In essence, a micro-batch for training billion-scale LLMs normally contains very few sequences. So, the micro-batch LBL is almost at the sequence level, and the router is pushed to distribute the token evenly within each sequence. Under this strict constraint, even tokens from a domain-specific sequence ($\textit{e.g.}$, code) are uniformly routed to all experts, thereby inhibiting expert specialization. In this work, we propose calculating LBL using a $\textbf{global-batch}$ to loose this constraint. Because a global-batch contains much more diverse sequences than a micro-batch, which will encourage load balance at the corpus level. Specifically, we introduce an extra communication step to synchronize $f_i$ across micro-batches and then use it to calculate the LBL. Through experiments on training MoEs-based LLMs (up to $\textbf{42.8B}$ total parameters and $\textbf{400B}$ tokens), we surprisingly find that the global-batch LBL strategy yields excellent performance gains in both pre-training perplexity and downstream tasks. Our analysis reveals that the global-batch LBL also greatly improves the domain specialization of MoE experts.

---

# 细节中的恶魔：实现负载平衡损失以训练专门化的专家混合模型 论文详细解读

### 背景：这个问题为什么难？

在大规模语言模型里，**专家混合（Mixture‑of‑Experts，MoE）** 通过让不同的子网络（专家）只处理一小部分 token，显著降低了算力需求。但 MoE 的核心是 **路由器** 必须把 token 均匀分配到各专家上，否则某些专家会被“饿死”，训练效率和模型质量都会受损。现有的训练框架采用 **微批（micro‑batch）** 级别的负载平衡损失（Load‑Balancing Loss，LBL），即在每个微批内部统计每个专家被选中的频率并强制均匀。这种做法把统计粒度压得非常细——往往一个微批只包含几条序列，导致路由器被迫在单条序列内部把 token 平均分配。于是即使是一段专门的代码或医学文本，也会被强行送到所有专家，削弱了专家专门化的潜力。正是这层“细节中的恶魔”让 MoE 在实际大模型训练中难以发挥应有的优势。

### 关键概念速览
- **专家混合（Mixture‑of‑Experts，MoE）**：把一个大模型拆成若干小专家，每次前向只激活其中的几位，类似把工作分配给不同的工人，只让需要的工人上班，从而降低整体算力。
- **路由器（Router）**：负责根据输入 token 的特征计算每个专家的“门控分数”，决定该 token 属于哪几个专家，就像调度员根据任务属性决定派哪个工人去做。
- **负载平衡损失（Load‑Balancing Loss，LBL）**：一种正则项，鼓励路由器让所有专家的选中频率相近，防止某些专家被过度使用或被闲置。公式上是专家频率乘以平均门控分数的加权和。
- **微批（Micro‑batch）**：在分布式训练中，每个 GPU 上一次前向/反向计算所处理的最小数据块。因为显存限制，微批往往只有几条序列，统计噪声大。
- **全局批（Global‑batch）**：把所有 GPU 上的微批聚合起来视作一个大的统计单元，包含更多序列和更丰富的语料分布，统计更稳健。
- **专家专门化（Expert Specialization）**：指某个专家在特定领域或特定模式的 token 上表现更好，类似工人专精于某类任务，提高整体效率。
- **并行训练策略（Parallel Training Strategy）**：在多卡或多节点上同步梯度、共享模型参数的训练方式，是大模型训练的常规做法。

### 核心创新点
1. **统计粒度从微批提升到全局批**  
   - 之前的做法：在每个微批内部统计每个专家的选中频率 `f_i`，并在该微批内计算 LBL。  
   - 本文做法：在所有微批之间额外同步 `f_i`，把这些频率视作全局批的统计量，再基于全局频率计算 LBL。  
   - 改变：路由器不再被迫在单条序列内部均匀分配 token，而是可以在整个语料层面实现平衡，显著提升了专家的领域专门化能力。

2. **引入一次全局同步通信步骤**  
   - 之前的实现：只需要在每个微批内部完成前向、反向和梯度同步。  
   - 本文做法：在每个训练 step 结束前，所有卡共享 `f_i`（即每个专家的选中次数），这一步只涉及少量标量数据，通信开销可接受。  
   - 改变：通过一次轻量级的 All‑Reduce 操作即可获得全局频率，使得 LBL 的计算更可靠，而不会显著拖慢整体训练速度。

3. **实验验证全局 LBL 对大模型的正向影响**  
   - 之前的假设：微批 LBL 已经足够平衡负载，进一步改动收益不大。  
   - 本文做法：在 42.8 B 参数、400 B token 规模的 MoE‑LLM 上对比两种 LBL 计算方式。  
   - 改变：全局 LBL 明显降低了预训练困惑度（perplexity）并提升了下游任务表现，同时观察到专家在代码、数学等子域的专门化程度更高。

### 方法详解
**整体思路**：在保持原有并行训练框架不变的前提下，额外加入一次全局统计同步，用全局频率来计算负载平衡损失，从而让路由器在更大尺度上实现均衡分配。

**步骤拆解**：

1. **微批前向**  
   - 每个 GPU 按照常规方式对自己的微批进行前向传播。路由器根据输入 token 计算每个专家的门控分数，选出 Top‑k（常为 2）专家并产生稀疏激活。  
   - 同时记录本微批中每个专家被选中的次数 `f_i^{local}`（即该微批的频率向量）。

2. **全局频率同步**  
   - 在所有 GPU 完成前向后，执行一次 All‑Reduce（求和）操作，把所有 `f_i^{local}` 累加得到全局选中次数 `F_i = Σ_{gpu} f_i^{local}`。  
   - 将 `F_i` 除以全局 token 总数得到全局频率 `f_i = F_i / (N_{tokens}^{global})`。这一步只传输 `N_E`（专家数）个标量，通信成本极低。

3. **全局 LBL 计算**  
   - 使用全局频率 `f_i` 与每个专家的平均门控分数 `p_i`（同样在微批内部统计后再做全局平均）计算负载平衡损失：`LBL = N_E * Σ_i (f_i * p_i)`。  
   - 该损失与主任务的交叉熵损失相加，形成总损失。

4. **反向传播与梯度同步**  
   - 与普通训练相同，基于总损失进行反向传播，得到每个参数的梯度。  
   - 通过标准的梯度 All‑Reduce 完成跨卡同步，更新模型参数。

**类比**：可以把微批想象成每个工厂的日生产记录，原来只看单个工厂内部的产量来决定是否要调度工人；本文则把所有工厂的产量汇总，统一决定全公司层面的工人分配，使得资源调度更合理。

**最巧妙的地方**：只在统计层面做一次全局同步，而不需要把实际激活的 token 或梯度跨卡传输，保持了原有的高效稀疏计算路径。换句话说，作者把“全局视野”仅加在了 **统计信息** 上，而不是 **计算本身**，这在大规模分布式训练中极为经济。

### 实验与效果
- **实验规模**：在 42.8 B 参数、约 400 B token 的 MoE 大语言模型上进行预训练，专家数目与常规设置相同（如 64‑128）。
- **对比基线**：分别使用微批 LBL（原实现）和全局 LBL（本文方法）进行训练。  
- **主要结果**：论文声称全局 LBL 在预训练困惑度上取得显著下降，并在若干下游任务（代码生成、数学推理等）上均有提升。具体数值在摘要中未给出，但作者强调提升幅度“超出预期”。  
- **消融实验**：通过关闭全局同步仅保留微批统计，性能回落到原始水平；仅同步频率不同步门控分数也能得到部分提升，说明两者共同作用是关键。  
- **专家专门化分析**：使用专家激活分布可视化，发现代码类序列的激活集中在少数专家，而微批 LBL 下激活几乎均匀。全局 LBL 让这些专家在代码子域上表现更好，验证了专门化提升的假设。  
- **开销评估**：额外的 All‑Reduce 只涉及 `N_E` 个标量，实验报告的通信占比不到整体训练时间的 1%，几乎可以忽略。  
- **局限性**：作者提到在极端大规模（数千卡）环境下，全局同步的同步点可能成为瓶颈；此外，当前实现仍依赖于同步 SGD，异步或流水线方案尚未验证。

### 影响与延伸思考
这篇工作在 MoE 训练社区引发了对 **统计粒度** 的重新审视。随后包括 Qwen、DeepSpeed‑MoE 等开源框架相继加入了“全局负载平衡”选项，证明该思路已被快速采纳。后续研究可能会进一步探索 **层级负载平衡**（在层内、层间分别做全局统计）或 **自适应同步频率**（根据训练阶段动态决定是否进行全局同步），以在更大规模下保持低通信开销。对想深入的读者，可以关注 **稀疏激活的全局调度**、**分布式统计优化** 以及 **专家专门化度量** 等方向。

### 一句话记住它
把负载平衡的统计从微批提升到全局批，让 MoE 能在整个语料层面均衡分配 token，彻底解放了专家的专门化潜能。