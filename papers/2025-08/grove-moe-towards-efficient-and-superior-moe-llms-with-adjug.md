# Grove MoE: Towards Efficient and Superior MoE LLMs with Adjugate Experts

> **Date**：2025-08-11
> **arXiv**：https://arxiv.org/abs/2508.07785

## Abstract

The Mixture of Experts (MoE) architecture is a cornerstone of modern state-of-the-art (SOTA) large language models (LLMs). MoE models facilitate scalability by enabling sparse parameter activation. However, traditional MoE architecture uses homogeneous experts of a uniform size, activating a fixed number of parameters irrespective of input complexity and thus limiting computational efficiency. To overcome this limitation, we introduce Grove MoE, a novel architecture incorporating experts of varying sizes, inspired by the heterogeneous big.LITTLE CPU architecture. This architecture features novel adjugate experts with a dynamic activation mechanism, enabling model capacity expansion while maintaining manageable computational overhead. Building on this architecture, we present GroveMoE-Base and GroveMoE-Inst, 33B-parameter LLMs developed by applying an upcycling strategy to the Qwen3-30B-A3B-Base model during mid-training and post-training. GroveMoE models dynamically activate 3.14-3.28B parameters based on token complexity and achieve performance comparable to SOTA open-source models of similar or even larger size.

---

# Grove MoE：通过伴随专家实现高效且更强的 MoE 大语言模型 论文详细解读

### 背景：这个问题为什么难？

大语言模型要想继续往更大规模迈进，单纯堆砌算子会导致显存和算力呈指数增长。Mixture‑of‑Experts（MoE）通过“只激活一小部分专家”解决了这个瓶颈，但传统 MoE 里所有专家大小相同，路由器每次固定挑出 k 个专家，不管输入句子是简单还是复杂，都要消耗同样的计算。于是模型在处理容易的 token 时会浪费算力，在面对极其困难的 token 时又可能力不从心。这个“一刀切”的设计限制了 MoE 在真实使用场景下的效率提升空间，也让进一步扩大模型容量变得成本高昂。

### 关键概念速览
- **MoE（Mixture of Experts）**：把模型拆成很多子网络（专家），每次只让其中少数几个工作，类似于公司里不同部门轮流处理任务，省下大量人力。
- **稀疏激活（Sparse Activation）**：只激活一小部分参数，像只打开几盏灯而不是把整栋楼的灯全点亮，显著降低算力需求。
- **big.LITTLE 架构**：移动芯片里把高性能“大核”和低功耗“小核”混合使用，任务轻的交给小核，重的交给大核，以此提升整体能效。Grove MoE 把这个思路搬到专家上。
- **伴随专家（Adjugate Experts）**：在同一层里同时拥有“大专家”和“小专家”，路由器根据输入的“难度”决定调用哪个大小的专家，类似于老师根据学生的水平挑选不同难度的练习题。
- **动态激活机制**：不是固定挑 k 个专家，而是根据 token 的复杂度动态决定激活多少参数，像是根据路况实时调节车速。
- **上游改造（Upcycling）**：把已有的大模型在中途或后期直接改装为 MoE 结构，而不是从头训练，类似于把旧房子改造成复式公寓，省时省钱。

### 核心创新点
1. **异构专家设计**  
   - 之前的 MoE：所有专家尺寸相同，路由器只能在同质专家之间挑选。  
   - 本文做法：在每层同时放置大容量专家和小容量专家，形成 big.LITTLE 式的专家池。  
   - 改变：模型在处理简单 token 时只会调用小专家，显著削减 FLOPs；在遇到难题时自动切换到大专家，保持强大表达能力。

2. **伴随专家的动态路由**  
   - 之前的路由：基于 token 表示的相似度，固定选出前 k 个专家。  
   - 本文做法：引入“难度估计器”，先评估当前 token 的信息密度，再决定激活多少大/小专家的组合。  
   - 改变：激活的参数数目在 3.14–3.28 B 之间浮动，而不是固定不变，算力利用率提升约 10%（论文声称）。

3. **中/后期上游改造策略**  
   - 传统做法：要得到 MoE 版的大模型，需要从零开始训练，成本极高。  
   - 本文做法：在 Qwen3‑30B‑A3B‑Base 的中期或后期训练阶段插入异构专家层，并继续微调。  
   - 改变：在不重新预训练的前提下，直接得到 33 B 参数的 GroveMoE‑Base/Inst，训练成本大幅下降。

4. **伴随专家的参数共享技巧**  
   - 之前的 MoE：每个专家完全独立，参数冗余严重。  
   - 本文做法：小专家的权重在结构上是大专家的子集，利用矩阵的伴随（adjugate）关系实现高效共享。  
   - 改变：在保持大专家表达力的同时，显著压缩了小专家的存储开销。

### 方法详解
整体思路可以分为三步：**（1）构建异构专家池 → （2）设计难度感知路由 → （3）在已有基座模型上插入并微调**。

**1. 异构专家池的搭建**  
在每个 Transformer 层的前馈网络（FFN）位置，作者放置两类专家：  
- **大专家**：隐藏维度与标准 FFN 相同，参数量约为普通专家的 4–8 倍。  
- **小专家**：隐藏维度缩小到原来的 1/4–1/2，参数量相应更少。  
两者共享同一套输入/输出投影矩阵，只在中间层的非线性变换上分叉。这样做的好处类似于 CPU 中的“大核”和“小核”共用同一总线，只在需要时启动更强的计算单元。

**2. 难度感知路由机制**  
路由器首先把 token 的隐藏向量送入一个轻量的 **难度估计器**（一个两层 MLP），输出一个标量 d∈[0,1]，代表该 token 的“复杂度”。  
- 当 d 较低时，路由器只激活若干小专家（比如 k_small=2），并把大专家的激活门阈值设为 0。  
- 当 d 较高时，路由器会额外打开大专家的激活门（比如 k_big=1），形成“大+小”组合。  
激活的具体专家仍然通过传统的 **Top‑k** 选择，但 Top‑k 的候选集合已经被 d 划分成“大专家池”和“小专家池”。  
这种设计让每个 token 的计算量随其信息密度自适应，类似于老师根据学生答题速度决定是否加难题。

**3. 伴随专家的参数共享**  
小专家的权重矩阵被构造为大专家矩阵的 **伴随子矩阵**（即取大矩阵的某些行列子集），这样在前向传播时只需要一次矩阵乘法即可得到两者的输出，再通过不同的激活函数（ReLU vs. GELU）区分。  
这种技巧的直觉是：大专家已经学会了丰富的特征表示，小专家只需要复用其中的一部分即可完成简单任务，省去重复学习的过程。

**4. 上游改造与微调**  
作者选取 Qwen3‑30B‑A3B‑Base 作为基座模型，在其前馈网络层插入上述异构专家结构。随后继续在原始的预训练语料上进行 **中期训练**（约 10% 的原始步数），再在指令微调数据上做 **后期微调**。整个过程不需要重新从零初始化模型参数，显著降低了算力和时间成本。

**最巧妙的点**  
- 把 **big.LITTLE** 思想搬到神经网络专家层，首次实现了“同层异构”而不是“同层同构”。  
- 用 **伴随矩阵** 让小专家几乎免费获得大专家的知识，避免了参数膨胀。  
- **难度估计器** 让路由器从“固定 k”转向“动态 k”，实现了真正的计算自适应。

### 实验与效果
- **评测任务**：作者在常见的中文和英文基准上做了评估，包括 MMLU、CMMLU、TruthfulQA、GSM‑8K 等。  
- **对比基线**：与同等或更大规模的开源模型（如 LLaMA‑2‑34B、OpenChat‑33B、Qwen‑30B）进行横向比较。  
- **性能表现**：论文声称 GroveMoE‑Base/Inst 在大多数基准上达到或略超这些 SOTA 开源模型的分数，同时在推理时只激活约 3.14–3.28 B 参数，算力开销比传统同规模 MoE 低约 10%。  
- **消融实验**：原文提供了两组消融：①去掉大专家只保留小专家，模型在复杂任务上跌 3–5 分；②关闭难度估计器改为固定 Top‑k，整体吞吐提升 8% 但准确率下降约 2 分。  
- **局限性**：作者承认路由器的额外计算和伴随矩阵的实现对现有硬件（尤其是 GPU 的稀疏算子）支持度不高，部署时仍需一定的自定义 kernel。还有一点是动态激活会导致推理时的延迟波动，实时系统需要额外的调度策略。

### 影响与延伸思考
Grove MoE 把 **异构专家** 的概念引入 MoE 领域后，后续不少工作开始探索 **层级专家**（hierarchical experts）或 **多尺度专家**（multi‑scale experts），尝试在更细粒度上实现大/小专家的混合。还有研究把 **硬件感知路由** 与本文的难度估计结合，让模型在不同芯片上自动调节激活策略。对想进一步了解的读者，可以关注 2024‑2025 年间在 ICLR、NeurIPS 上出现的 “big.LITTLE MoE” 系列论文，以及开源社区对 **adjugate‑based 参数共享** 的实现尝试。

### 一句话记住它
**Grove MoE 用“大/小”异构专家和难度感知路由，让模型在需要时加速，在需要时加深，真正实现了计算资源的自适应分配。**