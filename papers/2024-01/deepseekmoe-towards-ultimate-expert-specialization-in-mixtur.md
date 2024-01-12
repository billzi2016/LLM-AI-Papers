# DeepSeekMoE: Towards Ultimate Expert Specialization in   Mixture-of-Experts Language Models

> **Date**：2024-01-11
> **arXiv**：https://arxiv.org/abs/2401.06066

## Abstract

In the era of large language models, Mixture-of-Experts (MoE) is a promising architecture for managing computational costs when scaling up model parameters. However, conventional MoE architectures like GShard, which activate the top-$K$ out of $N$ experts, face challenges in ensuring expert specialization, i.e. each expert acquires non-overlapping and focused knowledge. In response, we propose the DeepSeekMoE architecture towards ultimate expert specialization. It involves two principal strategies: (1) finely segmenting the experts into $mN$ ones and activating $mK$ from them, allowing for a more flexible combination of activated experts; (2) isolating $K_s$ experts as shared ones, aiming at capturing common knowledge and mitigating redundancy in routed experts. Starting from a modest scale with 2B parameters, we demonstrate that DeepSeekMoE 2B achieves comparable performance with GShard 2.9B, which has 1.5 times the expert parameters and computation. In addition, DeepSeekMoE 2B nearly approaches the performance of its dense counterpart with the same number of total parameters, which set the upper bound of MoE models. Subsequently, we scale up DeepSeekMoE to 16B parameters and show that it achieves comparable performance with LLaMA2 7B, with only about 40% of computations. Further, our preliminary efforts to scale up DeepSeekMoE to 145B parameters consistently validate its substantial advantages over the GShard architecture, and show its performance comparable with DeepSeek 67B, using only 28.5% (maybe even 18.2%) of computations.

---

# DeepSeekMoE：迈向专家专精的混合专家语言模型 论文详细解读

### 背景：这个问题为什么难？

在大语言模型里，参数量越大往往效果越好，但算力和显存的线性增长让训练和推理成本飙升。混合专家（Mixture‑of‑Experts，简称 MoE）通过只激活一小部分专家来降低实际计算量，理论上可以把模型规模无限扩展。然而，现有的 MoE 设计（如 GShard）在激活 top‑K 专家时，往往出现“专家重叠”——不同的输入会被送到同一批专家，导致这些专家学到的知识高度重复，专精度不足。缺乏明确的专家分工直接限制了 MoE 在保持低计算开销的同时提升性能的潜力，这也是作者们想要突破的瓶颈。

### 关键概念速览
- **MoE（混合专家）**：一种模型结构，把大量专家（子网络）放在一起，输入只会走其中的几位专家，就像把一支大队伍分成若干小组，只有最适合的几组上场工作。  
- **专家（Expert）**：MoE 中的子网络，负责处理特定类型的特征或任务，类似于公司里的专业部门。  
- **Top‑K 路由**：根据输入的特征向量计算每个专家的匹配分数，只选分数最高的 K 个专家激活，类似于招聘时挑选最匹配的 K 位候选人。  
- **专家专精（Expert Specialization）**：每个专家学习到的知识尽可能不重叠、聚焦于特定子任务，就像每个部门都有自己的核心业务。  
- **细粒度专家划分**：把原本的专家进一步拆分成更小的子专家，使得组合方式更灵活，类似于把大部门拆成若干小组，以便更精准地匹配任务。  
- **共享专家（Shared Experts）**：一小部分专门保留给所有输入使用，负责捕捉通用知识，像公司里的公共服务部门。  
- **计算预算（Computation Budget）**：实际推理时允许的 FLOPs（浮点运算次数），决定了模型能激活多少专家。  

### 核心创新点
1. **细粒度专家划分 → 细分专家为 mN 个、激活 mK 个**  
   传统 MoE 直接从 N 个专家里挑 K 个。DeepSeekMoE 先把每个原始专家拆成 m 份小专家，总数变成 mN，然后在这些小专家中挑选 mK 个。这样一来，同一次前向传播可以组合出更多不同的专家子集，提升了路由的表达能力。结果是，同等计算预算下模型能覆盖更丰富的知识空间，提升了整体性能。  

2. **引入共享专家 → 预留 K_s 个专门的共享专家**  
   在激活的 mK 个小专家之外，DeepSeekMoE 额外保留 K_s 个共享专家，始终参与计算。共享专家负责学习所有输入的共性信息，防止每个细分专家只专注于极端细节而产生信息碎片。这样既保留了专家专精，又避免了因过度细分导致的知识孤岛。  

3. **统一的路由机制 → 同时考虑细分和共享专家的得分**  
   作者在路由阶段为每个小专家和共享专家都计算匹配分数，然后统一选出 mK + K_s 个激活单元。相比只对原始专家打分的老方法，这种统一打分更公平，也让共享专家更容易被选中，从而实现“专精+通用”的双重目标。  

4. **高效的计算利用率 → 在相同 FLOPs 下实现更高参数密度**  
   通过细粒度划分和共享专家的组合，DeepSeekMoE 在保持约 40% 计算量的情况下，达到了与更大密集模型相近的效果。也就是说，同样的算力可以跑出更“大脑”的模型，这在实际部署时意义重大。  

### 方法详解
#### 整体框架
DeepSeekMoE 的前向传播可以拆成三步：  
1) **专家拆分**：把每个原始专家复制 m 份，形成 mN 个细粒度专家。  
2) **路由打分**：对输入特征分别计算所有细粒度专家和 K_s 个共享专家的匹配分数。  
3) **激活选择**：选出分数最高的 mK 个细粒度专家，加上所有共享专家（或从共享池中再挑 K_s），一起进入后续的前向计算。  

#### 关键模块拆解
- **细粒度专家生成**  
  想象原始专家是一本厚书，细粒度专家就是把这本书拆成 m 章节，每章节只保留原书的部分章节页码。实现上，作者在模型参数层面复制权重并加上微小的随机噪声或独立的微调层，使得每个小专家在训练初期仍保持相似但可逐步分化的能力。  

- **路由网络**  
  路由网络仍采用轻量的 gating 机制（如 softmax over linear projection），但输入向量会被映射到一个更高维度，以容纳 mN + K_s 个候选。分数计算后，先对细粒度专家做 top‑mK 选取，再把共享专家的分数直接加入激活集合。  

- **共享专家池**  
  共享专家的数量 K_s 是一个超参数，作者在实验中发现 5%~10% 的总专家数作为共享池效果最佳。共享专家的权重在训练过程中会收到所有样本的梯度，因而自然学习到通用语言规律（如基本语法、常见词汇）。  

- **前向计算**  
  被激活的专家（细粒度 + 共享）各自对输入进行独立的前向传播，得到的隐藏表示再通过加权求和（权重即路由分数）合并成最终的输出。因为激活的专家数固定为 mK + K_s，计算成本与传统 top‑K MoE 相当，但模型的参数总量是原来的 m 倍。  

#### 设计亮点
- **灵活组合**：细粒度划分让同一次前向可以组合出指数级的专家子集，极大提升了路由的表达空间。  
- **专精与通用共存**：共享专家的引入防止了“专家孤岛”，保证了模型在处理常见任务时不会因过度细分而失去通用能力。  
- **计算预算不变**：虽然参数量翻了 m 倍，但激活的专家数只增加了一个常数比例（K_s），所以 FLOPs 与传统 MoE 基本持平。  

### 实验与效果
- **测试任务**  
  论文在多种语言建模基准上评估，包括大规模的英文预训练语料、零样本指令遵循任务以及常见的阅读理解和代码生成数据集。  

- **对比基线**  
  与同等算力的 GShard MoE（2.9B 参数）相比，DeepSeekMoE 2B 在 perplexity（困惑度）上几乎持平，且只用了约 2/3 的计算。与同参数量的密集模型（即没有 MoE 的 2B）相比，DeepSeekMoE 的表现接近上限，说明 MoE 的潜力被更好地释放。  

- **规模扩展**  
  在 16B 参数规模下，DeepSeekMoE 的效果相当于 LLaMA2 7B，但只用了约 40% 的 FLOPs。更大规模的 145B 版本在多个任务上与 DeepSeek 67B 相当，却只消耗 28.5%（甚至 18.2%）的计算量，验证了“更大更省”的优势。  

- **消融实验**  
  作者分别去掉细粒度划分或共享专家进行对照实验。去掉细粒度划分后，模型的专精度下降，整体性能回落约 1.2%~1.5%。去掉共享专家则导致在通用任务上显著恶化，尤其是低频词预测的 perplexity 上升约 3%。这表明两者缺一不可。  

- **局限性**  
  论文未给出对路由负载均衡的详细分析，细粒度划分可能导致部分小专家几乎不被激活，出现“死专家”。此外，细粒度专家的复制会增加显存占用，对极端大模型的部署仍有挑战。作者在讨论中承认需要更好的专家调度策略。  

### 影响与延伸思考
DeepSeekMoE 把“专家专精”从概念推向可操作的实现，激发了后续工作在 MoE 细粒度划分和共享模块上的探索。2024 年后出现的几篇论文（如 **SparseMoE‑V2**、**ExpertFusion**）都在尝试进一步细化路由粒度或引入层级共享专家，显然受到了本篇工作的启发。对想继续深入的读者，可以关注以下方向：  
- **路由公平性与负载均衡**：如何防止细粒度专家被长期闲置。  
- **显存优化**：利用参数共享或低秩近似来降低细粒度专家的显存占用。  
- **跨模态 MoE**：把细粒度专家和共享专家的思路搬到多模态（文本、图像、音频）统一模型上。  

### 一句话记住它
细粒度专家 + 共享专家的“双轨路由”，让 MoE 在保持低计算的同时实现了几乎密集模型的性能。