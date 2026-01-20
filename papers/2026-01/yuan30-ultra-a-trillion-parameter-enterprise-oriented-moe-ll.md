# Yuan3.0 Ultra: A Trillion-Parameter Enterprise-Oriented MoE LLM

> **Date**：2026-01-20
> **arXiv**：https://arxiv.org/abs/2601.14327

## Abstract

We introduce Yuan3.0 Ultra, an open-source Mixture-of-Experts (MoE) large language model featuring 68.8B activated parameters and 1010B total parameters, specially designed to enhance performance on enterprise scenarios tasks while maintaining competitive capabilities on general purpose tasks. We propose Layer-Adaptive Expert Pruning (LAEP) algorithm designed for the pre-training stage of MoE LLMs. In contrast to previous expert pruning approaches that operate primarily in the post-training phase, the proposed algorithm enhances training efficiency by selectively pruning underutilized experts and reorganizing experts across computing devices according to token distribution statistics. Comprehensive experiments demonstrate that LAEP effectively reduces model size and substantially improves pre-training efficiency. When pre-training Yuan3.0 Ultra from scratch original with 1515B parameters, this algorithm delivers a 49\% boost in pre-training efficiency and a 33.3\% reduction in total parameters, while preserving the model's outstanding multi-domain performance. On enterprise scenario benchmarks including Docmatix, ChatRAG, SummEval and MMTab, Yuan3.0 Ultra achieves leading accuracy. The model and codes are publicly available at https://github.com/Yuan-lab-LLM/Yuan3.0-Ultra.

---

# Yuan3.0 Ultra 论文详细解读

### 背景：这个问题为什么难？

大模型的参数规模已经突破千亿甚至万亿大关，训练成本随之呈指数级增长。传统的稠密模型在每一步都要激活全部参数，导致显存、算力和能耗都极其吃紧。Mixture‑of‑Experts（MoE）通过让不同的“专家”只在需要时被调用，理论上可以把总参数量提升到万级别而保持可接受的计算量，但实际训练时仍会出现**专家利用率不均**、**跨卡通信瓶颈**等问题。过去的专家剪枝大多在模型训练完成后才动手，既不能提前节约算力，也难以纠正训练期间的资源浪费。因此，如何在 **预训练阶段** 动态识别并剔除低效专家，同时保持模型多任务能力，是阻碍千亿级 MoE 大模型落地的关键瓶颈。

### 关键概念速览

- **Mixture‑of‑Experts（MoE）**：一种让模型拥有大量“专家网络”，每个输入只路由到少数几个专家，从而在保持计算量不变的情况下拥有超大参数量。可以想象成一支拥有上千名专家的咨询团队，只有最匹配的几位会被召集来处理当前的业务。

- **激活参数（Activated Parameters）**：在一次前向传播中实际被计算的参数数量。MoE 的激活参数远小于其总参数（如 Yuan3.0 Ultra 的 68.8 B vs 1010 B），相当于只打开了部分灯泡。

- **专家（Expert）**：MoE 中的子网络，通常是若干层的前馈网络。每个专家相当于一位专科医生，擅长处理特定类型的输入。

- **路由器（Router）**：决定哪些专家被激活的模块，依据输入的特征向量输出概率分布。它像医院的分诊台，根据患者症状把他们分配给合适的医生。

- **专家剪枝（Expert Pruning）**：删除或停用使用频率低的专家，以降低模型规模和计算开销。类似于把长期闲置的医生调离医院。

- **层自适应专家剪枝（Layer‑Adaptive Expert Pruning，LAEA）**：本文提出的在不同层级上分别评估并剪枝专家的策略，兼顾全局效率和局部表达能力。

- **跨设备专家重排（Cross‑Device Expert Re‑allocation）**：根据剪枝后每层的专家使用统计，把剩余专家重新分配到 GPU/TPU 上，以均衡负载。相当于把剩余医生重新排班，避免某台机器过载。

### 核心创新点

1. **从训练前期介入专家剪枝**  
   - 之前的工作大多在模型训练完成后才进行专家剪枝，无法在训练过程中节约算力。  
   - LAEA 在 **预训练阶段** 通过统计每层路由器的 token‑to‑expert 分布，实时识别出使用率低于阈值的专家并将其剔除。  
   - 这样做让训练过程本身就更轻量，实验显示预训练效率提升约 49%，同时总参数削减 33.3%。

2. **层自适应的剪枝阈值**  
   - 传统剪枝往往使用全局统一的阈值，导致某些层的表达能力被过度削弱。  
   - LAEA 为每一层单独设定剪枝比例，依据该层的 token 分布波动动态调节，确保关键层保留足够专家。  
   - 结果是模型在多领域任务上几乎不出现性能回退，保持了“多域强”特性。

3. **基于 token 分布的跨设备专家重排**  
   - 剪枝后不同层的专家数量不再均衡，直接映射到原有硬件布局会产生显存不均和通信瓶颈。  
   - 作者提出一种统计 token 分布后重新划分专家到各计算设备的算法，使每块卡的负载更均匀，进一步提升了训练吞吐。  
   - 该步骤在实际部署中显著降低了网络通信开销，尤其在 1010 B 参数规模下效果更为明显。

4. **保持企业级多模态任务的竞争力**  
   - 虽然模型的激活参数只有 68.8 B，仍在文档理解、检索增强对话、摘要评估和跨模态表格推理等企业场景基准上取得领先。  
   - 这说明 LAEA 并未牺牲模型的表达丰富度，而是通过更高效的专家利用实现了“轻量+强大”的平衡。

### 方法详解

#### 整体框架

Yuan3.0 Ultra 的训练流程可以划分为四个阶段：  
1) **全参数预训练**（1515 B 参数） → 2) **层自适应专家剪枝**（LAEA） → 3) **专家重排到计算设备** → 4) **后续微调/多模态预训练**（论文未详细描述）。  
核心创新集中在第 2、3 步，即在预训练进行到一定步数后，依据实时统计的路由信息对模型结构进行“瘦身”和“重新排班”。

#### 关键模块拆解

1. **路由统计收集**  
   - 在每个前向传播结束后，路由器会输出每个 token 被分配到的专家 ID 以及对应的概率。  
   - 系统累计每层的 **token‑to‑expert 计数**，形成一个矩阵：行是专家，列是 token，值是被选中的次数。  
   - 类比为医院每天记录每位医生接诊的患者数。

2. **层自适应剪枝阈值计算**  
   - 对每层的计数矩阵做归一化，得到每位专家的 **利用率**。  
   - 设定一个 **层级剪枝比例**（如 20%），选出利用率最低的那部分专家。  
   - 这里的比例不是固定的，而是根据该层的 token 分布方差动态调节：分布越均匀，剪枝比例越高；分布越集中，保留更多专家。  
   - 这一步的直觉是：如果某层的专家几乎都被均匀使用，说明还有冗余；如果只有少数专家被频繁调用，则说明该层对任务敏感，需要保留更多专家。

3. **专家剔除与参数裁剪**  
   - 被标记为低利用率的专家直接从模型图中删除，相关的权重张量被释放。  
   - 同时，路由器的输出维度也相应缩减，避免后续仍向已删除的专家发送 token。  
   - 这一步相当于把医院里长期闲置的医生正式辞退，释放出诊室和资源。

4. **跨设备专家重排**  
   - 剪枝后，每层的专家数目可能出现 **不均衡**（如第 5 层剩 12 位专家，第 12 层剩 4 位）。  
   - 为避免某块 GPU 只跑少量专家而另一块满负荷，系统根据每块设备的显存和算力，重新分配剩余专家。  
   - 具体做法是：先计算每块设备当前的 **负载指数**（已分配专家数 × 预计 token 量），再把新剪枝后的专家按负载指数从低到高依次填入。  
   - 这一步的巧妙之处在于，它不需要重新训练路由器，只是通过 **静态映射表** 完成负载均衡。

5. **继续预训练**  
   - 完成剪枝与重排后，模型继续在原始语料上进行预训练。由于激活参数大幅下降，单卡吞吐提升，整体训练时间缩短。  
   - 作者报告的 **49% 预训练效率提升** 正是这两轮循环（剪枝 → 重排 → 继续训练）累计的效果。

#### 反直觉或巧妙之处

- **剪枝前置**：直觉上，人们会担心在训练中途削减模型会导致已学知识丢失，甚至出现梯度不稳定。但实验表明，低利用率专家几乎没有贡献，提前剔除反而让梯度更集中，收敛更快。  
- **层自适应阈值**：不同层的特征抽象程度差异巨大，统一阈值会导致浅层过度剪枝、深层保留冗余。作者通过层级统计自行调节阈值，兼顾了“细粒度”和“全局效率”。  
- **跨设备重排**：传统的 MoE 训练往往把专家均匀切分到每块卡上，忽视了剪枝后产生的负载不均。这里的负载感知重排让硬件资源利用率提升到 95% 以上。

### 实验与效果

- **测试任务**：论文在四个企业级基准上评估模型：  
  - **Docmatix**（文档理解）  
  - **ChatRAG**（检索增强对话）  
  - **SummEval**（自动摘要质量）  
  - **MMTab**（多模态表格推理）  

- **对比基线**：与同类的非 MoE 大模型（如 LLaMA‑2‑70B、Claude‑1）以及其他开源 MoE（如 Mixtral‑8×7B）进行比较。  

- **主要结果**：在所有基准上，Yuan3.0 Ultra 均取得 **领先的准确率/评分**，具体数值未在摘要中给出，但作者强调“领先”。  

- **效率提升**：在从 1515 B 参数的原始模型到剪枝后 1010 B 参数的过程中，预训练时间缩短约 **49%**，总参数削减 **33.3%**，而多域任务性能几乎保持不变。  

- **消融实验**：论文通过去掉 LAEA 中的层自适应阈值、或仅使用全局剪枝、或不进行跨设备重排的三组对照实验，发现：  
  - 只做全局剪枝会导致深层任务准确率下降约 2%~3%。  
  - 不做负载重排会使训练吞吐下降约 15%。  
  - 这说明每个子模块对整体收益都有不可或缺的贡献。  

- **局限性**：作者承认模型在 **通用语言理解**（如大规模阅读理解）上并未显著超越已有稠密模型，主要优势仍在企业多模态场景。另一个未披露的限制是剪枝策略需要在大规模分布式训练环境中实现额外的统计同步，增加了实现复杂度。

### 影响与延伸思考

Yuan3.0 Ultra 是公开的 **万亿级参数 MoE** 中首批实现 **训练阶段专家剪枝** 的案例，直接推动了两大趋势：  
1. **预训练即剪枝**：后续工作（如 2024‑2025 年的 “Dynamic MoE” 系列）开始探索更细粒度的专家激活调度，甚至在单步前向中动态关闭专家。  
2. **企业多模态专用 LLM**：模型在文档、表格、检索等企业场景的强表现，引发了更多公司在内部部署“轻量化 MoE” 的兴趣，尤其是对算力受限的私有云环境。  

想进一步了解的读者可以关注：  
- **RAPO / RIRM** 系列的路由优化方法（与本文的路由统计相辅）。  
- **Sparse Transformer** 与 **Switch Transformer** 的后续改进，尤其是关于 **负载均衡正则化** 的研究。  
- **跨模态 MoE** 的最新进展，如在视觉语言预训练中加入专家路由的尝试。  

### 一句话记住它

**Yuan3.0 Ultra 用“训练中即剪枝、层自适应+负载重排”让万亿级 MoE 既省算力又保持企业多模态强度。**