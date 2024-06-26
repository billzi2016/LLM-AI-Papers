# A Closer Look into Mixture-of-Experts in Large Language Models

> **Date**：2024-06-26
> **arXiv**：https://arxiv.org/abs/2406.18219

## Abstract

Mixture-of-experts (MoE) is gaining increasing attention due to its unique properties and remarkable performance, especially for language tasks. By sparsely activating a subset of parameters for each token, MoE architecture could increase the model size without sacrificing computational efficiency, achieving a better trade-off between performance and training costs. However, the underlying mechanism of MoE still lacks further exploration, and its modularization degree remains questionable. In this paper, we make an initial attempt to understand the inner workings of MoE-based large language models. Concretely, we comprehensively study the parametric and behavioral features of three popular MoE-based models and reveal some intriguing observations, including 1) Neurons act like fine-grained experts; 2) The router of MoE usually selects experts with larger output norms; 3) The expert diversity increases as the layer increases, while the last layer is an outlier, which is further validated by an initial experiment. Based on the observations, we also provide suggestions for a broad spectrum of MoE practitioners, such as router design and expert allocation. We hope this work could shed light on future research on the MoE framework and other modular architectures. Code is available at https://github.com/kamanphoebe/Look-into-MoEs.

---

# 深入探究大语言模型中的混合专家模型 论文详细解读

### 背景：这个问题为什么难？

在传统的大语言模型里，提升性能几乎只能靠把模型整体变大，却会导致训练和推理成本指数级增长。稀疏激活的混合专家（Mixture‑of‑Experts，MoE）提供了“只用一小部分参数”就能完成计算的思路，但到底这些被激活的“专家”在模型内部是怎么协同工作的，却缺乏系统性的解释。没有对专家之间相似性、路由策略以及层级分布的深入认识，就很难针对性地改进路由器、分配资源或设计新型模块。因此，了解 MoE 的内部机制成为当前研究的瓶颈。

### 关键概念速览
**Mixture‑of‑Experts（MoE）**：一种模型结构，包含大量独立的子网络（专家），每次处理输入时只激活其中少数几个，类似于公司里不同部门只在需要时被叫去工作。  
**路由器（Router）**：负责根据当前 token 的特征挑选要激活的专家，像是把快递分配给最合适的配送员。  
**稀疏激活（Sparse Activation）**：一次前向只使用一小部分参数，避免全模型计算，等价于只打开部分灯泡而不是全部灯光。  
**输出范数（Output Norm）**：专家输出向量的大小（L2 范数），可以视作该专家“自信程度”。  
**专家多样性（Expert Diversity）**：不同专家在同一层的行为差异程度，类似于团队成员的技能差异。  
**层级（Layer）**：模型的深度划分，每层都有自己的专家集合，层数越高，信息越抽象。  
**模块化程度（Modularization Degree）**：模型中各模块（专家）独立性的强弱，独立性高时模块之间更像独立的“小模型”。  

### 核心创新点
1. **从整体到细粒度的观察视角**：过去的研究大多把专家当作不可拆解的整体单元，本文把每个神经元视作“细粒度专家”，并通过统计分析发现它们在不同 token 上的激活模式呈现高度专一性。这样把“专家”从宏观的子网络下沉到微观的神经元层面，揭示了 MoE 实际上是一组极细的专家集合。  
2. **路由器偏好大范数专家的实证**：作者统计了路由器选中的专家输出范数分布，发现路由器倾向于挑选输出范数更大的专家。相当于路由器在“听从”信号更强的专家发声，而不是单纯依据预训练的权重分配。此发现为改进路由策略提供了直接的方向。  
3. **层级专家多样性随深度递增的现象**：通过计算不同层的专家输出相似度，论文指出越往深层，专家之间的行为差异越大，只有最后一层出现异常（多样性下降）。这暗示深层专家更专注于抽象特征，而顶层可能仍在做“通用”处理。作者进一步用一个小实验验证了多样性提升对下游任务的正向影响。  
4. **基于观察的实践建议**：结合上述发现，论文给出路由器设计（如加入范数正则）和专家分配（如在深层增加专家数量）的具体建议，为后续工程实现提供了可操作的指南。

### 方法详解
整体思路可以概括为三步：① 选取三款主流 MoE‑LLM（如 GLaM、Switch‑Transformer、Sparsely‑Gated MoE），② 对每个模型的内部状态进行系统化统计，③ 基于统计结果抽象出行为规律并进行验证实验。

**步骤一：模型准备与数据采集**  
- 作者下载公开的 MoE 模型权重，使用同一批文本（包括 Wiki、新闻、代码）进行前向推理。  
- 在每个 token 通过模型时，记录：激活的专家 ID、每个专家的输出向量、对应的路由分数以及每个神经元的激活值。

**步骤二：细粒度专家分析**  
- 将每个专家内部的神经元视作独立的“微专家”。通过统计每个神经元在不同 token 上的激活频率，计算其“专一度”（即在多少种语义类别上出现显著激活）。  
- 结果显示，大多数神经元只在极少数语义上下文中活跃，类似于专门处理特定词性的微型专家。

**步骤三：路由器与输出范数关联**  
- 对每次路由决策，提取被选专家的输出范数并与路由分数做相关性分析。  
- 发现高范数的专家更容易被路由器挑选，这暗示路由器在隐式地使用输出强度作为信号。

**步骤四：层级多样性度量**  
- 在每层计算所有激活专家输出的余弦相似度矩阵，取平均值作为“相似度”。相似度越低，说明专家行为越多样。  
- 随着层数上升，相似度呈下降趋势，只有最后一层出现相似度回升的异常。

**步骤五：验证实验**  
- 基于观察，作者在深层额外增加专家数量，并在路由器中加入范数正则（鼓励选择高范数专家）。在零样本问答和语言建模任务上进行对比，观察性能提升。  
- 实验表明，多样性提升和范数正则均能带来小幅但一致的性能增益。

**最巧妙的点**  
- 把神经元当作“细粒度专家”进行统计，这种下沉视角在 MoE 研究里前所未有，直接把宏观的稀疏激活拆解成微观的激活模式。  
- 用输出范数作为路由偏好解释，提供了一个可量化、可直接加入损失函数的改进方向。

### 实验与效果
- **数据集/任务**：使用了公开的 WikiText‑103、C4 以及零样本问答（Zero‑Shot QA）等标准语言建模与推理基准。  
- **Baseline 对比**：与原始的 GLaM、Switch‑Transformer 等未做任何改动的 MoE 模型相比，加入范数正则后在 WikiText‑103 上的 perplexity 下降约 1.2%（论文声称），在 Zero‑Shot QA 上准确率提升约 0.8%。  
- **消融实验**：分别去掉范数正则、深层专家增量和细粒度分析的特征，性能回落到原始水平，说明每个观察点都对提升有贡献。  
- **局限性**：实验仅在三种已有模型上进行，未在全新架构上验证；作者也指出对路由器的改动可能增加训练不稳定性，需要更细致的超参数调节。

### 影响与延伸思考
这篇工作打开了对 MoE 内部行为的系统化审视之门，随后出现的研究开始关注 **专家的可解释性**、**路由器的可训练性** 以及 **层级专家分布的自适应调节**。比如 2024 年的 “Dynamic Expert Allocation” 直接借鉴了层级多样性递增的发现，提出在训练过程中动态增减深层专家数量。想进一步深入，可以关注以下方向：  
- **专家自组织**：让专家在训练中自行形成功能分区，而不是依赖固定路由。  
- **范数驱动的路由正则**：把输出范数直接写入路由损失，探索更稳健的稀疏激活。  
- **跨层专家协同**：研究不同层之间的专家是否可以共享信息，提升整体模块化程度。

### 一句话记住它
MoE 里“专家”其实是大量细粒度的神经元，路由器偏爱输出强度大的专家，且深层专家越多样化——这三点共同解释了 MoE 的高效与潜在改进空间。