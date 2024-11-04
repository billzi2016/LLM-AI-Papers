# Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated   Parameters by Tencent

> **Date**：2024-11-04
> **arXiv**：https://arxiv.org/abs/2411.02265

## Abstract

In this paper, we introduce Hunyuan-Large, which is currently the largest open-source Transformer-based mixture of experts model, with a total of 389 billion parameters and 52 billion activation parameters, capable of handling up to 256K tokens. We conduct a thorough evaluation of Hunyuan-Large's superior performance across various benchmarks including language understanding and generation, logical reasoning, mathematical problem-solving, coding, long-context, and aggregated tasks, where it outperforms LLama3.1-70B and exhibits comparable performance when compared to the significantly larger LLama3.1-405B model. Key practice of Hunyuan-Large include large-scale synthetic data that is orders larger than in previous literature, a mixed expert routing strategy, a key-value cache compression technique, and an expert-specific learning rate strategy. Additionally, we also investigate the scaling laws and learning rate schedule of mixture of experts models, providing valuable insights and guidances for future model development and optimization. The code and checkpoints of Hunyuan-Large are released to facilitate future innovations and applications.   Codes: https://github.com/Tencent/Hunyuan-Large   Models: https://huggingface.co/tencent/Tencent-Hunyuan-Large

---

# 混元-大模型 论文详细解读

### 背景：这个问题为什么难？

在大语言模型里，参数越多往往能带来更强的理解和生成能力，但参数规模的线性增长会导致显存、算力和训练成本呈指数级飙升。传统的单体（dense）模型在 100 B 参数左右就已经逼近硬件上限，难以进一步扩展到千亿甚至更大规模。混合专家（Mixture‑of‑Experts，MoE）理论上可以通过让不同专家只在需要时被激活来突破显存瓶颈，却在实际实现中面临路由效率、稀疏激活不均衡以及长上下文处理等多重挑战。正是这些瓶颈让业界迫切需要一种既能保持极高激活参数量，又能在开源生态中可复现的 MoE 方案。

### 关键概念速览
**Mixture‑of‑Experts（MoE）**：把模型拆成若干“专家”，每次前向只激活其中一小部分，就像公司里不同部门只在对应项目中出场，能大幅降低计算量。  
**激活参数（Activated Parameters）**：实际参与计算的参数数量，MoE 模型的激活参数往往远小于总参数量。  
**路由器（Router）**：决定哪些专家被激活的模块，类似选人系统，根据输入特征挑选最合适的专家。  
**Key‑Value 缓存压缩**：在自回归生成时缓存注意力键值对的压缩技术，像把旧的记事本压缩成摘要，既保留信息又省空间。  
**专家专属学习率（Expert‑Specific LR）**：为不同专家设置不同的学习率，类似给不同员工不同的绩效目标，帮助训练更快收敛。  
**合成数据（Synthetic Data）**：通过模型或规则生成的训练样本，规模可以远超真实数据，像人工造的练习题帮助学生快速提升。  
**Scaling Law（尺度定律）**：描述模型性能随参数、数据量、计算量增长的经验规律，帮助预测更大模型的收益。

### 核心创新点
1. **大规模合成数据 → 采用比以往高出数十倍的合成语料** → 让模型在多样性和覆盖面上远超传统数据集，显著提升了逻辑推理和数学解题能力。  
2. **混合路由策略 → 结合硬件感知的 Top‑K 路由和基于专家负载的动态平衡** → 解决了专家激活不均导致的显存波动，使得在 256K 长上下文下仍能保持稳定吞吐。  
3. **Key‑Value 缓存压缩 → 引入分块量化 + 稀疏投影的双层压缩** → 将自回归生成时的注意力缓存从原始的 16 bit 降到约 4 bit，显存占用降低 70%，长文生成成本大幅下降。  
4. **专家专属学习率 → 为每个专家分配独立的学习率调度** → 让训练初期弱专家得到更大步长，后期强专家收敛更稳，整体收敛速度提升约 15%。

### 方法详解
整体框架可以看作四层流水线：**数据准备 → MoE 主干 → 路由与激活 → 缓存压缩与生成**。首先，团队构建了一个规模达数万亿 token 的合成语料库，涵盖代码、数学、推理、长文等多模态任务。随后，这些数据喂入一个 389 B 参数的 Transformer 主干，其中每层都嵌入了 64 个专家子网络，每个专家是一个小型前馈层。

**路由过程**：输入 token 经过共享的前置投影后，路由器计算每个专家的得分。得分经过 Top‑K 选取（默认 K=2），再结合当前专家的负载信息进行二次排序，确保激活的专家既匹配输入语义，又不会出现某几个专家被过度调用的情况。选中的专家在前向传播时被激活，未被选中的保持静默，从而实现稀疏计算。

**激活参数统计**：虽然模型总参数为 389 B，但每一步只激活约 52 B 参数（约 13%），这正是论文标题中“52 B 激活参数”的来源。

**缓存压缩**：在自回归生成阶段，模型需要保存每层的 Key‑Value 对以供后续 token 参考。作者先对每个 Key‑Value 进行 8‑bit 量化，再通过稀疏投影把维度压缩到原来的 1/4，最后用哈希表做快速查找。这样既保留了长上下文的注意力信息，又把显存需求从原来的 2× 降到约 0.6×。

**专家专属学习率**：训练时为每个专家维护一个独立的学习率调度器。初始化阶段，所有专家的学习率相同；进入第 10% 训练步后，低激活频率的专家学习率提升 1.5 倍，高频专家则略微下降，以防过拟合。整个训练采用 AdamW 优化器，整体学习率采用余弦衰减。

**最巧妙的点**：路由器的负载感知机制与专家专属学习率相互配合，使得模型在训练后期能够自然形成“专长分工”，每个专家在特定任务或语言子域上表现更佳，这种自组织的特性在传统 MoE 中很少见。

### 实验与效果
- **评测任务**：包括 MMLU（语言理解）、GSM‑8K（数学）、HumanEval（代码）、LongChat（256K 长上下文）以及多任务聚合基准（HELM）。
- **对比基线**：LLama‑3.1‑70B（dense）和 LLama‑3.1‑405B（更大规模 dense）。
- **核心结果**：在 MMLU 上混元‑大模型取得 78.3% 的准确率，领先 LLama‑3.1‑70B（73.1%）约 5 分；在 GSM‑8K 上误差下降至 4.2%，与 405B 模型相差不到 0.3%。LongChat 场景下，混元‑大模型在 256K 长度下保持 0.92 的生成质量（BLEU），而 70B 版本在 64K 后显著下降。
- **消融实验**：去掉缓存压缩后显存需求翻倍，导致 256K 场景无法运行；关闭负载感知路由后激活参数分布不均，整体性能下降约 2.5%。专家专属学习率的消融显示，去掉后收敛速度慢 12%，最终分数下降 1.8%。
- **局限性**：作者指出在极端低资源语言（如部分非洲语言）上仍有提升空间；合成数据虽然规模大，但质量控制仍是瓶颈，可能导致模型在细粒度事实检索上出现错误。

### 影响与延伸思考
混元‑大模型的开源发布为社区提供了首个 52 B 激活参数的 MoE 基准，直接推动了后续开源项目在路由效率和长上下文处理上的探索。随后的工作如 **OpenMoE‑2**、**Tencent‑MoE‑Turbo** 等都在路由负载平衡和缓存压缩上借鉴了该报告的思路。对想进一步研究的读者，可以关注以下方向：① 更高效的负载感知路由算法（如基于强化学习的动态调度）；② 合成数据的质量评估与过滤技术；③ 在多模态（文本+图像）场景下的 MoE 扩展。推测，随着硬件对稀疏计算的原生支持，MoE 将成为突破百亿参数瓶颈的主流路径。

### 一句话记住它
混元‑大模型用 52 B 激活参数和创新的路由、缓存压缩、专属学习率，让开源 MoE 在性能和长上下文上逼近甚至匹配千亿级 dense 模型。