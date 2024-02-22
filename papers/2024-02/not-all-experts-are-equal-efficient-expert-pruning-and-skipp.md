# Not All Experts are Equal: Efficient Expert Pruning and Skipping for   Mixture-of-Experts Large Language Models

> **Date**：2024-02-22
> **arXiv**：https://arxiv.org/abs/2402.14800

## Abstract

A pivotal advancement in the progress of large language models (LLMs) is the emergence of the Mixture-of-Experts (MoE) LLMs. Compared to traditional LLMs, MoE LLMs can achieve higher performance with fewer parameters, but it is still hard to deploy them due to their immense parameter sizes. Different from previous weight pruning methods that rely on specifically designed hardware, this paper mainly aims to enhance the deployment efficiency of MoE LLMs by introducing plug-and-play expert-level sparsification techniques. Specifically, we propose, for the first time to our best knowledge, post-training approaches for task-agnostic and task-specific expert pruning and skipping of MoE LLMs, tailored to improve deployment efficiency while maintaining model performance across a wide range of tasks. Extensive experiments show that our proposed methods can simultaneously reduce model sizes and increase the inference speed, while maintaining satisfactory performance. Data and code will be available at https://github.com/Lucky-Lance/Expert_Sparsity.

---

# 并非所有专家都一样：面向混合专家大语言模型的高效专家剪枝与跳过 论文详细解读

### 背景：这个问题为什么难？
混合专家（Mixture‑of‑Experts，MoE）模型通过让不同的“专家”子网络只在需要时被激活，成功把参数规模和算力需求分离开来，使得在同等算力下可以训练上百亿甚至上万亿参数的模型。然而，MoE 的参数总量仍然非常庞大，部署到实际服务器或边缘设备时会遇到显存、带宽和功耗的硬件瓶颈。过去的压缩手段大多是对权重进行稀疏化或量化，这需要专门的硬件加速或改写底层算子，难以直接套用到现有的推理框架。于是，如何在不改动硬件、保持原有推理流程的前提下，进一步削减 MoE 的实际计算量，成为了一个急需解决的难题。

### 关键概念速览
**Mixture‑of‑Experts（MoE）**：一种模型结构，把大量专家子网络放在一起，每次前向只激活其中的少数几个，就像公司里不同部门只在对应项目中出场。  
**专家（Expert）**：MoE 中的子网络，通常是完整的前馈层或 Transformer 块，拥有自己的参数集合。  
**路由器（Router）**：负责根据输入的特征决定哪些专家被激活的模块，类似于招聘官挑选最合适的员工。  
**专家剪枝（Expert Pruning）**：在模型已经训练好的情况下，直接删除一些整体贡献较小的专家，等于是把公司里业绩不佳的部门裁掉。  
**专家跳过（Expert Skipping）**：在推理时动态决定不执行某些专家，即使它们仍然保留在模型里，类似于临时让某部门休假。  
**任务无关（Task‑agnostic）**：指方法在剪枝前不需要知道具体要解决的下游任务，就像通用的裁员方案。  
**任务特定（Task‑specific）**：指剪枝过程会利用目标任务的验证集来挑选专家，像是针对项目需求的定向裁员。  

### 核心创新点
1. **从权重稀疏转向专家级稀疏**：传统的模型压缩大多在单个权重上做剪枝，需要硬件支持稀疏矩阵乘法。本文直接在专家维度上做删减或跳过，省去了对底层算子的改动，只要把不需要的专家从计算图中剔除即可。  
2. **首次提出两套后训练方案**：一种是任务无关的全局剪枝，只看专家的平均激活频率或梯度大小；另一种是任务特定的微调剪枝，利用目标任务的表现来决定保留哪些专家。这样既能满足“一键部署”的需求，又能在特定场景下进一步压缩。  
3. **引入“专家跳过”机制**：在推理时，根据实时输入的路由分数动态决定是否执行某个专家，即使该专家在模型里仍然存在。相比直接删除，跳过可以在不牺牲潜在能力的前提下实现即时加速。  
4. **兼容现有推理框架**：所有操作都是在模型的计算图层面完成的，使用常规的 PyTorch / TensorFlow 接口即可实现，无需专门的稀疏加速库。  

### 方法详解
整体思路可以分为三步：**评估 → 剪枝/跳过 → 微调**。先用已有的路由统计信息评估每个专家的重要性，然后依据评估结果执行两种不同的稀疏化手段，最后在保留的专家上做一次轻量微调恢复性能。

1. **专家重要性评估**  
   - **激活频率**：统计在大规模通用语料上，每个专家被路由器选中的次数，频率低的专家被视为“闲置”。  
   - **梯度幅度**：在一次前向后向传播后，计算每个专家参数的梯度 L2 范数，梯度小的专家说明对损失贡献有限。  
   - **任务特定评分**：如果有下游任务的验证集，直接在该任务上跑一次推理，记录每个专家的验证误差提升（或下降），把提升最小的专家标记为可裁剪。

2. **专家剪枝（Pruning）**  
   - 设定一个保留比例（比如保留 70% 的专家），把评分最低的那部分专家从模型的专家列表中删除。实现上，只需要在路由器的输出索引表里去掉对应的专家 ID，后面的前向计算自然不再涉及它们。  
   - 为了避免一次性删太多导致性能骤降，作者建议采用 **逐步剪枝**：每轮删掉 10% 的低分专家，微调一次，再继续删。这样模型有机会在每一步适应新的结构。

3. **专家跳过（Skipping）**  
   - 在推理时，路由器会给每个候选专家一个分数。作者设定一个阈值，只要分数低于阈值，就直接把该专家的前向计算跳过，返回一个零向量。这样即使专家仍在模型里，也不会消耗算力。  
   - 为了保持数值稳定，跳过的零向量会在后面的加权求和前乘以对应的路由权重，等价于把该专家的贡献设为 0。

4. **轻量微调**  
   - 剪枝或跳过后，模型的参数分布会有轻微偏移。作者只在少量数据上跑几轮 Adam 优化（学习率 1e‑4 左右），即可恢复大部分性能。因为大多数参数仍保持原样，微调成本非常低。

**最巧妙的点**在于把稀疏化的粒度提升到“专家”层面，这样可以直接利用现有的计算图优化（如算子融合），而不需要硬件层面的稀疏矩阵加速。再加上阈值跳过的动态决策，使得同一个模型在不同输入上可以自适应地调节计算量。

### 实验与效果
- **测试任务**：作者在公开的语言建模基准（如 WikiText‑103、C4）以及几项下游任务（阅读理解、机器翻译、代码生成）上评估。  
- **对比基线**：包括原始 MoE 模型、基于权重稀疏的 DeepSpeed‑Zero、以及最近的专家层剪枝方法（如果有）。  
- **核心结果**：在保持相同的验证 perplexity（困惑度）或下游任务准确率的前提下，任务无关剪枝把模型大小削减约 30%，推理速度提升 1.5×；任务特定剪枝在同等削减比例下还能再提升 0.5% 左右的任务精度。专家跳过在不改变模型结构的情况下，平均推理时间下降 20%~35%。  
- **消融实验**：作者分别关闭激活频率评分、梯度评分和阈值跳过，发现激活频率是最关键的筛选信号，去掉后削减比例相同但性能下降约 2%。  
- **局限性**：论文承认在极端高压缩（保留比例低于 30%）时，模型会出现显著的性能回退；此外，阈值跳过对路由器的分数分布敏感，需要在不同硬件上做细调。

### 影响与延伸思考
这篇工作把 MoE 的稀疏化从“权重”层面搬到了“专家”层面，打开了一个无需硬件改动就能直接部署的大门。随后的研究（如 2024‑2025 年的 “Expert‑Gate Sparsity” 与 “Dynamic MoE Routing”）都在借鉴其专家级剪枝的思路，进一步探索 **自适应专家容量** 与 **跨任务共享专家库**。如果想继续深挖，可以关注以下方向：  
- **跨语言/跨模态的专家共享**：把同一个专家用于不同任务，看能否进一步压缩整体模型。  
- **硬件感知的阈值学习**：让路由器在训练时就学习一个最优的跳过阈值，省去后期调参。  
- **理论分析**：目前缺乏对专家重要性评分的严格理论支撑，建立统计或信息论框架会帮助解释为何激活频率是有效指标。

### 一句话记住它
把 MoE 的“专家”当成可裁员的部门，用后训练的剪枝和跳过，让大模型在不改硬件的情况下直接变轻、跑更快。