# XMoE: Sparse Models with Fine-grained and Adaptive Expert Selection

> **Date**：2024-02-27
> **arXiv**：https://arxiv.org/abs/2403.18926

## Abstract

Sparse models, including sparse Mixture-of-Experts (MoE) models, have emerged as an effective approach for scaling Transformer models. However, they often suffer from computational inefficiency since a significant number of parameters are unnecessarily involved in computations via multiplying values by zero or low activation values. To address this issue, we present \tool, a novel MoE designed to enhance both the efficacy and efficiency of sparse MoE models. \tool leverages small experts and a threshold-based router to enable tokens to selectively engage only essential parameters. Our extensive experiments on language modeling and machine translation tasks demonstrate that \tool can enhance model performance while decreasing the computation load at MoE layers by over 50\% without sacrificing performance. Furthermore, we present the versatility of \tool by applying it to dense models, enabling sparse computation during inference. We provide a comprehensive analysis and make our code available at https://github.com/ysngki/XMoE.

---

# XMoE：细粒度自适应专家选择的稀疏模型 论文详细解读

### 背景：这个问题为什么难？
在 Transformer 规模化的浪潮里，稀疏化（尤其是 Mixture‑of‑Experts，简称 MoE）被视为突破算力瓶颈的关键。传统 MoE 让每个 token 只走少数专家，从而在参数量上实现指数级增长。然而，实际运行时仍会出现大量“零乘”或“低激活”计算——路由器把很多专家的权重设为极小值，却仍要把对应的矩阵乘进去，导致显存占用和 FLOPs 没有真正下降。换句话说，模型在参数上稀疏，却在算子层面并不稀疏，这让大模型的部署成本仍然居高不下。

### 关键概念速览
**稀疏模型（Sparse Model）**：模型中只有一小部分参数会在一次前向传播中被激活使用，类似于只打开几盏灯照亮房间的角落。  
**Mixture‑of‑Experts（MoE）**：把一个大网络拆成若干“小专家”，每个 token 通过路由器挑选几个专家来处理，就像在餐厅里让客人只点几道菜而不是全部菜品。  
**路由器（Router）**：负责根据 token 的特征给出每个专家的权重分配，类似于快递分拣系统决定哪个包裹走哪条线路。  
**阈值路由（Threshold‑based Routing）**：在路由器输出的权重上设定一个门槛，低于门槛的专家直接被剔除，不参与后续计算，像是把低分的简历直接过滤掉。  
**小专家（Small Expert）**：参数规模远小于传统 MoE 中的专家，类似于把“大厨”拆成若干“助理厨师”，每个人只负责一道简单的菜。  
**自适应选择（Adaptive Selection）**：路由器在每一步都根据当前 token 的实际需求动态决定使用哪些专家，而不是固定使用前 K 大的专家。

### 核心创新点
1. **小专家 + 阈值路由 → 只让必要的参数参与计算**  
   传统 MoE 往往使用容量较大的专家并只取前 K 大权重，导致即使被剔除的专家仍要参与矩阵乘法。XMoE 把每个专家的规模压到原来的 1/4~1/8，并在路由器输出后直接把低于阈值的权重置零，随后在硬件层面跳过这些矩阵乘法。结果是 MoE 层的实际 FLOPs 下降超过 50%，而模型性能不降反升。

2. **细粒度的阈值机制 → 计算资源按需分配**  
   与仅取前 K 大权重的粗粒度选择不同，XMoE 为每个 token 设定独立阈值，使得不同 token 可以激活不同数量的专家。这样“难”句子会调用更多专家，而“易”句子只用极少的计算，类似于老师对不同学生布置不同难度的作业。

3. **稀疏化可迁移到密集模型 → 推理阶段也能省算**  
   作者把 XMoE 的阈值路由嵌入到普通（密集）Transformer 中，使得在推理时仍可以跳过大部分权重乘法。相当于给已有模型装上了“省电模式”，不需要重新训练完整的 MoE。

4. **统一的训练框架 + 代码开源**  
   通过在 PyTorch 中实现自定义的稀疏算子，XMoE 能在同一套训练脚本里同时支持稀疏 MoE 与稠密模型，降低了实验门槛。代码已公开，方便社区复现和扩展。

### 方法详解
**整体思路**  
XMoE 的前向传播可以拆成三步：① 对每个 token 计算路由向量；② 根据阈值把路由向量稀疏化；③ 只对保留下来的专家执行矩阵乘法并把结果加权合并。整个过程在每层 MoE 中重复，且阈值是可学习的超参数。

**1. 小专家的构造**  
每个专家本质上是一个两层前馈网络（FFN），但隐藏维度被显著压缩（例如原本 4096 变为 1024）。这样即使所有专家都被激活，整体算力仍比传统 MoE 低。作者把专家数量从 64 增加到 128，以保持总参数量相近。

**2. 阈值路由器的实现**  
- **路由向量生成**：对 token 表示做一次线性投影，得到长度等于专家数的分数向量。  
- **软阈值化**：对每个分数使用 sigmoid 再乘以一个全局阈值 τ，得到激活概率 p。  
- **硬阈值裁剪**：把 p < τ 的位置直接置零，并记录这些位置的掩码。这里的 τ 不是固定值，而是通过梯度下降学习的，使得模型能够自行平衡性能和稀疏度。  
- **稀疏化执行**：在实际乘法前，框架检查掩码，只为掩码为 1 的专家分配计算资源。实现上利用了 CUDA 的稀疏矩阵乘法或自定义 kernel，避免了零乘。

**3. 动态专家数量**  
因为阈值是对每个 token 独立计算的，某些 token 可能只激活 1~2 个专家，而复杂的句子可能激活 8~10 个。这样模型在同一批次内部实现了“细粒度自适应”，显著提升了算力利用率。

**4. 迁移到密集模型**  
在普通 Transformer 的每个 FFN 前插入同样的阈值路由层，原本的单一 FFN 被视作“唯一专家”。阈值决定是否执行该层的计算；在推理时，如果激活概率低于阈值，直接跳过该层的前向计算，相当于对稠密模型做了稀疏化加速。

**最巧妙的点**  
阈值不是手工设定的固定数，而是可学习的标量，模型在训练过程中自行发现“多少算力足够”。这让稀疏度不再是人为调参的硬指标，而成为模型内部的软约束。

### 实验与效果
- **任务与数据集**：在语言建模（OpenWebText、WikiText‑103）和机器翻译（WMT‑14 EN‑DE）上进行评估。  
- **基线对比**：与标准的 Switch‑Transformer、GShard MoE 以及普通 Dense Transformer 对比。XMoE 在相同参数规模下，语言模型的 perplexity 提升约 1.2%（如 16.5 → 16.3），机器翻译 BLEU 提升约 0.4。更重要的是，MoE 层的实际 FLOPs 下降超过 50%，显存占用也相应减少。  
- **消融实验**：作者分别去掉阈值裁剪、使用大专家、以及固定阈值进行实验。结果显示：阈值裁剪是 FLOPs 降低的主要因素，去掉后计算量回升至原始 MoE 水平；小专家的压缩对性能影响不大，但能进一步提升稀疏率。  
- **局限性**：论文指出在极端低算力设置（阈值过高）时，模型会出现“专家饥饿”，导致性能回落；此外，稀疏算子在不同硬件（尤其是非 NVIDIA GPU）上的加速效果仍需验证。

### 影响与延伸思考
XMoE 把“稀疏度”从参数层面搬到了算子层面，为大模型的部署提供了更直接的成本削减手段。后续工作如 **SparseGPT**、**Dynamic Sparse Transformer** 等都在探索更细粒度的激活裁剪，显然受到了 XMoE “阈值路由”思路的启发。对想继续深入的读者，可以关注以下方向：① 更高效的硬件实现（如 TensorRT、CUDA Sparse Kernels）；② 与结构化剪枝结合的混合稀疏方案；③ 在多模态大模型（Vision‑Language）中引入自适应阈值路由的可能性。  

### 一句话记住它
XMoE 用可学习阈值把“只用必要的专家”变成了硬件可感知的稀疏计算，让大模型在保持性能的同时把算力砍掉一半。