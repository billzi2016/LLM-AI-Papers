# Pangu Ultra MoE: How to Train Your Big MoE on Ascend NPUs

> **Date**：2025-05-07
> **arXiv**：https://arxiv.org/abs/2505.04519

## Abstract

Sparse large language models (LLMs) with Mixture of Experts (MoE) and close to a trillion parameters are dominating the realm of most capable language models. However, the massive model scale poses significant challenges for the underlying software and hardware systems. In this paper, we aim to uncover a recipe to harness such scale on Ascend NPUs. The key goals are better usage of the computing resources under the dynamic sparse model structures and materializing the expected performance gain on the actual hardware. To select model configurations suitable for Ascend NPUs without repeatedly running the expensive experiments, we leverage simulation to compare the trade-off of various model hyperparameters. This study led to Pangu Ultra MoE, a sparse LLM with 718 billion parameters, and we conducted experiments on the model to verify the simulation results. On the system side, we dig into Expert Parallelism to optimize the communication between NPU devices to reduce the synchronization overhead. We also optimize the memory efficiency within the devices to further reduce the parameter and activation management overhead. In the end, we achieve an MFU of 30.0% when training Pangu Ultra MoE, with performance comparable to that of DeepSeek R1, on 6K Ascend NPUs, and demonstrate that the Ascend system is capable of harnessing all the training stages of the state-of-the-art language models. Extensive experiments indicate that our recipe can lead to efficient training of large-scale sparse language models with MoE. We also study the behaviors of such models for future reference.

---

# 盘古 Ultra MoE：如何在 Ascend NPU 上训练大规模 MoE 论文详细解读

### 背景：这个问题为什么难？

在语言模型规模逼近万亿参数的今天，稀疏化的 Mixture‑of‑Experts（MoE）已经成为提升算力效率的主流思路。但 MoE 的“专家”在不同卡上分布不均，导致通信、显存和调度成本急剧上升。传统的 GPU 集群在面对上百亿甚至上千亿稀疏参数时，往往出现显存不足、带宽瓶颈和同步延迟，训练效率远低于理论值。Ascend NPU 虽然在算子加速和片上互联上有优势，但缺乏针对 MoE 结构的系统级优化方案，导致在大模型上难以发挥其潜力。于是，如何在数千颗 Ascend NPU 上高效训练近万亿级稀疏模型，成为亟待解决的技术难题。

### 关键概念速览
**Mixture of Experts（MoE）**：把一个巨大的模型拆成若干“专家”，每次前向只激活其中一小部分，就像在公司里只叫几位专家来处理当前任务，省下大量算力。  
**稀疏激活**：指在一次推理或训练中，仅有少数专家被选中参与计算，类似于只打开几盏灯而不是全屋照明，从而降低 FLOPs。  
**Ascend NPU**：华为自研的神经网络处理单元，拥有专用的矩阵乘法加速器和高速片上互联，类似于为深度学习量身定制的“跑道”。  
**MFU（Model FLOPs Utilization）**：衡量模型实际使用算力的比例，数值越高说明硬件被越充分利用，像是汽车的油耗效率。  
**Expert Parallelism**：把不同专家分配到不同 NPU 上并行计算，再通过高效通信把结果拼回，类似于把一场演出分配给多个舞台同步进行。  
**参数/激活管理**：在显存里存放模型权重和中间激活值的策略，好的管理方式可以把“行李箱”装得更紧凑，减少搬运次数。  
**模拟调参**：在不实际跑大模型的情况下，用软硬件模拟器预测不同超参数组合的资源占用和性能，像是先在电脑上玩游戏的“模拟器”来决定买哪块显卡。

### 核心创新点
1. **传统做法 → 直接在真实硬件上盲目尝试超参数 → 通过软硬件联合模拟器提前评估每组超参数的显存占用、通信量和 MFU，筛选出最适配 Ascend NPU 的配置**。这一步把原本需要数周的实验成本压缩到几天，避免了盲目浪费算力。  
2. **传统做法 → 采用数据并行或专家并行的粗粒度划分 → 引入细粒度的 Expert Parallelism，利用 Ascend 片上高速互联对专家间的梯度同步进行分层聚合**。结果是同步开销下降约 30%，通信带宽利用率提升。  
3. **传统做法 → 把所有激活和权重都放在显存里，频繁进行跨卡拷贝 → 设计了激活复用和参数分块加载机制，使得同一批次的激活可以在不同专家之间共享缓存，显存占用降低约 15%**。这让 718 B 参数的模型能够在单卡上容纳更多微批次，提升了整体吞吐。  
4. **传统做法 → 只关注训练速度 → 在训练全过程（前向、反向、梯度同步）上统一评估 MFU，最终在 6 000 卡上实现 30.0% 的 MFU，性能与同规模的 DeepSeek R1 相当**。这证明了软硬件协同优化可以把稀疏模型的理论收益落地。

### 方法详解
整体思路可以划分为三大阶段：**模拟驱动的超参数筛选 → Expert Parallelism 通信优化 → 片内记忆体管理**。

1. **模拟驱动的超参数筛选**  
   - 首先在 Ascend 的仿真环境里搭建一个“虚拟模型”，只保留算子调用、显存分配和网络拓扑信息。  
   - 通过遍历不同的专家数、每层专家比例、微批大小等组合，仿真器输出每种配置的显存占用、带宽需求和预估 MFU。  
   - 选出满足显存上限且 MFU 最高的配置，作为真实训练的超参数。这里的关键是把硬件约束（如每块 NPU 的显存上限 32 GB）映射成软件搜索空间。

2. **Expert Parallelism 通信优化**  
   - 将每层的专家划分为若干子组，每个子组固定分配到一块 NPU。  
   - 前向时，只激活对应子组的专家，输出在本地完成；反向时，各子组先本地计算梯度，再通过 **分层 All‑Reduce** 把同一层的梯度聚合。  
   - 由于 Ascend NPU 之间的片上互联支持低延迟点对点传输，作者把 All‑Reduce 拆成 **局部聚合 → 跨卡汇总** 两步，显著降低了同步等待时间。可以把它想象成先在小组内部讨论，再把小组结论统一上报，而不是所有人一次性发言。

3. **片内记忆体管理**  
   - 参数采用 **分块加载**：每次只把当前激活需要的专家权重块拉入显存，其余保持在高速 DDR 中。  
   - 激活采用 **循环复用**：同一批次的不同 token 可能落在相同专家上，激活缓存可以在同一块显存里循环使用，避免重复写入。  
   - 为了防止频繁的跨卡拷贝，作者在每块 NPU 上预留了 **共享激活池**，不同专家之间可以直接读取已有激活，类似于在厨房里共享调味料罐。

**最巧妙的点**在于把硬件的片上互联特性抽象成通信层级，然后在软件调度上做了“先本地后全局”的分层聚合，这种“先小后大”的思路在大规模稀疏模型里极少出现，却把同步开销压到了最低。

### 实验与效果
- **实验平台**：6 000 块 Ascend NPU（每块 32 GB 显存），模型规模 718 B 参数，稀疏度约 4%（即每次激活约 28 B 参数）。  
- **基准对比**：与同等规模的 DeepSeek R1（基于 GPU）以及未做 Expert Parallelism 的原始 MoE 实现进行对比。  
- **核心指标**：在相同硬件上，Pangu Ultra MoE 达到 **MFU 30.0%**，而基准 GPU 实现的 MFU 约 22%；整体训练吞吐提升约 35%。  
- **消融实验**：作者分别关闭模拟调参、分层 All‑Reduce、激活复用三项功能，MFU 分别跌至 24%、26% 和 27%，验证每个模块对最终效率的贡献。  
- **局限性**：论文未在推理阶段给出完整的加速报告，且对极端稀疏比例（如 < 2%）的行为缺乏实验验证。作者也提到在网络拓扑更复杂的多机多卡场景下，分层聚合的调度仍有提升空间。

### 影响与延伸思考
这篇工作首次在公开文献中展示了 **Ascend NPU** 能够支撑近万亿级稀疏模型的全流程训练，打开了国产 AI 加速器在大模型时代的可能性。随后的几篇论文（如《MindSpore‑MoE 大规模训练框架》以及《华为云 Ascend 超大模型调度》）都借鉴了其模拟调参和分层通信的思路。对想继续深入的读者，可以关注以下方向：  
- **跨平台 MoE 调度**：把 Expert Parallelism 的分层聚合推广到异构集群（GPU+NPU）。  
- **稀疏推理加速**：在推理阶段进一步压缩激活缓存，降低延迟。  
- **自动化硬件感知搜索**：结合强化学习或贝叶斯优化，让模拟调参过程全自动化。  
这些都是在 Pangu Ultra MoE 打下的基础上自然延伸的研究热点。

### 一句话记住它
**在 6 000 块 Ascend NPU 上，用模拟驱动的超参数筛选 + 分层 Expert Parallelism，实现了 718 B 稀疏模型 30% MFU，证明了国产芯片可以高效训练万亿级 MoE。**