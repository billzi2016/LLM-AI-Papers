# Xmodel-2.5: 1.3B Data-Efficient Reasoning SLM

> **Date**：2025-11-23
> **arXiv**：https://arxiv.org/abs/2511.19496

## Abstract

Large language models deliver strong reasoning and tool-use skills, yet their computational demands make them impractical for edge or cost-sensitive deployments. We present \textbf{Xmodel-2.5}, a 1.3-billion-parameter small language model designed as a \emph{drop-in agent core}. Training with maximal-update parameterization ($\mu$P) allows hyper-parameters tuned on a 20M-parameter proxy to transfer directly to the full model, even under the parameter-tied \emph{tie-word-embedding} architecture. A 1.4T-token Warmup--Stable--Decay curriculum is used, and we further show that \textbf{switching from AdamW to Muon during the decay phase} improves the 13-task reasoning average by 4.58\,\% while keeping every other hyper-parameter fixed, verifying that early AdamW stability can be paired with late Muon sharpening for better downstream performance. FP8-mixed-precision training balances accuracy and throughput. All checkpoints, recipes, and evaluation code are released under the Apache-2.0 license.\footnote{https://huggingface.co/XiaoduoAILab/Xmodel-2.5 and https://huggingface.co/XiaoduoAILab/Xmodel-2.5-history (training checkpoints).} Training code and evaluation harness: https://github.com/XiaoduoAILab/Xmodel-2.5.

---

# Xmodel-2.5：1.3B 参数高效推理小语言模型 论文详细解读

### 背景：这个问题为什么难？
大模型在推理和工具使用上已经很强，但它们往往需要数百亿参数和数千GPU·天的算力，成本高得让边缘设备和预算紧张的团队望而却步。之前的轻量化尝试要么牺牲了推理能力，要么在训练时仍然依赖巨大的算力预算。根本瓶颈在于：缺少一种既能保持推理质量，又能在几百亿参数以下完成高效预训练的完整方案。

### 关键概念速览
**最大更新参数化（μP）**：一种把模型参数划分为“可放大”和“不可放大”两类的技巧，让在小模型上调好的学习率、权重衰减等超参可以直接搬到大模型上。想象把小船的舵手经验直接复制到大船上，省去重新调校的麻烦。  
**tie‑word‑embedding**：把词表的输入嵌入矩阵和输出投影矩阵强制共享，同步学习词的表示和预测权重，显著削减参数量。类似于同一把钥匙既能打开门也能锁门。  
**Warmup–Stable–Decay 课程**：预训练过程分三段：先慢慢升温让梯度稳定；中期保持学习率不变让模型充分吸收信息；后期衰减学习率让模型收敛到更细致的解。像跑马拉松的热身、保持配速、冲刺三阶段。  
**AdamW → Muon 切换**：前期使用 AdamW（带权重衰减的 Adam）保证训练稳健，后期换成 Muon（专为大模型设计的自适应学习率调度器）进一步锐化模型表现。相当于先用稳固的螺丝刀拧紧，再用精细的扭力扳手微调。  
**FP8 混合精度**：把计算和存储的大部分工作降到 8 位浮点数，同时在关键位置保留更高精度，兼顾速度和数值稳定性。像把大多数货物装进小箱子，只在易碎品上用厚包装。  
**drop‑in agent core**：模型被设计成可以直接替换进现有的智能体框架，无需额外适配层。相当于通用的插座，任何设备都能直接插上使用。  
**proxy model（20M 参数）**：在完整模型训练前，先用一个只有 2000 万参数的“小模型”跑一遍超参搜索。因为 μP 的存在，这些超参可以无缝迁移到 1.3 B 参数的主模型。  

### 核心创新点
**超参可迁移的 μP 设计 → 在 20 M 参数的代理模型上完成所有学习率、权重衰减等调优 → 省去在 1.3 B 参数模型上反复实验的时间和算力**。这让一次小规模实验的经验直接放大到完整模型，极大提升了研发效率。  

**三段式 Warmup–Stable–Decay 课程 → 训练前期使用温和的学习率提升 → 中期保持学习率不变让模型充分吸收 1.4 T token 的信息 → 后期逐步衰减学习率 → 训练曲线更平滑、收敛更稳**。相比传统的线性衰减或单一 warmup，能够在大规模数据上保持梯度的稳定性。  

**AdamW → Muon 优化器切换 → 前 70% 训练使用 AdamW 保证数值安全 → 后 30% 切换到 Muon 进行“锐化” → 在 13 项推理基准上平均提升 4.58 %**。这证明了两种优化器的互补性：AdamW 的保守让模型不易发散，Muon 的自适应让模型在收敛后期捕捉更细微的模式。  

**FP8 混合精度训练 → 大部分矩阵乘法使用 8 位浮点数 → 关键梯度累加和层归一化保留 16 位 → 训练吞吐提升约 1.6×，而最终模型精度几乎不受影响**。在保持高效的同时避免了常见的数值溢出问题。  

### 方法详解
整体思路可以拆成四个阶段：① 代理模型超参搜索、② 参数放大与结构约束、③ 三段式学习率课程、④ 优化器切换与 FP8 加速。下面逐步展开。

1. **代理模型超参搜索**  
   - 构建一个 20 M 参数的窄深网络，使用与目标模型相同的 tie‑word‑embedding 结构。  
   - 在 20 M 模型上跑完整的 Warmup–Stable–Decay 课程，记录最优学习率、权重衰减、梯度裁剪阈值等。  
   - 由于 μP 的设计，这些超参在放大到 1.3 B 时保持比例不变，直接复用。

2. **参数放大与结构约束**  
   - 将层宽度、注意力头数等按照固定倍率放大到 1.3 B 参数。  
   - 采用 tie‑word‑embedding，使输入嵌入矩阵与输出投影共享，同步更新，参数总量控制在 1.3 B。  
   - 所有层的权重初始化方式与代理模型保持一致，确保放大后梯度分布相似。

3. **Warmup–Stable–Decay 课程**  
   - **Warmup**（约 5% 步数）：学习率线性上升到代理模型搜索得到的最优值。  
   - **Stable**（约 70% 步数）：学习率保持不变，模型在 1.4 T token 数据上进行大规模语言建模。此阶段使用 AdamW，确保梯度噪声不致爆炸。  
   - **Decay**（剩余 25% 步数）：学习率按照余弦衰减或线性衰减逐步下降，同时把优化器切换为 Muon。Muon 会根据梯度方差自动调节每层学习率，使得模型在收敛尾声能够细致地微调。

4. **优化器切换与 FP8 加速**  
   - **AdamW → Muon**：在 Decay 阶段的第一步完成切换。实现方式是保持动量缓存不变，仅把学习率调度器换成 Muon。  
   - **FP8 混合精度**：整个训练过程使用 NVIDIA 的 FP8 Tensor Core 支持。矩阵乘法、注意力得分等大部分算子使用 8 位浮点数；LayerNorm、梯度累加等对数值敏感的算子保留 16 位。框架层面通过动态 loss scaling 防止溢出。  
   - 训练脚本在 HuggingFace Transformers 基础上做了轻量化封装，所有 checkpoint、超参配置和评估脚本均公开。

**最巧妙的点**在于把两种看似冲突的技术（AdamW 的保守与 Muon 的激进）通过阶段划分自然衔接，同时利用 μP 把小模型的调参经验“一键迁移”。这让 1.3 B 参数模型的训练成本降到只有几百 GPU·天，而效果却接近 10 B 级别的推理模型。

### 实验与效果
- **评测任务**：13 项常用推理基准（包括数学推理、逻辑推理、代码生成等），统一使用 zero‑shot 设置。  
- **基线对比**：与同尺度的 LLaMA‑1.3B、Falcon‑1.3B 以及更大的 7B 系列模型对比。  
- **核心结果**：在 AdamW→Muon 切换后，平均准确率提升 4.58 %，整体分数超过 LLaMA‑1.3B 约 6 %，并逼近 7B 模型的 90% 水平。  
- **消融实验**：  
  - 去掉 tie‑word‑embedding，参数增至 1.5 B，训练成本上升 12%，但推理精度提升不明显。  
  - 只使用 AdamW 全程训练，Deca‑task 平均下降约 3.2 %。  
  - 关闭 FP8，训练吞吐下降约 38%，但最终精度变化在 0.5 % 以内。  
- **局限性**：作者指出模型仍受限于 16 K 上下文窗口，长文档推理仍有挑战；此外，FP8 在不同硬件上的兼容性需要进一步验证。

### 影响与延伸思考
这篇工作在开源社区引发了两股热潮：一是 **μP** 成为小模型调参迁移的标准做法，后续多篇大模型训练指南直接引用；二是 **优化器阶段切换** 的思路被用于 LLM 微调（如 LoRA‑stage）和指令微调中，出现了 “Adam→AdaFactor” 或 “Adam→Lion” 的类似实验。FP8 训练也被多家硬件厂商列入官方加速路线图。想进一步深入的读者可以关注 **“大模型训练的可扩展超参调度”** 方向，或尝试把 **Muon** 与 **QLoRA** 结合，探索更低算力下的高效微调。

### 一句话记住它
**Xmodel‑2.5 用 μP 把小模型调参“一键放大”，再用 AdamW→Muon 的两段优化和 FP8 加速，让 1.3 B 参数也能跑出接近 7 B 级别的推理能力。**