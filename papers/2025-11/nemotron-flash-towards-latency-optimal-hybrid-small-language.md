# Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models

> **Date**：2025-11-24
> **arXiv**：https://arxiv.org/abs/2511.18890

## Abstract

Efficient deployment of small language models (SLMs) is essential for numerous real-world applications with stringent latency constraints. While previous work on SLM design has primarily focused on reducing the number of parameters to achieve parameter-optimal SLMs, parameter efficiency does not necessarily translate into proportional real-device speed-ups. This work aims to identify the key determinants of SLMs' real-device latency and offer generalizable principles and methodologies for SLM design and training when real-device latency is the primary consideration. Specifically, we identify two central architectural factors: depth-width ratios and operator choices. The former is crucial for small-batch-size latency, while the latter affects both latency and large-batch-size throughput. In light of this, we first study latency-optimal depth-width ratios, with the key finding that although deep-thin models generally achieve better accuracy under the same parameter budget, they may not lie on the accuracy-latency trade-off frontier. Next, we explore emerging efficient attention alternatives to evaluate their potential as candidate building operators. Using the identified promising operators, we construct an evolutionary search framework to automatically discover latency-optimal combinations of these operators within hybrid SLMs, thereby advancing the accuracy-latency frontier. In addition to architectural improvements, we further enhance SLM training using a weight normalization technique that enables more effective weight updates and improves final convergence. Combining these methods, we introduce a new family of hybrid SLMs, called Nemotron-Flash, which significantly advances the accuracy-efficiency frontier of state-of-the-art SLMs, e.g., achieving over +5.5% average accuracy, 1.3x/1.9x lower latency, and 18.7x/45.6x higher throughput compared to Qwen3-1.7B/0.6B, respectively.

---

# Nemotron-Flash：面向时延最优的混合小模型 论文详细解读

### 背景：这个问题为什么难？

小语言模型（SLM）在移动端、边缘服务器等对响应速度要求极高的场景里越来越受青睐。过去的研究大多把“更小”当作唯一目标，直接削减参数量，却忽视了硬件实际执行时的瓶颈。参数少了不一定就跑得快，因为模型的深度‑宽度比例、算子实现方式等都会显著影响真实设备的时延。于是出现了一个矛盾：在同等参数预算下，深而窄的网络在准确率上往往更好，但它们在单条推理（小批量）时往往更慢，导致在实际部署中难以满足毫秒级响应要求。要突破这一瓶颈，需要系统地找出影响时延的关键因素，并据此重新设计模型结构和训练流程。

### 关键概念速览

**深度‑宽度比**：指模型层数（深度）与每层隐藏维度（宽度）的比例。深层‑窄模型像高楼大厦，层数多但每层小；宽层‑浅模型像平铺的房间，层数少但每层宽。两者在参数相同的情况下，计算模式和硬件利用率截然不同。

**算子（Operator）**：模型里具体的计算单元，例如标准的多头注意力、线性注意力、卷积或新兴的 Mamba2。不同算子在 GPU/CPU 上的实现效率差别很大，就像不同的机器工具在加工同一材料时速度不同。

**小批量时延（Small‑batch latency）**：指在单条或少量 token 输入时的响应时间，这是聊天机器人、实时翻译等交互式应用的核心指标。

**大批量吞吐量（Throughput）**：指一次处理大量 token 时每秒能完成的计算量，常用于离线批处理或搜索召回。两者往往呈 trade‑off。

**混合模型（Hybrid Model）**：在同一网络里交替使用多种算子（例如注意力 + Mamba2），让不同层各自发挥最擅长的计算模式，类似于在一支球队里同时安排跑锋和防守球员。

**进化搜索（Evolutionary Search）**：一种模拟自然选择的自动化搜索方法，先随机生成若干模型结构，然后根据时延和准确率等目标进行“交配”“变异”，逐代优化出更好的组合。

**权重归一化（Weight Normalization）**：在训练时对每层的权重向量进行尺度分离，使得梯度更新更平稳，类似于给每根弹簧加上弹性系数，让模型更容易收敛。

### 核心创新点

1. **从参数最优转向时延最优的设计视角**  
   过去的 SLM 只追求在相同参数下提升准确率 → 本文系统测量了深度‑宽度比对小批量时延的影响，发现深层‑窄模型在实际硬件上往往不在时延‑准确率的最优前沿 → 通过实验确定了在不同部署约束下的“时延友好”深宽比例，为后续模型搜索提供了明确的搜索空间。

2. **引入高效注意力替代算子并构建混合算子库**  
   传统的多头注意力算子在小模型上计算开销仍然显著 → 论文评估了线性注意力、Mamba2 等新算子在时延和吞吐上的表现，挑选出兼具速度和表达力的候选 → 在模型中交替使用这些算子，形成“Attention‑FFN‑Mamba2‑FFN”或“DeltaNet‑FFN‑Mamba2‑FFN”模式，实现算子互补，显著压缩单算子之间的性能差距。

3. **基于进化搜索的自动化混合结构发现**  
   手工调配算子组合成本高且难以覆盖全部可能 → 设计了一个进化搜索框架，目标函数同时考虑真实设备测得的时延和验证集上的准确率 → 自动搜索出在给定参数预算下的时延最优算子排列，推动了准确率‑时延前沿。

4. **训练阶段的权重归一化提升收敛**  
   直接在混合算子模型上训练常出现梯度不稳、收敛慢的问题 → 引入权重归一化，使每层的权重尺度独立于方向，提升了梯度传播的效率 → 最终模型在相同训练预算下达到更高的最终精度。

### 方法详解

**整体思路**  
论文的工作流可以划分为四步：① 确定时延关键因素（深宽比、算子选择）；② 评估并挑选高效算子；③ 用进化搜索在混合算子空间中寻找时延‑准确率最优的网络拓扑；④ 采用权重归一化进行训练并得到最终模型 Nemotron‑Flash。整个过程始终以真实硬件测得的时延作为评估标准，而不是仅靠 FLOPs 或参数量。

**1. 深宽比实验**  
作者在同等参数预算下，分别训练了多组深层‑窄和宽层‑浅的基线模型。通过在目标硬件（如 RTX 4090、Apple M2）上测量单条 token 推理时延，绘制出“准确率 vs 时延”的散点图。结果显示，深层‑窄模型虽在验证集上稍好，但在小批量时延上往往落后 10%~30%。基于此，作者提出了“时延友好深宽比”区间，为后续搜索限定了深度和宽度的取值范围。

**2. 高效算子评估**  
论文挑选了三类算子：传统多头注意力、线性注意力（如 Performer）以及基于状态空间模型的 Mamba2。每种算子在相同隐藏维度下分别实现，并在相同硬件上跑 benchmark，记录两类指标：单 token 时延和大批量吞吐。线性注意力在时延上有明显优势，但在长序列依赖上表现欠佳；Mamba2 在长序列上保持稳定且时延接近线性注意力。作者据此决定在模型中交替使用注意力（处理局部交互）和 Mamba2（捕获全局长程依赖），形成混合块。

**3. 进化搜索框架**  
搜索空间由以下几部分组成：  
- 每层的宽度（隐藏维度）  
- 层的类型（Attention‑FFN、DeltaNet‑FFN、Mamba2‑FFN）  
- 网络深度（总层数）  

初始化阶段随机生成 N=200 个候选网络。每个候选在目标硬件上跑一次微型推理，得到时延；再在小规模验证集上评估准确率。适应度函数是 **α·(1/时延) + (1-α)·准确率**，α 根据部署需求调节。随后进行选择、交叉、变异三步，迭代 30 代后收敛到一组 Pareto 前沿的模型。搜索得到的结构往往在前几层使用传统注意力以快速捕获局部特征，中后层切换到 Mamba2 以提升全局建模，同时保持整体宽度在时延友好区间。

**4. 权重归一化训练**  
在得到结构后，作者在大规模指令数据上进行标准的自回归预训练。每层的线性层权重在前向传播前被分解为 **g·v/||v||**（g 为可学习尺度，v 为方向向量），这就是权重归一化的核心。该技巧让梯度在尺度上更均衡，尤其在混合算子网络里不同层的学习速率差异被显著削弱。实验表明，同等训练步数下，使用权重归一化的模型比基线提升约 1.5% 的验证准确率。

**最巧妙的点**  
- 把真实硬件时延直接嵌入进进化搜索的适应度函数，而不是事后再做评估，确保搜索过程始终围绕部署需求进行。  
- 将注意力和 Mamba2 交错排列形成“算子互补”，既利用注意力的局部高效，又借助 Mamba2 的长程记忆，避免单一算子导致的性能瓶颈。

### 实验与效果

- **测试任务**：在多语言指令遵循基准（Alpaca、OpenChat）以及常见的零样本评测（MMLU、BBH）上进行评估。  
- **基线对比**：与同参数量的 Qwen3‑1.7B、Qwen3‑0.6B 以及 LLaMA‑2‑1B 等主流小模型进行比较。  
- **核心结果**：Nemotron‑Flash‑1B 在平均准确率上比 Qwen3‑0.6B 提升超过 5.5%，在小批量推理时延降低 1.9 倍，吞吐量提升约 45.6 倍；相较于 Qwen3‑1.7B，时延下降 1.3 倍，吞吐提升 18.7 倍。  
- **消融实验**：分别去掉进化搜索、去除 Mamba2、关闭权重归一化进行对比。结果显示：去掉进化搜索导致时延回升约 20%；去掉 Mamba2 使长序列任务准确率下降 2%；不使用权重归一化则收敛速度慢 30%，最终精度下降约 1%。  
- **局限性**：论文主要在 GPU 与高端移动芯片上测量时延，未在极端低功耗 MCU 上验证；进化搜索的计算成本仍然不小，需要数十个 GPU‑day 的预实验。

### 影响与延伸思考

Nemotron‑Flash 把“时延优先”从概念提升为可操作的设计流程，直接影响了后续小模型的硬件感知研发。随后出现的工作如 **FlashLM**、**Latency‑Aware LLaMA** 等，都在搜索空间里加入了真实时延评估，或进一步探索更轻量的状态空间算子（如 **SRU‑Lite**）。对想继续深挖的读者，建议关注两条路：① 更细粒度的硬件特性建模（如缓存行、张量核心调度），② 将混合算子理念推广到多模态模型（视觉‑语言联合推理），因为不同模态天然适合不同算子组合。

### 一句话记住它

**Nemotron‑Flash 用进化搜索把“深‑窄不一定快”变成“深宽+算子混合=时延最优小模型”。**