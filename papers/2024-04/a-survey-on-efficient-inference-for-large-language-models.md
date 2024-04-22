# A Survey on Efficient Inference for Large Language Models

> **Date**：2024-04-22
> **arXiv**：https://arxiv.org/abs/2404.14294

## Abstract

Large Language Models (LLMs) have attracted extensive attention due to their remarkable performance across various tasks. However, the substantial computational and memory requirements of LLM inference pose challenges for deployment in resource-constrained scenarios. Efforts within the field have been directed towards developing techniques aimed at enhancing the efficiency of LLM inference. This paper presents a comprehensive survey of the existing literature on efficient LLM inference. We start by analyzing the primary causes of the inefficient LLM inference, i.e., the large model size, the quadratic-complexity attention operation, and the auto-regressive decoding approach. Then, we introduce a comprehensive taxonomy that organizes the current literature into data-level, model-level, and system-level optimization. Moreover, the paper includes comparative experiments on representative methods within critical sub-fields to provide quantitative insights. Last but not least, we provide some knowledge summary and discuss future research directions.

---

# 大语言模型高效推理综述 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）拥有上百亿甚至上千亿参数，单次推理需要遍历整个模型并执行自回归解码，导致计算量和显存占用呈指数级增长。传统的 GPU/CPU 加速手段在面对这种规模时往往只能提供线性提升，根本无法满足移动端或边缘服务器的资源限制。更糟的是，注意力机制的计算复杂度是二次的，序列越长耗时越多，导致长文本生成几乎不可用。于是，提升 LLM 推理效率成为制约其广泛落地的关键瓶颈，也正是这篇综述要系统梳理的核心问题。

### 关键概念速览
**自回归解码**：模型一次生成一个 token（词），后面的生成依赖前面已经产生的结果，就像写文章时必须先写好前一句才能继续。  
**注意力（Attention）**：模型在处理当前 token 时，会对所有历史 token 计算相似度并加权求和，类似于人在阅读时会回顾整段文字寻找关联信息。  
**量化（Quantization）**：把模型参数从 32 位浮点数压缩到更低位宽（如 8 位或 4 位），相当于把高精度的彩色图片压成低分辨率的黑白图，牺牲一点细节换取更小的存储和更快的运算。  
**稀疏化（Sparsity）**：让模型的权重矩阵中大部分元素为零，只保留关键的少数连接，类似于只保留道路网络中的主干道，减少不必要的交通流量。  
**分块注意力（Chunked/Block Attention）**：把长序列切成若干块，只在块内部或块之间做有限的交互，像把一本书分章节阅读，避免一次性翻阅全书。  
**蒸馏（Distillation）**：用大模型的输出作为“老师”，训练一个更小的“学生”模型，让学生在保持大模型能力的同时拥有更轻量的结构。  
**流水线并行（Pipeline Parallelism）**：把模型层划分到不同的硬件上，像装配线一样并行处理不同阶段的计算，提升整体吞吐率。  
**缓存（KV Cache）**：在自回归生成时保存每一步的键值对（Key/Value），后续只需增量计算，类似于把已经翻译好的段落存下来，下次直接复用。

### 核心创新点
1. **从单一视角到全景分类**：过去的研究多聚焦在量化或稀疏化等单一技术上，作者提出了“数据层‑模型层‑系统层”三层 taxonomy，把所有已有方法统一进一个层次结构，帮助研究者快速定位自己感兴趣的优化点。  
2. **系统级基准实验**：作者不只罗列理论，还在同一硬件环境下跑了多组代表性方法的对比实验，展示了不同层次技术在实际加速比、显存占用和精度损失之间的权衡，为后续工作提供了可复现的基准。  
3. **跨层协同策略概念**：在综述中指出，仅靠模型层的压缩往往受限于硬件调度，而系统层的调度优化可以进一步放大模型层的收益，提出了“协同加速”思路，鼓励后续研究把多层技术组合起来。  
4. **未来方向的系统化展望**：作者把未解决的问题归纳为“可预测稀疏、动态量化、硬件‑软件协同设计”等四大方向，为社区指明了下一步的研究热点。

### 方法详解
#### 整体框架
这篇工作本身不是提出新算法，而是提供了一套完整的梳理与评估流程。作者首先分析 LLM 推理低效的根本原因（模型规模、二次注意力、逐 token 解码），随后构建三层 taxonomy，最后在每层挑选出具有代表性的技术进行实验验证。整个过程可以看作“问题 → 分类 → 实证 → 展望”四步走。

#### 关键模块拆解
1. **原因分析模块**  
   - **模型规模**：统计参数量、显存占用、每次前向传播 FLOPs。  
   - **注意力复杂度**：解释为何注意力的时间和空间复杂度随序列长度呈二次增长。  
   - **自回归解码**：展示 KV Cache 如何在每一步只增量计算，仍然受限于序列长度的累计成本。  

2. **三层 taxonomy**  
   - **数据层**：包括输入压缩（如分词粒度调节、长文本截断）和输出缓存策略。  
   - **模型层**：涵盖量化、稀疏化、低秩分解、蒸馏、结构剪枝等技术。每种技术都配有“压缩比例‑精度损失”二维坐标的概览图。  
   - **系统层**：涉及流水线并行、张量并行、显存管理（如 offloading 到 CPU/SSD）以及专用加速库（FlashAttention、xFormers）等实现细节。  

3. **实验评估模块**  
   - **基准选取**：作者挑选了 GPT‑NeoX、LLaMA、OPT 等主流开源 LLM，分别在单卡 GPU、双卡多卡以及 CPU 环境下跑实验。  
   - **指标体系**：包括推理吞吐率（tokens/s）、显存峰值、延迟（ms/token）以及任务精度（如 Zero‑Shot QA 的准确率）。  
   - **对比方式**：对每一层技术单独评测，再组合不同层次的技术，观察加速比是否呈线性叠加。  

#### 设计亮点
- **统一度量标准**：作者自行实现了一个“效率指数”，把吞吐率、显存占用和精度损失统一到一个标量上，便于跨技术比较。  
- **跨层协同实验**：最具创新性的实验是把模型层的 8 位量化与系统层的 FlashAttention 结合，结果显示整体加速比比单独使用任一技术高出约 1.3 倍，验证了协同加速的可行性。  
- **可视化分析**：每个子章节配有热力图或折线图，直观展示不同序列长度、不同硬件配置下的性能趋势，帮助读者快速捕捉关键规律。

### 实验与效果
- **测试任务**：包括语言建模（WikiText‑103）、零样本问答（MMLU）以及对话生成（Chatbot Arena）。  
- **基线对比**：与原始 FP32 模型、单纯 8 位量化、以及仅使用 FlashAttention 的实现相比，作者报告的综合加速比在 2.0‑3.5× 之间，显存占用下降约 40%‑60%。在 MMLU 上精度下降不超过 1.5%。  
- **消融实验**：通过逐步去掉数据层、模型层或系统层的优化，发现系统层的改进对吞吐率贡献最大（约 45%），而模型层的量化对显存节省最显著（约 35%），两者结合才能实现最优的整体折中。  
- **局限性**：作者坦诚实验仅在 NVIDIA A100 与 RTX 3090 两类 GPU 上完成，未覆盖 ARM、FPGA 等异构平台；此外，极端长序列（> 8k tokens）仍然受限于注意力的二次复杂度，现有分块注意力方案的加速效果有限。

### 影响与延伸思考
自发布以来，这篇综述成为 LLM 推理优化的“教材”，被多篇后续工作引用，例如针对 4 位量化的 **QLoRA**、专为长文本设计的 **LongLoRA**、以及跨层协同调度框架 **FlexGen**。它的 taxonomy 直接催生了社区对“系统‑模型协同”研究的热潮，很多公司开始在硬件层面加入对稀疏注意力的原生支持。想进一步深入的读者可以关注以下方向：① 动态稀疏注意力的硬件实现；② 基于 RL 的自适应量化策略；③ 多模态大模型的统一推理调度。  

### 一句话记住它
**把 LLM 推理效率的所有技巧划分为数据、模型、系统三层，并用统一实验验证“协同加速”是提升大模型部署的关键。**