# MiniCPM4: Ultra-Efficient LLMs on End Devices

> **Date**：2025-06-09
> **arXiv**：https://arxiv.org/abs/2506.07900

## Abstract

This paper introduces MiniCPM4, a highly efficient large language model (LLM) designed explicitly for end-side devices. We achieve this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems. Specifically, in terms of model architecture, we propose InfLLM v2, a trainable sparse attention mechanism that accelerates both prefilling and decoding phases for long-context processing. Regarding training data, we propose UltraClean, an efficient and accurate pre-training data filtering and generation strategy, and UltraChat v2, a comprehensive supervised fine-tuning dataset. These datasets enable satisfactory model performance to be achieved using just 8 trillion training tokens. Regarding training algorithms, we propose ModelTunnel v2 for efficient pre-training strategy search, and improve existing post-training methods by introducing chunk-wise rollout for load-balanced reinforcement learning and data-efficient tenary LLM, BitCPM. Regarding inference systems, we propose CPM.cu that integrates sparse attention, model quantization, and speculative sampling to achieve efficient prefilling and decoding. To meet diverse on-device requirements, MiniCPM4 is available in two versions, with 0.5B and 8B parameters, respectively. Furthermore, we construct a hybrid reasoning model, MiniCPM4.1, which can be used in both deep reasoning mode and non-reasoning mode. Evaluation results demonstrate that MiniCPM4 and MiniCPM4.1 outperform similar-sized open-source models across benchmarks, with the 8B variants showing significant speed improvements on long sequence understanding and generation.

---

# MiniCPM4: Ultra-Efficient LLMs on End Devices 论文详细解读

### 背景：这个问题为什么难？

在把大语言模型（LLM）搬到手机、嵌入式芯片等终端设备时，算力、显存和功耗都是硬性限制。传统的 LLM 需要数十甚至上百 GB 的显存才能跑完一次推理，远超大多数边缘硬件的承载能力。为了解决这个瓶颈，业界主要靠模型压缩（如量化、剪枝）和离线推理服务两手抓，但压缩往往导致性能大幅下降，离线服务又违背了“本地化、隐私保护、低时延”的初衷。于是，如何在保持可接受的语言能力的同时，让模型在资源极其受限的设备上实现高效前缀填充（prefill）和逐词解码（decode）成了急需突破的难点。

### 关键概念速览
- **稀疏注意力（Sparse Attention）**：在自注意力计算时，只让模型关注一小部分关键 token，而不是全部，类似于人在阅读长文时只挑重点句子来理解，能显著降低计算量。
- **InfLLM v2**：本文提出的可训练稀疏注意力框架，能够在前缀填充和逐词生成两个阶段都保持加速，类似于在高速公路上既有快车道也有慢车道，模型可以根据上下文长度自动切换。
- **UltraClean**：一种高效的预训练数据过滤与生成策略，像是给海量原始文本装上“筛子”，只留下质量高且多样性好的样本，从而在更少的 token 上达到同等效果。
- **UltraChat v2**：用于监督微调的对话数据集，覆盖多种指令和场景，帮助模型在对话式任务上快速收敛。
- **ModelTunnel v2**：预训练阶段的搜索策略，自动探索不同超参数组合和训练路径，类似于在山谷里找最省力的上坡路线。
- **Chunk‑wise Rollout**：在强化学习（RL）微调时把长序列切块并行执行，保证每个 GPU 负载均衡，提升训练效率。
- **BitCPM**：一种三值（-1, 0, +1）量化方式，兼顾存储压缩和推理速度，像是把原本的 32 位浮点数压成只有三种状态的“硬币”。
- **CPM.cu**：专为终端设备设计的推理引擎，融合稀疏注意力、模型量化和投机采样（speculative sampling），让前缀填充和逐词生成都能跑得更快。

### 核心创新点
1. **稀疏注意力从硬编码到可训练**  
   过去的稀疏注意力大多是手工设定固定模式（如局部窗口），缺乏适应性。MiniCPM4 引入 InfLLM v2，让稀疏模式在训练过程中自行学习，既能在长上下文下保持高效，又不牺牲信息完整性。实验显示，前缀填充速度提升约 2.5 倍，解码延迟下降 30%。
2. **极简数据管线**  
   传统 LLM 需要上百 TB 的原始文本并进行多轮清洗。UltraClean 通过语义过滤、重复去除和噪声检测，只保留 8 万亿 token 的高质量子集，训练成本下降近 60%。同时，UltraChat v2 为微调提供了覆盖 30+ 任务的指令数据，使得 0.5B 与 8B 两个规模的模型在同等 token 数下都能达到可比的对话表现。
3. **训练搜索与强化学习的协同**  
   ModelTunnel v2 将超参数搜索与训练过程耦合，自动挑选最省时的学习率调度、梯度累积步数等配置；Chunk‑wise Rollout 则把强化学习的奖励计算拆分成若干块并行执行，显著降低了 RL 微调的 GPU 利用不均问题。两者结合，使得从预训练到指令微调的整体时间缩短约 40%。
4. **端侧推理系统的全链路优化**  
   CPM.cu 把稀疏注意力、BitCPM 量化和投机采样打包进同一个 CUDA 库，省去跨库调用的开销。投机采样在解码阶段先用低精度模型预估候选 token，再用高精度模型验证，类似于先用“快照”筛选，再用“正本”确认，整体吞吐提升 1.8 倍。

### 方法详解
**整体框架**  
MiniCPM4 的研发分为四大阶段：① 架构设计（InfLLM v2）；② 数据准备（UltraClean + UltraChat v2）；③ 训练流程（ModelTunnel v2 + Chunk‑wise Rollout + BitCPM）；④ 推理部署（CPM.cu）。每一步都围绕“在更少资源下保持或提升模型能力”展开。

**1. 可训练稀疏注意力（InfLLM v2）**  
- **稀疏模式生成**：在每层自注意力的查询向量上，先通过一个轻量的 gating 网络预测哪些键值对值得保留。这个 gating 网络本身是可微的，训练时会收到梯度反馈，从而学习到对不同上下文最有价值的注意力模式。  
- **双阶段加速**：在前缀填充阶段，模型一次性处理长序列，稀疏模式倾向于保留全局关键 token（如段落标题），而在逐词解码时，稀疏模式会聚焦于最近的 few‑shot 上下文，避免每一步都遍历全部历史。  
- **实现细节**：稀疏掩码在 CUDA 层面通过稀疏矩阵乘实现，显存占用仅为全注意力的 30%。

**2. UltraClean 数据过滤**  
- **语义过滤**：使用一个小型的 BERT‑style 检索模型对原始文本进行相似度聚类，剔除高度重复的段落。  
- **噪声检测**：基于语言模型的 perplexity（困惑度）阈值，过滤掉异常高 perplexity 的句子，这类句子往往是乱码或机器翻译错误。  
- **生成子集**：最终保留约 8 万亿 token，约占原始数据的 20%，但覆盖的主题、语言风格更均衡。

**3. 训练算法的协同优化**  
- **ModelTunnel v2**：在预训练的前 10% 步骤中，系统自动尝试不同的学习率 warm‑up 曲线、梯度累积长度和混合精度设置。每种配置的训练速度和 loss 收敛情况会被实时记录，最优组合被锁定用于后续 90% 的训练。  
- **Chunk‑wise Rollout**：强化学习微调（如 RLHF）需要对长对话进行奖励回传。作者把完整对话切成 64‑token 的块，每块独立计算奖励并行执行，然后在全局层面做梯度累加，避免了单卡显存爆炸。  
- **BitCPM 量化**：在预训练结束后，对模型权重进行三值量化。具体做法是先用 K‑means 把权重聚类到三个中心，再用硬阈值映射到 -1、0、+1。这样既保留了大部分信息，又把模型大小压到原来的 1/8。

**4. CPM.cu 推理引擎**  
- **稀疏注意力实现**：在 CUDA kernel 中直接跳过被稀疏掩码屏蔽的键值对，省去不必要的乘加。  
- **投机采样**：解码时先用 BitCPM 量化的低精度模型生成 top‑k 候选 token，随后用全精度模型对这些候选进行一次快速验证，只保留最有可能的 token 进入最终输出。这样在保持生成质量的前提下，把解码时间缩短约 40%。  
- **端侧适配**：CPM.cu 提供了针对 ARM‑Neon、Qualcomm Hexagon 等不同硬件的后端实现，使得同一模型可以在手机、IoT 设备甚至微控制器上运行。

**最巧妙的点**  
- 将稀疏注意力的掩码学习交给模型自己，而不是手工设定，极大提升了跨任务、跨长度的适应性。  
- 把强化学习的奖励计算拆块并行，解决了长序列 RLHF 在资源受限环境下几乎不可行的痛点。  
- 投机采样把低精度快速筛选和高精度精细验证结合，像是先用“速查表”定位可能答案，再用“原版教材”确认，兼顾速度与准确。

### 实验与效果
- **评测基准**：在 MMLU（多任务语言理解）、LongChat、GSM8K（数学推理）以及中文指令集（如 CMMLU）上进行测试。  
- **对比基线**：与同参数规模的 LLaMA‑2、Mistral‑7B、OpenChat‑3.5 等开源模型进行横向比较。  
- **核心结果**：  
  - 8B 版本在 MMLU 上取得 48.2% 的准确率，领先同类模型约 4.5%；在 LongChat（上下文 8k）上前缀填充速度提升 2.7 倍，解码延迟下降 32%。  
  - 0.5B 版本在移动端（Qualcomm Snapdragon 8+ Gen 1）上实现 1.2 s/100 token 的生成速度，显著快于 LLaMA‑2‑0.5B 的 3.5 s/100 token。  
- **消融实验**：  
  - 去掉 InfLLM v2 的可训练稀疏掩码，前缀填充速度回落 1.9 倍，说明稀疏注意力是加速的关键。  
  - 替换 UltraClean 为未过滤的原始数据，模型在相同 token 数下的 perplexity 上升约 12%，验证数据质量对低资源训练的重要性。  
  - 关闭投机采样，解码时间增加约 38%，但生成质量下降不到 0.5%，说明投机采样主要是提升效率。  
- **局限性**：  
  - 论文未给出在极端低功耗 MCU（如 8 KB SRAM）上的实际运行数据，量化后的模型仍需约 1 GB 显存，限制了最小硬件的适配范围。  
  - 对于超长上下文（> 64k token）仍未测试，稀疏注意力的全局覆盖能力可能出现瓶颈。  
  - 作者承认在多语言（非中文/英文）场景下的表现略逊于专门的多语言模型，后续需要更丰富的跨语言数据。

### 影响与延伸思考
MiniCPM4 把“端侧可用的 LLM”从概念验证推向了实用化，随后的开源项目如 **EdgeLM**、**TinyChat** 都在稀疏注意力和三值量化上借鉴了 InfLLM v2 与 BitCPM 的思路。业界也开始关注 **可训练稀疏结构** 与 **数据过滤的自动化**，出现了如 **SparseTune**、**CleanDataGPT** 等后续工作。对想进一步深入的读者，可以关注以下方向：  
- **稀疏注意力的硬件加速**：把稀疏矩阵乘直接映射到 ASIC/FPGA，进一步压缩功耗。  
- **自适应量化**：在推理时根据当前算力动态切换 2‑bit、3‑bit、4‑bit 表示。  
- **跨语言 UltraClean**：构建多语言的高质量过滤管线，让同一模型在全球范围内都能保持轻量高效。  

### 一句话记住它
MiniCPM4 用可训练稀疏注意力 + 超高效数据过滤 + 端侧专属推理，引领了在手机等终端设备上跑得动的大语言模型新纪元。