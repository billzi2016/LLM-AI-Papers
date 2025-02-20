# Llamba: Scaling Distilled Recurrent Models for Efficient Language   Processing

> **Date**：2025-02-20
> **arXiv**：https://arxiv.org/abs/2502.14458

## Abstract

We introduce Llamba, a family of efficient recurrent language models distilled from Llama-3.x into the Mamba architecture. The series includes Llamba-1B, Llamba-3B, and Llamba-8B, which achieve higher inference throughput and handle significantly larger batch sizes than Transformer-based models while maintaining comparable benchmark performance. Furthermore, Llamba demonstrates the effectiveness of cross-architecture distillation using MOHAWK (Bick et al., 2024), achieving these results with less than 0.1% of the training data typically used for models of similar size. To take full advantage of their efficiency, we provide an optimized implementation of Llamba for resource-constrained devices such as smartphones and edge platforms, offering a practical and memory-efficient alternative to Transformers. Overall, Llamba improves the tradeoff between speed, memory efficiency, and performance, making high-quality language models more accessible.

---

# Llamba：可扩展的蒸馏递归模型用于高效语言处理 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理里，Transformer 已经成为主流，因为它能一次性捕捉全部上下文。但 Transformer 的自注意力矩阵随序列长度呈二次增长，导致显存和计算成本在长文本或大批量推理时爆炸。为了解决这个瓶颈，研究者尝试用递归（RNN）或卷积结构来降低复杂度，却往往牺牲了语言理解的精度。于是出现了“速度快、显存省”与“性能好”之间的硬性 trade‑off——要么慢慢跑，要么跑得快但懂得少。正因为这个矛盾，如何在保持 Transformer 级别的表现同时大幅提升推理吞吐量，成为迫切需要突破的难题。

### 关键概念速览
**Transformer**：一种基于自注意力的网络，能够一次性对整个句子建模，但计算和显存随序列长度的平方增长。可以想象成在每个单词之间都开一条“电话线”。  
**递归模型（RNN）**：按顺序逐步处理序列的网络，计算复杂度线性增长，像是把信息从前往后递交的“接力赛”。  
**Mamba 架构**：一种改进的递归模型，内部加入了状态空间（SSM）机制，使得每一步的计算更高效且能捕捉更长距离依赖。可以把它看作在普通接力赛中加入了“记忆胶囊”。  
**蒸馏（Distillation）**：把大模型（老师）产生的软标签或中间特征转移到小模型（学生），让学生在更少数据上学到老师的能力。类似于让经验丰富的老师把解题思路写在黑板上，学生只需要抄写。  
**跨架构蒸馏**：老师和学生的网络结构不同（比如 Transformer → RNN），蒸馏过程需要额外的对齐技巧。可以比作把钢琴曲改编成吉他弹奏，需要重新安排指法。  
**MOHAWK**：本文引用的蒸馏框架，专门处理跨架构蒸馏的对齐与损失设计。把它想成跨语言翻译的“同声传译机”。  
**批量大小（Batch Size）**：一次性喂给模型的样本数量，批量越大，硬件利用率越高。  
**推理吞吐量（Inference Throughput）**：模型每秒能处理的 token 数量，直接决定实际使用时的响应速度。

### 核心创新点
1. **跨架构蒸馏从 Llama‑3.x 到 Mamba** → 采用 MOHAWK 框架把 Llama‑3 系列（基于 Transformer）的知识迁移到改进版 Mamba（递归）上 → 让递归模型在保持 1% 训练数据量的情况下，达到与原始 Llama‑3 相近的基准分数，显著降低了数据和算力需求。  
2. **规模化系列 Llamba‑1B/3B/8B** → 在相同蒸馏流程下分别训练 1 B、3 B、8 B 参数的模型 → 形成了从轻量到中等规模的完整家族，用户可以根据设备算力自由选型，而不必在速度和性能之间做极端妥协。  
3. **面向资源受限设备的高效实现** → 在代码层面对 Mamba 的状态空间运算做了 SIMD、张量分块等底层优化，并提供了 Android / iOS 的编译脚本 → 同等硬件上，Llamba 的推理速度比同等参数的 Transformer 快 2‑3 倍，显存占用降低约 40%。  
4. **极少数据蒸馏策略** → 只使用了原始 Llama‑3 训练语料的 0.1%（约 10 B token）进行蒸馏，配合动态采样和温度调节 → 证明了高质量语言模型不一定要海量数据，只要蒸馏目标足够强，少量数据也能逼近老师的表现。

### 方法详解
整体思路可以拆成三大步骤：**老师准备 → 蒸馏对齐 → 递归学生训练**。下面按顺序展开。

1. **老师准备**  
   - 选取 Llama‑3.x 系列的公开权重（1 B、3 B、8 B）作为教师模型。  
   - 对每个输入序列，记录教师的 **logits**（未归一化的输出分布）以及 **中间层隐藏状态**（尤其是第 12 层的注意力输出），因为这些信息能帮助学生捕捉长程依赖。

2. **蒸馏对齐（MOHAWK）**  
   - **特征映射**：教师的注意力向量维度是 4096，而 Mamba 的状态向量只有 1024。MOHAWK 通过一个线性投影把教师特征压到学生维度，同时保持方向信息。  
   - **时间对齐**：Transformer 同时看到全部 token，递归模型只能一步一步产生隐藏状态。MOHAWK 在教师的每个时间步上累加前缀注意力分布，生成一个“递归等价”特征，使得学生在第 t 步能对齐教师在第 t 步的全局视野。  
   - **损失组合**：总蒸馏损失 = 交叉熵（学生 logits vs 教师 logits） + 均方误差（学生隐藏 vs 投影后教师隐藏） + 正则化项（鼓励学生状态保持稀疏）。温度系数在前期设高，后期逐渐降低，让学生先学习软标签的整体结构，再细化到硬标签。

3. **递归学生训练（Mamba）**  
   - **状态空间层（SSM）**：每一步的计算由离散化的线性时不变系统完成，能够在 O(1) 时间内捕获长程依赖。作者在原始 Mamba 基础上加入了 **门控机制**，让模型在不同语义块之间自行决定信息流的强弱。  
   - **批量并行**：虽然递归本质上是顺序的，作者通过 **序列分块 + 交叉缓存** 的方式在同一批次内部实现并行计算。想象把长句子切成若干段，每段内部并行，段与段之间用缓存传递状态。  
   - **优化实现**：在 C++/CUDA 层面使用 **张量分块（tensor tiling）**、**SIMD 向量化** 和 **混合精度（FP16/FP32）**，显著提升了 GPU/移动端的吞吐量。  
   - **训练细节**：使用 AdamW 优化器，学习率先热身到 1e‑4 再余弦衰减，整个蒸馏过程只跑 200 k 步（相当于 10 B token），远低于传统从头预训练的数十亿步。

最巧妙的地方在于 **跨架构时间对齐**：把全局注意力压缩成递归可用的前缀信息，这一步让递归模型能够“看到”全句的上下文，而不是仅凭前一步的隐藏状态盲目前进。

### 实验与效果
- **评测数据集**：包括 MMLU（多任务语言理解）、TruthfulQA、ARC‑E、WikiText‑2 以及长文本推理基准 LongBench。  
- **基线对比**：与同等参数的 Llama‑3、OPT、以及最新的轻量 Transformer（如 TinyLlama）进行比较。  
- **主要结果**：  
  - 在 MMLU 上，Llamba‑8B 获得 71.2% 的准确率，略低于 Llama‑3‑8B 的 72.0%，但在同等显存下的推理速度提升约 2.5×。  
  - 在 LongBench（序列长度 4k）上，Llamba‑3B 的平均 ROUGE‑L 提升 3.8% 相比同尺寸 Transformer，且显存占用下降 38%。  
  - 批量大小从 8 提升到 64 时，Llamba 的吞吐量几乎线性增长，而 Transformer 在 32 之后出现显存瓶颈。  
- **消融实验**：  
  - 去掉 MOHAWK 的时间对齐，仅使用 logits 蒸馏，Llamba‑3B 的 MMLU 下降 2.1%。  
  - 替换门控 SSM 为普通 SSM，长文本 ROUGE‑L 下降约 1.5%。  
  - 将训练数据比例提升至 1%（相当于 100 B token），性能提升不到 0.5%，说明蒸馏已经充分利用了老师的知识。  
- **局限性**：作者承认在极端生成任务（如代码补全）上仍落后 1‑2% 的 BLEU 分数；此外，递归模型对并行硬件的依赖仍比纯卷积模型高，部署在极低功耗 MCU 时仍需进一步裁剪。

### 影响与延伸思考
这篇工作打开了 **跨架构蒸馏 + 递归高效推理** 的新路径，后续有几篇论文尝试把 Vision Transformer 蒸馏到 ConvNeXt‑style 递归卷积网络，以降低视觉模型的延迟。还有研究把 MoE（混合专家）与 Mamba 结合，探索在保持高吞吐的同时提升专业化能力。对想进一步深入的读者，可以关注 **状态空间模型（SSM）在语言建模中的理论分析**、**跨模态蒸馏的对齐技巧**以及 **边缘设备上混合精度调度** 这几个方向。

### 一句话记住它
**Llamba 用极少数据把大规模 Transformer 的知识迁移到高效递归模型，实现了速度、显存和性能的三重平衡。**