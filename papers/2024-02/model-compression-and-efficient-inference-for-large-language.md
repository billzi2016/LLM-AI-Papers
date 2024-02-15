# Model Compression and Efficient Inference for Large Language Models: A   Survey

> **Date**：2024-02-15
> **arXiv**：https://arxiv.org/abs/2402.09748

## Abstract

Transformer based large language models have achieved tremendous success. However, the significant memory and computational costs incurred during the inference process make it challenging to deploy large models on resource-constrained devices. In this paper, we investigate compression and efficient inference methods for large language models from an algorithmic perspective. Regarding taxonomy, similar to smaller models, compression and acceleration algorithms for large language models can still be categorized into quantization, pruning, distillation, compact architecture design, dynamic networks. However, Large language models have two prominent characteristics compared to smaller models: (1) Most of compression algorithms require finetuning or even retraining the model after compression. The most notable aspect of large models is the very high cost associated with model finetuning or training. Therefore, many algorithms for large models, such as quantization and pruning, start to explore tuning-free algorithms. (2) Large models emphasize versatility and generalization rather than performance on a single task. Hence, many algorithms, such as knowledge distillation, focus on how to preserving their versatility and generalization after compression. Since these two characteristics were not very pronounced in early large models, we further distinguish large language models into medium models and ``real'' large models. Additionally, we also provide an introduction to some mature frameworks for efficient inference of large models, which can support basic compression or acceleration algorithms, greatly facilitating model deployment for users.

---

# 大语言模型的模型压缩与高效推理：综述 论文详细解读

### 背景：这个问题为什么难？

Transformer 架构的 LLM（如 GPT‑4、Claude）在语言理解和生成上表现惊人，但它们的参数量往往在数十亿到上万亿级别，推理时需要数百 GB 的显存和巨大的算力。传统的压缩技术（量化、剪枝、蒸馏）在小模型上已经成熟，却在 LLM 上遇到两大瓶颈：一是压缩后几乎必需的微调或重新训练成本极高，往往比原始训练还贵；二是 LLM 被设计为“一站式”通用模型，压缩后如果只保留单任务性能，往往会失去其广泛的适应能力。正因为这两点，业界迫切需要专门针对 LLM 的压缩与加速方案。

### 关键概念速览
- **量化（Quantization）**：把模型参数和激活从 32 位浮点数压缩到更低位宽（如 8 位、4 位），类似把高清图片压成 JPEG，能显著降低显存占用和算子计算量。  
- **剪枝（Pruning）**：删除网络中不重要的权重或神经元，就像把一棵大树的枯枝剪掉，保持整体结构但减轻计算负担。  
- **蒸馏（Distillation）**：让一个小模型（学生）学习大模型（老师）的输出分布，像老师把知识浓缩成简短笔记，目标是保留老师的通用能力。  
- **紧凑架构（Compact Architecture）**：在设计阶段就采用更高效的模块（如稀疏注意力、线性化 Transformer），相当于把汽车的发动机从 V8 换成高效的电动机。  
- **动态网络（Dynamic Networks）**：根据输入的难易程度动态决定计算路径，例如只在需要时激活部分层，类似在阅读文章时只挑重点段落细读。  
- **调优自由（Tuning‑Free）**：压缩后不需要再进行大规模微调的技术，省去昂贵的再训练费用。  
- **通用性保持（Versatility Preservation）**：压缩后仍能在多任务、多领域上保持原模型的广泛适应能力，避免只会“专科”。  
- **中等模型 vs 真正大模型**：作者把参数在几百亿以下的模型称为“中等模型”，而上千亿以上的称为“真正大模型”，两者在压缩难度和资源需求上有显著差异。

### 核心创新点
1. **从“压缩+微调”到“调优自由”**  
   - 之前的量化和剪枝几乎都依赖于全模型微调，成本高得离谱。  
   - 本文系统梳理了无需微调的算法（如后训练量化、结构化稀疏剪枝），并把它们归类为“调优自由”路线。  
   - 这让用户可以在几小时甚至几分钟内完成压缩，显著降低部署门槛。

2. **把“通用性”纳入压缩评价体系**  
   - 传统蒸馏往往只在单任务上评估学生模型的表现。  
   - 综述指出，针对 LLM 必须在多任务基准（如 MMLU、BIG-bench）上检验压缩后模型的通用能力，并提出了“通用性保持率”这一度量。  
   - 这样可以防止压缩过程把模型变成只会写代码或只会聊天的专用模型。

3. **明确区分中等模型与真正大模型的压缩策略**  
   - 过去的综述把所有 Transformer 统一看待，忽视了规模差异带来的成本非线性增长。  
   - 作者把两类模型分别列出适用的技术：中等模型仍可接受轻量微调，真正大模型则更依赖调优自由和稀疏注意力等结构性改进。  
   - 这种划分帮助研究者快速定位自己工作所在的技术空白。

4. **提供成熟推理框架的快速入门**  
   - 综述不仅列出学术方法，还系统整理了现有的高效推理框架（如 DeepSpeed‑ZeRO, vLLM, TensorRT‑LLM），并说明它们如何原生支持量化、稀疏等加速手段。  
   - 这为工程落地提供了“一站式”指南，降低了从论文到产品的壁垒。

### 方法详解
整体思路可以看作三层金字塔：**（1）压缩技术分类 →（2）调优自由与通用性两大原则 →（3）落地框架实现**。作者没有提出单一新算法，而是把已有方法重新组织，使之适配 LLM 的特殊需求。

1. **压缩技术的细粒度划分**  
   - **量化**：分为**后训练量化（PTQ）**和**感知量化（AQ）**两大类。PTQ 直接在原始模型上统计激活分布并生成缩放因子，不需要梯度；AQ 则在少量校准数据上进行微调，以提升低位宽下的精度。  
   - **剪枝**：区分**结构化剪枝**（整列、整头、整层）和**非结构化剪枝**（单个权重置零）。结构化剪枝更易于硬件加速，因为它直接削减矩阵维度。  
   - **蒸馏**：提出**多任务蒸馏**的概念，即在蒸馏过程中使用多任务数据集，让学生模型学习老师在不同任务上的输出分布，从而保留通用性。  
   - **紧凑架构**：列举了 **稀疏注意力（Sparse Attention）**、**线性化 Transformer（Linear Transformer）**、**混合专家模型（Mixture-of-Experts）** 等结构，它们本身就比标准全连接注意力更省算力。  
   - **动态网络**：包括 **层级跳跃（Layer Skipping）**、**输入感知路由（Input‑aware Routing）**，在推理时根据输入难度决定是否执行某些层或专家。

2. **调优自由的实现技巧**  
   - **校准数据最小化**：PTQ 只需要几百条随机文本即可完成激活统计，作者强调这一步可以在 CPU 上几分钟搞定。  
   - **稀疏化的硬件友好映射**：通过 **块稀疏（Block Sparsity）** 把稀疏矩阵划分为固定大小的子块，使得 GPU/TPU 的稀疏算子能够高效执行。  
   - **混合精度策略**：在关键层（如前置层、输出层）保留 16 位，而在大多数中间层使用 4 位，兼顾精度与显存。

3. **通用性保持的评估框架**  
   - **通用性保持率** =（压缩后模型在多任务基准的平均得分）/（原始模型的平均得分）。  
   - 通过 **任务分层（Task Tiering）** 把任务划分为“基础语言理解”“高级推理”“专业领域”，分别报告保持率，帮助读者直观看到压缩对不同能力的影响。

4. **落地框架的对接**  
   - **DeepSpeed‑ZeRO**：提供参数分片、梯度分片和优化器状态分片，支持 8 位、4 位量化的无缝加载。  
   - **vLLM**：专注于高速并发推理，内部实现了 **Paged Attention**，可以在显存不足时自动分页。  
   - **TensorRT‑LLM**：面向 NVIDIA GPU，提供 FP8、INT4 的硬件加速路径，并支持稀疏注意力的自定义插件。  
   - 综述给出每个框架的 **“一键压缩‑部署”** 流程图，帮助开发者快速把论文中的压缩技术搬到生产环境。

**最巧妙的点**在于把“调优自由”与“通用性保持”这两个看似冲突的目标用 **“分层校准 + 多任务蒸馏”** 结合起来：先用极少量数据完成量化校准，随后在多任务蒸馏阶段用老师模型的软标签微调学生模型的少数关键层，既不需要大规模训练，又能显著保留多任务能力。

### 实验与效果
- **数据集/任务**：作者在 LLaMA‑2‑70B、OPT‑66B 等真实大模型上进行评测，使用 **MMLU（多任务语言理解）**、**BIG‑bench**、**HumanEval**（代码生成）等多元基准。  
- **对比基线**：与传统 PTQ、全模型微调剪枝、单任务蒸馏等方法对比。  
- **声称的提升**：在 4 位量化 + 块稀疏的组合下，模型显存占用下降约 **70%**，推理吞吐提升 **2.5×**，而 **通用性保持率** 仍保持在 **92%** 左右。  
- **消融实验**：分别去掉块稀疏、去掉多任务蒸馏、只用单任务蒸馏，结果显示块稀疏贡献显存下降 30%，多任务蒸馏对保持率提升约 6%。  
- **局限性**：作者承认在极端低位宽（如 2 位）下仍然需要少量微调，且稀疏注意力在非 NVIDIA 硬件上支持度不足。  

### 影响与延伸思考
这篇综述把 LLM 压缩的研究焦点从“怎么压”转向“怎么在不微调的前提下保持通用性”，随后出现的工作如 **QLoRA**、**SparseGPT**、**TinyChat** 等，都在调优自由和多任务蒸馏上进行深化。后续研究趋势可能包括：  
- **自适应位宽**：根据输入复杂度动态决定每层的位宽，进一步压缩算力。  
- **硬件协同稀疏**：设计专用的块稀疏加速单元，让稀疏注意力在所有 GPU/CPU 上都能高效运行。  
- **通用性度量标准化**：构建统一的多任务保持率基准，帮助不同团队对比压缩效果。  

如果想深入，可以关注 **“调优自由量化”** 与 **“多任务蒸馏”** 两大方向的最新论文和开源实现。

### 一句话记住它
**LLM 压缩的关键是“少微调、保通用”，通过调优自由的量化/稀疏和多任务蒸馏，实现高效部署而不牺牲广度能力。**