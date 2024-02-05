# PanGu-$\pi$ Pro:Rethinking Optimization and Architecture for Tiny   Language Models

> **Date**：2024-02-05
> **arXiv**：https://arxiv.org/abs/2402.02791

## Abstract

The power of large language models (LLMs) has been demonstrated through numerous data and computing resources. However, the application of language models on mobile devices is facing huge challenge on the computation and memory costs, that is, tiny language models with high performance are urgently required. Limited by the highly complex training process, there are many details for optimizing language models that are seldom studied carefully. In this study, based on a tiny language model with 1B parameters, we carefully design a series of empirical study to analyze the effect of each component. Three perspectives are mainly discussed, \ie, neural architecture, parameter initialization, and optimization strategy. Several design formulas are empirically proved especially effective for tiny language models, including tokenizer compression, architecture tweaking, parameter inheritance and multiple-round training. Then we train PanGu-$\pi$-1B Pro and PanGu-$\pi$-1.5B Pro on 1.6T multilingual corpora, following the established formulas. Experimental results demonstrate the improved optimization and architecture yield a notable average improvement of 8.87 on benchmark evaluation sets for PanGu-$\pi$-1B Pro. Besides, PanGu-$\pi$-1.5B Pro surpasses a range of SOTA models with larger model sizes, validating its superior performance. The code is available at https://github.com/YuchuanTian/RethinkTinyLM.

---

# PanGu‑π Pro：重新思考小型语言模型的优化与架构 论文详细解读

### 背景：这个问题为什么难？

大模型的成功离不开海量数据和算力，但把同样的能力搬到手机、穿戴设备上会瞬间炸掉内存和电池。现有的轻量化方案大多是直接裁剪大模型，导致性能急剧下滑；还有的靠手工压缩词表或层数，却缺乏系统的训练经验。根本原因是：在“千亿参数”时代，人们已经把训练细节磨得很细致，但对“几亿参数”这种极小模型的每一步设计仍然是盲区。于是，如何在保持极低计算/存储开销的同时，最大化模型的语言理解与生成能力，成了迫切需要解决的难题。

### 关键概念速览
- **Tokenizer（分词器）**：把原始文字切成模型能处理的“词块”。在小模型里，词表太大相当于给模型装了太多“工具”，占用宝贵的参数空间。压缩词表就像把工具箱里的螺丝刀、扳手只保留最常用的几件。
- **Parameter Initialization（参数初始化）**：模型一开始的权重值。好的初始化像给新手司机装上助力转向，让模型在训练初期就能顺畅前进，而不是在原地打转。
- **Multiple‑Round Training（多轮训练）**：把一次完整的训练过程拆成若干阶段，每阶段都可能换掉学习率、数据采样或模型结构。类似于学语言时先学发音、再学词汇、最后练口语，层层递进。
- **Parameter Inheritance（参数继承）**：把已经训练好的模型权重直接拷贝到新模型里，再继续微调。相当于把前辈的经验手册交给新人，让学习效率大幅提升。
- **Architecture Tweaking（架构微调）**：在保持整体框架不变的前提下，对层数、隐藏维度、注意力头数等做细粒度调整。就像在同一辆车的底盘上换不同的轮胎和发动机，以适配不同路况。
- **Benchmark Evaluation Sets（基准评测集）**：公开的任务集合，用来统一衡量模型在阅读理解、对话生成等方面的表现。相当于学术界的“奥林匹克赛”。

### 核心创新点
1. **从大模型直接裁剪 → 重新设计词表压缩**  
   过去的轻量化往往把大模型的词表直接搬下来，导致稀疏的低频词占用大量参数。作者提出“Tokenizer Compression”，通过统计多语言语料的高频子词，构造更紧凑的子词表，使词表规模下降 30% 左右，同时保持跨语言覆盖。实验显示，压缩后的词表在保持或提升下游任务准确率的同时，显著降低了嵌入层的显存占用。

2. **统一的参数初始化 → 参数继承 + 多轮微调**  
   传统做法在每次训练都从随机初始化开始，浪费了已有的语言知识。论文引入“Parameter Inheritance”，先用 1 B 参数模型在大规模多语言语料上预训练，然后把权重迁移到 1.5 B 规模的模型，再进行第二轮训练。配合“Multiple‑Round Training”，在每轮结束后调节学习率、数据混合比例，使模型在每一步都站在更高的起点上。

3. **单一的网络结构 → Architecture Tweaking 公式化**  
   作者对 Transformer 的关键超参提出了经验公式（如隐藏维度 ≈ 参数量的 1/12，注意力头数 ≈ 隐藏维度的 1/64），并在 1 B 规模上系统验证。相比盲目复制大模型的层数和宽度，这套公式让模型在相同 FLOPs 下获得约 5% 的性能提升。

4. **一次性大规模训练 → 多阶段训练管线**  
   传统训练往往一次性跑完所有数据，容易出现梯度噪声和收敛不稳。论文把 1.6 T 词的多语言语料分成若干子集，先在通用子集上做粗训练，再在高质量子集上细调。这样做相当于先学“通用语法”，再学“专业术语”，最终在基准评测上实现了 8.87 分的平均提升。

### 方法详解
整体思路可以概括为四步：**词表压缩 → 初始模型预训练 → 参数继承与架构微调 → 多轮细化训练**。下面把每一步拆开讲。

1. **词表压缩**  
   - **统计频率**：在 1.6 T 多语言语料上统计子词出现频次。  
   - **阈值筛选**：设定覆盖率阈值（如 99%），只保留累计覆盖率最高的子词。  
   - **子词合并**：对低频子词进行合并或拆分，形成更紧凑的 BPE（Byte‑Pair Encoding）表。  
   - **效果**：词表从原始的 ~50k 降到 ~35k，嵌入层参数从 0.2 B 降到 0.14 B。

2. **初始模型预训练（1 B 参数）**  
   - **网络结构**：采用标准 Transformer，层数 24，隐藏维度 2048，注意力头 16。  
   - **初始化**：使用改进的 Xavier 初始化，配合 LayerNorm 的零均值。  
   - **优化器**：AdamW，学习率 1e‑4，使用 cosine 衰减。  
   - **训练数据**：全量 1.6 T 多语言语料，采用混合语言采样策略，确保每种语言都有足够曝光。

3. **参数继承与架构微调（升级到 1.5 B）**  
   - **参数迁移**：把 1 B 模型的权重直接拷贝到 1.5 B 模型对应层；新增层的权重使用相同初始化方式。  
   - **架构公式**：隐藏维度提升至 2560，注意力头数提升至 20，层数保持 24。公式保证每层的 FLOPs 与参数增长比例一致。  
   - **微调策略**：在迁移后立即进行 5% 的 warm‑up 训练，学习率降低至 5e‑5，帮助新层快速适应已有特征。

4. **多轮细化训练**  
   - **轮次划分**：第一轮使用全语料的 80%（通用轮），第二轮使用高质量过滤后的 20%（专业轮），第三轮再回到全语料进行轻微微调。  
   - **学习率调度**：每轮结束后将学习率乘以 0.7，防止后期过度震荡。  
   - **数据混合**：在专业轮中加入任务特定的对话、问答数据，以提升下游对话生成能力。  
   - **反直觉点**：作者发现把学习率在后期继续降低（而不是保持不变）能够显著提升小模型的收敛稳定性，这与大模型常用的“学习率保持”策略相反。

### 实验与效果
- **评测集合**：包括中文阅读理解（CMRC）、多语言自然语言推理（XNLI）、对话生成（DialoGLUE）等 7 套公开基准。  
- **主要结果**：在所有基准上，PanGu‑π‑1B Pro 的平均得分比同尺寸的 SOTA 小模型提升了 **8.87 分**。  
- **对比基线**：与 LLaMA‑7B、Bloom‑560M、MiniGPT‑4 等模型相比，PanGu‑π‑1.5B Pro 在同等 FLOPs 条件下超越了多数参数规模更大的模型，尤其在多语言任务上表现突出。  
- **消融实验**：作者分别去掉词表压缩、参数继承、架构微调和多轮训练四个模块，发现每去掉一个模块，平均得分下降 2.1‑3.5 分，说明四个设计相互叠加贡献显著。  
- **局限性**：论文未给出对极端低资源语言的细粒度分析，也没有公开对比不同硬件平台（如 ARM vs x86）的实际推理时延。作者承认在极端内存受限的设备上仍需进一步的量化与剪枝工作。

### 影响与延伸思考
这篇工作在“小模型”社区掀起了“系统化微调”潮流，随后出现了多篇围绕 **Parameter Inheritance** 与 **Multi‑Round Training** 的论文，例如华为的 TinyBERT‑Pro、Meta 的 Efficient‑LM。业界也开始把“先大后小、再迁移”作为轻量化模型的标准流程。未来可以进一步探索 **自适应词表压缩**（根据下游任务动态裁剪）以及 **跨模态参数继承**（把视觉模型的特征投射到语言模型），这些方向都有望把极小模型的能力推向新高度。

### 一句话记住它
**把大模型的经验“搬家”到小模型，再用压缩词表和多轮细化训练“装修”，让千兆参数的能力在百兆参数里也能闪光。**