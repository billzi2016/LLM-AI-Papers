# ChartMoE: Mixture of Diversely Aligned Expert Connector for Chart   Understanding

> **Date**：2024-09-05
> **arXiv**：https://arxiv.org/abs/2409.03277

## Abstract

Automatic chart understanding is crucial for content comprehension and document parsing. Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in chart understanding through domain-specific alignment and fine-tuning. However, current MLLMs still struggle to provide faithful data and reliable analysis only based on charts. To address it, we propose ChartMoE, which employs the Mixture of Expert (MoE) architecture to replace the traditional linear projector to bridge the modality gap. Specifically, we train several linear connectors through distinct alignment tasks, which are utilized as the foundational initialization parameters for different experts. Additionally, we introduce ChartMoE-Align, a dataset with nearly 1 million chart-table-JSON-code quadruples to conduct three alignment tasks (chart-table/JSON/code). Combined with the vanilla connector, we initialize different experts diversely and adopt high-quality knowledge learning to further refine the MoE connector and LLM parameters. Extensive experiments demonstrate the effectiveness of the MoE connector and our initialization strategy, e.g., ChartMoE improves the accuracy of the previous state-of-the-art from 80.48\% to 84.64\% on the ChartQA benchmark.

---

# ChartMoE：多样化对齐专家连接器的混合模型用于图表理解 论文详细解读

### 背景：这个问题为什么难？

图表是信息密度极高的视觉载体，机器要把图片里的坐标、颜色、文字映射成结构化数据并进行推理，涉及跨模态的语义对齐。早期的多模态大语言模型（MLLM）往往只在大规模图文对上做统一的线性投影，缺乏针对图表特殊结构的细粒度对齐，导致在仅凭图表就要给出数值答案或分析时容易出现“幻觉”。此外，现有模型在训练时只用单一的对齐目标（比如图像‑文本），难以同时学习图表‑表格、图表‑JSON、图表‑代码等多种关系，导致在真实文档中遇到复杂表格或代码生成任务时表现不稳。正是这些根本性的对齐瓶颈，让提升图表理解的可靠性成为迫切需求。

### 关键概念速览
- **多模态大语言模型（MLLM）**：把视觉特征和文字特征统一进同一个语言模型的框架里，类似于把“看”和“说”两种能力装进同一台机器。
- **投影器（projector）**：把视觉特征向量映射到语言模型的隐藏空间，就像把一张图片的颜色信息翻译成文字的语义代码。
- **混合专家模型（Mixture of Experts, MoE）**：由多个“专家”子网络组成，输入会根据某种路由机制分配给最擅长的专家，类似于公司里不同部门分别处理各自擅长的事务。
- **对齐任务**：让模型学习两种模态之间的对应关系，例如把图表对应到表格、对应到JSON结构或对应到可执行代码。
- **ChartMoE‑Align 数据集**：近百万条“图表‑表格‑JSON‑代码”四元组，用来分别训练三种对齐任务，提供了丰富的跨模态监督信号。
- **路由权重（routing weight）**：决定哪个专家在当前输入上发挥作用的系数，类似于在会议中决定哪个专家发言的投票结果。

### 核心创新点
1. **线性投影 → MoE 投影**  
   传统做法是用单一的线性层把视觉特征映射到语言模型空间，这相当于让所有图表都走同一条路。ChartMoE 把这个投影层换成 MoE 结构，让不同的专家专门负责不同的对齐任务。这样，图表‑表格、图表‑JSON、图表‑代码的映射可以在各自最擅长的子网络里完成，提升了跨模态的表达能力。

2. **多任务初始化 → 多样化专家**  
   直接随机初始化 MoE 的专家会导致各专家之间差异不大，路由机制难以发挥作用。作者先分别在三种对齐任务上训练独立的线性连接器，然后把这些训练好的参数作为对应专家的初始权重。相当于让每个专家一开始就“懂”一种语言，从而在后续联合训练中保持多样性。

3. **ChartMoE‑Align 数据集的构建**  
   为了让模型看到足够丰富的跨模态对应，团队自行收集并生成了近 100 万条图表‑表格‑JSON‑代码四元组。每条数据同时提供了三种对齐标签，使得模型可以在同一次前向传播中学习多任务对齐，显著提升了对图表细节的捕捉能力。

4. **高质量知识学习的双向微调**  
   在 MoE 初始化完成后，作者并没有只微调 MoE 本身，而是同步对大型语言模型（LLM）的参数进行微调，形成“投影‑语言模型”双向协同。这样，语言模型可以适应更精准的视觉特征输入，视觉投影也能得到语言侧的反馈，整体系统的鲁棒性得到提升。

### 方法详解
整体思路可以拆成三大步骤：①准备多任务对齐数据，②用单任务线性连接器初始化 MoE 专家，③在联合任务上微调 MoE 与 LLM。下面逐层展开。

1. **数据层**  
   ChartMoE‑Align 包含四个元素：原始图表图片、对应的结构化表格、等价的 JSON 描述以及可直接执行的代码片段。构造过程先从公开的图表库抽取图片，再利用 OCR 与表格解析工具生成表格，随后用规则或模型把表格转成 JSON，最后用模板把 JSON 编译成代码。每条记录因此拥有三套对齐目标，提供了丰富的监督信号。

2. **单任务线性连接器的训练**  
   对每一种对齐任务（图表→表格、图表→JSON、图表→代码），分别训练一个只有一个线性层的投影器。输入是图表的视觉特征（如 CLIP‑ViT 提取），输出直接映射到目标空间的向量。训练目标是最小化投影后向量与目标向量的距离（如 MSE），这一步相当于让模型学会“把图表翻译成表格”或“把图表翻译成代码”。

3. **MoE 投影器的构建**  
   MoE 由 N（论文中 N≈8）个专家组成，每个专家本质上是一个线性层。作者把前一步得到的三个单任务线性层的参数分别复制到三个专家里，剩余的专家保持随机初始化。这样，系统里已经有了“表格专家”“JSON 专家”“代码专家”，其余专家则可以在后续训练中学习到更细粒度的特征。

4. **路由机制**  
   对每个输入图表，系统先通过一个轻量的路由网络（通常是两层 MLP）计算每个专家的权重，这些权重在 0‑1 之间并归一化。随后，所有专家的输出按照权重加权求和，得到最终的投影向量。路由网络的目标是让输入自动匹配最合适的专家，类似于把不同类型的图表交给最擅长的翻译员。

5. **联合微调**  
   投影向量送入预训练的大语言模型（如 LLaMA）进行下游任务的生成或问答。整个系统的损失由三部分组成：①对齐损失（保持投影与目标的对应），②语言模型的生成损失（如交叉熵），③路由正则化（鼓励专家使用均衡）。在梯度回传时，MoE 的专家参数、路由网络以及 LLM 参数都会同步更新，实现“投影‑语言模型”协同学习。

**最巧妙的点**在于把单任务线性投影的参数直接搬进 MoE 专家，使得每个专家一开始就拥有明确的功能定位，避免了 MoE 常见的“专家塌陷”（所有输入都走同一个专家）问题。

### 实验与效果
- **测试数据**：主要在 ChartQA 基准上评估，ChartQA 是一个包含真实文档中图表问答的公开数据集。还使用了自建的 ChartEval 小样本集验证代码生成质量。
- **对比基线**：与之前的最强模型（基于单线性投影的 MLLM）以及几种常规 MoE 变体进行比较。ChartMoE 将准确率从 80.48% 提升到 84.64%，提升幅度约 4.2%。
- **消融实验**：作者分别去掉（1）多任务初始化、（2）路由网络、（3）联合微调，仅保留单一线性投影。结果显示，去掉任何一环都会导致整体性能下降 2‑3% 之间，说明每个模块都对最终提升有贡献。
- **局限性**：论文指出，MoE 的路由开销在大规模推理时仍然比纯线性投影高，且对显存要求更大；此外，ChartMoE‑Align 虽然规模大，但仍偏向于结构化、规则化的图表，面对手绘或极度噪声的图表时表现未必理想。

### 影响与延伸思考
ChartMoE 把 MoE 思想成功搬进了跨模态投影层，打开了“多任务专家初始化”在视觉‑语言对齐中的新路径。后续工作已经开始探索把更复杂的非线性专家（如小型 Transformer）加入投影层，或在路由阶段加入图表类型的显式标签，以进一步提升专业化程度。对想深入的读者，可以关注以下方向：①更高效的 MoE 路由算法（如稀疏注意力），②跨模态对齐的自监督预训练（利用未标注图表），③把 ChartMoE 思路推广到其他结构化视觉任务（如流程图、思维导图）上。

### 一句话记住它
把图表的多种跨模态映射分别交给“专门的翻译员”，再让它们一起协作，ChartMoE 用 MoE 投影把图表理解的准确率直接推上新高度。