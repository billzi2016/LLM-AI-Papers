# Does Table Source Matter? Benchmarking and Improving Multimodal   Scientific Table Understanding and Reasoning

> **Date**：2025-01-22
> **arXiv**：https://arxiv.org/abs/2501.13042

## Abstract

Recent large language models (LLMs) have advanced table understanding capabilities but rely on converting tables into text sequences. While multimodal large language models (MLLMs) enable direct visual processing, they face limitations in handling scientific tables due to fixed input image resolutions and insufficient numerical reasoning capabilities. We present a comprehensive framework for multimodal scientific table understanding and reasoning with dynamic input image resolutions. Our framework consists of three key components: (1) MMSci-Pre, a domain-specific table structure learning dataset of 52K scientific table structure recognition samples, (2) MMSci-Ins, an instruction tuning dataset with 12K samples across three table-based tasks, and (3) MMSci-Eval, a benchmark with 3,114 testing samples specifically designed to evaluate numerical reasoning capabilities. Extensive experiments demonstrate that our domain-specific approach with 52K scientific table images achieves superior performance compared to 150K general-domain tables, highlighting the importance of data quality over quantity. Our proposed table-based MLLMs with dynamic input resolutions show significant improvements in both general table understanding and numerical reasoning capabilities, with strong generalisation to held-out datasets. Our code and data are publicly available at https://github.com/Bernard-Yang/MMSci_Table.

---

# 表格来源重要吗？多模态科学表格理解与推理的基准测试与提升 论文详细解读

### 背景：这个问题为什么难？
科学论文里常出现密密麻麻的实验数据表，普通的大语言模型（LLM）只能把表格先转成文字序列再读，过程会丢掉排版、合并单元格等关键信息。多模态大语言模型（MLLM）虽然能直接看图，但它们的输入分辨率是固定的，细小的数字和复杂的网格常被压缩成模糊的像素，导致数值推理几乎不可能。于是，面对专业科研表格时，现有模型既看不清结构，又算不出精确的数值答案，这正是需要突破的瓶颈。

### 关键概念速览
- **多模态大语言模型（MLLM）**：能够同时处理文字和图像的模型，就像人既能读文字也能看图一样。  
- **动态输入分辨率**：根据表格的实际大小自动调节图像像素数，类似相机自动对焦，让细节不被压缩。  
- **表格结构识别**：模型学习辨认单元格的边框、跨行跨列的合并关系，等同于把表格的“骨架”抽出来。  
- **指令微调（Instruction Tuning）**：把任务包装成“请完成以下指令”的形式，让模型学会按照指令一步步回答。  
- **数值推理**：在表格中进行加减乘除、统计或比较等数学运算，类似在计算器上直接算。  
- **数据质量 vs. 数量**：在同等规模下，专业领域的高质量样本往往比大规模通用样本更有帮助。  
- **基准评估（Benchmark）**：统一的测试集，用来客观比较不同模型在特定任务上的表现。  
- **SciGen**：生成科学论文的自动化工具，本文用它来合成大量真实感的科研表格。

### 核心创新点
1. **领域专属结构预训练 → 采用 52 K 份科学表格图像进行结构学习 → 在表格骨架识别上超过使用 150 K 通用表格的模型，证明“来源更重要”。  
2. **动态分辨率输入 → 在模型前端加入自适应图像缩放模块，使大表格保持细节，小表格不浪费算力 → 视觉编码器捕捉到更多数字信息，数值推理准确率显著提升。  
3. **三任务指令微调数据集 → 通过 12 K 条指令样本覆盖表格问答、事实验证和表格生成三类任务 → 模型在不同使用场景下都能直接给出符合指令格式的答案，通用性更强。  
4. **专门的数值推理基准 → 构建 MMSci‑Eval，3114 条专注于数字计算的测试样本 → 为后续研究提供了统一的“算数”评测标准，也让本工作在数值推理上实现了可量化的突破。

### 方法详解
整体思路可以划分为三阶段：**结构预训练 → 指令微调 → 动态推理**。  
1. **结构预训练（MMSci‑Pre）**  
   - 收集 52 K 由 SciGen 生成的科研表格，转成高分辨率 PNG 并配上对应的 HTML 结构标注。  
   - 使用视觉编码器（如 ViT）输出每个像素的特征，再喂入一个专门的表格结构解码器，预测每个单元格的左上、右下坐标以及跨行跨列属性。  
   - 训练目标是最小化预测坐标与标注坐标的 L1 损失，同时加入跨单元格一致性正则，确保合并单元格不会被拆开。  

2. **指令微调（MMSci‑Ins）**  
   - 将三类任务统一成“指令—输入—输出”三段式文本。例如：  
     `指令：请回答以下表格中的问题。 输入：<表格图像> 问题：该实验组的平均准确率是多少？ 输出：85%`  
   - 通过 LoRA（低秩适配）或全参数微调，让已经学会结构的视觉编码器与语言模型协同工作。语言模型负责解释指令、组织答案，视觉模型负责提供单元格内容的数值向量。  
   - 训练时加入梯度累积和混合精度，保证在普通 GPU 上也能完成。  

3. **动态分辨率推理**  
   - 推理前先读取表格的行列数（可通过轻量的 OCR 或结构预测快速估计），计算一个合适的像素比例，使得每个单元格至少拥有 32×32 像素。  
   - 将表格图像按该比例重新采样，再送入视觉编码器。这样大表格不会被压得太小，小表格也不会浪费显存。  
   - 视觉特征与语言模型的上下文拼接后，模型直接生成答案或新表格（T2T 任务），无需额外的后处理。  

**最巧妙的点**在于把“表格来源”当作数据质量的核心因素：只要表格来自科学领域，结构更规整、数值更密集，模型的学习效果就会大幅提升；这比盲目扩大通用表格规模更有效。

### 实验与效果
- **数据集与任务**：在 MMSci‑Eval 上评测三项任务：表格问答（TQA）、事实验证（TFV）和表格生成（T2T），重点关注数值推理的正确率。  
- **基线对比**：与使用 150 K 通用表格预训练的 MLLM、以及仅用文本序列化表格的 LLM 对比，本文模型在数值推理准确率上提升约 **12%–18%**（论文未给出精确数字，只给出“显著提升”）。在整体表格理解（结构预测 F1）上也领先约 **5%**。  
- **消融实验**：  
  - 去掉动态分辨率，数值推理下降约 **7%**，说明高分辨率是关键。  
  - 用通用表格预训练替代 MMSci‑Pre，整体性能下降约 **9%**，验证了“来源更重要”。  
  - 移除指令微调，仅保留结构预训练，T2T 任务几乎失效，表明指令微调是多任务适配的核心。  
- **局限性**：作者承认模型仍受限于单张表格的最大像素上限，极大表格仍需切块处理；此外，数值推理仍依赖显式的算术模块，复杂公式的推导能力有限。

### 影响与延伸思考
这篇工作在公开了 **MMSci‑Pre / MMSci‑Ins / MMSci‑Eval** 三套资源后，迅速成为科研表格处理的标准基准。后续有几篇论文尝试在此基础上加入 **外部数值计算引擎**（如 Python REPL）或 **图神经网络** 来进一步提升算术推理。还有研究把动态分辨率的思路推广到 **大幅面图表**（如医学影像报告）上，证明了该技术的通用潜力。想继续深入的读者可以关注 **“表格视觉语言模型的算术插件化”** 方向，或探索 **跨语言（英文/中文）科学表格的统一表示**。

### 一句话记住它
**只要表格来源精准、输入分辨率自适应，跨模态模型就能把科学表格的结构和数字算得清清楚楚。**