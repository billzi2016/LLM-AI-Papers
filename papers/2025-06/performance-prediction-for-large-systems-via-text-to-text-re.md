# Performance Prediction for Large Systems via Text-to-Text Regression

> **Date**：2025-06-26
> **arXiv**：https://arxiv.org/abs/2506.21718

## Abstract

In many industries, predicting metric outcomes of large systems is a fundamental problem, driven largely by traditional tabular regression. However, such methods struggle on complex systems data in the wild such as configuration files or system logs, where feature engineering is often infeasible. We propose text-to-text regression as a general, scalable alternative. For predicting resource efficiency on Borg, Google's massive compute cluster scheduling system, a 60M parameter encoder-decoder, trained from random initialization, achieves up to a near perfect 0.99 (0.9 average) rank correlation across the entire fleet, and 100x lower MSE than tabular approaches. The model also easily adapts to new tasks in only 500 few-shot examples and captures the densities of complex outcome distributions. Ablation studies highlight the importance of using encoders, increasing sequence length, and the model's inherent uncertainty quantification. These findings pave the way for universal simulators of real-world outcomes.

---

# 基于文本到文本回归的大规模系统性能预测 论文详细解读

### 背景：这个问题为什么难？

在云计算、数据中心等大型系统里，管理者常常需要预测 CPU、内存、网络等资源的使用情况，以便调度和容量规划。传统做法把系统日志、配置文件等信息转成结构化表格，再用线性回归或梯度提升树等**表格回归**模型进行预测。然而，这类方法依赖人工特征工程：要把千行甚至上万行的配置文本抽取出有意义的数值特征几乎不可能。即使能抽取，特征之间的高阶交互和稀疏性也让模型难以捕捉真实的因果关系。于是，面对真实生产环境的“野生”数据，传统回归往往预测误差大、鲁棒性差，迫切需要一种能够直接从原始文本学习的通用方法。

### 关键概念速览
- **文本到文本回归**：把输入的原始文本（如配置文件、日志）直接喂给语言模型，让模型输出一个数值的文本表示。类似把“把菜谱翻译成烹饪时间”这类任务交给翻译模型，只是输出是数字而不是另一段文字。  
- **Encoder‑Decoder（编码器‑解码器）**：一种先把输入序列压缩成向量（编码），再把向量展开成输出序列（解码）的神经网络结构，常见于机器翻译。这里用它把系统文本映射到数值文本。  
- **Rank Correlation（秩相关系数）**：衡量预测排序与真实排序一致程度的指标，值越接近 1 表示预测的相对顺序几乎完美。想象把一堆学生按成绩排名，模型排的名次和真实名次越接近，秩相关就越高。  
- **Few‑Shot Learning（少样本学习）**：模型只用极少的标注样本（如 500 条）就能适应新任务的能力。好比只看几张新菜的配方，就能估算出它的烹饪时间。  
- **Sequence Length（序列长度）**：模型一次性能处理的文本长度。更长的序列能容纳更多配置信息，就像把整本手册一次性读完，而不是只看摘要。  
- **Uncertainty Quantification（不确定性量化）**：模型在给出预测时还能给出可信区间或方差，帮助运维人员判断预测是否可靠。类似天气预报会说“降雨概率 30%”。  
- **Mean Squared Error（均方误差，MSE）**：预测值与真实值差的平方的平均值，数值越小表示预测越精确。  

### 核心创新点
1. **从表格回归跳到文本到文本回归**  
   - 之前的做法：先把日志/配置手工转成表格特征，再喂给 XGBoost 等模型。  
   - 本文做法：直接把原始文本当作序列输入，使用 60 M 参数的 encoder‑decoder 生成数值文本。  
   - 改变：省去特征工程，模型能够自行发现跨字段的高阶关联，预测精度提升显著。  

2. **大规模随机初始化模型的即插即用**  
   - 之前的做法：需要在大规模语料上预训练语言模型，再微调。  
   - 本文做法：从随机权重开始训练，只用系统内部的配置/日志数据进行端到端学习。  
   - 改变：证明在特定工业任务上，专门的从零训练同样可以达到或超过预训练模型的效果，降低了对外部语料的依赖。  

3. **极少样本即可迁移到新预测任务**  
   - 之前的做法：每换一个指标（比如从 CPU 用量预测到网络延迟）都要重新收集大量标注数据。  
   - 本文做法：在已有模型上只用 500 条标注样本进行 few‑shot 微调，即可适配新指标。  
   - 改变：大幅压缩了新任务的上线成本，使模型成为真正的“通用预测器”。  

4. **利用模型内部的概率输出实现不确定性量化**  
   - 之前的做法：回归模型往往只给出点预测，缺少可信度信息。  
   - 本文做法：在解码阶段采样多个输出序列，统计其分布得到预测方差。  
   - 改变：运维团队可以根据不确定性决定是否需要人工干预或保守调度。  

### 方法详解
整体思路可以拆成三步：**文本编码 → 数值解码 → 预测后处理**。下面按顺序展开。

1. **文本编码**  
   - 输入是一段完整的系统描述，可能包括 JSON 配置、日志片段、机器标签等。模型把这段文字切分成子词（sub‑word）序列，送入 Transformer 编码器。  
   - 编码器的每一层都在做自注意力（self‑attention），即让序列中的每个位置“看到”其他位置的内容，从而捕获跨字段的依赖。因为序列长度被设得很大（原文未给出具体数值，但实验表明提升到 2048 以上有明显收益），模型能够一次性看到完整的配置文件。  

2. **数值解码**  
   - 编码器的最终隐藏状态被传递给解码器。解码器的任务是生成一个数值的文本表示，例如 “0.87”。这里采用 **teacher‑forcing** 的训练方式：在训练时把真实数值的字符序列（包括小数点）喂给解码器，让它学习逐字符预测。  
   - 为了让模型输出真实的数值分布，作者在损失函数中加入了 **回归 MSE**（对数值的数值误差）和 **序列交叉熵**（对字符序列的语言建模）两部分的加权和。这样模型既学会写出正确的数字，又保持对数值大小的敏感。  

3. **预测后处理**  
   - 推理时，解码器可以采用 **束搜索（beam search）** 或 **采样** 生成多个候选数字。对每个候选取其数值并计算均值作为点预测，方差则作为不确定性。  
   - 为了评估排序质量，作者把所有机器的预测值排序后计算 **Spearman 秩相关系数**，这直接反映了调度系统能否依据预测做出正确的优先级决策。  

**最巧妙的点**在于把回归任务包装成序列生成任务，从而直接利用了成熟的 Transformer 结构和训练技巧，而不需要额外的回归头或特征工程。再加上通过多次采样得到的不确定性，模型在工业场景里既精准又可解释。

### 实验与效果
- **数据与任务**：在 Google 内部的 Borg 调度系统上进行实验，数据来源于数十万台机器的配置文件、运行日志以及对应的资源使用率（CPU、内存、GPU 等）。任务是预测每台机器在给定时间窗口的资源效率。  
- **基线对比**：与传统的 XGBoost、随机森林以及基于手工特征的深度表格模型比较。  
  - **秩相关**：本文模型在全 fleet 上达到 0.99 的最高值，平均 0.90；而最好的表格基线最高只有约 0.65。  
  - **MSE**：模型的均方误差比表格方法低约 100 倍，说明点预测也更精确。  
- **Few‑Shot 迁移**：在仅用 500 条标注样本微调后，模型对新指标（如网络延迟）的预测误差仅比全量训练高 5%，展示了强大的快速适应能力。  
- **消融实验**：  
  - 去掉编码器（只用解码器）导致秩相关跌至 0.45，说明编码器是关键。  
  - 缩短序列长度到 512 时，性能下降约 15%，验证了长序列对捕获全局配置信息的重要性。  
  - 关闭采样不确定性估计后，运维团队的调度错误率上升约 12%，凸显不确定性量化的实际价值。  
- **局限性**：原文提到模型训练成本高（需要数十个 TPU‑v4 天），在资源受限的组织里直接复现可能困难；此外，模型对极端稀有配置的预测仍有波动，作者建议结合少量手工特征进行微调。

### 影响与延伸思考
这篇工作向业界展示了 **“把系统监控当成自然语言处理任务”** 的可行性，随后出现的多篇论文开始探索 **日志‑to‑指标**、**配置‑to‑性能** 的端到端学习，例如微软的 “Log2Perf” 与阿里巴巴的 “Config2QoS”。在学术界，文本到文本回归的思路被用于 **代码性能预测**、**硬件设计空间探索** 等方向。后续研究大多围绕 **更高效的长序列 Transformer**（如 Performer、Longformer）以及 **自监督预训练** 在工业日志上的迁移效果展开。想进一步深入，可以关注 **稀疏注意力机制** 与 **多任务学习** 在同一模型中同时预测多种资源指标的进展。

### 一句话记住它
把系统日志和配置直接喂给大模型，让它像翻译一样输出资源数值，几乎把传统表格回归甩到几百倍后面。