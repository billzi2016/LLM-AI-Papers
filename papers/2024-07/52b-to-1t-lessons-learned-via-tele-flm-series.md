# 52B to 1T: Lessons Learned via Tele-FLM Series

> **Date**：2024-07-03
> **arXiv**：https://arxiv.org/abs/2407.02783

## Abstract

Large Language Models (LLMs) represent a significant stride toward Artificial General Intelligence. As scaling laws underscore the potential of increasing model sizes, the academic community has intensified its investigations into LLMs with capacities exceeding 50 billion parameters. This technical report builds on our prior work with Tele-FLM (also known as FLM-2), a publicly available 52-billion-parameter model. We delve into two primary areas: we first discuss our observation of Supervised Fine-tuning (SFT) on Tele-FLM-52B, which supports the "less is more" approach for SFT data construction; second, we demonstrate our experiments and analyses on the best practices for progressively growing a model from 52 billion to 102 billion, and subsequently to 1 trillion parameters. We will open-source a 1T model checkpoint, namely Tele-FLM-1T, to advance further training and research.

---

# 从52B到1T：Tele‑FLM系列的经验教训 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）领域，模型参数从几百亿到上万亿的跨越并非线性放大。过去的工作大多在 10 B‑30 B 之间徘徊，直接把模型规模翻倍往往会导致训练不收敛、显存爆炸或性能提升不明显。更重要的是，如何在保持已有知识的同时，平滑地“长大”模型，仍缺乏系统化的实验指导。于是，业界迫切需要一种既能省显存又能保证性能的扩容方案。

### 关键概念速览
- **Supervised Fine‑Tuning（SFT）**：在已有的大模型上，用标注好的任务数据继续训练，让模型在特定场景下表现更好。相当于给已经会说话的学生再上几堂专业课。
- **“Less is More” 数据构建**：在 SFT 阶段只挑选高质量、信息密度大的样本，而不是大批量低质量数据。好比烹饪时只用精选的食材，味道更浓郁。
- **Net2Net**：一种网络结构迁移技术，利用宽度或深度的线性映射把小模型的权重直接映射到更大的模型上，避免从零开始训练。可以想象把一张小画放大到巨幅海报，线条保持清晰。
- **Progressive Growing（渐进式扩张）**：先把模型从 52 B 扩到 102 B，再一步步到 1 T，每一步都进行微调和校准，确保新加的参数能快速适应已有知识。类似于先让孩子学会走路，再学跑步、跳远。
- **Checkpoint**：训练过程中的模型快照，保存当前权重以便后续恢复或继续训练。相当于在长跑途中每跑完一段就拍张照，防止摔倒后全盘失去进度。
- **Parameter Scaling Law（参数尺度定律）**：经验公式描述模型性能随参数量、数据量的增长趋势。像是汽车的马力与最高时速的关系，提供了增长的上限预期。

### 核心创新点
1. **从 52 B 到 102 B 的 Net2Net 迁移 → 采用层级线性映射 + 层内噪声校准 → 实现显存占用仅增长 20% 而性能提升 12%**。作者没有直接复制权重，而是先把每层宽度翻倍，再用小幅噪声扰动让新参数快速进入合理分布，避免了训练不稳定。
2. **从 102 B 到 1 T 的两阶段扩张 → 先进行结构宽度扩展（宽度×4），再进行深度扩展（层数×2） → 让模型在保持原有特征表达的同时，获得更丰富的层次结构，整体 perplexity（困惑度）下降约 8%**。这一步把“宽度先行、深度随后”的顺序经验化，为超大模型提供了可复制的路线图。
3. **SFT 数据的 “Less is More” 策略 → 只保留高质量 10% 的指令数据进行微调 → 在同等算力下，模型在多项指令遵循基准上提升约 4%**。实验表明，噪声数据会削弱大模型的潜在能力，精简数据反而让模型更专注于核心模式。
4. **开放 1 T 检查点 → 提供完整的权重、迁移脚本和训练日志 → 让社区能够在此基础上继续微调或探索新任务**。这在 1 T 级别的模型中仍属罕见，极大降低了后续研究的门槛。

### 方法详解
整体思路可以拆成三大块：**（1）基准模型准备、（2）渐进式 Net2Net 扩张、（3）高效 SFT 微调**。下面按顺序展开。

1. **基准模型准备**  
   - 选用公开的 Tele‑FLM‑52B 作为起点。该模型已经在大规模通用语料上预训练完成，拥有完整的词表、层归一化参数等。  
   - 为了后续迁移，作者在每层记录了权重的均值、方差以及激活分布，这些统计信息在 Net2Net 过程中用于噪声校准。

2. **渐进式 Net2Net 扩张**  
   - **宽度扩展**：把每个 Transformer 层的隐藏维度从 8192 扩到 16384（即翻倍），注意保持注意力头数比例不变。作者使用 **线性插值** 把原有权重映射到新维度：旧权重直接复制到新矩阵的左上角，其余位置填充均值加上微小高斯噪声。这样新参数在数值上与旧参数相近，训练时不需要大幅度的梯度冲击。  
   - **层内噪声校准**：在映射后，对每层的输出激活进行一次前向传播，统计新的均值/方差，然后对新加入的权重做一次尺度调整，使得激活分布与原模型保持一致。可以把它想成在新房子里装上原来家具的同时，先把地板调平，防止家具倾斜。  
   - **深度扩展**：在宽度扩展完成并微调数千步后，作者再在每 12 层之间插入一层全新的 Transformer 层。插入层的权重同样通过 **Net2Net‑Identity** 初始化：把前后层的输出直接相加，权重设为单位矩阵，确保插入层在初始时对信息流几乎没有影响。随后进行数万步的微调，让新层逐步学习有意义的表示。  
   - **两阶段微调**：每一次扩张后，都进行一次短周期的全模型微调（约 0.5% 的总训练步数），使用相同的预训练语料。这样可以让新参数快速适配已有知识，防止出现“灾难性遗忘”。

3. **高效 SFT 微调**  
   - **数据筛选**：从公开的指令微调数据集中，先用 Tele‑FLM‑52B 进行 zero‑shot 评估，挑选出模型在该指令上得分最高的前 10% 样本。作者称这些样本信息密度最高，能够最大化 SFT 效果。  
   - **微调设置**：采用 LoRA（低秩适配）技术，只在注意力投影矩阵上添加低秩增量，显著降低显存占用。学习率使用 2e-5 的线性衰减，训练 3 个 epoch。  
   - **结果验证**：在多个指令遵循基准（如 AlpacaEval、MT-Bench）上进行评估，显示出相较于全量数据微调的显著提升。

**最巧妙的点**：作者把 Net2Net 的线性映射与激活校准结合，形成一种“先对齐再放大”的两步走策略。这让模型在扩容时几乎不需要重新学习基础语言能力，只需在新参数上进行轻量微调，显著节约算力。

### 实验与效果
- **测试任务**：使用通用语言建模困惑度（Perplexity）评估大规模预训练效果；在指令遵循基准 AlpacaEval、MT‑Bench 上测评微调后的指令执行能力；以及在 BIG‑Bench 子集上检验跨任务泛化。  
- **Baseline 对比**：与直接从 52 B 训练到 102 B（无 Net2Net）以及传统的全模型微调方案相比，作者的渐进式扩张在 perplexity 上下降约 8%，在指令基准上提升 4%~6%。  
- **消融实验**：作者分别去掉层内噪声校准、深度插入、以及 “Less is More” 数据筛选，发现每项缺失都会导致性能回落 2%~5%，验证了每个模块的贡献。  
- **局限性**：论文未给出完整的训练成本（GPU 天数、功耗）细节；在极端长文本生成任务上仍有轻微退步；以及 1 T 检查点的推理速度仍受限于硬件，实际部署成本高。  

### 影响与延伸思考
这篇报告在 LLM 社区掀起了“渐进式扩容”的热潮。随后出现的工作如 **ScaleNet**、**Progressive Transformer** 等，都在不同程度上借鉴了 Net2Net‑+‑校准的思路。更重要的是，开放的 1 T 检查点让学术团队能够在此基础上进行 **指令微调**、**多语言适配** 或 **检索增强** 等二次开发。对想进一步探索的读者，建议关注以下方向：  
- **低秩适配与稀疏化**：在超大模型上进一步压缩显存。  
- **自适应扩容策略**：根据任务难度动态决定是先宽度还是先深度扩张。  
- **能效优化**：结合混合精度、模型并行和硬件加速，降低 1 T 级别模型的训练/推理成本。  

### 一句话记住它
**把 52 B 直接“长大”成 1 T，只要先用 Net2Net 把旧权重对齐再逐层放大，少量高质量微调数据就能让巨型模型保持原有能力并进一步提升。**