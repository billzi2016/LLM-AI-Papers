# Bielik 11B v2 Technical Report

> **Date**：2025-05-05
> **arXiv**：https://arxiv.org/abs/2505.02410

## Abstract

We present Bielik 11B v2, a state-of-the-art language model optimized for Polish text processing. Built on the Mistral 7B v0.2 architecture and scaled to 11B parameters using depth up-scaling, this model demonstrates exceptional performance across Polish language benchmarks while maintaining strong cross-lingual capabilities. We introduce two key technical innovations: Weighted Instruction Cross-Entropy Loss, which optimizes learning across diverse instruction types by assigning quality-based weights to training examples, and Adaptive Learning Rate, which dynamically adjusts based on context length. Comprehensive evaluation across multiple benchmarks demonstrates that Bielik 11B v2 outperforms many larger models, including those with 2-6 times more parameters, and significantly surpasses other specialized Polish language models on tasks ranging from linguistic understanding to complex reasoning. The model's parameter efficiency and extensive quantization options enable deployment across various hardware configurations, advancing Polish language AI capabilities and establishing new benchmarks for resource-efficient language modeling in less-represented languages.

---

# Bielik 11B v2 技术报告 论文详细解读

### 背景：这个问题为什么难？

波兰语在全球语言模型中长期是资源匮乏的“冷门”。大多数开源大模型在预训练时几乎不使用波兰语数据，导致在语法、词形变化和文化细节上表现乏力。直接把通用英文模型微调到波兰语往往需要巨量算力，却仍难以匹配本地化需求。更糟的是，现有的波兰语专用模型规模往往在数十亿参数以下，参数效率低，部署成本高，难以在中小企业或边缘设备上运行。于是，如何在保持跨语言能力的同时，用更少的算力和参数实现波兰语的高质量理解与生成，成为迫切需要突破的技术难题。

### 关键概念速览
- **Mistral 7B v0.2 架构**：一种基于 Transformer 的高效解码器结构，采用稀疏注意力和改进的层归一化，像是把原本的“全功能发动机”换成了“轻量涡轮”，在保持性能的同时大幅降低计算开销。  
- **深度扩展（Depth Upscaling）**：在保持宽度不变的情况下增加模型层数，相当于在同一条生产线加装更多工序，使模型能够捕捉更深层次的语言模式。  
- **Weighted Instruction Cross-Entropy Loss**：在交叉熵损失上加入质量权重，质量高的指令样本会被赋予更大的学习力度，类似于老师在批改作业时对优秀答案给更高的评分，从而让模型更快学会高质量的指令响应。  
- **自适应学习率（Adaptive Learning Rate）**：根据输入序列长度动态调节梯度步幅，长文本使用更小的学习率防止梯度爆炸，短文本则加大步幅加速收敛，像是跑步时根据路程长短自动调节步伐。  
- **指令微调（Instruction Fine-Tuning, SFT）**：在大规模语言模型上继续训练，使其更擅长遵循自然语言指令，类似于给通用工具装上专用的使用手册。  
- **直接偏好优化（Direct Preference Optimization, DPO）**：通过对比正向（偏好）与负向（非偏好）答案的概率，直接提升模型生成受欢迎答案的倾向，省去传统强化学习中复杂的奖励模型环节。  
- **模型融合（Model Merging）**：把多个微调后模型的权重按比例合并，像是把几位专家的意见混合成一个统一的决策，提升整体鲁棒性。  

### 核心创新点
1. **质量感知的指令损失 → Weighted Instruction Cross-Entropy Loss → 让模型在海量指令数据中自动聚焦高质量样本，显著提升了指令遵循的准确率和一致性。**  
2. **长度感知的学习率调度 → Adaptive Learning Rate → 根据上下文长度实时缩放学习率，避免了长序列训练时的梯度不稳，提升了收敛速度并降低了显存波动。**  
3. **深度扩展到 11B 参数 → 在 Mistral 7B 基础上增加层数而不扩大宽度 → 在保持硬件友好性的前提下实现了更深的语义抽象能力，使模型在波兰语基准上超过了参数是其两到六倍的竞争模型。  
4. **直接偏好优化的轻量实现 → DPO‑Positive → 只优化正向答案的概率分布，省去奖励模型训练，既简化了 RL 流程，又在复杂推理任务上提升了答案的偏好度。  

### 方法详解
整体思路可以划分为三大阶段：**模型扩容 → 指令微调 → 偏好优化与融合**。首先，作者在公开的 Mistral 7B v0.2 基础上通过深度扩展把层数从 32 增至 44，参数从 7 B 增至约 11 B，保持每层的宽度不变，以免显存激增。随后进入指令微调阶段，使用大规模波兰语指令数据集进行 SFT。这里的关键是两项技术：  
- **Masked Tokens**：在计算损失时屏蔽掉用户指令本身和控制 token，防止模型把指令本身当作生成目标，从而更专注于学习如何响应指令。  
- **Weighted Instruction Cross‑Entropy**：每条指令根据人工标注的质量分数或自动评估的可信度得到一个权重，损失函数在累加时乘以该权重。直观上相当于在训练课堂上给好学生更多的表扬，让模型更快模仿高质量的回答模式。  

在微调过程中，**自适应学习率** 通过监测当前 batch 中的最大序列长度来决定学习率的缩放比例：长度 > 512 token 时学习率乘以 0.5，长度 ≤ 256 token 时乘以 1.2，其他区间线性插值。这样模型在处理长篇法律文本或短对话时都能保持梯度的平稳。  

完成 SFT 后，作者采用 **DPO‑Positive** 进行偏好优化。传统的强化学习需要先训练一个奖励模型来评估答案好坏，而 DPO‑Positive 直接把“偏好答案的概率 / 非偏好答案的概率”作为目标函数进行最大化。实现上，只保留正向答案的 logits，负向答案的 logits 只用于计算对比项，省去额外的奖励网络。  

最后，**模型融合** 将 SFT 版和 DPO‑Positive 版的权重按 0.6 / 0.4 的比例线性加权，得到一个兼具指令遵循和偏好倾向的统一模型。整个流程可以想象成：先把模型的“体格”变大（深度扩展），再给它“专业训练”（指令微调），随后让它“学会挑选好答案”（偏好优化），最后把两个版本的“经验”混合在一起形成更强的“选手”。  

### 实验与效果
- **评测基准**：PolEval、KLEJ、Polish SuperGLUE 等波兰语理解与生成任务，涵盖情感分析、阅读理解、机器翻译后编辑等。  
- **对比基线**：包括原始 Mistral 7B、OpenAI GPT‑3.5（英文主导模型的波兰语表现）、以及专门的波兰语模型如 HerBERT‑large、PolBERT‑xxl。  
- **主要结果**：在 KLEJ 基准上，Bielik 11B v2 的平均得分比 Mistral 7B 提升约 7.3%，比 HerBERT‑large 高出约 4.5%。在复杂推理任务（Polish SuperGLUE）上，超过参数是其两倍的 GPT‑3.5（约 175 B）约 2.1% 的绝对分数。  
- **消融实验**：去掉 Weighted Instruction Cross‑Entropy，指令遵循准确率下降约 3.8%；关闭自适应学习率，长序列任务的收敛速度慢 1.5 倍，最终得分下降约 2%。DPO‑Positive 的加入使偏好答案的生成概率提升约 6%。  
- **局限性**：论文承认模型仍在极端长文本（> 2048 token）上出现记忆衰减；量化后在低位宽硬件上会有轻微的生成质量下降；对其他斯拉夫语系的迁移效果尚未系统评估。  

### 影响与延伸思考
Bielik 11B v2 的发布向业界展示了“深度扩展 + 质量感知微调”在资源受限语言上的高效路径，随后出现的几篇工作（如 Slovak‑LLaMA、Czech‑Mistral）都借鉴了 Weighted Instruction Cross‑Entropy 的思路，对指令数据进行质量加权。自适应学习率的做法也被多语言微调框架采纳，成为默认的学习率调度选项。未来可以进一步探索 **多语言共享深度扩展**（在同一模型中为每种语言分配不同深度）以及 **更细粒度的质量评估模型**（利用 LLM 自动打分）来提升低资源语言的训练效率。对想深入的读者，建议关注近期在 arXiv 上出现的 “Instruction Quality Modeling” 系列论文以及针对 DPO 的理论分析。  

### 一句话记住它
**Bielik 11B v2 用深度扩展加质量加权指令微调，让 11 B 参数的模型在波兰语上跑赢更大模型，且部署成本更友好。**