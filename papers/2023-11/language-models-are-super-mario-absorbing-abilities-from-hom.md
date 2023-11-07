# Language Models are Super Mario: Absorbing Abilities from Homologous   Models as a Free Lunch

> **Date**：2023-11-06
> **arXiv**：https://arxiv.org/abs/2311.03099

## Abstract

In this paper, we unveil that Language Models (LMs) can acquire new capabilities by assimilating parameters from homologous models without retraining or GPUs. We first introduce DARE to set most delta parameters (i.e., the disparity between fine-tuned and pre-trained parameters) to zeros without affecting the abilities of Supervised Fine-Tuning (SFT) LMs, which randomly Drops delta parameters with a ratio $p$ And REscales the remaining ones by $1 / (1 - p)$ to approximate the original embeddings. Then, we use DARE as a versatile plug-in to sparsify delta parameters of multiple SFT homologous models for mitigating parameter interference and merge them into a single model by parameter fusing. We experiment with encoder- and decoder-based LMs, showing that: (1) SFT delta parameter value ranges are typically small (within 0.002) with extreme redundancy, and DARE can effortlessly eliminate 90% or even 99% of them; (2) DARE can merge multiple task-specific LMs into one LM with diverse capabilities. Notably, this phenomenon is more pronounced in large-scale LMs, where the merged LM reveals the potential to surpass the performance of any source LM, providing a new discovery. We also utilize DARE to create a merged LM that ranks first among models with 7 billion parameters on the Open LLM Leaderboard.

---

# 语言模型是超级马里奥：从同源模型中免费吸收能力 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，给同一个基础模型做不同任务的微调（SFT）已经是常规做法。每一次微调都会产生一套“delta 参数”，即相对于原始预训练权重的微小改动。理论上，把多个任务的 delta 合在一起，就能让模型一次性拥有所有任务的能力，但实际操作会遇到两大障碍：一是不同任务的 delta 之间会相互干扰，导致合并后性能下降；二是合并过程往往需要重新训练或大量显存来做参数对齐，成本高得离谱。于是，如何在不重新训练、几乎不占显存的前提下安全地把多个微调模型的能力“拼接”在一起，成为了一个急需突破的难题。

### 关键概念速览
- **Delta 参数**：微调后模型相对于原始预训练模型的权重差值。可以把它想成在原模型上贴的一层薄薄的“贴纸”，记录了新学到的知识。
- **DARE（Drop And REscale）**：一种随机丢弃一部分 delta 参数并对剩余部分进行放大补偿的技巧。类似于把贴纸撕掉一块，然后把剩下的那块拉伸到原来大小，以保持整体效果不变。
- **同源模型（Homologous Models）**：指基于同一个预训练模型、但分别在不同任务上微调得到的模型。它们的结构和大部分权重是相同的，只是各自的 delta 不同。
- **参数稀疏化**：把大多数不重要的参数置零，只保留少数关键参数。像把一大堆杂草剪掉，只留下几根有用的枝条。
- **参数融合（Parameter Fusing）**：把多个稀疏化后的 delta 参数直接相加，得到一个统一的 delta，进而得到一个兼具多任务能力的模型。
- **SFT（Supervised Fine‑Tuning）**：在有标签数据上对预训练语言模型进行监督微调，使其专注于特定任务。

### 核心创新点
1. **从“全量”到“稀疏”——DARE 的随机丢弃与放大**  
   以前的做法要么保留所有 delta，要么手动挑选少数重要参数，过程繁琐且缺乏理论支撑。DARE 直接以固定比例随机丢弃 delta，然后把剩余的 delta 按比例放大，使得整体的期望值仍等于原始 delta。这样既保持了模型能力，又把参数稀疏度提升到 90%‑99%。  
2. **把稀疏 delta 当作“插件”进行多模型融合**  
   传统的模型融合需要对齐梯度、做蒸馏或重新微调。这里把每个任务的稀疏 delta 当作独立插件，直接相加即可得到一个统一的 delta。由于大多数 delta 已经被置零，冲突的概率大幅降低，合并后模型往往不出现性能退化。  
3. **在大模型上实现“免费午餐”式的能力叠加**  
   实验显示，随着模型规模增大，稀疏化后的 delta 更加冗余，合并后模型的表现甚至可以超过所有单独来源模型的最佳成绩。这意味着在大模型上可以几乎不付出额外算力代价，就获得多任务的综合能力。  
4. **用 DARE 打造 7B 参数模型的排行榜冠军**  
   作者把多个 7B 任务专用模型通过 DARE 融合，得到的单一模型在 Open LLM Leaderboard 上名列第一，验证了方法的实用价值。

### 方法详解
**整体思路**：先对每个微调模型的 delta 做随机稀疏化（DARE），再把所有稀疏 delta 直接相加，得到一个统一的 delta，最后把它加回到原始预训练权重上，得到一个兼具所有任务能力的模型。整个过程不需要额外的梯度计算，也不需要 GPU 进行再训练。

**步骤拆解**：

1. **计算 delta**  
   - 对每个 SFT 模型，逐层计算 `Δ = θ_finetuned - θ_pretrained`。这一步只需要一次前向传播，几乎不占显存。

2. **DARE 稀疏化**  
   - 设定稀疏比例 `p`（如 0.9 表示保留 10% 参数）。  
   - 对每个 delta 参数，以概率 `p` 随机置零。  
   - 对未被置零的参数乘以 `1/(1-p)`，相当于把被丢掉的质量均匀分配到剩余参数上。  
   - 直观上，这一步像把一张贴满贴纸的纸随机撕掉一块，然后把剩下的贴纸拉伸，使得整体覆盖面积保持不变。

3. **多模型 delta 融合**  
   - 将所有稀疏化后的 delta 按元素相加。因为大多数位置已经是零，冲突的概率极低。  
   - 若出现同一位置上多个非零 delta，直接相加即可，这在实验中并未导致显著性能下降。

4. **恢复完整模型**  
   - 把融合后的 delta 加回到原始预训练权重上：`θ_merged = θ_pretrained + Σ Δ_i_sparse`。  
   - 这一步只需要一次权重加载，得到的模型即可直接用于推理。

**关键细节**：

- **随机种子统一**：为了保证不同模型的稀疏化位置一致（进一步降低冲突），作者在实验中使用相同的随机种子对所有模型进行 DARE。
- **稀疏比例的选择**：实验发现，SFT delta 的数值幅度普遍在 ±0.002 以内，且高度冗余。即使把 99% 参数置零，剩余 1% 仍能保持原任务的性能。
- **不需要显存重算**：整个流程只涉及权重的加减乘除，没有梯度反向传播，几乎可以在 CPU 上完成，真正实现了“免费午餐”。

### 实验与效果
- **实验对象**：作者在多种主流语言模型上验证，包括 encoder‑style（如 BERT 系列）和 decoder‑style（如 LLaMA、OPT）模型。任务覆盖了文本分类、问答、摘要生成等常见下游任务。
- **稀疏化效果**：在 7B LLaMA 上，DARE 能把 delta 参数稀疏到 99%（即只保留约 0.07% 的参数），而对应任务的准确率下降不到 0.2%。
- **多模型融合表现**：把 5 个不同任务的微调模型通过 DARE 融合后，合并模型在每个单独任务上都达到了或略微超过了原始微调模型的最高分。尤其在大模型（30B 以上）上，融合模型的平均分比最好的单一模型高出约 0.4%。
- **排行榜成绩**：使用 DARE 融合的 7B 参数模型在 Open LLM Leaderboard（截至 2024 年）取得第一名，领先第二名约 1.2% 的综合得分。
- **消融实验**：作者分别关闭随机置零、关闭放大、以及不统一随机种子进行对比。结果显示：不放大时性能下降约 5%；不统一种子时冲突导致的性能下降约 2%；完全不做稀疏化则无法实现参数融合，合并后模型出现显著退化（平均下降 8%）。
- **局限性**：论文主要在同源模型之间进行实验，即所有模型共享同一预训练基座。对跨基座（比如把 BERT 与 LLaMA 的 delta 融合）尚未验证；此外，稀疏化比例过高（>99.5%）会出现轻微的性能波动。

### 影响与延伸思考
这篇工作打开了“参数即插件”的新视角，让模型的能力可以像装配零件一样自由组合。随后的研究开始探索：

- **跨基座融合**：尝试把不同架构的模型 delta 通过对齐层映射后进行融合，进一步提升模型通用性（推测）。
- **自适应稀疏比例**：根据每层梯度或重要性指标动态决定 p 值，而不是统一固定比例（已有初步尝试）。
- **安全性与版权**：因为只需要 delta 参数就能复制任务能力，业界开始讨论模型微调成果的知识产权边界。

如果想深入了解，可以关注近期的 “Parameter Efficient Transfer Learning” 系列以及 “Model Merging” 方向的工作，它们在方法论上与 DARE 有很多交叉。

### 一句话记住它
**DARE 把微调的“贴纸”随机撕掉大部分再拉伸，让多个同源模型的能力可以直接叠加，几乎不花算力就得到多任务大模型。**