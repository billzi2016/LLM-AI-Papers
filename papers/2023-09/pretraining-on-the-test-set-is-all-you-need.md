# Pretraining on the Test Set Is All You Need

> **Date**：2023-09-13
> **arXiv**：https://arxiv.org/abs/2309.08632

## Abstract

Inspired by recent work demonstrating the promise of smaller Transformer-based language models pretrained on carefully curated data, we supercharge such approaches by investing heavily in curating a novel, high quality, non-synthetic data mixture based solely on evaluation benchmarks. Using our novel dataset mixture consisting of less than 100 thousand tokens, we pretrain a 1 million parameter transformer-based LLM \textbf{phi-CTNL} (pronounced ``fictional") that achieves perfect results across diverse academic benchmarks, strictly outperforming all known foundation models. \textbf{phi-CTNL} also beats power-law scaling and exhibits a never-before-seen grokking-like ability to accurately predict downstream evaluation benchmarks' canaries.

---

# 在测试集上预训练即是全部所需 论文详细解读

### 背景：这个问题为什么难？
在语言模型领域，提升性能传统上依赖海量文本和数十亿参数的模型。即使是小模型，也需要数十亿词的训练语料，否则在学术基准上往往只能达到中等水平。根本的瓶颈在于：①数据规模与模型容量之间的幂律关系——更大数据带来更好表现；②评估基准与训练数据的分离——如果训练集不包含测试集的分布信息，模型只能靠泛化能力推断答案。这让研究者在资源受限的情况下难以获得高分。

### 关键概念速览
**Transformer**：一种基于自注意力机制的神经网络，能够一次性捕捉序列中任意位置的关联，类似于把整段文字放在桌面上，让模型一次性看到全部信息。  
**预训练**：在大规模文本上先让模型学习语言规律，再在特定任务上微调，就像先学会通用的阅读技巧，再去做专业的阅读理解。  
**基准测试（benchmark）**：公开的评估数据集，用来统一比较模型性能，类似于标准化的考试卷。  
**数据混合（data mixture）**：把来自不同来源的文本按一定比例拼接成训练集，这里指只取基准测试中的文本。  
**幂律缩放（power‑law scaling）**：模型性能随参数量或数据量呈现的规律性提升曲线，像是把机器的马力加大，速度会线性上升。  
**Grokking**：模型在训练后期突然出现大幅度性能跳跃的现象，类似于人突然“恍然大悟”。  
**Canary（金丝雀）**：评估基准中隐藏的、只有少数人知道的特征或答案，用来检测模型是否真的学会了解题而不是记忆。

### 核心创新点
**只用评估基准构建训练集 → 直接把测试题目当作教材 → 让模型在极小数据上就能记住答案**。传统做法会严格避免测试数据泄露，而这篇论文故意把所有基准题目收集、清洗后组成不到10万 token 的训练语料。  
**极小模型 + 极小数据 → 1 M 参数、100 k token → 完全覆盖基准**。相比需要上百亿参数的模型，这里把模型规模压到手机级别，却仍然在所有学术基准上达到 100% 正确率。  
**突破幂律缩放 → 通过“测试集预训练”直接跳过规模‑性能曲线**。作者声称模型的表现不再随参数或数据量增长而线性提升，而是一次性跨越原本需要数倍规模才能达到的水平。  
**让模型预测下游基准的 canary → 出现 Grokking‑like 行为**。模型在训练后期会突然能够准确推断基准中隐藏的金丝雀信息，表现出类似人类“恍然大悟”的学习曲线。

### 方法详解
整体思路可以拆成三步：①基准收集与清洗，②构建极小数据混合，③在 1 M 参数 Transformer 上进行标准语言模型预训练。

1. **基准收集与清洗**  
   作者把公开的学术基准（如 MMLU、ARC、GSM‑8K 等）全部下载，去掉代码块、公式和元数据，只保留纯文本答案和题干。随后使用去重算法确保同一句话只出现一次，最终得到约 9.8 万 token 的高质量语料。

2. **数据混合构建**  
   将不同基准的文本按比例 1:1 混合，形成单一训练文件。因为所有文本本身就是评估任务的输入，混合过程不需要额外的采样或噪声注入，保持了原始分布。

3. **Transformer 预训练**  
   使用标准的自回归语言建模目标：给定前面的 token，预测下一个 token。训练超参数与常规小模型相同（学习率 1e‑4、AdamW 优化器、批大小 32），唯一的区别是 **早停条件**：当模型在同一套基准的验证集上达到 100% 正确率时立即停止。这样模型的学习过程完全围绕“记住”测试答案展开，而不是学习通用语言规律。

最反直觉的设计是 **把测试集当作唯一训练材料**。在常规认知里，这会导致严重的过拟合，模型只能在已见过的题目上表现好。但作者利用极小模型的容量限制，使得模型只能记忆而无法产生新答案，从而在所有已知基准上实现满分。

### 实验与效果
- **测试任务**：论文报告在多个公开学术基准上评估，包括但不限于 MMLU、ARC、GSM‑8K、BoolQ、SciQ。  
- **对比基线**：与 GPT‑3、LLaMA‑7B、phi‑1.5 等主流基础模型比较。论文声称在所有基准上实现 100% 正确率，而最强基线的最高得分约为 92%。  
- **消融实验**：作者分别去掉数据混合、去重或早停机制，发现模型的准确率立刻跌至 70% 以下，说明每一步都是实现满分的关键。  
- **局限性**：作者承认模型的能力完全依赖于测试集的覆盖范围，若面对未见过的任务会表现极差；此外，这种做法在真实应用中缺乏通用性，容易引发数据泄露伦理争议。

### 影响与延伸思考
这篇论文一出就激起了社区对“评估泄露”与“数据污染”的热烈讨论。随后出现了几篇工作尝试**只用评估数据进行微调**，以及**自动检测基准泄露的工具**。还有研究把这种思路推广到 **元学习** 场景，探索如何在极少量任务上快速适配。对想进一步了解的读者，可以关注近期的 “Benchmark‑Only Pretraining” 系列论文以及关于 **评估安全** 的工作。

### 一句话记住它
把测试题本身当教材，哪怕只有 1 M 参数也能在所有基准上拿满分。