# Not All Correct Answers Are Equal: Why Your Distillation Source Matters

> **Date**：2025-05-20
> **arXiv**：https://arxiv.org/abs/2505.14464

## Abstract

Distillation has emerged as a practical and effective approach to enhance the reasoning capabilities of open-source language models. In this work, we conduct a large-scale empirical study on reasoning data distillation by collecting verified outputs from three state-of-the-art teacher models-AM-Thinking-v1, Qwen3-235B-A22B, and DeepSeek-R1-on a shared corpus of 1.89 million queries. We construct three parallel datasets and analyze their distributions, revealing that AM-Thinking-v1-distilled data exhibits greater token length diversity and lower perplexity. Student models trained on each dataset are evaluated on reasoning benchmarks including AIME2024, AIME2025, MATH500, and LiveCodeBench. The model distilled from AM-Thinking-v1 consistently achieves the best performance (e.g., 84.3 on AIME2024, 72.2 on AIME2025, 98.4 on MATH500, and 65.9 on LiveCodeBench) and demonstrates adaptive output behavior-producing longer responses for harder tasks and shorter ones for simpler tasks. These findings highlight the value of high-quality, verified reasoning traces. We release the AM-Thinking-v1 and Qwen3-235B-A22B distilled datasets to support future research on open and high-performing reasoning-oriented language models. The datasets are publicly available on Hugging Face\footnote{Datasets are available on Hugging Face: \href{https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled}{AM-Thinking-v1-Distilled}, \href{https://huggingface.co/datasets/a-m-team/AM-Qwen3-Distilled}{AM-Qwen3-Distilled}.}.

---

# 并非所有正确答案都等价：为何你的蒸馏来源很重要 论文详细解读

### 背景：这个问题为什么难？
开源大模型在推理题目上常常表现平平，主要因为它们缺少高质量的思考过程示例。传统的蒸馏方法往往直接把强模型的最终答案当作“老师”，忽视了答案背后的推理路径。这样得到的训练数据既缺乏多样性，又可能包含错误或不完整的思考，导致学生模型学不到真正的推理技巧。要想让开源模型在数学、代码等高阶任务上赶上闭源大模型，必须解决“老师到底该提供什么样的答案”这一根本问题。

### 关键概念速览
**蒸馏（Distillation）**：把一个大、强的模型（老师）产生的输出，用来训练一个更小、更快的模型（学生），相当于把“大师的手艺”压缩进新手的练习册。  
**思考链（Chain‑of‑Thought, CoT）**：在给出答案前先把推理步骤写出来，就像解题时先列出草稿，帮助模型学习逐步推理的逻辑。  
**验证输出（Verified Output）**：老师模型的答案经过人工或自动校验，确保答案和推理过程都是正确的，类似于老师批改后确认的答案。  
**Token 长度多样性**：生成文本的长度分布是否丰富，长短不一的回答能让学生学会在不同难度下调节输出。  
**困惑度（Perplexity）**：语言模型对一段文本的预测难度，数值越低说明文本越符合模型的语言习惯，类似于阅读流畅度。  
**适配性输出行为**：模型根据任务难度自动产生更长或更短的答案，像人一样会在复杂题目上写得详细，在简单题目上直接给出结论。  

### 核心创新点
1. **大规模并行蒸馏数据构建**：以前的工作往往只用单一老师模型或少量数据。本文同步使用三种最前沿的老师模型（AM‑Thinking‑v1、Qwen3‑235B‑A22B、DeepSeek‑R1），在同一 1.89 百万查询集合上生成三套平行数据集。这样可以直接比较不同老师的“教学质量”。  
2. **从分布角度审视蒸馏数据**：作者不仅统计了答案的正确率，还分析了 token 长度分布和困惑度。发现 AM‑Thinking‑v1 的数据更长、更丰富且语言更流畅，这说明“老师的表达方式”本身会影响学生的学习效果。  
3. **针对推理任务的学生模型评估**：把每套蒸馏数据分别训练学生模型，然后在 AIME2024/2025、MATH500、LiveCodeBench 等高阶推理基准上做对比。结果显示，使用 AM‑Thinking‑v1 数据的学生在所有基准上均领先，验证了高质量思考链的实际价值。  
4. **公开高质量蒸馏数据**：把 AM‑Thinking‑v1 与 Qwen3‑235B‑A22B 的蒸馏数据全部开源，为后续研究提供了可复现的基准，推动社区共同提升推理能力。  

### 方法详解
整体思路可以拆成三步：① 统一查询集合准备；② 多老师并行生成验证答案；③ 用生成的平行数据分别训练学生模型并评估。

**步骤 1：统一查询集合**  
作者收集了 1.89 百万条来自公开推理数据集的查询，覆盖数学、代码、逻辑等多种任务。所有查询在三套老师模型之间保持完全一致，确保后续比较只受老师输出差异影响。

**步骤 2：并行生成验证答案**  
- 每条查询分别送入 AM‑Thinking‑v1、Qwen3‑235B‑A22B、DeepSeek‑R1。  
- 为保证答案的可靠性，作者使用了两层验证：一是模型自带的置信度阈值，二是自动化的答案校验脚本（比如数学公式的符号检查、代码的单元测试）。只有通过双重校验的答案才会被写入对应的蒸馏数据集。  
- 生成的文本不仅包括最终答案，还保留完整的思考链，这让学生模型能够学习到“怎么思考”。  

**步骤 3：学生模型训练与评估**  
- 采用相同的模型架构和超参数，对三套数据分别进行微调。  
- 训练过程不做任何特殊技巧，只是普通的语言模型微调，目的是让老师的“教学质量”本身显现出来。  
- 训练完成后，在四个推理基准上做零-shot 与 few-shot 测试，记录准确率、答案长度等指标。  

**关键细节**  
- **长度多样性控制**：在生成思考链时，老师模型被设置了不同的最大 token 限制，导致 AM‑Thinking‑v1 的输出自然更长。作者把这视为一种“教学风格”，而不是人为调参。  
- **困惑度评估**：在数据构建阶段，作者计算每条生成文本的困惑度，发现 AM‑Thinking‑v1 的平均困惑度最低，说明它的语言更符合模型的统计规律，学生更容易学习。  
- **适配性输出行为**：实验中观察到，使用 AM‑Thinking‑v1 数据训练的学生在难题上会自动生成更长的思考链，在易题上则倾向于简短回答，这种行为在其他两套数据上不明显。  

### 实验与效果
- **测试基准**：AIME2024、AIME2025（美国数学竞赛模拟题），MATH500（数学推理集合），LiveCodeBench（代码生成与调试任务）。  
- **主要结果**：使用 AM‑Thinking‑v1 数据的学生在四个基准上分别取得 84.3、72.2、98.4、65.9 的分数，均显著高于 Qwen3‑235B‑A22B（约低 5‑10%）和 DeepSeek‑R1（约低 8‑12%）的学生。  
- **基线对比**：与未蒸馏的原始开源模型相比，所有蒸馏学生都有明显提升，尤其是 AM‑Thinking‑v1 数据带来的提升最为显著。  
- **消融实验**：论文提供了去掉思考链、只保留最终答案的实验，结果显示准确率下降约 6‑9%，说明完整的推理过程是关键因素。  
- **局限性**：作者承认只在单一模型架构上做了微调，未探索不同学生模型大小的差异；此外，验证过程主要依赖自动脚本，可能遗漏一些细微错误。  

### 影响与延伸思考
这篇工作让社区重新审视“老师的质量”在蒸馏中的作用，推动了以验证思考链为核心的蒸馏数据构建方式。后续有几篇论文（如 2025 年的 “Verified Chain‑of‑Thought Distillation”）直接引用了该数据集，尝试在更小模型上复现相同的适配性输出行为。对想继续深入的读者，可以关注以下方向：① 自动化高效的答案验证技术；② 不同模型规模下的蒸馏效应曲线；③ 将思考链蒸馏与自监督预训练结合的混合策略。  

### 一句话记住它
老师提供的高质量、验证过的思考链决定了蒸馏学生的推理水平，好的老师比好的答案更重要。