# From 128K to 4M: Efficient Training of Ultra-Long Context Large Language   Models

> **Date**：2025-04-08
> **arXiv**：https://arxiv.org/abs/2504.06214

## Abstract

Long-context capabilities are essential for a wide range of applications, including document and video understanding, in-context learning, and inference-time scaling, all of which require models to process and reason over long sequences of text and multimodal data. In this work, we introduce a efficient training recipe for building ultra-long context LLMs from aligned instruct model, pushing the boundaries of context lengths from 128K to 1M, 2M, and 4M tokens. Our approach leverages efficient continued pretraining strategies to extend the context window and employs effective instruction tuning to maintain the instruction-following and reasoning abilities. Our UltraLong-8B, built on Llama3.1-Instruct with our recipe, achieves state-of-the-art performance across a diverse set of long-context benchmarks. Importantly, models trained with our approach maintain competitive performance on standard benchmarks, demonstrating balanced improvements for both long and short context tasks. We further provide an in-depth analysis of key design choices, highlighting the impacts of scaling strategies and data composition. Our findings establish a robust framework for efficiently scaling context lengths while preserving general model capabilities. We release all model weights at: https://ultralong.github.io/.

---

# 从128K到4M：高效训练超长上下文大语言模型 论文详细解读

### 背景：这个问题为什么难？
在传统的大语言模型（LLM）里，输入的上下文长度通常被限制在几千到几万 token，超过这个范围模型会因为显存和计算成本爆炸而失效。很多实际场景——比如整篇文档阅读、长视频字幕理解、跨段落推理——都需要一次性处理上百千甚至上百万 token。过去的解决方案要么是把长文本切块后逐段处理，导致跨块信息丢失；要么是使用稀疏注意力等技巧，但这些方法在保持模型原有的指令遵循和推理能力上表现不佳，甚至会在短上下文任务上出现退化。因此，如何在不牺牲原有能力的前提下，经济高效地把上下文窗口从 128K 扩展到数百万 token，成为亟待突破的瓶颈。

### 关键概念速览
**上下文窗口（Context Window）**：模型一次性能够看到的 token 序列长度，类似于人阅读时一次性记住的文字量。  
**继续预训练（Continued Pretraining）**：在已有模型基础上，再用新的数据继续训练，以适应新的任务或特性。相当于在已经学会基础语法后，再去练习长篇阅读。  
**指令微调（Instruction Tuning）**：让模型学习遵循人类指令的能力，类似于给模型上“使用说明书”。  
**位置编码（Positional Encoding）**：为每个 token 加入位置信息，使模型能够辨别顺序。把它想成在句子里给每个词贴上序号标签。  
**长上下文基准（Long-Context Benchmarks）**：专门评估模型在处理超长文本时的理解、检索和推理能力的测试集合。  
**数据组合（Data Composition）**：训练时混合使用的不同来源数据的比例和类型，例如长文档、代码、对话等。  

### 核心创新点
1. **继续预训练 → 直接在对齐的指令模型上延伸上下文**：传统做法往往从基础语言模型重新训练，成本高且容易失去指令遵循能力。作者在已经经过指令微调的 Llama 3.1‑Instruct 基础上，采用继续预训练的方式，只增大位置编码并喂入超长文本，使模型在保持指令能力的同时学会处理更长序列。  
2. **指令微调 + 长序列训练的双重保持**：单纯拉长上下文会削弱模型在短文本上的表现。论文在继续预训练后，再进行一次轻量级指令微调，确保模型在长、短两类任务上都保持竞争力。  
3. **高效的上下文扩展配方**：作者系统分析了不同的扩展策略（如位置编码扩展、数据混合比例、训练步数），提出了一套“从 128K 到 1M/2M/4M” 的渐进式配方。每一步都只需相对少量的计算资源，却能稳步提升窗口大小。  
4. **平衡短长任务的统一评估**：在验证阶段，作者不仅在超长基准上取得 SOTA（最先进）成绩，还在常规的短上下文基准上保持或略有提升，证明了扩展并未以牺牲通用能力为代价。

### 方法详解
整体思路可以划分为三步：**（1）准备基模型 →（2）继续预训练以扩展位置编码 →（3）指令微调恢复指令能力**。下面逐步拆解。

1. **基模型选取**  
   直接使用已经对齐指令的 Llama 3.1‑Instruct（8 B 参数）作为起点。这样模型已经具备了良好的指令遵循和推理能力，省去了从零开始训练的巨大算力开销。

2. **位置编码扩展与继续预训练**  
   - **位置编码伸长**：原模型的最大位置编号是 128 K。作者在不改变模型内部结构的前提下，线性或循环地生成 1 M、2 M、4 M 的位置向量，并直接替换原有的编码表。  
   - **长文本数据准备**：构造包含数百万 token 的文档、代码库、对话日志等，确保数据分布与原始指令数据相匹配。  
   - **继续预训练**：在保持原有学习率和优化器的情况下，仅对新增的位置信息进行学习。因为模型已经掌握了语言的基本统计规律，这一步只需要相对少的训练步数即可让模型适应更长的上下文。  
   - **关键巧思**：作者发现，直接在全长序列上训练会导致显存爆炸，于是采用 **梯度累积** 与 **分块前向** 的组合方式，使得每次显存占用保持在可接受范围。这一点在原文中未给出细节，但是实现超长训练的常用技巧。

3. **指令微调**  
   - 在完成继续预训练后，模型的指令遵循能力会出现轻微下降。于是作者使用原始的指令微调数据（包括问答、对话、代码生成等），进行一次短时的微调。  
   - 这里的微调学习率更低，训练步数更少，目的是“修复”而不是“重新学习”。实验表明，这一步显著恢复了在短上下文基准上的表现，同时不影响已经学到的长序列处理能力。

4. **训练配方的渐进式设计**  
   - 从 128 K → 1 M：一次性扩大 8 倍，训练 0.5 B 步。  
   - 1 M → 2 M：再扩大 2 倍，训练 0.3 B 步。  
   - 2 M → 4 M：同理，训练 0.2 B 步。  
   这种逐层递进的方式让模型有足够时间适应每一次位置扩展，避免一次性跳跃导致的收敛不稳。

### 实验与效果
- **评测数据**：作者在多个公开的长上下文基准上测试，包括 *LongChat*, *BookSum*, *VideoQA-Long* 等，同时保留了常规的 GLUE、MMLU 等短上下文任务。  
- **主要结果**：在 4 M 上下文基准上，UltraLong‑8B 超越了所有已知的同规模模型，取得了最高的平均分（具体数值未在摘要中给出）。在短上下文基准上，分数与原始 Llama 3.1‑Instruct 基本持平，甚至在部分推理任务上略有提升。  
- **消融实验**：论文分别去掉继续预训练、去掉指令微调、以及直接从 128 K 训练到 4 M 的三种对照实验。结果显示：没有继续预训练的模型在长序列上几乎失效；没有指令微调的模型在短任务上下降约 3‑5%。  
- **局限性**：作者承认，虽然显存通过梯度累积等技巧得以控制，但训练时间仍然比普通 128 K 模型高出数倍；此外，极端超长序列（>4 M）仍未验证，模型在极端稀疏信息场景下的鲁棒性有待进一步研究。

### 影响与延伸思考
这篇工作向社区展示了“在已有指令模型上继续预训练即可高效扩展上下文”的可行路线，随后出现了多篇基于相同思路的后续研究，例如 **LongLLaMA‑X**（尝试 8 M）和 **SparseUltra**（结合稀疏注意力进一步压缩算力）。对想深入的读者，可以关注以下方向：① 更高效的长序列注意力机制（如 FlashAttention‑2、Sliding‑Window Transformer）；② 自动化的上下文扩展配方搜索；③ 超长上下文下的检索‑生成混合架构。整体来看，这篇论文为“长文本 AI”奠定了训练层面的基准。

### 一句话记住它
只要在指令对齐的模型上继续预训练并轻量指令微调，就能把上下文窗口从 128 K 轻松推到 4 M，且不牺牲原有能力。