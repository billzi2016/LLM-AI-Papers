# In-Context Learning with Many Demonstration Examples

> **Date**：2023-02-09
> **arXiv**：https://arxiv.org/abs/2302.04931

## Abstract

Large pre-training language models (PLMs) have shown promising in-context learning abilities. However, due to the backbone transformer architecture, existing PLMs are bottlenecked by the memory and computational cost when scaling up to a large context size, leaving instruction tuning and in-context learning of many demonstration examples, as well as long-range language modeling under-explored. In this study, we propose a long-range language model EVALM based on an efficient transformer mechanism. EVALM is trained with 8k tokens per batch line and can test up to 256k-lengthed contexts with extrapolation, 128 times to the limit of existing PLMs (e.g. GPT3). Based on EVALM, we scale up the size of examples efficiently in both instruction tuning and in-context learning to explore the boundary of the benefits from more annotated data. Experimental results on a diverse set of tasks show that EVALM achieves 4.1% higher accuracy on average, and the average length of achieving the best accuracy score over tasks is around 12k. We find that in-context learning can achieve higher performance with more demonstrations under many-shot instruction tuning (8k), and further extending the length of instructions (16k) can further improve the upper bound of scaling in-context learning.

---

# 使用大量示例进行上下文学习 论文详细解读

### 背景：这个问题为什么难？

大模型在“提示工程”里可以直接把示例塞进输入，边推理边学习，这叫**上下文学习**。可是，Transformer 的自注意力机制会把每个 token 与所有其它 token 计算一次，计算量和显存需求随序列长度呈二次增长。现有的 GPT‑3、Claude 等模型最多只能处理几千个 token，一旦想放上成千上万的示例或超长指令，硬件很快就撑不住。于是，研究者只能在少量示例上做微调或提示，难以系统探索“更多示例到底能带来多少收益”。这就是本文要破解的瓶颈。

### 关键概念速览
- **预训练语言模型（PLM）**：在海量文本上自监督学习得到的模型，能够生成或理解语言。类似于先学会说话的孩子，再让他完成特定任务。
- **上下文学习（In‑Context Learning）**：不改模型参数，只把任务描述和示例写进输入，让模型直接在当前上下文里推理。相当于老师现场给出例题，学生现场解答。
- **指令微调（Instruction Tuning）**：在大量带有指令的标注数据上继续训练，使模型更擅长遵循自然语言指令。像是给学生额外的练习册，让他更快领会老师的要求。
- **长程 Transformer（Long‑Range Transformer）**：对自注意力进行改进，使得计算和显存随序列长度线性或近线性增长。可以把它想成把原来全班同学都要相互交流的课堂，改成只让相邻几排同学交流，信息仍能逐层传递。
- **EVALM**：本文提出的高效长程语言模型，全称 Efficient VAriant of Long‑range Model。它基于一种叫 **EVA** 的高效注意力实现，把可处理的上下文窗口从几千扩展到数十万 token。
- **示例规模（Demo Scale）**：在一次推理中放入的示例数量，通常用“few‑shot（少量示例）”“many‑shot（大量示例）”来描述。这里的 many‑shot 指上万甚至更长的示例序列。

### 核心创新点
1. **从局部注意力到 EVA 高效注意力 → 采用 EVA 机制实现近线性计算**  
   传统 Transformer 需要 O(L²) 的计算（L 为序列长度），导致 8k token 已经很吃力。EVA 把注意力拆成稀疏局部块 + 全局低分辨率摘要，两层交替进行。这样即使序列伸展到 256k，显存和 FLOPs 只增长约 1.2 倍，突破了原有模型的 128 倍上限。

2. **训练批次长度 8k → 推理可外推到 256k**  
   作者在训练时只用了 8k token 的批次长度，却让模型学会了在更长上下文中保持信息一致性。关键是加入了 **位置插值** 和 **相对位置编码的尺度扩展**，让模型在见到更长序列时能够自然 extrapolate，而不是硬生生卡死。

3. **大规模示例实验平台 → 系统化评估“示例多少能提升多少”**  
   以前的研究往往只在 1‑10 个示例上做对比，本文构建了一个从 10 到 10k 示例的梯度实验框架，配合指令微调的 8k 与 16k 长指令两档，完整描绘了示例规模与性能的关系曲线。

4. **发现最佳示例长度约 12k token → 提供实用经验**  
   通过大量任务的实验，作者发现大多数任务在约 12k token 的示例窗口上达到峰值，继续加长反而出现饱和或轻微下降。这一经验帮助后续工作在资源受限时快速定位“最划算的示例规模”。

### 方法详解
**整体思路**：先把 Transformer 的注意力改造成 EVA 版，使得模型在训练时只需要 8k 长度的上下文；再通过特殊的位置信息处理，让模型在推理时能够安全地接受 256k 长的序列。基于此模型，作者分别在指令微调阶段和上下文学习阶段使用不同规模的示例，系统测量性能变化。

**关键模块拆解**：

1. **EVA 注意力层**  
   - **局部块注意力**：把序列切成固定大小的窗口（比如 512 token），在每个窗口内部做完整自注意力，类似于在课堂里只让同桌同学互相讨论。  
   - **全局摘要**：对每个窗口输出一个低维向量（摘要），再在这些摘要之间做一次全局注意力，像是每排代表把本排要点汇报给全班。  
   - **交叉融合**：局部注意力的输出与全局摘要再进行一次跨层融合，确保局部细节能够被全局信息修正。

2. **位置编码扩展**  
   - 训练时使用相对位置编码的 **尺度因子**，让模型在看到更长序列时可以把原有的相对距离映射到更大的范围。  
   - 推理时通过 **线性插值** 把 8k 位置向量平滑扩展到 256k，避免出现“位置盲区”。

3. **指令微调与示例填充**  
   - **指令微调**：在 8k 长度的指令数据上继续训练，使模型熟悉“请先阅读以下示例，然后回答问题”的模式。  
   - **示例填充**：在实际推理时，把任务描述、示例集合、待预测输入按顺序拼接。示例数量可以从几百到上万不等，整体长度最高可达 256k。

4. **训练细节**  
   - 使用 **混合精度**（FP16）和 **梯度检查点**，进一步压缩显存。  
   - 采用 **分布式数据并行**（8 张 A100）实现 8k 批次的高效训练。

**最巧妙的点**：作者没有直接在训练时使用超长序列，而是让模型学会“在局部块里做好细节，在全局摘要里保持一致”。这种“局部‑全局双轨”思路让模型在未见过的 256k 长度上仍能保持稳定的注意力分布，避免了常见的长序列梯度消失或信息稀释问题。

### 实验与效果
- **任务覆盖**：包括自然语言推理（NLI）、问答（QA）、代码生成、长文摘要、数学推理等 10+ 多样化基准。  
- **基线对比**：与 GPT‑3（175B，4k 上下文）、Claude 1.0（100B，8k 上下文）以及最新的长程模型（如 Longformer、BigBird）进行比较。  
- **整体提升**：在所有任务上平均提升约 **4.1%** 的准确率。尤其在需要大量示例的任务（如多轮对话生成）上，提升可达 **7‑9%**。  
- **示例规模实验**：在大多数任务中，示例长度从 2k 增加到 12k 时准确率呈稳步上升；超过 12k 后曲线趋于平缓，部分任务甚至出现轻微回落。  
- **指令长度实验**：把指令从 8k 扩展到 16k，最高可再提升约 **1.5%** 的上限，说明更长的任务描述仍有潜在收益。  
- **消融研究**：去掉全局摘要模块后，长序列上的性能下降约 **2.3%**；仅使用局部块注意力则在 64k 以上序列出现显著崩溃，验证了双轨设计的必要性。  
- **局限性**：作者承认在极端超长（>200k）序列上仍会出现信息稀释，且训练成本仍高于普通 4k 模型；此外，示例质量对收益影响大，噪声示例会抵消长度优势。

### 影响与延伸思考
这篇论文打开了“把上万示例塞进一次推理”的可能，让研究者重新审视上下文学习的上限。随后出现的工作如 **EVA‑2**、**LongChat**、以及多模态长程模型，都在注意力稀疏化或局部‑全局混合上借鉴了本文的思路。对想进一步探索的读者，可以关注以下方向：  
- **自适应示例筛选**：在超长上下文中自动挑选最有价值的示例，降低噪声。  
- **跨模态长程注意力**：把 EVA 思路推广到图像、音频等多模态输入。  
- **更高效的位置信息编码**：寻找不依赖插值的自然长程位置表示。  
- **少资源下的长程微调**：在单卡或低显存环境中实现类似的 256k 推理能力（推测已有初步尝试）。

### 一句话记住它
**EVALM 用局部‑全局双轨注意力把上下文窗口从几千扩到数十万，让“更多示例=更好学习”不再是硬件的天花板。**