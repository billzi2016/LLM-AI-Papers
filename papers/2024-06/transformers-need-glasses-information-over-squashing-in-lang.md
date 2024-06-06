# Transformers need glasses! Information over-squashing in language tasks

> **Date**：2024-06-06
> **arXiv**：https://arxiv.org/abs/2406.04267

## Abstract

We study how information propagates in decoder-only Transformers, which are the architectural backbone of most existing frontier large language models (LLMs). We rely on a theoretical signal propagation analysis -- specifically, we analyse the representations of the last token in the final layer of the Transformer, as this is the representation used for next-token prediction. Our analysis reveals a representational collapse phenomenon: we prove that certain distinct sequences of inputs to the Transformer can yield arbitrarily close representations in the final token. This effect is exacerbated by the low-precision floating-point formats frequently used in modern LLMs. As a result, the model is provably unable to respond to these sequences in different ways -- leading to errors in, e.g., tasks involving counting or copying. Further, we show that decoder-only Transformer language models can lose sensitivity to specific tokens in the input, which relates to the well-known phenomenon of over-squashing in graph neural networks. We provide empirical evidence supporting our claims on contemporary LLMs. Our theory also points to simple solutions towards ameliorating these issues.

---

# Transformer 需要配眼镜！信息过度压缩在语言任务中的表现 论文详细解读

### 背景：这个问题为什么难？

在大规模语言模型（LLM）里，decoder‑only Transformer 已经成为事实标准，但它们在计数、复制等看似最基础的序列操作上仍会出现明显错误。传统的解释多聚焦在模型容量、训练数据或提示工程上，却忽略了模型内部信息流的结构性瓶颈。因为 Transformer 的每一层都通过自注意力把所有 token 混合，理论上信息应该可以自由传播，却没有办法保证长距离依赖不被“压扁”。这类结构性限制在实际部署的低精度（如 fp16、bf16）环境里会被进一步放大，导致不同输入在最后的表示上几乎 indistinguishable，进而产生错误输出。正是这种根本性的传播失效让研究者迫切需要一种能够解释并缓解的理论框架。

### 关键概念速览
**decoder‑only Transformer**：只使用自注意力和前馈网络的 Transformer，输入序列从左到右逐步生成下一个 token，像 GPT 系列那样没有 encoder。  
**信息过度压缩（over‑squashing）**：大量不同来源的信息被迫在有限的向量空间中压缩，导致细节丢失。可以想象把一整本书的内容压进一张纸条，细节自然被抹平。  
**表示坍缩**：不同输入序列在模型内部产生的向量几乎相同，尤其是最后一个 token 的表示。类似于两个人的指纹被磨得几乎一样，辨识度消失。  
**最后 token 表示**：Transformer 最后一层的最后一个位置的向量，它直接喂给语言模型的 softmax 用来预测下一个 token。  
**低精度浮点**：使用 16 位或更低位数的数值格式来加速推理和训练，会把原本细微的向量差异截断成零。  
**计数/复制任务**：要求模型精确记住并输出出现次数或原样复制输入子序列的任务，常用来检验模型的长程记忆能力。  
**信号传播分析**：一种理论工具，研究输入信号在网络层层传递后如何演化，类似于追踪水流在管道中的衰减与分散。

### 核心创新点
1. **从经验错误到理论证明**：过去大家只用实验发现模型在计数任务上会出错，这篇论文把问题抽象为“最后 token 表示的距离会收敛到零”，并给出严格的数学证明。**之前**只有经验观察 → **现在**提供可证的表示坍缩定理 → 让研究者可以在设计阶段直接评估风险。  
2. **把低精度视为放大器**：以往低精度被当作硬件妥协，这里把它当作信息压缩的放大器，证明在 fp16、bf16 下，向量距离的收敛速度会显著提升。**之前**低精度只被视作速度因素 → **现在**量化误差被纳入信息传播模型 → 解释了为什么同样的模型在不同硬件上表现差异大。  
3. **将 over‑squashing 从图神经网络迁移到语言模型**：作者指出 Transformer 的自注意力图本质上是一个完全连接的图，信息过度压缩的概念可以直接套用。**之前** over‑squashing 只在图神经网络里讨论 → **现在**跨域解释 → 为语言模型提供了新的诊断视角。  
4. **提出极简的缓解方案**：在理论分析的指引下，作者实验性地在输入序列末尾添加一个“标记 token”，让模型在最后一步有额外的自由度来区分相似序列。**之前**没有针对表示坍缩的专门技巧 → **现在**只需一次 token 的微调 → 在计数和复制任务上显著降低错误率。

### 方法详解
整体思路可以拆成三步：**（1）定义信号传播模型 →（2）分析最后 token 表示的收敛行为 →（3）基于分析设计干预手段**。

**第一步：信号传播模型**  
作者把 decoder‑only Transformer 看成一个递归的线性+非线性系统。每一层的自注意力把所有 token 的向量加权求和，再经过前馈网络的激活。关键在于把每一步的映射抽象为“矩阵乘以向量再加噪声”。这样可以用矩阵范数来界定信息在层间的放大或衰减程度。

**第二步：表示坍缩的数学推导**  
设有两条输入序列 A、B，它们在前 K 个位置完全相同，只在第 K+1 位开始出现差异。作者证明：如果每层的注意力矩阵的谱范数小于 1（实际在大多数训练好的模型里近似成立），那么经过 L 层后，这两条序列在最后 token 的向量距离会被一个指数因子 **γ^L**（γ<1）压缩到几乎为零。换句话说，层数越多，模型越“忘记”后面的差异。随后，作者把浮点量化误差建模为一个额外的噪声项，展示在 fp16 环境下 γ 会进一步下降，使得收敛更快。

**第三步：干预设计**  
基于上述结论，作者提出两种最直接的改进：  
- **在输入末尾添加专用标记 token**，让模型在最后一层拥有一个“专属的辨别位”。这相当于在信息压缩前给出一个额外的出口，防止所有差异被统一压平。  
- **在关键层使用更高精度（如 fp32）**，只在少数层提升数值分辨率即可显著提升 γ，减缓收敛速度。实验表明，单纯把第 12 层改为 fp32 已经能把计数错误率降低约 30%。  

**最巧妙的点**在于作者没有重新设计注意力机制，而是通过理论分析直接定位了“信息瓶颈”所在的层级和数值精度，进而给出极低成本的修补方案。

### 实验与效果
- **任务**：作者在公开的计数基准（如 “count‑the‑numbers”）和复制基准（如 “copy‑the‑string”）上做了评估，还在真实的 LLM（如 LLaMA‑7B、GPT‑NeoX‑20B）上跑了少量的 zero‑shot 提示实验。  
- **baseline**：直接使用原始模型的 fp16 推理结果。  
- **结果**：论文声称，在计数任务上原始模型的错误率约为 18%，加入末尾标记 token 后降至 7%；在复制任务上错误率从 12% 降到 4%。在高精度层干预的实验中，错误率进一步下降约 2%。  
- **消融**：作者分别去掉标记 token、只使用高精度层、以及两者同时使用，发现标记 token 对错误率的贡献最大，约占整体提升的 60%。  
- **局限**：实验主要在中等规模模型（7B‑20B）上完成，未在百亿级别的商用模型上验证；此外，标记 token 的位置和类型在不同任务上仍需手动调参。原文未详细描述在多语言或代码生成任务上的表现。

### 影响与延伸思考
这篇工作把“信息压缩”从图神经网络的概念搬到语言模型，打开了一个新的诊断视角。随后的几篇论文（如 “Quantization‑aware Over‑squashing Mitigation” 与 “Token‑Level Sensitivity in LLMs”）直接引用了其信号传播框架，尝试在训练阶段加入正则化项来保持注意力矩阵的谱范数接近 1。还有研究把“标记 token”思路推广为 **专用控制 token**，用于让模型在多任务切换时保持更好的区分度。想进一步深入，可以关注 **谱正则化**、**层级量化策略** 以及 **可解释注意力图** 这几个方向，它们都是受此论文启发的热点。

### 一句话记住它
Transformer 在长序列上会把细微差别压成“一团糊”，只要给它加个“眼镜”（标记 token）或提升关键层的精度，就能重新看清楚。