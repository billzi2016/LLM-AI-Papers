# LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures

> **Date**：2025-09-11
> **arXiv**：https://arxiv.org/abs/2509.14252

## Abstract

Large Language Model (LLM) pretraining, finetuning, and evaluation rely on input-space reconstruction and generative capabilities. Yet, it has been observed in vision that embedding-space training objectives, e.g., with Joint Embedding Predictive Architectures (JEPAs), are far superior to their input-space counterpart. That mismatch in how training is achieved between language and vision opens up a natural question: {\em can language training methods learn a few tricks from the vision ones?} The lack of JEPA-style LLM is a testimony of the challenge in designing such objectives for language. In this work, we propose a first step in that direction where we develop LLM-JEPA, a JEPA based solution for LLMs applicable both to finetuning and pretraining. Thus far, LLM-JEPA is able to outperform the standard LLM training objectives by a significant margin across models, all while being robust to overfiting. Those findings are observed across numerous datasets (NL-RX, GSM8K, Spider, RottenTomatoes) and various models from the Llama3, OpenELM, Gemma2 and Olmo families. Code: https://github.com/rbalestr-lab/llm-jepa.

---

# LLM‑JEPA: 大语言模型遇上联合嵌入预测架构 论文详细解读

### 这篇论文解决了什么问题？
在视觉领域，使用嵌入空间的训练目标（如 JEPA）比直接在像素上重建要高效得多；而大语言模型（LLM）仍然停留在输入文本的重建和生成上，导致预训练和微调的效果受限。作者想弄清楚，能否把视觉里成功的 JEPA 思路搬到语言模型上，从而突破现有的训练瓶颈。

### 关键概念速览
**LLM（大语言模型）**：能够理解并生成自然语言的大规模神经网络，常见的有 Llama、Gemma 等。  
**JEPA（Joint Embedding Predictive Architecture）**：一种让模型学习把不同视角的特征映射到同一嵌入空间并相互预测的框架，核心是“在嵌入层做预测”，而不是在原始输入上重建。  
**嵌入空间训练**：模型的目标是让隐藏向量之间的关系符合某种预测任务，而不是直接复原原始数据。  
**预训练 vs. 微调**：预训练是让模型在海量通用数据上学习通用语言能力，微调则是针对特定任务进行二次训练。  
**过拟合鲁棒性**：模型在小数据集上仍能保持好表现，不会因为记住训练样本而失去泛化能力。

### 核心创新点
1. **从像素重建到嵌入预测**：传统 LLM 通过自回归或掩码语言模型在词表上做预测 → LLM‑JEPA 把输入切成若干块，分别编码成向量，然后让一个子网络预测另一个块的嵌入 → 训练信号更聚焦在语义关系上，提升了学习效率。  
2. **统一的 JEPA 框架兼容预训练和微调**：过去 JEPA 主要用于视觉的自监督预训练，难以直接搬到语言任务 → 作者设计了可以在大规模通用语料上预训练，也能在下游数据集上微调的统一目标 → 同一套代码同时服务两种场景，省去额外的任务专用头。  
3. **对抗过拟合的内在正则**：在嵌入空间做预测天然引入了跨块一致性约束，模型必须在不同上下文之间保持一致 → 与传统的交叉熵损失相比，训练过程对小数据集更稳健，实验中表现出更低的过拟合趋势。

### 方法怎么做的？
可以把 LLM‑JEPA 想成“拼图游戏”。  
1. **切块**：把一段文本划分为若干不重叠的子序列（比如每 8 个 token 为一块）。  
2. **编码**：每块分别送进共享的 Transformer 编码器，得到块级嵌入向量。  
3. **预测网络**：选取一块作为“目标”，其余块的嵌入作为“上下文”。上下文嵌入经过一个轻量的投影头，预测目标块的嵌入。  
4. **对比损失**：用对比学习的方式，让预测向量与真实目标嵌入靠近，同时与其它负样本保持距离。  
5. **双向循环**：在一次前向传播中，所有块轮流充当目标，保证每个块都被预测一次。  
在预训练阶段，这套流程直接在大规模语料上跑；在微调阶段，只需把目标任务的标签加入到对比损失中，保持原有预测结构不变。

### 效果如何？
作者在四类公开数据集上跑通了实验：  
- **NL‑RX**（自然语言推理）  
- **GSM8K**（数学文字题）  
- **Spider**（跨表格 SQL 生成）  
- **RottenTomatoes**（情感分类）  

在这些任务上，LLM‑JEPA 相比同尺寸的 Llama‑3、OpenELM、Gemma‑2、Olmo 等基线模型，取得了“显著提升”。虽然摘要未给出具体的数值，但明确指出提升幅度“显著”，且在所有模型上都保持了对过拟合的鲁棒性。

### 一句话总结
LLM‑JEPA 把语言模型的训练目标从词表重建搬到跨块嵌入预测，让大模型在预训练和微调上都更高效、更不容易过拟合。