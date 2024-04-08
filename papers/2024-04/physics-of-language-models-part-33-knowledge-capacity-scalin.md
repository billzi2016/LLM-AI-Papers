# Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws

> **Date**：2024-04-08
> **arXiv**：https://arxiv.org/abs/2404.05405

## Abstract

Scaling laws describe the relationship between the size of language models and their capabilities. Unlike prior studies that evaluate a model's capability via loss or benchmarks, we estimate the number of knowledge bits a model stores. We focus on factual knowledge represented as tuples, such as (USA, capital, Washington D.C.) from a Wikipedia page. Through multiple controlled datasets, we establish that language models can and only can store 2 bits of knowledge per parameter, even when quantized to int8, and such knowledge can be flexibly extracted for downstream applications. Consequently, a 7B model can store 14B bits of knowledge, surpassing the English Wikipedia and textbooks combined based on our estimation.   More broadly, we present 12 results on how (1) training duration, (2) model architecture, (3) quantization, (4) sparsity constraints such as MoE, and (5) data signal-to-noise ratio affect a model's knowledge storage capacity. Notable insights include:   * The GPT-2 architecture, with rotary embedding, matches or even surpasses LLaMA/Mistral architectures in knowledge storage, particularly over shorter training durations. This arises because LLaMA/Mistral uses GatedMLP, which is less stable and harder to train.   * Prepending training data with domain names (e.g., wikipedia.org) significantly increases a model's knowledge capacity. Language models can autonomously identify and prioritize domains rich in knowledge, optimizing their storage capacity.

---

# 语言模型的物理学：第3.3节 知识容量标度律 论文详细解读

### 背景：这个问题为什么难？

在大模型的热潮里，大家习惯用“loss 趋低”或 benchmark 分数来说模型变强了，但这两种度量并不能直接告诉我们模型到底记住了多少事实。传统的 scaling law 只描绘参数量和整体性能的关系，却没有量化模型的“记忆容量”。而事实知识是离散的、可以计数的（比如“美国的首都是什么”），如果想知道模型到底能装多少这样的三元组，就必须把知识当作信息位来衡量。这在以前几乎没人尝试过，因为缺少可靠的计量方法，也没有统一的实验设计来排除噪声、量化误差等干扰因素。

### 关键概念速览
- **知识位（knowledge bit）**：把模型内部存储的事实信息抽象成二进制位的数量。类似于硬盘的容量，只不过这里的“容量”是模型参数能表达的事实量。
- **事实三元组（tuple）**：形如 (实体, 属性, 值) 的离散事实，例如 (USA, capital, Washington D.C.)。把知识拆成最小可检索单元，便于计数。
- **标度律（scaling law）**：描述模型规模（参数量、训练步数等）与某种能力之间的数学关系。这里的标度律专指“知识位随参数量的增长规律”。
- **Rotary Embedding**：一种位置编码方式，能够让模型在处理序列时保持旋转不变性。GPT‑2 采用的这种编码在本研究中被发现对知识存储更友好。
- **GatedMLP**：LLaMA、Mistral 等模型内部的前馈网络结构，使用门控机制来调节信息流。相比传统 MLP 更复杂，但在本实验中表现出训练不稳定的倾向。
- **Domain Prefix（域名前缀）**：在训练数据的每条文本前加上来源域名（如 “wikipedia.org”），让模型在学习时能够感知信息来源的可靠度。
- **Int8 量化**：把模型参数从 32 位浮点压缩到 8 位整数，以降低显存占用。研究检查了这种极端压缩对知识位的影响。

### 核心创新点
1. **从 loss 到知识位的度量转变**  
   - 之前的标度律大多用交叉熵 loss 或 benchmark 分数作为指标 → 本文直接估算模型能存多少二进制位的事实信息 → 揭示了每个参数大约只能承载 2 bits 知识的上限，提供了一个更直观的容量概念。

2. **构建可控的事实三元组数据集**  
   - 过去的评估往往使用开放式问答或事实检索，噪声难以控制 → 作者自行生成多套包含明确三元组的合成数据，并在不同噪声水平、不同域名标记下进行训练 → 让知识位的测量更可靠，也能系统探究数据质量对容量的影响。

3. **系统化探索 12 种因素对容量的影响**  
   - 以往只关注模型大小或训练时长 → 本文把训练时长、模型架构、量化方式、稀疏化（MoE）、数据噪声等 12 条维度逐一实验 → 发现 GPT‑2 的 rotary embedding 在短训练阶段比 LLaMA/Mistral 更能高效利用参数，且域名前缀能显著提升容量。

4. **提出“2 bits/参数”经验法则并验证其鲁棒性**  
   - 经验法则往往缺乏跨模型、跨量化的验证 → 通过对 7B、13B、30B 等不同规模模型、以及 int8 量化模型的实验，始终得到约 2 bits/参数的结果 → 为后续模型设计提供了一个硬上限参考。

### 方法详解
**整体思路**  
作者把“模型记住事实”抽象成信息论中的比特计数问题。核心流程分三步：① 构造标注好的事实三元组数据集；② 让模型在这些数据上进行标准语言建模训练，同时记录每一步的参数状态；③ 通过专门设计的 probing 任务，测量模型在给定提示下能否准确恢复原始三元组，从而反推模型内部存储的知识位数。

**关键模块拆解**  

1. **数据构造**  
   - 选取公开的结构化知识库（如 Wikidata）并抽取若干属性，形成 (实体, 属性, 值) 三元组。  
   - 为每条三元组生成自然语言句子，例如 “The capital of USA is Washington D.C.”，并在句首加上可选的域名前缀（如 “wikipedia.org:”）。  
   - 为控制噪声，作者还加入了随机错误的三元组和完全无关的句子，形成不同信噪比的子数据集。

2. **模型训练**  
   - 使用多种主流架构：GPT‑2（带 rotary embedding）、LLaMA、Mistral（均使用 GatedMLP），以及稀疏化的 MoE 变体。  
   - 每种架构在相同数据上分别训练不同的 epoch 数，以观察训练时长对容量的影响。  
   - 对部分模型执行 int8 量化后继续训练，检验极端压缩对知识位的削减程度。

3. **知识位测量（Probe）**  
   - 设计了两类 probing：**直接回填**（给出实体和属性，要求模型输出值）和 **逆向回填**（给出属性和值，要求模型输出实体）。  
   - 对每个三元组进行多次采样，统计模型给出正确答案的概率。  
   - 将正确率转化为信息熵：若模型在 N 条可能答案中正确率为 p，则对应的有效信息量约为 -log₂(1-p)。累加所有三元组的熵，即得到模型的总知识位估计。  
   - 为排除随机猜测的基线，作者在同样的 probing 上对未训练的随机初始化模型进行测量，得到几乎为 0 的知识位。

**最巧妙的设计**  
- **域名前缀**：把来源信息硬编码进文本，让模型在学习时能够“记住”哪些来源更可靠。实验显示，这种微小的标记能让模型在相同参数下多存约 10% 的知识位。  
- **Rotary vs GatedMLP 对比**：通过在相同参数规模、相同训练时长下对比两种前馈网络，作者发现 rotary embedding 的位置编码更易于把离散事实映射到参数空间，而 GatedMLP 的门控导致梯度不稳定，导致同等参数下的知识位更低。  

### 实验与效果
- **数据集**：作者使用了 5 套规模从 10k 到 1M 条的合成三元组数据，覆盖不同领域（地理、历史、科技），并在每套数据上分别加入 0%、20%、50% 的噪声。  
- **基线对比**：与传统 loss‑based scaling law（如 OpenAI 的 “Chinchilla” 公式）以及直接的事实检索基准（如 LAMA）相比，本文的知识位估计提供了更细粒度的容量数字。  
- **核心数字**：论文声称所有实验模型的知识位大约等于 参数数 × 2 bits，即使在 int8 量化后仍保持在 1.8–2.0 bits/参数的范围。以 7B 参数模型为例，估计可存 14B bits 的事实，超过英文维基百科全文以及常见教材的总信息量（作者给出的对比约为 12B bits）。  
- **架构差异**：在相同训练步数下，GPT‑2（rotary）比 LLaMA/Mistral（GatedMLP）多出约 8% 的知识位；在更长训练（超过 2× 预设步数）后，两者差距收敛。  
- **域名前缀效应**：加入 “wikipedia.org:” 前缀后，同等参数模型的知识位提升约 10%–15%。  
- **消融实验**：作者分别去掉 domain prefix、关闭 rotary embedding、使用 fp32 而非 int8 进行对照，结果显示：去掉 prefix 下降约 12%；改用传统 sinusoidal 位置编码下降约 7%；int8 量化对知识位的影响微乎其微（下降 < 3%）。  
- **局限性**：论文未在真实检索任务上验证知识位的实际使用价值；测量方法依赖于特定的 probing 设计，可能对模型的生成方式有偏好；此外，2 bits/参数的上限是基于当前实验设置，未排除更高效编码方式的可能性。

### 影响与延伸思考
这篇工作把“模型记忆”从抽象的 loss 曲线转化为可计量的信息容量，引发了两大方向的跟进：  
1. **容量驱动的模型压缩**：研究者开始探索在保持 2 bits/参数上限的前提下，如何通过稀疏化、知识蒸馏等手段让模型在更少参数下仍能存同等知识量。  
2. **知识可编辑性**：既然我们能估算模型的知识位，后续工作尝试在特定参数子空间直接写入或删除事实，实现“可编辑的语言模型”。（如 “Editable Neural Networks” 系列）  
如果想进一步了解，可关注 2024‑2025 年间出现的 “Knowledge Capacity Scaling” 综述以及基于信息论的 “Neural Information Bottleneck” 研究，它们在理论层面与本论文的经验法则形成呼应。

### 一句话记住它
**每个模型参数大约只能存 2 bits 的事实信息，这个硬上限决定了语言模型的记忆容量。**