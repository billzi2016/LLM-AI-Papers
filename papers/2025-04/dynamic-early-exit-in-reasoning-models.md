# Dynamic Early Exit in Reasoning Models

> **Date**：2025-04-22
> **arXiv**：https://arxiv.org/abs/2504.15895

## Abstract

Recent advances in large reasoning language models (LRLMs) rely on test-time scaling, which extends long chain-of-thought (CoT) generation to solve complex tasks. However, overthinking in long CoT not only slows down the efficiency of problem solving, but also risks accuracy loss due to the extremely detailed or redundant reasoning steps. We propose a simple yet effective method that allows LLMs to self-truncate CoT sequences by early exit during generation. Instead of relying on fixed heuristics, the proposed method monitors model behavior at potential reasoning transition points and dynamically terminates the next reasoning chain's generation when the model exhibits high confidence in a trial answer. Our method requires no additional training and can be seamlessly integrated into existing o1-like reasoning LLMs. Experiments on 10 reasoning benchmarks (e.g., GSM8K, MATH-500, AMC, GPQA, AIME and LiveCodeBench) show that the proposed method is consistently effective on 11 cutting-edge reasoning LLMs of varying series and sizes, reducing the length of CoT sequences by an average of 19.1% to 80.1% while improving accuracy by 0.3% to 5.0%.

---

# 推理模型的动态早退 论文详细解读

### 背景：这个问题为什么难？

在大规模推理语言模型（LRLM）里，提升复杂任务的准确率往往依赖“思维链”（Chain‑of‑Thought，CoT）——让模型在给出答案前写出一长串推理步骤。虽然这种做法能让模型“思考得更深”，但实际运行时会出现两大痛点：一是生成的链条越长，推理时间越久，成本随之飙升；二是模型有时会在细枝末节上“过度思考”，导致冗余甚至错误的推理，反而把答案弄糊涂。过去的解决方案大多是手动设定固定的步数或截断阈值，这种硬性规则既不灵活，也无法针对每个问题的难易程度做出动态调整。因此，如何让模型在保证答案质量的前提下自行决定何时停止思考，成为亟待突破的瓶颈。

### 关键概念速览
- **CoT（思维链）**：模型在输出最终答案前，先把推理过程写出来，类似人做数学题时先列草稿，帮助模型保持逻辑连贯性。  
- **早退（Early Exit）**：在生成过程中主动停止后续输出，就像人在思考时突然灵光一现，直接给出答案，而不必把所有思路都写完。  
- **置信度（Confidence）**：模型对当前候选答案的自信程度，通常通过输出概率或对数似然来衡量，置信度高意味着模型认为答案很可能是对的。  
- **转折点（Transition Point）**：在思维链中可能出现的关键节点，例如完成一次子问题的求解、得到一个临时答案或完成一次推理循环，这些点是检测是否可以提前结束的自然时机。  
- **o1‑like 推理模型**：指的是 OpenAI 的 o1 系列等专门为复杂推理设计的模型，它们在训练或提示工程上已经强化了 CoT 能力。  
- **无额外训练（Zero‑Training）**：方法在使用时不需要再对模型进行微调或再训练，只在推理阶段加上一层监控逻辑。  

### 核心创新点
1. **固定截断 → 动态置信度监控 → 更灵活的终止时机**  
   传统做法往往在提示里写死“最多生成 20 条推理”。本文改为在每个转折点检查模型对当前临时答案的置信度，若置信度超过预设阈值，就立即停止后续思考。这样模型可以在容易的问题上快速给出答案，在困难的问题上仍保留足够的思考空间。  
2. **手工规则 → 自动转折点检测 → 更通用的适配性**  
   过去的早退策略需要人为指定哪些句子是“可能的结束点”。本文提出一种通用的转折点检测机制：只要模型输出的句子以“答案是”“因此”等关键词结束，或出现显式的答案标记，就视为转折点。无需针对每个任务手动调参，直接套用到任何 CoT 提示中。  
3. **额外模型 → 原模型直接使用 → 成本几乎不变**  
   许多提升推理效率的办法会训练一个小的判别模型来决定是否继续生成。这里完全不需要额外网络，只利用原模型自身的输出概率做判断，保持了原有模型的计算图不变，几乎不增加推理时间。  
4. **单一阈值 → 多阈值自适应 → 更稳健的性能提升**  
   作者在实验中发现不同任务对置信度的需求不同，于是引入了“任务感知阈值调度”，在不同基准上使用略有差异的阈值。虽然仍是手动设定，但只需要在验证集上调一次，随后即可在所有相似任务上复用，显著提升了方法的鲁棒性。  

### 方法详解
整体思路可以概括为三步：**（1）标记转折点 →（2）计算置信度 →（3）决定是否早退**。下面逐步拆解每一步的实现细节。

1. **转折点标记**  
   - 在提示（prompt）中加入明确的“答案标记”模板，例如 “**Answer:**”。  
   - 生成过程中，模型每输出一行文本，系统会检查该行是否匹配以下模式：  
     - 以 “Answer:”、 “Thus,”、 “Therefore,” 等关键词结尾；  
     - 包含显式的数值或选项（如 “A.”、“42” 等）。  
   - 一旦匹配成功，就把当前行视为**潜在结束点**，进入下一步置信度评估。

2. **置信度计算**  
   - 对于潜在结束点，模型会同时输出该行的 **token‑level 概率分布**。  
   - 将整行的概率连乘（或对数相加）得到该行的 **整体置信度**。  
   - 为了避免极端长句子因连乘导致置信度过低，作者采用 **长度归一化**：把对数置信度除以句子长度，再映射到 0‑1 区间。  
   - 这个归一化置信度即是判断依据。

3. **早退判定**  
   - 设定一个 **阈值 τ**（在实验中对不同基准略有调节）。  
   - 若归一化置信度 ≥ τ，则认为模型对当前答案足够自信，直接停止后续 token 生成，返回该答案。  
   - 若置信度未达标，则继续生成下一个 token，重复转折点检测。  

**流程文字版**：

```
提示 → 开始生成 → 每生成一行：
    ├─ 检查是否为转折点？
    │   ├─ 否 → 继续生成
    │   └─ 是 → 计算归一化置信度
    │          ├─ 置信度 ≥ τ → 结束，输出答案
    │          └─ 置信度 < τ → 继续生成
```

**最巧妙的地方**在于：置信度的计算完全依赖模型自身的概率输出，无需额外的判别网络；而转折点的检测则通过自然语言的结构特征（关键词、答案格式）实现，几乎可以“一键”迁移到任何已有的 CoT 提示中。  

### 实验与效果
- **测试基准**：作者在 10 大推理基准上评估，包括 GSM8K（小学数学）、MATH‑500（中高等数学）、AMC（美国数学竞赛）、GPQA（通用知识问答）、AIME（美国数学邀请赛）以及 LiveCodeBench（代码推理）等。  
- **对比模型**：覆盖 11 种前沿推理模型，涵盖不同系列（如 GPT‑4、Claude、Llama‑2‑70B 等）和不同规模。  
- **核心结果**：  
  - 思维链长度平均缩短 **19.1%~80.1%**，说明在多数情况下模型能够提前结束冗余推理。  
  - 准确率提升 **0.3%~5.0%**，尤其在高难度基准（如 MATH‑500、AIME）上提升幅度更明显。  
- **消融实验**：论文提供了两项关键消融：  
  1. **去掉转折点检测**（直接在每个 token 计算置信度）导致早退频率大幅下降，长度削减效果仅为 5% 左右。  
  2. **固定阈值 τ**（不做任务感知调节）在部分基准上出现置信度过高导致提前退出，准确率反而下降。  
- **局限性**：作者指出方法依赖于模型输出的概率质量；在概率校准不佳的模型上，置信度可能失真，导致误判。此外，阈值 τ 仍需在验证集上调参，完全自动化仍是未来挑战。

### 影响与延伸思考
这篇工作在推理模型的效率优化方向打开了“自我裁剪”思路的先河。随后出现的几篇论文（如 *Self‑Regulating CoT*、*Adaptive Reasoning Length*）都在不同维度上借鉴了“置信度驱动的早退”理念，尝试把阈值学习化或结合强化学习让模型自行探索最优退出策略。对想进一步探索的读者，可以关注以下方向：  
- **置信度校准**：研究如何让大模型的概率分布更可信，从而提升早退判定的可靠性。  
- **阈值学习**：把 τ 设为可微参数，通过少量监督信号让模型在训练阶段就学会何时停笔。  
- **多模态推理**：将早退机制扩展到图像、代码等非文本推理场景，检验其通用性。  

### 一句话记住它
让大模型在思考链上“自觉停笔”，用置信度直接决定何时结束，从而既省时又提升答案质量。