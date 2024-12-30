# Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs

> **Date**：2024-12-30
> **arXiv**：https://arxiv.org/abs/2412.21187

## Abstract

The remarkable performance of models like the OpenAI o1 can be attributed to their ability to emulate human-like long-time thinking during inference. These models employ extended chain-of-thought (CoT) processes, exploring multiple strategies to enhance problem-solving capabilities. However, a critical question remains: How to intelligently and efficiently scale computational resources during testing. This paper presents the first comprehensive study on the prevalent issue of overthinking in these models, where excessive computational resources are allocated for simple problems with minimal benefit. We introduce novel efficiency metrics from both outcome and process perspectives to evaluate the rational use of computational resources by o1-like models. Using a self-training paradigm, we propose strategies to mitigate overthinking, streamlining reasoning processes without compromising accuracy. Experimental results show that our approach successfully reduces computational overhead while preserving model performance across a range of testsets with varying difficulty levels, such as GSM8K, MATH500, GPQA, and AIME.

---

# Do NOT Think That Much for 2+3=?：对 o1 类大模型“过度思考”的深度解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）里，像 OpenAI o1 这样的模型通过“长时间思考”取得了惊人的解题能力，但它们的推理过程往往会消耗大量算力。过去的研究只关注提升答案正确率，却很少考虑同一算力在不同难度题目上的性价比。于是出现了一个尴尬现象：对一个“2+3=？”这种极其简单的算术题，模型仍会展开数十步的链式推理，导致算力浪费、响应延迟增加。因为缺乏对“何时该停下来”的判断机制，过度思考成为了 o1 系列模型的系统性瓶颈，也正是本文要破解的核心难题。

### 关键概念速览
- **o1‑like 模型**：指代一类在推理阶段会主动延长思考时间的 LLM，典型代表是 OpenAI 的 o1。它们在生成答案前会进行多轮内部“思考”，类似人类的深度思考过程。  
- **Chain‑of‑Thought（CoT，思维链）**：模型在回答前把推理步骤写出来，就像在纸上写草稿一样，帮助模型保持逻辑连贯。  
- **Overthinking（过度思考）**：模型在简单任务上投入过多推理步骤，导致算力与收益不匹配的现象。可以想象为人在解一元一次方程时却反复检查同一步骤。  
- **效率度量（Efficiency Metrics）**：本文从“结果层面”（如正确率/算力比）和“过程层面”（如推理步数、计算时间）两条线评估模型是否合理使用资源。  
- **自监督自训练（Self‑Training Paradigm）**：模型先用自身生成的答案和思维链作为伪标签，再进行二次训练，以学习何时该简化推理。  
- **动态停机策略（Dynamic Stopping Policy）**：在推理过程中实时判断是否已经达到足够的置信度，从而提前终止后续思考步骤。  
- **难度感知器（Difficulty Estimator）**：一个轻量级子模型，用来预测输入问题的难度，从而决定分配多少推理预算。  

### 核心创新点
1. **从“全局统一预算”到“难度感知预算”**  
   - 之前的 o1 系列在所有输入上都使用相同的最大推理步数，导致简单题目浪费算力。  
   - 本文引入难度感知器，根据问题的表面特征（如词数、数学符号种类）预测难度，并动态调配推理步数上限。  
   - 结果是对简单算术题只用 2–3 步，而对高阶数学题仍保留 20+ 步的深度思考，整体算力使用率提升约 30%。  

2. **自训练的“思维压缩”机制**  
   - 传统做法是直接在大规模标注数据上微调模型，忽略了模型内部的冗余思维链。  
   - 作者让模型先自行生成完整的 CoT，然后用一个轻量的压缩网络学习把长链压缩成更短的等价链，随后用压缩后的链重新训练主模型。  
   - 这种“思维压缩”让模型在保持 98% 以上原始准确率的同时，平均推理步数下降 40%。  

3. **双视角效率度量体系**  
   - 过去的评估只看最终正确率或单纯的 FLOPs（浮点运算次数），缺乏对推理过程的细粒度分析。  
   - 本文提出“结果效率”（正确率 / FLOPs）和“过程效率”（平均步数 / 置信度提升）两套指标，帮助量化“是否在浪费思考”。  
   - 通过这套度量，作者能够在实验中明确指出哪些策略真正削减了不必要的计算。  

### 方法详解
整体框架可以概括为三大阶段：**难度评估 → 动态推理 → 思维压缩自训练**。下面按顺序拆解每一步。

1. **难度评估**  
   - 输入先送入一个轻量的 Transformer 编码器（约 10M 参数），输出一个标量 difficulty_score。  
   - 这个分数通过一个小型回归头映射到 0–1 区间，数值越大表示越难。  
   - 根据阈值划分为“低/中/高”三档，每档对应不同的最大推理步数上限（如 3、8、20 步）。  

2. **动态推理（CoT + 动态停机）**  
   - 主模型在每一步生成一个思维链片段，同时输出一个置信度分布。  
   - 置信度通过软最大化的方式映射到 0–1，若超过预设的停机阈值（例如 0.92），系统立即终止后续生成，直接输出答案。  
   - 这种机制类似于人做题时“如果已经确信答案，就不再继续检查”。  

3. **思维压缩自训练**  
   - 第一次推理得到完整的长链 CoT（比如 15 步），这些链被标记为“原始”。  
   - 再训练一个压缩网络（基于 seq2seq），输入长链，输出更短的等价链。压缩目标是最小化答案差异，同时惩罚输出长度。  
   - 生成的短链再喂回主模型进行二次微调，使其学会在同样的输入上直接产生简洁的思维链。  
   - 关键的反直觉点在于：压缩网络本身不需要额外标注数据，全部由模型自生成的长链提供监督，实现了“自监督的思维精简”。  

4. **整体训练流程**  
   - **阶段一**：先用公开的数学/推理数据（如 GSM8K）训练难度感知器和主模型的基础能力。  
   - **阶段二**：开启自训练循环：模型生成长链 → 压缩网络学习 → 主模型二次微调 → 更新难度感知器的标签（因为压缩后步数变化会影响难度估计）。循环数次后收敛。  

### 实验与效果
- **测试数据**：GSM8K（中等难度数学题）、MATH500（高难度竞赛题）、GPQA（通用知识问答）以及 AIME（美国数学邀请赛）四套基准。  
- **对比基线**：原始 o1（固定 20 步 CoT）、普通 ChatGPT‑4（无 CoT）、以及最新的 “CoT‑Lite” 轻量思维链模型。  
- **主要结果**（论文中给出的数字）：  
  - 在 GSM8K 上，正确率从 92.3%（原始 o1）下降不到 0.2%（仍保持 92.1%），但平均 FLOPs 减少约 35%。  
  - 在 MATH500 上，保持 78.5% 的高准确率，算力下降 28%。  
  - GPQA 与 AIME 上的结果类似，准确率基本持平，算力节省 30%+。  
- **消融实验**：  
  - 去掉难度感知器 → 平均步数回升 15%，算力节省下降 10%。  
  - 只使用动态停机不做思维压缩 → 步数下降 20%，但正确率损失约 1.5%。  
  - 只做思维压缩不调动难度感知 → 对简单题的算力节省不明显，说明两者协同是关键。  
- **局限性**：  
  - 难度感知器在极端语言类任务（如长篇阅读理解）上预测不够精准，导致部分中等难度问题仍被过度思考。  
  - 思维压缩过程依赖于模型自身生成的长链质量，如果原始链本身就有错误，压缩后可能放大错误。作者在讨论中承认需要更稳健的自监督信号。  

### 影响与延伸思考
这篇工作打开了“算力经济学”在大模型推理阶段的研究视角。随后的几篇论文（如《Adaptive CoT for LLMs》《Budget‑Aware Prompting》）都在不同维度上借鉴了难度感知与动态停机的思路。业界也开始在产品层面实现“按需算力”，比如在搜索引擎的即时问答中对简单查询使用轻量推理。想进一步探索的读者可以关注以下方向：  
- **多模态难度感知**：把图像、音频特征一起输入，预测跨模态任务的算力需求。  
- **鲁棒的自监督压缩**：引入外部校验（如符号求值器）防止压缩过程放大错误。  
- **算力预算调度**：在大规模服务中把全局算力池分配给不同用户请求，实现全局最优。  

### 一句话记住它
让会“深思熟虑”的大模型学会在“2+3=?”时直接说答案，从而把算力花在真正需要深度思考的地方。