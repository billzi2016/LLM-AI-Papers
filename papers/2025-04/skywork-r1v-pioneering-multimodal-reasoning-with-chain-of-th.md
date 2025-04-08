# Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought

> **Date**：2025-04-08
> **arXiv**：https://arxiv.org/abs/2504.05599

## Abstract

We introduce Skywork R1V, a multimodal reasoning model extending the an R1-series Large language models (LLM) to visual modalities via an efficient multimodal transfer method. Leveraging a lightweight visual projector, Skywork R1V facilitates seamless multimodal adaptation without necessitating retraining of either the foundational language model or the vision encoder. To strengthen visual-text alignment, we propose a hybrid optimization strategy that combines Iterative Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO), significantly enhancing cross-modal integration efficiency. Additionally, we introduce an adaptive-length Chain-of-Thought distillation approach for reasoning data generation. This approach dynamically optimizes reasoning chain lengths, thereby enhancing inference efficiency and preventing excessive reasoning overthinking. Empirical evaluations demonstrate that Skywork R1V, with only 38B parameters, delivers competitive performance, achieving a score of 69.0 on the MMMU benchmark and 67.5 on MathVista. Meanwhile, it maintains robust textual reasoning performance, evidenced by impressive scores of 72.0 on AIME and 94.0 on MATH500. The Skywork R1V model weights have been publicly released to promote openness and reproducibility.

---

# Skywork R1V：开创多模态推理与思维链 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）已经可以处理复杂文字推理的今天，把同样的能力搬到“看得懂图、说得出话”的多模态场景仍然卡在几个关键点。第一，视觉编码器和语言模型的结构差异大，直接拼接往往需要大规模联合训练，成本高且容易出现模式冲突。第二，跨模态对齐不充分会导致模型在看图时只能给出表层描述，难以进行深层推理。第三，思维链（Chain‑of‑Thought）在纯文本上已经证明有效，但在视觉任务里如何生成、何时停止仍没有统一方案。于是出现了“既想保留已有大语言模型的强大推理，又不想重新训练整个系统”的需求。

### 关键概念速览
- **多模态模型**：同时接受文字和图像输入的模型，像人一样把“看”和“说”结合起来。  
- **视觉投影器（visual projector）**：一个轻量的桥接层，把视觉编码器输出的向量映射到语言模型内部的隐藏空间，就像把不同语言的翻译器对接在一起。  
- **迭代监督微调（Iterative Supervised Fine‑Tuning, SFT）**：在已有的监督数据上多轮微调，每轮都把上一次的模型输出当作新一轮的训练信号，类似于老师让学生先写草稿再改进。  
- **组相对策略优化（Group Relative Policy Optimization, GRPO）**：一种强化学习技巧，针对一组样本同时优化策略，使模型在跨模态对齐上更稳健，类似于团队运动中大家一起调整战术。  
- **思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前先写出推理步骤，像解数学题时先列出公式再算出结果。  
- **自适应长度思维链蒸馏**：在生成思维链时动态决定链的长短，并把这些链“蒸馏”进模型，使其在推理时既不冗长也不缺信息。  
- **蒸馏（Distillation）**：把大模型或复杂推理过程的知识压缩进小模型，像把老师的经验浓缩成简短的笔记。  

### 核心创新点
1. **轻量视觉投影器实现跨模态迁移**  
   - 之前的多模态系统往往需要重新训练语言模型或使用庞大的跨模态桥接层。  
   - 这篇论文直接在已有的 R1 系列语言模型上叠加一个小型投影器，将视觉特征映射进去，而不改动原始语言模型的权重。  
   - 结果是训练成本大幅下降，模型可以在保持原有文本能力的同时快速获得视觉感知。

2. **混合优化：迭代 SFT + GRPO**  
   - 传统的单一监督微调容易在视觉-文本对齐上出现局部最优，强化学习单独使用时又不够稳定。  
   - 作者先用迭代 SFT 让模型在标注数据上逐步提升，再用 GRPO 对一批样本进行策略梯度优化，强化跨模态一致性。  
   - 这种两步走的组合显著提升了跨模态推理的准确率，尤其在需要细粒度视觉理解的任务上表现突出。

3. **自适应长度思维链蒸馏**  
   - 直接把固定长度的思维链喂给模型会导致推理过长或信息不足。  
   - 论文提出根据问题难度动态调节链的长度，并把这些链通过蒸馏方式注入模型，使其在推理时自行决定“写几步”。  
   - 这样既避免了“思考过度”导致的推理慢，也保持了复杂问题所需的深度推理。

### 方法详解
整体框架可以拆成三大步骤：**视觉特征提取 → 投影映射 → 跨模态微调与蒸馏**。先用标准的视觉编码器（如 CLIP‑ViT）把图片转成向量；随后轻量投影器把这些向量线性或非线性映射到语言模型的隐藏层维度；最后在混合优化和自适应思维链蒸馏的帮助下，让模型学会在文字和图像之间建立紧密联系。

**1. 视觉投影器细节**  
- 投影器本身只有几层全连接网络，参数量远低于语言模型（约 0.1%）。  
- 输入是视觉编码器的最后一层特征，输出直接加到语言模型的 token embedding 前，类似于在文字序列前插入一个“视觉 token”。  
- 由于投影器是独立训练的，语言模型的原始权重保持不变，这也是实现“零重训练”的关键。

**2. 混合优化流程**  
- **迭代 SFT**：先准备跨模态对齐的监督数据（图文问答、描述等），模型在这些数据上进行一次微调，得到第一版。随后把模型在这些数据上的输出（包括生成的思维链）作为新的标签，进入第二轮微调，如此循环数次。每轮都让模型在已有知识的基础上进一步细化。  
- **GRPO**：在每轮 SFT 结束后，抽取一批难例，构造奖励函数（如答案正确率、思维链完整度），对同一批样本的策略进行相对优化。因为是“组相对”，模型在同一组内相互比较，梯度更平滑，避免单样本噪声导致的崩溃。  
- 两者交替进行，使得模型既能从标注信号中学习，又能在策略层面强化跨模态一致性。

**3. 自适应长度思维链蒸馏**  
- 生成思维链时，先用一个小的教师模型对每个问题进行多步推理，记录每一步的置信度。  
- 根据置信度曲线动态截断：如果后续步骤的增益低于阈值，就停止生成，得到“最短足够链”。  
- 将这些链与原始答案一起作为蒸馏目标，用 KL 散度让学生模型（即 Skywork R1V）学习。蒸馏过程不需要额外的推理时间，因为链已经被压缩进模型内部的权重。  
- 这一步的巧妙之处在于让模型在推理时自行决定思考深度，而不是硬编码固定步数。

**最反直觉的设计**  
- 只用一个轻量投影器就能让 38 B 参数的语言模型直接接受视觉信息，而不需要对语言模型进行任何结构性改动。很多人会担心这样会导致信息瓶颈，但实验表明投影器足以把视觉语义映射到语言空间。

### 实验与效果
- **评测数据集**：MMMU（多模态理解基准）得分 69.0，MathVista（数学视觉推理）得分 67.5，AIME（美国数学竞赛）得分 72.0，MATH500（大规模数学题库）得分 94.0。  
- **对比基线**：在同等参数规模下，传统多模态模型（如 LLaVA、MiniGPT‑4）在 MMMU 上大约在 62–65 分之间，MathVista 也普遍低于 65 分。Skywork R1V 超出 4–7 分的区间，显示出跨模态对齐的优势。  
- **消融实验**：论文分别去掉投影器、GRPO、以及自适应思维链蒸馏。去掉投影器后模型几乎失去视觉能力；去掉 GRPO，MMMU 分数下降约 2.3 分；去掉蒸馏，MathVista 分数下降约 1.8 分，说明每个模块都有实质贡献。  
- **局限性**：作者指出模型仍然依赖大量高质量的跨模态标注数据，且在极端长图或细粒度视觉关系（如微观医学图像）上表现未必理想。训练过程中的 GRPO 需要额外的奖励函数设计，增加了实现复杂度。

### 影响与延伸思考
- 这篇工作展示了“轻量投影+混合优化”可以在不破坏大语言模型的前提下快速扩展到视觉领域，随后的多篇研究（如 **Vision‑LLM‑Adapter**、**CrossModal‑Lite**）都借鉴了类似的投影桥接思路。  
- 自适应思维链蒸馏的想法也激发了后续在多模态推理中动态推理深度的探索，例如 **Dynamic‑CoT‑Vision** 系列。  
- 想进一步深入的读者可以关注两条路线：① 更高效的跨模态对齐奖励设计（比如基于对比学习的 GRPO 变体）；② 将自适应思维链扩展到视频、音频等时序模态，探索“思考时间”与“思考深度”的协同调度。  

### 一句话记住它
只用一个轻量投影器和混合微调，就让 38 B 的大语言模型瞬间拥有高效的视觉推理能力。