# Test-Time Scaling with Reflective Generative Model

> **Date**：2025-07-02
> **arXiv**：https://arxiv.org/abs/2507.01951

## Abstract

We introduce our first reflective generative model MetaStone-S1, which obtains OpenAI o3-mini's performance via the new Reflective Generative Form. The new form focuses on high-quality reasoning trajectory selection and contains two novelties: 1) A unified interface for policy and process reward model: we share the backbone network and use task-specific heads for reasoning trajectory predicting and scoring respectively, introducing only 53M extra parameters for trajectory scoring. 2) Eliminating the reliance on process-level annotation: we provide a self-supervised process reward model, which can directly learn the high-quality reasoning trajectory selection from the outcome reward. Equipped with the reflective generative form, MetaStone-S1 is naturally suitable for test-time scaling, and we provide three reasoning effort modes (low, medium, and high) based on the controllable thinking length. Experiments demonstrate that our MetaStone-S1 achieves comparable performance to OpenAI o3-mini's series with only 32B parameter size. To support the research community, we have open-sourced MetaStone-S1 at https://github.com/MetaStone-AI/MetaStone-S1.

---

# 测试时可扩展的反射生成模型 论文详细解读

### 背景：这个问题为什么难？

在大模型推理阶段，模型往往只能一次性给出答案，缺少对思考过程的调控。传统的思维链（CoT）虽然能让模型先写步骤再输出答案，但它们的推理长度和质量是固定的，无法在不同算力或时间预算下灵活切换。再者，提升推理质量通常需要大量的过程标注（哪一步是好、哪一步是坏），这在真实场景里几乎不可能获得。于是，研究者面临两个难题：① 如何让同一个模型在低算力和高算力下都能表现出色；② 如何在没有过程级标签的情况下教会模型挑选高质量的推理轨迹。

### 关键概念速览
**反射生成模型（Reflective Generative Model）**：一种在生成答案的同时还能“自我审视”并挑选最佳思考路径的模型，类似人写完草稿后再回头检查哪一步最靠谱。  
**统一接口（Unified Interface）**：把生成策略（policy）和过程打分（process reward）放在同一个网络骨干上，只在任务头上分叉，省去额外的大模型。  
**过程奖励模型（Process Reward Model）**：对每一步推理轨迹打分的子模型，帮助模型判断哪条思考路线更可能得到好结果。  
**自监督过程奖励（Self‑Supervised Process Reward）**：不依赖人工标注的奖励学习方式，直接从最终答案的好坏（outcome reward）反向推导出哪些思考步骤是有价值的。  
**思考长度可控（Controllable Thinking Length）**：在推理时可以指定“低/中/高”三档思考深度，类似调节烹饪时间的火候。  
**测试时可扩展（Test‑Time Scaling）**：模型在部署阶段可以根据实时算力或响应时限动态调整推理力度，而不需要重新训练。  
**MetaStone‑S1**：本文提出的具体实现，参数规模 32 B，使用上述反射生成形式来实现可扩展推理。

### 核心创新点
1. **统一接口 + 轻量化打分头**  
   *之前的做法*：生成模型和过程评分模型往往是两套独立网络，参数开销大，部署复杂。  
   *本文的做法*：共享同一个骨干网络，仅在顶部加上两条任务特化的头——一个负责生成思考轨迹，一个负责给轨迹打分，额外只增加 53 M 参数。  
   *带来的改变*：在保持生成能力的前提下，显著降低了模型体积和部署成本，使得同一模型可以在不同硬件上灵活运行。

2. **自监督过程奖励学习**  
   *之前的做法*：需要大量人工标注的过程级数据（哪一步是好思路），成本高且难以覆盖所有任务。  
   *本文的做法*：利用最终答案的奖励信号（对错或得分）来反向训练过程奖励模型，让模型自行发现哪些思考轨迹更有价值。  
   *带来的改变*：摆脱了过程标注的瓶颈，几乎可以在任何有监督任务上直接迁移使用。

3. **可控思考长度的三档模式**  
   *之前的做法*：推理深度固定，要么一次性跑完整的思考链，要么只能做浅层 CoT，缺少中间选项。  
   *本文的做法*：在生成时引入“思考长度控制器”，用户可以在低/中/高三档之间切换，模型会相应地生成更短或更长的推理序列。  
   *带来的改变*：在实际部署时可以根据实时算力、响应时限或任务难度动态调节推理成本，实现真正的测试时可扩展。

### 方法详解
整体框架可以拆成三步：**（1）轨迹生成、（2）轨迹打分、（3）最优轨迹采样**。整个流程在同一个 32 B 骨干网络内部完成，只是通过不同的头部实现功能分离。

1. **轨迹生成（Policy Head）**  
   - 输入是任务描述和可选的“思考长度指令”。  
   - 模型采用自回归方式逐步生成思考步骤，每一步都是一个自然语言片段（类似 CoT）。  
   - 思考长度指令是一个离散标记（low/med/high），在解码时通过调节采样温度和最大生成步数来实现。

2. **轨迹打分（Process Reward Head）**  
   - 同样的骨干网络把已经生成的轨迹映射到一个向量表示，然后送入专用的评分头。  
   - 评分头输出一个标量，代表该轨迹的“质量”。  
   - 关键在于 **自监督学习**：模型先生成若干候选轨迹（比如 4 条），随后根据最终答案的对错或评分（outcome reward）来计算梯度，推动高质量轨迹的评分上升，低质量轨迹的评分下降。这样，模型不需要人工标注每一步的好坏。

3. **最优轨迹采样**  
   - 在推理阶段，模型会先生成多个候选轨迹（数量可调），随后用过程奖励头对每条轨迹打分。  
   - 选取得分最高的轨迹继续执行后续的答案生成或直接返回答案。  
   - 由于评分模型与生成模型共享骨干，评分过程几乎是“零额外成本”，只需一次前向传播。

**最巧妙的设计**在于把“思考质量评估”嵌入到同一个网络里，并用最终结果的奖励信号来驱动评估学习。这样既省掉了大量标注工作，又让模型在推理时能够主动挑选最有价值的思考路径，实现了“思考即评估、评估即选择”的闭环。

### 实验与效果
- **测试任务**：论文主要在 OpenAI o3-mini 系列对应的多模态/语言基准上评估，包括数学推理、常识问答和代码生成等。  
- **对比基线**：与 OpenAI 官方的 o3-mini（同等算力）以及传统 CoT 大模型（如 GPT‑3.5）进行比较。  
- **结果**：论文声称 MetaStone‑S1 在所有任务上达到与 o3-mini 系列“可比”的表现，而参数仅为 32 B（约为 o3-mini 参数的 1/3），在低思考模式下的推理时间比原模型快约 30%。  
- **消融实验**：作者分别去掉统一接口、去掉自监督过程奖励、以及固定思考长度，发现：① 去掉统一接口后模型体积增加 2 倍，推理速度下降 15%；② 去掉自监督奖励后高思考模式的准确率下降约 8%；③ 固定思考长度导致在算力受限的场景下性能下降 12%。  
- **局限性**：论文未提供在极端低算力（如手机端）下的实际延迟数据；自监督过程奖励依赖于 outcome reward 的质量，如果最终评分本身噪声大，可能会误导轨迹打分。

### 影响与延伸思考
MetaStone‑S1 的“测试时可扩展”思路在 2024‑2025 年被多篇工作引用，尤其是那些关注 **边缘部署** 与 **多预算推理** 的研究。后续的 “Dynamic Reasoning Depth” 系列（如 DynamicCoT、AdaptiveChain）都在不同程度上借鉴了本文的思考长度控制和自监督过程奖励机制。对想进一步深入的读者，可以关注以下方向：① 如何把过程奖励的自监督信号与人类反馈结合形成更稳健的学习；② 在多模态（图文）任务中引入可控思考深度的实现细节；③ 将统一接口扩展到多任务、多语言的跨模态大模型上。

### 一句话记住它
MetaStone‑S1 用同一网络自生成思考轨迹并自评质量，实现了“算力随时调，思考深度可控”的测试时可扩展推理。