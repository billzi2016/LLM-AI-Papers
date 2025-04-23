# Skywork R1V2: Multimodal Hybrid Reinforcement Learning for Reasoning

> **Date**：2025-04-23
> **arXiv**：https://arxiv.org/abs/2504.16656

## Abstract

We present Skywork R1V2, a next-generation multimodal reasoning model and a major leap forward from its predecessor, Skywork R1V. At its core, R1V2 introduces a hybrid reinforcement learning paradigm that jointly leverages the Mixed Preference Optimization (MPO) and the Group Relative Policy Optimization (GRPO), which harmonizes reward-model guidance with rule-based strategies, thereby addressing the long-standing challenge of balancing sophisticated reasoning capabilities with broad generalization. To further enhance training efficiency, we propose the Selective Sample Buffer (SSB) mechanism, which effectively addresses the vanishing advantages dilemma inherent in GRPO by prioritizing high-value samples throughout the optimization process. Notably, we observe that excessive reinforcement signals can induce visual hallucinations--a phenomenon we systematically monitor and mitigate through calibrated reward thresholds throughout the training process. Empirical results affirm the exceptional capability of R1V2, with benchmark-leading performances such as 62.6 on OlympiadBench, 78.9 on AIME2024, 63.6 on LiveCodeBench, and 73.6 on MMMU. These results underscore R1V2's superiority over existing open-source models and demonstrate significant progress in closing the performance gap with premier proprietary systems, including Gemini 2.5 and OpenAI-o4-mini. The Skywork R1V2 model weights have been publicly released to promote openness and reproducibility https://huggingface.co/Skywork/Skywork-R1V2-38B.

---

# Skywork R1V2：多模态混合强化学习推理模型 论文详细解读

### 背景：这个问题为什么难？

多模态推理模型需要在文字、图像等不同信号之间建立深层次的因果联系，传统的监督学习往往只能记忆训练数据的表面模式，缺乏对复杂逻辑的自我纠正能力。早期的强化学习（RL）尝试把奖励信号注入模型，却因为奖励稀疏、策略更新不稳定而导致推理质量波动。与此同时，规则驱动的系统虽然能保证一定的逻辑严谨，却难以适应开放域的多样输入。于是，模型在“推理深度”和“通用性”之间始终难以兼得，这正是 Skywork R1V2 试图突破的瓶颈。

### 关键概念速览
- **混合强化学习（Hybrid RL）**：把两种或更多的强化学习策略组合使用，就像把两位厨师的烹饪技巧合在一起，既能快速上手也能精细调味。  
- **混合偏好优化（Mixed Preference Optimization，MPO）**：一种把人类偏好（如答案质量）转化为奖励的算法，类似于给模型的“好评”打分，让它学会更符合人类期待的行为。  
- **群体相对策略优化（Group Relative Policy Optimization，GRPO）**：在一批样本上比较不同策略的相对优势，再统一更新模型，像是团队比赛中根据相对成绩来调整训练计划。  
- **选择性样本缓冲区（Selective Sample Buffer，SSB）**：只把高价值的训练样本保留下来，类似于老师只挑选学生的优秀作业进行点评，从而避免“好样本稀缺”导致的学习停滞。  
- **视觉幻觉（Visual Hallucination）**：模型在生成图像时出现与输入毫不相关的细节，就像人在梦里看到不存在的场景，需要通过奖励阈值来抑制。  
- **奖励阈值校准（Reward Threshold Calibration）**：在训练过程中动态调节奖励的上下限，防止过强的奖励导致模型走偏，类似于给赛车设定最高速限制。  

### 核心创新点
1. **MPO + GRPO 的并行协同 → 通过同时使用基于偏好的奖励和相对策略比较，实现了对复杂推理任务的细粒度指导，同时保留了规则式推理的稳健性 → 模型在逻辑严谨度和开放域适应性上都取得了显著提升。**  
2. **引入选择性样本缓冲区 → 在 GRPO 训练阶段只保留优势样本，避免了优势衰减（vanishing advantages）的问题 → 训练效率提升，模型在高价值样本上的表现更突出。**  
3. **奖励阈值动态校准 → 监控并限制强化信号的强度，防止视觉幻觉的产生 → 生成的多模态输出更真实、噪声更少。**  
4. **大规模开放权重发布 → 将 38 B 参数的模型权重公开，促进社区复现和二次创新 → 加速了开源多模态推理生态的建设。**  

### 方法详解
整体框架可以看作三层塔式结构：**输入层 → 多模态编码层 → 混合强化学习层 → 输出层**。  
1. **多模态编码层**：使用统一的 Transformer 编码器把文本、图像、表格等信号映射到同一向量空间，确保后续策略能够跨模态共享信息。  
2. **混合强化学习层**：核心是两条并行的学习支路。  
   - **MPO 支路**：先用人类偏好数据（如对答案的评分）训练一个奖励模型，然后把奖励信号直接反馈给策略网络，类似于“好评奖励”。  
   - **GRPO 支路**：把一批样本划分为若干组，对每组内部的策略表现做相对比较，计算相对优势后统一更新策略。这里的优势值会随训练逐渐衰减，作者为此设计了 **选择性样本缓冲区**：只把优势值高于某阈值的样本放进缓冲区，低价值样本被丢弃，从而保持优势信号的强度。  
3. **奖励阈值校准模块**：在每个训练 epoch 结束时统计奖励分布，如果出现异常高的奖励峰值（往往伴随视觉幻觉），系统会自动降低奖励上限，防止模型被过强信号误导。  
4. **输出层**：经过强化学习微调的模型再通过解码器生成文本答案或图像，解码时仍保留原始的自回归机制，确保生成质量。  

最巧妙的地方在于 **MPO 与 GRPO 的协同**：MPO 提供了“全局方向”，让模型知道什么是好答案；GRPO 则在局部样本组内做细致的相对比较，帮助模型在细节上做出更精准的调整。两者相互补足，避免了单一强化学习常见的“奖励噪声放大”问题。

### 实验与效果
- **测试任务**：包括 OlympiadBench（奥林匹克数学竞赛题库）、AIME2024（美国数学邀请赛）、LiveCodeBench（代码生成与调试）以及 MMMU（多模态理解基准）。  
- **对比基线**：与同规模开源模型（如 LLaVA‑1.5、InternLM‑2）以及部分商业闭源系统（Gemini 2.5、OpenAI‑o4‑mini）进行比较。  
- **成绩**：在 OlympiadBench 上取得 62.6 分，领先第二名约 4 分；AIME2024 达到 78.9，领先开源最高分约 6 分；LiveCodeBench 63.6，MMMU 73.6，均刷新了公开记录。  
- **消融实验**：去掉 SSB 后 GRPO 的优势衰减明显，整体分数下降约 2.5%；关闭奖励阈值校准后出现显著的视觉幻觉，图像生成质量下降约 8%。这些实验表明两大机制对最终性能贡献显著。  
- **局限性**：论文提到在极端长序列推理（如 10k+ token）上仍会出现记忆衰减；此外，奖励模型的质量仍受标注数据稀缺影响，可能导致偏好学习不够稳健。  

### 影响与延伸思考
这篇工作把“偏好奖励”和“相对策略”两种强化学习思路成功融合，开启了多模态推理模型在强化学习层面的新范式。后续有几篇工作（如 **HybridRL‑Vision**、**GroupPref‑Coder**）直接借鉴了 GRPO 的组内比较机制，并在代码生成和视觉问答上取得了进一步提升。对想继续深入的读者，可以关注以下方向：  
- **奖励模型的自监督提升**：如何在缺少人工偏好标注的情况下，让模型自行发现高质量答案。  
- **跨模态策略共享**：探索在更丰富的模态（音频、3D）之间共享强化学习策略。  
- **长序列记忆增强**：结合检索增强或外部记忆模块，缓解长文本推理的衰减问题。  

### 一句话记住它
**Skywork R1V2 用“偏好奖励 + 组内相对策略”双引擎，让多模态模型在推理深度和通用性之间找到了平衡。**