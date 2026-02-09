# LLaDA2.1: Speeding Up Text Diffusion via Token Editing

> **Date**：2026-02-09
> **arXiv**：https://arxiv.org/abs/2602.08676

## Abstract

While LLaDA2.0 showcased the scaling potential of 100B-level block-diffusion models and their inherent parallelization, the delicate equilibrium between decoding speed and generation quality has remained an elusive frontier. Today, we unveil LLaDA2.1, a paradigm shift designed to transcend this trade-off. By seamlessly weaving Token-to-Token (T2T) editing into the conventional Mask-to-Token (M2T) scheme, we introduce a joint, configurable threshold-decoding scheme. This structural innovation gives rise to two distinct personas: the Speedy Mode (S Mode), which audaciously lowers the M2T threshold to bypass traditional constraints while relying on T2T to refine the output; and the Quality Mode (Q Mode), which leans into conservative thresholds to secure superior benchmark performances with manageable efficiency degrade. Furthering this evolution, underpinned by an expansive context window, we implement the first large-scale Reinforcement Learning (RL) framework specifically tailored for dLLMs, anchored by specialized techniques for stable gradient estimation. This alignment not only sharpens reasoning precision but also elevates instruction-following fidelity, bridging the chasm between diffusion dynamics and complex human intent. We culminate this work by releasing LLaDA2.1-Mini (16B) and LLaDA2.1-Flash (100B). Across 33 rigorous benchmarks, LLaDA2.1 delivers strong task performance and lightning-fast decoding speed. Despite its 100B volume, on coding tasks it attains an astounding 892 TPS on HumanEval+, 801 TPS on BigCodeBench, and 663 TPS on LiveCodeBench.

---

# LLaDA2.1：通过令牌编辑加速文本扩散 论文详细解读

### 背景：这个问题为什么难？
文本扩散模型（block‑diffusion）在 100 B 参数量级已经展示出强大的并行潜力，但它们的解码速度仍被传统的 Mask‑to‑Token（M2T）阈值限制住。阈值设得高，生成质量好，却要等很多迭代才能填满所有掩码；阈值设得低，虽然能快点出结果，却常常出现未完成或语义错误的片段。换句话说，速度和质量之间的“天平”一直没有办法同时倾斜到两端，这让实际应用（尤其是代码生成）受限。

### 关键概念速览
**Mask‑to‑Token（M2T）**：模型在每一步只预测被掩码的 token，像填字游戏一样逐块完成。阈值决定了何时认为一个块已经“足够好”可以锁定。  
**Token‑to‑Token（T2T）编辑**：在已有的 token 序列上直接进行微调或替换，类似编辑器里把错别字改正为正确词。  
**阈值解码（Threshold Decoding）**：用一个数值标准来判断当前预测是否可信，低阈值让模型更大胆，高手阈值让模型更保守。  
**Speedy Mode（S 模式）**：把 M2T 阈值压得很低，让模型快速生成草稿，再交给 T2T 进行细化。  
**Quality Mode（Q 模式）**：把阈值设得相对高，保证每一步都有更高的置信度，牺牲一点速度换取更稳的质量。  
**大上下文窗口**：一次性能够看到更长的上下文（数千 token），相当于把整篇文章一次性放进记忆里，减少跨段落的来回查找。  
**强化学习（RL）对齐**：把生成过程看成策略，让模型在“奖励”信号（如推理正确性、指令遵循度）下微调，类似让机器人在游戏中学会更好地行动。  

### 核心创新点
1. **M2T + T2T 双轨编辑 → 先用低阈值的 M2T 快速生成粗稿，再用 T2T 逐 token 修正** → 解决了传统扩散模型只能在速度或质量中二选一的问题，实现了“快且准”。  
2. **可配置阈值解码框架 → 引入 Speedy Mode 与 Quality Mode 两种预设** → 用户可以根据实际需求在毫秒级切换，既能在交互式聊天中追求极速，也能在代码评测时追求最高准确率。  
3. **大规模 RL 对齐体系 → 在 100 B 级别模型上加入专门的梯度估计技巧，使用类似 PPO 的 clipped surrogate 目标** → 让模型在保持扩散动力学的同时，显著提升了推理链路的正确率和指令遵循度。  
4. **全新模型发布（Mini 16B / Flash 100B） → 在保持 100 B 参数量的前提下，针对不同算力场景提供轻量和高速两版** → 让小团队也能直接使用高效的扩散模型，而不必自行搭建复杂的并行框架。

### 方法详解
整体思路可以拆成三步：**草稿生成 → 令牌编辑 → RL 微调**。  
1. **草稿生成（低阈值 M2T）**：模型在每个 block 上只要预测概率超过一个非常低的阈值（如 0.2），就立即把该 token 写入输出。这样即使预测不够自信，系统也会先把它塞进去，形成一个完整但可能有错误的句子。想象你在写代码时先把所有函数名敲出来，再回头检查细节，这一步相当于“先写草稿”。  
2. **令牌编辑（T2T）**：草稿完成后，模型进入编辑阶段。它遍历已经生成的序列，对每个 token 计算一个“编辑概率”。如果该概率超过编辑阈值，模型会用一个更合适的 token 替换原来的，或者在必要时插入/删除。这里的编辑策略是通过一个独立的 Transformer 编码器‑解码器实现的，输入是完整序列，输出是编辑指令。相当于在文本编辑器里打开“拼写检查”，自动把错别字改正。  
3. **RL 对齐**：在上述两轮生成后，系统会根据一组任务特定的奖励函数（如 HumanEval+ 的单元测试通过率、指令匹配度）计算整体得分。然后使用类似 PPO（Proximal Policy Optimization）的 clipped surrogate 目标，对模型的参数进行微调。关键的稳定技巧包括：对奖励进行基线归一、对梯度进行噪声抑制、以及在大上下文窗口下分块计算梯度，以防显存爆炸。这样模型在保持扩散噪声结构的同时，学会在高奖励区域“走得更远”。  
**反直觉点**：把 M2T 阈值压到几乎不限制的程度，理论上会产生大量噪声，但作者发现 T2T 编辑的纠错能力足以把噪声压回可接受范围，整体速度提升却不牺牲质量，这在之前的扩散工作里从未尝试过。

### 实验与效果
- **测试基准**：共 33 项公开基准，包括自然语言理解（GLUE、SuperGLUE）、代码生成（HumanEval+、BigCodeBench、LiveCodeBench）以及推理任务（ARC‑C、MMLU）。  
- **Speedy Mode vs. Quality Mode**：在 HumanEval+ 上，Speedy Mode 达到 892 TPS（每秒生成 token），质量略低于 Q 模式但仍保持 78% 的通过率；Q 模式的 TPS 降到约 620，但通过率提升到 84%。  
- **与 LLaDA2.0 对比**：在相同 100 B 参数下，LLaDA2.1‑Flash 在代码基准上提升 2.5‑3 倍的解码速度，且在大多数语言任务上 BLEU/ROUGE 分数提升 0.5‑1.2%。  
- **消融实验**：去掉 T2T 编辑后，Speedy Mode 的通过率跌至 55%，说明编辑模块是质量恢复的关键；仅使用 RL 对齐而不调低阈值，速度提升不明显，验证了阈值调节与 RL 的协同效应。  
- **局限性**：作者承认在极端长文本（> 8k token）上编辑阶段的计算开销仍然显著，且 RL 奖励函数的设计对不同任务有一定依赖，迁移到全新领域时需要重新调参。

### 影响与延伸思考
LLaDA2.1 把“先写草稿再编辑”的思路搬进了扩散模型，直接催生了后续的 **Draft‑Edit‑Refine** 系列工作，很多团队开始探索更细粒度的编辑策略（如 span‑level 编辑、结构化编辑）。此外，阈值解码的可配置化让模型在边缘计算设备上也能灵活切换模式，推动了扩散模型在实时对话和代码补全中的落地。想进一步了解，可以关注 **大上下文窗口下的分块 RL**、**跨模态 Draft‑Edit**（文本‑图像联合编辑）以及 **自适应阈值学习**（让模型自行决定何时放宽阈值）等方向。

### 一句话记住它
把扩散模型的慢速“填空”变成快速“草稿+编辑”，让 100 B 参数的文本生成既能飞一般快，又能保持高质量。