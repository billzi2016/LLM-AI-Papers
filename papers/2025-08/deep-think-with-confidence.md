# Deep Think with Confidence

> **Date**：2025-08-21
> **arXiv**：https://arxiv.org/abs/2508.15260

## Abstract

Large Language Models (LLMs) have shown great potential in reasoning tasks through test-time scaling methods like self-consistency with majority voting. However, this approach often leads to diminishing returns in accuracy and high computational overhead. To address these challenges, we introduce Deep Think with Confidence (DeepConf), a simple yet powerful method that enhances both reasoning efficiency and performance at test time. DeepConf leverages model-internal confidence signals to dynamically filter out low-quality reasoning traces during or after generation. It requires no additional model training or hyperparameter tuning and can be seamlessly integrated into existing serving frameworks. We evaluate DeepConf across a variety of reasoning tasks and the latest open-source models, including Qwen 3 and GPT-OSS series. Notably, on challenging benchmarks such as AIME 2025, DeepConf@512 achieves up to 99.9% accuracy and reduces generated tokens by up to 84.7% compared to full parallel thinking.

---

# 深度思考与置信度 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上做推理，常用的办法是一次性让模型生成多个思考路径，然后用多数投票决定答案，这叫自一致性（self‑consistency）。虽然能把准确率往上推，但每增加一次思考就要完整生成一遍答案，算力和时间会线性飙升。更糟的是，很多生成的路径质量参差不齐，低质量的“噪声”不仅浪费算力，还会把投票结果拉低，出现收益递减的现象。于是，如何在保持甚至提升推理准确率的同时，削减不必要的计算，成了迫切需要解决的难题。

### 关键概念速览
- **大语言模型（LLM）**：能够理解并生成自然语言的深度神经网络，像 GPT、Qwen 等。把它想象成会写作文的“机器人”。
- **自一致性（self‑consistency）**：让模型并行生成多条思考链，再用多数投票决定最终答案，类似让几个人讨论后取多数意见。
- **思考链（Chain‑of‑Thought, CoT）**：模型在给出答案前把推理步骤写出来，像在纸上列草稿一样帮助模型保持逻辑连贯。
- **置信度信号（confidence signal）**：模型在每一步生成时内部产生的概率分布（logits）所对应的“自信程度”，可以看作模型对当前词的“确信度”。
- **推理轨迹（reasoning trace）**：一次完整的思考链，从问题到答案的全部生成过程。
- **动态过滤（dynamic filtering）**：在生成过程中或结束后，根据置信度把低质量轨迹剔除，而不是盲目保留所有轨迹。
- **令牌（token）**：模型处理的最小语言单元，类似文字的“拼图块”。生成的令牌越多，算力消耗越大。

### 核心创新点
1. **内部置信度直接用于轨迹筛选 → DeepConf 在生成或生成结束后读取模型的 token‑level 置信度，设定阈值把置信度低的思考链剔除 → 只保留高置信度轨迹即可投票，显著降低了无效计算。**
2. **无需额外训练或调参 → 传统提升推理的方法往往要微调模型或搜索最佳投票数，DeepConf 直接使用原始模型的输出概率，省去所有再训练步骤 → 直接在已有服务框架中挂上一个过滤层，部署成本几乎为零。**
3. **统一的“置信度阈值”可在不同模型间迁移 → 在 Qwen 3、GPT‑OSS 系列等多款开源模型上实验，使用同一阈值策略即可获得提升 → 说明置信度信号具有跨模型的通用性。**
4. **极致的令牌削减效果 → 在 AIME 2025 这类高难度数学推理基准上，DeepConf@512（最大生成 512 令牌）比全并行思考少生成约 84.7% 的令牌，同时准确率提升到 99.9% → 证明高置信度过滤不仅省时，还能提升答案质量。**

### 方法详解
DeepConf 的整体思路可以拆成三步：**生成 → 置信度评估 → 高置信度过滤 & 投票**。

1. **生成阶段**  
   - 给定问题，模型按照常规的 CoT 方式一次性生成 N 条思考链（N 通常是 5~10），每条链的长度上限设为 512 令牌（对应 DeepConf@512）。  
   - 与普通自一致性不同的是，模型在每生成一个 token 时会输出该 token 的概率分布，这就是置信度信号的来源。

2. **置信度评估**  
   - 对每条思考链，遍历其所有 token，取对应的最大概率（或对数概率）作为该 token 的置信度。  
   - 将整条链的置信度聚合为一个整体分数，常用的做法是取平均置信度或取最低置信度的倒数（类似“最弱环节决定整体强度”）。  
   - 设定一个经验阈值 τ（如 0.85），所有整体置信度低于 τ 的链被标记为“低质量”。

3. **高置信度过滤 & 投票**  
   - 只保留置信度 ≥ τ 的链进入多数投票环节。投票仍然基于最终答案的文字相同与否，但因为噪声链被剔除，投票结果更稳健。  
   - 若保留下来的链数不足以形成多数（极端情况下可能全被剔除），系统会回退到原始单链输出，保证不出现“无答案”情况。

**最巧妙的点**在于置信度的使用时机可以灵活切换：  
- **生成中途过滤**：当某条链的置信度在前半段已经跌破阈值，直接中止该链的继续生成，省下后续的令牌。  
- **生成后统一过滤**：完整生成后再统一评估，适用于需要完整轨迹进行后续分析的场景。两种模式只需在服务层面切换开关，无需改动模型本身。

### 实验与效果
- **测试任务**：论文在多个推理基准上评估，包括数学竞赛题目 AIME 2025、逻辑推理和常识问答等。  
- **模型**：使用最新开源的大模型 Qwen 3 系列以及 GPT‑OSS 系列，覆盖不同规模和架构。  
- **对比基线**：传统自一致性（全并行思考链）以及单一 CoT 生成。  
- **核心结果**：在 AIME 2025 上，DeepConf@512 将准确率提升至 99.9%，同时生成的令牌数比全并行思考少约 84.7%。其他任务也普遍出现 1%~3% 的准确率提升和 30%~70% 的令牌削减。  
- **消融实验**：作者分别关闭“生成中途过滤”和“后置过滤”，发现仅保留后置过滤时仍能削减约 60% 的令牌，但准确率提升幅度下降约 0.5%；若完全不使用置信度阈值，性能回到普通自一致性水平。  
- **局限性**：论文承认置信度阈值 τ 需要根据任务和模型大小做粗略调校，极端低置信度的任务（如开放域生成）可能会把大多数链都过滤掉，导致回退到单链输出。置信度本身仍受模型内部概率校准的影响，若模型本身概率不可靠，过滤效果会受限。

### 影响与延伸思考
DeepConf 把“模型自信”直接转化为计算资源的调度信号，开启了“置信度驱动推理加速”的新思路。随后的工作（如 Confidence‑Guided Sampling、Adaptive CoT）纷纷借鉴其动态过滤机制，尝试在更细粒度（如 token 级）上做预算分配。对想进一步探索的读者，可以关注以下方向：  
- **置信度校准**：提升模型概率的真实性，使过滤更可靠。  
- **多模态置信度**：把视觉或音频模型的置信度也纳入统一调度框架。  
- **学习式阈值**：让模型自行学习最优的置信度阈值，而不是手动设定。  
- **实时推理服务**：在大规模在线 API 中部署 DeepConf，评估实际延迟和成本节约。

### 一句话记住它
用模型自带的置信度把低质量思考链剔除，既省算力又提准确率——DeepConf 把“自信”变成了“自省”。