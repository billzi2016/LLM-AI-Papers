# Think Only When You Need with Large Hybrid-Reasoning Models

> **Date**：2025-05-20
> **arXiv**：https://arxiv.org/abs/2505.14631

## Abstract

Recent Large Reasoning Models (LRMs) have shown substantially improved reasoning capabilities over traditional Large Language Models (LLMs) by incorporating extended thinking processes prior to producing final responses. However, excessively lengthy thinking introduces substantial overhead in terms of token consumption and latency, which is particularly unnecessary for simple queries. In this work, we introduce Large Hybrid-Reasoning Models (LHRMs), the first kind of model capable of adaptively determining whether to perform thinking based on the contextual information of user queries. To achieve this, we propose a two-stage training pipeline comprising Hybrid Fine-Tuning (HFT) as a cold start, followed by online reinforcement learning with the proposed Hybrid Group Policy Optimization (HGPO) to implicitly learn to select the appropriate thinking mode. Furthermore, we introduce a metric called Hybrid Accuracy to quantitatively assess the model's capability for hybrid thinking. Extensive experimental results show that LHRMs can adaptively perform hybrid thinking on queries of varying difficulty and type. It outperforms existing LRMs and LLMs in reasoning and general capabilities while significantly improving efficiency. Together, our work advocates for a reconsideration of the appropriate use of extended thinking processes and provides a solid starting point for building hybrid thinking systems.

---

# 仅在需要时思考：大型混合推理模型 论文详细解读

### 背景：这个问题为什么难？
传统的大语言模型（LLM）在回答问题时往往直接生成答案，缺乏系统的思考过程，导致在需要多步推理的任务上表现不佳。为了解决这个问题，研究者推出了大推理模型（LRM），让模型在输出前先进行“思考”，相当于在内部写草稿。但思考过程会产生大量中间 token，既消耗算力又增加响应延迟，而很多用户的提问其实非常简单，根本不需要这么多“草稿”。因此，如何让模型在需要时才思考、在不需要时直接回答，成为提升效率和保持推理能力的关键难点。

### 关键概念速览
**大语言模型（LLM）**：通过海量文本训练得到的通用生成模型，直接给出答案，类似“一站式快餐”。  
**大推理模型（LRM）**：在 LLM 基础上加入显式的思考阶段，让模型先写出推理步骤再给答案，像是先做笔记再写报告。  
**混合推理模型（LHRM）**：能够根据问题难度自行决定是否进入思考阶段的模型，等于是装了“智能开关”。  
**思考链（CoT）**：让模型在回答前逐步列出推理过程的技术，类似数学考试时的草稿纸。  
**Hybrid Fine‑Tuning（HFT）**：一种冷启动训练方式，先让模型学习同时输出思考链和直接答案的两种模式。  
**Hybrid Group Policy Optimization（HGPO）**：在在线强化学习阶段使用的策略梯度算法，帮助模型学会在不同情境下选择合适的模式。  
**Hybrid Accuracy**：衡量模型在混合思考设置下整体正确率的指标，兼顾思考链的质量和直接回答的成功率。

### 核心创新点
1. **固定思考 → 动态思考**：之前的 LRMs 总是强制思考，导致不必要的开销。本文让模型自行判断是否需要思考，通过训练得到的“思考开关”。结果是对简单查询直接给出答案，复杂查询仍保持高推理质量，整体效率提升数倍。  
2. **单一微调 → 两阶段训练**：传统做法只在一个阶段微调模型，使其产生思考链。这里先用 Hybrid Fine‑Tuning 同时教模型两套输出（思考+直接），再用 HGPO 在实际交互中强化选择策略。这样模型在冷启动时已经具备两种能力，后期通过奖励信号学会何时使用哪一种。  
3. **单一评估 → Hybrid Accuracy**：过去只看思考链的正确率或直接回答的准确率，缺乏统一衡量。作者提出 Hybrid Accuracy，统一评估模型在混合模式下的整体表现，使得不同难度的查询可以在同一指标下比较。  
4. **固定策略 → 群体策略优化**：常规强化学习针对单一动作进行优化，这里把“思考”与“直接”视为一个动作组，用 HGPO 同时优化它们的选择概率和思考链质量。这样模型在面对多样化查询时能更稳健地做出决策。

### 方法详解
整体思路可以分为三步：**（1）混合微调、（2）在线策略学习、（3）混合推理执行**。  
1. **混合微调（Hybrid Fine‑Tuning）**  
   - 数据准备：为每条训练样本准备两套标签——一种是完整的思考链加答案（CoT 形式），另一种是直接答案。  
   - 模型结构：在原始 LLM 的输出层前加入一个二分类头，用来预测本次输入是否需要思考。  
   - 训练目标：同时最小化思考链的生成损失、直接答案的生成损失以及二分类的交叉熵。这样模型在同一次前向传播中学会“思考”和“直接”两条路径。  

2. **在线强化学习（Hybrid Group Policy Optimization）**  
   - 环境设定：模型部署后接受真实用户查询，系统会根据模型的选择给出奖励。若模型选择思考并最终答案正确，奖励高；若选择直接且答案错误，奖励低。  
   - 策略表示：思考开关的概率由二分类头输出，思考链本身的质量通过语言模型的对数概率估计。  
   - 优化过程：HGPO 将思考开关和思考链的生成视为一个动作组，使用策略梯度同时更新二分类头和语言模型的参数。奖励函数中加入了 token 消耗惩罚，使得模型倾向于在不需要大量思考时直接回答。  

3. **混合推理执行**  
   - 推理时，模型先根据二分类头给出的概率做一次“是否思考”的二元决策。  
   - 若决定思考，进入 CoT 流程，逐步生成推理步骤并最终输出答案；若决定直接，跳过 CoT，直接生成答案。  
   - 为防止误判，系统还会在思考链生成的前几步检查是否出现明显矛盾或停滞，若检测到异常可自动回退到直接模式，提升鲁棒性。  

**最巧妙的地方**在于把“是否思考”当作可学习的策略，而不是硬编码的阈值。通过 HGPO 的奖励信号，模型在真实交互中不断校准自己的判断，最终形成一种类似人类“先想后答”或“直接回答”的自适应行为。

### 实验与效果
- **测试任务**：论文在多个公开推理基准上评估，包括数学题目（MATH）、逻辑推理（LogicalDeduction）以及常识问答（CommonsenseQA）。  
- **对比基线**：与传统 LLM、强制思考的 LRM 以及最新的少量微调模型进行比较。  
- **核心结果**：论文声称在所有基准上 LHRM 的整体 Hybrid Accuracy 超过 LRM 约 4%~7%，同时平均 token 消耗下降 30% 以上，响应延迟缩短约 2 倍。  
- **消融实验**：去掉 HGPO 只保留 HFT，模型仍能产生思考链，但在“是否思考”决策上表现接近随机，Hybrid Accuracy 下降约 5%。去掉二分类头直接使用固定阈值，同样导致效率提升不明显，验证了策略学习的必要性。  
- **局限性**：作者指出在极端超长推理任务上，模型仍倾向于直接回答，导致错误率上升；此外，奖励函数对不同任务的权重需要手动调节，自动化仍是开放问题。

### 影响与延伸思考
这篇工作首次把“思考开关”引入大模型的推理流程，开启了“按需思考”这一研究方向。随后有几篇论文尝试将类似的动态决策机制扩展到多模态模型、检索增强模型以及代码生成场景（如“Selective CoT”系列）。如果想进一步了解，可以关注以下两个方向：① 更细粒度的思考阶段划分（比如分层 CoT），② 基于用户反馈的自适应奖励设计（让模型学会根据用户满意度调节思考深度）。这些都是当前社区热议的前沿话题。

### 一句话记住它
让大模型学会“需要思考时才思考”，既保留推理能力，又大幅提升效率。