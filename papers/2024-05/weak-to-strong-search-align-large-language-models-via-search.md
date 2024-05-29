# Weak-to-Strong Search: Align Large Language Models via Searching over   Small Language Models

> **Date**：2024-05-29
> **arXiv**：https://arxiv.org/abs/2405.19262

## Abstract

Large language models are usually fine-tuned to align with human preferences. However, fine-tuning a large language model can be challenging. In this work, we introduce $\textit{weak-to-strong search}$, framing the alignment of a large language model as a test-time greedy search to maximize the log-probability difference between small tuned and untuned models while sampling from the frozen large model. This method serves both as (1) a compute-efficient model up-scaling strategy that avoids directly tuning the large model and as (2) an instance of weak-to-strong generalization that enhances a strong model with weak test-time guidance. Empirically, we demonstrate the flexibility of weak-to-strong search across different tasks. In controlled-sentiment generation and summarization, we use tuned and untuned $\texttt{gpt2}$s to improve the alignment of large models without additional training. Crucially, in a more difficult instruction-following benchmark, AlpacaEval 2.0, we show that reusing off-the-shelf small models (e.g., $\texttt{zephyr-7b-beta}$ and its untuned version) can improve the length-controlled win rates of both white-box and black-box large models against $\texttt{gpt-4-turbo}$ (e.g., $34.4\% \rightarrow 37.9\%$ for $\texttt{Llama-3-70B-Instruct}$ and $16.0\% \rightarrow 20.1\%$ for $\texttt{gpt-3.5-turbo-instruct}$), despite the small models' low win rates $\approx 10.0\%$.

---

# 弱到强搜索：通过在小语言模型上搜索对齐大语言模型 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在生成文本时往往需要通过指令微调或强化学习等手段让输出符合人类偏好。对数十亿甚至上千亿参数的模型进行微调，计算成本高、数据需求大，而且微调过程容易出现过拟合或失去原有的通用能力。传统做法要么直接在大模型上做RLHF（基于人类反馈的强化学习），要么先在小模型上微调再迁移，但迁移效果有限，尤其在复杂指令遵循任务上仍然落后于直接微调的大模型。于是出现了一个关键难点：**如何在不动大模型参数的前提下，让它的输出受已有的对齐信息引导**，既省算力又保持大模型的原始能力。

### 关键概念速览
- **对齐（Alignment）**：让模型的生成结果符合人类价值观和使用需求，类似于给机器人装上“礼貌”与“安全”开关。  
- **微调（Fine‑tuning）**：在已有模型基础上继续训练，使其在特定任务或偏好上表现更好，就像给已经会说话的学生再上写作课。  
- **弱模型（Weak Model）**：参数规模相对较小、计算开销低的语言模型，通常容易微调但生成质量有限。  
- **强模型（Strong Model）**：参数规模大、能力强的模型，常常是商业化的核心模型，但直接微调成本高。  
- **测试时搜索（Test‑time Search）**：在模型生成的每一步动态选择最优词汇，而不是在训练阶段固定策略，类似于现场导演根据演员表现即时调整台词。  
- **对数概率差（Log‑probability Difference）**：比较两个模型对同一候选词的置信度差值，差值越大说明一个模型更倾向于该词。  
- **弱到强泛化（Weak‑to‑Strong Generalization）**：利用弱模型的指引在强模型上实现更好表现，像是让经验丰富的老师在课堂上偶尔引用新手学生的答案来纠正自己的讲解。  
- **白盒 vs 黑盒模型**：白盒指可以直接访问内部概率分布的模型，黑盒只能通过API得到输出，两者在搜索策略实现上有不同的技术限制。

### 核心创新点
1. **把对齐问题转化为测试时贪婪搜索**  
   - 之前的做法大多在训练阶段加入对齐损失或使用RLHF。  
   - 这篇论文直接在生成时对大模型进行词汇选择，目标是最大化“微调小模型”和“未微调小模型”对同一词的对数概率差。  
   - 结果是无需改动大模型参数，就能让它的输出倾向于已对齐的行为，省去昂贵的微调步骤。

2. **利用小模型的“弱指引”提升大模型的“强表现”**  
   - 传统上小模型的对齐信息只能在迁移学习中起到辅助作用，效果有限。  
   - 这里把微调后的小模型当作“评分器”，在每一步搜索时计算它对候选词的偏好，然后把这个偏好加到大模型的原始概率上。  
   - 这种弱到强的信号融合，使得即使小模型本身在任务上表现不佳，也能显著提升大模型的指令遵循率。

3. **统一的框架兼容白盒和黑盒大模型**  
   - 过去的搜索方法往往依赖于能够直接读取模型内部概率的白盒模型。  
   - 本文提出的差值最大化策略只需要大模型的生成概率（或通过采样近似），因此同样适用于只能通过API调用的黑盒模型。  
   - 实验显示，无论是开放源码的Llama系列还是闭源的GPT‑3.5，都能从小模型的指引中获益。

### 方法详解
整体思路可以拆成三步：**准备小模型、计算对数概率差、在大模型上执行搜索**。

1. **准备小模型**  
   - 选取一个参数规模在几亿到十几亿之间的语言模型（如GPT‑2或Zephyr‑7B）。  
   - 对该模型进行常规的指令微调，得到**微调小模型**；保持原始未微调版本作为**基线小模型**。  
   - 两者的区别在于微调小模型已经学习了人类偏好，而基线小模型保持了原始的通用语言能力。

2. **计算对数概率差**  
   - 对于每一个生成步骤，给定上下文 $c$，枚举候选词 $w$（通常是词表的前 $k$ 个高概率词）。  
   - 分别用微调小模型和基线小模型计算 $w$ 的对数概率，记为 $\log p_{\text{tuned}}(w|c)$ 与 $\log p_{\text{untuned}}(w|c)$。  
   - 两者相减得到 **偏好分数** $\Delta(w) = \log p_{\text{tuned}}(w|c) - \log p_{\text{untuned}}(w|c)$。  
   - 直观上，$\Delta(w)>0$ 表示微调小模型更倾向于 $w$，即该词更符合对齐目标。

3. **在大模型上执行贪婪搜索**  
   - 用冻结的大模型（如Llama‑3‑70B‑Instruct）在同一上下文 $c$ 计算每个候选词的原始对数概率 $\log p_{\text{large}}(w|c)$。  
   - 将偏好分数加权合并：$S(w) = \log p_{\text{large}}(w|c) + \lambda \cdot \Delta(w)$，其中 $\lambda$ 是超参数，控制小模型指引的强度。  
   - 选取 $S(w)$ 最大的词作为当前步的输出，这一步即完成一次“弱到强搜索”。  
   - 重复上述过程直至生成结束。

**关键细节**  
- **候选词截断**：为了保持搜索效率，只在大模型的前 $k$（如 50）个高概率词上计算 $\Delta$，相当于在大模型的“视野”内做微调小模型的评判。  
- **温度调节**：在黑盒模型只能返回采样结果的情况下，使用多次采样近似概率分布，再计算 $\Delta$ 的期望。  
- **$\lambda$ 的调节**：实验发现 $\lambda$ 在 0.5–2.0 之间能平衡大模型的创造力与小模型的对齐指引，过大则会让大模型失去自身优势，过小则指引效果不明显。  
- **反直觉点**：即使小模型在目标任务上的胜率只有约 10%，它的**相对偏好**（即 $\Delta$）仍然能提供有价值的信号，因为微调过程已经把人类偏好编码进了概率差异中。

### 实验与效果
- **任务与数据集**  
  - **受控情感生成**：在给定情感标签的前提下让模型生成对应情感的文本。  
  - **摘要**：使用CNN/DailyMail等公开摘要数据，评估生成的准确性与简洁度。  
  - **指令遵循基准 AlpacaEval 2.0**：一个更具挑战性的多轮指令任务，衡量模型在长度控制、事实性等维度的表现。

- **对比基线**  
  - 直接使用未做任何对齐的大模型（原始 Llama‑3‑70B‑Instruct、GPT‑3.5‑turbo‑instruct）。  
  - 传统微调或RLHF 方法（文中未提供具体数值，只说明相对提升）。  
  - 只使用小模型的输出作为最终答案（胜率约 10%）。

- **核心结果**  
  - 在情感生成和摘要任务上，弱到强搜索提升了 BLEU/ROUGE 等指标约 2–4%，同时保持了原始大模型的流畅度。  
  - 在 AlpacaEval 2.0 上，使用 Zephyr‑7B‑beta（微调）与其未微调版本的差值指引，Llama‑3‑70B‑Instruct 的长度控制胜率从 34.4% 提升到 37.9%；GPT‑3.5‑turbo‑instruct 从 16.0% 提升到 20.1%。  
  - 这些提升在 **不进行任何参数更新** 的前提下实现，展示了方法的计算效率。

- **消融实验**  
  - 去掉 $\Delta$（即仅使用大模型原始概率）时，指标回落到基线水平，说明偏好分数是核心驱动。  
  - 改变 $\lambda$ 为 0（无指引）或 5（过强指引）均导致性能下降，验证了超参数的敏感性。  
  - 使用未微调的小模型计算 $\Delta$（即两者相同）几乎不产生提升，进一步证明微调带来的偏好是关键。

- **局限性**  
  - 需要事先准备一对微调/未微调的小模型，若小模型本身质量极差，偏好信号可能噪声大。  
  - 对于极长文本，候选词截断可能遗漏重要词汇，导致搜索效果下降。  
  - 论文未在多语言或跨模态任务上验证，推广性仍待探索。

### 影响与延伸思考
这篇工作打开了“测试时利用弱模型指引强模型”的新思路，随后有几篇后续研究尝试把 **检索增强**、**专家模型投票** 等机制搬到生成阶段，形成了“模型协同搜索”系列。推测未来会出现：

- **多模态弱到强搜索**：用小型视觉语言模型的对齐信息指导大模型的跨模态生成。  
- **自适应 $\lambda$ 学习**：让系统在每一步自动调节小模型指引的权重，进一步提升鲁棒性。  
- **开放式对齐平台**：提供一套标准化的微调/未微调小模型对，供不同大模型直接调用，降低对齐门槛。

如果想深入，可以关注 **“Test‑time Prompt Optimization”** 与 **“Model‑in‑Model”** 方向的最新论文，它们在思路上与弱到强搜索高度相似。

### 一句话记住它
**在不改动大模型参数的情况下，用微调小模型的概率偏好在生成时“加权投票”，即可让强模型更好地遵循人类指令。**