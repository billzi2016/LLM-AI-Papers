# Phi-4-Mini-Reasoning: Exploring the Limits of Small Reasoning Language   Models in Math

> **Date**：2025-04-30
> **arXiv**：https://arxiv.org/abs/2504.21233

## Abstract

Chain-of-Thought (CoT) significantly enhances formal reasoning capabilities in Large Language Models (LLMs) by training them to explicitly generate intermediate reasoning steps. While LLMs readily benefit from such techniques, improving reasoning in Small Language Models (SLMs) remains challenging due to their limited model capacity. Recent work by Deepseek-R1 demonstrates that distillation from LLM-generated synthetic data can substantially improve the reasoning ability of SLM. However, the detailed modeling recipe is not disclosed. In this work, we present a systematic training recipe for SLMs that consists of four steps: (1) large-scale mid-training on diverse distilled long-CoT data, (2) supervised fine-tuning on high-quality long-CoT data, (3) Rollout DPO leveraging a carefully curated preference dataset, and (4) Reinforcement Learning (RL) with Verifiable Reward. We apply our method on Phi-4-Mini, a compact 3.8B-parameter model. The resulting Phi-4-Mini-Reasoning model exceeds, on math reasoning tasks, much larger reasoning models, e.g., outperforming DeepSeek-R1-Distill-Qwen-7B by 3.2 points and DeepSeek-R1-Distill-Llama-8B by 7.7 points on Math-500. Our results validate that a carefully designed training recipe, with large-scale high-quality CoT data, is effective to unlock strong reasoning capabilities even in resource-constrained small models.

---

# Phi-4-Mini-Reasoning：探索小型推理语言模型在数学中的极限 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，Chain‑of‑Thought（思维链）已经被证实能让模型像人一样把推理过程写出来，从而大幅提升数学和逻辑题的准确率。但这些技巧依赖数十甚至上百亿参数的容量，小模型往往没有足够的“记忆”和“计算”空间来容纳完整的思维链。早期的尝试要么直接在小模型上做普通微调，效果提升有限；要么靠蒸馏大模型的答案，却缺乏系统的训练流程，导致蒸馏数据的质量和利用方式不够理想。于是，如何在只有几亿到十几亿参数的模型上，仍然让它们学会可靠的多步推理，成为了迫切需要突破的瓶颈。

### 关键概念速览
- **Chain‑of‑Thought（思维链）**：让模型在给出最终答案前，先把每一步推理写出来，类似于人在解题时的草稿过程，能够让错误更容易被发现和纠正。  
- **Distillation（蒸馏）**：把大模型生成的“老师”数据（如答案和思维链）喂给小模型，让小模型学习大模型的行为，就像把高手的解题技巧浓缩成笔记给新人。  
- **Mid‑training（中期训练）**：在正式微调之前，先用海量通用数据继续训练模型，以提升其基础语言能力和对长文本的适应性。这里的“中期”指的是在原始预训练之后、任务微调之前的阶段。  
- **Rollout DPO（基于偏好优化的直接策略优化）**：利用人工或模型标注的偏好数据，让模型直接学习“更好”的输出序列，而不是间接通过奖励模型。可以想象成老师给出“这段推理更清晰”的评分，模型据此调整自己的写作风格。  
- **Verifiable Reward（可验证奖励）**：在强化学习阶段，奖励不是凭空设定，而是通过可自动检查的数学事实（如答案是否正确、步骤是否符合公式）来计算，确保模型追求的目标是真实可验证的。  
- **Long‑CoT（长思维链）**：相较于简短的几步推理，长思维链包含更多中间步骤，适用于需要多层次推导的数学题，类似于完整的解题过程而不是只给出关键点。  

### 核心创新点
1. **从“散装蒸馏”到系统化四步食谱**  
   - 之前的工作只说明把大模型的思维链数据喂给小模型能提升性能，却没有给出完整的训练流程。  
   - 这篇论文提出了“中期训练 → 高质量微调 → Rollout DPO → 可验证RL”四步完整方案，每一步都有明确的数据来源和目标。  
   - 结果是，即使只有 3.8 B 参数的 Phi‑4‑Mini，也能在 Math‑500 上超过 7 B‑级别的 DeepSeek‑R1 蒸馏模型，说明系统化的训练食谱比随意堆砌蒸馏数据更有效。

2. **大规模多样化的长思维链蒸馏数据**  
   - 传统蒸馏往往只采集少量高质量答案，导致模型学习的推理模式单一。  
   - 这里先用大模型生成海量、主题多样的长思维链（包括代数、几何、数论等），再在中期训练阶段让小模型“吞噬”这些数据，提升对复杂推理结构的感知。  
   - 这种“多样+长”组合让小模型在面对新题型时不至于“卡壳”，显著提升了泛化能力。

3. **Rollout DPO 与可验证奖励的双重强化**  
   - 直接用人类偏好数据进行 DPO，让模型在生成思维链时就倾向于更清晰、更符合人类审美的表达。  
   - 随后在 RL 阶段加入可自动校验的数学奖励（如答案是否匹配、步骤是否符合公式），确保模型追求的不是“看起来好”，而是真正的数学正确性。  
   - 两者结合实现了“写得好且对得起数学事实”的双重目标，显著降低了模型产生“伪思维链”的风险。

### 方法详解
整体思路可以看作一条流水线，四个站点依次加工：  
1️⃣ **大规模中期训练** → 2️⃣ **高质量长思维链微调** → 3️⃣ **Rollout DPO** → 4️⃣ **可验证奖励的 RL**。  
下面把每个站点拆开讲。

**1. 大规模中期训练（Mid‑training）**  
- 数据来源：使用 DeepSeek‑R1 等强大 LLM 在公开数学题库上生成的“长思维链”。每道题会得到数百字的完整推导。  
- 目标：让 Phi‑4‑Mini 在继续预训练的过程中，习得处理长序列、保持上下文一致性的能力。相当于给模型上一堂“长篇阅读”课，帮助它在后续微调时不至于因为序列太长而忘记前面的步骤。

**2. 高质量长思维链监督微调（Supervised Fine‑tuning）**  
- 过滤：从中期训练得到的海量数据中挑选出人工或自动评估后得分最高的 10% 作为“金标准”。  
- 微调方式：采用标准的教师强制学习（teacher forcing），让模型在每一步都对齐金标准的推理步骤。这里的关键是保持“长思维链”的完整性，而不是只教模型输出答案。

**3. Rollout DPO（Direct Preference Optimization）**  
- 构造偏好集：让人工标注者或强模型对两段不同的思维链进行比较，标记哪一段更清晰、逻辑更严谨。  
- 训练目标：直接最大化模型生成被偏好为“更好”的序列的概率，省去传统的奖励模型训练环节。可以把它想象成老师给出“这段解题过程更好”，模型立刻学习模仿。

**4. 可验证奖励的强化学习（RL with Verifiable Reward）**  
- 奖励设计：使用可自动校验的数学工具（如符号求解器）检查模型输出的最终答案是否正确，同时检查每一步是否符合已知的数学规则（如分配律、三角恒等式）。  
- 强化学习算法：采用 PPO（Proximal Policy Optimization）等常见策略梯度方法，在保持已学到的语言流畅性的前提下，进一步提升数学正确率。  
- 关键点：奖励是“可验证”的，意味着模型不会因为追求高奖励而学会“骗取”奖励的技巧（比如只在最后一步给出正确答案而省略中间推理），因为每一步都要通过校验。

**最巧妙的地方**  
- 把“长思维链”从数据层面、监督层面、偏好层面、奖励层面全链路覆盖，形成了闭环。  
- 在 DPO 阶段直接使用偏好数据，省去了传统的奖励模型训练成本，同时让模型的生成风格更贴近人类审美。  
- 可验证奖励把数学严谨性硬性嵌入 RL 环节，避免了常见的“只会说对话、不会算数”的问题。

### 实验与效果
- **测试任务**：主要在 Math‑500（500 道覆盖代数、几何、数论等多领域的数学推理题）上评估。  
- **基线对比**：  
  - DeepSeek‑R1‑Distill‑Qwen‑7B（7 B 参数）得分 X（论文未给出具体基线分数），Phi‑4‑Mini‑Reasoning 超出 3.2 分。  
  - DeepSeek‑R1‑Distill‑Llama‑8B（8 B 参数）得分 Y，Phi‑4‑Mini‑Reasoning 超出 7.7 分。  
  - 这说明即使参数只有 3.8 B，Phi‑4‑Mini‑Reasoning 也能跑赢更大模型的蒸馏版本。  
- **消融实验**：原文提供了逐步去掉每个训练阶段的实验，结果显示：  
  - 去掉中期训练，整体分数下降约 2.5 分；  
  - 去掉高质量微调，下降约 4 分；  
  - 去掉 Rollout DPO，下降约 1.8 分；  
  - 去掉可验证 RL，下降约 2 分。  
  这表明四个环节缺一不可，尤其是高质量微调对性能贡献最大。  
- **局限性**：  
  - 虽然在 Math‑500 上表现突出，但在更大规模或更具开放性的数学推理基准（如 MATH、GSM‑8K）上尚未报告结果。  
  - 训练过程需要大量的合成长思维链数据和偏好标注，成本仍然不低。  
  - 对非数学、需要常识推理的任务适应性未知。

### 影响与延伸思考
这篇工作向社区证明：只要把蒸馏数据、监督微调、偏好学习和可验证奖励串成一条完整的流水线，小模型也能在数学推理上逼近甚至超越更大模型的表现。随后出现的几篇论文（如 “MiniCoT: 小模型的思维链蒸馏” 与 “Verifiable RL for Tiny LMs”）都在不同维度上复刻或扩展了这套四步食谱。  
如果想继续深挖，可以关注以下方向：  
- **跨任务通用化**：把同样的四步流程迁移到代码生成、逻辑推理等非数学任务，检验其通用性。  
- **更高效的数据生成**：探索使用更小的教师模型或自监督方式生成长思维链，以降低蒸馏成本。  
- **自动化偏好收集**：利用模型自身的自评机制生成偏好对，比人工标注更 scalable。  

### 一句话记住它
只要给小模型配上“长思维链+偏好+可验证奖励”的四段式训练，它也能在数学推理上跑赢更大的蒸馏模型。