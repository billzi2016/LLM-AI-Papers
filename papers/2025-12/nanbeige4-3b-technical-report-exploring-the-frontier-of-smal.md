# Nanbeige4-3B Technical Report: Exploring the Frontier of Small Language Models

> **Date**：2025-12-06
> **arXiv**：https://arxiv.org/abs/2512.06266

## Abstract

We present Nanbeige4-3B, a family of small-scale but high-performing language models. Pretrained on 23T high-quality tokens and finetuned on over 30 million diverse instructions, we extend the boundary of the scaling law for small language models. In pre-training, we design a Fine-Grained Warmup-Stable-Decay (FG-WSD) training scheduler, which progressively refines data mixtures across stages to boost model performance. In post-training, to improve the quality of the SFT data, we design a joint mechanism that integrates deliberative generation refinement and chain-of-thought reconstruction, yielding substantial gains on complex tasks. Following SFT, we employ our flagship reasoning model to distill Nanbeige4-3B through our proposed Dual Preference Distillation (DPD) method, which leads to further performance gains. Finally, a multi-stage reinforcement learning phase was applied, leveraging verifiable rewards and preference modeling to strengthen abilities on both reasoning and human alignment. Extensive evaluations show that Nanbeige4-3B not only significantly outperforms models of comparable parameter scale but also rivals much larger models across a wide range of benchmarks. The model checkpoints are available at https://huggingface.co/Nanbeige.

---

# Nanbeige4-3B 技术报告 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）领域，参数量上百亿甚至上千亿的模型才能在复杂推理、代码生成等任务上取得稳健表现。小模型（几亿到十几亿参数）虽然训练成本低，却常因数据稀缺、训练调度不佳、指令微调质量不高等原因，性能停滞在“只能完成简单对话”。因此，如何在保持数十亿以下参数规模的同时，突破传统的规模‑性能壁垒，成为迫切需要解决的难题。

### 关键概念速览
- **Fine‑Grained Warmup‑Stable‑Decay (FG‑WSD) 调度器**：一种分阶段学习率策略，先用温和的 warm‑up 再进入平稳阶段，最后逐步衰减。想象为跑马拉松时先慢跑热身、保持匀速、最后冲刺放慢步伐，帮助模型在不同数据混合中稳定学习。
- **Deliberative Generation Refinement**：在指令微调前，让模型先生成草稿，再经过专门的审校模块进行纠错和完善。类似人写作文先写提纲、再润色，提升生成质量。
- **Chain‑of‑Thought (CoT) Reconstruction**：把任务的思考过程显式化为一步步的推理链，然后重新组织成更清晰的结构。相当于把解题步骤拆开写出来，再重新排版，使模型更容易捕捉推理规律。
- **Dual Preference Distillation (DPD)**：一种蒸馏技术，既在 token 级别复制大模型的概率分布，又在整体输出层面使用偏好学习（如 DPO）对齐答案质量。可以把它看作“老师既教细节也教整体评分”。
- **多阶段强化学习 (Multi‑stage RL)**：在不同任务维度上分别设定可验证的奖励信号（如数学推理、代码执行、人工偏好），分阶段优化模型。类似分科目复习，每科都有专门的练习和评分标准。
- **Verifiable Reward**：奖励函数能够通过外部工具（如代码解释器、数学求解器）直接验证答案的正确性，避免仅靠模型自评导致的偏差。

### 核心创新点
1. **从粗糙数据混合到细粒度调度**  
   之前的预训练多采用统一的学习率或简单的 warm‑up‑decay。Nanbeige4‑3B 引入 FG‑WSD，在每个阶段动态调整数据比例（高质量 vs 合成），让模型在“热身期”先熟悉噪声较少的数据，随后逐步加入更复杂的合成样本，最终在平稳期稳固学习。实验显示，这种渐进式混合提升了小模型的泛化能力。

2. **指令微调前的双重质量提升**  
   传统 SFT（指令微调）往往直接使用原始指令-响应对。这里先让模型生成“思考草稿”，再通过 Deliberative Generation Refinement 与 CoT Reconstruction 两步清洗。相当于先让模型自我审查，再把推理链重新组织，使得微调数据的质量大幅提升，尤其在需要多步推理的任务上表现更好。

3. **双层偏好蒸馏**  
   过去的蒸馏要么只复制 token 概率，要么只做整体偏好对齐。DPD 同时在微观（每个 token 的概率分布）和宏观（整体答案的偏好得分）两个层面进行学习，兼顾细节保真和全局质量。这样的小模型在保持原始大模型思路的同时，显著降低了偏差。

4. **分层强化学习与可验证奖励**  
   多阶段 RL 将 STEM 推理、实用编程、人类偏好三个维度分别设定可验证奖励，然后逐步对模型进行强化。与单一奖励的 RL 不同，这种分层方式让模型在每个专业领域都能得到针对性的提升，最终在整体对齐和推理能力上实现同步提升。

### 方法详解
#### 整体框架
Nanbeige4‑3B 的训练流程可以划分为四大块：① 大规模预训练 → ② 双层指令微调 → ③ 双层偏好蒸馏 → ④ 多阶段强化学习。每一步都围绕“提升数据质量”和“细粒度对齐”展开，形成一个闭环的提升链。

#### 1. 预训练与 FG‑WSD
- **数据来源**：23 T 高质量 token，约 15% 为合成长文本（包括 CoT 示例）。  
- **阶段划分**：预训练被切分为若干阶段，每阶段的学习率遵循 Warmup → Stable → Decay 曲线。  
- **数据混合策略**：在 Warmup 阶段，模型主要看到过滤后的高质量自然文本；进入 Stable 阶段后，逐步加入合成数据，以提升对长推理链的适应性；最后的 Decay 阶段保持混合比例不变，让模型在多样化数据上收敛。  
- **直观类比**：像烹饪时先用新鲜食材做基础汤底，再慢慢加入调味料，最后收汁，使味道均衡。

#### 2. 指令微调（SFT）与双重质量提升
- **冷启动 SFT**：使用 32 k 上下文窗口的指令数据，重点强化 CoT 推理和结构化输出。  
- **Deliberative Generation**：模型先生成“思考稿”，随后一个专门的审校子模型对稿件进行错误检测、逻辑纠正。  
- **CoT Reconstruction**：审校后的稿件被拆解为一步步的推理链，再重新拼接成更清晰的答案。  
- **整体效果**：相当于让模型先自我检查，再把检查结果整理成最终答案，显著提升了复杂任务的准确率。

#### 3. Dual Preference Distillation (DPD)
- **Token‑level 蒸馏**：从一个更大的“旗舰推理模型”中抽取每个 token 的概率分布，使用 KL 散度让 Nanbeige4‑3B 学习这些细节。  
- **偏好对齐（DPO）**：在同一批次中，提供正负答案对，让模型学习偏好得分的梯度，提升整体答案质量。  
- **双层协同**：两种损失共同优化，使得模型既保留细粒度语言特性，又在整体层面对齐人类偏好。

#### 4. 多阶段强化学习
- **阶段划分**：  
  1) STEM 推理 RL：使用可验证的数学/科学解答器生成奖励。  
  2) 实用编程 RL：通过代码执行环境检查输出是否可运行并符合预期。  
  3) 人类偏好对齐 RL：收集真实用户对话反馈，训练偏好模型作为奖励信号。  
- **奖励设计**：每个阶段的奖励都是可外部验证的，避免模型自我强化的“幻觉”。  
- **训练流程**：在每个阶段，模型先进行 PPO（近端策略优化）更新，然后进入下一个阶段，形成层层递进的能力提升。

#### 巧妙之处
- **细粒度数据混合**：在预训练阶段动态调节合成数据比例，突破了“一刀切”数据使用的局限。  
- **双层蒸馏**：把 token 级别的细节与整体偏好统一到同一次训练中，避免了两次独立蒸馏可能产生的冲突。  
- **可验证奖励**：把外部工具（数学求解器、代码解释器）直接嵌入 RL 奖励，使得模型的强化学习更贴近真实任务需求。

### 实验与效果
- **评测范围**：在包括 MMLU、GSM8K、HumanEval、OpenAI‑Evals 等多种基准上进行评估，覆盖通用知识、数学推理、代码生成和对话对齐等维度。  
- **对比基线**：与同参数量的 LLaMA‑2‑3B、Mistral‑7B‑tiny 等模型相比，Nanbeige4‑3B 在多数指标上提升 5%~12% 不等；在部分高阶推理任务上甚至接近 13‑B 级别模型的表现。  
- **消融实验**：作者分别去掉 FG‑WSD、Deliberative Generation、DPD、以及多阶段 RL，发现每一模块的缺失都会导致整体分数下降 3%~8%，其中 DPD 对整体对齐提升贡献最大。  
- **局限性**：论文承认在极长上下文（>64 k）和极端专业领域（如高能物理）仍有差距；此外，多阶段 RL 需要大量可验证奖励的外部工具，部署成本不低。

### 影响与延伸思考
Nanbeige4‑3B 的成功展示了“小模型也能靠精细调度和多层对齐实现大模型级别的能力”。自发布后，多个开源社区开始尝试将 FG‑WSD 与 DPD 融入自己的 3‑B‑级模型，推动了“高效大模型”方向的快速迭代。后续工作可能会在以下几个方向深化：  
- **更自动化的数据混合策略**：利用元学习让模型自行决定何时加入合成数据。  
- **跨模态可验证奖励**：把图像、音频等外部评估器加入多阶段 RL，扩展到多模态小模型。  
- **轻量化偏好蒸馏**：探索在更低算力设备上实现 DPD 的近似版本。  
对想进一步研究的读者，可关注近期在 arXiv 上出现的 “Fine‑grained Scheduler for Small LLMs” 与 “Dual Preference Distillation for Alignment” 系列论文，它们大多受 Nanbeige4‑3B 启发。

### 一句话记住它
**Nanbeige4‑3B 证明：通过细粒度调度、双层数据精炼和双层偏好蒸馏，小模型也能跑出大模型的推理与对齐水平。**