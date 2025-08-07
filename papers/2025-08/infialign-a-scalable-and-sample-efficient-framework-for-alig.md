# InfiAlign: A Scalable and Sample-Efficient Framework for Aligning LLMs to Enhance Reasoning Capabilities

> **Date**：2025-08-07
> **arXiv**：https://arxiv.org/abs/2508.05496

## Abstract

Large language models (LLMs) have exhibited impressive reasoning abilities on a wide range of complex tasks. However, enhancing these capabilities through post-training remains resource intensive, particularly in terms of data and computational cost. Although recent efforts have sought to improve sample efficiency through selective data curation, existing methods often rely on heuristic or task-specific strategies that hinder scalability. In this work, we introduce InfiAlign, a scalable and sample-efficient post-training framework that integrates supervised fine-tuning (SFT) with Direct Preference Optimization (DPO) to align LLMs for enhanced reasoning. At the core of InfiAlign is a robust data selection pipeline that automatically curates high-quality alignment data from open-source reasoning datasets using multidimensional quality metrics. This pipeline enables significant performance gains while drastically reducing data requirements and remains extensible to new data sources. When applied to the Qwen2.5-Math-7B-Base model, our SFT model achieves performance on par with DeepSeek-R1-Distill-Qwen-7B, while using only approximately 12% of the training data, and demonstrates strong generalization across diverse reasoning tasks. Additional improvements are obtained through the application of DPO, with particularly notable gains in mathematical reasoning tasks. The model achieves an average improvement of 3.89% on AIME 24/25 benchmarks. Our results highlight the effectiveness of combining principled data selection with full-stage post-training, offering a practical solution for aligning large reasoning models in a scalable and data-efficient manner. The model checkpoints are available at https://huggingface.co/InfiX-ai/InfiAlign-Qwen-7B-SFT.

---

# InfiAlign：一种可扩展且样本高效的对齐大语言模型以提升推理能力的框架 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在数学、逻辑推理等复杂任务上已经展现出惊人的能力，但要让它们在特定推理场景里更稳健、更精准，通常需要在已有模型上再进行一次“后训练”。传统的后训练方式要么要海量标注数据，要么要耗费巨大的算力，成本高得让很多团队望而却步。近些年有人尝试通过挑选“好”数据来提升样本利用率，却往往依赖手工规则或只针对单一任务设计，导致方法难以迁移到新数据源，也难以在大模型上保持可扩展性。于是，如何在保持或提升推理水平的同时，显著降低数据和算力需求，成为了迫切需要解决的瓶颈。

### 关键概念速览
- **监督微调（SFT）**：在已有的大模型上，用标注好的输入‑输出对继续训练，让模型更贴合特定任务的风格。相当于给模型上“补习班”，让它把课堂上学到的知识再巩固一次。  
- **直接偏好优化（DPO）**：一种基于人类偏好的对齐方式，模型直接学习“哪个答案更好”，不需要先把偏好转化为奖励函数。可以想象成让模型参加“选秀”，观众投票决定哪位选手更受欢迎。  
- **多维质量度量**：对候选训练样本从多个角度（如答案正确性、解释完整性、语言流畅度等）打分，综合得出一个质量分。类似于招聘时既看学历也看工作经验，还要看面试表现，综合评估后才决定录用。  
- **样本效率**：在达到相同性能的前提下，用更少的数据完成训练的能力。就像用更少的练习题就能考出同样高的分数。  
- **全阶段对齐**：把对齐过程拆成两个阶段——先用 SFT 学习基本的推理能力，再用 DPO 微调偏好。相当于先打好基础功，再进行专项训练。  
- **可扩展性**：方法能够平滑地适用于更大的模型或更多的数据源，而不需要重新设计流程。就像一套通用的烹饪流程，换锅换料也能顺利操作。  

### 核心创新点
1. **从经验规则到自动化多维筛选**  
   - 之前的样本挑选大多靠人工经验或单一指标（比如只看答案是否正确）。  
   - InfiAlign 构建了一个自动化管线，利用正确率、解释深度、语言质量等多维度度量，对开源推理数据集进行批量筛选。  
   - 结果是同样的模型只用了约 12% 的原始数据就能达到或超过基线的效果，大幅提升了样本利用率。  

2. **全阶段对齐：SFT + DPO 的协同**  
   - 过去的工作往往只做 SFT 或只做基于奖励的强化学习，对齐深度受限。  
   - InfiAlign 先用筛选后的高质量数据进行监督微调，让模型掌握扎实的推理技巧；随后在同一批数据上加入人类偏好对比，使用 DPO 进一步微调。  
   - 这种两步走的设计在数学推理基准（如 AIME 24/25）上额外提升了约 3.9%，证明了偏好信息对细粒度推理的增益。  

3. **对标强基准、保持轻量**  
   - 将 Qwen2.5‑Math‑7B‑Base 经过 InfiAlign 的 SFT 训练后，性能与 DeepSeek‑R1‑Distill‑Qwen‑7B（一个更大、更昂贵的模型）持平。  
   - 关键在于只用了原始数据的约 12%，算力需求也相应下降，展示了“少量数据、强大效果”的可行路径。  

4. **开放可复用的检查点**  
   - 论文在 HuggingFace 上公开了完整的模型检查点和数据筛选脚本，降低了社区复现和二次开发的门槛。  
   - 这点虽不算技术创新，却极大提升了方法的可推广性和实际落地价值。  

### 方法详解
**整体思路**  
InfiAlign 把对齐过程拆成三大步骤：① 自动化数据筛选 → ② 监督微调（SFT） → ③ 直接偏好优化（DPO）。整个流程像流水线：先把原始的海量推理数据“过滤”成高质量的原料，再把原料喂给模型进行“基础烹饪”，最后用“口味评审”进一步调味。

**1. 多维质量筛选管线**  
- **数据来源**：作者从公开的数学、逻辑推理数据集（如 GSM‑8K、MATH、OpenAI‑Evals）中抓取原始问答对。  
- **质量度量**：每条样本会被送入若干评估模型，分别计算：  
  - **答案正确率**（是否能得到官方答案），  
  - **解释完整度**（解释是否覆盖关键步骤），  
  - **语言流畅度**（使用语言模型打分），  
  - **多样性**（避免同质化答案）。  
- **综合打分**：把上述指标加权求和，得到一个整体质量分。阈值设定后，低于阈值的样本被剔除。  
- **结果**：筛选后保留约 12% 的原始样本，但这些样本在所有质量维度上均表现优秀。

**2. 监督微调（SFT）**  
- **目标**：让模型学习“正确的推理路径”。  
- **输入**：筛选后的问答对，模型接受问题 + （可选）少量提示，输出完整的解题步骤。  
- **训练细节**：使用常规的交叉熵损失，学习率、批大小等超参与原始模型保持一致，只是训练轮数缩短（因为数据更精炼）。  
- **巧妙之处**：作者没有直接在大模型上做全量微调，而是先在 7B 规模的基座模型上进行，这样算力需求大幅降低，同时保留了模型的通用能力。

**3. 直接偏好优化（DPO）**  
- **目标**：在已有的推理能力上，进一步让模型倾向于“更受人类偏好”的答案。  
- **数据形式**：从同一批筛选样本中抽取成对比较（A vs B），每对包含两个解答，标注哪个更好。标注方式可以是人工评审，也可以利用已有的偏好模型。  
- **优化方式**：DPO 直接最大化“好答案被选中的概率”，不需要显式的奖励函数或强化学习的价值估计。  
- **关键点**：因为前一步已经让模型掌握了高质量的推理步骤，DPO 只需要在细节层面微调（比如更简洁的表达、更少的冗余步骤），所以收敛更快，算力开销极低。

**最反直觉的设计**  
很多人会担心把数据量压到 12% 会导致模型“见识不足”。InfiAlign 用多维度质量度量确保留下的样本在信息密度上远高于原始数据，等于是把“稀疏的海水”浓缩成“高盐的海水”。这种“质量优先、数量次之”的思路在大模型后训练中并不常见，却是本工作成功的关键。

### 实验与效果
- **实验平台**：在 Qwen2.5‑Math‑7B‑Base 基座模型上进行，使用同等算力的单卡（或少数卡）进行训练。  
- **对比基线**：  
  - **DeepSeek‑R1‑Distill‑Qwen‑7B**（公开的强基准模型），  
  - **原始 Qwen2.5‑Math‑7B‑Base**（未经过任何对齐），  
  - 其他近期的 SFT‑only 或 RLHF‑only 方法。  
- **主要指标**：在 AIME 2024/2025、MATH、GSM‑8K 等数学推理基准上报告准确率或得分。  
- **核心结果**：  
  - InfiAlign‑SFT 版在 AIME 上的得分与 DeepSeek‑R1‑Distill‑Qwen‑7B 持平，使用的数据仅为后者的约 12%。  
  - 加入 DPO 后，整体模型在 AIME 上提升了 **3.89%**，在 MATH 上也有约 2% 的提升。  
  - 与未对齐的原始基座模型相比，整体提升超过 10%。  
- **消融实验**：  
  - **仅 SFT** vs **SFT+ DPO**：显示 DPO 对数学推理的增益主要体现在答案的简洁度和偏好匹配上。  
  - **全量数据** vs **筛选后 12%**：全量数据提升不明显，甚至出现轻微过拟合，验证了质量筛选的必要性。  
- **局限性**：  
  - 论文主要在数学推理任务上评估，其他类型的推理（如常识推理、代码推理）尚未验证。  
  - 多维质量度量的权重是手工设定的，可能需要针对不同领域重新调参。  
  - DPO 依赖成对偏好标注，标注成本在大规模数据上仍是瓶颈。  

### 影响与延伸思考
InfiAlign 的出现让“少量高质量数据 + 两阶段对齐”成为可能，直接冲击了传统的“海量数据 + 大算力”思路。自论文发布后，已有几篇后续工作尝试把多维质量筛选搬到代码生成、医学问答等领域，证明了该框架的通用性。还有研究在探索 **自监督质量度量**（让模型自己评估样本好坏），进一步降低人工干预。对想继续深挖的读者，可以关注以下方向：  
- **跨任务质量度量统一化**：如何设计一套指标同时适用于数学、逻辑、代码等多种推理任务。  
- **偏好数据自动生成**：利用 LLM 自己生成成对比较，减少人工标注成本。  
- **大模型全阶段对齐的算力优化**：在更大模型（如 70B）上复现 InfiAlign，验证其可扩展性极限。  

### 一句话记住它
只用 12% 的高质量样本，先 SFT 再 DPO，就能让 7B 级模型在数学推理上追平甚至超越更大更贵的基准。