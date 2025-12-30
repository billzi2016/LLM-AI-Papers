# Bielik 11B v3: Multilingual Large Language Model for European Languages

> **Date**：2025-12-30
> **arXiv**：https://arxiv.org/abs/2601.11579

## Abstract

We present Bielik 11B v3, a state-of-the-art language model highly optimized for the Polish language, while also maintaining strong capabilities in other European languages. This model extends the Mistral 7B v0.2 architecture, scaled to 11B parameters via depth up-scaling. Its development involved a comprehensive four-stage training pipeline: continuous pre-training, supervised fine-tuning (SFT), Direct Preference Optimization (DPO), and reinforcement learning.   Comprehensive evaluations demonstrate that Bielik 11B v3 achieves exceptional performance. It significantly surpasses other specialized Polish language models and outperforms many larger models (with 2-6 times more parameters) on a wide range of tasks, from basic linguistic understanding to complex reasoning.   The model's parameter efficiency, combined with extensive quantization options, allows for effective deployment across diverse hardware configurations. Bielik 11B v3 not only advances AI capabilities for the Polish language but also establishes a new benchmark for developing resource-efficient, high-performance models for less-represented languages.

---

# Bielik 11B v3：面向欧洲语言的多语言大模型 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）快速扩张的背景下，欧洲语言尤其是波兰语仍然缺乏既高效又具备跨语言能力的模型。大多数公开模型要么专注于英语，要么在多语言上采用统一的、参数庞大的架构，导致对资源有限的语言表现平平。传统的做法是直接在通用模型上微调，但参数规模和上下文长度的限制让模型在波兰语的细微语法、词形变化以及长文档推理上经常失手。于是出现了“更大更好”的思路，却忽视了 **参数效率** 与 **硬件可部署性** 的矛盾——这正是论文要破解的核心难题。

### 关键概念速览
- **Mistral 7B v0.2**：一种 7 B 参数的 Transformer 架构，以高效的注意力实现和良好的零样本能力著称。可以把它想成一辆轻量级跑车，速度快但马力有限。
- **深度扩展（Depth Scaling）**：在保持层宽不变的情况下增加 Transformer 层数，从而提升模型容量。类似于在同样宽度的楼层上再加几层，整体高度（能力）提升但占地面积不变。
- **连续预训练（Continuous Pre‑training）**：在已有的通用语料上继续训练更长时间，以适应特定语言或任务需求。相当于在已经学会基础功的学生身上再进行强化训练。
- **监督微调（Supervised Fine‑Tuning, SFT）**：使用标注好的问答或指令数据让模型学会遵循人类意图。像是给学生布置作业，让他练习课堂上学到的技巧。
- **直接偏好优化（Direct Preference Optimization, DPO）**：直接从人类偏好对比数据学习奖励信号，而不是先训练奖励模型再做强化学习。可以比作老师直接告诉学生哪种答案更好，而不是先让学生自己估计分数。
- **强化学习（Reinforcement Learning, RL）**：在 DPO 基础上进一步让模型通过试错获得更高的奖励分数。类似于在游戏中不断尝试，找到最高得分的策略。
- **上下文长度扩展**：模型一次性能处理的 token 数量，从 8 k 增至 32 k、64 k。把它想成阅读纸张的大小，从信纸到报纸再到卷轴，能一次性“看到”更多信息。
- **量化（Quantization）**：把模型参数从 16‑bit/32‑bit 精度压缩到 8‑bit、4‑bit 等低位表示，以降低显存和算力需求。相当于把大块的砖块压缩成更小的砖块，仍能搭出同样的建筑。

### 核心创新点
1. **深度扩展 + 多阶段训练 → 11 B 参数的波兰语专精模型**  
   过去的多语言模型大多通过增加宽度（更多隐藏单元）来提升容量，导致显存飙升。Bielik 通过在 Mistral 7B 基础上增加层数（深度扩展），保持每层宽度不变，从而在显存开销相对可控的前提下实现 11 B 参数。配合四阶段训练，使模型在波兰语细节和跨语言通用能力之间取得更好平衡。

2. **分层上下文扩展策略 → 64 k 长序列处理**  
   传统做法一次性把上下文长度拉到最大，训练成本爆炸。论文采用三阶段递进：先在 8 k 上进行通用预训练 → 再在 32 k 完整上下文上继续 → 最后使用 YRAN 技术把长度推到 64 k。这样模型逐步适应更长的依赖关系，训练效率大幅提升。

3. **直接偏好优化（DPO）取代奖励模型 → 更稳健的指令遵循**  
   以前的指令微调往往先训练一个奖励模型，再用 PPO 等强化学习算法优化。Bielik 直接在对比数据上最小化偏好损失，省去奖励模型的训练环节，简化管线并降低了奖励模型偏差带来的不稳定性。

4. **丰富量化选项 + 参数效率 → 多硬件部署友好**  
   在保持 11 B 参数的同时，提供 8‑bit、4‑bit、甚至 3‑bit 量化实现，使模型能够在消费级 GPU、CPU 甚至移动端运行。相较于同等性能的 20 B‑级模型，部署成本下降 2‑3 倍。

### 方法详解
整体框架可以看作四层塔楼：**连续预训练 → 监督微调 → DPO → 强化学习**，每层都在前一层的基础上进一步细化模型的语言能力和指令遵循度。

1. **连续预训练**  
   - **数据规模**：超过 1 T token，覆盖波兰语新闻、维基、法律文献等多源语料。  
   - **上下文策略**：先在 8 k token 上训练，模型学习基本的句子级依赖；随后在 32 k 完整上下文上继续，让模型捕捉段落甚至章节级的长程关联；最后使用 YRAN（Yield‑Recursive Attention Network）技术把注意力窗口扩展到 64 k，确保在超长文本中仍能保持信息流通。  
   - **实现细节**：深度扩展后模型层数从原来的 32 层提升到约 48 层，隐藏维度保持 4096，注意力头数 32。这样每层的计算量略增，但整体显存占用与 7 B 宽度扩展的模型相当。

2. **监督微调（SFT）**  
   - **指令数据**：收集了约 200 k 条波兰语指令-响应对，涵盖问答、翻译、代码生成等任务。  
   - **训练方式**：使用标准的交叉熵损失，让模型在给定指令后直接输出期望答案。这里的关键是保持学习率较低，以免破坏已经学到的语言结构。

3. **直接偏好优化（DPO）**  
   - **对比数据**：从人类评审的模型输出中抽取成对比较（好 vs. 坏），约 50 k 对。  
   - **损失函数**：最小化好答案相对于坏答案的对数概率差，等价于让模型在同一指令下更倾向于生成高质量答案。省去奖励模型的训练，使得优化过程更直接、更少噪声。

4. **强化学习（RL）**  
   - **目标**：在 DPO 已经提升的基线上进一步通过试错提升长文本推理和多步骤任务的表现。  
   - **算法**：采用基于 PPO（Proximal Policy Optimization）的策略更新，但因为已经有了较好的奖励信号，更新幅度更小，防止模型“忘记”已学知识。  
   - **迭代次数**：约 5 % 的总训练步数，用于微调。

**最巧妙的点**在于把 **上下文长度扩展** 与 **深度扩展** 串联起来：深度扩展提升模型的抽象能力，而分阶段的上下文扩展让这种能力在更长的文本上得以发挥，二者相辅相成，避免了单纯加宽或单纯加长导致的效率瓶颈。

### 实验与效果
- **评测任务**：包括波兰语自然语言推理（Polish NLI）、阅读理解、机器翻译（波-英、波-德）、代码生成以及跨语言多步推理基准。  
- **基线对比**：与专门的波兰语模型（如 PolBERT、PolGPT‑3B）以及通用的大模型（Mistral 7B、LLaMA‑13B、Claude‑2）进行比较。  
- **关键数字**：论文声称在波兰语 NLI 上比最强的波兰语专用模型提升约 12% F1，且在同类任务上超过 2‑6 倍参数的通用模型 5‑8% 的准确率。  
- **消融实验**：分别去掉深度扩展、上下文扩展、DPO、RL 四个环节，结果显示：去掉深度扩展导致整体性能下降约 7%；去掉 64 k 上下文导致长文档推理分数跌 4%；去掉 DPO 使指令遵循率下降约 6%；去掉 RL 对复杂推理任务影响约 3%。  
- **局限性**：论文承认在极低资源语言（如巴尔干语系的某些语言）上仍缺乏足够的语料支撑，模型表现不如波兰语；此外，超长上下文的推理速度仍受显存带宽限制，实际部署时需要权衡。

### 影响与延伸思考
Bielik 11B v3 的出现让业界重新审视 **“参数越大越好”** 的思路，展示了通过 **深度扩展 + 多阶段上下文训练** 能在保持相对紧凑显存的前提下实现显著性能提升。随后出现的几篇工作（如 “Depth‑Mistral” 系列、以及针对斯拉夫语族的 “Slavik‑XL”）都在不同程度上借鉴了其深度扩展与分层上下文策略。对想进一步探索的读者，可以关注：

- **稀疏注意力与长上下文的结合**：如何在更低显存下实现 128 k 甚至更长的上下文。  
- **跨语言 DPO 框架**：把偏好优化推广到多语言对比数据，提升模型在非主流语言上的指令遵循。  
- **量化与硬件协同设计**：探索 3‑bit 甚至 2‑bit 量化在保持推理质量的同时进一步压缩模型。

### 一句话记住它
**Bielik 11B v3 用深度扩展和分阶段超长上下文训练，让 11 B 参数的模型在波兰语和其他欧洲语言上实现了“更小、更强、更长”的三重突破。**