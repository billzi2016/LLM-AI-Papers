# CoT-Valve: Length-Compressible Chain-of-Thought Tuning

> **Date**：2025-02-13
> **arXiv**：https://arxiv.org/abs/2502.09601

## Abstract

Chain-of-Thought significantly enhances a model's reasoning capability, but it also comes with a considerable increase in inference costs due to long chains. With the observation that the reasoning path can be easily compressed under easy tasks but struggle on hard tasks, we explore the feasibility of elastically controlling the length of reasoning paths with only one model, thereby reducing the inference overhead of reasoning models dynamically based on task difficulty. We introduce a new tuning and inference strategy named CoT-Valve, designed to allow models to generate reasoning chains of varying lengths. To achieve this, we propose to identify a direction in the parameter space that, when manipulated, can effectively control the length of generated CoT. Moreover, we show that this property is valuable for compressing the reasoning chain. We construct datasets with chains from long to short for the same questions and explore two enhanced strategies for CoT-Valve: (1) a precise length-compressible CoT tuning method, and (2) a progressive chain length compression approach. Our experiments show that CoT-Valve successfully enables controllability and compressibility of the chain and shows better performance than the prompt-based control. We applied this method to QwQ-32B-Preview, reducing reasoning chains on GSM8K from 741 to 225 tokens with a minor performance drop (95.07% to 94.92%) and on AIME from 6827 to 4629 tokens, with only one additional incorrect answer.

---

# CoT‑Valve：可压缩长度的思维链调优 论文详细解读

### 背景：这个问题为什么难？

在大模型里加入 **Chain‑of‑Thought（思维链）** 能显著提升复杂推理的准确率，但每一步都要生成文字，导致推理链往往很长，推理时的计算和显存开销随之暴涨。过去的做法只能在 **提示词** 里硬性规定要不要写思维链，或者直接让模型自行决定长度，缺乏对推理过程“长短”的细粒度控制。于是出现了两个矛盾：① 需要足够长的链来解决难题，② 但在大多数简单题目上，冗长的链是浪费。如何让同一个模型根据题目难度自动调节思维链的长度，成为了一个未被解决的瓶颈。

### 关键概念速览
- **Chain‑of‑Thought（思维链）**：模型在给出最终答案前，先把推理步骤写出来，类似人做数学题时的草稿。  
- **推理链长度**：思维链中生成的 token 数量，越长说明模型在“思考”更多步骤。  
- **参数空间方向**：在模型的参数空间里选取一条特定的向量（方向），沿着这条向量微调或加权会产生可预测的行为变化。可以把它想象成调音台上的一个旋钮，顺时针转会让模型说得更啰嗦，逆时针转则更简洁。  
- **LoRA 微调**：Low‑Rank Adaptation 的缩写，只在模型的某些层加入低秩矩阵进行轻量微调，既不破坏原模型，又能快速学习新功能。这里把 LoRA 当作在参数空间里插入“可调节的方向”。  
- **可压缩性（Compressibility）**：指在保持答案正确率的前提下，能够把原本冗长的思维链压缩成更短的版本。  
- **Progressive Chain Compression（渐进式链压缩）**：一种训练策略，先让模型学会完整的长链，再逐步教它在同样的输入下输出更短的链，类似先让学生写完整解答，再训练他用更简洁的语言复述。  
- **Prompt‑based Control（提示词控制）**：通过在输入前加特定文字（如 “简要思考：”）来诱导模型生成不同长度的思维链，这是之前最常见的控制手段。  

### 核心创新点
1. **从提示词到参数控制的转变**  
   - 之前：只能在提示词里写 “思考一步一步” 或 “直接回答”，控制力度有限且对不同任务的适配度低。  
   - 本文：在模型参数空间里找到一条专门调节思维链长度的方向，并通过 LoRA 学习这条方向。  
   - 改变：长度不再依赖文字提示，而是通过调节 LoRA 权重的标量系数即可实现，控制更细致、可在推理时动态调节。

2. **精确的长度可压缩调优方法**  
   - 之前：没有办法让模型在同一道题上生成“长链”和“短链”两种版本，只能分别训练两个模型。  
   - 本文：在同一模型上同时学习长链和对应的短链，通过对比学习让 LoRA 参数形成“长度映射”。  
   - 改变：只需一次微调，就能让模型在推理时通过一个标量切换长短，省去多模型部署的成本。

3. **渐进式链长度压缩策略**  
   - 之前：直接让模型一次性学习压缩往往导致答案质量下降，因为模型缺少对完整推理过程的记忆。  
   - 本文：先让模型熟练生成完整的长链，再逐步引入更短的目标链，形成“先大后小”的学习路径。  
   - 改变：压缩过程更平滑，最终得到的短链在保持答案正确率的同时，显著削减 token 数。

4. **实证证明参数方向真的能压缩链**  
   - 之前：理论上可以调节参数，但缺乏大规模实验验证。  
   - 本文：在 GSM8K、AIME 等数学推理基准上，展示了同一模型在不同长度标量下的推理效果，证明了可压缩性。  
   - 改变：为后续研究提供了可复制的实验范式，打开了“可调推理成本”的新方向。

### 方法详解
**整体框架**  
CoT‑Valve 的核心是：① 用 LoRA 在原模型上学习一条“长度方向”；② 在推理时通过调节该 LoRA 的标量系数 λ，控制生成的思维链长短。整个流程分为两阶段：**长度可压缩调优**（训练阶段）和 **动态长度控制**（推理阶段），并提供两种训练细节：精确调优和渐进式压缩。

**1️⃣ 找到长度方向的 LoRA**  
- 选取模型的若干关键层（通常是自注意力层的投影矩阵），插入低秩矩阵 A·Bᵀ（LoRA）。  
- 训练目标是让模型在同一道题上分别输出 **长链**（L）和 **短链**（S），并最小化两者的交叉熵损失，同时加入一个 **长度对齐损失**：让 LoRA 参数的变化与链长差值呈线性关系。  
- 训练结束后，LoRA 参数本身就编码了一条“往长往短”的方向。

**2️⃣ 精确长度可压缩调优**  
- 在同一批次里，给每个样本随机采样一个目标长度比例 α∈[0,1]（0 表示最短，1 表示最长）。  
- 通过在前向传播时把 LoRA 权重乘以 α，模型直接生成对应长度的链。  
- 损失函数同时考虑答案正确性和 **长度误差**（生成的 token 数与目标长度的差），确保模型学会在不同 α 下产生合适的链。

**3️⃣ 渐进式链长度压缩**  
- **阶段 1**：只训练长链（α≈1），让模型熟悉完整推理步骤。  
- **阶段 2**：逐步降低 α 的上界，例如从 1→0.8→0.6…，每一步都继续微调 LoRA。  
- 这种“先大后小”的 curriculum 学习让模型在压缩链的过程中保留关键推理信息，避免一次性压缩导致信息丢失。

**4️⃣ 推理时的动态控制**  
- 给定一道新题，先用一个轻量的 **难度估计器**（可以是模型自身的置信度或外部分类器）判断任务难度。  
- 根据难度映射到 λ（或 α），例如：简单题 λ=0.3 → 生成短链；难题 λ=0.9 → 生成长链。  
- 在实际生成时，只需把 LoRA 权重乘以 λ，然后正常解码，模型会自动输出对应长度的思维链。

**最巧妙的点**  
- **单模型多长度**：不需要为每种长度训练独立模型，只通过一个标量即可切换，极大降低部署和维护成本。  
- **参数方向的线性可解释性**：作者发现 LoRA 权重的放大/缩小与链长呈近似线性关系，这在高维参数空间里非常罕见，提供了直观的“调节旋钮”。  

### 实验与效果
- **数据集**：GSM8K（中等难度数学文字题）和 AIME（高难度美国数学竞赛题），以及作者自行构造的同一道题对应的长短链对齐数据。  
- **基线**：传统的 **Prompt‑based Control**（在提示词里加 “简要思考” 或 “详细思考”），以及未做任何长度控制的原始模型。  
- **主要结果**：  
  - 在 GSM8K 上，使用 CoT‑Valve 将平均思维链长度从 741 token 压缩到 225 token，答案正确率仅从 95.07% 下降到 94.92%（下降 0.15%）。  
  - 在 AIME 上，链长从 6827 token 降至 4629 token，仅多出 1 例错误答案，整体得分几乎不变。  
  - 与 Prompt‑based Control 相比，CoT‑Valve 在相同长度下的准确率提升约 0.4%~0.7%。  
- **消融实验**：  
  - 去掉 LoRA（直接在原模型上调节 λ）几乎不产生长度变化，验证 LoRA 是关键。  
  - 只使用精确调优不做渐进式压缩时，短链的答案正确率下降约 0.6%，说明渐进式策略对保持质量有显著帮助。  
- **局限性**：  
  - 论文未在大规模多模态或非数学任务上验证，长度控制的通用性仍待探索。  
  - 需要一个可靠的难度估计器来决定 λ，若估计错误可能导致过度压缩或不必要的冗长。  

### 影响与延伸思考
CoT‑Valve 为 **可调推理成本** 提供了实用的实现路径，打开了“在同一模型上按需调节推理深度”的新局面。后续工作已经开始尝试把这种参数方向控制扩展到 **多模态**（如图文推理）和 **检索增强** 场景，甚至把长度方向与 **答案置信度** 结合，形成自适应的 “思考-回答” 循环。对想进一步研究的读者，可以关注以下方向：  
- **自动难度估计**：如何在不额外模型的情况下精准预测题目难度，以决定 λ。  
- **多任务共享方向**：是否可以在同一个 LoRA 中同时控制不同任务的推理深度（如数学、常识、代码）。  
- **理论解释**：为何在高维参数空间里会出现如此线性的长度方向，是否与模型内部的 “思考模块” 结构有关。  

### 一句话记住它
CoT‑Valve 让大模型只动一个旋钮，就能在长链和短链之间自由切换，实现“按难度付费”式的思维链压缩。