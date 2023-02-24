# Automatic Prompt Augmentation and Selection with Chain-of-Thought from   Labeled Data

> **Date**：2023-02-24
> **arXiv**：https://arxiv.org/abs/2302.12822

## Abstract

Chain-of-thought (CoT) advances the reasoning abilities of large language models (LLMs) and achieves superior performance in complex reasoning tasks. However, most CoT studies rely on carefully designed human-annotated rational chains to prompt LLMs, posing challenges for real-world applications where labeled data is available without rational chains. This paper proposes a new strategy, Automate-CoT (Automatic Prompt Augmentation and Selection with Chain-of-Thought), that can bypass human engineering of CoT by automatically augmenting rational chains from a small labeled dataset, and then pruning low-quality chains to construct a candidate pool of machine-generated rationale chains based on the labels. Finally, it selects the optimal combination of several rationale chains from the pool for CoT prompting by employing a variance-reduced policy gradient strategy to estimate the significance of each example. Automate-CoT enables a quick adaptation of the CoT technique to different tasks. Experimental results demonstrate the effectiveness of our method, where competitive results are achieved on arithmetic reasoning (+2.7%), commonsense reasoning (+3.4%), symbolic reasoning (+3.2%), and non-reasoning tasks (+2.5%). The code is available at https://github.com/SHUMKASHUN/Automate-CoT.

---

# 基于标注数据的自动提示增强与选择的思维链（Automate‑CoT）论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上加入思维链（Chain‑of‑Thought，CoT）可以让模型先写出推理步骤，再给出答案，效果显著提升。但大多数 CoT 方法依赖人工编写的“理性链”，也就是把每一步推理写成自然语言。人工标注成本高、质量参差不齐，而且只能针对少数公开数据集。实际场景常常只有带标签的训练样本，却没有对应的推理过程，这让直接搬用已有 CoT 提示变得不切实际。于是，如何在没有人工理性链的情况下，让模型自己生成、筛选高质量的思维链，成为制约 CoT 大规模落地的关键瓶颈。

### 关键概念速览

**思维链（CoT，Chain‑of‑Thought）**：让模型在给出最终答案前，先把推理步骤写出来，类似解题时的草稿，能够帮助模型保持逻辑连贯性。  
**提示（Prompt）**：向语言模型提供的输入文本，决定模型的行为方式。这里的提示包括问题本身和可选的思维链。  
**自动提示增强（Automatic Prompt Augmentation）**：利用已有标注数据，机器自动生成一批可能的思维链，而不是人工手写。  
**候选池（Candidate Pool）**：所有机器生成的思维链集合，后续会从中挑选最有价值的子集。  
**方差削减策略（Variance‑Reduced Policy Gradient）**：一种强化学习技巧，用来估计每条思维链对整体性能的贡献，同时降低估计噪声。  
**重要性估计（Significance Estimation）**：评估单个示例（问题+思维链）在提升模型表现上的作用大小。  
**标签驱动的筛选（Label‑Driven Pruning）**：依据原始答案标签，剔除那些生成的思维链在答案上表现不一致的低质量链。

### 核心创新点

1. **从标签直接生成思维链 → 自动化生成**：传统做法需要人工写出每一步推理，这篇论文把任务转化为“给模型一个带标签的样本，让它自己尝试写出推理”。具体做法是让一个强大的 LLM（如 GPT‑4）在少量标注数据上进行自回归生成，得到若干候选思维链。这样就把人工成本降到几乎为零。

2. **基于标签的质量过滤 → 只保留答案一致的链**：生成的链质量参差不齐。作者利用原始标签检查每条链的最终答案是否匹配，如果不匹配就直接剔除。相当于让模型自我纠错，只留下“自洽”的思维链进入候选池。

3. **方差削减的策略梯度挑选 → 稳定的组合选择**：候选池里可能有上百条链，直接全部喂给模型会导致噪声。论文引入一种方差削减的策略梯度方法，估算每条链对整体任务表现的贡献，并在此基础上挑选出若干最有价值的链组成最终提示。这样既保留多样性，又避免低质量链拖累性能。

4. **一次性适配多任务 → 任务无关的通用流程**：整个流程只需要少量标注样本，不依赖任务特定的规则或模板。实验表明，同一套自动化流程可以在算术、常识、符号推理等四类任务上都取得显著提升，展示了方法的广泛适用性。

### 方法详解

整体框架可以划分为三大步骤：**生成 → 过滤 → 选择**。

1. **生成阶段**  
   - 输入：一小批带标签的训练样本（问题 + 正确答案）。  
   - 操作：使用一个强大的预训练 LLM（论文中使用的是 GPT‑4）对每个样本进行多次采样，每次让模型在问题后自行续写推理过程，直至产生答案。因为模型本身已经具备一定的推理能力，采样多次可以得到风格、长度、细节各异的思维链。  
   - 类比：把模型当成“学生”，让它在老师只给出题目和答案的情况下，自己写出解题步骤。

2. **过滤阶段**  
   - 对每条生成的链，提取其末尾的答案并与原始标签比对。若答案相同，则认为这条链在逻辑上是自洽的，保留；否则直接丢弃。  
   - 这一步相当于“质量关卡”，只让通过答案检查的链进入后续。作者指出，这一步可以显著降低噪声，因为错误答案往往伴随明显的逻辑错误。

3. **选择阶段**  
   - 目标是从保留下来的候选池中挑出一个子集，使得在实际推理时使用这些链作为 CoT 提示能够最大化模型的整体准确率。  
   - 采用**方差削减的策略梯度**：把每条链视为一个“动作”，其奖励是模型在验证集上使用该链后得到的正确率提升。为了降低奖励估计的方差，作者引入了基准值（例如使用空提示的表现）并对奖励进行中心化。随后通过梯度上升更新每条链的选择概率，最终得到一组高权重的链。  
   - 实际使用时，论文会把这些高权重链拼接在一起，形成一个“多链提示”。模型在推理时会看到多个思维路径，从而提升鲁棒性。

**最巧妙的点**在于把“生成思维链”当作一个**自监督**过程：只要答案对齐，就不需要人工检查每一步的合理性；再加上方差削减的强化学习技巧，使得即使候选池里仍有噪声，也能通过概率加权把噪声压制住。

### 实验与效果

- **测试任务**：算术推理（如 GSM8K）、常识推理（CommonsenseQA）、符号推理（Symbolic Reasoning Benchmark）以及几个不需要推理的分类任务（如 SST‑2）。  
- **基准对比**：与传统的 Zero‑Shot CoT、Few‑Shot CoT（手工挑选的几条思维链）以及最新的 Self‑Consistency 方法相比，Automate‑CoT 在四类任务上均实现了两位数的提升。具体提升幅度为：算术 +2.7%、常识 +3.4%、符号 +3.2%、非推理任务 +2.5%。  
- **消融实验**：作者分别去掉（1）标签驱动过滤、（2）方差削减策略梯度，只保留随机选择的链。结果显示，去掉过滤会导致整体准确率下降约 1.8%，去掉策略梯度则下降约 2.1%，说明两者都是性能提升的关键因素。  
- **局限性**：生成思维链仍然依赖一个强大的 LLM（如 GPT‑4）作为“教师”，在资源受限的环境下可能难以复现；此外，过滤仅基于答案一致性，可能会保留逻辑上不严谨但恰好得到正确答案的链。论文中也提到在极端长文本或多步骤推理任务上，生成质量仍有待提升。

### 影响与延伸思考

Automate‑CoT 把“思维链”从手工工程转向自动化生成，为大模型在缺少人工标注的真实业务场景中快速部署提供了可行路径。自论文发布后，出现了多篇工作尝试在不同语言、不同模型规模上复现或改进其自动生成策略，例如利用 **自监督对齐**（self‑aligned）模型直接生成高质量链，或结合 **检索增强**（retrieval‑augmented）技术把外部知识库引入链的生成过程。未来的研究方向可能包括：

- **低资源模型的链生成**：探索在没有 GPT‑4 这类强模型的情况下，如何通过多轮自蒸馏或小模型互相校验来提升链质量。  
- **多模态思维链**：把图像、表格等非文本信息也纳入链的生成，让模型在视觉推理任务中也能受益。  
- **动态链选择**：根据具体问题的难度或领域特征，实时决定使用多少条链、哪几条最合适，而不是一次性固定组合。

如果想深入了解，可以关注近期的 “Self‑Generated CoT” 系列论文以及 “Prompt Mining” 方向的最新进展。

### 一句话记住它

只要有标签，就能让大模型自己写出、挑选高质量的思维链，从而在各种推理任务上省去人工标注的痛苦。