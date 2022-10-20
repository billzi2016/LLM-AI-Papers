# Large Language Models Can Self-Improve

> **Date**：2022-10-20
> **arXiv**：https://arxiv.org/abs/2210.11610

## Abstract

Large Language Models (LLMs) have achieved excellent performances in various tasks. However, fine-tuning an LLM requires extensive supervision. Human, on the other hand, may improve their reasoning abilities by self-thinking without external inputs. In this work, we demonstrate that an LLM is also capable of self-improving with only unlabeled datasets. We use a pre-trained LLM to generate "high-confidence" rationale-augmented answers for unlabeled questions using Chain-of-Thought prompting and self-consistency, and fine-tune the LLM using those self-generated solutions as target outputs. We show that our approach improves the general reasoning ability of a 540B-parameter LLM (74.4%->82.1% on GSM8K, 78.2%->83.0% on DROP, 90.0%->94.4% on OpenBookQA, and 63.4%->67.9% on ANLI-A3) and achieves state-of-the-art-level performance, without any ground truth label. We conduct ablation studies and show that fine-tuning on reasoning is critical for self-improvement.

---

# 大语言模型可以自我提升 论文详细解读

### 背景：这个问题为什么难？

在过去，想让大语言模型（LLM）在推理类任务上更强，通常需要大量标注好的答案来微调模型。标注成本高、质量参差不齐，而且很多真实场景根本没有现成的标签。人类却可以通过自己思考、写下推理过程来提升解题能力，这种“自我练习”在机器上几乎没有实现。于是，如何在没有任何人工标签的情况下，让模型自己产生高质量的训练信号，成为了一个急需突破的难题。

### 关键概念速览
- **Chain-of-Thought（思维链）**：让模型在给出最终答案前先把推理步骤写出来，类似于人做数学题时先列草稿，再写答案。这样可以让模型的思考过程可见、可纠错。
- **Self‑Consistency（自洽性）**：对同一道题多次采样答案，取出现频率最高的答案作为最终结果。相当于让模型“投票”，降低一次采样的偶然错误。
- **高置信度答案**：模型在多次采样后产生的、出现频率最高且推理链最完整的答案，作者把它当作“可信的自标签”。
- **无监督微调**：使用模型自己生成的答案作为训练目标进行微调，而不依赖任何人工标注的数据。
- **推理微调**：微调的目标不是单纯的答案，而是包括完整推理链的文本，使模型学会更系统的思考方式。

### 核心创新点
1. **自生成高置信度推理 → 直接用于微调**  
   过去的自监督方法往往只利用模型的输出概率或隐藏状态，而这篇工作直接把模型自己写的完整推理链当作标签。这样做把“思考过程”本身变成了监督信号，显著提升了微调的质量。

2. **Chain-of-Thought + Self‑Consistency 双重过滤**  
   仅用一次 CoT 生成的答案往往噪声较大。作者先让模型多次生成答案（Self‑Consistency），再挑选出现频率最高的那条推理链作为高置信度样本。这个两步筛选相当于先让模型“思考”，再让它“自我审查”，大幅降低错误标签的比例。

3. **在 540B 参数模型上实现大幅提升**  
   之前的自监督微调大多在几十亿参数的模型上实验，提升有限。这篇论文在一个 5400 亿参数的超大模型上进行实验，验证了方法在规模化模型上的可行性，并实现了 GSM8K 从 74.4% 提升到 82.1% 的跨越。

4. **细致的消融实验证明“推理微调”是关键**  
   作者分别去掉 CoT、Self‑Consistency、只微调答案等设置，发现只有同时保留完整推理链的微调才能带来显著提升，说明模型真正学会了更好的推理，而不是单纯记忆答案。

### 方法详解
**整体思路**可以概括为三步：  
1) 用预训练好的 LLM 对未标注的问题集合进行多次 CoT 推理；  
2) 通过 Self‑Consistency 选出出现频率最高的推理链，视为高置信度标签；  
3) 用这些自生成的“问题‑推理‑答案”三元组对同一个 LLM 进行有监督微调。

**步骤拆解**  
- **数据准备**：收集大量无标签的问答数据（如数学题、阅读理解等），只要求有问题文本，不需要答案。  
- **多次 CoT 生成**：对每个问题，使用 Chain-of-Thought 提示让模型输出完整的思考过程和最终答案。这里的提示大致是：“请一步一步思考并给出答案”。模型会一次性返回类似“第一步…第二步…答案是…”。  
- **Self‑Consistency 过滤**：对同一道题重复上述生成过程 N 次（论文中 N≈10），得到 N 条推理链。统计每条完整文本出现的次数，选取出现次数最多的那一条。因为同一条推理链在多次采样中重复出现，说明模型对这条路径的置信度更高。  
- **构造训练样本**：把问题文本、选中的推理链以及答案拼接成一个完整的输入‑输出对，作为微调的训练样本。这里的输出不仅是答案，而是包括所有推理步骤的完整文本。  
- **微调过程**：使用标准的自回归语言模型微调流程（如 AdamW 优化器、学习率调度），在上述自标签数据上继续训练模型。因为标签来源于同一个模型，训练过程相当于让模型“自我纠错”，逐步强化正确的推理模式。

**最巧妙的地方**在于把模型的“思考过程”直接当作监督信号，而不是仅仅依赖最终答案。这样做的好处是：  
- 推理链本身携带了解题思路，模型可以学习到通用的推理模板；  
- 当出现错误时，错误的推理链往往在多次采样中不具备自洽性，容易被过滤掉；  
- 微调时模型会被迫对齐自己的内部推理路径，提升整体一致性。

### 实验与效果
- **测试任务**：GSM8K（小学数学）、DROP（阅读理解数值推理）、OpenBookQA（常识问答）以及 ANLI‑A3（自然语言推理）。这些任务都需要模型进行多步推理，能够真实检验“思考能力”。  
- **主要结果**：在 540B 参数模型上，使用自监督微调后，GSM8K 正确率从 74.4% 提升到 82.1%；DROP 从 78.2% 提升到 83.0%；OpenBookQA 从 90.0% 提升到 94.4%；ANLI‑A3 从 63.4% 提升到 67.9%。这些提升在没有任何人工标签的前提下，已经接近或超过了当时的 SOTA（有监督）水平。  
- **Baseline 对比**：与直接在同样无标签数据上进行普通语言模型微调（不使用 CoT 或 Self‑Consistency）相比，后者的提升幅度只有 1% 左右，说明推理链和自洽过滤是关键。  
- **消融实验**：去掉 CoT（只生成答案）或去掉 Self‑Consistency（只取一次采样）都会导致性能回落到原始模型水平，进一步验证了两者的必要性。  
- **局限性**：方法依赖于模型本身已有一定的推理能力；如果基模型在某类任务上几乎没有正确的推理链，过滤后仍可能得到错误标签，导致微调效果有限。作者也提到，生成过程的计算成本较高（需要多次采样），在资源受限的环境下不易直接推广。

### 影响与延伸思考
这篇工作打开了“模型自我教练”的新思路，随后出现了多篇围绕自监督推理、自动数据标注的研究。例如，利用 LLM 生成对抗样本进行自我对齐、在多语言环境下做自我翻译标注等，都可以视为对该思路的延伸。未来的方向可能包括：  
- **更高效的自洽采样**：设计更轻量的投票机制，降低多次生成的计算开销；  
- **跨任务自监督**：把一种任务的自生成推理链迁移到另一种任务，探索通用推理能力的共享；  
- **与人类反馈结合**：在自监督的基础上加入少量人类审校，形成“少量标注+大量自标签”的混合训练框架。想深入了解的读者可以关注 2024‑2025 年的 “Self‑Improving LLM” 系列论文，以及 OpenAI、DeepMind 在自监督微调方面的最新报告。

### 一句话记住它
让大语言模型自己写出高置信度的推理链，再用这些自生成的答案进行微调，就能在没有任何人工标签的情况下显著提升推理能力。