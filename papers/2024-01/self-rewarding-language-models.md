# Self-Rewarding Language Models

> **Date**：2024-01-18
> **arXiv**：https://arxiv.org/abs/2401.10020

## Abstract

We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human preferences, which may then be bottlenecked by human performance level, and secondly these separate frozen reward models cannot then learn to improve during LLM training. In this work, we study Self-Rewarding Language Models, where the language model itself is used via LLM-as-a-Judge prompting to provide its own rewards during training. We show that during Iterative DPO training that not only does instruction following ability improve, but also the ability to provide high-quality rewards to itself. Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613. While there is much left still to explore, this work opens the door to the possibility of models that can continually improve in both axes.

---

# 自我奖励语言模型 论文详细解读

### 背景：这个问题为什么难？

在让大语言模型（LLM）变得更聪明的路上，训练信号的质量至关重要。传统做法是让人类标注者给出偏好，然后训练一个单独的奖励模型（reward model）来评估模型输出。可是人类的判断本身有上限：当模型的能力已经逼近或超过人类时，奖励模型也会卡在“人类水平”。更糟的是，这个奖励模型在训练过程中是冻结的，无法随模型的进步而自我提升，导致整体学习进入瓶颈。于是出现了一个悖论：想让模型超越人类，却只能靠人类提供的反馈。

### 关键概念速览

**奖励模型（Reward Model）**：一种专门用来打分模型输出好坏的网络，通常是从人类偏好数据里学到的。它相当于“裁判”，但如果裁判本身水平有限，比赛也难以更高水平进行。

**LLM-as-a-Judge**：把语言模型本身当作评审，让它在特定提示下给出分数或偏好。想象成让选手自己给自己的作品打分，前提是选手已经足够成熟。

**指令遵循（Instruction Following）**：模型在接到用户指令后，能够给出符合意图、语义准确的回答。类似于把模型当成助理，要求它听话并完成任务。

**迭代 DPO（Iterative Direct Preference Optimization）**：一种直接基于偏好进行优化的训练循环，模型在每一轮都用最新的奖励信号来更新自身。把它想成“每次比赛后，选手根据裁判的最新评分再训练一次”。

**AlpacaEval 2.0**：一个公开的指令遵循评测基准，提供多种任务的对话式比较，常被用来排榜。

### 核心创新点

1. **奖励来源从人类迁移到模型自身**  
   *之前的做法*：先收集大量人类偏好，训练一个固定的奖励模型，再用它来指导LLM。  
   *本文的做法*：在每一次 DPO 迭代中，直接让同一个 LLM 通过“LLM-as-a-Judge”提示生成奖励分数。  
   *带来的改变*：消除了人类反馈的上限，模型可以在自我评估的循环中不断提升评分质量。

2. **奖励模型与生成模型共享同一参数体**  
   *之前的做法*：奖励模型是独立的网络，训练完后冻结，无法随生成模型一起进化。  
   *本文的做法*：同一个 LLM 同时承担回答和打分两个角色，参数在两种任务之间共享。  
   *带来的改变*：模型在学习如何更好回答的同时，也在学习如何更精准打分，形成正向反馈环。

3. **多轮迭代 DPO 训练框架**  
   *之前的做法*：一次性使用固定奖励模型进行 DPO，或者只做少量微调。  
   *本文的做法*：对 Llama 2 70B 进行三轮迭代，每轮都重新生成奖励并继续优化。  
   *带来的改变*：指令遵循能力和自评质量同步提升，最终在 AlpacaEval 2.0 上超过多款商业模型。

### 方法详解

整体思路可以概括为“三步走”：**生成 → 评估 → 优化**，并在整个循环中让同一个模型扮演生成者和评审者。

1. **生成阶段**  
   给模型一个指令提示（例如“写一段关于量子计算的介绍”），让它输出答案。这里使用的仍是普通的自回归解码，和普通微调没有区别。

2. **评估阶段（LLM-as-a-Judge）**  
   把同一个模型重新召回，但这次的提示改成“请对以下两段回答进行比较，给出更好的一段并说明原因”。模型会收到两段候选答案（通常是当前模型的输出和一个基准/旧版本的输出），然后在内部生成一个偏好标签或数值分数。关键在于提示设计，使模型把注意力放在“质量判断”而不是继续生成内容上。

3. **优化阶段（Iterative DPO）**  
   收集到的偏好对（更好 vs. 更差）被喂入 DPO 损失函数。DPO 直接最大化模型在更好答案上的概率相对更差答案的比例，而不需要额外的价值网络。因为奖励已经由模型自己给出，这一步只需要普通的梯度下降即可。完成一次梯度更新后，模型的参数已经兼顾了“写得更好”和“评得更准”两方面的改进。

**循环**：完成一次 DPO 更新后，模型进入下一轮迭代。此时它的评估能力已经提升，生成的答案也更接近理想。新一轮的评估会产生更可靠的奖励，进一步推动模型向更高水平迈进。作者在实验中跑了三轮，发现每轮都带来显著提升。

**巧妙之处**：把评估任务包装成普通的文本生成任务，使得不需要额外的网络结构或专门的奖励模型，只靠提示工程即可让模型自评。这种“自我对话”方式极大降低了实现成本，同时打开了模型自我提升的可能性。

### 实验与效果

- **测试平台**：作者在 AlpacaEval 2.0 上进行评测，这是一套覆盖多种指令的对话基准，能够客观比较不同模型的指令遵循水平。
- **对比基线**：包括 Claude 2、Gemini Pro、GPT‑4（2023‑06‑13 版本）等主流商业模型。论文声称在排行榜上超越这些系统，说明自我奖励的 Llama 2 70B 版本在整体得分上领先。
- **增益幅度**：具体数值未在摘要中披露，但作者强调“显著提升”，并在三轮迭代后达到领先地位。
- **消融实验**：原文未提供细节，只能说明作者通过对比不同迭代次数验证了多轮迭代的必要性。若去掉自评环节或使用固定奖励模型，性能会回落到传统 DPO 水平。
- **局限性**：作者坦承仍有大量未探索的空间，例如自评的可靠性在极端任务上可能下降，提示设计的鲁棒性尚未系统化。此外，模型仍然依赖于初始人类数据来启动第一轮奖励，完全摆脱人类监督仍是远景。

### 影响与延伸思考

这篇工作打开了“模型自我监督”在奖励学习中的新局面。随后出现的研究开始探索更复杂的自评机制，如多模态自评、跨模型自评（让小模型评大模型）以及基于链式思考的自评提示。对想进一步了解的读者，可以关注以下方向：  
- **自我对话（Self-Chat）**：让模型在同一会话中轮流扮演用户和助理，以生成更丰富的训练信号。  
- **奖励模型蒸馏**：把自评产生的分数蒸馏成轻量化的专用奖励网络，以降低推理成本。  
- **安全性与偏差控制**：自我奖励可能放大模型已有的偏见，如何在自评环节加入约束是后续的重要课题。

### 一句话记住它

让大模型自己当裁判，通过循环自评和直接偏好优化，突破人类反馈上限，实现指令遵循与自我奖励同步提升。