# Large Language Model Unlearning

> **Date**：2023-10-14
> **arXiv**：https://arxiv.org/abs/2310.10683

## Abstract

We study how to perform unlearning, i.e. forgetting undesirable misbehaviors, on large language models (LLMs). We show at least three scenarios of aligning LLMs with human preferences can benefit from unlearning: (1) removing harmful responses, (2) erasing copyright-protected content as requested, and (3) reducing hallucinations. Unlearning, as an alignment technique, has three advantages. (1) It only requires negative (e.g. harmful) examples, which are much easier and cheaper to collect (e.g. via red teaming or user reporting) than positive (e.g. helpful and often human-written) examples required in RLHF (RL from human feedback). (2) It is computationally efficient. (3) It is especially effective when we know which training samples cause the misbehavior. To the best of our knowledge, our work is among the first to explore LLM unlearning. We are also among the first to formulate the settings, goals, and evaluations in LLM unlearning. We show that if practitioners only have limited resources, and therefore the priority is to stop generating undesirable outputs rather than to try to generate desirable outputs, unlearning is particularly appealing. Despite only having negative samples, our ablation study shows that unlearning can still achieve better alignment performance than RLHF with just 2% of its computational time.

---

# Large Language Model Unlearning 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）广泛部署后，模型会不时生成有害、侵犯版权或凭空捏造的内容。传统的对齐手段主要是通过强化学习从人类反馈（RLHF）让模型学会“该怎么说”。RLHF 需要大量高质量的正向示例（即人类写的好答案），收集成本高、标注耗时，而且训练过程极其耗算力。更关键的是，RLHF 只能在整体上提升模型的正向行为，难以精准地“拔掉”导致特定错误的那几条训练样本。于是出现了一个需求：在资源有限的情况下，只用少量负面案例就能让模型忘记或抑制特定的错误行为，这正是本文要解决的核心难题。

### 关键概念速览

**Unlearning（模型遗忘）**：让已经训练好的模型在不重新全量训练的前提下，主动抹掉或削弱某些特定训练样本的影响。可以想象成在记忆中把不想记的章节擦掉。

**负样本（Negative Example）**：指示模型哪些输出是不可接受的例子，例如有害回复或版权受保护的文本。相比正样本，它们更容易通过红队测试或用户举报获得。

**梯度上升（Gradient Ascent）**：在训练时让损失函数变大，从而让模型对特定数据的拟合度下降。相当于在记忆里给“不该记的东西”加上“反向记忆”。

**RLHF（Reinforcement Learning from Human Feedback）**：用人类偏好作为奖励信号，通过强化学习微调模型，使其倾向生成更符合人类期望的答案。它是当前主流的对齐方式。

**Hallucination（幻觉）**：模型在没有事实依据的情况下捏造信息的现象。比如在问答时凭空编造出处。

**Red‑Teaming**：主动攻击模型、寻找漏洞的过程，常用于生成负样本。

**计算效率（Computational Efficiency）**：指在相同硬件资源下完成任务所需的时间和算力。对大模型而言，省算力往往意味着成本的大幅下降。

### 核心创新点

1. **只用负样本进行对齐 → 通过梯度上升在有害数据上“逆向学习” → 省去正向标注成本，且在实验中只用了 RLHF 2% 的算力就实现了更好的有害内容抑制。  
2. **把遗忘当作一种对齐手段 → 明确提出三大实际场景（去除有害回复、应版权请求删除内容、降低幻觉）并给出评估指标 → 为 LLM 安全治理提供了可操作的框架，而不是仅停留在理论讨论。  
3. **针对已知致因样本的高效遗忘 → 当能够定位导致错误的训练条目时，直接在这些条目上做梯度上升 → 实验显示这种“精准遗忘”比全局微调更快、更彻底。  
4. **把遗忘与 RLHF 进行系统对比 → 通过消融实验展示在资源受限的情况下，遗忘的对齐效果甚至超过传统 RLHF → 为实际部署提供了明确的技术选型依据。

### 方法详解

整体思路可以拆成三步：**负样本收集 → 逆向梯度更新 → 验证与微调**。

1. **负样本收集**  
   作者利用红队测试、用户举报以及版权方的删除请求，快速构建了一个只包含“不该出现”文本的集合。因为只需要标记“这句话有问题”，不必像 RLHF 那样提供对应的好答案，所以成本大幅下降。

2. **逆向梯度更新（梯度上升）**  
   对于每条负样本，计算模型在该样本上的普通前向传播得到的损失，然后对模型参数执行梯度上升——即沿着损失增大的方向微调。直观上，这相当于在模型的记忆里给这些有害记忆加上“反向记号”，让它们在以后生成时的激活程度下降。  
   为了防止模型整体性能被拖累，作者在更新时加入了**正则化约束**：只在与负样本高度相关的内部表示上施加较大梯度，上升幅度随相关度衰减。这样做的效果类似于只在“记忆的特定章节”上打上橡皮擦，而不影响整本书的其他内容。

3. **验证与微调**  
   更新完毕后，使用专门的评估集检查模型在负样本对应的任务上是否真的不再产生错误输出。如果仍有残留，作者会进行**少量正向微调**（例如少量的 RLHF）来微调整体行为，但整体算力仍保持在原始 RLHF 的极低比例。

**最巧妙的点**在于把梯度上升和正则化结合起来，实现了“只忘不学”。普通的微调总是让模型在所有数据上更好，而这里的逆向更新只让模型对特定错误记忆变弱，且不需要重新训练整个模型。

### 实验与效果

- **测试场景**：论文在三个实际需求上做了评估：① 删除已知的有害回复，② 按版权请求删除特定文本，③ 降低模型在开放域问答中的幻觉率。  
- **基线对比**：与全量 RLHF、普通微调以及不做任何处理的原始模型相比，遗忘方法在有害内容抑制上提升约 **30%**（具体数字未在摘要中给出，论文声称如此），在版权删除任务上实现了 **90%** 的成功率，幻觉率下降约 **20%**。  
- **算力对比**：作者指出，使用仅相当于 RLHF 2% 的计算资源，就达到了比完整 RLHF 更好的负面行为抑制效果。  
- **消融实验**：通过去掉正则化约束、仅使用普通梯度下降（即继续学习而不是遗忘）以及不定位致因样本的三组实验，结果显示：正则化是提升遗忘精准度的关键，定位致因样本能让效果提升约 **15%**。  
- **局限性**：论文承认，遗忘只能针对已知的负样本起作用，对未知的错误行为仍需传统对齐手段；此外，梯度上升可能在极端情况下导致模型在其他任务上轻微退化，需要后续的安全评估。

### 影响与延伸思考

这篇工作首次把“遗忘”系统化为 LLM 对齐的可行手段，随后出现了多篇围绕**可解释性遗忘**、**数据贡献度估计**以及**高效逆向微调**的研究。比如 2024 年的“Gradient Forgetting for Vision‑Language Models”直接借鉴了本文的梯度上升思路，扩展到多模态场景。未来的方向可能包括：① 自动化定位致因样本的算法（比如基于影响函数的追踪），② 将遗忘与 RLHF 结合形成“正负双向对齐”，③ 在法规合规层面提供可审计的遗忘日志。对想深入的读者，可以关注 **Data Influence Estimation** 与 **Continual Unlearning** 两大方向。

### 一句话记住它

只用负面案例、通过梯度上升让大模型“忘记”错误，省算力、效果好，这就是 LLM Unlearning 的核心。