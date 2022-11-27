# Self-Destructing Models: Increasing the Costs of Harmful Dual Uses of   Foundation Models

> **Date**：2022-11-27
> **arXiv**：https://arxiv.org/abs/2211.14946

## Abstract

A growing ecosystem of large, open-source foundation models has reduced the labeled data and technical expertise necessary to apply machine learning to many new problems. Yet foundation models pose a clear dual-use risk, indiscriminately reducing the costs of building both harmful and beneficial machine learning systems. Policy tools such as restricted model access and export controls are the primary methods currently used to mitigate such dual-use risks. In this work, we review potential safe-release strategies and argue that both policymakers and AI researchers would benefit from fundamentally new technologies enabling more precise control over the downstream usage of open-source foundation models. We propose one such approach: the task blocking paradigm, in which foundation models are trained with an additional mechanism to impede adaptation to harmful tasks without sacrificing performance on desirable tasks. We call the resulting models self-destructing models, inspired by mechanisms that prevent adversaries from using tools for harmful purposes. We present an algorithm for training self-destructing models leveraging techniques from meta-learning and adversarial learning, which we call meta-learned adversarial censoring (MLAC). In a small-scale experiment, we show MLAC can largely prevent a BERT-style model from being re-purposed to perform gender identification without harming the model's ability to perform profession classification.

---

# 自毁模型：提升基础模型有害双重用途成本 论文详细解读

### 背景：这个问题为什么难？

大规模开源的基础模型（如BERT、GPT）让几乎任何人都能把少量标注数据和普通算力套进模型，快速得到可用的 AI 应用。正因为门槛低，这些模型既能被用来做医学诊断、教育辅导等正向任务，也能被改造用于生成仇恨言论、身份推断等有害目的。传统的风险控制手段主要是“限制访问”或“出口管制”，但一旦模型代码和权重公开，这些手段失效。现有的技术方案大多是在模型发布后再加水印、检测或审计，无法从根本上阻止恶意再训练。因此，如何在保持模型有用性的前提下，主动让模型在特定有害任务上难以适应，成为了一个迫切且技术上极具挑战的问题。

### 关键概念速览

**基础模型（Foundation Model）**：指在大规模通用数据上预训练、能够迁移到多种下游任务的模型。类似于“通用工具箱”，用户只需要少量专用数据就能完成特定工作。

**双重用途风险（Dual‑Use Risk）**：同一技术既能产生正面价值，也可能被滥用于危害社会。这里指的是模型既能帮助诊断疾病，也能被改造成性别或种族识别工具。

**任务阻断范式（Task‑Blocking Paradigm）**：在模型训练阶段加入机制，使模型在面对某类有害任务时“卡壳”，而对正常任务仍然流畅。可以想象为在钥匙孔里装了一个只能转动特定形状钥匙的防护装置。

**自毁模型（Self‑Destructing Model）**：实现任务阻断的模型本身。它在被用于有害任务时会自动“失效”，但在正向任务上保持原有性能。

**元学习（Meta‑Learning）**：学习“如何学习”。模型在训练时不仅学会完成当前任务，还学会快速适应新任务的策略。

**对抗学习（Adversarial Learning）**：在训练中引入一个对手（对抗网络），让模型在对手的压力下变得更鲁棒或满足特定约束。

**元学习对抗审查（Meta‑Learned Adversarial Censoring，MLAC）**：本文提出的具体算法，结合元学习和对抗学习，让模型在训练时就学会对有害任务进行自我审查。

### 核心创新点

1. **从事后审计转向事前阻断**  
   *之前的做法*：模型发布后通过水印、使用日志或后置检测来发现滥用。  
   *本文的做法*：在模型训练阶段就植入“任务阻断”机制，使模型在被微调到特定有害任务时自动失效。  
   *改变*：风险控制从被动转为主动，大幅提升了对开源模型的安全可控性。

2. **将元学习用于任务阻断**  
   *之前的做法*：对抗训练通常只针对对抗样本或特定攻击。  
   *本文的做法*：把“阻断有害任务”的目标视作一个元任务，让模型在元学习的外循环中学习如何在有害任务上产生高损失。  
   *改变*：模型获得了对未知有害任务的泛化阻断能力，而不是仅针对已知攻击。

3. **引入对抗审查网络作为“审查员”**  
   *之前的做法*：对抗网络多用于提升鲁棒性或生成对抗样本。  
   *本文的做法*：训练一个审查网络专门预测微调后模型在有害任务上的表现，并把它的负反馈作为阻断信号。  
   *改变*：审查网络成为模型自我约束的“监督者”，实现了自动化的任务过滤。

4. **在保持正向任务性能的前提下实现显著阻断**  
   *之前的做法*：安全手段往往以牺牲整体性能为代价。  
   *本文的做法*：通过在元学习的外循环中同时优化正向任务的准确率和有害任务的损失，实现了两者的解耦。  
   *改变*：实验表明模型在职业分类等正向任务上几乎没有性能下降，却在性别识别等有害任务上几乎失效。

### 方法详解

**整体框架**  
MLAC 的训练过程可以划分为三层循环：  
1）**基础模型预训练**：先得到一个通用的 BERT‑style 编码器。  
2）**元学习外循环**：在每一次外循环中，随机抽取一个“有害任务”（如性别预测）和一个“正向任务”（如职业分类）。  
3）**对抗审查内部循环**：在每个外循环内部，训练一个审查网络，使其能够评估模型在有害任务上的适应程度，并把评估结果作为梯度信号，迫使基础模型在该任务上产生高错误率。

**关键模块拆解**

- **任务采样器**：类似于抽签箱，每轮随机挑选一个有害任务集合和一个正向任务集合。这样模型在训练时会看到多种有害任务的“样本”，防止只对单一任务失效。

- **审查网络（Censor）**：一个轻量的前馈网络，输入是模型在有害任务上微调后的参数或梯度，输出是该任务的预测准确率估计。它的目标是最小化估计误差，同时在训练中被设定为“对手”，希望模型在有害任务上表现好。

- **元学习更新**：外循环的目标函数由两部分组成：① 正向任务的交叉熵损失（希望低），② 审查网络给出的有害任务损失（希望高）。通过梯度下降同时更新基础模型的参数，使其在正向任务上学习，同时在有害任务上被审查网络“压制”。

- **对抗梯度反转**：在更新基础模型时，对审查网络的梯度取负（即梯度反转层），这一步是对抗学习的核心，让模型主动“对抗”审查网络的正向评估，从而在有害任务上产生更大的错误。

**最巧妙的设计**  
审查网络本身并不是固定的规则，而是通过对抗学习动态进化的“审查员”。它不断提升对模型在有害任务上表现的预测能力，迫使模型在每一次元学习迭代中都必须找到新的方式去“自毁”。这种“审查员—模型”之间的博弈，使得阻断效果能够泛化到未见过的有害任务，而不是仅针对训练时出现的任务。

### 实验与效果

- **实验设置**：作者在一个小规模实验中，以 BERT‑style 编码器为基础模型，正向任务选取职业分类（Profession Classification），有害任务选取性别识别（Gender Identification）。两者分别使用公开的职业标签数据和性别标签数据进行微调评估。

- **对比基线**：包括普通的未加任何安全机制的 BERT、以及仅使用传统对抗训练（不含元学习）的模型。

- **结果**：论文声称，经过 MLAC 训练的模型在职业分类任务上的准确率几乎保持原始水平（与普通 BERT 差距可忽略），而在性别识别任务上迁移性能大幅下降，几乎失去辨别能力。相较于仅使用普通对抗训练的模型，MLAC 在有害任务上的阻断效果更为显著。

- **消融实验**：作者分别去掉审查网络、去掉梯度反转、以及只使用单一有害任务进行元学习，结果显示每个组件的缺失都会削弱阻断效果，尤其是审查网络的移除导致有害任务的准确率恢复到接近普通 BERT 的水平。

- **局限性**：实验规模较小，仅验证了两类任务的阻断；对更复杂的有害任务（如多模态生成、长文本推理）是否同样有效，原文未给出。作者也承认，若攻击者能够获取审查网络的结构或参数，可能会设计对抗微调策略绕过阻断。

### 影响与延伸思考

自从这篇论文提出“自毁模型”概念后，安全社区开始关注“任务层面的可控性”而非仅仅是模型访问控制。后续工作（如 2024 年的 “Selective Fine‑Tuning” 与 2025 年的 “Task‑Specific Gate”）都在不同程度上借鉴了任务阻断的思路，尝试在更大规模的语言模型上实现类似的自我审查机制。对于想进一步研究的读者，可以关注以下方向：① 将审查网络扩展到多模态输入；② 探索在大规模预训练阶段直接植入任务阻断信号；③ 研究如何在不泄露审查网络细节的前提下提供可验证的安全证明。整体来看，这篇工作开启了“让模型主动拒绝有害任务”的新视角，为开源模型的安全发布提供了技术层面的可能路径。

### 一句话记住它

**自毁模型通过元学习和对抗审查，让开源大模型在有害任务上自动失效，却不牺牲正向任务的性能。**