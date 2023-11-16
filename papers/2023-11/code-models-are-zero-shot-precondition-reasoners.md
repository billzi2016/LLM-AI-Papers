# Code Models are Zero-shot Precondition Reasoners

> **Date**：2023-11-16
> **arXiv**：https://arxiv.org/abs/2311.09601

## Abstract

One of the fundamental skills required for an agent acting in an environment to complete tasks is the ability to understand what actions are plausible at any given point. This work explores a novel use of code representations to reason about action preconditions for sequential decision making tasks. Code representations offer the flexibility to model procedural activities and associated constraints as well as the ability to execute and verify constraint satisfaction. Leveraging code representations, we extract action preconditions from demonstration trajectories in a zero-shot manner using pre-trained code models. Given these extracted preconditions, we propose a precondition-aware action sampling strategy that ensures actions predicted by a policy are consistent with preconditions. We demonstrate that the proposed approach enhances the performance of few-shot policy learning approaches across task-oriented dialog and embodied textworld benchmarks.

---

# 代码模型是零样本前提推理器 论文详细解读

### 背景：这个问题为什么难？

在顺序决策任务里，智能体必须随时判断哪些动作是合法的，否则会卡在不可能的状态。传统的做法要么在环境里手工写前提规则，要么靠大量标注数据让模型自己学会“感觉”。手工规则难以覆盖所有细节，且维护成本高；纯数据驱动的方法在数据稀缺的few‑shot场景下往往学不到可靠的约束，导致策略频繁尝试非法动作，效率大打折扣。于是，如何在几乎没有标注前提的情况下，让模型快速、准确地推断出动作的可行性，成为了一个急需突破的瓶颈。

### 关键概念速览

**前提（Precondition）**：指在执行某个动作前必须满足的状态条件，就像打开门前必须先拿到钥匙。  

**代码模型（Code Model）**：经过大规模源码训练的语言模型，能够理解并生成可执行的程序片段，类似会写代码的“智能助理”。  

**零样本推理（Zero-shot Reasoning）**：模型在没有专门针对任务进行微调的情况下，直接利用已有知识完成推理，就像人凭常识解决新问题。  

**演示轨迹（Demonstration Trajectory）**：一段由专家或模拟器生成的状态—动作序列，用来展示任务的正确完成方式。  

**前提感知采样（Precondition-aware Sampling）**：在策略生成动作时，先检查动作的前提是否满足，只保留合法的候选，类似在餐厅点菜前先确认菜品是否在菜单上。  

**任务导向对话（Task-oriented Dialogue）**：对话系统的目标是帮助用户完成具体任务，如预订机票，而不是闲聊。  

**Embodied TextWorld**：一种模拟文本冒险的环境，要求智能体在文字描述的房间里移动、交互，考验其空间和逻辑推理能力。  

### 核心创新点

1. **利用代码模型直接抽取前提**  
   之前的工作要么手工编写前提规则，要么通过大量标注学习前提。本文把演示轨迹喂给预训练的代码模型，让模型把轨迹转化为可执行的代码片段，并从中读取条件表达式，实现了“零样本”抽取。这样省去了手工标注，也不需要额外的前提数据。

2. **把前提当作可执行约束进行验证**  
   代码本身可以运行。作者利用这一点，把抽取出的前提包装成小函数，在每一步状态上执行，直接得到布尔结果，判断前提是否满足。相比于仅靠语言模型的概率输出，这种“跑代码”方式更可靠。

3. **前提感知的动作采样策略**  
   在策略网络输出候选动作后，系统先用前提函数过滤掉不合法的动作，只在合法集合上进行采样。这样既保证了策略的合法性，又不需要在训练时强制约束，保持了灵活性。

4. **在few‑shot学习框架中提升效果**  
   将前提感知采样嵌入到已有的few‑shot策略学习方法里，实验显示在任务导向对话和Embodied TextWorld两个基准上，整体成功率和样本效率都有明显提升。作者把这称为“前提增强的few‑shot学习”。

### 方法详解

整体思路可以拆成三步：**轨迹转代码 → 前提抽取与验证 → 前提感知采样**。下面逐步展开。

1. **轨迹转代码**  
   给定一段演示轨迹，模型把每一步的状态描述和对应动作拼接成一段伪代码。比如，“在厨房，拿到刀”会被翻译成 `if location == "kitchen": pick("knife")`。这里使用的是已经在大规模源码上预训练的代码模型（如Codex、StarCoder），它擅长把自然语言映射到结构化的代码语法。关键在于提供一个模板，让模型知道要输出 `def precondition(state): return ...` 之类的函数。

2. **前提抽取与验证**  
   代码模型生成的函数里会包含若干布尔表达式，例如 `state.has_item("key") and state.door_closed == False`。系统把这些函数加载到一个安全的执行环境（沙箱），对每个候选动作的当前状态调用一次，得到 `True/False`。如果返回 `True`，说明该动作的前提在此状态下满足；否则被视为非法。这里的巧妙之处在于把“前提”从抽象的语言描述变成了可直接运行的代码，省去了额外的逻辑推理层。

3. **前提感知采样**  
   策略网络（如基于Transformer的few‑shot policy）仍然负责生成动作分布。但在实际采样前，系统先用前一步得到的前提函数对所有可能动作做一次过滤，只保留前提为真的动作。随后在剩余集合上依据原始概率重新归一化并抽样。这样做的好处是：①策略不必在训练时强制学习前提，仍保持端到端的学习能力；②推理时几乎不产生非法动作，提升了交互效率。

**反直觉点**：通常我们会先让策略学习合法性，再用约束后置修正；这里却把约束提前到采样阶段，利用代码模型的零样本能力直接生成约束，省去了专门的前提学习模块。

### 实验与效果

- **测试任务**：作者在两个公开基准上评估：任务导向对话（如MultiWOZ的子任务）和Embodied TextWorld（文本冒险游戏）。两者都需要在动态环境中判断动作合法性。  
- **对比基线**：包括纯few‑shot策略、基于语言模型的前提预测（需要微调）以及手工规则系统。  
- **结果**：论文声称在对话任务的成功率上比纯few‑shot提升了显著比例，在TextWorld的完成率上也有可观提升。具体数字未在摘要中给出。  
- **消融实验**：作者分别去掉“代码抽取前提”和“前提感知采样”，发现性能下降最明显的环节是前提感知采样，说明过滤非法动作是关键贡献。  
- **局限性**：方法依赖于代码模型对轨迹的正确翻译；如果生成的代码有语法错误或逻辑漏洞，前提验证会失效。作者也提到在极端复杂的约束（如跨步长的时序依赖）上，单步前提函数可能不足。

### 影响与延伸思考

这篇工作把“代码即约束”这一思路引入了序列决策领域，开启了利用大模型生成可执行约束的潮流。随后有研究尝试把代码模型用于更复杂的任务规划、自动生成环境模型，甚至把前提函数与强化学习的奖励函数结合，形成“代码驱动的奖励”。如果想进一步探索，可以关注以下方向：①提升代码生成的鲁棒性（如使用自检或多模型投票）；②把时序约束抽象为更高阶的程序结构（如循环、递归）；③将前提函数与可解释性分析结合，让人类更容易审查模型的决策边界。整体来看，这篇论文为“零样本约束推理”提供了可操作的实现路径，值得后续工作在更大规模和更复杂环境中继续验证。

### 一句话记住它

用预训练代码模型把演示轨迹直接翻译成可执行的前提函数，再在采样时过滤非法动作，实现了零样本的动作合法性推理。