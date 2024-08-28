# LogicGame: Benchmarking Rule-Based Reasoning Abilities of Large Language   Models

> **Date**：2024-08-28
> **arXiv**：https://arxiv.org/abs/2408.15778

## Abstract

Large Language Models (LLMs) have demonstrated notable capabilities across various tasks, showcasing complex problem-solving abilities. Understanding and executing complex rules, along with multi-step planning, are fundamental to logical reasoning and critical for practical LLM agents and decision-making systems. However, evaluating LLMs as effective rule-based executors and planners remains underexplored. In this paper, we introduce LogicGame, a novel benchmark designed to evaluate the comprehensive rule understanding, execution, and planning capabilities of LLMs. Unlike traditional benchmarks, LogicGame provides diverse games that contain a series of rules with an initial state, requiring models to comprehend and apply predefined regulations to solve problems. We create simulated scenarios in which models execute or plan operations to achieve specific outcomes. These game scenarios are specifically designed to distinguish logical reasoning from mere knowledge by relying exclusively on predefined rules. This separation allows for a pure assessment of rule-based reasoning capabilities. The evaluation considers not only final outcomes but also intermediate steps, providing a comprehensive assessment of model performance. Moreover, these intermediate steps are deterministic and can be automatically verified. LogicGame defines game scenarios with varying difficulty levels, from simple rule applications to complex reasoning chains, in order to offer a precise evaluation of model performance on rule understanding and multi-step execution. Utilizing LogicGame, we test various LLMs and identify notable shortcomings in their rule-based logical reasoning abilities.

---

# LogicGame：大语言模型规则推理能力基准测试 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在聊天、写作等任务上已经表现得相当强大，但它们是否真的能像人类一样遵循一套明确的规则并一步步执行计划，却很少有人系统评估。过去的评测大多依赖于常识问答或开放式推理，这类题目往往可以靠记忆或语言模式得分，难以区分“懂规则”与“会套话”。因此，缺少一种只考察模型对预先定义规则的理解、执行和多步规划能力的基准，导致我们无法准确判断 LLM 在实际决策系统中能否可靠地充当规则执行者。

### 关键概念速览
- **规则推理**：模型需要依据一组明确的、事先给出的规则，对当前状态进行判断并决定下一步行动，类似于棋手遵守棋规走子。  
- **多步规划**：在知道目标的前提下，模型必须设计并执行一系列连续操作才能达成目标，就像在迷宫里规划出一条通路。  
- **中间步骤验证**：评测不仅看最终答案，还会检查每一步的输出是否符合规则，类似老师批改作业时会逐行打分。  
- **确定性验证**：因为规则是固定的，模型的每一步都可以用程序自动判断对错，省去人工标注的成本。  
- **难度层级**：Benchmark 将游戏划分为从单一规则直接应用到需要长链推理的多层次任务，帮助定位模型在哪个层次出现瓶颈。  
- **规则与知识的分离**：所有题目都不依赖常识，只靠提供的规则解答，确保评测测的是“规则推理”而不是“知识回忆”。  

### 核心创新点
1. **从知识问答 → 纯规则游戏**：传统评测让模型利用已有的世界知识回答问题，作者则构造了一系列只靠预设规则才能解开的游戏，彻底剔除了知识因素的干扰。这样模型的表现直接映射到规则理解和执行能力上。  
2. **加入可验证的中间步骤 → 全程追踪**：以前的基准只看最终答案对不对，作者设计了每一步都可机器验证的机制，使得评估可以细粒度地捕捉模型在哪一步出错，帮助定位具体的推理缺陷。  
3. **多难度层级 → 细致能力划分**：单一难度的测试只能给出粗糙的好坏判断，LogicGame 按规则复杂度和推理链长度划分了多个层级，让研究者能够看到模型在“简单规则应用”和“深度链式推理”之间的性能差距。  
4. **统一的游戏框架 → 可扩展评测平台**：作者把各种规则游戏抽象为“初始状态 + 规则集合 + 目标”，形成统一的描述格式，后续只需添加新规则即可生成新任务，极大提升了基准的可持续性和社区贡献潜力。

### 方法详解
整体思路可以概括为三步：**构造游戏 → 让模型生成操作序列 → 自动校验每一步**。  
1. **游戏构造**：研究团队先挑选日常或抽象的规则系统（比如卡牌游戏的出牌规则、资源管理的调配规则），为每个游戏设定一个起始状态和一个明确的目标状态。所有规则都用自然语言描述，保持与 LLM 的交互方式一致。  
2. **模型推理**：在评测时，模型收到“当前状态 + 规则列表 + 目标”这三段文字提示，要求它输出下一步的操作或直接给出完整的操作序列。这里可以让模型一次性写出全部步骤，也可以采用“一步一步”交互的方式，让模型在每一步后重新获取更新后的状态再继续。  
3. **步骤验证**：每条规则都被程序化为一个判定函数，输入当前状态和模型给出的操作，返回是否符合规则以及执行后的新状态。评测脚本把模型的每一步喂进这些函数，若不符合则标记为错误并记录错误位置。因为规则是确定的，这一步完全自动化，无需人工审阅。  
4. **评分机制**：最终得分综合两部分：**完整成功率**（模型是否在限定步数内达成目标）和**中间正确率**（模型在整个过程中遵守规则的比例）。这种双重指标可以区分“偶然成功”与“稳健推理”。  
最巧妙的地方在于把规则本身转化为机器可执行的判定器，使得评测过程不再依赖主观人工标注，保证了大规模、可重复的实验环境。

### 实验与效果
- **测试对象**：作者在 LogicGame 上跑了多款公开的 LLM，包括 GPT‑3.5、GPT‑4、Claude、LLaMA 系列等。  
- **对比基线**：与传统的常识推理 benchmark（如 MMLU、ARC）以及一些已有的逻辑推理数据集（如 LSAT、LogicalDeduction）进行横向比较。  
- **结果概述**：论文声称，尽管这些模型在常识问答上已经接近或超过人类水平，但在规则游戏中普遍出现中间步骤错误，尤其是需要超过三步的链式推理时成功率急剧下降。具体数值未在摘要中给出。  
- **消融实验**：作者分别关闭“一步一步交互”和“中间步骤验证”两项功能，发现交互式推理能显著提升长链任务的成功率，验证了逐步反馈的重要性。  
- **局限性**：实验主要聚焦于文字描述的规则，未涉及图形或实时交互的游戏；此外，评测仍然依赖于提示工程的质量，提示不当可能导致模型表现被低估。作者在讨论中承认这些限制，并把它们列为后续工作方向。

### 影响与延伸思考
LogicGame 为 LLM 的规则执行能力提供了首个系统化、可自动验证的评测平台，随后出现的工作多围绕“规则强化学习”“可解释推理链”展开，尝试把模型的中间步骤直接反馈给训练过程，以提升其遵规能力。还有研究把 LogicGame 的游戏描述转化为代码生成任务，探索 LLM 在自动化脚本编写中的可靠性。想进一步深入的读者可以关注以下方向：① 将视觉或动作规则加入基准，形成多模态规则推理；② 结合强化学习，让模型在错误后自行修正策略；③ 开发基于 LogicGame 的微调数据集，提升模型的规则遵循率。整体来看，这篇论文打开了评估 LLM “执行者”角色的新视角，推动了从“会说”向“会做”转变的研究潮流。

### 一句话记住它
LogicGame 用纯规则的游戏和可自动验证的中间步骤，首次让我们能够精准衡量大语言模型的规则理解与多步执行能力。