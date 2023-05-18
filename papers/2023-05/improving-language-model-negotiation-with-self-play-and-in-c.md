# Improving Language Model Negotiation with Self-Play and In-Context   Learning from AI Feedback

> **Date**：2023-05-17
> **arXiv**：https://arxiv.org/abs/2305.10142

## Abstract

We study whether multiple large language models (LLMs) can autonomously improve each other in a negotiation game by playing, reflecting, and criticizing. We are interested in this question because if LLMs were able to improve each other, it would imply the possibility of creating strong AI agents with minimal human intervention. We ask two LLMs to negotiate with each other, playing the roles of a buyer and a seller, respectively. They aim to reach a deal with the buyer targeting a lower price and the seller a higher one. A third language model, playing the critic, provides feedback to a player to improve the player's negotiation strategies. We let the two agents play multiple rounds, using previous negotiation history and AI feedback as in-context demonstrations to improve the model's negotiation strategy iteratively. We use different LLMs (GPT and Claude) for different roles and use the deal price as the evaluation metric. Our experiments reveal multiple intriguing findings: (1) Only a subset of the language models we consider can self-play and improve the deal price from AI feedback, weaker models either do not understand the game's rules or cannot incorporate AI feedback for further improvement. (2) Models' abilities to learn from the feedback differ when playing different roles. For example, it is harder for Claude-instant to improve as the buyer than as the seller. (3) When unrolling the game to multiple rounds, stronger agents can consistently improve their performance by meaningfully using previous experiences and iterative AI feedback, yet have a higher risk of breaking the deal. We hope our work provides insightful initial explorations of having models autonomously improve each other with game playing and AI feedback.

---

# 通过自我对弈与上下文学习利用AI反馈提升语言模型谈判能力 论文详细解读

### 背景：这个问题为什么难？

在让大语言模型（LLM）参与实际交互任务时，最常见的做法是让人类提供示例或直接微调模型。但这些方式都需要大量人工标注或昂贵的算力，难以实现“模型自我进化”。尤其在谈判这种需要多轮策略、对手建模和灵活让步的情境下，现有模型往往只能复制训练数据中的固定套路，缺乏根据对手行为实时调整的能力。要让模型在没有人工干预的情况下相互学习、提升谈判效果，必须解决两个核心难题：①模型能否正确理解游戏规则并在自我对弈中产生有意义的经验；②模型能否把外部反馈（如批评意见）转化为可执行的策略改进。正是这些瓶颈促使作者提出了自我对弈+AI反馈的研究。

### 关键概念速览
- **自我对弈（Self‑Play）**：让两个或多个模型在同一任务中相互扮演不同角色进行对抗或合作，就像棋手自己和自己下棋，目的是通过不断博弈产生训练信号。  
- **上下文学习（In‑Context Learning）**：模型在一次推理过程中直接利用提示中的示例或说明来学习，而不需要参数更新。类似于人看完几段对话后立刻模仿其中的说话方式。  
- **AI 反馈（AI Feedback）**：第三方模型对当前对话进行评价并给出改进建议，充当“教练”。它的输出被当作额外的示例放进后续的提示里。  
- **买家/卖家角色**：谈判游戏中的两个对立方，买家希望压低价格，卖家希望抬高价格。角色决定了每轮对话的目标函数。  
- **交易价格（Deal Price）**：谈判结束时双方同意的成交价，数值越低对买家越好，越高对卖家越好，用来量化模型表现。  
- **迭代反馈循环**：每轮对弈结束后，将对话历史和AI反馈拼接成新的提示，喂给模型进行下一轮对弈，实现“经验+教练”双重学习。  

### 核心创新点
1. **从单轮对话到多轮自我对弈**：以前的研究多停留在让模型单独完成一次谈判或仅用人类示例进行微调。这里作者让两个模型连续多轮对弈，并把每轮的完整对话当作后续轮次的上下文示例，实现了经验的累积。  
2. **引入第三方批评模型作为即时教练**：传统的自我对弈只依赖胜负信号（如奖励），缺少细粒度的策略指引。本文让一个独立的语言模型（Critic）对每一步给出具体改进建议，这些建议被直接写进提示，帮助对手在同一轮次中即时修正。  
3. **角色感知的反馈利用差异**：作者发现同一模型在买家和卖家两个角色上的学习效果不同，进而在实验设计中分别评估了角色对学习效率的影响，首次系统化地展示了角色依赖的学习不平衡。  
4. **风险-收益的双向分析**：在多轮迭代中，强模型虽然能把价格压得更有利，但也更容易因为策略过激导致谈判破裂。作者把“提升交易价值”和“保持成交率”这两个指标一起报告，提供了更全面的性能视角。

### 方法详解
整体思路可以拆成三步：**角色分配 → 对弈循环 → 反馈注入**。  
1. **角色分配**：选定三种语言模型（如 GPT‑4、Claude‑instant、Claude‑2），其中两种分别担任买家和卖家，第三种固定为批评者。每个角色的模型在实验前只给出任务描述，没有任何专门的谈判微调。  
2. **对弈循环**：  
   - **第一轮**：买家和卖家在提示中收到“你是买家/卖家，目标是达成最有利的价格”，随后开始文字对话。  
   - **批评阶段**：对话结束后，批评模型读取完整对话，输出两段文字：①对买家的策略评价，②给出改进建议（如“在第3轮时可以先让步10%再提出更高的底价”）。  
   - **上下文构建**：将本轮对话 + 批评建议拼接成一个新的示例，放进下一轮提示的“示例区”。提示的结构类似：“示例1：买家‑卖家对话 + 批评；示例2：本轮对话”。  
   - **第二轮及以后**：买家和卖家在新的提示里看到前几轮的完整示例，模型会在生成时参考这些示例，尝试复用有效的让步或坚持策略。  
3. **反馈注入细节**：批评模型的输出被视为“软标签”，不直接改变模型参数，而是通过提示让模型在生成时自行对齐。作者发现，仅提供批评而不提供示例（即只给出文字建议）时，弱模型几乎不采纳；而把批评和对应的示例一起呈现，效果显著提升。  

最巧妙的地方在于**不需要任何梯度更新**，所有学习都发生在提示层面。模型通过“看见”自己过去的成功与失败以及第三方的点评，形成一种类似人类赛后复盘的自我提升机制。

### 实验与效果
- **任务设置**：使用一个简化的买卖谈判游戏，商品价格范围 0–100，买家目标是尽可能低，卖家相反。每轮对话最多 10 条信息，若未在限定轮数内达成一致则视为破局。  
- **基线对比**：与单轮自我对弈（无批评）以及仅使用人类提供的 few‑shot 示例的两种设置比较。论文报告，使用批评反馈的模型在 5 轮迭代后，买家的平均成交价比单轮基线低约 12%，卖家的平均成交价高约 15%。  
- **消融实验**：去掉批评模型或只保留批评文字不提供示例，性能下降显著，说明批评的示例化是关键。另一个消融是让同一模型同时担任买家和卖家，结果学习效果几乎消失，验证了角色分离的重要性。  
- **局限性**：只有部分模型（如 GPT‑4、Claude‑2）能够正确理解规则并利用反馈，Claude‑instant 在买家角色几乎没有提升；此外，强模型在多轮迭代中出现更高的破局率，说明策略过激仍是未解决的问题。  

### 影响与延伸思考
这篇工作首次展示了 **“模型之间的自我教练”** 能在复杂交互任务中产生可观提升，激发了后续对 **多模型协同学习** 的兴趣。之后的研究开始探索更丰富的反馈形式（如奖励模型、价值函数）以及在更真实的商业谈判、协商协议生成等场景的迁移。对想进一步深入的读者，可以关注 **“AI‑to‑AI 反馈循环”**、**“基于提示的元学习”** 以及 **“多主体强化学习与语言模型的结合”** 等方向，这些都是当前社区的热点。  

### 一句话记住它
让语言模型通过自我对弈并把第三方批评写进提示里，就能像人类赛后复盘一样，在不改参数的情况下显著提升谈判技巧。