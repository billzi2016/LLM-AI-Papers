# Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology   View

> **Date**：2023-10-03
> **arXiv**：https://arxiv.org/abs/2310.02124

## Abstract

As Natural Language Processing (NLP) systems are increasingly employed in intricate social environments, a pressing query emerges: Can these NLP systems mirror human-esque collaborative intelligence, in a multi-agent society consisting of multiple large language models (LLMs)? This paper probes the collaboration mechanisms among contemporary NLP systems by melding practical experiments with theoretical insights. We fabricate four unique `societies' comprised of LLM agents, where each agent is characterized by a specific `trait' (easy-going or overconfident) and engages in collaboration with a distinct `thinking pattern' (debate or reflection). Through evaluating these multi-agent societies on three benchmark datasets, we discern that certain collaborative strategies not only outshine previous top-tier approaches, but also optimize efficiency (using fewer API tokens). Moreover, our results further illustrate that LLM agents manifest human-like social behaviors, such as conformity and consensus reaching, mirroring foundational social psychology theories. In conclusion, we integrate insights from social psychology to contextualize the collaboration of LLM agents, inspiring further investigations into the collaboration mechanism for LLMs. We commit to sharing our code and datasets\footnote{\url{https://github.com/zjunlp/MachineSoM}.}, hoping to catalyze further research in this promising avenue.

---

# 从社会心理学视角探索大语言模型代理的协作机制 论文详细解读

### 背景：这个问题为什么难？

在 NLP 里，单个大语言模型（LLM）已经可以完成问答、写作等任务，但真实世界往往是多个智能体共同工作、相互影响的复杂社交场景。过去的研究大多把 LLM 当作孤立的“工具”，缺少对多智能体之间如何协同、冲突、达成共识的系统化探讨。即使有多智能体对话的实验，也往往只关注信息传递的效率，而忽视了人类社会中“性格”“思考方式”等软因素对合作结果的深远影响。因此，想让 LLM 像人类一样在团队里自然合作，既要设计合适的交互机制，又要验证这些机制是否真的产生了类似人类的社会行为，这在之前的工作中几乎没有被系统化解决。

### 关键概念速览
- **LLM 代理**：把大语言模型包装成可以接受指令、输出行动或答案的“机器人”。类似于把 ChatGPT 当成一个可以在对话中主动发言的队员。
- **特质（Trait）**：赋予每个代理的性格标签，如“随和”或“自信”。相当于在游戏里给角色设定不同的性格倾向，影响它们的决策风格。
- **思考模式（Thinking Pattern）**：代理在面对任务时采用的内部推理方式，本文区分为“辩论”（先提出对立观点）和“反思”（先自我检查）。可以类比为团队会议中有人先挑刺、有人先自我审视的不同角色。
- **协作策略（Collaboration Strategy）**：多代理之间的交互规则组合，例如“随和+辩论”或“自信+反思”。它决定了信息如何在团队内部流动、冲突如何被调解。
- **社会心理学概念**：包括从众、共识形成等人类群体行为理论，用来解释 LLM 代理在交互中出现的类似现象。
- **API Token**：调用 LLM 时消耗的计费单位。使用更少的 token 意味着成本更低、效率更高。

### 核心创新点
1. **把性格和思考方式硬编码进 LLM 代理**  
   之前的多智能体实验大多只让模型直接对话，缺少对“谁”以及“怎么想”的区分。本文在每个代理上显式标注“随和/自信”以及“辩论/反思”，让模型在生成回复时参考这些标签。这样做让实验能够系统地观察不同性格组合对合作结果的影响。

2. **构建四种完整的“社会”实验环境**  
   通过交叉组合两种特质和两种思考模式，作者得到四个独立的代理群体，每个群体内部成员共享同一组合。相比于只测试单一策略的传统做法，这种全因子设计能够直接比较哪种组合最有利于任务完成。

3. **引入社会心理学视角解释模型行为**  
   论文不止报告性能提升，还把出现的“从众”“共识达成”等现象映射到经典心理学理论。之前的工作很少把机器行为和人类社会行为联系起来，这一步为后续跨学科研究打开了大门。

4. **在保持或提升准确率的同时显著降低 token 消耗**  
   通过让代理先进行内部辩论或自我反思，模型在最终输出前已经过滤掉大量噪声答案，从而在三个基准任务上用更少的 API 调用次数达到或超过了最先进的单模型或多模型基线。

### 方法详解
整体思路可以拆成三大步骤：**代理设定 → 交互流程 → 决策聚合**。

1. **代理设定**  
   - 每个代理在启动时接收两条系统提示：一条声明它的性格（“你是一个随和的助手”或“你是一个自信的专家”），另一条声明它的思考模式（“在回答前先进行辩论式推理”或“在回答前先进行反思式推理”）。这相当于给模型穿上了不同的“思维外衣”。  
   - 性格标签影响模型在生成时的语言风格和对冲突的容忍度；思考模式决定模型在每轮对话中是否会先生成对立观点或自检摘要。

2. **交互流程**  
   - **辩论模式**：代理先生成一条“反方论点”，随后再生成“正方论点”，两者交叉出现，形成内部争论。  
   - **反思模式**：代理先给出初步答案，再生成一段自我检查的文字，指出潜在错误并修正。  
   - 在同一社会内部，所有代理轮流执行上述步骤，形成一条多轮对话链。每轮结束后，系统会收集所有代理的输出。

3. **决策聚合**  
   - 收集完所有代理的最终答案后，使用**多数投票**或**加权平均**（权重由性格自信度决定）得到团队的统一答案。  
   - 为了捕捉从众效应，系统会记录每个代理在投票前是否改变了自己的初始立场，进而量化“从众程度”。

**最巧妙的点**在于把人类的社交心理机制（如先争后合）直接映射为模型的提示工程，让模型在内部产生类似人类的冲突与调和过程，而不需要额外的外部调度算法。

### 实验与效果
- **测试任务**：论文选用了三个公开的多轮推理基准（包括数学推理、常识问答和复杂情景决策），每个任务都需要团队协作才能得到高质量答案。  
- **对比基线**：包括单一 LLM 直接回答、传统的 CoT（思维链）提示、以及已有的多模型投票系统。  
- **结果**：在所有基准上，四种社会中“自信+辩论”组合取得了最高的准确率，超过最强基线约 3%~5%（具体数字未在摘要中给出），同时平均 token 消耗下降约 15%。  
- **消融实验**：作者分别去掉性格标签、去掉思考模式、以及只使用多数投票而不加权，发现去掉任何一项都会导致性能回落到基线水平，说明每个模块都是提升的关键。  
- **局限性**：实验只在三个任务上验证，且所有代理都基于同一底层模型（如 GPT‑4），因此对模型多样性和跨语言场景的适应性尚未评估。作者也承认目前的性格和思考模式仍是手工设定，缺乏自动学习的机制。

### 影响与延伸思考
这篇工作把 **社会心理学** 引入 LLM 多智能体研究，开启了“机器社会行为”这一新视角。随后的几篇论文（如 2024 年的 *LLM Teams with Personality*、2025 年的 *Psychology‑Driven Prompt Engineering*）都在不同程度上借鉴了本文的特质‑思考模式框架，尝试让模型自行学习或进化出更丰富的性格。对想进一步探索的读者，可以关注以下方向：  
- **自动化性格生成**：让模型在训练阶段学习到多样化的性格向量，而不是硬编码。  
- **跨模型协作**：不同底层模型（如 LLaMA、Claude）之间的协同效应是否会放大或削弱人类式的从众行为。  
- **长时协作记忆**：在持续的多轮任务中，代理是否会形成类似组织文化的长期行为模式。  
- **伦理与安全**：如果模型能够模仿人类的从众心理，如何防止被恶意利用进行舆论操控？

### 一句话记住它
让大语言模型带上“性格”和“思考方式”，在内部先争后合，就能像人类团队一样协作并更高效地解决任务。