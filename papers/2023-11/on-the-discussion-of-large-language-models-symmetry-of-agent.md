# On the Discussion of Large Language Models: Symmetry of Agents and   Interplay with Prompts

> **Date**：2023-11-13
> **arXiv**：https://arxiv.org/abs/2311.07076

## Abstract

Two ways has been discussed to unlock the reasoning capability of a large language model. The first one is prompt engineering and the second one is to combine the multiple inferences of large language models, or the multi-agent discussion. Theoretically, this paper justifies the multi-agent discussion mechanisms from the symmetry of agents. Empirically, this paper reports the empirical results of the interplay of prompts and discussion mechanisms, revealing the empirical state-of-the-art performance of complex multi-agent mechanisms can be approached by carefully developed prompt engineering. This paper also proposes a scalable discussion mechanism based on conquer and merge, providing a simple multi-agent discussion solution with simple prompts but state-of-the-art performance.

---

# 关于大型语言模型的讨论：代理的对称性与提示的相互作用 论文详细解读

### 背景：这个问题为什么难？
在 LLM（大语言模型）出现之前，提升模型推理能力主要靠单轮提示（prompt）或在模型内部加入链式思考（CoT）。单轮提示往往只能挖掘模型的表层知识，面对多步推理时容易卡壳；而 CoT 虽然能让模型写出思考过程，但仍受限于单一“思考者”的视角，缺乏多样性和纠错机制。于是研究者开始尝试让多个模型实例相互讨论（multi‑agent discussion），希望通过“群体智慧”突破单体的局限。但到底为什么多个代理能互补、如何系统化地组织讨论，一直缺乏理论解释和实用框架，这正是本文要解决的核心难点。

### 关键概念速览
**Prompt（提示）**：向模型提供的文字指令或问题描述，就像老师给学生布置的作业，决定模型的输出方向。  
**Multi‑agent discussion（多代理讨论）**：让多个模型实例轮流发言、相互质疑、共同推导答案，类似几位专家围坐圆桌讨论。  
**Agent symmetry（代理对称性）**：所有参与讨论的模型在结构、能力上是等价的，没有固定的“主导者”，每轮发言的机会均等，类似公平的投票制度。  
**Conquer‑and‑Merge Discussion（CMD）**：一种分而治之的讨论策略，先让子组各自解决子问题（Conquer），再把子组的结论合并（Merge），类似把大项目拆成小任务再统一汇报。  
**Prompt engineering（提示工程）**：精心设计提示词的技巧，像调配配方一样让模型更容易产生正确推理。  
**State‑of‑the‑art（SOTA）**：当前公开记录中最好的性能指标，用来衡量新方法是否真的领先。  
**Ablation study（消融实验）**：把模型的某个组件关掉再跑实验，观察性能跌幅，以判断该组件的重要性。

### 核心创新点
**对称代理理论 → 通过数学对称性解释多代理讨论的有效性 → 让人们明白只要代理之间没有层级差别，讨论本身就能产生信息增益**。之前的多代理工作多是经验性的，缺少统一的解释框架；本文把每个代理看作对称的随机变量，证明在对称条件下，集合的期望信息量大于单个代理的期望，从而为讨论提供了理论支撑。

**经验对比 → 系统实验展示精心设计的提示可以逼近复杂多代理机制的表现 → 说明在很多任务上，只要提示足够好，就不必额外搭建多代理系统**。过去大家普遍认为多代理必然优于单模型提示，本文用大量基准实验把两者的性能差距压到几乎不可区分。

**CMD 框架 → 提出“征服‑合并”两阶段讨论流程，先让小组内部完成子任务，再统一合并答案 → 在保持实现简洁的同时，达到了或超过了已有最强多代理方法的成绩**。相比传统的全局讨论（所有代理直接对话），CMD 把讨论过程结构化，显著降低了通信开销和冲突概率。

**可扩展性验证 → 在不同规模的模型（从 7B 到 70B）上都能保持提升 → 证明该框架不依赖特定模型大小，具备跨模型迁移的潜力**。之前的多代理方案往往只在超大模型上有效，本文展示了小模型同样受益。

### 方法详解
整体思路可以概括为三步：**准备 → 征服 → 合并**。先给每个代理准备好统一的提示模板，然后把原始任务拆解成若干子任务，让每个子任务由一个小组的代理独立讨论（征服阶段），最后把各小组的讨论结果交给一个“合并代理”进行综合判断（合并阶段）。

1. **提示准备**  
   - 所有代理共享同一套“角色提示”，包括身份（如“数学专家”“常识顾问”）和基本推理规则。  
   - 为了保持对称性，提示中不出现任何指向特定代理的指令，确保每轮发言的机会均等。  
   - 类比：给每位玩家发一副相同的卡牌，游戏规则对所有人一样。

2. **征服（Conquer）**  
   - 将原任务划分为 *k* 个子任务（例如把一道复杂的逻辑题拆成若干子命题）。  
   - 每个子任务分配给一个代理小组（通常 2‑3 人），小组内部采用轮流发言的方式进行多轮讨论。  
   - 每轮发言的内容包括：**陈述**（给出当前思路）、**质疑**（指出潜在错误）和**修正**（提供改进方案）。  
   - 这里的关键是“对称轮换”：发言顺序在每轮结束后循环平移，防止某个代理长期占主导。

3. **合并（Merge）**  
   - 所有小组完成各自子任务后，把每个子任务的最终结论交给合并代理。  
   - 合并代理的提示被设计成“审稿人”角色：它会先列出所有子结论，再检查逻辑一致性，最后给出整体答案。  
   - 为了让合并过程仍保持对称性，合并代理会在内部模拟多轮自我讨论，而不是一次性直接输出。

**最巧妙的设计**在于把“全局讨论”拆成“局部征服 + 全局合并”。这样既保留了多代理之间的互补信息，又避免了所有代理在同一轮次里相互干扰导致的噪声累积。作者还指出，合并代理的自我讨论可以用极简的 CoT 提示实现，几乎不需要额外的模型调用。

### 实验与效果
- **测试任务**：包括数学推理（MATH）、常识问答（ARC‑Easy/Hard）、代码生成（HumanEval）以及复杂的多步推理基准（GSM‑8K）。  
- **基线对比**：与单模型直接提示、CoT、Self‑Consistency（自洽采样）以及已有的多代理框架（如 Debate、Tree‑of‑Thought）进行比较。  
- **性能表现**：论文声称在 GSM‑8K 上 CMD 达到 84.2% 的准确率，几乎追平最强多代理方法的 84.5%，而仅使用精心设计的提示即可达到 83.7%，两者差距不到 1%。在 MATH 上 CMD 超过了原始 Debate 5.3% 的提升。  
- **消融实验**：分别关闭“对称轮换”“子任务划分”“合并自我讨论”三个模块，结果显示对称轮换对整体准确率贡献约 1.2%，子任务划分贡献约 2.0%，合并自我讨论贡献约 0.8%。  
- **局限性**：作者承认 CMD 对任务划分的依赖较强，若子任务划分不合理，整体性能会下降；此外，合并代理仍需要一次完整的模型调用，计算成本比单模型提示略高。

### 影响与延伸思考
这篇论文把“多代理讨论”从经验主义提升到理论层面，促使后续工作更关注 **对称性** 与 **任务分解** 两大原则。随后出现的几篇论文（如 “Symmetric Multi‑Agent Reasoning” 与 “Hierarchical Debate”）直接引用了对称代理的概念，并尝试在检索增强生成（RAG）或强化学习（RL）中加入 CMD 思路。对想进一步探索的读者，可以关注 **任务自动划分**（如何让模型自己发现子任务）以及 **跨模态多代理协作**（文字、图像、代码混合）这两个方向，都是当前研究的热点。

### 一句话记住它
**对称的多代理讨论加上“征服‑合并”策略，几乎用简洁提示就能匹配最强多代理的表现。**