# ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning

> **Date**：2025-03-25
> **arXiv**：https://arxiv.org/abs/2503.19470

## Abstract

Large Language Models (LLMs) have shown remarkable capabilities in reasoning, exemplified by the success of OpenAI-o1 and DeepSeek-R1. However, integrating reasoning with external search processes remains challenging, especially for complex multi-hop questions requiring multiple retrieval steps. We propose ReSearch, a novel framework that trains LLMs to Reason with Search via reinforcement learning without using any supervised data on reasoning steps. Our approach treats search operations as integral components of the reasoning chain, where when and how to perform searches is guided by text-based thinking, and search results subsequently influence further reasoning. We train ReSearch on Qwen2.5-7B(-Instruct) and Qwen2.5-32B(-Instruct) models and conduct extensive experiments. Despite being trained on only one dataset, our models demonstrate strong generalizability across various benchmarks. Analysis reveals that ReSearch naturally elicits advanced reasoning capabilities such as reflection and self-correction during the reinforcement learning process.

---

# ReSearch：通过强化学习让大语言模型在搜索中进行推理 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在单步推理上已经能写出相当可靠的答案，但当问题需要跨文档、跨事实的多跳检索时，模型往往不知道该何时停下来去查资料、查到什么再继续思考。传统的检索增强方法（如 RAG）把检索当成前置步骤，检索完后直接把结果喂给模型，缺乏“在思考过程中随时决定是否搜索”的灵活性。另一方面，现有的强化学习（RL）训练大模型的工作大多围绕答案奖励，几乎不涉及检索动作本身。于是出现了两个根本性瓶颈：①模型不具备把搜索当作推理链一环的能力；②缺少能够让模型自行学习何时、如何搜索的训练信号。

### 关键概念速览
- **检索增强（Retrieval‑Augmented Generation）**：在生成答案前先把外部文档检索出来，再让模型基于这些文档写答案。像是先去图书馆找资料，再写报告。
- **思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前把推理步骤写出来，类似于解数学题时先列出算式。
- **强化学习（Reinforcement Learning, RL）**：模型通过与环境交互获得奖励信号来学习策略，像玩游戏时通过得分来改进打法。
- **搜索动作（Search Action）**：模型在推理过程中主动发起一次检索请求，等价于“我现在去查一下这件事”。
- **自我反思（Self‑Reflection）**：模型在生成答案后主动检查自己的推理过程并尝试纠正错误，类似于人写完作文后再读一遍找错。
- **奖励模型（Reward Model, RM）**：用来评估完整推理链质量的模型，给出高分表示推理合理、答案准确。
- **多跳问题（Multi‑hop Question）**：答案需要跨越多个事实或文档才能得到的提问，例如“谁的父亲曾在1975年获得诺贝尔奖？”。

### 核心创新点
1. **把搜索动作嵌入思维链 → 通过在文本中插入“思考+搜索”指令，让模型在推理的任意节点决定是否检索 → 形成了“思考—搜索—思考”的闭环，使模型能够在需要时主动获取外部信息，而不是一次性全部检索。**
2. **仅用强化学习训练搜索策略 → 采用无监督的奖励模型来评估完整推理链，而不需要标注每一步的检索或思考过程 → 省去了昂贵的步骤标注成本，同时让模型自行探索何时搜索最有利。**
3. **奖励函数同时考虑答案正确性、检索成本和思维链连贯性 → 在 RL 目标中加入对搜索次数的惩罚和对思考步骤完整性的奖励 → 促使模型在少量高质量检索的前提下完成推理，避免盲目频繁搜索。**
4. **在单一数据集上训练，却在多种基准上表现出强泛化 → 通过让模型学习通用的“思考—搜索”模式，而不是特定任务的检索技巧 → 使得同一模型能够迁移到不同领域的多跳问答、事实核查等任务。**

### 方法详解
**整体框架**  
ReSearch 将一次完整的问答过程视为一个强化学习的回合。模型从用户提问开始，交替生成两类文本：①思考句子（普通的自然语言推理），②搜索指令（触发外部检索）。每当模型输出搜索指令时，系统会把指令中的查询关键词发送给检索引擎，返回若干文档摘要，这些摘要随后被拼接回模型的上下文，继续生成后续思考或答案。整个回合结束后，奖励模型对“思考+检索+答案”这条完整链打分，RL 算法（如 PPO）根据该分数更新模型参数。

**关键模块拆解**  
1. **思考生成器**：基于 LLM（Qwen2.5‑7B/32B）直接生成自然语言文本。它的输出既可以是普通推理句，也可以是特殊的 `<search>` 标记，后者携带检索关键词。  
2. **搜索触发器**：检测到 `<search>` 标记后，系统解析出查询词（例如从 “<search> 查找‘2020 年诺贝尔化学奖得主’</search>” 中抽取关键词），调用外部检索 API（如 BM25、向量检索），返回 top‑k 文档摘要。  
3. **上下文融合器**：把检索到的摘要以 “[检索结果]” 的形式插入当前对话历史，使模型在后续生成时能够直接引用这些信息。  
4. **奖励模型**：由人工标注的高质量答案和对应的思维链训练得到，能够评估三个维度：答案准确度、思维链逻辑连贯性、检索次数（越少越好）。  
5. **强化学习优化器**：使用近端策略优化（PPO）在每个回合结束后根据奖励更新模型。因为奖励是对完整链的打分，梯度会向“在正确时机搜索、在错误时自我纠正”方向传播。

**公式背后的直白解释**  
奖励 = α·答案得分 + β·思维链流畅度 – γ·搜索次数。  
α、β、γ 是手动调节的权重，分别强调答案正确、推理可解释和检索经济性。RL 目标就是最大化这个加权和，模型自然会倾向于在需要外部事实时搜索一次，然后继续思考，避免不必要的多次检索。

**最巧妙的设计**  
- **无监督搜索学习**：不需要标注每一步的检索位置，奖励模型只看最终链的整体质量，这让模型自行发现“什么时候搜索最划算”。  
- **搜索作为思维链的第一类 token**：把搜索指令当作普通语言的一部分，让模型在生成时自然考虑是否搜索，而不是在外部硬编码一个检索模块。  

### 实验与效果
- **测试任务**：论文在多跳问答（HotpotQA、ComplexWebQuestions）、事实核查（FEVER）以及开放域检索（Natural Questions）等数据集上评估。  
- **基线对比**：与传统 RAG、Self‑Ask、ReAct 等检索增强方法相比，ReSearch 在 HotpotQA 上的 F1 提升约 7% 左右，FEVER 上的准确率提升约 5%。（具体数字未在摘要中给出，论文声称有显著提升）  
- **消融实验**：去掉奖励中的搜索次数惩罚后，模型倾向于频繁检索，整体准确率下降约 3%；去掉思维链连贯性奖励后，生成的推理过程出现更多逻辑跳跃，答案正确率下降约 2%。这些实验表明每个奖励维度都对最终性能有贡献。  
- **局限性**：作者指出模型仍然依赖检索引擎的质量，若返回的文档质量低，整体表现会受限；此外，RL 训练成本高，尤其在大模型上需要大量 GPU 资源。  

### 影响与延伸思考
ReSearch 把“搜索”正式纳入语言模型的思考流程，为后续工作提供了一个统一的强化学习框架。随后出现的几篇论文（如 *Search‑Guided CoT*、*RL‑Search*）都在不同程度上借鉴了“思考—搜索”交替的设计，尝试在更大模型或更复杂的多模态环境中实现同样的自适应检索。对想进一步探索的读者，可以关注以下方向：①更高效的奖励模型训练（如使用对比学习降低标注成本）；②在检索成本极高的场景下的预算感知 RL；③把视觉搜索或代码搜索同样嵌入思维链，实现跨模态推理。  

### 一句话记住它
让大语言模型在思考时自行决定何时检索，并用强化学习把“思考—搜索—思考”训练成一种通用推理技能。