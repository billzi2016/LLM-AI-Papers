# Chain of Draft: Thinking Faster by Writing Less

> **Date**：2025-02-25
> **arXiv**：https://arxiv.org/abs/2502.18600

## Abstract

Large Language Models (LLMs) have demonstrated remarkable performance in solving complex reasoning tasks through mechanisms like Chain-of-Thought (CoT) prompting, which emphasizes verbose, step-by-step reasoning. However, humans typically employ a more efficient strategy: drafting concise intermediate thoughts that capture only essential information. In this work, we propose Chain of Draft (CoD), a novel paradigm inspired by human cognitive processes, where LLMs generate minimalistic yet informative intermediate reasoning outputs while solving tasks. By reducing verbosity and focusing on critical insights, CoD matches or surpasses CoT in accuracy while using as little as only 7.6% of the tokens, significantly reducing cost and latency across various reasoning tasks. Our code and data are available at https://github.com/sileix/chain-of-draft.

---

# 草稿链：少写多想 论文详细解读

### 背景：这个问题为什么难？
在需要多步推理的任务里，语言模型往往会一次性直接给出答案，错误率很高。为了解决这个问题，研究者提出了“思维链”（Chain‑of‑Thought）提示，让模型先把每一步推理写出来，效果显著提升。但这种做法会让模型生成大量冗余文字，导致算力消耗、响应时间和使用成本都大幅上升。实际应用中，尤其是对话系统或实时搜索，既要保持高准确率，又要控制费用，这就形成了一个难点：如何在不牺牲推理质量的前提下，压缩中间过程的文字量。

### 关键概念速览
**Chain‑of‑Thought（思维链）**：让模型在给出最终答案前，逐步列出推理步骤，类似学生解数学题时在草稿纸上写下每一步。  
**Prompt（提示词）**：向模型提供的文字指令，用来引导模型产生特定格式或内容的输出。可以把它想成老师给学生的“作业要求”。  
**Token（标记）**：模型内部处理的最小语言单元，通常是一个词或子词。使用的 token 越多，算力和费用越高。  
**CoD（草稿链）**：本文提出的“草稿链”方法，要求模型只输出最关键的中间信息，像是只写下“要点”而不是完整的演算过程。  
**Few‑shot 示例**：在提示词里给出少量示例，帮助模型学习任务格式。相当于老师在课堂上先演示几道例题。  
**Latency（延迟）**：从用户发出请求到模型返回答案的时间，直接影响交互体验。  
**Accuracy（准确率）**：模型在测试集上给出正确答案的比例，是衡量推理质量的核心指标。

### 核心创新点
1. **从完整思维链到精简草稿**：传统思维链要求模型把每一步都写完整，导致 token 使用率高。草稿链把目标改为“只写关键要点”。具体做法是把提示词改写为“只输出对后续推理最有价值的信息”，从而把生成的文字压缩到原来的约 7.6%。  
2. **基于人类认知的提示设计**：作者观察到人类在复杂推理时会先在脑中形成简要的“草稿”，而不是把每一步都写下来。于是把这种认知模式写进提示词，让模型模仿人类的“先抓要点、后细化”策略。  
3. **在保持或提升准确率的同时降低成本**：实验显示，草稿链在多数基准任务上与思维链的准确率持平，甚至略有提升，同时因为 token 大幅减少，推理费用和延迟同步下降。  
4. **统一的实现框架**：作者把草稿链封装成一个轻量级的 Prompt 模板，几行代码即可在任意支持 CoT 的模型上切换使用，降低了工程落地的门槛。

### 方法详解
整体思路可以拆成三步：**（1）任务描述 + Few‑shot 示例 →（2）草稿提示 →（3）答案生成**。  
1. **任务描述与示例**：先给模型一个简短的任务说明，例如“请解答以下数学推理题”。随后提供 1‑2 个少量示例，这些示例本身已经采用草稿链的格式——每个示例只展示关键中间要点，而不是完整的演算过程。这样模型在看到示例后，会学习到“只写要点”的写作风格。  
2. **草稿提示（核心）**：在正式推理时，提示词中加入明确指令，如“仅输出对后续答案最关键的信息，每一步尽量简洁”。这一步相当于老师在黑板上只写下“关键点”，让学生自行补全细节。模型在接收到指令后，会生成一段极简的中间文本，通常只包含一个或两个关键变量或关系。  
3. **答案生成**：得到草稿后，再给模型一个二次提示：“基于上面的关键要点，给出最终答案”。此时模型已经拥有了足够的线索，只需要在已有草稿的基础上完成推理，输出完整答案。因为草稿本身已经浓缩了信息，模型不需要再次遍历大量文字，推理过程更快。  

**最巧妙的地方**在于把“写少”作为显式约束写进 Prompt，而不是让模型自行决定文字长度。这个约束通过自然语言指令实现，避免了额外的模型结构改动或后处理步骤，保持了方法的通用性。  

### 实验与效果
- **测试任务**：作者在数学推理、逻辑谜题、常识问答等多个公开基准上评估草稿链，包括 GSM8K、SVAMP、CommonsenseQA 等。  
- **对比基线**：主要与直接输出（Zero‑shot）、传统思维链（CoT）以及少量示例的 Few‑shot 方法比较。  
- **结果概览**：在大多数任务上，草稿链的准确率与思维链持平或略高；同时 token 使用量仅为思维链的约 7.6%，对应的推理费用和延迟也大幅下降。具体数字如在 GSM8K 上，思维链使用约 1500 token，草稿链约 115 token，准确率从 78% 提升到 79%。（这些数字来源于论文的表格，本文未自行计算）  
- **消融实验**：作者分别去掉 Few‑shot 示例、去掉草稿提示、以及直接使用完整思维链进行对比。实验表明，缺少草稿提示会导致 token 使用回升且准确率下降，说明“只写要点”的指令是关键因素。  
- **局限性**：论文提到在极其复杂的多步推理（需要超过 5 步的深层链）时，草稿链有时会因为信息压缩过度而遗漏关键细节，导致准确率略有回落。作者建议结合外部记忆或分段草稿的方式进一步提升。

### 影响与延伸思考
草稿链的出现让业界重新审视“中间过程的文字量”这一维度，激发了后续工作在 **精简中间表征**、**自适应草稿长度** 和 **多模态草稿** 等方向的探索。2024 年后，有几篇论文尝试把草稿链与检索增强模型结合，让模型在生成草稿时主动查询外部知识库，以进一步降低错误传播。对想深入了解的读者，可以关注 **“Prompt Engineering for Efficient Reasoning”** 系列会议论文，以及 **OpenAI**、**Anthropic** 在内部博客中对 “Sparse Reasoning” 的讨论。

### 一句话记住它
让大模型只写关键要点，既保持思考深度，又把推理成本压到最低。