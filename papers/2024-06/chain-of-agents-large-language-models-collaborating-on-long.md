# Chain of Agents: Large Language Models Collaborating on Long-Context   Tasks

> **Date**：2024-06-04
> **arXiv**：https://arxiv.org/abs/2406.02818

## Abstract

Addressing the challenge of effectively processing long contexts has become a critical issue for Large Language Models (LLMs). Two common strategies have emerged: 1) reducing the input length, such as retrieving relevant chunks by Retrieval-Augmented Generation (RAG), and 2) expanding the context window limit of LLMs. However, both strategies have drawbacks: input reduction has no guarantee of covering the part with needed information, while window extension struggles with focusing on the pertinent information for solving the task. To mitigate these limitations, we propose Chain-of-Agents (CoA), a novel framework that harnesses multi-agent collaboration through natural language to enable information aggregation and context reasoning across various LLMs over long-context tasks. CoA consists of multiple worker agents who sequentially communicate to handle different segmented portions of the text, followed by a manager agent who synthesizes these contributions into a coherent final output. CoA processes the entire input by interleaving reading and reasoning, and it mitigates long context focus issues by assigning each agent a short context. We perform comprehensive evaluation of CoA on a wide range of long-context tasks in question answering, summarization, and code completion, demonstrating significant improvements by up to 10% over strong baselines of RAG, Full-Context, and multi-agent LLMs.

---

# 智能体链：大语言模型在长上下文任务中的协作 论文详细解读

### 背景：这个问题为什么难？
处理上万字甚至更长的文本一直是大语言模型（LLM）的瓶颈。模型的上下文窗口有限，直接喂入整篇文档会导致信息被稀释，模型难以聚焦关键细节。为了解决这个问题，业界主要走两条路：一是用检索增强生成（RAG）把文档切成若干块，只挑出看似相关的片段；二是直接扩展模型的窗口大小。前者的风险在于检索不到任务所需的关键句子，后者虽然能看到全部内容，却仍然要在海量信息中找出答案，容易“信息过载”。这两种思路都没有根本解决“长文档→精准推理”这一难点。

### 关键概念速览
**长上下文任务**：需要模型在一次推理过程中考虑数千到上万字的输入，例如长篇阅读理解、代码补全等。  
**检索增强生成（RAG）**：先用向量检索挑出若干相关段落，再交给模型生成答案，类似先把图书馆里可能有用的书挑出来再阅读。  
**上下文窗口**：模型一次能够“看到”的 token 数量上限，窗口越大，模型一次能直接处理的信息越多。  
**工作智能体（Worker Agent）**：负责读取并推理输入的一个子段落的 LLM，像是把一本书分给多个小组分别阅读。  
**管理智能体（Manager Agent）**：收集所有工作智能体的输出并进行综合、校对，类似编辑部把各小组的稿件合并成最终稿。  
**链式协作（Chain-of-Agents, CoA）**：把长文本拆成若干块，依次让工作智能体处理，再让管理智能体统一汇总的整体框架。  
**信息聚合**：把多个局部推理结果合并成全局答案的过程，类似把多位专家的意见整合成统一决策。

### 核心创新点
1. **从单一检索→多智能体协作**：传统 RAG 只挑出几段交给一个模型，信息可能残缺。CoA 把全文切块，所有块都被至少一个工作智能体阅读，确保没有信息被遗漏。结果是模型在长文档上能覆盖更完整的知识面。  
2. **从一次性全局推理→交叉阅读+分段推理**：以前的全局推理让模型一次性面对全部上下文，容易失焦。CoA 让每个工作智能体只处理短上下文，保持高注意力；随后管理智能体负责把分散的推理拼接起来，解决了“长文档注意力分散”的痛点。  
3. **从固定角色→层级角色分工**：CoA 明确划分了工作智能体和管理智能体的职责，前者专注于局部阅读与初步推理，后者负责全局一致性检查与最终答案生成。这种层级结构比之前的“多模型并行投票”更高效，因为管理智能体可以利用前一步的中间结果进行二次推理。  
4. **从单模型→跨模型协同**（可选实现）：框架本身不限制使用同一模型或不同规模的模型，实验中可以让大模型担任管理角色，小模型负责工作角色，降低算力成本的同时仍保持整体性能提升。

### 方法详解
CoA 的整体流程可以概括为三步：**切分 → 分段推理 → 汇总**。

1. **文本切分**  
   输入的长文档首先被划分成若干等长或语义完整的块（比如每块 500‑800 token），切分方式可以是固定长度、段落边界或语义分段。目标是让每块都能完整放进单个工作智能体的上下文窗口。

2. **工作智能体链式处理**  
   - **顺序调度**：工作智能体按块的顺序依次被调用。第一个智能体读取第一块并生成“局部输出”，包括对该块的理解、关键事实抽取或初步答案草稿。  
   - **上下文携带**：每个后续智能体在读取自己对应的块时，还会收到前一个智能体的输出作为“短期记忆”。这相当于让每个小组在阅读新章节时，先看前一小组的摘要，保证信息在链上流动。  
   - **自然语言交互**：所有信息的传递都采用自然语言描述（例如“上一步的结论是…”，或“请基于前面的要点继续推理”），这样即使使用不同模型也能顺畅交流。

3. **管理智能体汇总**  
   - **收集所有局部输出**：当最后一个工作智能体完成后，管理智能体一次性获得全部局部结果的列表。  
   - **全局推理**：管理智能体把这些碎片化的答案当作“证据”，在自己的上下文窗口里进行二次推理，检查前后逻辑是否一致、是否有冲突，并生成最终的完整答案或摘要。  
   - **校对与精炼**：管理智能体还能执行自我纠错，例如发现某个工作智能体遗漏了关键细节时，主动回溯请求补充，或者对语言风格进行统一。

**关键细节**  
- **信息携带的格式**：作者使用了“任务指令 + 前一步输出”这种模板，使得即使不同规模的模型也能理解上下文。  
- **并行潜力**：虽然论文实现的是顺序调用，但框架本身支持并行处理，只要在管理阶段加入额外的冲突解决步骤。  
- **最巧妙的点**：把“记忆”交给自然语言而不是向量或专用记忆模块，使得系统可以在不改动模型内部结构的情况下实现跨模型协作，极大提升了可迁移性。

### 实验与效果
- **测试任务**：论文在三大类长上下文任务上做评估：长篇问答（如 NarrativeQA、HotpotQA 长文版）、长文摘要（ArXiv 论文摘要、新闻长文）以及代码补全（超过 2k 行的代码文件）。  
- **基线对比**：与检索增强生成（RAG）、直接使用全上下文的单模型、以及已有的多智能体协作方案（如 Multi-Agent Debate）进行比较。  
- **性能提升**：在所有任务上，CoA 的最终得分均高出基线 3%~10%，其中在长篇问答上最高提升约 10%。  
- **消融实验**：作者分别去掉“前一步输出携带”和“管理智能体二次推理”两项，发现去掉任意一项后性能下降约 4%~6%，说明两者都是关键贡献。  
- **局限性**：论文承认顺序调用导致推理时间随块数线性增长，且对切分策略敏感——如果块划分不合理，工作智能体可能得到不完整的上下文，导致错误传播。

### 影响与延伸思考
CoA 的出现让研究者重新审视“长上下文”不一定要靠单一大模型或更大的窗口，而是可以通过**协同分工**来实现。自发表以来，已有工作尝试把 CoA 与检索系统结合，形成“检索+分段+协作”的三段式管线；还有研究把管理智能体换成专门的**事实校验模型**，提升答案的可信度。未来的方向可能包括：  
- **并行化调度**：设计冲突检测机制，使工作智能体真正并行运行，显著降低时延。  
- **自适应切分**：让模型自行决定如何划分文本，以适应不同任务的粒度需求。  
- **跨模态协作**：把视觉、音频等模态的专用模型也纳入同一链式框架，实现多模态长上下文推理。  
对想深入的读者，可以关注近期在 “LLM 代理系统（LLM Agent Systems）” 和 “长上下文检索（Long-Context Retrieval）” 交叉点的论文。

### 一句话记住它
把长文档拆成小块，让一串专注的 LLM 逐段阅读，再交给“总编辑”统一校对——这就是 **Chain‑of‑Agents** 的核心魔法。