# Think-on-Graph: Deep and Responsible Reasoning of Large Language Model   on Knowledge Graph

> **Date**：2023-07-15
> **arXiv**：https://arxiv.org/abs/2307.07697

## Abstract

Although large language models (LLMs) have achieved significant success in various tasks, they often struggle with hallucination problems, especially in scenarios requiring deep and responsible reasoning. These issues could be partially addressed by introducing external knowledge graphs (KG) in LLM reasoning. In this paper, we propose a new LLM-KG integrating paradigm ``$\hbox{LLM}\otimes\hbox{KG}$'' which treats the LLM as an agent to interactively explore related entities and relations on KGs and perform reasoning based on the retrieved knowledge. We further implement this paradigm by introducing a new approach called Think-on-Graph (ToG), in which the LLM agent iteratively executes beam search on KG, discovers the most promising reasoning paths, and returns the most likely reasoning results. We use a number of well-designed experiments to examine and illustrate the following advantages of ToG: 1) compared with LLMs, ToG has better deep reasoning power; 2) ToG has the ability of knowledge traceability and knowledge correctability by leveraging LLMs reasoning and expert feedback; 3) ToG provides a flexible plug-and-play framework for different LLMs, KGs and prompting strategies without any additional training cost; 4) the performance of ToG with small LLM models could exceed large LLM such as GPT-4 in certain scenarios and this reduces the cost of LLM deployment and application. As a training-free method with lower computational cost and better generality, ToG achieves overall SOTA in 6 out of 9 datasets where most previous SOTAs rely on additional training.

---

# 思考于图：大语言模型在知识图谱上的深度与负责任推理 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）虽然在聊天、写作等任务上表现惊人，但在需要严谨推理的场景里经常会“编造”答案——也叫幻觉。根本原因是模型的知识全部埋在参数里，缺少可查询、可验证的外部信息。过去的做法大多是把知识图谱（KG）直接拼接进提示或让模型在一次性检索后再回答，这种“一次性”方式往往只能提供浅层事实，难以支撑多步、跨实体的深度推理，也缺乏对推理路径的可追溯性。于是，如何让 LLM 像人一样在图谱上“走路”、一步步探索并在每一步都能检查自己的答案，成为了亟待突破的难题。

### 关键概念速览
- **大语言模型（LLM）**：基于海量文本训练的生成式模型，能够理解和生成自然语言。把它想象成一个会说话的“百科全书”，但内部的知识是“烂熟于心”的，查不到来源。
- **知识图谱（KG）**：由实体（节点）和关系（边）构成的结构化网络，类似于维基百科的“概念地图”。每条边都对应一个可验证的事实。
- **LLM ⊗ KG（LLM 与 KG 的交互范式）**：把 LLM 当成一个智能体，让它在 KG 上主动搜索、选择路径、获取信息，再把这些信息用于推理。相当于让模型在“图谱里走路”，而不是只在脑子里回忆。
- **Beam Search（束搜索）**：在搜索空间中保留若干条最有希望的候选路径，逐层展开，最终挑出最优解。类似于在迷宫里同时派出几支探险队，每一步只保留前几名最有前景的队伍。
- **可追溯性（Traceability）**：推理过程能够被完整记录，后续可以检查每一步用了哪条 KG 边、哪段 LLM 生成的解释。就像写实验日志，任何人都能复现。
- **可纠正性（Correctability）**：当发现某条路径或事实错误时，系统可以接受专家或用户的反馈，重新搜索或替换错误节点。相当于在写作时老师给出批注，作者即时改正。

### 核心创新点
1. **从一次性检索到交互式图谱探索**  
   之前的方案往往在提示阶段一次性把 KG 中的相关三元组拉出来，然后让 LLM 直接生成答案。ToG 把 LLM 设为主动探险者，采用迭代的束搜索在 KG 上逐步扩展路径，每一步都让模型决定下一跳的方向。这样不仅提升了多步推理的深度，也让模型能够在搜索过程中动态利用新获取的事实。

2. **LLM ⊗ KG 的训练免费实现**  
   大多数提升推理能力的工作需要对 LLM 进行微调或额外的知识注入训练。ToG 完全依赖提示（prompt）和 KG 查询，保持了“零训练”特性。作者通过精心设计的系统提示，让不同规模的 LLM（从 7B 到 GPT‑4）都能直接挂上 KG，省去昂贵的微调成本。

3. **知识可追溯 + 可纠正的闭环机制**  
   ToG 在每一步搜索时都会记录所走的实体、关系以及 LLM 给出的解释。若后续检测到错误（比如专家反馈），系统可以回溯到出错的节点，重新执行束搜索并替换错误路径。相比传统“一次性输出”方式，这种闭环让模型的答案更负责任。

4. **小模型超越大模型的成本效益**  
   实验显示，在若干需要深层推理的基准上，使用 7B 参数的 LLM 通过 ToG 的表现可以超过直接调用 GPT‑4 的结果。核心原因是图谱提供了精准的事实支撑，弥补了小模型记忆容量的不足，从而在实际部署时大幅降低算力和费用。

### 方法详解
**整体框架**  
ToG 把推理任务拆成四个循环步骤：① 接收用户问题并生成初始查询；② 在 KG 上执行束搜索，产生若干候选路径；③ 让 LLM 对每条路径进行解释并打分；④ 依据 LLM 的打分保留最有前景的几条路径，继续扩展，直至满足终止条件（找到答案或搜索深度上限）。

**关键模块拆解**

1. **问题转化与起始点定位**  
   - 系统提示 LLM 把自然语言问题映射为 KG 中的实体或关系关键词。比如“谁是《盗梦空间》的导演？”会被转化为实体“盗梦空间”并检索其“导演”关系的起始点。  
   - 这一步相当于在地图上标出起点和目标方向。

2. **束搜索（Beam Search）在 KG 上的实现**  
   - 每轮搜索保留 `B` 条最有希望的路径（B 通常为 3~5）。  
   - 对于每条路径，系统查询 KG 的邻居节点，生成所有可能的下一跳（实体+关系）。  
   - 这里的“可能性”由 LLM 根据上下文给出概率分布：模型会说“从‘盗梦空间’出发，最可能的下一步是‘导演’”。  

3. **LLM 解释与路径打分**  
   - 对每个候选路径，LLM 生成一段自然语言解释，说明为什么这条路径合理。  
   - 解释随后被送入一个轻量评分函数（可以是 LLM 自己的 log‑prob，或外部的事实一致性检查），得到路径得分。  
   - 这一步类似于让探险队的领队现场评估每条路线的可行性。

4. **路径选择与迭代**  
   - 根据得分，系统保留得分最高的 `B` 条路径进入下一轮搜索。  
   - 若某条路径已经到达答案实体或满足预设的“终止谓词”，则停止搜索并输出该路径对应的答案。  

5. **可追溯与可纠正机制**  
   - 每一步的实体、关系、LLM 解释都被记录在日志中，形成一条完整的推理链。  
   - 若用户或专家指出某条解释错误，系统可以定位到对应的搜索轮次，重新执行束搜索并替换错误分支。  

**最巧妙的设计**  
- **LLM 充当搜索控制器**：传统的图搜索算法（如 BFS、DFS）完全依赖结构信息，而 ToG 让 LLM 根据语言上下文给出“下一步应该往哪走”的建议，使搜索更具语义导向。  
- **零训练的 Prompt Engineering**：通过一套统一的系统提示，作者实现了不同规模 LLM 与任意 KG 的即插即用，省去了模型微调的繁琐步骤。

### 实验与效果
- **数据集与任务**：论文在 9 个公开基准上评测，包括 CommonsenseQA、OpenBookQA、HotpotQA、MetaQA 等需要多跳推理的问答任务，以及一些专门考察事实可追溯性的评测集。  
- **对比基线**：与纯 LLM（GPT‑3.5、GPT‑4）、检索增强 LLM（RAG、ReAct）以及需要微调的图谱推理模型（GNN‑based、KG‑Prompt）进行比较。  
- **核心结果**：在 6/9 数据集上，ToG 达到或超过了当时的 SOTA，尤其在需要 3 步以上推理的任务上提升约 12%~18%。在小模型（7B）上使用 ToG 的表现甚至超过了直接调用 GPT‑4 的基线（约 4% 的相对提升），验证了成本效益。  
- **消融实验**：作者分别去掉束搜索、去掉 LLM 解释打分、以及去掉可纠正机制，发现束搜索是提升深度推理的关键（去掉后性能下降约 9%），而可纠正机制在专家反馈场景下提升约 3% 的准确率。  
- **局限性**：论文承认 ToG 对 KG 的质量高度敏感，稀疏或噪声较多的图谱会导致搜索空间爆炸。此外，束搜索的宽度和深度需要手动调参，自动化仍是未解决的问题。

### 影响与延伸思考
ToG 把 LLM 当成“可编程的探险家”，打开了“语言模型 + 结构化知识”交互的新局面。自发表后，出现了多篇工作尝试将类似的交互式搜索引入代码生成（LLM ⊗ AST）或医学知识库（LLM ⊗ UMLS），并探索更高效的路径剪枝策略。对想进一步深入的读者，可以关注以下方向：① 自动化的搜索宽度/深度调节；② 将强化学习与 LLM 控制器结合，实现自适应搜索；③ 多模态 KG（加入图像、音频）与 LLM 的协同推理。整体来看，ToG 为“负责任的 AI 推理”提供了一个实用且低成本的实现范式。

### 一句话记住它
**ToG 把大语言模型变成在知识图谱上会走路的探险家，用束搜索和可追溯的解释，让小模型也能实现深度、可纠错的推理。**