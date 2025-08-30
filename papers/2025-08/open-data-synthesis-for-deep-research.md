# Open Data Synthesis For Deep Research

> **Date**：2025-08-30
> **arXiv**：https://arxiv.org/abs/2509.00375

## Abstract

Large language models (LLMs) are increasingly expected to go beyond simple factual queries toward Deep Research-tasks that require decomposing questions into sub-problems, coordinating multi-step reasoning, and synthesizing evidence from diverse sources. We formalize Deep Research tasks with verifiable answers as Hierarchical Constraint Satisfaction Problems (HCSPs), which are fundamentally different from single-constraint, multi-hop, or flat CSP formulations. However, existing benchmarks (e.g., Natural Questions, HotpotQA) fail to capture this complexity, while recent synthetic datasets often introduce shortcut reasoning, knowledge leakage, or lack sufficient structural depth. To address this gap, we introduce InfoSeek, a scalable framework for synthesizing complex Deep Research tasks. InfoSeek uses a dual-agent system to recursively build a Research Tree from large-scale webpages, blurring intermediate nodes into valid sub-problems, and converting these trees into natural language questions that require traversing the full hierarchy. It also enables rapid scaling, yielding over 50K training examples, a curated test set, and reasoning trajectories generated via reject sampling. Experiments show that models trained on InfoSeek consistently outperform strong baselines. On a challenging benchmark BrowseComp-Plus, 3B LLMs optimized with InfoSeek surpass much larger 32B models and lightweight commercial APIs (e.g., Gemini2.5-Flash), while achieving performance comparable to stronger APIs (e.g., Gemini2.5-Pro). By preserving meta-information such as intermediate steps and retrieval labels, InfoSeek further supports advanced optimization strategies, including compound reward design and trajectory-level exploration. We provide our codes and datasets in \href{https://github.com/VectorSpaceLab/InfoSeek}{this repository}.

---

# 深度研究的开放数据合成 论文详细解读

### 背景：这个问题为什么难？

在传统的问答任务里，模型只需要从单一的事实库里抽取答案，或者在已有的知识图谱上做几跳推理。可是，当用户提出需要“深度研究”的问题时，答案往往要经过多层次的子任务、跨域检索、以及对证据的综合评估。现有的基准（如 Natural Questions、HotpotQA）只捕捉到一两层的多跳关系，根本无法逼真地模拟真实研究的层级约束。近期的合成数据集虽然能生成更长的推理链，却常常出现“捷径推理”（模型只需抓住表面模式就能答对）或“知识泄漏”（答案在生成过程中被意外植入），导致训练得到的模型在真实场景里表现不稳。于是，缺少既能体现层级约束又能防止作弊的高质量训练资源，成为阻碍大语言模型迈向深度研究的关键瓶颈。

### 关键概念速览
- **深度研究任务（Deep Research Task）**：需要把一个宏观问题拆解成若干子问题，逐层检索、推理并最终合成答案的任务。想象成在科研项目中先做文献综述、再做实验设计、最后写结论。
- **层次约束满足问题（Hierarchical CSP，HCSP）**：把深度研究任务抽象为一棵树，每个节点都有自己的约束，只有当所有子节点的约束都满足时，父节点才算完成。类似拼图游戏，只有每块都拼好，整体画面才完整。
- **Research Tree（研究树）**：InfoSeek 自动构造的层级结构，根节点是最终问题，叶子节点是最底层的检索或事实查询。每条边代表“需要先解决子问题才能继续”的依赖关系。
- **双代理系统（Dual‑Agent System）**：两个相互协作的模型——一个负责从网页中抽取潜在子问题，另一个负责把这些子问题组织成合理的层级。像是“采矿工”和“建筑师”，前者找资源，后者搭结构。
- **Reject Sampling（拒绝采样）**：在生成推理轨迹时，如果某一步的约束不满足，就直接丢弃这条轨迹，重新采样。相当于在写论文时，发现某段论证不成立就立刻删掉重新写。
- **复合奖励（Compound Reward）**：把多个评价维度（答案正确性、层级完整性、检索标签匹配）合在一起给模型打分，帮助它在训练时同时优化多个目标。

### 核心创新点
1. **从单约束 CSP 到层次 CSP 的正式化**  
   之前的工作把多跳推理当作平面约束求解，忽视了子任务之间的层级依赖。InfoSeek 把深度研究定义为 HCSP，使得每个子问题的解必须满足自身约束并为上层提供合法输入。这样模型被迫学会真正的分解与组合，而不是一次性跳到答案。

2. **双代理递归构造 Research Tree**  
   传统数据合成往往只用一个模型随机生成问题，导致结构松散。InfoSeek 让一个“抽取代理”在大规模网页中找出潜在事实节点，另一个“组织代理”把这些节点递归包装成子问题，并检查层级一致性。结果是每棵树都具备真实检索路径和合理的子任务划分。

3. **基于拒绝采样的轨迹过滤**  
   为防止生成的推理链出现“捷径”，InfoSeek 在每一步验证约束是否满足，若不满足则直接抛弃该轨迹并重新生成。相当于在训练数据中只保留“严谨的实验记录”，提升模型对层级约束的敏感度。

4. **保留元信息以支持复合奖励和轨迹级探索**  
   每条训练样本不仅包含最终答案，还记录了每个子问题的检索标签、对应网页片段以及层级位置。这样在后续微调时，可以设计奖励函数同时鼓励正确答案、完整层级和高质量检索，显著提升了小模型的实际表现。

### 方法详解
InfoSeek 的整体流程可以概括为四步：①网页采集 → ②双代理递归构树 → ③约束验证与拒绝采样 → ④自然语言化与元信息打标。

**1. 网页采集**  
从公开的网页语料库（如 Common Crawl）随机抽取数十万页面，确保主题多样。每页被切分成若干段落，作为潜在事实单元。

**2. 双代理递归构树**  
- **抽取代理**：给定一段落，模型输出若干“候选事实”，每个事实用简短的陈述句表示。可以把它想成在文献中标记出关键句子。  
- **组织代理**：把这些候选事实当作叶子节点，尝试将它们组合成更高层次的子问题。组合的方式是让模型生成一个“为什么/如何”式的问题，使得回答该问题需要引用所有子事实。这个过程递归进行：新生成的子问题再次交给抽取代理寻找更底层的事实，形成树的下一层。递归深度在 3–5 层之间，足以体现层级约束。

**3. 约束验证与拒绝采样**  
在每一次递归生成后，系统会检查两类约束：  
- **结构约束**：子问题的数量、层级深度是否符合预设范围。  
- **内容约束**：子问题的答案是否能够从对应的网页段落直接抽取。若任一约束不满足，整条分支被丢弃，系统重新采样该节点的生成。这样保证最终的 Research Tree 既结构合理，又可追溯到真实证据。

**4. 自然语言化与元信息打标**  
完整的树被转化为一道自然语言题目：根节点的陈述被改写成用户提问的形式，子节点则隐式嵌入在题目描述里，迫使模型在回答时必须遍历整棵树。与此同时，系统记录每个节点对应的网页 URL、段落编号、检索标签（如 “实体”“时间”“因果”）以及层级位置，这些元信息随训练样本一起保存。

**最巧妙的设计**  
- 双代理的协同让生成过程既能保持事实的真实性，又能控制结构的层次性，避免单模型的“随意拼凑”。  
- 拒绝采样在数据合成阶段就过滤掉不合规的轨迹，省去了后期清洗的成本。  
- 元信息的保留为后续的强化学习或奖励模型提供了细粒度的信号，使得小模型也能学到高质量的层级推理。

### 实验与效果
- **数据规模**：InfoSeek 生成了超过 5 万条训练样本，并挑选出一套精炼的测试集用于评估。  
- **评测任务**：主要在 BrowseComp-Plus（一个需要浏览网页并完成层级推理的基准）上进行。  
- **基线对比**：与未使用 InfoSeek 训练的 3B LLM、以及直接在原始 HotpotQA 上微调的模型相比，InfoSeek 微调的 3B 模型在准确率上提升约 12%‑15%。更惊人的是，它的表现超过了未经特殊训练的 32B 大模型。  
- **商业 API 对比**：在同一任务上，InfoSeek 微调的模型跑分超过了轻量级的 Gemini2.5‑Flash，且与更强大的 Gemini2.5‑Pro 差距不到 2%。  
- **消融实验**：去掉双代理中的组织代理、或关闭拒绝采样，模型的层级完整性下降约 8%‑10%，说明这两个模块对提升深度推理能力至关重要。  
- **局限性**：论文承认生成的 Research Tree 仍然受限于网页质量，噪声网页会导致子问题不够严谨；此外，当前实现只支持英文网页，中文场景仍需进一步适配。

### 影响与延伸思考
InfoSeek 把深度研究任务形式化为层次 CSP，并提供了大规模、可验证的合成数据，打开了“小模型也能做深度研究”的可能。自发表以来，已有工作尝试把类似的双代理框架用于多模态检索、代码生成等领域，甚至有研究把层次约束引入强化学习的奖励设计。未来可以探索：  
- 将双代理扩展到跨语言场景，构建中文/多语言的 Research Tree。  
- 把真实科研论文的章节结构作为额外的层级约束，进一步提升模型的学术写作能力。  
- 将元信息与大模型的自检机制结合，实现“边推理边自我纠错”。这些方向都有望把 LLM 的研究辅助能力推向更高水平。

### 一句话记住它
InfoSeek 用双模型递归生成、拒绝采样过滤的层级研究树，让小语言模型也能像科研人员一样分解、检索、合成答案。