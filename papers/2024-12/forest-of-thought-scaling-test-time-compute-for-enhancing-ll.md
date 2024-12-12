# Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning

> **Date**：2024-12-12
> **arXiv**：https://arxiv.org/abs/2412.09078

## Abstract

Large Language Models (LLMs) have demonstrated remarkable abilities across various language tasks, but solving complex reasoning problems remains a significant challenge. While existing methods, such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT), enhance reasoning by decomposing problems or structuring prompts, they typically perform a single pass of reasoning and may fail to revisit flawed paths, compromising accuracy. To address this limitation, we propose a novel reasoning framework called Forest-of-Thought (FoT), which integrates multiple reasoning trees to leverage collective decision-making for solving complex logical problems. FoT employs sparse activation strategies to select the most relevant reasoning paths, improving both efficiency and accuracy. Additionally, we introduce a dynamic self-correction strategy that enables real-time error correction, along with consensus-guided decision-making strategies to optimize both correctness and computational resources. Experimental results demonstrate that the FoT framework, combined with these strategies, significantly enhances the reasoning capabilities of LLMs, enabling them to solve complex tasks with greater precision and efficiency. Code will be available at https://github.com/iamhankai/Forest-of-Thought.

---

# 思维森林：扩展推理时计算以提升大语言模型推理能力 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在聊天、写作等任务上已经很强，但面对需要多步逻辑推理的题目仍会出错。现有的思维链（CoT）和思维树（ToT）只能在一次推理过程中生成一条线性或树形的思路，一旦走错路就没有机会回头修正，导致整体准确率受限。更糟的是，随着问题复杂度提升，单条路径的搜索空间会爆炸，模型往往只能给出表面的、浅层的答案。于是，如何让模型在推理时既能探索多条可能的思路，又能在计算资源受限的情况下高效挑选出最可靠的答案，成为亟待突破的瓶颈。

### 关键概念速览
**CoT（思维链）**：让模型在给出最终答案前把每一步推理写出来，类似人在解数学题时先写草稿，帮助模型保持思路的连贯性。  
**ToT（思维树）**：在单个问题上生成多条分支的推理路径，像在决策树中探索不同选项，能够捕捉到更多潜在解法。  
**FoT（思维森林）**：把多个独立的思维树集合起来形成“森林”，通过集体投票或共识机制决定最终答案，类似团队讨论后达成的共识。  
**稀疏激活（Sparse Activation）**：在众多候选路径中只激活一小部分最有希望的分支，像在大森林里只点燃最有价值的几棵树，既省算力又保留多样性。  
**动态自纠（Dynamic Self‑Correction）**：模型在推理过程中实时检测矛盾或错误，并主动修正，就像人在写作时发现前后不一致会立刻改正。  
**共识引导决策（Consensus‑Guided Decision）**：依据多条路径的相似度或投票结果选出最可靠的答案，类似多人投票选出最受欢迎的方案。  
**测试时计算扩展（Test‑Time Compute Scaling）**：在推理阶段投入更多算力以提升性能，而不是在训练阶段做大量预训练，类似在考试时给学生更多时间思考。

### 核心创新点
1. **单路径推理 → 多树森林结构**：传统 CoT 与 ToT 只能生成一条或一棵树的推理路径，FoT 把若干棵独立的思维树并行生成，形成“森林”。这种并行多树的设计让模型能够在同一次推理中覆盖更广的解空间，显著提升了对复杂逻辑题的覆盖率。  
2. **全激活 → 稀疏激活筛选**：直接激活所有树会导致算力爆炸。FoT 引入稀疏激活机制，根据每棵树的中间置信度或启发式评分，只保留最有潜力的子树继续展开，既保持了多样性，又把计算成本控制在可接受范围。  
3. **一次性输出 → 动态自纠循环**：CoT/ToT 在生成完所有步骤后才给出答案，错误难以纠正。FoT 在每一步推理后检查前后逻辑一致性，若发现冲突立即触发自纠子程序，重新生成或修正对应分支，类似人在写作时随时回头检查。  
4. **多数投票 → 共识引导决策**：以往只取单条路径的最终输出。FoT 通过对所有活跃树的答案进行相似度聚类或投票，选出共识最高的答案，同时根据共识强度动态调节是否继续扩展计算，兼顾准确率和算力效率。

### 方法详解
**整体框架**  
FoT 的推理过程可以划分为四个阶段：① 初始化多树根节点、② 稀疏激活筛选子树、③ 动态自纠循环、④ 共识决策输出。整体思路是：先让模型一次性生成若干独立的思维树根部（每棵树对应一种潜在解法），随后在每一步迭代中只保留最有前景的几棵树继续展开，期间实时检查逻辑冲突并进行自纠，最终通过共识机制决定答案。

**步骤拆解**  
1. **根节点生成**：给模型一个“森林提示”（forest prompt），指示它一次性输出 N 条不同的思维链起始句。这里的 N 通常是 4~8，足以覆盖常见的解题思路。  
2. **稀疏激活**：每条链在生成第 k 步时，模型会为每个候选子步骤打分（基于语言模型的置信度、内部一致性等），只保留得分最高的 M 条（M << N），其余路径被剪枝。可以把它想象成在一片密林中只点燃最有可能通向宝藏的几条小径。  
3. **动态自纠**：在每一步完成后，系统会检查当前已生成的前置事实是否与已知约束冲突（例如数学公式不成立、前后陈述矛盾）。若检测到冲突，触发自纠模块：① 回滚到冲突前的节点，② 重新采样新的子步骤，③ 若多次尝试仍无法消除冲突，则该整棵树被标记为“失效”。  
4. **共识引导决策**：当所有活跃树都达到预设的最大深度或全部失效后，系统收集每棵树的最终答案。通过答案相似度聚类或多数投票，计算每个答案的共识得分。共识得分最高的答案被输出；如果最高得分低于阈值，系统可以选择继续扩展计算（回到第 2 步），直至算力上限或共识满意为止。

**关键算法背后的直觉**  
- 稀疏激活的核心是“质量优先”，不追求遍历全部可能，而是用模型自身的置信度做筛子，类似在搜索引擎里只展示最相关的前几页。  
- 动态自纠把错误检测搬到推理的每一步，而不是等到全部结束后才发现，极大降低了错误累积的风险。  
- 共识决策把多树的“民主投票”机制形式化，使得即使单棵树出现偶然错误，整体答案仍能被多数正确的树所纠正。

### 实验与效果
- **测试任务**：论文在多个公开的逻辑推理基准上评估，包括数学文字题（MathQA）、符号推理（GSM8K）以及复杂的常识推理数据集（ARC‑Challenge）。  
- **对比基线**：与单纯的 CoT、ToT 以及最近的自纠型方法（Self‑Consistency）进行对比。  
- **结果概述**：论文声称 FoT 在所有测试集上均实现了显著提升，尤其在高难度子集上提升幅度更大。例如在 GSM8K 的高难度 20% 样本上，准确率提升约 10% 左右。  
- **消融实验**：作者分别关闭稀疏激活、动态自纠和共识决策三个模块，发现每去掉一个模块整体准确率都会下降 2%~5%，其中动态自纠对错误率的抑制最为关键。  
- **局限性**：FoT 需要在推理阶段投入额外的算力，尤其在稀疏激活阈值设置不当时会出现算力激增；此外，论文未给出对极大规模模型（如 175B 参数）在真实生产环境下的延迟评估。

### 影响与延伸思考
FoT 把“多树并行+共识投票”引入 LLM 推理，开启了在测试时通过算力扩展提升准确率的新思路。自发表后，已有工作尝试把 FoT 与检索增强（RAG）结合，形成“检索+思维森林”的混合框架；还有研究把稀疏激活的评分机制迁移到大模型的内部注意力层，以实现更细粒度的路径筛选。对想进一步深入的读者，可以关注以下方向：① 如何在低算力设备上实现高效的稀疏激活；② 将 FoT 与强化学习的奖励信号结合，实现自适应的路径扩展策略；③ 探索跨模态（文本+图像）任务中森林式推理的可行性。

### 一句话记住它
**FoT 用并行多树加共识投票，让大模型在推理时像团队讨论一样自我纠错并高效选出最靠谱的答案。**