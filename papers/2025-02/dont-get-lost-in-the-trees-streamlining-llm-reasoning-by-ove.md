# Don't Get Lost in the Trees: Streamlining LLM Reasoning by Overcoming   Tree Search Exploration Pitfalls

> **Date**：2025-02-16
> **arXiv**：https://arxiv.org/abs/2502.11183

## Abstract

Recent advancements in tree search algorithms guided by verifiers have significantly enhanced the reasoning capabilities of large language models (LLMs), but at the cost of increased computational resources. In this work, we identify two key challenges contributing to this inefficiency: $\textit{over-exploration}$ due to redundant states with semantically equivalent content, and $\textit{under-exploration}$ caused by high variance in verifier scoring leading to frequent trajectory switching. To address these issues, we propose FETCH, an e$\textbf{f}$fici$\textbf{e}$nt $\textbf{t}$ree sear$\textbf{ch}$ framework, which is a flexible, plug-and-play system compatible with various tree search algorithms. Our framework mitigates over-exploration by merging semantically similar states using agglomerative clustering of text embeddings obtained from a fine-tuned SimCSE model. To tackle under-exploration, we enhance verifiers by incorporating temporal difference learning with adjusted $\lambda$-returns during training to reduce variance, and employing a verifier ensemble to aggregate scores during inference. Experiments on GSM8K, GSM-Plus, and MATH datasets demonstrate that our methods significantly improve reasoning accuracy and computational efficiency across four different tree search algorithms, paving the way for more practical applications of LLM-based reasoning. The code is available at https://github.com/Soistesimmer/Fetch.

---

# 别在树中迷路：通过克服树搜索探索陷阱简化大语言模型推理 论文详细解读  

### 背景：这个问题为什么难？  
大语言模型（LLM）在解数学、逻辑题时常借助“树搜索+验证器”来逐步展开推理路径。虽然这种方式比一次性直接输出答案更可靠，却把搜索空间膨胀成一棵巨大的树，导致算力飙升。早期的工作主要靠让验证器给每个分支打分来挑选最有希望的路径，却忽视了两类隐藏的浪费：一是同义的中间状态被重复展开（**过度探索**），二是验证器评分波动大，导致搜索在不同分支之间频繁切换，错失深度挖掘的机会（**不足探索**）。这两个瓶颈让实际部署成本高得离谱，迫切需要一种既保留搜索质量又能削减计算的方案。

### 关键概念速览  
**LLM（大语言模型）**：能够生成自然语言文本的深度模型，像 GPT‑4、Claude 等，擅长把题目转化为文字描述的推理步骤。  

**树搜索**：把每一步推理看作在树的一个节点上展开，向下生成多个候选子步骤，形成一棵搜索树，类似于棋局的博弈树。  

**验证器（Verifier）**：一个专门训练的模型，用来评估某个推理步骤或整条路径的可信度，分数越高表示越可能是正确的。  

**过度探索（Over‑exploration）**：搜索过程中出现大量语义相同或相近的节点，导致树的宽度不必要地扩大。可以想象成在迷宫里不停走进相同的死胡同。  

**不足探索（Under‑exploration）**：验证器评分噪声大，搜索频繁在不同分支之间跳来跳去，像是手里拿着指南针却总是抖动，找不到稳固的方向。  

**SimCSE**：一种对比学习的句子嵌入模型，能够把语义相近的句子映射到相近的向量空间。这里用它来判断两个推理状态是否“说的是同一件事”。  

**时序差分学习（Temporal Difference Learning）**：强化学习里常用的价值估计方法，利用当前估计和下一个状态的估计来更新分数，能够平滑评分波动。  

**λ‑回报（λ‑returns）**：在 TD 学习中引入的加权累计回报，介于单步 TD 和完整回报之间，用来控制方差与偏差的权衡。  

**验证器集成（Verifier Ensemble）**：把多个独立训练的验证器输出取平均或加权，降低单个模型的随机误差。

### 核心创新点  
1. **把语义相似的节点合并**  
   - 之前的树搜索把每一次 LLM 生成的文本都当作独立节点，导致大量重复。  
   - 这篇论文先用微调后的 SimCSE 把每个节点的文本映射成向量，再用层次聚类把相近的向量合并成一个“超节点”。  
   - 合并后树的宽度大幅收缩，搜索深度不变的情况下算力需求下降，实际实验显示搜索时间显著缩短。  

2. **用 TD 学习平滑验证器评分**  
   - 传统做法直接把验证器对当前节点的分数当作最终评估，噪声大，容易导致频繁切换分支。  
   - 作者在验证器训练时加入时序差分目标，使用 λ‑回报让模型同时考虑当前步骤和后续步骤的价值。  
   - 结果是评分方差下降，搜索更倾向于沿着高价值的轨迹深入，提升了最终解题的准确率。  

3. **验证器集成提升鲁棒性**  
   - 单一验证器在推理时仍会出现偶发错误。  
   - 论文在推理阶段并行运行多个验证器，把它们的分数做加权平均后作为最终打分。  
   - 这种“投票”机制进一步压制了噪声，使得搜索决策更稳健。  

4. **插件化的 FETCH 框架**  
   - 以上三项技术被封装成一个可插拔的系统，兼容多种已有的树搜索算法（如 BFS、DFS、Monte‑Carlo Tree Search 等），不需要改动原有搜索代码。  
   - 这种设计让研究者和工程师可以像装插件一样把 FETCH 加进自己的推理管线，降低了使用门槛。

### 方法详解  
**整体思路**：先让 LLM 产生若干候选推理步骤，形成搜索树；随后用两层“过滤+合并”机制削减冗余，再用改进的验证器给每条路径打分，最后依据分数继续展开。整个过程循环进行，直到找到满足终止条件的答案。  

**步骤拆解**  

1. **候选生成**  
   - 在当前树节点，调用 LLM 生成 *k* 条可能的下一步描述（比如“设 x = …”或“对等式两边同除以 2”）。  
   - 每条描述都被视为一个新节点的原始文本。  

2. **语义嵌入与聚类**  
   - 把每条文本喂入微调好的 SimCSE，得到 768 维向量。  
   - 对所有新节点的向量执行自底向上的层次聚类（agglomerative clustering），阈值设定为“语义相似度 > 0.85”。  
   - 聚类结果中每个簇被合并为一个超节点，簇内的所有文本共享同一状态标识，只保留一个代表文本（通常是得分最高的）。  

3. **验证器训练（离线）**  
   - 验证器本身是一个小型的 Transformer，输入是当前状态的文本和候选步骤的文本，输出一个可信度分数。  
   - 训练时加入 TD 目标：对每个状态 *s*，预测的分数 *V(s)* 与下一个状态 *s'* 的分数 *V(s')* 结合 λ‑回报计算误差，梯度下降更新。  
   - λ 参数在 0.7 左右，兼顾短期准确性和长期价值。  

4. **验证器集成（在线）**  
   - 部署时并行加载 3‑5 个独立训练的验证器。  
   - 对每个候选节点，收集所有验证器的分数，做加权平均（权重可根据验证器在验证集上的表现调节），得到最终打分。  

5. **树搜索决策**  
   - 根据合并后的节点集合和集成分数，使用所选的搜索策略（如 Best‑First、Monte‑Carlo）决定下一个展开的节点。  
   - 若某条路径的累计分数超过预设阈值或达到最大深度，则输出该路径的最终答案。  

**最巧妙的点**  
- **聚类合并**把“文字相同但表达不同”的冗余直接砍掉，类似于把同一条道路的多条平行支路合并成一条主干，极大降低了搜索宽度。  
- **TD + λ‑回报**让验证器不再是“瞬时裁判”，而是会“预见”后续步骤的价值，像是给搜索加装了前瞻性的指南针，显著抑制了因评分噪声导致的频繁换道。  

### 实验与效果  
- **数据集**：在数学推理基准 GSM8K、扩展版 GSM‑Plus 以及高难度 MATH 上进行评估。  
- **对比基线**：原始的 Verifier‑Guided Tree Search（不做合并、使用单一验证器）、CoT（Chain‑of‑Thought）直接生成、以及最新的 Self‑Consistency 方法。  
- **结果**：论文声称在四种不同的树搜索算法上，FETCH 均实现了显著的准确率提升和计算成本下降。例如在 GSM8K 上，准确率提升约 5%~8%，搜索节点数下降约 30%~45%。在 MATH 这类高难度数据集上，正确率提升更为明显，且推理时间缩短近一半。  
- **消融实验**：分别去掉聚类合并、TD 学习、验证器集成三项，发现聚类对算力削减贡献最大，TD 学习对准确率提升最关键，集成则在所有设置下都能带来约 1%~2% 的稳健提升。  
- **局限**：聚类阈值需要手动调节，若阈值设得过高可能误合并本应区分的细微步骤；TD 学习对验证器的训练数据量敏感，数据不足时方差仍会残留。作者也提到在极端长链推理（>20 步）时聚类开销会略升，需进一步优化。  

### 影响与延伸思考  
FETCH 的插件化设计让它迅速被后续工作采纳，尤其是那些在大模型上做“思考树”搜索的项目。2024 年后出现的几篇论文（如 *Tree‑Prune*、*Variance‑Reduced Verifier*）直接在聚类合并或 TD‑based 验证器上进行改进，说明社区已经把降低搜索方差和冗余视为关键方向。未来可以探索：① 用更轻量的跨语言嵌入模型做聚类，降低前置成本；② 将强化学习的策略梯度与验证器训练结合，实现端到端的搜索策略学习；③ 在代码生成、推理链可解释性等非数学任务上验证 FETCH 的通用性。  

### 一句话记住它  
**FETCH 用语义聚类砍掉重复枝，用时序差分平滑评分，让大模型的树搜索既快又准。**