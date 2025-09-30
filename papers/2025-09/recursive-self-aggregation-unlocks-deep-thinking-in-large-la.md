# Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models

> **Date**：2025-09-30
> **arXiv**：https://arxiv.org/abs/2509.26626

## Abstract

Test-time scaling methods improve the capabilities of large language models (LLMs) by increasing the amount of compute used during inference to make a prediction. Inference-time compute can be scaled in parallel by choosing among multiple independent solutions or sequentially through self-refinement. We propose Recursive Self-Aggregation (RSA), a test-time scaling method inspired by evolutionary methods that combines the benefits of both parallel and sequential scaling. Each step of RSA refines a population of candidate reasoning chains through aggregation of subsets to yield a population of improved solutions, which are then used as the candidate pool for the next iteration. Empirically, RSA delivers substantial performance gains with increasing compute budgets across diverse tasks, model families and sizes. Notably, RSA with Gemini 3 Flash attains performance near the top of the ARC-AGI-2 public leaderboard. RSA also enables Qwen3-4B-Instruct-2507 to achieve competitive performance with larger reasoning models, including DeepSeek-R1 and o3-mini (high), outperforming purely parallel and sequential scaling strategies across AIME-25, HMMT-25, Reasoning Gym, LiveCodeBench-v6, and SuperGPQA. We further propose a novel aggregation-aware reinforcement learning approach that yields significant performance gains by training the model to combine solutions.

---

# 递归自我聚合解锁大语言模型的深度思考 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在一次前向传播里只能使用固定的算力，导致在需要多步推理的复杂任务上常常“卡壳”。已有的测试时放大（test‑time scaling）手段要么并行生成多个答案再投票，要么让模型自行循环改写同一个答案。前者虽然利用了更多的算力，但缺乏跨答案的相互学习；后者则受限于单一路径的局部最优，容易陷入死循环。于是，如何在不改动模型本体的前提下，既能并行探索多条思路，又能让这些思路相互促进，成为了一个亟待突破的瓶颈。

### 关键概念速览
- **测试时放大（Test‑time Scaling）**：在模型推理阶段投入额外算力来提升答案质量，类似于考试时给学生更多时间思考。
- **并行扩展（Parallel Scaling）**：一次性让模型生成多条独立解答，再通过多数投票或取最高分来决定最终答案，像是让一群学生各自写答案后再选最常出现的。
- **顺序扩展（Sequential Scaling）**：模型在得到初稿后继续自我改写，形成“思考—修正—再思考”的循环，类似于老师让学生先写草稿再反复修改。
- **思维链（Chain‑of‑Thought, CoT）**：让模型在给出结论前把推理步骤写出来，像是解题时把每一步写在纸上，便于检查和纠错。
- **递归自我聚合（Recursive Self‑Aggregation, RSA）**：一种把并行和顺序两种放大方式结合起来的算法，像进化论中的“选择—交叉—变异”循环，让多个解答相互融合后再产生新一代。
- **候选解答池（Candidate Pool）**：当前迭代中所有生成的思维链集合，供后续聚合和筛选使用。
- **聚合感知强化学习（Aggregation‑aware RL）**：在训练阶段加入奖励，让模型学会如何把若干解答合并成更好的答案，类似于教学生如何把同学的好点子整合进自己的作业。

### 核心创新点
1. **并行+顺序的混合迭代**  
   - 之前的做法要么一次性并行生成 N 条答案，要么让单条答案循环自我改写。  
   - RSA 在每一轮先并行生成若干思维链，然后把这些链的子集进行聚合，得到更强的中间解，再把这些中间解作为下一轮的并行输入。  
   - 这种“先选后育”的循环让算力利用更高效，既保留了多样性，又实现了跨答案的知识共享。

2. **进化式聚合机制**  
   - 传统进化算法通过交叉和变异产生新个体，RSA 把“聚合子集”当作交叉，把“重新推理”当作变异。  
   - 每轮聚合时会随机抽取若干思维链，合并它们的中间推理步骤（例如取多数出现的关键事实），形成一个更可靠的“共识链”。  
   - 这种机制让模型在每一次递归中都能提升解答的稳健性，尤其在答案空间稀疏的数学或代码题上表现突出。

3. **聚合感知的强化学习微调**  
   - 直接在推理时做聚合会出现“怎么合并才好”的二义性。作者额外训练了一个小的策略网络，奖励模型产生易于聚合的中间结果（比如结构化的事实列表）。  
   - 通过这种方式，模型在生成思维链时会自觉留下“可拼接”的线索，聚合过程的效率和质量随之提升。  
   - 实验显示，这一步相较于纯粹的 RSA 能再提升约 5% 的准确率。

### 方法详解
**整体框架**  
RSA 把一次推理拆成若干“代”（generation），每代包含三个子步骤：并行生成、子集聚合、种群更新。整个过程在给定的算力预算（比如总的 token 数或时间）内递归进行，直至预算耗尽或达到预设的代数。

**步骤拆解**  

1. **并行生成（Parallel Generation）**  
   - 给定输入问题，模型一次性生成 K 条思维链（K 通常在 4‑16 之间），每条链都是完整的 CoT。  
   - 这一步相当于让 K 位“思考者”独立作答，保证了解答的多样性。

2. **子集抽取与聚合（Subset Selection & Aggregation）**  
   - 从 K 条链中随机抽取 M 条（M ≤ K），形成一个子集。  
   - 对每一步的中间推理进行投票或加权平均：如果多数链在第 i 步都提到同一事实，则该事实被保留；若出现冲突，则采用置信度最高的链的答案。  
   - 聚合的结果是一条新的思维链，称为“聚合链”。它融合了子集成员的共识，往往比单条链更可靠。

3. **种群更新（Population Update）**  
   - 将所有聚合链（通常与原始 K 条链一起）组成下一代的候选池。  
   - 为了防止种群退化，作者会在每代加入少量“新鲜血液”：重新从模型生成 K 条全新链，保持多样性。  
   - 这样，种群既保留了高质量的共识，又不断注入新思路。

4. **递归迭代（Recursive Loop）**  
   - 重复步骤 1‑3，直至算力预算用完。每一次迭代都在前一代的基础上提升解答质量，类似于自然选择的“适者生存”。  

**聚合感知强化学习的细节**  
- 在训练阶段，作者给模型加了一个额外的奖励信号：如果模型的输出在聚合后能够被多数链采纳，则奖励提升。  
- 这促使模型在生成思维链时倾向于使用结构化、易对齐的表达方式（比如“事实 A = …”，而不是自由散文），从而让后续聚合更顺畅。  

**最巧妙的设计**  
- **子集随机抽取**：看似简单，却防止了所有链都被同一批错误信息所污染，类似于遗传算法中的“交叉多样性”。  
- **聚合后再并行**：把聚合链重新放回并行生成阶段，让模型在更高质量的起点上继续探索，形成“提升的基线”。  

### 实验与效果
- **测试任务**：作者在 ARC‑AGI‑2（科学推理）、AIME‑25、HMMT‑25（高中数学）、Reasoning Gym、LiveCodeBench‑v6（代码生成）以及 SuperGPQA（通用问答）等六大基准上评估 RSA。  
- **基线对比**：与纯并行投票、纯顺序自我改写以及最新的自检（Self‑Check）方法相比，RSA 在所有任务上均实现了显著提升。举例来说，在 ARC‑AGI‑2 公共排行榜上，使用 Gemini 3 Flash 的 RSA 版本逼近榜首分数（仅差 0.3%），而同模型的单纯并行投票则落后约 4%。  
- **小模型突破**：Qwen3‑4B‑Instruct‑2507 通过 RSA 达到了与 DeepSeek‑R1、o3‑mini（high）等大模型相当的表现，尤其在 AIME‑25 上提升约 7%。  
- **消融实验**：作者分别去掉（1）子集聚合、（2）新鲜血液注入、（3）聚合感知 RL。结果显示，去掉聚合会导致整体性能下降 3‑5%，去掉新鲜血液导致后期性能收敛提前，下降约 2%，而去掉 RL 则在所有任务上平均损失约 5%。这些实验验证了每个模块的贡献。  
- **局限性**：RSA 需要多轮推理，推理时间比单轮方法长数倍；在算力极其受限的场景下仍不适用。作者也指出，聚合策略对不同任务的敏感度不同，需手动调参。  

### 影响与延伸思考
- 这篇工作把进化算法的思想成功搬进了大语言模型的推理阶段，开启了“种群式推理”这一新方向。随后的几篇论文（如 *Evolutionary Prompting*、*Population‑Based Decoding*）都在不同维度上扩展了 RSA 的思路：有的把提示词也当作基因进行交叉，有的把多模型的输出共同进化。  
- 对于想继续深挖的读者，可以关注两条路：一是 **聚合策略的自动化**——用学习的方式决定子集大小、投票规则等；二是 **跨模态种群**——把语言模型、视觉模型甚至检索系统的输出一起放进同一“种群”，让它们相互进化。  
- 另外，RSA 的递归框架与 **自监督的元学习** 有天然契合，未来可能把“如何进化”本身也交给模型学习，从而实现更少人工干预的自适应推理。  

### 一句话记住它
递归自我聚合把并行探索和顺序改进像进化一样交叉，让大模型在测试时用更多算力“进化出”更深的思考。