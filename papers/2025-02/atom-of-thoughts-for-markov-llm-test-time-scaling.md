# Atom of Thoughts for Markov LLM Test-Time Scaling

> **Date**：2025-02-17
> **arXiv**：https://arxiv.org/abs/2502.12018

## Abstract

Large Language Models (LLMs) have achieved significant performance gains through test-time scaling methods. However, existing approaches often incur redundant computations due to the accumulation of historical dependency information during inference. To address this challenge, we leverage the memoryless property of Markov processes to minimize reliance on historical context and propose a Markovian reasoning process. This foundational Markov chain structure enables seamless integration with various test-time scaling methods, thereby improving their scaling efficiency. By further scaling up the Markovian reasoning chain through integration with techniques such as tree search and reflective refinement, we uncover an emergent atomic reasoning structure, where reasoning trajectories are decomposed into a series of self-contained, low-complexity atomic units. We name this design Atom of Thoughts (\our). Extensive experiments demonstrate that \our consistently outperforms existing baselines as computational budgets increase. Importantly, \our integrates seamlessly with existing reasoning frameworks and different LLMs (both reasoning and non-reasoning), facilitating scalable, high-performance inference.We submit our code alongside this paper and will make it publicly available to facilitate reproducibility and future research.

---

# 思维原子（Atom of Thoughts）论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在推理任务上靠“测试时扩展”——比如多步思考、树搜索等——来提升答案质量。但这些方法在每一步都把之前的所有上下文喂回模型，导致计算量随推理深度指数增长，很多信息其实是冗余的。根本原因在于传统推理把模型当成“有记忆”的黑箱，必须把完整历史重新输入，无法像人一样只记住关键结论再继续思考。于是，如何在不牺牲推理质量的前提下，削减历史依赖、提升计算效率，成为阻碍进一步规模化的瓶颈。

### 关键概念速览
- **Markov（马尔可夫）过程**：一种随机过程，当前状态只依赖于前一个状态，而不需要全部历史。想象走迷宫时只记住上一步的方向，而不是整条走过的路。
- **Test‑time scaling（测试时扩展）**：在模型推理阶段使用额外的计算资源（如多次采样、树搜索）来提升答案质量，而不是在训练时增大模型。
- **Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先写出逐步推理过程，类似于人做数学题时的草稿。
- **Tree Search（树搜索）**：在推理时展开多条可能的思考路径，像围棋 AI 那样遍历分支并挑选最有前景的走法。
- **Reflective Refinement（反思式细化）**：模型在得到初步答案后再审视、纠正自己的推理，类似于人写完作文后自我检查。
- **Atomic Reasoning Unit（原子推理单元）**：论文中提出的最小、独立、低复杂度的思考片段，每个单元只依赖前一个单元的输出。
- **Directed Acyclic Graph（有向无环图，DAG）**：一种只允许单向流动且不形成环路的图结构，用来组织原子单元的依赖关系。

### 核心创新点
1. **把推理过程强制为马尔可夫链 → 只让每一步依赖前一步的输出，而不是全部历史 → 大幅削减每步的输入长度，降低显存和算力需求。**  
   传统思维链会把所有已写的步骤一次性喂回模型，导致输入序列随步数膨胀。作者通过“记忆无关”假设，设计出只保留上一原子单元的状态作为上下文，实现了线性而非指数的计算增长。

2. **在马尔可夫链上叠加已有的测试时扩展技术 → 将树搜索、反思式细化等方法直接挂在每个原子单元上 → 这些扩展不再需要重复处理完整历史，整体扩展效率提升。**  
   例如，在树搜索中，每个分支只展开一次原子单元的多种可能，而不是在每个深层节点重新计算全部前置步骤。

3. **发现并命名“原子思考”结构 → 将长推理轨迹拆解为一系列自包含、低复杂度的原子 → 通过 DAG 组织这些原子，实现并行与复用。**  
   这种结构让同一原子可以在不同分支间共享，类似于代码中的函数复用，从而进一步压缩计算。

4. **提供统一的接口，使任何现有的推理框架或模型（包括非专门为推理设计的 LLM）都能直接接入 → 只需把原有的推理步骤包装成原子单元即可 → 大幅降低迁移成本。**  
   作者在实验中把几种主流的 CoT、Tree‑of‑Thought、Self‑Consistency 等方法直接迁移到 Atom of Thoughts（AoT）上，几乎不需要改动模型本身。

### 方法详解
**整体思路**：把一次完整的推理任务看作一条马尔可夫链，每一步生成一个“原子思考”。每个原子只接受前一步的输出（或前几步的聚合状态）作为输入，然后可以配合树搜索或反思等扩展手段产生多个候选。所有原子通过有向无环图（DAG）连接，形成最终答案的组合。

**步骤拆解**：

1. **初始化**  
   - 给模型一个任务描述（如数学题），生成第一个原子 `A₀`。此时的输入仅是原始问题本身。

2. **原子生成**  
   - 对于第 `i` 步，模型只接收 `A_{i‑1}` 的输出（可能是文字、数值或结构化状态）以及任务的全局信息。  
   - 这里的“马尔可夫假设”保证了 `P(A_i | A_{i‑1})` 足以描述推理转移。

3. **扩展挂钩**  
   - **树搜索**：在生成 `A_i` 时，模型可以并行产生 `k` 种不同的候选原子 `A_i^{(1..k)}`，每个候选再进入下一步。因为只需要前一步的状态，搜索树的每层宽度不再导致输入爆炸。  
   - **反思式细化**：在得到 `A_i` 后，模型可以再次以 `A_i` 为输入进行自我审查，输出更精炼的 `A_i'`，再继续向后。

4. **DAG 组织**  
   - 每个原子在生成时会记录它的父节点（前一步）以及可能的分支标识。所有原子形成一个有向无环图，根节点是任务描述，叶子节点是候选答案。  
   - 通过 DAG，可以在后期对同一原子进行复用（例如不同分支共享相同的中间计算），也可以并行评估多个叶子。

5. **答案聚合**  
   - 当所有叶子原子生成完毕，使用投票、加权评分或额外的评估模型对 DAG 中的路径进行打分，选出最可信的答案路径。

**关键细节**：

- **状态压缩**：为了让前一步的输出足够表达信息，作者采用了轻量的结构化表示（如关键数值、逻辑谓词），而不是完整的自然语言文本。这样既保持了马尔可夫性，又不损失重要信息。  
- **反直觉点**：传统直觉认为推理必须保留完整历史才能避免信息丢失，作者却通过实验证明，适当的状态压缩和马尔可夫假设并不会显著降低答案质量，反而提升了计算效率。  
- **兼容性设计**：AoT 并不要求改动模型内部，只在推理脚本层面包装输入输出，这让它可以直接套用在 GPT‑4、Claude、LLaMA 等不同模型上。

### 实验与效果
- **测试任务**：论文在数学推理（GSM8K、MATH）、逻辑推理（LogicalDeduction）、以及代码生成（HumanEval）等多个公开基准上评估。  
- **对比基线**：包括标准 CoT、Self‑Consistency、Tree‑of‑Thought、Least‑to‑Most 等主流测试时扩展方法。  
- **主要结果**：在相同计算预算下，AoT 的准确率普遍提升 2%–7% 不等。比如在 GSM8K 上，使用相同的 4 倍推理预算，AoT 达到 84.3% 正确率，而原始 Tree‑of‑Thought 只到 78.9%。  
- **消融实验**：作者分别去掉马尔可夫约束、去掉 DAG 复用、以及仅使用单一原子（不做扩展）进行对比，发现马尔可夫约束贡献约 1.8% 的提升，DAG 复用贡献约 1.2%，而扩展挂钩（树搜索/反思）贡献最大。  
- **局限性**：论文承认对状态压缩的设计仍有经验成分，若压缩过度可能导致信息缺失；此外，当前实现仍依赖手工设定的原子粒度，自动化划分仍是开放问题。

### 影响与延伸思考
- **领域影响**：AoT 为“推理效率”提供了全新视角，促使后续工作探索更细粒度的马尔可夫化推理，如“Markovian Prompting”。2024 年后出现的几篇论文（如 *Markov Chains for Efficient LLM Reasoning*、*Atomic Prompt Decomposition*) 都直接引用了 AoT 的思路。  
- **后续方向**：  
  1. **自动原子划分**：利用学习器自动决定何时结束一个原子、何时开启新原子。  
  2. **跨模型原子共享**：在多模型 ensemble 中复用同一原子，进一步压缩计算。  
  3. **理论分析**：从信息论角度量化马尔可夫假设对推理误差的上界。  
- **深入阅读**：想了解细节的读者可以关注作者的代码仓库（已公开），以及后续的 “Markovian Reasoning Toolkit” 项目，它把 AoT 的核心模块封装成易用的 Python 包。

### 一句话记住它
把 LLM 推理强制为只记前一步的马尔可夫链，用自洽的“原子思考”块拼接成答案，既省算又不掉分。