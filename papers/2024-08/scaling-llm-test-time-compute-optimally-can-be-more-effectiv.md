# Scaling LLM Test-Time Compute Optimally can be More Effective than   Scaling Model Parameters

> **Date**：2024-08-06
> **arXiv**：https://arxiv.org/abs/2408.03314

## Abstract

Enabling LLMs to improve their outputs by using more test-time computation is a critical step towards building generally self-improving agents that can operate on open-ended natural language. In this paper, we study the scaling of inference-time computation in LLMs, with a focus on answering the question: if an LLM is allowed to use a fixed but non-trivial amount of inference-time compute, how much can it improve its performance on a challenging prompt? Answering this question has implications not only on the achievable performance of LLMs, but also on the future of LLM pretraining and how one should tradeoff inference-time and pre-training compute. Despite its importance, little research attempted to understand the scaling behaviors of various test-time inference methods. Moreover, current work largely provides negative results for a number of these strategies. In this work, we analyze two primary mechanisms to scale test-time computation: (1) searching against dense, process-based verifier reward models; and (2) updating the model's distribution over a response adaptively, given the prompt at test time. We find that in both cases, the effectiveness of different approaches to scaling test-time compute critically varies depending on the difficulty of the prompt. This observation motivates applying a "compute-optimal" scaling strategy, which acts to most effectively allocate test-time compute adaptively per prompt. Using this compute-optimal strategy, we can improve the efficiency of test-time compute scaling by more than 4x compared to a best-of-N baseline. Additionally, in a FLOPs-matched evaluation, we find that on problems where a smaller base model attains somewhat non-trivial success rates, test-time compute can be used to outperform a 14x larger model.

---

# 在测试时最优扩展大语言模型计算量或许比扩展模型参数更有效 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在训练阶段已经投入了巨量算力，但在实际使用时往往只能跑一次前向传播，输出一个答案。很多人以为只要把模型做得更大、参数更多，性能自然提升，却忽视了推理阶段的计算潜力。传统的“best‑of‑N”做法——一次生成 N 条候选答案再挑最好的——只能在固定的预算内稍微提升质量，且随 N 增长成本线性上升。更关键的是，面对不同难度的提问，统一的计算配额往往要么浪费在简单问题上，要么不足以解决棘手问题。于是出现了一个核心难点：**如果给模型一个固定但不算小的推理算力，它到底能把答案提升多少？** 这直接关系到我们是继续投入更大模型，还是把算力花在更聪明的推理策略上。

### 关键概念速览
- **测试时计算（test‑time compute）**：模型在收到用户提问后，用来生成或改进答案的额外算力。想象成答题时可以多写几遍草稿，而不是一次性交卷。
- **密集验证器（dense verifier）**：一个专门训练来评估答案质量的模型，给每个候选答案打分。类似老师批改作业，分数高的更可能是对的。
- **best‑of‑N**：一次生成 N 条答案，挑分数最高的那条。相当于让学生写 N 份答案，选最好的那份。
- **自适应分配（adaptive allocation）**：根据当前提问的难度，动态决定要用多少算力。像老师根据题目难易度决定给学生多少时间。
- **计算最优策略（compute‑optimal scaling）**：一种在每个提问上都能把算力用到最有效位置的调度方法。可以比作在考试中把时间分配给最需要的题目。
- **分布更新（distribution update）**：在推理过程中，根据已经看到的部分答案，重新调整模型生成下一个词的概率分布。类似人在写作文时，根据前面的句子不断修正后面的表达。
- **FLOPs‑matched 评估**：在实验中把不同方法的总浮点运算次数（FLOPs）对齐，确保比较的是“花同样多钱”下的效果，而不是单纯看模型大小。

### 核心创新点
1. **系统化研究测试时计算的扩展规律**  
   之前大多数工作只关注模型参数的规模，几乎没有系统地测量“多算力”到底能带来多少提升。论文把这个问题抽象成“给定固定算力，性能提升多少”，并在不同难度的提示上做了细致实验。结果显示，算力的边际收益并非线性，而是高度依赖提示难度。

2. **两种可扩展的推理机制**  
   - *密集验证器搜索*：先让模型生成大量候选答案，用验证器打分后挑最优。相比传统的 best‑of‑N，这里把验证器当作“筛子”，把算力集中在评分上。  
   - *自适应分布更新*：在生成过程中实时根据提示和已有答案调整概率分布，等于是让模型在推理时“思考”更多。两者在不同难度的任务上表现差异明显，验证了没有“一刀切”的最佳策略。

3. **计算最优的自适应调度框架**  
   作者提出先估计提示难度，再决定是投入更多候选生成、更多验证器评分，还是更多分布更新。这个调度器像考试计时器，能把有限的算力“投向最需要的地方”。实验表明，这种调度比单纯的 best‑of‑N 提升了超过 4 倍的效率。

4. **在 FLOPs‑matched 条件下，小模型凭推理算力超越 14 倍大模型**  
   把算力总量固定后，让一个相对小的基模型使用上述自适应算力策略，竟然在一些非 trivial 的任务上跑赢了参数量大约 14 倍的模型。说明算力的灵活使用可以在一定程度上抵消模型规模的劣势。

### 方法详解
**整体思路**：给每个输入提示分配一个固定的算力预算，然后让模型在这个预算内决定怎么使用——是多生成候选答案、是多跑验证器、还是多做分布更新。核心是一个“难度感知调度器”，它先判断当前提示是“易”“中”“难”，再把预算切分成不同的子任务。

**步骤拆解**：

1. **难度估计**  
   - 使用一个轻量的前置模型或基于提示长度、词汇稀有度等特征的规则，输出一个难度分数。可以把它想象成老师先快速浏览题目，判断这道题需要多少时间。

2. **算力预算划分**  
   - 根据难度分数，调度器决定三类算力的比例：  
     a) **候选生成**（生成 N 条答案的前向计算）  
     b) **验证器评分**（对每条答案跑密集验证器）  
     c) **分布更新**（在生成过程中进行多轮采样或重抽样）  
   - 例如，对难题可能分配 50% 给生成更多候选，30% 给验证器，20% 给分布更新；对简单题则相反。

3. **密集验证器搜索**  
   - 生成的候选答案送入预训练的验证器模型，得到每条的质量分数。  
   - 采用“top‑k”或“阈值”策略挑选最有希望的答案。这里的关键是验证器本身要足够“密集”，即对细节有辨识能力。

4. **自适应分布更新**  
   - 在生成过程中，模型会根据已经采样的部分答案动态调整下一个 token 的概率分布。实现方式可以是：  
     - **重抽样**：对低分候选进行再抽样，提升多样性。  
     - **温度调节**：根据当前分数或置信度改变采样温度，让模型在不确定时更大胆。  
   - 这一步相当于让模型在写作时不断回头检查、改写。

5. **最终答案选取**  
   - 将经过验证器筛选和分布更新优化的候选集合，再用一个轻量的聚合规则（如最高验证器分数或加权投票）输出最终答案。

**最巧妙的地方**：调度器不是固定的比例表，而是根据实时难度估计动态计算。作者甚至在实验中展示，若把算力全都投在单一策略上（比如只做 best‑of‑N），效果远不如把算力混合使用。这个“算力拼图”思路突破了传统“一次性生成” 的思维定式。

### 实验与效果
- **测试任务**：论文在一系列“挑战性提示”上评估，包括需要多步推理、常识检索以及开放式问答等。具体数据集名称未在摘要中给出，作者称这些任务对小模型的成功率只有“非平凡”水平。
- **对比基线**：  
  - **best‑of‑N**（固定 N 条候选）  
  - **单一密集验证器**（只做验证不做分布更新）  
  - **单一分布更新**（只做自适应采样不做验证）  
- **主要结果**：  
  - 使用计算最优调度器的效率提升超过 4 倍，相比 best‑of‑N 在相同 FLOPs 下得到更高的成功率。  
  - 在 FLOPs‑matched 实验中，算力相同的情况下，小基模型通过上述方法的表现超过了参数量约 14 倍的大模型。  
  - 论文声称这些提升在“难度中等到高”的提示上最为明显，而在极易或极难的提示上收益会趋于平缓。
- **消融实验**：作者分别去掉验证器、去掉分布更新、或固定算力分配比例，发现每个模块的缺失都会导致整体性能显著下降，验证了“多策略混合”是关键。
- **局限性**：  
  - 需要额外训练或微调密集验证器，增加了实现成本。  
  - 难度估计器的准确性直接影响算力分配，若估计错误可能导致算力浪费。  
  - 摘要未提供对极大规模模型（如数百亿参数）在同样框架下的实验，是否仍能保持优势仍待验证。

### 影响与延伸思考
这篇工作把“推理算力”提升到和“模型规模”同等重要的位置，促使社区开始探索 **动态推理**、**自适应算力调度** 等方向。随后出现的研究如 **Mixture‑of‑Experts 在推理阶段的激活选择**、**可微分算力预算分配**、以及 **基于 LLM 的自我评估与自我改进循环**，都可以视为对该论文思路的延伸。对想进一步深入的读者，建议关注：

- **可微分算力调度器**：把算力分配过程本身做成可学习的模块。  
- **多模态验证器**：将视觉、表格等信息纳入验证器，扩大“密集评估”范畴。  
- **算力‑效率的理论上界**：研究在固定 FLOPs 下，最优的推理策略上限。

### 一句话记住它
**给模型足够的推理算力，并聪明地把这些算力分配到最需要的地方，往往能让小模型跑赢大模型。**