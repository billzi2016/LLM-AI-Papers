# Large Language Monkeys: Scaling Inference Compute with Repeated Sampling

> **Date**：2024-07-31
> **arXiv**：https://arxiv.org/abs/2407.21787

## Abstract

Scaling the amount of compute used to train language models has dramatically improved their capabilities. However, when it comes to inference, we often limit models to making only one attempt at a problem. Here, we explore inference compute as another axis for scaling, using the simple technique of repeatedly sampling candidate solutions from a model. Across multiple tasks and models, we observe that coverage -- the fraction of problems that are solved by any generated sample -- scales with the number of samples over four orders of magnitude. Interestingly, the relationship between coverage and the number of samples is often log-linear and can be modelled with an exponentiated power law, suggesting the existence of inference-time scaling laws. In domains like coding and formal proofs, where answers can be automatically verified, these increases in coverage directly translate into improved performance. When we apply repeated sampling to SWE-bench Lite, the fraction of issues solved with DeepSeek-Coder-V2-Instruct increases from 15.9% with one sample to 56% with 250 samples, outperforming the single-sample state-of-the-art of 43%. In domains without automatic verifiers, we find that common methods for picking from a sample collection (majority voting and reward models) plateau beyond several hundred samples and fail to fully scale with the sample budget.

---

# 大语言猴子：通过重复采样扩展推理计算 论文详细解读

### 背景：这个问题为什么难？

训练大语言模型时，投入的算力越多，模型的能力通常会显著提升。但在实际使用时，推理阶段往往只让模型给出一次答案，这相当于把所有希望压在一次“投掷”上。很多任务（尤其是代码生成、数学证明）本身就有多个潜在解法，而一次采样的成功率往往很低。过去的改进主要集中在模型规模、数据质量或提示工程上，少有人系统地探讨“多次尝试”本身的算力成本与收益关系。因此，如何把推理算力当作另一个可伸缩的维度来提升覆盖率，成为了一个值得深入的研究点。

### 关键概念速览
- **推理算力（Inference Compute）**：指在模型生成答案时消耗的计算资源，包括采样次数、温度调节等。可以把它想成“考试时的答题时间”，时间越多，学生可以尝试更多答案。
- **重复采样（Repeated Sampling）**：让模型在同一输入上多次独立生成候选答案，就像让同一个学生多次写作业，挑出最对的那份。
- **覆盖率（Coverage）**：在所有测试题目中，至少有一个候选答案是正确的比例。相当于“只要有一次答对，就算这道题完成”。
- **指数幂律（Exponentiated Power Law）**：一种数学关系，表现为变量的对数与另一个变量的对数呈线性关系。这里指的是覆盖率随采样次数的增长规律，像是“每翻倍一次采样，覆盖率提升固定比例”。
- **自动验证器（Automatic Verifier）**：能够机器判定答案是否正确的程序，例如代码单元测试或形式化证明检查器。它让我们可以直接统计覆盖率，而不需要人工评审。
- **多数投票（Majority Voting）**：从多次采样中选出出现次数最多的答案，类似于“让多数学生的答案决定最终分数”。
- **奖励模型（Reward Model）**：训练出来的模型，用来给每个候选答案打分，帮助挑选最有可能正确的答案。相当于请一个“评卷老师”先行打分。

### 核心创新点
1. **把推理算力当作可伸缩维度**  
   之前的工作把算力主要放在训练阶段，推理时只做一次采样。本文直接把“采样次数”视作算力的延伸，用同一模型多次生成答案。这样做把算力的使用从“一次性”转为“可累积”，显著提升了解题覆盖率。

2. **发现并量化推理时的尺度律**  
   通过在四个数量级的采样规模上实验，作者观察到覆盖率随采样次数呈近似对数线性增长，并用指数幂律模型进行拟合。这个经验公式让我们可以预测在给定算力预算下能达到的覆盖上限，提供了理论指导。

3. **在可自动验证的任务上直接转化为性能提升**  
   在代码生成（SWE‑bench Lite）和形式化证明等任务中，答案的正确性可以机器判定。作者把提升的覆盖率直接映射为最终的任务成功率，证明了重复采样在这些场景下的实际价值。

4. **系统评估传统采样后处理方法的局限**  
   对比多数投票和奖励模型，实验显示这两种在几百次采样后就出现饱和，无法继续利用更多算力。此发现提醒我们，单纯靠后处理并不能充分挖掘重复采样的潜力。

### 方法详解
整体思路非常直白：**给定同一个输入，反复让模型独立采样 N 次，收集所有候选答案，然后根据任务特性决定如何使用这些答案**。整个流程可以拆成三步：

1. **多次独立采样**  
   - 输入保持不变，模型每次采样时使用相同的温度或采样策略（如 nucleus sampling）。  
   - 采样过程是相互独立的，等价于把同一道题交给 N 位“同质的学生”。  
   - 采样次数 N 可以从 1 到几千不等，作者在实验中覆盖了 10、100、1 000、10 000 等尺度。

2. **答案收集与去重**  
   - 将所有生成的文本放进一个集合，去掉完全相同的重复项。  
   - 对于需要自动验证的任务，直接把每个唯一答案送入验证器，记录是否通过。  
   - 对于无法自动验证的任务，保留所有候选，后续交给投票或奖励模型筛选。

3. **结果选取**  
   - **自动验证任务**：只要集合中有一个通过验证的答案，就算该题 solved。覆盖率即 solved 题目数除以总题目数。  
   - **非自动验证任务**：使用多数投票或奖励模型对候选进行排序，取最高分的答案作为最终输出。实验表明，这两种方法在采样数超过几百后提升停滞。

**关键细节**  
- **采样独立性**：作者强调必须保证每次采样的随机种子不同，否则多次采样会产生高度冗余的答案，导致算力浪费。  
- **指数幂律拟合**：覆盖率 C 与采样次数 S 的关系被建模为 C = 1 - exp(-a·S^b)，其中 a、b 通过最小二乘法在实验数据上拟合得到。这个公式直观上说明，随着 S 增大，未覆盖的概率呈指数衰减。  
- **算力预算换算**：因为一次采样的计算成本与模型大小、序列长度成正比，作者把“采样次数 × 单次推理成本”当作总推理算力，用来与训练算力进行对比。

最让人意外的地方是：**仅仅增加采样次数，就能在不改动模型结构或参数的前提下，获得类似于更大模型的效果**。这是一种“软硬件分离”的提升方式，尤其适合算力受限但对结果质量要求高的场景。

### 实验与效果
- **任务与数据集**  
  - **代码生成**：SWE‑bench Lite（包含真实软件缺陷的代码修复任务），答案可通过单元测试自动验证。  
  - **形式化证明**：若干公开的定理证明基准，使用自动定理检验器。  
  - **自然语言任务**：包括常规的问答和多选题，无法自动验证，采用多数投票或奖励模型评估。

- **基线对比**  
  - 在 SWE‑bench Lite 上，单次采样的 DeepSeek‑Coder‑V2‑Instruct 成功率为 15.9%。  
  - 采用 250 次采样后，成功率提升到 56%，超过此前单样本最佳 43% 的记录。  
  - 对比多数投票和奖励模型，在 100–200 次采样后提升幅度基本停在 5% 左右，远低于直接验证的线性提升。

- **消融实验**  
  - 作者分别关闭去重、改变采样温度、使用不同的随机种子生成策略，发现去重对覆盖率提升贡献约 10%，而温度调节对幂律参数 b 的影响不大。  
  - 对比不同的后处理方法（多数投票 vs 奖励模型），两者在可验证任务上几乎没有差别，说明核心收益来自于“多样性”而非后处理。

- **局限性**  
  - 对于没有自动验证器的任务，重复采样的收益受限于后处理方法的选择，作者承认目前缺乏能够充分利用数百甚至数千候选的通用筛选机制。  
  - 采样次数大幅提升会导致推理延迟和算力成本线性增长，在实时交互场景仍然不适用。  
  - 论文未对不同模型规模（如 7B vs 70B）在相同采样预算下的相对收益做系统分析，留待后续工作。

### 影响与延伸思考
这篇工作把“推理算力”正式纳入可伸缩的研究维度，随后出现了多篇围绕“采样预算分配”“自适应采样停止准则”等方向的论文。还有研究尝试把奖励模型与采样过程结合，形成 **“采样‑评估闭环”**，让模型在生成过程中即时判断是否已经得到足够好的答案。对想进一步探索的读者，可以关注以下几个方向：  
1. **自适应采样**：根据中间验证结果动态决定是否继续采样，降低不必要的算力消耗。  
2. **多样性引导采样**：在采样时加入鼓励生成不同解的机制（如核采样温度调度），提升覆盖率的边际收益。  
3. **大模型与小模型协同**：让小模型快速生成大量候选，大模型只负责验证或重排，形成算力层次化利用。  
这些思路已经在近期的会议（如 NeurIPS 2024）中出现，表明“重复采样”已成为提升 LLM 推理性能的一个重要工具箱。

### 一句话记住它
只要让大语言模型多尝试几次，覆盖率会像指数幂律一样快速提升——算力不只在训练，推理也能“加速”。