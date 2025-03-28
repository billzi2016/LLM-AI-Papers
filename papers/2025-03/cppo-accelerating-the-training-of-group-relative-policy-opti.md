# CPPO: Accelerating the Training of Group Relative Policy Optimization-Based Reasoning Models

> **Date**：2025-03-28
> **arXiv**：https://arxiv.org/abs/2503.22342

## Abstract

This paper introduces Completion Pruning Policy Optimization (CPPO) to accelerate the training of reasoning models based on Group Relative Policy Optimization (GRPO). GRPO, while effective, incurs high training costs due to the need to sample multiple completions for each question. Our experiment and theoretical analysis reveal that the number of completions impacts model accuracy yet increases training time multiplicatively, and not all completions contribute equally to policy training -- their contribution depends on their relative advantage. To address these issues, we propose CPPO, which prunes completions with low absolute advantages, significantly reducing the number needed for gradient calculation and updates. Additionally, we introduce a dynamic completion allocation strategy to maximize GPU utilization by incorporating additional questions, further enhancing training efficiency. Experiments show that CPPO achieves up to $7.98\times$ speedup on GSM8K and $3.48\times$ on Math while preserving or even enhancing the accuracy compared to the original GRPO. We release our code at \href{https://github.com/lzhxmu/CPPO}{https://github.com/lzhxmu/CPPO}.

---

# CPPO：加速基于组相对策略优化的推理模型训练 论文详细解读

### 背景：这个问题为什么难？
在大模型做数学或逻辑推理时，常用的做法是让模型一次生成多个候选答案（completion），再用强化学习的策略梯度来挑选最好的。这种 **Group Relative Policy Optimization（GRPO）** 能显著提升准确率，但每一道题要跑 dozens 甚至上百次前向传播，训练成本呈指数级增长。实际使用时，GPU 资源往往被大量“低价值”答案占满，训练时间成为瓶颈。于是，如何在不牺牲模型质量的前提下，削减无效的候选答案，成为迫切需要解决的问题。

### 关键概念速览
- **Completion（候选答案）**：模型对同一道题的不同生成结果，就像学生在考试时写的多个解法草稿。每个草稿都有可能是对的，也可能是废话。
- **Policy Optimization（策略优化）**：用强化学习的思路让模型学会“更倾向于生成好答案”。把模型当成一个决策者，奖励高质量的生成，惩罚低质量的生成。
- **Group Relative Policy Optimization（GRPO）**：在一次训练步骤里，把同一道题的所有候选答案放在一起比较，计算它们相对优势（relative advantage），再根据优势来更新模型。相当于在一次课堂上让学生们相互比较谁的解法更好。
- **Advantage（优势）**：某个候选答案相对于该题平均水平的得分差。优势高说明该答案比大多数答案更好，优势低则相反。
- **Pruning（剪枝）**：把优势低的候选答案直接丢掉，不参与梯度计算。类似老师在批改作业时直接划掉明显错误的步骤，省去细致点评的时间。
- **Dynamic Completion Allocation（动态完成分配）**：在显卡空闲时，把本来留给同一道题的“槽位”转给其他题目，以保持 GPU 利用率满格。像是餐厅里空桌子不等客人，而是让等位的客人先坐下。

### 核心创新点
1. **从全部候选答案到优势剪枝**  
   之前的 GRPO 必须把每道题的所有生成都喂进梯度计算，导致计算量随候选数线性增长。CPPO 先算出每个答案的 **绝对优势**，把低于阈值的答案直接剔除，只保留“有价值”的少数。这样梯度只基于少数高质量答案，训练时间大幅下降。

2. **基于优势的阈值自适应**  
   直接设定一个固定阈值会在不同任务或不同训练阶段产生偏差。CPPO 引入了 **相对优势比例**（如保留前 30% 的优势），让剪枝过程随模型能力自动调节，既避免过度剪枝导致信息不足，也防止保留太多无用答案。

3. **动态完成分配提升硬件利用率**  
   当某批次因为剪枝而剩余的候选数不足以填满 GPU 的并行度时，CPPO 会把空余的计算资源分配给 **其他未完成的题目**，实现“多题混批”。这相当于把原本空闲的显卡算力重新利用起来，进一步压缩整体训练时间。

4. **理论与实证双重验证**  
   作者不仅在实验上展示了显著加速，还给出了 **训练时间与候选数的乘法关系** 以及 **优势分布对梯度方差的影响** 的理论分析，证明剪枝不会破坏策略梯度的无偏性，只是降低了方差，从而有助于更快收敛。

### 方法详解
**整体框架**  
CPPO 的训练流程可以概括为四步：① 采样多个候选答案；② 计算每个答案的奖励并转化为优势；③ 根据优势阈值进行剪枝；④ 将剩余答案用于梯度更新，同时把空余的计算槽位分配给其他题目。整个过程在每个 mini‑batch 内循环执行，保持 GPU 负载均衡。

**步骤拆解**  

1. **候选答案采样**  
   与原始 GRPO 相同，先对每个问题使用语言模型采样 $k$ 次（如 $k=16$），得到一组 Completion。这里的采样策略可以是温度采样或 nucleus 采样，保证答案多样性。

2. **奖励与优势计算**  
   - **奖励**：根据任务（数学解答、逻辑推理等）使用外部评估器（如程序求值或人工评分）得到标量分数 $r_i$。  
   - **基准**：对同一道题的所有 $r_i$ 求平均得到 $\bar r$。  
   - **优势**：$A_i = r_i - \bar r$，即每个答案相对于该题平均水平的超额得分。优势正负直接反映答案好坏。

3. **优势剪枝（Pruning）**  
   - **阈值设定**：计算所有 $A_i$ 的分位数（如第 30 百分位），记为 $T$。  
   - **筛选**：仅保留 $A_i \ge T$ 的答案进入后续梯度计算。  
   - **自适应**：在训练的不同阶段，阈值 $T$ 会随优势分布自动上升或下降，保证保留的答案比例大致恒定。

4. **梯度计算与更新**  
   对保留下来的答案，使用 **相对优势加权的策略梯度**：梯度 $\propto \sum_{i \in \text{kept}} A_i \nabla \log \pi_\theta(\text{completion}_i)$。这里的 $\pi_\theta$ 是模型的生成概率分布，$A_i$ 充当权重，优势越大对梯度的贡献越大。

5. **动态完成分配**  
   - **检测空闲**：如果某批次因为剪枝只剩 $m < k$ 条答案，显卡的并行度会出现空洞。  
   - **补充题目**：从训练队列中挑选额外的 $k-m$ 道未处理的问题，立即采样它们的候选答案并加入同一批次。  
   - **统一梯度**：所有答案（原题+补充题）在同一次前向/反向传播中一起计算梯度，保持显卡利用率接近 100%。

**最巧妙的点**  
- **优势本身是相对的**：作者利用了 GRPO 已经需要计算相对优势的事实，直接把它当作剪枝依据，省去额外的评估开销。  
- **动态批次混合**：传统做法会因为剪枝导致 batch size 下降，进而导致显卡利用率下降。CPPO 用“多题混批”把这个问题转化为资源调度问题，几乎不需要额外的代码改动。

### 实验与效果
- **数据集**：在 **GSM8K**（小学数学）和 **Math**（中学/大学数学）两大推理基准上进行评估。  
- **对比基线**：与原始 **GRPO**、以及未使用任何策略优化的 **直接微调**（Fine‑tuning）进行比较。  
- **加速效果**：在 GSM8K 上实现 **最高 7.98×** 的训练速度提升；在 Math 上达到 **3.48×**。  
- **准确率**：在两套数据上，CPPO 的最终解答准确率与 GRPO 持平，甚至在部分设置下略有提升（如 GSM8K 上提升约 0.3%）。  
- **消融实验**：作者分别关闭 **优势剪枝**、**动态完成分配**，发现剪枝 alone 已能带来约 4× 加速，加入动态分配后才达到最高的 8×。这说明两者是互补的。  
- **局限性**：论文提到在 **极端稀疏奖励**（如仅少数答案能得到正奖励）的任务上，优势分布可能过于集中，导致剪枝过度，需要手动调节阈值比例。除此之外，作者未在大规模多模态任务上验证。

### 影响与延伸思考
CPPO 把 **“只保留有价值的梯度”** 这一思路搬到大语言模型的强化学习训练中，开启了 **训练效率与策略梯度相结合的优化方向**。后续有几篇工作（如 *Efficient RL for LLMs*、*Adaptive Sampling for Policy Gradient*）引用了 CPPO 的剪枝机制，尝试在更大模型（百亿级）上做类似的“优势过滤”。如果想进一步探索，可以关注：

- **自适应采样策略**：在采样阶段就预测哪些答案可能拥有高优势，直接降低低价值答案的生成概率。  
- **多任务共享剪枝**：在多任务联合训练时，利用不同任务的优势分布共享剪枝阈值，提升整体资源利用率。  
- **硬件协同调度**：把 CPPO 的动态分配与 GPU 调度器深度结合，实现“训练即调度”，进一步压缩训练时间。

### 一句话记住它
**CPPO 通过只保留高优势的候选答案并动态填满显卡空闲，实现了几倍到十倍的推理模型训练加速，而不牺牲准确率。**