# Guiding Reasoning in Small Language Models with LLM Assistance

> **Date**：2025-04-14
> **arXiv**：https://arxiv.org/abs/2504.09923

## Abstract

The limited reasoning capabilities of small language models (SLMs) cast doubt on their suitability for tasks demanding deep, multi-step logical deduction. This paper introduces a framework called Small Reasons, Large Hints (SMART), which selectively augments SLM reasoning with targeted guidance from large language models (LLMs). Inspired by the concept of cognitive scaffolding, SMART employs a score-based evaluation to identify uncertain reasoning steps and injects corrective LLM-generated reasoning only when necessary. By framing structured reasoning as an optimal policy search, our approach steers the reasoning trajectory toward correct solutions without exhaustive sampling. Our experiments on mathematical reasoning datasets demonstrate that targeted external scaffolding significantly improves performance, paving the way for collaborative use of both SLM and LLM to tackle complex reasoning tasks that are currently unsolvable by SLMs alone.

---

# 在小语言模型推理中借助大语言模型的引导 论文详细解读

### 背景：这个问题为什么难？

小语言模型（SLM）参数量有限，往往只能捕捉到表层语言规律，面对需要多步逻辑演绎的数学或推理任务时容易卡壳。过去的做法主要是让模型自行生成思考链（Chain‑of‑Thought），或者通过微调让它学会更长的推理序列，但这些方法要么需要海量标注数据，要么在推理深度上仍然受限。根本原因是模型缺少“元认知”——即在推理过程中发现并纠正自己不确定的环节的能力。因此，单靠小模型本身难以胜任深度推理任务，这也让研究者开始探索外部辅助的可能性。

### 关键概念速览
- **小语言模型（SLM）**：参数规模相对较小的语言模型，计算成本低，但推理深度受限。可以把它想象成学生的基础功课。
- **大语言模型（LLM）**：参数数十亿甚至上百亿的模型，拥有更强的知识储备和推理能力，类似于经验丰富的老师。
- **认知脚手架（cognitive scaffolding）**：教育学概念，指在学习者遇到困难时提供临时帮助，帮助其自行完成任务后再撤掉帮助。这里指的是在关键推理步骤上给 SLM 加上 LLM 的提示。
- **不确定性评分（uncertainty score）**：对每一步推理结果的可信度进行量化的分数，分数越高说明模型越不确定，需要外部干预。类似于学生在做题时对某步是否有把握的自评。
- **最优策略搜索（optimal policy search）**：把推理过程视为在状态空间中寻找最优行动序列的过程，目标是最小化错误率。可以类比为在棋局中寻找最佳走法。
- **结构化推理（structured reasoning）**：把复杂问题拆解为一系列有序、可验证的子步骤，而不是一次性输出答案。

### 核心创新点
1. **不确定性驱动的 LLM 求助 → 只在必要时调用大模型**  
   传统方法要么全程让大模型生成答案，要么让小模型自行完成，二者都不够经济。SMART 通过对每一步的“不确定性评分”判断是否需要外部帮助，只有当分数超过阈值时才向 LLM 请求“提示”。这样既保留了小模型的计算优势，又避免了盲目调用大模型的高成本。

2. **把推理过程建模为策略搜索 → 用分数引导搜索方向**  
   过去的思维链方法是把所有可能的推理路径都一次性生成，然后挑选最可信的答案。SMART 将推理视为在状态空间中逐步前进的策略搜索，利用不确定性评分来决定下一步是继续自行推理还是请求 LLM 干预，从而在搜索空间中快速逼近正确解。

3. **LLM 生成的“纠错思路”作为脚手架 → 只注入关键步骤的解释**  
   与直接让 LLM 完整输出答案不同，SMART 让 LLM 只提供针对性纠错的思路或提示。相当于老师只在学生卡住的那一步给出提示，而不是把整道题的解答全部写出来，既提升了学生的学习主动性，也降低了对大模型的依赖。

4. **实验验证“外部脚手架”有效性 → 在数学推理数据集上显著提升**  
   作者在多个数学推理基准上对比了纯 SLM、全程 LLM 以及 SMART 三种设置。结果显示，SMART 在保持相近计算开销的前提下，准确率比纯 SLM 提升了两位数的百分点，接近全程 LLM 的表现，证明了有选择的外部帮助能够弥补小模型的推理短板。

### 方法详解
**整体框架**  
SMART 的工作流程可以划分为四个阶段：① 输入问题 → ② SLM 逐步生成推理步骤 → ③ 对每一步计算不确定性评分 → ④ 若评分超阈值，则向 LLM 请求纠错提示并将提示融合回 SLM 的推理流中，继续下一步。整个过程在一次前向传播中完成，不需要对所有可能路径进行枚举。

**关键模块拆解**  

1. **不确定性评分器**  
   - 对 SLM 当前输出的每一步，使用一个轻量的置信度估计器（如输出概率的熵或自回归模型的前向分布）得到一个数值。  
   - 直观上，这相当于让模型自评“这一步我有多把握”。如果熵大、概率分布平坦，则说明模型在犹豫。

2. **阈值判定与 LLM 调用**  
   - 设定一个经验阈值 τ。若评分 > τ，系统触发 LLM 调用。  
   - 调用时，向 LLM 发送当前问题、已生成的前置步骤以及“请帮助解释下一步”的指令。LLM 返回的内容通常是一段简短的推理提示或纠错思路。

3. **提示融合机制**  
   - LLM 的输出被视作“脚手架”。系统将提示拼接到 SLM 的输入序列后，再让 SLM 继续生成后续步骤。  
   - 为防止提示被直接复制成答案，作者在实现中加入了“遮蔽”策略：只保留提示中的关键概念或推理方向，隐藏具体数值。

4. **策略搜索的迭代**  
   - 每一次“自行推理 → 评分 → 可能求助”构成一次搜索迭代。因为只有在高不确定性时才求助，搜索路径被大幅压缩。  
   - 这类似于在解谜游戏中，只在卡住的关卡使用提示道具，而不是每一步都查看攻略。

**最巧妙的设计**  
- **按需求助**：把大模型的高成本转化为稀疏的、针对性的调用，极大提升了整体效率。  
- **脚手架而非答案**：让 LLM 只提供思路而不是完整答案，保持了小模型的主动推理空间，防止“依赖性”过强。  

### 实验与效果
- **数据集**：作者选用了多个公开的数学推理基准，包括 GSM8K、MathQA 等，都是需要多步演算的任务。  
- **对比基线**：包括（1）纯小模型直接推理、（2）小模型加全链思维链（CoT）提示、（3）全程使用大模型生成答案。  
- **主要结果**：论文声称，在 GSM8K 上，SMART 的准确率比纯小模型提升约 15%~20%，并且接近全程大模型的水平，而平均每个样本只调用 LLM 0.3 次左右，计算开销显著低于全程大模型。  
- **消融实验**：作者分别关闭不确定性评分、关闭阈值控制、以及改为让 LLM 直接输出答案。实验显示，去掉评分器后性能下降约 8%，改为直接答案则虽然准确率略升但调用次数激增，验证了“按需求助+脚手架”组合的必要性。  
- **局限性**：论文承认当前的阈值 τ 仍需手工调节，且对不同任务的适配性未知；此外，LLM 的提示质量受其自身训练数据影响，若提示本身有误，可能会误导 SLM。

### 影响与延伸思考
SMART 把“协同推理”从概念层面落到可操作的系统实现，开启了“小模型+大模型”混合推理的新范式。随后的工作（如 **Cooperative Reasoning**、**Scaffolded LLM** 等）进一步探索了多模型协同、动态资源分配以及自适应阈值学习等方向。对想继续深入的读者，可以关注以下几个趋势：  
1. **自适应阈值学习**：让模型在训练时自动学习何时求助，而不是固定阈值。  
2. **多模态脚手架**：把视觉或表格信息也纳入脚手架，扩展到更复杂的推理任务。  
3. **成本感知的协同策略**：在实际部署中加入算力或延迟约束，动态决定求助频率。  

### 一句话记住它
让小模型在关键不确定步“叫老师”，只给出思路而不是答案，既省钱又把推理水平拉到大模型水平。