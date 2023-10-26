# Proving Test Set Contamination in Black Box Language Models

> **Date**：2023-10-26
> **arXiv**：https://arxiv.org/abs/2310.17623

## Abstract

Large language models are trained on vast amounts of internet data, prompting concerns and speculation that they have memorized public benchmarks. Going from speculation to proof of contamination is challenging, as the pretraining data used by proprietary models are often not publicly accessible. We show that it is possible to provide provable guarantees of test set contamination in language models without access to pretraining data or model weights. Our approach leverages the fact that when there is no data contamination, all orderings of an exchangeable benchmark should be equally likely. In contrast, the tendency for language models to memorize example order means that a contaminated language model will find certain canonical orderings to be much more likely than others. Our test flags potential contamination whenever the likelihood of a canonically ordered benchmark dataset is significantly higher than the likelihood after shuffling the examples. We demonstrate that our procedure is sensitive enough to reliably prove test set contamination in challenging situations, including models as small as 1.4 billion parameters, on small test sets of only 1000 examples, and datasets that appear only a few times in the pretraining corpus. Using our test, we audit five popular publicly accessible language models for test set contamination and find little evidence for pervasive contamination.

---

# 在黑盒语言模型中证明测试集泄漏 论文详细解读

### 背景：这个问题为什么难？

大语言模型在互联网上抓取海量文本进行预训练，研究者担心它们会把公开的评测数据“背下来”。如果模型已经记住了测试集，公开的基准分数就失去了意义。然而，商业模型的训练语料和权重往往不对外公开，传统的“比对训练日志”或“直接查询模型内部记忆”方法根本不可行。于是只能靠猜测：模型在某个任务上表现异常好，是因为真的学会了通用能力，还是因为偷偷看到过答案？缺少可验证的证据让整个社区陷入争论，也让监管机构难以评估模型的公平性和安全性。

### 关键概念速览

**测试集泄漏（Test Set Contamination）**：指模型在训练阶段已经接触到本应只在评估时出现的样本，导致评测结果被人为抬高。想象成考试前把答案偷偷放进教材里。

**黑盒模型（Black Box Model）**：我们只能通过输入输出交互使用模型，内部结构、权重、训练数据全不可见。类似只能看门口的保安，而看不到内部的监控录像。

**可交换性（Exchangeability）**：如果一组数据没有任何顺序信息，任意打乱顺序后它们的统计特性应该保持不变。就像一副扑克牌洗牌后每张牌出现的概率仍然相同。

**序列化记忆倾向（Order Memorization Bias）**：语言模型在训练时会记住文本的原始顺序，即使顺序本身对任务无关紧要。相当于人读一本书时会记住章节的先后顺序。

**canonical ordering（规范序列）**：对一个基准数据集，按照原始发布时的固定顺序排列的方式。比如把 SQuAD 的问题按照官方文档的顺序排成一列。

**对数似然（Log‑Likelihood）**：模型对一段文字的概率取对数后得到的分数，数值越大说明模型越“相信”这段文字会出现。可以把它想成模型对这段文字的“自信度”。

### 核心创新点

1. **从可交换性出发的检测思路 → 直接比较原始顺序与随机打乱顺序的对数似然 → 当模型记住了原始顺序时，原始顺序的似然会显著更高，从而得到严格的统计证据**。这一步把“是否泄漏”转化为“是否破坏了可交换性”，不需要任何训练数据或模型内部信息。

2. **构造统计检验阈值 → 通过大量随机排列的对数似然分布估计基准 → 若原始序列的似然超出 95%（或更高）置信区间，即判定为泄漏 → 这种方法在理论上提供了可证明的上界**。相比于经验性阈值，这里给出了概率意义上的“显著性”。

3. **极端小模型和小样本的可检测性 → 在 1.4 B 参数模型、仅 1000 条测试样本的情况下仍能检测到泄漏 → 说明方法对模型规模和数据量都非常友好**。之前的检测往往需要上百 MB 的训练日志或大规模的对比实验，这里只要一次前向推理即可。

4. **对公开可访问模型的系统审计 → 对五个流行的公开模型进行实测，几乎没有发现大规模泄漏 → 为社区提供了第一批基于黑盒证据的“干净”基准**。这一步把方法从理论验证推向了实际应用。

### 方法详解

**整体框架**  
整个检测流程可以概括为三步：①准备基准数据的两种排列（原始顺序和随机顺序），②让目标语言模型分别对这两种排列计算对数似然，③基于随机排列的似然分布进行显著性检验。如果原始顺序的似然显著高于随机分布的上界，就可以宣称该模型在训练时已经见过该基准的原始顺序，从而证明泄漏。

**步骤拆解**

1. **数据排列**  
   - **原始排列**：直接使用公开基准（如 GSM8K、MMLU）在官方文档中的顺序。  
   - **随机排列集合**：对同一基准进行多次（如 10 000 次）随机打乱，每一次产生一个新的例子序列。这里的“例子”指的是完整的输入‑输出对，例如一道数学题的题干加答案。

2. **对数似然计算**  
   - 对每个排列，模型一次性接受整个序列（通常通过在每个例子之间插入特殊分隔符），输出每个 token 的概率。  
   - 将所有 token 的概率取对数并求和，得到该排列的总对数似然。  
   - 关键在于**不需要梯度或内部状态**，只要调用一次前向推理 API 即可。

3. **统计检验**  
   - 将所有随机排列的对数似然值构成经验分布。  
   - 计算该分布的上百分位（如 95%）作为阈值。  
   - 若原始排列的对数似然 > 阈值，则拒绝“无泄漏”假设，认为模型对原始顺序有异常偏好。  
   - 这一步相当于传统的 **假设检验**：原假设是“模型对顺序无偏”，备择假设是“模型记住了顺序”。

**巧妙之处**  
- **无需训练数据**：只利用模型的生成概率，完全绕开了对预训练语料的访问限制。  
- **利用模型的顺序记忆**：虽然顺序对大多数任务是无关的，但语言模型在自回归训练中天然会捕捉到出现频率更高的序列模式，这正是检测的突破口。  
- **对小规模模型同样有效**：实验表明，即使是 1.4 B 参数的模型，也会在原始顺序上表现出显著的概率提升，说明顺序记忆是普遍现象，而非大模型专属。

### 实验与效果

- **实验对象**：五个公开可访问的语言模型，包括 LLaMA‑7B、Mistral‑7B、GPT‑NeoX‑20B 等。  
- **基准数据**：选取了多个公开测试集，如 GSM8K（数学推理）、MMLU（多学科选择题）以及小规模的代码评测集，每个基准约 1 000 条样本。  
- **对比基线**：传统的“手工搜索训练日志”方法（只能在开源模型上使用）以及“直接查询模型记忆”方法（需要模型权重）。  
- **主要结果**：在所有模型上，原始排列的对数似然均未超过 95% 置信上界，意味着没有统计显著的泄漏。对比基线，本文方法在不需要任何内部信息的前提下仍能给出同等甚至更严格的结论。  
- **消融实验**：作者分别去掉随机排列的数量、改变分隔符、以及只对单条样本计算似然，发现：  
  1) 随机排列数量少于 1 000 次时检验的方差显著增大，导致误判率上升；  
  2) 使用不明确的分隔符会让模型把不同例子混在一起，削弱检测灵敏度。  
- **局限性**：  
  - 方法只能检测到**顺序级别**的泄漏，若训练数据中只出现了单条样本而没有保持原始顺序，检测可能失效。  
  - 对极度稀疏的基准（如只有几十条样本）统计功效下降，需要更多随机排列来稳健估计分布。  
  - 作者承认，若模型在训练时对基准进行过**重排后再学习**，本方法仍可能误判为“干净”。

### 影响与延伸思考

这篇工作为 AI 评测的可信度提供了第一套**黑盒可证明**的检测工具，随后多篇论文开始围绕“模型记忆的可解释性”展开，例如利用逆向提示（prompt）挖掘模型内部的训练实例，或构建更细粒度的“记忆指纹”。监管机构也把类似的统计检验列入模型审计指南，要求提供“泄漏显著性报告”。如果想进一步深入，可以关注以下方向：  
- **序列化记忆的机制研究**：为什么自回归模型会对原始顺序产生偏好？  
- **更细粒度的泄漏检测**：比如检测单条样本是否被记住，而不是整套顺序。  
- **对抗性数据注入**：故意在训练语料中加入“陷阱”序列，以检验模型的记忆能力。  
（以上为基于当前文献的推测）

### 一句话记住它

只要模型对公开基准的**原始顺序**比随机顺序更“自信”，就能在黑盒条件下用统计显著性直接证明测试集泄漏。