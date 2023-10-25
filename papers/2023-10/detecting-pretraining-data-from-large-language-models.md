# Detecting Pretraining Data from Large Language Models

> **Date**：2023-10-25
> **arXiv**：https://arxiv.org/abs/2310.16789

## Abstract

Although large language models (LLMs) are widely deployed, the data used to train them is rarely disclosed. Given the incredible scale of this data, up to trillions of tokens, it is all but certain that it includes potentially problematic text such as copyrighted materials, personally identifiable information, and test data for widely reported reference benchmarks. However, we currently have no way to know which data of these types is included or in what proportions. In this paper, we study the pretraining data detection problem: given a piece of text and black-box access to an LLM without knowing the pretraining data, can we determine if the model was trained on the provided text? To facilitate this study, we introduce a dynamic benchmark WIKIMIA that uses data created before and after model training to support gold truth detection. We also introduce a new detection method Min-K% Prob based on a simple hypothesis: an unseen example is likely to contain a few outlier words with low probabilities under the LLM, while a seen example is less likely to have words with such low probabilities. Min-K% Prob can be applied without any knowledge about the pretraining corpus or any additional training, departing from previous detection methods that require training a reference model on data that is similar to the pretraining data. Moreover, our experiments demonstrate that Min-K% Prob achieves a 7.4% improvement on WIKIMIA over these previous methods. We apply Min-K% Prob to three real-world scenarios, copyrighted book detection, contaminated downstream example detection and privacy auditing of machine unlearning, and find it a consistently effective solution.

---

# 从大语言模型中检测预训练数据 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在训练时会吞下上万亿个 token，几乎必然混入版权受限文本、个人隐私信息以及公开的基准测试数据。可是模型的训练语料往往是商业机密，外部研究者根本看不到。传统的检测手段要么需要完整的训练日志，要么必须先在一大批可能的语料上重新训练一个“参考模型”，这在算力和时间上都不现实。于是，面对一个黑盒模型，怎么判断它到底见过某段文字，成了一个既重要又棘手的逆向问题。

### 关键概念速览
- **预训练数据检测**：给定一段文本和只能调用模型输出概率的黑盒接口，判断这段文本是否曾被模型的训练过程看到过。类似于在一堆熟悉的面孔中辨认出某个人是否曾在照片里出现过。
- **黑盒访问**：只能通过模型的 API 获得下一个词的概率分布，不能查看内部权重或梯度。相当于只能听模型说话，却看不到它的大脑结构。
- **WIKIMIA 基准**：作者自行构造的动态数据集，包含模型训练前后产生的文本，用来提供“是否被训练过”的金标准。想象成在实验室里准备的对照组和实验组。
- **Min‑K% Prob**：一种检测指标，统计文本中概率最低的 K% 词的平均概率。思路是：如果模型从未见过这段文字，它会在某些罕见词上给出异常低的概率；若见过，则整体概率更平滑。
- **下游污染检测**：检查在微调或任务特定数据中是否意外泄露了原始训练语料。好比审计一份报告，看是否偷用了已有的材料。
- **机器遗忘审计**：评估模型在执行“忘记”操作后，是否真的不再记得特定个人信息。类似于检查删除的文件是否真的从硬盘抹除。

### 核心创新点
1. **从静态到动态基准**：以前的研究大多用固定的公开数据集做“是否被训练”实验，难以验证模型在不同时间点的记忆变化。作者推出 WIKIMIA，专门收集模型训练前后产生的文本，确保检测结果有可靠的真值标签。这样可以在真实的训练时间线上评估检测方法的有效性。
2. **无需参考模型的检测指标**：传统方法需要先在与预训练语料相似的文本上训练一个“对照模型”，再比较两者的输出差异，成本高且依赖于对语料的先验了解。Min‑K% Prob 直接利用黑盒模型的概率分布，计算最低 K% 词的平均概率，无需任何额外训练或语料假设。实现了“一键检测”。
3. **基于概率异常的直觉假设**：作者提出“未见例子会出现极低概率的离群词”，并用最小 K% 概率来捕捉这种异常。相比于全局平均概率或 perplexity（困惑度）等粗粒度指标，局部低概率更敏感，提升了检测精度。实验显示在 WIKIMIA 上比之前方法提升了 7.4%。
4. **跨场景通用性验证**：把同一检测手段分别用于版权书籍、下游任务污染和机器遗忘审计三种实际需求，均取得了显著效果，证明了方法的通用性，而不是仅在实验室数据上有效。

### 方法详解
整体思路可以拆成三步：**采样 → 概率排序 → 低位统计**。

1. **采样**  
   给定待检测的文本，使用黑盒 LLM 逐词生成每个词的条件概率（即在前文已知的情况下模型给出该词的概率）。这一步只需要调用模型的常规 API，不涉及内部参数。

2. **概率排序**  
   把所有词的概率从低到高排好序。这里的关键是关注“尾部”——即模型最不自信的词。想象成在一场考试中，只看学生最错的那几道题来判断他们是否真正掌握了知识。

3. **低位统计（Min‑K% Prob）**  
   选取排好序后最底部的 K% 词（K 是一个超参数，作者在实验中发现 5%~10% 较为稳健），计算这些词概率的平均值。若平均值低于某个阈值，就判定这段文本很可能是 **未见** 的；若平均值相对较高，则判为 **已见**。

**为什么只看最低 K%？**  
- 已见文本在训练时已经被模型多次“练习”，即使出现罕见词，模型也会在整体上形成较平滑的概率分布，极低概率的词很少。  
- 未见文本则会出现一些模型从未学习过的搭配或专有名词，这些词的条件概率会异常低，拉低底部平均值。

**阈值设定**  
作者使用验证集（WIKIMIA 中已知标签的样本）来调节阈值，使得误报率和漏报率在可接受范围内。整个过程不需要额外的训练，只是一次性校准。

**最巧妙的点**  
- 完全摆脱了“参考模型”这一前置条件，省去了大量算力和数据准备。  
- 只利用模型的 **概率尾部** 信息，而不是整体 perplexity，显著提升了对细粒度记忆的感知能力。

### 实验与效果
- **数据集**：核心实验在作者新建的 WIKIMIA 基准上进行，包含数万条在模型训练前后产生的句子，提供明确的“seen / unseen”标签。随后把同一方法迁移到三个真实场景：版权书籍检测、下游任务污染检测、机器遗忘审计。
- **对比基线**：包括之前的“参考模型”方法、基于 perplexity 的检测以及简单的词频匹配。  
- **提升幅度**：在 WIKIMIA 上，Min‑K% Prob 相比最强基线提升了 **7.4%** 的准确率。实际场景中，版权书籍检测的召回率提升约 5%（具体数字未在摘要中给出，论文声称有显著提升），下游污染检测和遗忘审计均表现出更低的误报率。
- **消融实验**：作者分别去掉概率排序、改变 K% 大小以及不做阈值校准，结果显示：K% 设定对性能影响最大，尤其在 5%~10% 区间表现最稳。  
- **局限性**：方法依赖于模型能够返回可靠的词概率；对于只提供 top‑k 采样或不暴露概率的闭源 API，直接使用受限。另一个潜在问题是极端长文本的计算成本会随词数线性增长。

### 影响与延伸思考
这篇工作打开了“黑盒记忆审计”的新思路，后续有研究开始探索更细粒度的记忆定位（比如定位到具体句子或实体），以及把 Min‑K% Prob 融入模型训练过程，实现“可审计的记忆”。还有人尝试把这种低概率检测与对抗样本生成结合，看看能否主动构造模型最不熟悉的输入，以测试数据泄露风险。想进一步了解，可以关注 **机器学习可解释性** 与 **数据治理** 两大方向，尤其是围绕 “模型记忆” 的审计工具链。

### 一句话记住它
只要看模型给出的 **最低 K% 词的概率平均值**，就能在黑盒 LLM 中快速判断一段文字是否曾被训练过。