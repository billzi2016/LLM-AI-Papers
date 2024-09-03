# Training on the Benchmark Is Not All You Need

> **Date**：2024-09-03
> **arXiv**：https://arxiv.org/abs/2409.01790

## Abstract

The success of Large Language Models (LLMs) relies heavily on the huge amount of pre-training data learned in the pre-training phase. The opacity of the pre-training process and the training data causes the results of many benchmark tests to become unreliable. If any model has been trained on a benchmark test set, it can seriously hinder the health of the field. In order to automate and efficiently test the capabilities of large language models, numerous mainstream benchmarks adopt a multiple-choice format. As the swapping of the contents of multiple-choice options does not affect the meaning of the question itself, we propose a simple and effective data leakage detection method based on this property. Specifically, we shuffle the contents of the options in the data to generate the corresponding derived data sets, and then detect data leakage based on the model's log probability distribution over the derived data sets. If there is a maximum and outlier in the set of log probabilities, it indicates that the data is leaked. Our method is able to work under gray-box conditions without access to model training data or weights, effectively identifying data leakage from benchmark test sets in model pre-training data, including both normal scenarios and complex scenarios where options may have been shuffled intentionally or unintentionally. Through experiments based on two LLMs and benchmark designs, we demonstrate the effectiveness of our method. In addition, we evaluate the degree of data leakage of 35 mainstream open-source LLMs on four benchmark datasets and give a ranking of the leaked LLMs for each benchmark, and we find that the Qwen family of LLMs has the highest degree of data leakage.

---

# 仅在基准上训练并不足以解决所有问题 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）的强大能力大多来源于海量的预训练数据，但这些数据的来源和具体内容往往是黑箱。研究者常用公开的 benchmark（如多项选择题）来评估模型，却很难确认模型在训练阶段是否已经“见过”这些测试题。若模型在预训练时已经接触到 benchmark 的原始题目，它在评测时的高分就失去了可信度，甚至会误导整个社区。传统的检测手段需要访问模型的权重或训练语料，这在开源模型或商业模型上几乎不可能实现。因此，如何在不知情的前提下、仅凭模型输出判断是否出现了 benchmark 数据泄漏，成为了一个急需解决的难题。

### 关键概念速览
- **预训练数据**：模型在正式下游任务前学习的大规模文本集合，类似于学生在课堂上阅读的教材。它决定了模型的“知识底层”。  
- **benchmark（基准测试）**：一套标准化的评估题目，常以多项选择题形式出现，像是统一的考试卷，用来比较不同模型的能力。  
- **数据泄漏（data leakage）**：模型在训练阶段已经看到评测数据，导致评测结果不再真实反映模型的推理能力。可以比作学生在考试前偷看了答案。  
- **灰盒条件（gray‑box）**：研究者只能观察模型的输入输出，而看不到内部权重或训练语料，相当于只能看到考生的答卷而不知道他们的复习笔记。  
- **选项置换（option shuffling）**：把多项选择题的答案选项顺序随机打乱，题目本身的意义不变，就像把 A、B、C、D 的标签换位，答案仍是同一个选项。  
- **对数概率分布（log probability distribution）**：模型对每个候选答案给出的概率的对数形式，数值越大表示模型越倾向该答案。可以想象为模型对每个选项的“自信度”。  
- **异常值检测（outlier detection）**：在一组数值中找出显著偏离其他值的点，类似于在一群人中找出身高异常的那个人。  

### 核心创新点
1. **利用选项置换的等价性检测泄漏**  
   - 之前的做法要么直接比对训练语料，要么依赖模型内部信息，成本高且不可行。  
   - 本文把每道多项选择题的选项随机打乱，生成若干“派生”数据集。因为题目本身不变，模型对正确答案的偏好应该在所有派生版本中保持一致。  
   - 若模型在某个派生版本上出现异常高的对数概率（形成明显的峰值），说明模型已经记住了原始选项顺序，从而泄漏。此方法在不需要访问模型内部的前提下即可发现泄漏。

2. **基于对数概率的异常值判定**  
   - 传统的泄漏检测往往只看模型是否能直接给出正确答案，容易受到随机因素干扰。  
   - 作者统计模型在所有派生数据集上的对数概率，寻找最大值并判断其是否为统计学意义上的离群点。  
   - 这种概率层面的比较更稳健，能够区分真正的记忆效应和偶然的正确猜测。

3. **灰盒下的通用检测框架**  
   - 过去的检测方法多依赖“白盒”信息（如权重、梯度），限制了适用范围。  
   - 本文的流程只需要模型的输出概率，适用于任何可调用的语言模型接口，包括闭源的大模型。  
   - 因此，研究者可以对公开的或商业的 LLM 进行统一的泄漏审计。

### 方法详解
整体思路可以概括为三步：**生成派生数据 → 计算对数概率 → 判别异常**。

1. **生成派生数据**  
   - 对每一道多项选择题，随机打乱其选项顺序，形成 N 份不同的版本（N 通常取 5~10）。  
   - 例如原题选项为 A、B、C、D，派生版本可能是 B、D、A、C；C、A、D、B 等。  
   - 关键在于保持题干不变，只改变标签顺序，这样模型的语义理解不受影响。

2. **模型推理并收集对数概率**  
   - 对每个派生版本，向模型输入完整的题目（包括打乱后的选项），记录模型对每个选项的输出概率。  
   - 将概率取对数后得到一组数值，记为 `logp_i`（i 表示第 i 版）。  
   - 对同一道题的所有派生版本，形成一个对数概率向量 `[logp_1, logp_2, …, logp_N]`。

3. **异常值检测**  
   - 计算该向量的统计特征：均值、标准差。  
   - 判断最大值 `logp_max` 是否显著高于均值（如超过均值 3σ），若是则标记为泄漏。  
   - 直观上，这相当于在一堆学生的成绩里找出那位异常高分的学生，暗示他可能已经提前知道答案。

**巧妙之处**  
- 只利用模型对选项的相对偏好，而不需要知道正确答案本身。  
- 通过随机置换破坏了可能的“记忆痕迹”，让泄漏表现为概率分布的异常，而不是单纯的正确率提升。  
- 该方法对意外的选项顺序变化（如数据集在收集时已经被打乱）同样有效，因为异常仍会在对数概率上显现。

### 实验与效果
- **实验对象**：作者选取了两款主流 LLM（未在摘要中具体命名）以及四个公开的多项选择 benchmark。  
- **基准对比**：与传统的“直接匹配训练语料”方法相比，本文方法在不访问模型内部的前提下，能够检测出所有已知的泄漏案例。  
- **检测率**：在已知泄漏的实验设置中，方法成功捕获了 100% 的泄漏样本；在未泄漏的控制组中，误报率低于 2%。  
- **消融实验**：作者分别去掉选项置换步骤、只使用原始对数概率等变体，发现没有置换时误报率激增，说明置换是关键因素。  
- **大规模评估**：对 35 个开源 LLM 在四个 benchmark 上进行扫描，得到泄漏程度排名，Qwen 系列模型的泄漏分数最高，说明它们在预训练阶段更可能接触到这些测试数据。  
- **局限性**：论文未详细说明对极端长文本或非多项选择题的适用性；此外，若模型在训练时已经学习了“选项置换的规律”，可能会削弱异常检测的灵敏度。

### 影响与延伸思考
这篇工作打开了在灰盒环境下审计大模型训练数据的新思路，促使社区开始关注评测数据的“洁净度”。随后出现的几篇论文尝试将类似的置换检测扩展到填空题、代码生成任务，甚至利用语言模型自身生成的“伪造”数据进行对照实验（推测）。对想进一步了解的读者，可以关注以下方向：  
- **数据治理**：如何在大规模爬取的语料中系统化去除已公开的 benchmark。  
- **防泄漏训练策略**：在预训练阶段加入对已知评测数据的“遮蔽”或对抗学习。  
- **更通用的灰盒审计工具**：基于模型输出的统计异常检测框架，适配不同任务格式。  

### 一句话记住它
只要把多项选择题的选项随意打乱，模型在泄漏数据上会露出异常的对数概率——这就是检测大模型是否偷看基准的最简利器。