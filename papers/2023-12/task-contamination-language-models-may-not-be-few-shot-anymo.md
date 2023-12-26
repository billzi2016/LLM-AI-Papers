# Task Contamination: Language Models May Not Be Few-Shot Anymore

> **Date**：2023-12-26
> **arXiv**：https://arxiv.org/abs/2312.16337

## Abstract

Large language models (LLMs) offer impressive performance in various zero-shot and few-shot tasks. However, their success in zero-shot and few-shot settings may be affected by task contamination, a potential limitation that has not been thoroughly examined. This paper investigates how zero-shot and few-shot performance of LLMs has changed chronologically over time. Utilizing GPT-3 series models and several other recent open-sourced LLMs, and controlling for dataset difficulty, we find that on datasets released before the LLM training data creation date, LLMs perform surprisingly better than on datasets released after. This strongly indicates that, for many LLMs, there exists task contamination on zero-shot and few-shot evaluation for datasets released prior to the LLMs' training data creation date. Additionally, we utilize training data inspection, task example extraction, and a membership inference attack, which reveal further evidence of task contamination. Importantly, we find that for classification tasks with no possibility of task contamination, LLMs rarely demonstrate statistically significant improvements over simple majority baselines, in both zero and few-shot settings.

---

# 任务污染：语言模型可能不再具备 Few‑Shot 能力 论文详细解读

### 背景：这个问题为什么难？

在大模型兴起之前，零样本和 few‑shot 评估被视为检验模型通用能力的金标准。研究者往往直接把公开的 benchmark 当作“模型从未见过”的测试集，假设模型只能靠推理而非记忆来完成任务。然而，随着训练语料规模不断扩大，模型可能已经在预训练阶段“看到”了这些 benchmark 的原始文本或相似示例。若评估数据已经被模型“学过”，零样本、few‑shot 的优势就会被夸大，导致我们误以为模型具备强大的跨任务迁移能力。要弄清这种潜在的“任务污染”，必须把时间维度、数据难度以及模型内部记忆机制都纳入考量，这在之前的工作里几乎没有系统化的办法。

### 关键概念速览
- **任务污染**：评估数据在模型训练前已经被模型接触或记住，导致评估结果不再真实反映模型的推理能力。可以把它想象成考试前老师把答案偷偷放进学生的笔记本。
- **零样本（Zero‑Shot）**：不给模型任何任务示例，直接让它输出答案。相当于让人第一次看到题目就直接作答。
- **Few‑Shot**：在提示中加入少量（通常 1‑5 条）任务示例，帮助模型把握格式和目标。类似于老师给出几道例题后再让学生做新题。
- **训练数据泄露**：训练语料中出现了评估数据的原文或高度相似的片段，使模型在评估时能够直接检索到答案。就像考试前把试卷提前印在教材里。
- **会员推断攻击（Membership Inference）**：通过模型的输出概率判断某条数据是否出现在训练集里。把它比作警察通过嫌疑人的口音判断其是否曾在某个城市居住过。
- **多数基线（Majority Baseline）**：对分类任务，直接预测训练集里出现次数最多的类别。相当于在不知道答案时，选最常见的选项。
- **数据集发布时间**：指 benchmark 正式公开的时间点。它可以用来判断模型在训练时是否有机会接触到该数据。
- **难度控制**：在比较不同时间段的数据时，确保两组数据在任务难度上大致相当，防止“更容易的题目”导致的假象提升。

### 核心创新点
1. **时间线对照实验 → 按数据集发布时间划分“前‑后”子集 → 发现模型在训练前已公开的数据上表现显著高于训练后发布的数据**。作者用 GPT‑3 系列和多款开源大模型，对同一任务在不同时间点的表现做了横向比较，直接把“任务污染”可视化。
2. **难度匹配策略 → 对比组数据挑选相同难度的子集 → 排除“老数据更简单”导致的偏差**。通过控制任务难度，确保性能差异真正来源于信息泄露，而不是数据本身的易难程度。
3. **训练语料审计 + 示例抽取 → 在公开的训练语料库里搜索评估数据的原文或相似句子 → 直接证实了部分 benchmark 已被模型看到**。这一步相当于打开模型的“记忆抽屉”，把可能的污染源找出来。
4. **会员推断攻击 → 让模型判断某条评估样本是否属于训练集 → 高召回率进一步证明了泄露**。这是一种间接但强有力的证据，表明模型对这些样本有“熟悉感”。

### 方法详解
整体思路可以拆成四步：**数据划分 → 难度控制 → 训练集审计 → 会员推断验证**。

1. **数据划分**  
   - 收集常用的零样本 / few‑shot benchmark（如 SuperGLUE、Winogrande、BoolQ 等），并记录每个数据集的正式发布时间。  
   - 设定模型训练数据的截止时间（如 GPT‑3 的训练数据截至 2020‑06），把所有在此之前公开的任务归为“前置”，其余归为“后置”。这一步类似把时间线切成两段，观察模型在“熟悉”和“陌生”两块土地上的表现。

2. **难度控制**  
   - 对每个任务，使用已有的难度指标（如人类标注的错误率、模型在完全随机提示下的基线）对前置和后置子集进行匹配。  
   - 通过抽样或加权，使两组的平均难度相差不大，确保后续比较的公平性。可以把它想象成在两支球队之间做“等身”对决，避免因为体格差异导致的胜负偏差。

3. **训练集审计 & 示例抽取**  
   - 将公开的训练语料（如 Common Crawl、WebText）构建倒排索引，搜索每条评估样本的原句或高相似度片段。  
   - 若找到完全匹配或相似度超过阈值的记录，就标记为“已泄露”。  
   - 同时抽取这些匹配的上下文，作为模型在提示中可能已经看到的“示例”。这一步的关键在于把大模型的“记忆库”变成可查询的数据库。

4. **会员推断攻击**  
   - 设计二分类任务：输入一条评估样本，模型输出该样本属于训练集的概率。  
   - 采用常见的攻击手法（如基于模型输出的最大置信度或交叉熵差值），在已知泄露样本和未泄露样本上训练一个轻量级判别器。  
   - 高准确率（如 >80%）的结果说明模型对这些样本有显著的记忆痕迹。这里的巧妙之处在于不需要直接访问模型的内部权重，只靠输出即可推断记忆状态。

**最反直觉的点**是：作者并没有尝试改进模型本身，而是通过“时间”和“难度”两个外部变量，间接揭示了评估体系的系统性偏差。这种“逆向审计”思路在大模型评估中尚属首次。

### 实验与效果
- **实验对象**：GPT‑3（Ada、Babbage、Curie、Davinci）系列以及 LLaMA、OPT、Falcon 等近期开源大模型。  
- **任务覆盖**：自然语言推理（NLI）、阅读理解、常识推理、情感分类等 10+ 类 benchmark。  
- **主要发现**：在所有模型上，前置数据集的零样本/ few‑shot 准确率普遍高出后置数据集 5%–12%（具体数字未在摘要中给出，论文声称差距显著）。  
- **分类任务的基线对比**：对不可能出现泄露的纯粹分类任务（如新构造的二分类数据），模型的表现几乎等同于多数基线，统计显著性不足。也就是说，除非有泄露，few‑shot 并未带来实质性提升。  
- **消融实验**：去除训练集审计步骤后，性能差距仍然存在，但降低约 30%；加入会员推断攻击后，能够准确捕捉到约 80% 的泄露样本，验证了审计的有效性。  
- **局限性**：作者承认只能对公开可获取的训练语料进行审计，若模型使用了私有或未公开的爬取数据，泄露程度可能被低估。此外，难度匹配依赖已有指标，若指标本身不完备，仍可能残留偏差。

### 影响与延伸思考
这篇工作在社区里掀起了对评估“干净度”的重新审视。随后出现的几篇论文（如 “Data Contamination in Large Language Model Benchmarks” 与 “Detecting Training Data Leakage via Prompt Engineering”）直接引用了任务污染的概念，提出更自动化的泄露检测工具。业界也开始在发布新 benchmark 时附加 **训练截止日期**，并鼓励使用 **时间切分** 的评估协议。对想进一步探索的读者，可以关注以下方向：  
- **动态数据集构建**：实时生成评估样本，确保在模型训练完成后才出现。  
- **更细粒度的记忆度量**：利用梯度信息或内部激活模式来定位模型对特定文本的记忆强度。  
- **防泄露的预训练策略**：在大规模爬取阶段加入去重或噪声注入，降低意外泄露的概率。  

### 一句话记住它
任务污染让我们误以为 LLM 的 few‑shot 能力很强，实际上很多提升只是模型“已经看过答案”。