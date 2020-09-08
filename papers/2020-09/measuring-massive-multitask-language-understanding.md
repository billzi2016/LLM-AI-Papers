# Measuring Massive Multitask Language Understanding

> **Date**：2020-09-07
> **arXiv**：https://arxiv.org/abs/2009.03300

## Abstract

We propose a new test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average. However, on every one of the 57 tasks, the best models still need substantial improvements before they can reach expert-level accuracy. Models also have lopsided performance and frequently do not know when they are wrong. Worse, they still have near-random accuracy on some socially important subjects such as morality and law. By comprehensively evaluating the breadth and depth of a model's academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings.

---

# 大规模多任务语言理解测评 论文详细解读

### 背景：这个问题为什么难？
语言模型在单一任务（比如翻译或问答）上已经能跑出不错的成绩，但真实世界的需求往往是“一次性搞定”多种专业领域的问题。过去的评测大多聚焦于单一数据集或特定技能，无法检验模型是否真的拥有跨学科的常识和推理能力。于是我们缺少一种统一的、覆盖广泛学科的“通用考试”，导致很难判断模型的整体知识深度和实用性。

### 关键概念速览
**多任务评测**：一次性让模型回答来自不同学科的题目，就像一次大学期末综合考试。  
**随机基线**：把模型的表现和纯随机猜测的正确率做对比，类似于把学生的成绩和随意填空的分数比。  
**专家水平**：人类专业人士在相同题目上能达到的准确率，作为上限参考。  
**任务不平衡**：模型在某些科目上表现好，在另一些科目上几乎和随机一样，像学生只会数学不会历史。  
**自信度校准**：模型给出答案时的置信度是否能反映真实正确率，类似于学生对自己答对题目的把握程度。  
**社会重要任务**：涉及伦理、法律等对社会影响大的题目，错误代价高。  
**模型规模**：指模型的参数数量，通常越大模型的学习能力越强。  

### 核心创新点
**从单一任务到57任务的全景测评 → 设计了一个覆盖数学、历史、计算机、法律等57个学科的大规模测评套件 → 让研究者能够一次性看到模型在整个知识体系上的强弱点，而不是碎片化的成绩。  

**把随机猜测作为统一基准 → 统计每个任务的随机正确率并与模型表现直接对比 → 揭示了即使是最强模型在多数任务上仍接近随机，凸显真实能力缺口。  

**对模型“知道自己错了”的能力进行评估 → 记录模型对每个答案的置信度并检查其与实际正确率的对应关系 → 发现模型普遍缺乏自我校准，很多时候自信满满却答错。  

**聚焦社会关键任务的表现 → 在道德、法律等敏感领域单独报告准确率 → 发现即便是最大模型在这些任务上仍几乎是随机，提醒了实际部署的风险。  

### 方法详解
整体思路可以拆成三步：**任务集合构建 → 模型答题 → 结果分析**。

1. **任务集合构建**  
   研究团队从公开教材、考试题库以及专业测评平台挑选出57个任务，确保每个任务都有明确的单选或填空答案。题目覆盖从基础算术到美国宪法解释，形成一个类似“全科考试”的题库。每道题都标记了所属学科、难度等级以及随机猜测的基准正确率（比如四选一的题目随机基线是25%）。

2. **模型答题**  
   对每个待评估的语言模型，使用统一的提示模板让模型生成答案。例如：“请回答以下选择题：…”。模型的输出被解析为选项或数值，同时记录模型给出的置信度分数（如果模型支持概率输出）。这一过程对所有模型保持完全相同的输入格式，避免因提示差异导致的评测偏差。

3. **结果分析**  
   - **准确率计算**：直接比较模型答案与标准答案，得到每个任务的准确率。  
   - **相对提升**：用模型准确率减去对应任务的随机基线，得到“相对提升”。  
   - **跨任务对比**：把57个任务的相对提升绘成雷达图，直观看出哪些学科模型强、哪些弱。  
   - **自信度校准**：把模型的置信度分段（比如0.8以上、0.5-0.8等）与实际正确率对应，检查是否呈正相关。  
   - **专家水平差距**：把模型的整体准确率与人类专家在同套题目上的表现对比，量化距离。

最巧妙的地方在于**把随机基线当作统一的“零分线”**，这样即使不同任务的难度天差地别，也能用同一个尺度评估模型的真实进步。再加上对置信度的系统检查，直接暴露了模型“自信但错误”的现象，这在传统评测里很少被量化。

### 实验与效果
- **测试对象**：包括公开的GPT-3系列（从125M到175B参数）以及若干开源模型。  
- **整体表现**：大多数模型的平均准确率几乎等于随机基线，说明它们在广域知识上并未真正掌握。唯一例外是最大规模的GPT-3（175B），其在57个任务上的平均相对提升约为20个百分点，仍然距离专家水平有明显差距。  
- **任务分布**：在数学、计算机科学等相对结构化的任务上，GPT-3的提升稍高；而在美国历史、伦理、法律等需要深层语义理解的任务上，准确率仍接近随机。  
- **自信度**：模型在错误答案上往往给出高置信度，校准曲线呈现明显的偏离，说明模型缺乏“知道自己不知道”的能力。  
- **消融实验**：原文未提供细粒度的消融结果，只是整体报告了不同规模模型的对比，未细化到单个模块的贡献。  
- **局限性**：评测只覆盖英文教材和美国本土法律，文化多样性不足；此外，题目形式主要是选择题，可能低估了模型在生成式任务上的潜力。  

### 影响与延伸思考
这篇工作在语言模型评测史上树立了“全科测评”的标杆，随后出现的MMLU（Massive Multitask Language Understanding）系列基准、BIG-bench等都在规模和任务多样性上向它看齐。研究者开始关注模型的**知识广度**与**自我校准**，而不是单一任务的极致表现。后续工作常把MMLU作为预训练或微调的检验点，甚至用它来指导模型规模的成本效益分析。想进一步深入，可以关注**跨语言多任务评测**、**动态难度自适应测评**以及**模型不确定性校准**等方向，这些都是在MMLU提出的缺口上自然延伸的研究热点。

### 一句话记住它
大模型虽能在部分学科略胜随机，但在全科测评上仍远未达到专家水平，且常自信满满却答错——这提醒我们，真正的通用智能仍在路上。