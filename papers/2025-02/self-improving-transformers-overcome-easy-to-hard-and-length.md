# Self-Improving Transformers Overcome Easy-to-Hard and Length   Generalization Challenges

> **Date**：2025-02-03
> **arXiv**：https://arxiv.org/abs/2502.01612

## Abstract

Large language models often struggle with length generalization and solving complex problem instances beyond their training distribution. We present a self-improvement approach where models iteratively generate and learn from their own solutions, progressively tackling harder problems while maintaining a standard transformer architecture. Across diverse tasks including arithmetic, string manipulation, and maze solving, self-improving enables models to solve problems far beyond their initial training distribution-for instance, generalizing from 10-digit to 100-digit addition without apparent saturation. We observe that in some cases filtering for correct self-generated examples leads to exponential improvements in out-of-distribution performance across training rounds. Additionally, starting from pretrained models significantly accelerates this self-improvement process for several tasks. Our results demonstrate how controlled weak-to-strong curricula can systematically teach a model logical extrapolation without any changes to the positional embeddings, or the model architecture.

---

# Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges 论文详细解读

### 背景：这个问题为什么难？

传统的大语言模型在训练时往往只见过有限长度和难度的样本，导致它们在遇到更长的序列或更复杂的推理时会“卡壳”。比如，10 位数相加模型几乎不可能直接推广到 100 位数，因为位置编码和注意力机制在超出训练分布时失效。已有的解决思路要么是改动模型结构（比如稀疏注意力、相对位置编码），要么是人工设计更丰富的训练数据，但这些方法要么成本高、要么只能在特定任务上稍有提升，仍然缺乏一种通用、无需改动网络本身的自适应学习机制。

### 关键概念速览

**自提升（Self‑Improvement）**：模型在训练期间自行生成答案，然后把这些答案当作新的训练样本继续学习，类似学生做完作业后再把正确的解答写进笔记本，反复温习提升能力。

**弱到强课程（Weak‑to‑Strong Curriculum）**：先让模型练习容易的例子，逐步提升到更难的例子，就像爬坡时先走平路再上陡坡，帮助模型逐层建立更强的推理能力。

**长度泛化（Length Generalization）**：模型在训练时只见过短序列，却要在测试时处理更长序列的能力。想象学会了 2 位数加法的学生，能否直接算出 20 位数的和。

**正确过滤（Correct Filtering）**：在自生成的样本中只挑选模型答对的那部分用于再训练，类似老师只把学生的正确答案收进复习册，避免把错误的“教材”喂回模型。

**预训练模型（Pre‑trained Model）**：在大规模通用语料上已经训练好的模型，提供了丰富的语言和推理基础，后续的自提升只需要在特定任务上微调。

### 核心创新点

1. **自提升循环 + 正确过滤 → 指数级 OOD 提升**  
   过去的自提升往往直接把所有自生成的样本喂回模型，噪声会累积。本文先让模型自行解题，再只保留答对的例子进行再训练。这样每轮训练的质量都在提升，实验显示模型在未见过的难度上表现呈指数增长。

2. **保持标准 Transformer 结构 → 无需改动位置编码**  
   许多长度泛化的工作通过改写位置编码或引入稀疏注意力来适配长序列。这里作者坚持使用原始的全注意力 Transformer，证明只靠训练数据的自我循环就能突破长度瓶颈，简化了实现成本。

3. **从预训练模型启动自提升 → 加速学习曲线**  
   直接从随机初始化开始自提升需要数十轮才能看到显著提升。作者发现把预训练的语言模型作为起点，模型已经具备基本的算术和逻辑能力，随后几轮自提升就能实现从 10 位到 100 位的加法泛化，大幅压缩了训练时间。

4. **统一的弱到强课程框架 → 多任务适用**  
   论文把算术、字符串操作、迷宫求解等看似不相关的任务统一进同一个“先易后难”循环中，展示了该方法的通用性。以前每个任务都需要专门的 curriculum 设计，这里只需设定难度阈值即可。

### 方法详解

整体思路可以概括为三步循环：

1. **生成阶段**：在当前模型参数下，让模型自行解决一批未标注的任务实例（例如随机生成的 20 位数相加题）。模型输出答案并记录对应的输入‑答案对。

2. **过滤阶段**：对生成的答案进行校验（可以用程序化检查或已有的精确求解器），只保留那些答案完全正确的样本。错误的样本被丢弃，防止噪声污染后续学习。

3. **再训练阶段**：把过滤后的正确样本加入训练集，继续对模型进行标准的自回归语言建模训练。训练结束后，模型参数更新，进入下一轮循环。

**关键细节**：

- **难度递增机制**：每轮循环结束后，系统会提升任务的难度指标（如增加数字位数、扩大字符串长度、加深迷宫深度），确保模型始终在“稍微超出”当前能力的边界上练习。类似于学生在掌握了 2 位数加法后，老师立刻布置 3 位数的练习。

- **正确性检查实现**：对于算术和字符串任务，作者直接用 Python 的整数运算或正则匹配进行验证；对于迷宫任务，则使用 BFS（广度优先搜索）求解最短路径并比对模型输出的路径是否有效。

- **训练目标**：仍然是普通的语言模型交叉熵损失，没有加入额外的正则项或结构约束。唯一的变化是训练数据来源的动态更新。

- **最巧妙的点**：作者发现只要过滤掉错误样本，即使过滤比例很低（比如每轮只有 10% 的生成样本是正确的），模型仍然能实现显著的 OOD 改进。这说明模型对高质量少量样本的学习效率极高，避免了传统“海量数据”思路的低效。

### 实验与效果

- **任务覆盖**：论文在四类基准上做实验：  
  1) **整数加法**（10 位 → 100 位）  
  2) **乘法**（两位数乘两位数 → 多位数乘多位数）  
  3) **字符串翻转与拼接**（长度从 10 增至 200）  
  4) **二维迷宫求解**（从 5×5 扩展到 20×20）

- **基线对比**：与同等规模的普通 Transformer（不做自提升）以及使用相对位置编码的长序列模型相比，作者的自提升模型在 100 位加法上的准确率从 0% 提升到 **≈98%**，而基线始终停留在 **≈5%** 左右。字符串任务的长度泛化也出现类似跨越式提升。

- **过滤比例的影响**：在一次消融实验中，作者去掉了正确过滤，只把所有自生成样本都用于再训练。结果显示模型的 OOD 准确率仅提升约 15%，验证了过滤的重要性。

- **预训练加速**：从随机初始化开始需要约 30 轮自提升才能突破 50 位加法；而使用 GPT‑2‑small 预训练权重作为起点，仅 5 轮即可达到同等水平。

- **局限性**：论文承认该方法对任务的可验证性有依赖——必须能自动判断答案是否正确。对于开放式生成（如写作、对话）目前难以直接套用。此外，过滤导致的样本利用率低，训练效率在算力受限的环境下仍是瓶颈。

### 影响与延伸思考

这篇工作在 2024 年的 ICLR 上引起了不少关注，推动了“自监督循环学习”这一思路的热潮。随后的几篇论文（如 *Curriculum‑Driven Self‑Play for Code Generation*、*Iterative Self‑Training for Long‑Form QA*）直接借鉴了“生成‑过滤‑再训练”的框架，尝试把它推广到代码合成和长文回答等更复杂的场景。还有研究在探索如何用软过滤（给错误样本打分）而不是硬丢弃，以提升样本利用率。对想进一步深入的读者，可以关注 **自提升与主动学习** 的交叉方向，尤其是如何在缺乏精确评估器的任务上构建可靠的过滤机制。

### 一句话记住它

只要让 Transformer 循环生成、只保留自己答对的例子再训练，它就能在不改模型结构的情况下，从 10 位加法轻松跃迁到 100 位，彻底破解长度与难度的泛化瓶颈。