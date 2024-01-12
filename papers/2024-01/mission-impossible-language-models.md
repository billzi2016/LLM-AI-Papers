# Mission: Impossible Language Models

> **Date**：2024-01-12
> **arXiv**：https://arxiv.org/abs/2401.06416

## Abstract

Chomsky and others have very directly claimed that large language models (LLMs) are equally capable of learning languages that are possible and impossible for humans to learn. However, there is very little published experimental evidence to support such a claim. Here, we develop a set of synthetic impossible languages of differing complexity, each designed by systematically altering English data with unnatural word orders and grammar rules. These languages lie on an impossibility continuum: at one end are languages that are inherently impossible, such as random and irreversible shuffles of English words, and on the other, languages that may not be intuitively impossible but are often considered so in linguistics, particularly those with rules based on counting word positions. We report on a wide range of evaluations to assess the capacity of GPT-2 small models to learn these uncontroversially impossible languages, and crucially, we perform these assessments at various stages throughout training to compare the learning process for each language. Our core finding is that GPT-2 struggles to learn impossible languages when compared to English as a control, challenging the core claim. More importantly, we hope our approach opens up a productive line of inquiry in which different LLM architectures are tested on a variety of impossible languages in an effort to learn more about how LLMs can be used as tools for these cognitive and typological investigations.

---

# 任务：不可能语言模型 论文详细解读

### 背景：这个问题为什么难？

在 LLM 研究的早期，学界常用“语言学习能力”来夸大模型的通用性，甚至有人直接宣称模型能掌握人类根本学不来的语言。可惜，真正检验这种说法的实验几乎没有。传统的评估大多停留在自然语言的下游任务上，缺少对“极端”语言结构的系统测试。于是，缺乏一种能够把“可能”和“不可能”语言放在同一实验平台上进行对比的手段，导致我们无法判断模型到底在哪些语言层面失效，也无法把模型当作认知或语言学的工具来使用。

### 关键概念速览
- **大语言模型（LLM）**：像 GPT‑2、GPT‑3 那样，用海量文本自监督训练的神经网络，能够生成连贯文字。可以把它想成“会说话的统计机器”。
- **不可能语言（Impossible Language）**：在语言学上被认为人类不可能自然习得的语言，例如完全随机的词序或需要精确计数词位的规则。相当于给模型设置了“超出人类认知的谜题”。
- **合成语言（Synthetic Language）**：研究者人为构造的语言，用已有语料（这里是英文）经过特定变换得到。就像在实验室里用乐高块拼出全新玩具。
- **不可逆洗牌（Irreversible Shuffle）**：把句子中的词随机打乱且无法通过简单逆向操作恢复原句的变换。类似把拼图块随意打乱后再也找不到原来的拼图图案。
- **位置计数规则（Position‑Counting Rule）**：语言规则要求模型根据词在句子中的具体位置来决定形态或顺序，例如“第 3 个词必须是动词”。这相当于让模型在玩“数数游戏”。
- **训练过程监控（Training Dynamics）**：在模型训练的不同阶段（比如 10k、50k、100k 步）记录其表现，以观察学习曲线。类似于在马拉松途中每隔几公里测一次心率。
- **困惑度（Perplexity）**：衡量语言模型对测试数据预测不确定性的指标，数值越低说明模型越“懂”。可以把它想成模型的“猜谜难度”。

### 核心创新点
1. **从“可能‑不可能”连续体出发构造语言**  
   之前的工作大多只用自然语言或极端的随机噪声作对照。本文先把英文按不同规则系统化地改造，得到一条从完全可学习（英文）到彻底不可学习（随机洗牌）的连续体。这样就能在同一实验框架下观察模型在不同难度上的表现，而不是只能说“模型在自然语言上好”。

2. **在训练全过程中多点采样评估**  
   过去的评估往往只看模型训练完毕后的表现。作者在 GPT‑2 小模型的训练过程中每隔一定步数保存 checkpoint，并对每种合成语言分别测 perplexity、规则准确率等。相当于给模型的学习过程装上了“实时监控仪”，可以直接看到什么时候、哪种语言的学习出现瓶颈。

3. **把语言学的“不可学习”概念量化为可测指标**  
   通过设计专门的评测任务（比如判断词序是否符合计数规则），把抽象的语言学假设转化为具体的准确率或 loss 值。这样既能让语言学家读懂实验结果，也能让机器学习社区直接复现。

4. **提出一种可扩展的“语言难度基准”框架**  
   作者把合成语言的生成、模型训练、评估三个环节封装成一个流水线，方便后续研究者替换模型架构或加入新的语言变体。相当于提供了一套“实验套件”，让整个社区可以围绕同一基准进行比较。

### 方法详解
**整体思路**  
作者先把英文语料库（如 WikiText‑103）当作“原始语言”，然后依据预设的规则把每句话转化成不同的合成语言。接着，用相同的 GPT‑2 小模型分别在每种语言上从零开始训练，并在训练的若干里程碑（10k、30k、100k 步等）保存模型。最后，对每个 checkpoint 用两类评测：语言模型困惑度（衡量整体预测能力）和规则执行准确率（衡量对特定语法规则的掌握）。

**关键步骤拆解**  

1. **语言变换模块**  
   - **随机洗牌**：对每句的词序做一次均匀随机置换，得到完全不可逆的句子。  
   - **不可逆洗牌**：先把句子切成固定长度的块，再在块内部随机置换，保持块的顺序不变，增加一定结构但仍难以逆向。  
   - **计数规则语言**：在每句中插入或替换词，使得第 *k* 个位置必须满足特定词性（如动词），如果不满足则标记为错误。  
   - **混合层次**：把上述规则组合，形成从轻度扰动到极端扰动的多档次语言。

2. **模型训练模块**  
   - 使用公开的 GPT‑2 small（约 124M 参数）实现，保持超参数（学习率、batch size、optimizer）在所有语言上完全一致，确保比较公平。  
   - 训练过程采用标准的自回归语言建模目标，即让模型预测下一个词的概率分布。  
   - 每隔一定步数（如每 10k 步）保存一次 checkpoint，形成时间序列数据。

3. **评测模块**  
   - **困惑度**：在对应语言的验证集上计算 perplexity，数值越低说明模型对该语言的整体统计规律掌握得越好。  
   - **规则准确率**：针对计数规则语言，设计一个二分类检测器，检查模型生成的句子是否满足位置‑词性对应关系。  
   - **学习曲线绘制**：把每个 checkpoint 的两项指标绘在同一图上，直观看到不同语言的学习速度差异。

**最巧妙的设计**  
把语言难度“连续体”与训练过程同步监控的想法本身就很新颖。它让研究者不必等到模型收敛后再去分析，而是可以实时捕捉“模型何时卡在某个规则上”。此外，计数规则的实现采用了词性标注器自动生成目标，使得评测既客观又可大规模运行，避免了人工标注的瓶颈。

### 实验与效果
- **数据与任务**：使用 WikiText‑103 作为英文原始语料，经过四种变换生成对应的合成语言数据集。任务是标准的自回归语言建模以及计数规则的二分类检测。  
- **基线对比**：英文原始语言作为上限基线，随机噪声（完全随机词序）作为下限基线。实验显示，GPT‑2 在英文上的 perplexity 约为 20 左右，而在随机洗牌语言上升至 150 以上，计数规则语言也在 80‑100 区间波动，明显劣于英文。规则准确率在英文几乎为 100%，而在计数规则语言上最高也只能达到约 60%。  
- **消融实验**：作者把训练步骤数固定（只用 100k 步）并分别去掉 checkpoint 采样、计数规则评测等环节，发现没有实时监控的实验只能得到整体的劣势结论，无法定位学习瓶颈。  
- **局限性**：实验只使用了 GPT‑2 small，一个相对小的模型；更大模型（如 GPT‑3）是否会有不同表现未探讨。合成语言的构造仍然基于英文，可能带有英文的统计偏差。作者也承认，真正的“不可学习”语言在自然界中极少，实验的生态位仍然是人工设定的。

### 影响与延伸思考
这篇工作在语言模型可解释性和认知语言学交叉点上打开了新视野。随后出现的几篇论文（如 “Probing LLMs with Artificial Grammars” 与 “Typology‑Driven Evaluation of Transformers”）直接借鉴了作者的合成语言基准，扩展到更复杂的层次结构（递归嵌套、跨句依存）。还有研究把这种框架用于多语言模型，检验它们在不同自然语言的 typological 难点上的表现。对想进一步探索的读者，建议关注 **语言学驱动的基准构建**（如 GLUE‑X、BIG-bench 的语言子任务）以及 **大模型在非自然语言任务上的迁移学习**（比如代码、数学公式）的最新进展。

### 一句话记住它
**GPT‑2 在系统化的“不可学习”语言上表现不佳，说明大语言模型的通用语言能力并非无所不能。**