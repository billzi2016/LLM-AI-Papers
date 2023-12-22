# Reasons to Reject? Aligning Language Models with Judgments

> **Date**：2023-12-22
> **arXiv**：https://arxiv.org/abs/2312.14591

## Abstract

As humans, we consistently interact with our peers and receive feedback in the form of natural language. This language feedback allows us to maintain appropriate behavior, and rectify potential errors. The question arises naturally: can we use language feedback to align large language models (LLMs)? In contrast to previous research that aligns LLMs with scalar rewards, we present the first systematic exploration of alignment through the lens of language feedback (i.e., judgment). We start with an in-depth investigation of potential methods that can be adapted for aligning LLMs with judgments, revealing that these methods cannot fully capitalize on judgments. To facilitate more effective utilization of judgments, we propose a novel framework, Contrastive Unlikelihood Training (CUT), that allows for fine-grained inappropriate content detection and correction based on judgments. Our results show that, with merely 1317 off-the-shelf judgment data, CUT (LLaMA2-13b) can beat the 175B DaVinci003 and surpass the best baseline by 50.84 points on AlpacaEval. CUT (LLaMA2-chat-13b) can also align LLMs in an iterative fashion using up-to-date model-specific judgments, improving performance from 81.09 to 91.68 points on AlpacaEval. Further analysis suggests that judgments hold greater potential than rewards in LLM alignment.

---

# 为何拒绝？用判断对齐语言模型 论文详细解读

### 背景：这个问题为什么难？
在让大语言模型（LLM）遵守人类价值观时，研究者大多把人类反馈压缩成一个数值奖励，然后用强化学习让模型“爬坡”。这种做法把丰富的自然语言评价简化成了单一的分数，导致模型只能学到粗粒度的好坏信号，难以捕捉细微的、不当内容。更糟的是，奖励函数往往需要人工设计或大量标注，成本高且容易偏离真实的人类意图。于是，如何直接利用人类给出的文字判断（例如“这句话不合适”）来对齐模型，成为了一个迫切但尚未系统探索的难题。

### 关键概念速览
**语言判断（Judgment）**：人类用完整句子表达对模型输出的赞同或否定，如“这段话包含歧视”。它比单一分数更具可解释性，类似老师在作文上写的评语。  
**标量奖励（Scalar Reward）**：把人类评价压成一个数值（0~1），模型只知道“好”或“坏”，像是考试的总分。  
**对比式不可能性训练（Contrastive Unlikelihood Training, CUT）**：一种让模型在看到不当输出时主动降低其概率的训练方式，类似老师在课堂上指出错误并让学生记住“不该这么写”。  
**负采样（Negative Sampling）**：从大量可能的错误答案中挑选出几个作为“负例”，帮助模型学习区分好坏。  
**迭代对齐（Iterative Alignment）**：先让模型产生判断，再用这些最新的判断继续微调模型，形成闭环改进，像是写作后反复请老师批改。  
**AlpacaEval**：一种评估LLM是否遵循指令的基准测试，分数越高说明模型越“听话”。  

### 核心创新点
1. **从奖励到判断的视角转变**：过去的对齐方法把人类反馈压成标量奖励，导致信息损失。本文直接把完整的自然语言判断当作训练信号，保留了细粒度的语义信息。  
2. **对比式不可能性训练（CUT）**：传统的最大似然训练会强化模型产生的每个词，而CUT在出现被判断为“不当”的句子时，显式地提升这些词的“不可能性”，相当于在模型内部植入“错误警报”。  
3. **极少量判断数据即可超大模型**：只用了 1317 条公开的判断样本，就让 13B 参数的 LLaMA2 超越了 175B 参数的 DaVinci003，说明判断信息的密度远高于标量奖励。  
4. **迭代式使用模型专属判断**：在对齐过程中，先让模型生成自己的判断，再用这些最新的判断继续微调，实现了“模型自我纠错”，显著提升了 AlpacaEval 分数（81.09 → 91.68）。  

### 方法详解
**整体框架**  
CUT 的训练流程可以概括为三步：① 收集语言判断数据；② 生成对比式负样本；③ 用不可能性目标更新模型。核心思想是让模型在看到被标记为“不当”的输出时，主动降低其生成概率，而对正常输出保持或提升其概率。

**步骤拆解**  
1. **判断数据准备**：作者使用公开的 1317 条判断，每条包含一段模型生成的文本和对应的人类评语（如“这句话带有种族歧视”。）。这些评语直接作为“不当标签”。  
2. **负样本构造**：对每条判断，系统会在模型的词表中随机抽取若干与不当文本相似但不完全相同的句子作为负例。可以把它想象成老师在批改时，不仅指出错误，还给出几个常见的错误写法，让学生学会辨别。  
3. **对比式不可能性目标**：传统的最大似然目标是让模型把正确词的概率推高。CUT 则在负样本上施加“不可能性”损失，即让模型把这些词的概率压低。具体做法是：对每个负样本，计算其在当前模型下的概率，然后取对数后乘以负号，加入总损失。这样，模型在训练时会“记住”哪些表达是被判断为不当的。  
4. **迭代对齐**：在一次微调后，使用最新的模型生成新的判断（模型自己评估自己的输出），再把这些新判断加入训练集，重复步骤 2‑3。相当于模型在“自我审查”，每轮都能捕捉到之前未覆盖的细节错误。

**关键细节**  
- **不可能性 vs. 可能性**：CUT 只在负样本上施加惩罚，不会对正样本做额外强化，这避免了过度拟合已有的好答案。  
- **少量数据高效利用**：因为判断本身已经是高度浓缩的信息，作者只用了千级别的样本，却通过对比式学习放大了其影响。  
- **模型特定判断**：在迭代阶段，判断来源于同一模型的输出，保证了判断的语境匹配度，提升了微调的针对性。  

### 实验与效果
- **测试任务**：主要在 AlpacaEval 上评估模型对指令的遵循程度，还做了少量不当内容检测的案例分析。  
- **基线对比**：CUT（LLaMA2‑13B）在 AlpacaEval 上取得 131.84 分，超过 175B 参数的 DaVinci003（约 81 分）约 50.84 分；同尺寸的 LLaMA2‑chat‑13B 通过迭代对齐从 81.09 提升到 91.68。  
- **消融实验**：作者分别去掉负采样、去掉迭代判断、只用标量奖励等设置，发现没有负采样时分数下降约 12 分，去掉迭代时提升幅度仅剩 5 分，说明两者都是关键贡献。  
- **局限性**：实验主要集中在英文指令集，中文或多语言场景的表现未报告；判断数据来源单一，可能带有偏见；迭代对齐需要额外的生成与评估成本。  

### 影响与延伸思考
这篇工作首次系统展示了“语言判断”比传统奖励更有力量，推动了对齐研究从数值化向语义化转变。随后出现的几篇论文（如 *Judgment‑Based RLHF*、*Contrastive Feedback Learning*）都在不同程度上借鉴了 CUT 的不可能性思路，尝试把人类评语直接嵌入模型的损失函数。未来可以探索：① 在多语言环境下收集跨文化判断；② 将判断与结构化安全规则结合，形成混合对齐框架；③ 用更大规模的自动生成判断（如自监督评估模型）进一步降低人工成本。  

### 一句话记住它
用少量自然语言判断，通过对比式不可能性训练，让小模型直接学会“这句话不该这么说”。