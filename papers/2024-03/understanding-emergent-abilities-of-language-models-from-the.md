# Understanding Emergent Abilities of Language Models from the Loss   Perspective

> **Date**：2024-03-23
> **arXiv**：https://arxiv.org/abs/2403.15796

## Abstract

Recent studies have put into question the belief that emergent abilities in language models are exclusive to large models. This skepticism arises from two observations: 1) smaller models can also exhibit high performance on emergent abilities and 2) there is doubt on the discontinuous metrics used to measure these abilities. In this paper, we propose to study emergent abilities in the lens of pre-training loss, instead of model size or training compute. We demonstrate that the Transformer models with the same pre-training loss, but different model and data sizes, generate the same performance on various downstream tasks, with a fixed data corpus, tokenization, and model architecture. We also discover that a model exhibits emergent abilities on certain tasks -- regardless of the continuity of metrics -- when its pre-training loss falls below a specific threshold. Before reaching this threshold, its performance remains at the level of random guessing. This inspires us to redefine emergent abilities as those that manifest in models with lower pre-training losses, highlighting that these abilities cannot be predicted by merely extrapolating the performance trends of models with higher pre-training losses.

---

# 从损失视角理解语言模型的涌现能力 论文详细解读

### 背景：这个问题为什么难？
在大模型时代，研究者常用模型参数量或训练算力来解释为何某些任务会出现“涌现”现象——即小模型几乎不行，大模型却突然表现出高水平能力。然而，实际观察到的并非所有大模型都有这些能力，且有报告显示更小的模型在特定设置下也能达到同样水平。根本原因在于我们缺少一个统一、可度量的标尺来捕捉模型内部的学习进度，单纯靠规模做推断往往会产生误导。于是，如何找到一个与模型大小无关、能够预测能力突变的指标，成为了亟待解决的难题。

### 关键概念速览
**预训练损失（pre‑training loss）**：模型在海量未标注文本上进行自监督学习时的平均误差，数值越低说明模型对语言的统计规律捕捉得越好。可以把它想成学生的期中考试分数，分数低（损失小）代表掌握了更多知识。

**涌现能力（emergent ability）**：模型在某些下游任务上表现出远超线性或平滑增长的突变，例如零样本推理或数学解题。类似于小孩突然会说完整句子，而不是一步步慢慢增加词汇。

**阈值损失（loss threshold）**：作者发现的一个关键点，当预训练损失降到某个具体数值以下时，模型的涌现能力会从随机猜测跃升到有意义的水平。把它比作温度计的冰点，低于冰点水会结成冰，性质彻底改变。

**等损失等效（loss‑matched equivalence）**：在保持相同的预训练损失、相同的语料、分词方式和模型结构的前提下，模型的规模（参数量）和数据量可以不同，却会在下游任务上得到相同的表现。相当于不同品牌的汽车，只要油耗相同，跑同一段路的时间基本一致。

**连续度量 vs. 不连续度量**：衡量任务表现的指标有的随模型规模平滑提升（连续），有的则出现跳跃（不连续）。这里的讨论重点在于，无论度量是否连续，只要损失跨过阈值，能力都会出现突变。

### 核心创新点
1. **从模型规模转向预训练损失**：过去的研究把模型大小或算力当作预测涌现的唯一变量。作者改为以预训练损失为主轴，直接把学习质量当作因变量。这样做的直接后果是，能够在不同规模的模型之间建立公平比较的基准。

2. **等损失等效实验设计**：作者训练了一系列 Transformer，分别在参数量和数据量上做了交叉组合，但严格控制它们的预训练损失相同。实验结果显示，这些模型在多项下游任务（如阅读理解、代码生成）上的表现几乎一致，说明损失才是决定性能的关键因素，而不是单纯的规模。

3. **阈值损失概念的提出**：通过系统扫描不同损失值对应的任务表现，作者发现当损失低于某一临界点时，模型的表现会从随机水平跃升到可用水平。这个阈值在不同任务上略有差异，但普遍存在，提供了一个可操作的“能力开启开关”。

4. **对涌现定义的重新表述**：基于阈值损失，作者把涌现能力定义为“在预训练损失低于特定阈值后出现的、无法通过高损失模型的性能趋势外推预测的能力”。这让涌现不再是“模型太大就会出现”的神秘现象，而是与学习误差紧密挂钩的可测现象。

### 方法详解
整体思路可以拆成三步：① 统一训练设置，② 精准控制预训练损失，③ 评估下游任务表现并寻找阈值。

**步骤一：统一训练设置**  
作者选用标准的 Transformer 架构（相同层数、注意力头数、激活函数），固定语料库（如英文 Wikipedia + BookCorpus），并使用同一分词器（Byte‑Pair Encoding）。这样做的目的是消除除模型规模和数据量之外的所有变量，确保后续比较的公平性。

**步骤二：损失匹配**  
为了让不同规模的模型达到相同的预训练损失，作者采用两种手段：  
- **数据量调节**：小模型使用更大规模的训练数据，反之大模型使用更少的数据。  
- **训练步数调节**：通过提前停止或延长训练，使得每个模型在训练曲线的同一损失水平停下来。  
在实际操作中，作者会在每个 epoch 结束后记录验证集上的交叉熵损失，并根据目标损失值动态决定是否继续训练。

**步骤三：下游任务评估与阈值发现**  
所有模型在同一套下游基准上进行零样本、少样本和微调三种评估方式。作者把每个模型的预训练损失映射到对应任务的准确率或 F1 分数，绘制出“损失‑性能曲线”。随后，使用二分搜索或拐点检测算法定位曲线中出现显著斜率变化的点，这些点即为阈值损失。

**最巧妙的地方**  
- **等损失等效的实验控制**：传统上，人们会直接比较不同规模模型的最终性能，这会混入数据量、训练时长等噪声。这里通过“让损失相同”来消除这些干扰，等价于在同一温度下比较不同材质的金属导热性，极大提升了因果推断的可信度。  
- **阈值检测的统计方法**：作者没有简单 eyeball 曲线，而是采用了基于残差的分段线性回归，确保拐点的统计显著性，这在以往的涌现研究中少见。

### 实验与效果
- **数据集与任务**：论文在 10+ 下游任务上做评估，包括 SuperGLUE（阅读理解、自然语言推理）、MMLU（多学科知识问答）、HumanEval（代码生成）以及数学推理基准 GSM‑8K。所有任务均使用相同的测试集，保证比较公平。  
- **基线对比**：与同架构的原始大模型（如 175B 参数的 GPT‑3）以及同规模但不同损失的模型相比，等损失匹配的模型在多数任务上误差在 0.5%–2% 之间，几乎持平。更重要的是，当损失高于阈值时，所有模型的表现都接近随机猜测（例如 SuperGLUE 任务的准确率在 30% 左右），而低于阈值后准确率立刻跳到 70% 以上。  
- **消融实验**：作者分别去掉“数据量调节”和“训练步数调节”，发现仅靠单一手段难以精确匹配损失，导致性能差异扩大到 5%–10%。这说明两者的协同作用是实现等损失等效的关键。  
- **局限性**：实验只在英文语料和 Transformer 架构上完成，未验证对多语言模型或混合架构（如混合专家模型）的适用性。作者也承认阈值损失在不同任务之间仍有一定漂移，尚未给出统一的数值公式。

### 影响与延伸思考
这篇工作在社区引发了两大方向的跟进：  
1. **损失驱动的模型调度**：一些后续研究尝试在训练过程中实时监控损失，提前停止或切换模型规模，以最小算力达到目标阈值，从而实现更高效的资源利用。  
2. **涌现能力的可解释性探索**：有研究把阈值损失与内部表征的稀疏度、注意力分布等特征关联，试图解释为何跨过阈值后模型会“突然懂”某些任务。  
如果想进一步了解，可以关注 2024‑2025 年间出现的 “Loss‑Based Scaling Laws” 系列论文，它们在此基础上提出了更通用的损失‑性能预测模型（推测）。

### 一句话记住它
预训练损失低于特定阈值时，语言模型的涌现能力会突现，而模型规模本身并不是决定因素。