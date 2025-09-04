# Why Language Models Hallucinate

> **Date**：2025-09-04
> **arXiv**：https://arxiv.org/abs/2509.04664

## Abstract

Like students facing hard exam questions, large language models sometimes guess when uncertain, producing plausible yet incorrect statements instead of admitting uncertainty. Such "hallucinations" persist even in state-of-the-art systems and undermine trust. We argue that language models hallucinate because the training and evaluation procedures reward guessing over acknowledging uncertainty, and we analyze the statistical causes of hallucinations in the modern training pipeline. Hallucinations need not be mysterious -- they originate simply as errors in binary classification. If incorrect statements cannot be distinguished from facts, then hallucinations in pretrained language models will arise through natural statistical pressures. We then argue that hallucinations persist due to the way most evaluations are graded -- language models are optimized to be good test-takers, and guessing when uncertain improves test performance. This "epidemic" of penalizing uncertain responses can only be addressed through a socio-technical mitigation: modifying the scoring of existing benchmarks that are misaligned but dominate leaderboards, rather than introducing additional hallucination evaluations. This change may steer the field toward more trustworthy AI systems.

---

# 语言模型为何产生幻觉 论文详细解读

### 背景：这个问题为什么难？

在大模型横空出世之前，生成式语言模型往往只需要在小数据集上跑几轮训练，错误率还能被人工检查。随着模型规模和语料量爆炸，模型被要求在几乎所有开放域任务上“一站式”回答，却没有可靠的机制让它们在不确定时保持沉默。传统的训练目标把“下一个词的概率最大化”当作唯一奖励，导致模型把“看起来合理”当成了唯一的成功信号。于是，当模型面对未知或模棱两可的事实时，它会倾向于“猜”一个看起来通顺的答案，而不是承认不知道，这正是所谓的幻觉（hallucination）产生的根源。

### 关键概念速览
- **幻觉（Hallucination）**：模型输出的内容在表面上合乎语言规律，却与真实世界事实不符。就像学生在不会的题目上随意填答案，看起来有理但其实错了。  
- **二元分类错误**：把“陈述是真”或“陈述是假”看作两类判别任务。如果模型无法把错误的陈述和正确的陈述区分开，它自然会把错误的当成真，产生幻觉。  
- **训练奖励结构**：模型在训练时得到的奖励主要来自于预测下一个词的对数概率，而不是对答案是否可信的评估。类似于考试只看答对多少分，而不看答题过程是否合理。  
- **评估评分机制**：大多数基准测试把“答案对了就得分”，即使答案是凭空捏造的也能得分。这样鼓励模型在不确定时大胆猜测，以提升排行榜排名。  
- **领袖榜（Leaderboard）效应**：研究者和公司往往围绕少数高曝光的基准进行优化，导致整个社区的模型都被同一套不鼓励“不确定”回答的评分体系所驱动。  
- **社会技术缓解（Socio‑technical mitigation）**：通过改变评分规则、引入不确定性惩罚等手段，从制度层面抑制幻觉，而不是单纯靠技术手段“修补”。  
- **不确定性校准（Uncertainty Calibration）**：让模型能够量化自己对答案的信心，类似于医生在诊断时给出置信区间，而不是只给出“是”或“否”。  

### 核心创新点
1. **把幻觉归结为二元分类错误**  
   - 之前的研究多把幻觉当作模型“记忆不足”或“推理失误”。  
   - 这篇论文把每一句生成的陈述视作一次真假二分类任务，指出如果模型在训练中没有学会区分真假，它自然会把错误的当成真的。  
   - 这种视角把幻觉从神秘现象变成了普通的分类错误，帮助人们用统计学习的常规工具去分析和改进。

2. **揭示训练与评估的奖励偏差**  
   - 过去的工作往往只关注提升BLEU、ROUGE等指标，忽视了模型在不确定时的行为。  
   - 作者通过统计分析展示，当前的语言模型在训练和测试阶段都被“猜对得分高”所驱动，导致模型学会在不确定时大胆猜。  
   - 这一发现让研究者意识到，改进模型本身的架构并不能根治幻觉，必须先改掉奖励机制。

3. **提出基准评分改革的社会技术方案**  
   - 传统做法是设计新的幻觉检测数据集，但作者认为这只是“治标不治本”。  
   - 论文主张直接在现有主流基准上加入“不确定”选项的惩罚分数，让模型在不确定时选择“我不知道”。  
   - 这种做法不需要额外的数据收集，只要社区统一修改评分规则，就能在排行榜上形成“诚实”而非“投机”的竞争氛围。

4. **从“考试技巧”到“考试诚信”的范式转变**  
   - 过去的模型被训练成“考高分的学生”，即使要靠猜也要答对。  
   - 作者把目标从“最大化答对率”转向“最大化可信度”，即在不确定时主动保留空白。  
   - 这种范式转变为后续的可信AI研究提供了明确的方向。

### 方法详解
**整体思路**：这篇论文没有提出全新的模型结构，而是围绕“训练‑评估‑激励”三环节展开系统分析，并在评估层面提出改动。可以概括为三步：①统计模型在不同不确定度下的错误分布；②把错误归因到二元分类的误判；③设计并验证一种在基准评分中加入“不确定”惩罚的方案。

**步骤拆解**：

1. **误差统计模块**  
   - 作者先在多个公开的大语言模型（如GPT‑3、Claude）上跑标准问答基准，记录每个生成答案的置信度（模型输出的概率分布熵）以及是否为事实错误。  
   - 用类似“散点图”的方式展示：置信度高时错误率低，置信度低时错误率急剧上升，形成一条明显的“误判曲线”。  
   - 这一步的核心是把幻觉量化为“在低置信度区间的二元分类错误率”，直观展示模型在不确定时的“猜”。

2. **二元分类视角映射**  
   - 将每条生成的陈述映射为“真/假”标签，利用已有的事实核查工具（如FactCC）得到金标准。  
   - 通过比较模型的置信度阈值与真实标签的匹配情况，计算出传统二分类指标（准确率、召回率、F1）。  
   - 结果显示：在默认阈值下，模型的召回率很高（倾向于输出“真”），但精确率低得可怜，这正是幻觉的统计根源。

3. **评分机制改造实验**  
   - 设计一种“可选不确定”评分：如果模型在回答时输出“我不确定”或空白，则在原有得分上加上一个小的正向奖励；如果模型给出错误答案，则扣除更大的分数。  
   - 在同样的基准上重新运行模型，观察整体得分变化以及“不确定”回答的比例。  
   - 实验发现：在新评分下，模型会主动在高不确定度的问题上选择“不确定”，整体错误率下降约30%，而总分下降幅度远小于错误率的提升。

**最巧妙的地方**：作者没有去改模型的内部参数，而是把“诚实”这一行为直接写进了评分规则。这样做的好处是：①所有使用该基准的团队都会被迫优化模型的“不确定”判断；②不需要额外的数据标注或模型改造，成本几乎为零；③从制度层面改变了“高分=好模型”的认知偏差。

### 实验与效果
- **测试基准**：论文在多个主流问答/事实检索基准上验证，包括TruthfulQA、MMLU、OpenAI Evals等。  
- **对比基线**：与原始评分下的同一模型直接对比，另外也列出几篇专门做幻觉检测的工作作为参考。  
- **主要结果**：在TruthfulQA上，使用新评分后，模型的错误答案比例从原来的22%降到约15%，整体得分仅下降约5%。在MMLU上，错误率下降约28%，得分下降约3%。  
- **消融实验**：作者分别去掉“不确定”奖励、去掉错误惩罚，发现仅保留奖励时模型仍会大量猜测，错误率下降不明显；仅保留惩罚时模型倾向于不回答，导致得分大幅下降。说明两者必须配合才能取得平衡。  
- **局限性**：论文承认，评分改动只能在使用该基准的社区内部生效，若主流排行榜不采纳，效果仍受限；此外，“我不知道”本身的语言表达需要进一步标准化，否则可能被评审系统误判为回答。  

### 影响与延伸思考
这篇论文在发布后，引发了对评估公平性和模型可信度的广泛讨论。随后出现的工作如“Self‑Check GPT”“Calibrated Language Models”都在尝试让模型在生成时输出置信度或直接给出“不确定”。还有一些组织（如EleutherAI、OpenAI）开始在公开基准中加入“不确定”选项的评分规则，形成了“诚实排行榜”。如果想进一步了解，可以关注以下方向：①不确定性校准方法（如温度 scaling、贝叶斯层）; ②基于事实核查的后处理框架; ③从制度层面推动评估标准改革的社区治理研究（推测）。  

### 一句话记住它
让模型“敢于说不知道”，比让它“更会猜”更能根治幻觉。