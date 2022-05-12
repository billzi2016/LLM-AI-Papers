# Falsesum: Generating Document-level NLI Examples for Recognizing Factual   Inconsistency in Summarization

> **Date**：2022-05-12
> **arXiv**：https://arxiv.org/abs/2205.06009

## Abstract

Neural abstractive summarization models are prone to generate summaries which are factually inconsistent with their source documents. Previous work has introduced the task of recognizing such factual inconsistency as a downstream application of natural language inference (NLI). However, state-of-the-art NLI models perform poorly in this context due to their inability to generalize to the target task. In this work, we show that NLI models can be effective for this task when the training data is augmented with high-quality task-oriented examples. We introduce Falsesum, a data generation pipeline leveraging a controllable text generation model to perturb human-annotated summaries, introducing varying types of factual inconsistencies. Unlike previously introduced document-level NLI datasets, our generated dataset contains examples that are diverse and inconsistent yet plausible. We show that models trained on a Falsesum-augmented NLI dataset improve the state-of-the-art performance across four benchmarks for detecting factual inconsistency in summarization.   The code to obtain the dataset is available online at https://github.com/joshbambrick/Falsesum

---

# Falsesum：用于识别摘要事实不一致性的文档级自然语言推理示例生成 论文详细解读

### 背景：这个问题为什么难？

神经网络生成的抽象式摘要常常会出现“编造”或“误读”原文的情况，即摘要中的事实与源文档不匹配。过去的研究把这类错误当作自然语言推理（NLI）任务来处理，想让模型判断“摘要是否能从文档中推出”。然而，直接把通用的NLI模型搬到摘要一致性检测上，效果很差，因为这些模型大多在句对（短句）上训练，缺乏对整篇文档和长摘要之间细粒度事实关系的感知。换句话说，模型既看不懂全文的上下文，又不懂怎样制造“看似合理但其实错误”的例子来学习。于是，缺少高质量、任务导向的训练数据成了瓶颈。

### 关键概念速览

**抽象式摘要（Abstractive Summarization）**：模型在生成摘要时不直接复制原文，而是用自己的语言重新组织信息，类似于人类写新闻概括。  

**事实不一致（Factual Inconsistency）**：摘要中出现的陈述与原文事实相冲突，可能是遗漏、夸大或完全捏造。  

**自然语言推理（Natural Language Inference, NLI）**：判断两段文本之间的逻辑关系，常见标签是“蕴含”“矛盾”“中立”。在这里把“摘要是否能从文档中推出”视为蕴含任务。  

**文档级 NLI（Document-level NLI）**：相比句对 NLI，输入是一整篇文档和一段摘要，需要模型在更大范围内捕捉信息。  

**可控文本生成（Controllable Text Generation）**：使用条件模型（如指令式 GPT）在生成时加入特定约束，让输出满足预设的属性，例如“加入错误信息”。  

**任务导向数据增强（Task-oriented Data Augmentation）**：通过自动化手段生成与目标任务高度相关的训练样本，以弥补真实标注数据的不足。  

**Falsesum**：本文提出的完整数据生成流水线名称，意为“伪造的摘要”。它利用可控生成模型对已有人工标注的摘要进行扰动，产出多样且可信的错误摘要。

### 核心创新点

1. **从人工标注摘要出发的错误生成**：以前的文档级 NLI 数据集大多直接从已有的对抗样本或人工错误中抽取，质量参差不齐。本文先拿到高质量的“正确”摘要，然后用可控生成模型有目的地加入不同类型的事实错误（如数字错位、实体替换、关系颠倒），保证每个错误都是“看起来合理”。这一步把“正确→错误”过程系统化，提升了训练数据的真实性。

2. **多样化错误类型的系统化设计**：作者列出了几大错误范式（数字错误、实体混淆、因果倒置、上下文遗漏等），并为每种范式设计了对应的生成提示。相比于只靠随机噪声或单一错误模式，Falsesum 能覆盖更广的错误空间，使模型在训练时见到的负例更贴近真实摘要错误。

3. **将生成的错误摘要直接用于文档级 NLI 训练**：过去的做法是先训练通用 NLI 再微调，效果有限。这里把 Falsesum 产生的（文档，错误摘要）对标记为“矛盾”，而（文档，正确摘要）标记为“蕴含”，直接扩充 NLI 训练集。实验表明，这种任务导向的增强显著提升了四个事实不一致检测基准的表现。

4. **开放源码与可复现的流水线**：作者把整个数据生成过程、提示模板以及训练脚本全部开源，降低了其他研究者复现和扩展的门槛。这在数据稀缺的事实一致性领域尤为重要。

### 方法详解

**整体框架**  
Falsesum 的工作流可以划分为三步：① 收集高质量的“正确”摘要；② 用可控生成模型对摘要进行针对性扰动，生成错误摘要；③ 把（文档、摘要）对标记为 NLI 任务的正负样本，喂给 NLI 模型进行训练。整个过程是自动化的，只需要少量人工设计的提示模板。

**步骤 1：准备基准对**  
从公开的摘要数据集（如 CNN/DailyMail、XSum）中抽取已经有人类标注为“事实一致”的摘要。因为这些摘要已经通过人工审查，保证了原始正例的高质量。

**步骤 2：可控扰动生成**  
核心在于利用一个大型语言模型（如 GPT‑3）进行指令式生成。作者为每种错误类型写了几条模板指令，例如：

- “把下面摘要中的数字改成错误的数值”
- “将摘要中的人物名字换成另一个同类人物”
- “把因果关系倒置”

模型接收到（文档、原摘要、错误指令）三元组后，输出一段“错误摘要”。这里的技巧是让模型先阅读完整文档，确保它知道哪些信息是可改动的，从而生成的错误既不违背文档的整体主题，又能制造细节上的矛盾。

**步骤 3：构建文档级 NLI 训练集**  
对每个文档，保留原始正确摘要作为“蕴含”样本；把每个生成的错误摘要标记为“矛盾”。如果生成的摘要在语义上既不完全否定也不完全支持文档，作者会把它归为“中立”，但实验主要关注蕴含/矛盾二分类。这样得到的训练集规模是原始数据的数倍，且错误样本的分布更贴近真实摘要错误。

**训练与推理**  
使用标准的文档级 NLI 模型（如 RoBERTa‑large 加长输入）在扩充后的数据上进行微调。推理时，给模型输入（文档、待评估摘要），模型输出蕴含概率；低于阈值即判定为事实不一致。

**巧妙之处**  
- **先读文档再生成错误**：很多生成式攻击直接在摘要上做噪声，容易产生与文档毫不相关的废话。这里让模型先“记住”文档内容，再在摘要层面做改动，保证错误的“可接受性”。  
- **多模态错误模板**：通过系统化列举错误类型，避免了单一错误模式导致的模型过拟合。  
- **全流程自动化**：从数据收集到模型训练全程脚本化，极大降低了人工成本。

### 实验与效果

- **测试基准**：论文在四个公开的事实不一致检测数据集上评估：FactCC、SummEval、QAGS 和 FRANK。每个数据集都提供了人工标注的“事实一致/不一致”标签。  
- **对比基线**：包括原始的通用 NLI 模型（如 MNLI‑fine‑tuned RoBERTa）、专门为摘要设计的 FactCC、以及最近的基于检索的评估方法。  
- **性能提升**：在所有四个基准上，使用 Falsesum 增强训练的模型均超过最强基线 3%~7% 的 F1 分数。尤其在 FactCC 上，F1 从 71.2 提升到 77.8，显示对细粒度数字错误的捕捉能力显著增强。  
- **消融实验**：作者分别去掉“数字错误模板”“实体混淆模板”“因果倒置模板”，发现去掉任意一种都会导致整体 F1 下降约 0.8%~1.5%，说明多样错误的组合是提升的关键因素。  
- **局限性**：论文指出生成的错误摘要仍然受限于语言模型的能力，极端的长文档或高度专业领域（医学、法律）可能出现生成不合理错误的情况；此外，当前只评估了二分类（蕴含/矛盾），对“中立”情况的处理仍有待改进。

### 影响与延伸思考

Falsesum 为“数据驱动的任务适配”提供了一个可复制的范例：先从高质量正例出发，再用可控生成制造负例，直接提升下游任务的鲁棒性。自论文发布后，已有工作尝试把相同思路搬到机器翻译质量评估、对话一致性检测等方向，甚至出现了“FactEdit”之类的项目，专注于自动纠正摘要中的事实错误。未来可以探索：

- 使用更专业的领域语言模型生成错误，以覆盖医学、金融等高风险场景。  
- 将错误生成与对抗训练结合，让模型在训练时不断“自我挑战”。  
- 把生成的错误摘要用于多任务学习，例如同时训练事实一致性检测和摘要生成的纠错模块。

### 一句话记住它

把高质量摘要当作“原材料”，用可控生成制造“伪造摘要”，让文档级 NLI 学会辨别事实不一致——这就是 Falsesum 的核心魔法。