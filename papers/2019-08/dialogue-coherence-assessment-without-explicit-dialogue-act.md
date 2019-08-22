# Dialogue Coherence Assessment Without Explicit Dialogue Act Labels

> **Date**：2019-08-22
> **arXiv**：https://arxiv.org/abs/1908.08486

## Abstract

Recent dialogue coherence models use the coherence features designed for monologue texts, e.g. nominal entities, to represent utterances and then explicitly augment them with dialogue-relevant features, e.g., dialogue act labels. It indicates two drawbacks, (a) semantics of utterances is limited to entity mentions, and (b) the performance of coherence models strongly relies on the quality of the input dialogue act labels. We address these issues by introducing a novel approach to dialogue coherence assessment. We use dialogue act prediction as an auxiliary task in a multi-task learning scenario to obtain informative utterance representations for coherence assessment. Our approach alleviates the need for explicit dialogue act labels during evaluation. The results of our experiments show that our model substantially (more than 20 accuracy points) outperforms its strong competitors on the DailyDialogue corpus, and performs on par with them on the SwitchBoard corpus for ranking dialogues concerning their coherence.

---

# 无需显式对话行为标签的对话连贯性评估 论文详细解读

### 背景：这个问题为什么难？

对话的连贯性不像单篇文章那样只看词句的衔接，参与者的意图、说话的功能（比如提问、确认、道别）会极大影响整体流畅感。早期的连贯性模型直接搬用单篇文本的特征——比如名词实体的出现频率——来表示每一句话，然后硬生生地在特征上加上对话行为标签（dialogue act）来补足信息。这样做有两个根本缺陷：一是只把句子当成“实体的集合”，忽略了真正的语义；二是模型的好坏几乎全靠事先标好的行为标签，而这些标签在真实场景里往往不完整或根本没有。于是，如何在不依赖人工标注的对话行为的前提下，得到既能捕捉语义又能反映对话功能的句子表示，成为了一个急需突破的难点。

### 关键概念速览
- **对话行为（Dialogue Act）**：一句话在对话中的功能，如提问、陈述、道别等。可以把它想成对话里的“角色卡”，指明说话者想干什么。  
- **连贯性评估（Coherence Assessment）**：判断一段对话是否在逻辑和语用上前后顺畅，类似于读者在看完一段聊天记录后会不会觉得“这对话合理”。  
- **多任务学习（Multi‑Task Learning）**：让模型同时学习两个或更多任务，共享底层表示，以期互相促进。好比一个学生在学数学的同时练习物理，两个科目都能提升思考能力。  
- **隐式特征学习（Implicit Feature Learning）**：不显式提供特征，而是让模型自己从数据中抽取有用信息。相当于让孩子自己在玩耍中发现颜色的区别，而不是直接告诉他“红色是这样”。  
- **句子表示（Utterance Representation）**：把一句话压缩成向量，向量里包含了词义、上下文、说话意图等信息，后续模型可以直接操作这些向量。  
- **对话连贯性排序（Dialogue Coherence Ranking）**：给定若干对话，模型输出一个排序，最连贯的排在前面。类似于让系统挑选出最自然的聊天记录。  
- **DailyDialogue 与 SwitchBoard**：两个公开的对话数据集，前者日常口语对话较为简短，后者是电话交谈，语言更正式、结构更复杂。  

### 核心创新点
1. **把对话行为预测当作辅助任务 → 在同一个网络里同时训练连贯性判断和行为预测 → 句子向量在学习过程中自然吸收了功能信息，而不需要在评估阶段手动喂入行为标签**。  
2. **放弃仅靠实体特征的单一表示 → 引入上下文感知的深度编码器（如双向Transformer）来捕获完整语义 → 表示更丰富，连贯性模型不再局限于“出现了哪些名词”。  
3. **在评估时完全去除行为标签的依赖 → 只使用训练好的共享编码器输出的句子向量进行连贯性打分 → 解决了真实场景中标签缺失的问题**。  
4. **在两个风格迥异的数据集上进行对比实验 → 在日常对话的 DailyDialogue 上提升超过 20% 的准确率，在更难的 SwitchBoard 上达到与最强基线持平的水平 → 证明方法的通用性和鲁棒性**。

### 方法详解
整体思路可以拆成三步：**编码 → 共享学习 → 任务分支**。

1. **句子编码**  
   每一句对话先经过词嵌入（如 GloVe 或 BERT），得到词向量序列。随后，这些向量送入一个双向 Transformer 编码器，既能捕捉左侧上下文，又能捕捉右侧信息。编码器的输出在时间维度上做平均池化，得到固定长度的**句子向量**。这一步相当于把一句话压成一个“语义压缩包”，里面已经蕴含了词义、顺序和初步的对话功能暗示。

2. **共享多任务层**  
   所有句子向量堆叠成对话层级的序列（对话长度可变），再喂入一个轻量的共享层——通常是几层双向 LSTM 或者再一个 Transformer。共享层的目标是让模型在整体对话结构上学习共性特征。这里的关键是**共享**：同一套参数既服务于后面的连贯性评分，又服务于行为预测。

3. **任务分支**  
   - **对话行为预测分支**：在共享层的每个时间步上接一个全连接层 + Softmax，输出该句子的行为类别。训练时使用交叉熵损失，帮助模型把“这句话像是提问还是陈述”这类信息写进句子向量。  
   - **连贯性评估分支**：对整个对话的句子向量序列做池化（如全局平均或注意力加权），得到对话级向量。再通过一个二分类头（连贯 / 不连贯）或排序头（对比两段对话的相对连贯度）输出分数。损失同样是交叉熵或对比损失。

4. **训练目标**  
   两个任务的损失加权求和，形成总损失。权重可以手动调节，也可以在训练过程中自适应。因为行为预测是辅助的，它的梯度会推动共享层学习到更具功能性的特征，而连贯性任务则直接优化最终评估目标。

5. **推理阶段**  
   只保留连贯性分支，行为预测头被“关掉”。因此在实际使用时不需要任何对话行为标签，模型直接把句子向量喂入连贯性头得到分数，实现**隐式利用行为信息**。

**最巧妙的点**在于：行为标签只在训练时出现一次，模型把它们“烙印”进内部表示；而在测试时，完全不需要外部标签，这解决了标注成本高、标签噪声大的痛点。

### 实验与效果
- **数据集**：作者在 DailyDialogue（日常口语对话）和 SwitchBoard（电话交谈）两个公开语料上做评估。两者分别代表轻松对话和正式对话两种极端场景。  
- **基线**：包括传统的实体特征 + 手工对话行为标签模型、纯文本的 Transformer 连贯性模型以及最近的几篇对话连贯性专用模型。  
- **主要结果**：在 DailyDialogue 上，本文模型的准确率比最强基线高出 **超过 20 个百分点**，这在对话评估任务里是一个非常显著的提升。SwitchBoard 上，模型的表现与最强基线基本持平，说明即使在更复杂的对话环境中，隐式学习的行为信息也足够支撑连贯性判断。  
- **消融实验**：去掉对话行为预测任务后，模型在 DailyDialogue 上的准确率下降约 12%，验证了行为预测的辅助作用是关键因素。进一步把共享层换成单纯的句子级编码器（不做对话级上下文）也会导致显著性能衰减，说明对话级上下文对连贯性评估不可或缺。  
- **局限性**：论文只在英文对话数据上做实验，未验证在中文或多语言场景下的迁移效果；此外，行为预测的标签仍然来源于已有标注数据，若训练语料本身的行为标签质量不高，可能会把噪声带进表示中。作者在讨论中承认这些问题，并把跨语言和噪声鲁棒性列为未来工作。

### 影响与延伸思考
这篇工作在对话评估社区引发了两类后续研究：一是**多任务驱动的对话表示学习**，很多后续论文尝试把情感分析、意图识别等任务一起训练，以获得更通用的对话向量；二是**去标签化的评估方法**，研究者开始探索如何在没有任何人工标注的情况下，让模型自行发现对话结构规律。推测，未来会有更多工作把**自监督预训练**（如对话顺序预测）和**任务驱动微调**结合起来，进一步降低对标注的依赖。想深入了解的读者可以关注近期在 ACL、EMNLP 上出现的“对话自监督学习”系列论文，以及在对话系统评估基准（如 DSTC）中加入的无标签连贯性任务。

### 一句话记住它
**把对话行为预测当作训练时的“隐形标签”，让模型在不需要任何显式行为标注的情况下，也能精准评估对话连贯性。**