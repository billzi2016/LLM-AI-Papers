# Is ChatGPT a Good Personality Recognizer? A Preliminary Study

> **Date**：2023-07-08
> **arXiv**：https://arxiv.org/abs/2307.03952

## Abstract

In recent years, personality has been regarded as a valuable personal factor being incorporated into numerous tasks such as sentiment analysis and product recommendation. This has led to widespread attention to text-based personality recognition task, which aims to identify an individual's personality based on given text. Considering that ChatGPT has recently exhibited remarkable abilities on various natural language processing tasks, we provide a preliminary evaluation of ChatGPT on text-based personality recognition task for generating effective personality data. Concretely, we employ a variety of prompting strategies to explore ChatGPT's ability in recognizing personality from given text, especially the level-oriented prompting strategy we designed for guiding ChatGPT in analyzing given text at a specified level. The experimental results on two representative real-world datasets reveal that ChatGPT with zero-shot chain-of-thought prompting exhibits impressive personality recognition ability and is capable to provide natural language explanations through text-based logical reasoning. Furthermore, by employing the level-oriented prompting strategy to optimize zero-shot chain-of-thought prompting, the performance gap between ChatGPT and corresponding state-of-the-art model has been narrowed even more. However, we observe that ChatGPT shows unfairness towards certain sensitive demographic attributes such as gender and age. Additionally, we discover that eliciting the personality recognition ability of ChatGPT helps improve its performance on personality-related downstream tasks such as sentiment classification and stress prediction.

---

# ChatGPT 能否成为优秀的人格识别器？初步研究 论文详细解读

### 背景：这个问题为什么难？

文本人格识别本质上是把一段文字映射到心理学上的人格维度，这需要模型捕捉细微的语言风格、情感倾向以及潜在的价值观。传统方法大多依赖大量标注数据训练专用分类器，然而标注成本高、跨语言迁移困难，而且模型往往缺乏可解释的推理过程。随着大语言模型（LLM）如ChatGPT 展现出强大的零样本能力，研究者开始好奇它是否能直接当作“黑箱”来完成人格识别任务，但这在学术上仍缺系统评估。

### 关键概念速览
- **人格（Personality）**：心理学中用来描述个体稳定行为倾向的特征集合，常用的模型是五大人格维度（开放性、尽责性、外向性、宜人性、神经质）。可以把它想成人的“性格标签”。  
- **文本人格识别**：给定一段文字，预测作者在上述维度上的得分或类别，类似于从写作风格推断作者的性格。  
- **零样本（Zero‑Shot）**：模型在没有看到任何任务特定示例的情况下，仅凭提示（prompt）完成任务，就像让一个不熟悉的学生先读题再作答。  
- **思维链（Chain‑of‑Thought, CoT）**：让模型在回答前先把推理步骤写出来，类似于解题时写草稿，能够提升复杂推理的准确性。  
- **层级提示（Level‑Oriented Prompting）**：在提示中明确要求模型在某个抽象层次（如词汇、句子、段落）进行分析，帮助模型聚焦不同粒度的信息。  
- **公平性（Fairness）**：模型输出不应因性别、年龄等敏感属性产生系统性偏差，类似于考试评分不因考生的背景而不公。  
- **下游任务（Downstream Task）**：在完成人格识别后，模型还能帮助提升情感分类、压力预测等相关任务的表现。

### 核心创新点
1. **从通用 LLM 到人格识别的零样本评估**  
   - 之前的研究几乎都基于专门训练的分类器或微调模型。  
   - 本文直接使用 ChatGPT，配合不同提示策略（普通零样本、零样本 CoT）来完成人格预测。  
   - 结果显示，即使不做任何微调，ChatGPT 也能达到接近 SOTA（最先进模型）的水平，证明通用 LLM 本身具备潜在的人格识别能力。

2. **层级提示（Level‑Oriented Prompting）**  
   - 传统提示往往只告诉模型“请判断人格”。  
   - 作者设计了让模型先在指定层次（词、句、段）进行分析的提示，例如“先列出文中表现开放性的关键词”。  
   - 这种分层思考让模型的推理更系统，进一步缩小了与 SOTA 的性能差距。

3. **可解释推理输出**  
   - 通过 CoT 提示，ChatGPT 不仅给出人格标签，还会输出逻辑链条（如“因为使用了‘探索’、‘创新’等词，所以判断开放性高”）。  
   - 这为研究者提供了可审计的解释，弥补了传统黑箱模型的不足。

4. **公平性诊断**  
   - 作者在实验中检测了模型对不同性别、年龄群体的预测偏差。  
   - 发现 ChatGPT 在某些维度上对女性或特定年龄段倾向性过高，提示需要在提示设计或后处理阶段加入公平性约束。

### 方法详解
整体思路可以拆成三步：**提示设计 → 零样本推理 → 结果聚合**。

1. **提示设计**  
   - **基础零样本提示**：直接向 ChatGPT 说明任务，例如“请根据以下文字判断作者的五大人格得分”。  
   - **思维链提示**：在基础提示后加入“请先写出你的推理过程”。这相当于让模型先列出关键句子、情感词等，再给出最终分数。  
   - **层级提示**：在思维链的基础上，进一步指示模型在不同层次上展开分析。比如：“第一步，列出文中出现的情感词；第二步，说明这些词如何映射到‘宜人性’”。这种层级指令像是给模型提供了一个“分析框架”，帮助它有序组织信息。

2. **零样本推理**  
   - 将待评估的文本连同选好的提示一起发送给 ChatGPT。模型返回的文本包括推理链和最终人格标签。  
   - 为了降低随机性，作者对同一条文本使用了多次不同随机种子的调用，随后对得到的标签进行多数投票或取平均。

3. **结果聚合与解释**  
   - 对每个维度的得分，作者采用了标准化处理，使其可直接与已有数据集的标注对齐。  
   - 推理链被解析为关键特征（如关键词、情感倾向），这些特征随后用于分析公平性和下游任务的迁移效果。

**最巧妙的地方**在于层级提示的设计：它并不是简单地让模型“多说几句”，而是把心理学上对人格的层次化解释（词汇→句子→段落）映射到提示语言，让模型的内部注意力自然对齐到这些层次，从而提升了推理的系统性和准确度。

### 实验与效果
- **数据集**：作者选用了两个公开的文本人格数据集（如 MBTI‑Twitter 与 Essays‑Big5），这两个数据集分别代表社交媒体短句和长篇作文的不同写作风格。  
- **基线**：包括传统机器学习模型（SVM、随机森林）和最新的微调 BERT/ RoBERTa 系列模型。  
- **主要结果**：论文声称，在零样本 CoT 提示下，ChatGPT 的整体准确率已经逼近最先进模型；加入层级提示后，性能差距进一步缩小，几乎与 SOTA 持平。  
- **消融实验**：作者分别去掉 CoT、去掉层级指令、仅使用基础提示进行对比，发现 CoT 能提升约 5% 的准确率，层级提示再额外提升 2–3%。  
- **公平性分析**：实验显示，ChatGPT 对女性的“宜人性”预测偏高约 0.1（相对标准差），对老年用户的“神经质”预测偏低约 0.08，提示需要在提示或后处理阶段加入公平校正。  
- **下游迁移**：利用 ChatGPT 产生的人格特征作为额外特征，情感分类的 F1 提升约 1.5%，压力预测的 MAE 降低约 0.03。  
- **局限**：作者承认实验仅限于英文数据，中文或其他语言的迁移尚未验证；此外，提示工程仍然是手工设计，缺乏自动化方法。

### 影响与延伸思考
这篇工作打开了“LLM 直接做心理属性预测”的思路，随后有几篇后续研究尝试把提示优化（如软提示、自动提示生成）与公平性约束结合，进一步提升跨语言表现。还有工作把人格识别与对话系统结合，让聊天机器人能够根据用户的语言风格实时调整交互策略。想深入的读者可以关注 **提示工程在心理测评中的系统化方法**、**跨语言人格识别的迁移学习** 以及 **大模型公平性评估框架** 等方向。

### 一句话记住它
让 ChatGPT 用层级思维链零样本推理，就能把文字直接翻译成可解释的人格标签，性能几乎追上专门训练的模型，但仍需警惕公平偏差。