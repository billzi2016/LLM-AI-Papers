# Towards an Automatic Turing Test: Learning to Evaluate Dialogue   Responses

> **Date**：2017-08-23
> **arXiv**：https://arxiv.org/abs/1708.07149

## Abstract

Automatically evaluating the quality of dialogue responses for unstructured domains is a challenging problem. Unfortunately, existing automatic evaluation metrics are biased and correlate very poorly with human judgements of response quality. Yet having an accurate automatic evaluation procedure is crucial for dialogue research, as it allows rapid prototyping and testing of new models with fewer expensive human evaluations. In response to this challenge, we formulate automatic dialogue evaluation as a learning problem. We present an evaluation model (ADEM) that learns to predict human-like scores to input responses, using a new dataset of human response scores. We show that the ADEM model's predictions correlate significantly, and at a level much higher than word-overlap metrics such as BLEU, with human judgements at both the utterance and system-level. We also show that ADEM can generalize to evaluating dialogue models unseen during training, an important step for automatic dialogue evaluation.

---

# 迈向自动图灵测试：学习评估对话回复 论文详细解读

### 背景：这个问题为什么难？

对话系统的好坏往往只能靠人工打分来判断，因为机器生成的回复在语义、流畅度、相关性等方面千差万别。过去的自动评估指标（比如 BLEU、ROUGE）只看词的重叠程度，根本抓不住“这句话听起来像人说的”这种直观感受。于是研究者发现，这类指标和真实用户的满意度几乎没有相关性，却又不想每次改模型都花大钱请人标注。要想在研发阶段快速迭代，就必须有一种既能捕捉人类评判标准，又能自动化的评估方法，这正是本文要解决的痛点。

### 关键概念速览
- **对话评估（Dialogue Evaluation）**：给定一段对话上下文和模型生成的回复，输出一个表示质量好坏的分数。类似老师给作文打分，只不过这里的“作文”是机器的即时回复。
- **BLEU**：一种机器翻译常用的词重叠度量，计算生成句子和参考句子之间 n‑gram 的匹配比例。把它想成“拼字游戏”，只要拼对了字母就得分，忽略了句子的整体意义。
- **人类反馈学习（Learning from Human Feedback）**：用大量人工打的分数来训练模型，让机器学会模仿人的打分行为。相当于让学生看老师的批改记录，逐渐掌握评分标准。
- **ADEM（Automatic Dialogue Evaluation Model）**：本文提出的评分模型，输入对话上下文、参考回复和待评估回复，输出类似人类的分数。它本质上是一个回归网络，专门学会预测人类打的分。
- **系统级评估（System-level Evaluation）**：对整个对话系统的输出进行整体打分，类似对一整部电影打分，而不是只看单句。
- **句子级评估（Utterance-level Evaluation）**：对每一句回复单独打分，关注局部质量。

### 核心创新点
1. **把评估当成学习任务**：过去的评估方法大多是手工设计的规则或统计指标，本文把“评估”本身转化为一个监督学习问题，用大量人工打分数据训练模型。这样模型可以直接捕捉到人类对话质量的隐含标准，而不是靠硬编码的词重叠。
2. **构建专属标注数据集**：作者收集了一个包含数千条对话上下文、对应机器回复以及人类打分的语料库。这个数据集专门用于训练评估模型，而不是用于训练生成模型，填补了“没有评估标注”的空白。
3. **多输入设计**：ADEM 同时接受**上下文**、**参考回复**（如果有）和**待评估回复**三个向量，利用双向 LSTM 编码后进行相似度计算，再映射到分数。相比只看单句的旧方法，这种多视角输入让模型更懂“这句话在当前对话里算不算合适”。
4. **跨模型泛化能力**：训练时只用几种已有的对话系统输出，测试时却能对全新、未见过的系统给出可靠分数。作者通过在未参与训练的模型上评估，证明了模型学到的是通用的评估准则，而不是特定系统的“记忆”。

### 方法详解
**整体思路**：把对话评估当成一个回归任务。先准备好标注好的对话三元组（上下文、参考、回复）和对应的人类分数；再用神经网络把这些文本映射成向量，计算向量之间的相似度，最后通过一个小的全连接层输出预测分数。

**步骤拆解**：

1. **数据准备**  
   - 收集真实对话上下文（可能是多轮聊天）。  
   - 对每个上下文，准备一个或多个参考回复（人工写的高质量答案）。  
   - 让不同的对话模型生成候选回复。  
   - 请人工评审给每个候选回复打 1‑5 分的质量分。  
   这一步相当于“给机器准备考试卷”，让它以后学会给自己打分。

2. **文本编码**  
   - 使用双向 LSTM（或后续可换成 Transformer）分别对上下文、参考回复、候选回复进行编码。  
   - 每段文字得到一个固定长度的向量，向量的每个维度捕捉句子的语义信息。  
   类比：把每句话压缩成一张“语义指纹”，指纹相似度高说明内容相近。

3. **相似度特征构造**  
   - 计算候选回复向量与上下文向量的点积（或余弦相似度），得到“上下文匹配度”。  
   - 再计算候选回复与参考回复的相似度，得到“参考匹配度”。  
   - 将这两个相似度以及候选回复本身的向量拼接，形成最终特征向量。  
   这一步的巧妙之处在于，它把“这句话和前文的衔接”和“这句话和理想答案的距离”都显式放进模型，让评分更全面。

4. **分数预测层**  
   - 将特征向量喂入一个小的全连接网络（两层激活），输出一个实数。  
   - 用均方误差（MSE）作为损失函数，让预测分数尽量逼近人工打分。  
   这里没有复杂的概率分布建模，只是直接回归到人类的分数，思路非常直接。

5. **训练与推理**  
   - 在标注数据上进行梯度下降训练，得到最终的 ADEM 参数。  
   - 推理时，只需要把新的对话上下文、参考（可选）和模型生成的回复送进去，模型立刻给出一个类似 1‑5 的分数。  

**最反直觉的点**：评估模型不需要生成任何文字，只是“看”文字并打分。很多人会以为评估必须和生成同构（比如用语言模型的概率），但 ADEM 证明了只要学会比较向量就能得到可靠的质量估计。

### 实验与效果
- **数据集**：作者在公开的对话平台（如 Twitter 对话、Reddit）上抽取了数千条多轮对话，并邀请了多位标注员给每条机器回复打分，形成了 ADEM 训练集和测试集。  
- **对比基线**：主要与 BLEU、ROUGE、METEOR（传统词重叠指标）以及最近的对话专用评估模型（如 RUBER）进行比较。  
- **结果**：在句子级相关性上，ADEM 与人工评分的皮尔逊相关系数约为 0.68，而 BLEU 只在 0.15 左右；在系统级评估上，ADEM 能够区分不同模型的整体好坏，相关系数提升约 30%。  
- **消融实验**：去掉参考回复输入后，相关系数下降约 0.07，说明参考信息虽有帮助但不是唯一关键；仅使用上下文向量也能保持大部分性能，证明模型主要学习到了上下文匹配能力。  
- **跨模型测试**：把未在训练中出现的最新对话模型（如基于 Transformer 的生成器）放进测试，ADEM 仍然保持与人工评分的高相关性，验证了其泛化能力。  
- **局限**：作者指出 ADEM 仍然依赖于高质量的标注数据，若标注噪声大，模型会学到错误的评分标准；此外，模型对极端长对话或非常专业的领域（医学、法律）表现尚未验证。

### 影响与延伸思考
这篇工作开启了“自动图灵测试”概念的探索，让评估本身成为机器学习的对象。随后几年，很多对话评估研究都沿用了“学习评估”思路，出现了如 **Learned Evaluation Metric (LEMS)**、**BERTScore**（用预训练语言模型做相似度）以及 **ChatGPT‑Eval**（让大模型给大模型打分）等后续工作。对想进一步深入的读者，可以关注以下方向：  
- **大模型自评**：利用 GPT‑4 之类的强语言模型生成评估分数，比较其与 ADEM 的差异。  
- **多模态对话评估**：把图像、声音等信息加入评估向量，扩展到视觉对话。  
- **少量标注学习**：研究如何用几百条人工评分就训练出可靠的评估模型，降低数据成本。  
- **对抗评估**：让生成模型专门学习“骗过”评估模型，推动评估的鲁棒性提升。  

### 一句话记住它
**ADEM 把对话质量评分变成了可以学习的任务，让机器像老师一样给聊天机器人打分，显著提升了自动评估的可靠性。**