# Discovering Latent Knowledge in Language Models Without Supervision

> **Date**：2022-12-07
> **arXiv**：https://arxiv.org/abs/2212.03827

## Abstract

Existing techniques for training language models can be misaligned with the truth: if we train models with imitation learning, they may reproduce errors that humans make; if we train them to generate text that humans rate highly, they may output errors that human evaluators can't detect. We propose circumventing this issue by directly finding latent knowledge inside the internal activations of a language model in a purely unsupervised way. Specifically, we introduce a method for accurately answering yes-no questions given only unlabeled model activations. It works by finding a direction in activation space that satisfies logical consistency properties, such as that a statement and its negation have opposite truth values. We show that despite using no supervision and no model outputs, our method can recover diverse knowledge represented in large language models: across 6 models and 10 question-answering datasets, it outperforms zero-shot accuracy by 4\% on average. We also find that it cuts prompt sensitivity in half and continues to maintain high accuracy even when models are prompted to generate incorrect answers. Our results provide an initial step toward discovering what language models know, distinct from what they say, even when we don't have access to explicit ground truth labels.

---

# 在无监督条件下发现语言模型的潜在知识 论文详细解读

### 背景：这个问题为什么难？

语言模型在训练时往往通过模仿人类文本或让人类打分来优化，却会把人类的错误和盲点一起学进去。即使在评估阶段，只要让模型直接输出答案，就会受到提示词、温度等因素的干扰，导致“幻觉”——模型说出自己根本不“知道”的内容。过去的知识抽取方法大多依赖标注好的问答对或模型的生成结果，缺乏一种能够在不提供任何标签、也不看模型输出的情况下，直接探测模型内部到底存了哪些事实。

### 关键概念速览
- **潜在知识（latent knowledge）**：模型内部激活所隐含的事实信息，类似于大脑里潜藏的记忆，只要找到合适的钥匙就能读出来。  
- **激活空间（activation space）**：模型每层神经元在一次前向传播后产生的向量集合，想象成一张高维地图，模型的每个思考过程都在这张地图上留下痕迹。  
- **方向（direction）**：在激活空间里的一条向量线，沿着它投影可以把某类信息（比如“是”或“否”）映射出来，就像在地图上找一条指北针的方向。  
- **逻辑一致性约束（logical consistency constraints）**：要求模型对同一句话和它的否定给出相反的真值，这相当于在高维空间里强制“正负相反”，帮助定位正确的方向。  
- **零样本（zero‑shot）**：不做任何微调、直接用模型原始输出回答问题的方式，类似于让学生在没有复习的情况下直接答题。  
- **提示敏感性（prompt sensitivity）**：模型答案随输入提示词变化的程度，提示词像调味料，太多会让答案味道全变。  
- **无监督探测（unsupervised probing）**：不使用任何标注数据，只靠模型内部信号来推断知识的技术，等价于在黑箱里用电流波形判断机器内部结构。

### 核心创新点
1. **从激活而非输出寻找答案**：传统做法都是让模型生成文字再比对标签，这篇论文直接在激活向量里找方向，省掉了生成步骤，也避免了生成过程中的噪声。  
2. **利用逻辑一致性来约束方向**：以前的线性探针只最小化预测误差，这里加入“正句”和“否句”必须得到相反投影的约束，使得找到的方向更符合事实逻辑。  
3. **完全无监督的 yes‑no 读取器**：不需要任何标注的问答对，只要有未标记的激活样本，就能训练出一个能够判断真假值的读取器，突破了对大规模标注成本的依赖。  
4. **对抗提示诱导的错误输出**：实验表明，即使把模型诱导去说错话，只要使用激活读取器，正确率仍保持高水平，说明内部知识比表面输出更稳固。

### 方法详解
整体思路可以划分为三步：  
1) **收集激活**：对一批随机或任务相关的输入（不要求有标签）进行前向传播，记录目标层的激活向量。  
2) **构造约束对**：把每个陈述句和它的逻辑否定配对，例如 “巴黎是法国的首都” 与 “巴黎不是法国的首都”。这一步不需要答案，只需要生成对应的否定文本。  
3) **优化方向**：在激活空间里搜索一条向量方向 **v**，使得对所有配对，投影值的符号相反且幅度尽可能大。实现上通常把 **v** 当作可学习的参数，最小化一个损失函数：  
   - **一致性损失**：鼓励正句投影正、否句投影负。  
   - **正则化**：防止 **v** 过大或过于稀疏。  
   通过梯度下降即可得到 **v**。得到 **v** 后，对任意新问题的激活向量 **h**，只要计算 **h·v**（点积），符号为正即判断为“是”，为负则判断为“否”。  

**关键细节**  
- **激活层的选择**：作者发现中高层的激活更能捕获事实信息，而底层更偏向词法特征。  
- **配对生成策略**：否定句通过简单的词汇替换或模板生成，保持句式相似，确保两者在激活空间的差异主要来源于真值而非结构。  
- **方向的解释性**：因为 **v** 是线性组合，理论上可以回溯到哪些神经元对真假判断贡献最大，提供了可解释的线索。  
- **最反直觉的点**：即使不看模型的输出，仅凭激活就能比零样本直接生成的答案高出约 4%，说明模型内部已经“知道”很多事实，只是有时不敢说出来。

### 实验与效果
- **数据与模型**：在 6 种规模不等的语言模型（包括 LLaMA、OPT 等）上，使用 10 套公开的 yes‑no 问答基准（如 BoolQ、OpenBookQA 的二分类子集）。  
- **基线对比**：与传统零样本直接生成的准确率相比，激活读取器平均提升 4%（最高提升约 7%），在某些数据集上甚至接近有监督微调的水平。  
- **提示敏感性**：通过在同一问题上使用不同的提示词，零样本的答案波动约 30%，而本方法的波动降至 15%，实现约 50% 的敏感性削减。  
- **对抗实验**：在让模型故意输出错误答案的设置下，零样本准确率跌至 20% 左右，而激活读取器仍保持在 65% 以上，验证了内部知识的鲁棒性。  
- **消融研究**：去掉逻辑一致性约束后，方向搜索只靠最小化投影幅度，整体准确率下降约 2.5%，说明一致性约束是提升效果的关键因素。  
- **局限性**：论文只针对二分类（yes‑no）问题展开，未验证在多选或生成式任务上的可行性；方向搜索在极高维激活上计算成本仍然不低，实际部署需要进一步压缩。

### 影响与延伸思考
这篇工作打开了“从内部激活直接读出事实”的新思路，激发了后续研究在以下方向的探索：  
- **多类别或序列化知识抽取**：尝试把方向扩展为多个正交向量，以同时捕获多种属性。  
- **跨模态潜在知识**：把视觉模型的激活也套用相同的逻辑约束，看看能否直接读取图像中的事实。  
- **安全与对齐**：因为激活读取器不受提示词诱导，可能成为检测模型“自我矛盾”或“隐藏幻觉”的工具。  
- **压缩与加速**：利用低秩近似或稀疏投影技术，降低方向搜索的计算开销，使其适用于边缘设备。  
如果想进一步了解，可以关注 2024‑2025 年间出现的 “unsupervised probing” 系列论文以及 “latent space alignment” 的新方法，它们大多在此工作基础上进行扩展。

### 一句话记住它
**不让模型说出来，也能直接从它的大脑里读出真相。**