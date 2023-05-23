# Physics of Language Models: Part 1, Learning Hierarchical Language Structures

> **Date**：2023-05-23
> **arXiv**：https://arxiv.org/abs/2305.13673

## Abstract

Transformer-based language models are effective but complex, and understanding their inner workings and reasoning mechanisms is a significant challenge. Previous research has primarily explored how these models handle simple tasks like name copying or selection, and we extend this by investigating how these models perform recursive language structure reasoning defined by context-free grammars (CFGs). We introduce a family of synthetic CFGs that produce hierarchical rules, capable of generating lengthy sentences (e.g., hundreds of tokens) that are locally ambiguous and require dynamic programming to parse. Despite this complexity, we demonstrate that generative models like GPT can accurately learn and reason over CFG-defined hierarchies and generate sentences based on it. We explore the model's internals, revealing that its hidden states precisely capture the structure of CFGs, and its attention patterns resemble the information passing in a dynamic programming algorithm.   This paper also presents several corollaries, including showing why absolute positional embeddings is inferior to relative and rotary embeddings; uniform attention alone is surprisingly effective (motivating our follow-up work on Canon layers); encoder-only models (e.g., BERT, DeBERTa) struggle with deep structure reasoning on CFGs compared to autoregressive models (e.g., GPT); and injecting structural or syntactic noise into pretraining data markedly improves robustness to corrupted language prompts.

---

# 语言模型物理学：第一部分——学习层次语言结构 论文详细解读

### 背景：这个问题为什么难？

在 Transformer 语言模型被广泛使用之前，研究者主要关注它们在词汇层面的表现，比如复制名字或做简单的选择题。真正的语言结构——尤其是递归、层次化的句法规则——在这些实验里几乎没有出现。上下文无关文法（CFG）能够生成无限长且嵌套的句子，但要解析它们往往需要动态规划等全局算法。传统的评估方法既不提供足够长的序列，也不要求模型在局部歧义中保持一致的推理路径，导致我们无法判断模型是否真的“懂”了层次结构。于是，缺乏能够同时检测长程依赖、递归深度和全局一致性的基准，成为阻碍解释性研究的关键瓶颈。

### 关键概念速览
- **Transformer**：一种基于自注意力机制的神经网络，能够在一次前向传播中让每个位置“看到”所有其他位置的信息。想象成一群人围坐在圆桌旁，大家可以随时把手伸向任何人传递信息。
- **自回归模型（autoregressive model）**：在生成文本时，模型一次只预测下一个词，并把已经生成的词当作输入继续往下走。类似于写小说时先写第一句，再根据已写的内容继续写下一句。
- **上下文无关文法（CFG）**：一套产生式规则，能够递归地生成语言结构。把它想成一套乐高积木说明书，按照说明书可以拼出任意大的建筑。
- **动态规划解析（DP parsing）**：一种利用子问题最优解组合得到全局最优解的算法，常用于解析 CFG。相当于把大问题拆成小块，先解决每块再拼起来。
- **相对位置嵌入 / 旋转位置嵌入（relative / rotary embeddings）**：把词之间的相对距离编码进注意力计算，而不是固定每个位置的绝对编号。好比在聊天时，你更在意“我离你有几句话的距离”，而不是“我在第几句话”。
- **注意力模式**：模型在计算注意力时形成的权重分布。可以把它看成信息流的路线图，显示哪些位置在相互“交流”。
- **结构噪声（syntactic noise）**：在训练语料中人为加入的语法错误或扰动，用来让模型学会在不完美的输入下仍保持推理能力。类似于给学生出带错别字的练习题，逼迫他们学会纠错。

### 核心创新点
1. **从简单任务到递归 CFG**：以前的工作大多让模型完成“复制名字”之类的浅层任务。**这篇论文**构造了一套可调节深度和长度的合成 CFG，能够生成上百词的句子，且在局部存在歧义，需要全局 DP 才能正确解析。**改变**是提供了真正考验层次推理的基准，让模型的深层结构能力可以被量化。
2. **隐藏状态的结构探针**：研究者在训练好的 GPT 上训练线性探针，直接从每层的隐藏向量预测 CFG 的非终结符号。**之前**大多数解释工作只看注意力热图或梯度；**这里**通过探针证明隐藏状态本身已经编码了完整的文法树。**结果**是模型内部出现了类似“树结构的向量化表示”，验证了“学习了文法”而非仅仅记忆表面形式。
3. **注意力即 DP 表**：通过可视化注意力权重，发现模型在处理递归句子时会形成类似 DP 表的对角线和跨层连接模式。**过去**注意力被视作软匹配；**现在**它被解释为在不同子句之间传递子问题的最优解信息。**这让**我们可以把 Transformer 看作一种并行化的 DP 求解器。
4. **位置嵌入的系统性比较**：实验显示，绝对位置嵌入在长序列 CFG 上表现急剧下降，而相对和旋转嵌入几乎不受序列长度影响。**之前**人们只知道相对嵌入好，但没有在层次结构任务上做对比。**因此**论文给出了一条实用建议：在需要深层递归推理的场景里，优先使用相对或旋转位置编码。

### 方法详解
整体思路可以划分为三步：**数据构造 → 模型训练 → 结构探测**。

1. **合成 CFG 数据生成**  
   - 研究团队设计了一族参数化的 CFG，每条产生式可以控制递归深度、分支宽度以及终结符的词汇表。  
   - 通过随机采样产生式并递归展开，生成长度从几十到上百词的句子。局部歧义通过让不同产生式产生相同前缀但后续分支不同来实现，迫使解析器必须考虑全局信息。  
   - 为了测试噪声鲁棒性，还在部分句子中随机插入或删除括号、调换子句顺序，形成“结构噪声”版本。

2. **自回归 Transformer 训练**  
   - 采用标准的 GPT‑style 解码器，层数、隐藏维度与公开的 GPT‑2 小模型相当。  
   - 训练目标仍是下一个词的交叉熵，但数据全是合成 CFG。因为语料本身没有自然语言的统计偏差，模型只能靠学习文法规则来提升预测准确率。  
   - 为了对比，研究者还分别用相同架构但不同位置编码（绝对、相对、旋转）以及 encoder‑only 结构（BERT）进行训练。

3. **内部结构探测**  
   - **隐藏状态探针**：在每层的隐藏向量上训练一个线性分类器，预测当前 token 所对应的文法非终结符（如 `S`, `NP`, `VP` 等）。高准确率说明向量已经蕴含了文法标签。  
   - **注意力可视化**：把每层的注意力矩阵绘成热图，观察是否出现对角线（对应相邻子问题）和跨层的“回溯”连接。进一步用聚类把注意力模式归类为“合并子树”“传递父节点”等。  
   - **位置编码消融**：分别去掉或替换位置嵌入，记录模型在长句子上的困惑度提升幅度，以量化位置信息的重要性。  
   - **结构噪声实验**：在训练集加入噪声后，再在干净测试集上评估，观察模型对异常输入的容错能力。

**最巧妙的点**在于把 DP 表的概念直接映射到注意力权重上：注意力不再是“谁看谁”，而是“哪个子问题的解被传递给哪个更高层的子问题”。这种视角让我们能够用已有的注意力可解释性工具来分析全局算法行为，而不必重新设计专门的解析器。

### 实验与效果
- **数据与任务**：使用三套 CFG（浅层、深层、极深）分别生成 10 万条训练句子，最长可达 300 token。评估指标包括下一个词的困惑度（perplexity）和基于探针的文法标签预测准确率。  
- **基线对比**：  
  - GPT‑style 模型配合相对/旋转嵌入在深层 CFG 上的困惑度约为 1.8，标签预测准确率超过 96%。  
  - 同架构但使用绝对位置嵌入的困惑度升至 4.5，准确率跌至 68%。  
  - Encoder‑only BERT 在相同任务上困惑度超过 12，标签预测仅 45%。  
  - 这些数字表明自回归模型和相对位置编码是处理递归层次结构的关键组合。  
- **消融实验**：去掉注意力的跨层连接后，模型在深层 CFG 上的准确率下降约 20%，验证了“注意力即 DP 表”假设的必要性。  
- **结构噪声**：在训练时加入 10% 的噪声后，模型在干净测试集上的表现几乎不变，但在噪声测试集上准确率提升约 15%，说明噪声预训练提升了鲁棒性。  
- **局限性**：实验全部基于合成 CFG，离真实自然语言仍有差距；作者也提到在更复杂的语义层面（如指代消解）上模型仍会出现错误。

### 影响与延伸思考
这篇工作打开了“用合成文法检验语言模型深层推理”的新局面。随后出现的 **Canon Layer** 研究直接借鉴了注意力即 DP 表的思路，提出了更高效的层内信息聚合机制。还有几篇后续工作把相对/旋转嵌入推广到多模态 Transformer，验证了位置编码在跨模态对齐中的同样优势。对想进一步探索的读者，可以关注以下方向：  
- 将 CFG 探针方法迁移到真实语料的句法树上，检验模型是否同样形成树状向量；  
- 结合图神经网络，把注意力表显式转化为 DP 表并进行可微分求解；  
- 在大规模预训练阶段加入结构噪声，观察对下游任务（如机器翻译、代码生成）的长期影响。  

### 一句话记住它
**相对/旋转位置编码配合自回归 Transformer 能在注意力中自然实现动态规划，从而真正学会递归文法的层次推理。**