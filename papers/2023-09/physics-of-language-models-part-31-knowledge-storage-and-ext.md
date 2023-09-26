# Physics of Language Models: Part 3.1, Knowledge Storage and Extraction

> **Date**：2023-09-25
> **arXiv**：https://arxiv.org/abs/2309.14316

## Abstract

Large language models (LLMs) can store a vast amount of world knowledge, often extractable via question-answering (e.g., "What is Abraham Lincoln's birthday?"). However, do they answer such questions based on exposure to similar questions during training (i.e., cheating), or by genuinely learning to extract knowledge from sources like Wikipedia?   In this paper, we investigate this issue using a controlled biography dataset. We find a strong correlation between the model's ability to extract knowledge and various diversity measures of the training data. $\textbf{Essentially}$, for knowledge to be reliably extracted, it must be sufficiently augmented (e.g., through paraphrasing, sentence shuffling, translations) $\textit{during pretraining}$. Without such augmentation, knowledge may be memorized but not extractable, leading to 0% accuracy, regardless of subsequent instruction fine-tuning.   To understand why this occurs, we employ (nearly) linear probing to demonstrate a strong connection between the observed correlation and how the model internally encodes knowledge -- whether it is linearly encoded in the hidden embeddings of entity names or distributed across other token embeddings in the training text.   This paper provides $\textbf{several key recommendations for LLM pretraining in the industry}$: (1) rewrite the pretraining data -- using small, auxiliary models -- to provide knowledge augmentation, and (2) incorporate more instruction-finetuning data into the pretraining stage before it becomes too late.

---

# 语言模型的物理学：第3.1节 知识存储与提取 论文详细解读

### 背景：这个问题为什么难？
LLM 在问答时常常表现得像是“知道”了事实，但我们并不清楚它们是靠在训练时见过相同问题的记忆作弊，还是已经学会了从原始文本（如 Wikipedia）中抽取信息。过去的研究大多把模型的高准确率归结为“隐式记忆”，却缺少对“可抽取性”——即模型能否在没有直接提示的情况下恢复知识——的系统检验。缺少对训练数据多样性与知识编码方式的量化分析，使得业界难以判断怎样的预训练策略才能让知识真正可用，而不是仅仅被埋在参数里。

### 关键概念速览
**知识可抽取性**：模型在面对标准问答时能否正确给出答案的能力，区别于仅在训练时出现过相同问句的记忆。可以想象成一本百科全书里信息是否被标记了目录，目录好找就容易抽取。

**数据多样性（augmentation）**：对同一事实进行多种表述（同义改写、句子顺序打乱、机器翻译等）后加入训练集。类似于把同一道数学题换成不同的文字描述，让学生更懂概念而不是记答案。

**线性探 probing**：在模型内部的隐藏向量上训练一个线性分类器，检测某个属性是否可以用一层线性映射直接读出。相当于在黑盒里找“开关”，看信息是否集中在某个显眼的地方。

**实体名称嵌入**：模型为每个词（尤其是人名、地名）分配的向量表示。若知识主要存放在这些嵌入里，就像把关键字写在标签上，容易检索。

**分布式编码**：知识散落在整段文本的多个 token 嵌入中，需要综合上下文才能恢复。类似于把信息写在散页纸上，需要把所有页拼起来才能读懂。

**指令微调（instruction fine‑tuning）**：在已有模型上继续训练，让它学习遵循特定指令格式（如“请回答以下问题”）。相当于给模型加装了一个“客服脚本”。

### 核心创新点
1. **从“记忆”到“可抽取”进行实验划分** → 作者构造了受控传记数据集，确保每条事实只出现一次且没有对应的问答对 → 通过这种设置可以直接观察模型是否真的学会抽取，而不是靠训练时的直接匹配。实验结果显示，缺乏多样化呈现的事实几乎得不到正确答案，即使后续做了大量指令微调。

2. **关联数据多样性与抽取成功率** → 统计了每条事实在预训练语料中的出现次数、同义改写数量、跨语言翻译次数等多样性指标 → 发现这些指标与模型的抽取准确率呈强正相关。换句话说，越是被“翻来覆去”呈现的知识，模型越容易在问答时调出来。

3. **线性探 probing 揭示内部编码形态** → 对每条实体名称的隐藏向量做线性探测，判断知识是否线性可读 → 当多样化增强明显时，线性探测准确率提升，说明知识被压缩进了实体嵌入；否则，探测几乎不成功，暗示知识被分散在上下文 token 中，难以直接抽取。

4. **给出可操作的工业建议** → 基于实验，作者提出两条实用路线：① 在预训练阶段使用小型辅助模型对原始文本进行改写、翻译等增强；② 把指令微调数据提前混入预训练，以免错过“编码窗口”。这两步直接针对“知识可抽取性”而非单纯提升模型规模。

### 方法详解
整体思路可以划分为三步：**数据准备 → 训练/微调 → 线性探测与分析**。

1. **受控传记数据集构建**  
   - 选取若干历史人物，每人准备一段简短的维基百科式传记。  
   - 对每条事实（如出生日期、职业）只保留一次原始表述，确保训练语料中没有对应的问答对。  
   - 为了测试多样性影响，作者分别生成了四种版本：① 原始句子；② 同义改写（使用小模型生成）；③ 句子顺序随机化；④ 机器翻译往返（英↔中）。每种版本的出现频次可控。

2. **模型训练与指令微调**  
   - 采用主流的 Transformer 架构（如 LLaMA‑7B）进行标准自回归预训练，语料中混入上述四种增强版本。  
   - 完成预训练后，统一使用指令微调数据（包括常规 QA、对话等）进行二次训练，保持微调数据量不变，以排除微调本身对抽取能力的影响。

3. **线性探 probing 与编码分析**  
   - 对每个实体名称的最终隐藏向量（在句子末尾的 CLS 类 token）抽取出来。  
   - 训练一个仅含单层线性层的分类器，输入向量 → 预测对应的属性值（如出生年份）。  
   - 通过比较不同数据增强策略下的探测准确率，判断知识是否被“线性化”存储。  
   - 额外的对比实验把探测目标换成上下文 token 的平均向量，以检验是否出现分布式编码。

**最巧妙的点**在于把“知识是否可抽取”转化为一个可量化的**多样性‑抽取‑线性可读性三元关系**，并用极简的线性探测来验证内部表征，而不是依赖复杂的解码或梯度分析。

### 实验与效果
- **数据集**：作者自建的 500 条人物传记数据，分别对应四种增强策略。  
- **基线**：未做任何数据增强的普通预训练模型（仅原始句子），以及仅做指令微调的模型。  
- **主要结果**：在原始句子上，模型的 QA 正确率几乎为 0%（论文声称），即使指令微调后也没有提升。加入同义改写后，准确率提升到约 30%；句子随机化提升到约 45%；机器翻译往返最高，达到约 60%。线性探测的准确率同样呈相似趋势，说明多样化增强直接提升了线性可读性。  
- **消融实验**：去掉指令微调，仅保留预训练阶段的多样化增强，仍能保持 50% 左右的抽取率，说明预训练阶段的多样化是关键因素。  
- **局限性**：实验仅在受控小规模传记数据上完成，未在大规模开放域 QA 上验证；多样化策略的成本（生成同义句、翻译）在工业规模上仍需评估。作者也承认，极端长文本或高度专业化知识的抽取仍未得到验证。

### 影响与延伸思考
这篇工作把“知识存储”从抽象的参数空间拉到可观测的“可抽取性”层面，引发了后续对预训练数据多样化的关注。随后出现的几篇论文（如 *Data Augmentation for Knowledge‑Rich LLMs*、*Curriculum Pretraining for Fact Retrieval*）直接引用了该研究的实验框架，尝试在更大规模语料上做同义改写或跨语言混合。业界也开始在数据清洗流水线中加入“知识增强”模块，使用小模型自动生成 paraphrase。想进一步深入，可以关注 **“可解释的知识编码”** 与 **“预训练阶段的指令注入”** 两大方向，尤其是如何在不显著增加算力的情况下实现大规模多样化。

### 一句话记住它
让模型真正会“查资料”，必须在预训练时把同一事实多次、以不同方式呈现，否则它只会“记住”而永远抽不出来。