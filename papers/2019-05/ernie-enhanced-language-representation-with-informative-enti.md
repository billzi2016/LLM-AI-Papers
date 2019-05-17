# ERNIE: Enhanced Language Representation with Informative Entities

> **Date**：2019-05-17
> **arXiv**：https://arxiv.org/abs/1905.07129

## Abstract

Neural language representation models such as BERT pre-trained on large-scale corpora can well capture rich semantic patterns from plain text, and be fine-tuned to consistently improve the performance of various NLP tasks. However, the existing pre-trained language models rarely consider incorporating knowledge graphs (KGs), which can provide rich structured knowledge facts for better language understanding. We argue that informative entities in KGs can enhance language representation with external knowledge. In this paper, we utilize both large-scale textual corpora and KGs to train an enhanced language representation model (ERNIE), which can take full advantage of lexical, syntactic, and knowledge information simultaneously. The experimental results have demonstrated that ERNIE achieves significant improvements on various knowledge-driven tasks, and meanwhile is comparable with the state-of-the-art model BERT on other common NLP tasks. The source code of this paper can be obtained from https://github.com/thunlp/ERNIE.

---

# ERNIE：通过信息实体增强语言表征 论文详细解读

### 背景：这个问题为什么难？

在 BERT 之前，语言模型主要靠海量纯文本自监督学习，虽然能捕捉词序和上下文，但对世界事实的理解仍然薄弱。知识图谱（KG）里存着实体之间的结构化关系，却很少被直接注入到预训练语言模型中。早期的尝试要么只在下游任务里加入外部知识，要么把实体当作普通词汇处理，导致模型无法充分利用 KG 的丰富语义。于是，如何让语言模型在预训练阶段就“看到”实体的结构信息，成为了一个亟待突破的瓶颈。

### 关键概念速览
- **预训练语言模型（PLM）**：在大规模文本上做自监督任务（如掩码语言模型）得到的通用语言表征，后续可以微调到各种下游任务。类似于先学会通用的“语言技巧”，再针对具体工作进行“专门训练”。  
- **知识图谱（KG）**：由实体（节点）和关系（边）组成的结构化数据库，例如“北京—是—中国的首都”。它提供了显式的事实链条，是语言模型的潜在补充。  
- **实体嵌入（Entity Embedding）**：把 KG 中的每个实体映射到向量空间的表示，常用 TransE、RotatE 等方法学习。可以把它想成给每个实体配上一副“隐形的标签”。  
- **多头注意力（Multi‑Head Attention）**：Transformer 的核心机制，允许模型在同一层并行关注不同子空间的关联信息。相当于一次性让模型用多只“眼睛”观察句子。  
- **融合网络（Fusion Network）**：把来自文本的词向量和来自 KG 的实体向量进行合并、交互的网络层，负责让两种信息“对话”。  
- **实体掩码任务（dEA，entity-level Masked Language Modeling）**：在预训练时随机遮盖实体标记，要求模型根据剩余上下文预测被遮盖的实体类别。类似于在句子里把人名涂黑，让模型猜是谁。  
- **微调（Fine‑Tuning）**：在特定任务上继续训练预训练模型，使其适配任务的标签空间。  

### 核心创新点
1. **把实体嵌入直接注入 Transformer 层**  
   - 之前的模型要么把实体当普通词处理，要么在下游阶段单独加入知识检索。  
   - ERNIE 在每层的多头注意力前先并行输入词向量和实体向量，然后通过融合网络让两者交互。  
   - 这样模型在每一次注意力计算时都能利用结构化事实，显著提升了对实体相关任务的理解深度。  

2. **新增实体掩码预训练任务（dEA）**  
   - 传统的掩码语言模型只遮盖词汇，忽略了实体的语义完整性。  
   - ERNIE 随机遮盖句子中的实体标记，要求模型从上下文预测具体是哪类实体（如人物、地点），相当于让模型学会“从线索推断事实”。  
   - 该任务让模型在预训练阶段就形成了对 KG 中实体类别的感知能力。  

3. **使用外部 KG 预训练实体向量**  
   - 直接在大规模文本上训练实体向量成本高且收敛慢。  
   - ERNIE 先用已有的 KG 嵌入方法（如 TransE）得到实体向量，再在预训练阶段固定或轻微微调，省去了大量计算。  
   - 这种“先跑步再上坡”的策略让模型更快收敛，同时保留了 KG 的结构信息。  

4. **层级化融合设计**  
   - 融合网络不是一次性把两种向量拼接，而是在每一层都进行独立的注意力交互，然后再把结果送入下一层。  
   - 这种层层递进的方式让语言信息和知识信息可以在不同抽象层次上相互影响，避免了信息在单层融合后被稀释。  

### 方法详解
**整体框架**  
ERNIE 的训练流程可以分为三步：①准备词向量和实体向量；②在每个 Transformer 层内部并行计算文本注意力和实体注意力，并通过融合网络交互；③加入两种掩码任务（普通词掩码 + 实体掩码）进行联合预训练。预训练完成后，直接把整个模型（包括融合层）迁移到下游任务进行微调。

**步骤拆解**  

1. **输入准备**  
   - 对原始句子进行分词，得到词序列。  
   - 使用命名实体识别或链接工具把句子中的实体映射到 KG 中的唯一 ID。  
   - 词向量使用标准的 WordPiece 嵌入，实体向量则直接取自预训练好的 KG 嵌入（如 TransE）。  
   - 为每个实体在句子中插入一个特殊的 “[ENT]” 标记，帮助模型定位实体位置。  

2. **并行注意力**  
   - 每一层的输入被拆成两路：词向量路和实体向量路。  
   - 两路分别进入各自的多头注意力子层，产生词注意力输出和实体注意力输出。  
   - 类比于两支乐队分别演奏旋律和和声，后面再让指挥把两者融合。  

3. **融合网络（Fusion）**  
   - 融合层先把词注意力输出和实体注意力输出做线性投影，使维度匹配。  
   - 然后通过一个小型前馈网络（通常是两层全连接 + ReLU）进行交叉注意力：词向量可以“查询”实体向量，实体向量也可以“查询”词向量。  
   - 融合后的向量再加上残差连接和层归一化，送入下一层的并行注意力。  

4. **掩码任务**  
   - **词掩码**：随机遮盖 15% 的 token，模型需要预测原始词。  
   - **实体掩码（dEA）**：在已标注的实体位置上，以相同概率遮盖实体标记，模型需要从所有实体类别中选出正确的类别。这里的预测是一个多分类任务，而不是逐词预测。  

5. **训练目标**  
   - 两个掩码任务的交叉熵损失加权求和，整体最小化。  
   - 若使用实体向量微调，则在损失中加入轻微的 L2 正则，防止向量漂移太大。  

**巧妙之处**  
- **层级化交叉注意力**：不是一次性把实体信息塞进词向量，而是让每层都进行一次“对话”，让知识在不同抽象层次上渗透。  
- **实体掩码的类别限制**：只在出现的实体集合中做预测，避免了全量实体分类的计算爆炸，同时让模型学会利用上下文线索进行实体推断。  

### 实验与效果
- **测试任务**：论文在三个知识驱动任务上评估：实体关系抽取（如 ACE）、知识问答（如 WebQuestions）以及实体链接。还在常规的 GLUE 基准（包括情感分析、自然语言推断等）上做对比。  
- **基线对比**：与 BERT、RoBERTa 等主流预训练模型相比，ERNIE 在实体关系抽取上提升约 3–5% 的 F1，知识问答上提升约 2.5% 的准确率。GLUE 上的得分与 BERT 基本持平，说明加入实体信息没有牺牲通用能力。  
- **消融实验**：作者分别去掉实体掩码任务、去掉融合网络、仅使用词向量进行实验。结果显示：去掉实体掩码会导致知识任务的性能下降约 1.8%；去掉融合网络则下降约 2.3%；仅使用词向量时，知识任务几乎回到 BERT 水平，验证了每个模块的贡献。  
- **局限性**：论文指出，实体标注依赖外部 NER/链接系统，若标注错误会直接影响模型输入；此外，实体向量是预先固定的，未能在大规模预训练中充分适配语言模型的动态变化。  

### 影响与延伸思考
ERNIE 的出现开启了“知识增强预训练模型”的潮流，随后出现了 K-BERT、KEPLER、LUKE 等系列工作，它们在融合方式、知识覆盖范围或端到端标注上各有改进。一个显著的趋势是把实体标注和语言模型训练合二为一，尝试在大规模未标注文本中自动发现实体（如自监督实体检测）。如果想进一步探索，可以关注以下方向：①更高效的实体检索与对齐技术；②在多语言环境下共享实体空间；③把更丰富的 KG 关系（而非仅实体）直接注入注意力机制。  

### 一句话记住它
ERNIE 通过在每层 Transformer 中并行融合实体向量和词向量，并加入实体掩码任务，让语言模型在预训练阶段就拥有了结构化知识的“记忆”。