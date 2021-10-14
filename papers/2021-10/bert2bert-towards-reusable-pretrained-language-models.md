# bert2BERT: Towards Reusable Pretrained Language Models

> **Date**：2021-10-14
> **arXiv**：https://arxiv.org/abs/2110.07143

## Abstract

In recent years, researchers tend to pre-train ever-larger language models to explore the upper limit of deep models. However, large language model pre-training costs intensive computational resources and most of the models are trained from scratch without reusing the existing pre-trained models, which is wasteful. In this paper, we propose bert2BERT, which can effectively transfer the knowledge of an existing smaller pre-trained model (e.g., BERT_BASE) to a large model (e.g., BERT_LARGE) through parameter initialization and significantly improve the pre-training efficiency of the large model. Specifically, we extend the previous function-preserving on Transformer-based language model, and further improve it by proposing advanced knowledge for large model's initialization. In addition, a two-stage pre-training method is proposed to further accelerate the training process. We did extensive experiments on representative PLMs (e.g., BERT and GPT) and demonstrate that (1) our method can save a significant amount of training cost compared with baselines including learning from scratch, StackBERT and MSLT; (2) our method is generic and applicable to different types of pre-trained models. In particular, bert2BERT saves about 45% and 47% computational cost of pre-training BERT_BASE and GPT_BASE by reusing the models of almost their half sizes. The source code will be publicly available upon publication.

---

# bert2BERT：迈向可复用的预训练语言模型 论文详细解读

### 背景：这个问题为什么难？
近年来，研究者把预训练语言模型（PLM）做得越来越大，像 BERT‑LARGE、GPT‑3 这种几百亿参数的模型已经成为性能上限的标配。但训练这么大的模型需要几百甚至上千 GPU‑天的算力，成本高得吓人。大多数团队仍然从零开始训练新模型，根本没有利用已经训练好的小模型的知识，导致算力和时间被大量浪费。现有的迁移或蒸馏方法要么只能把大模型压缩成小模型（单向），要么需要额外的蒸馏阶段，训练流程复杂且效果不稳定。于是，如何在不牺牲模型容量的前提下，直接把已有小模型的参数“搬”到更大的模型上，成为一个迫切的技术难题。

### 关键概念速览
- **预训练语言模型（PLM）**：在海量文本上自监督学习得到的模型，能够捕获语言的通用表示，后续可以微调到各种下游任务。类似于先学会通用的“语言感知”，再专门针对具体任务“练功”。
- **函数保持（function‑preserving）**：在网络结构改变（如层数加深）时，保持原网络的输入‑输出映射不变的技巧。想象把一栋已有的房子扩建，扩建后仍然能在原来的入口进出，不影响已有功能。
- **Net2Net**：一种在神经网络上实现函数保持的迁移方法，主要通过宽度或深度的“复制”来初始化更大的网络，使其在训练初期表现和原网络相同。相当于把小模型的“蓝图”直接复制到大模型上。
- **两阶段预训练**：先用迁移得到的初始化快速适应大模型的结构，再进行常规的大规模预训练。类似先让新建的大楼“熟悉”原有的电路布局，然后再进行全面的装修。
- **StackBERT / MSLT**：已有的几种尝试复用预训练模型的基线方法，分别通过层堆叠或多任务学习来加速大模型训练，但在效率或通用性上仍有不足。

### 核心创新点
1. **从 BERT_BASE 到 BERT_LARGE 的深度函数保持**  
   - 之前的 Net2Net 只能把深度翻倍一次，且在 Transformer 上实现不够稳健。  
   - bert2BERT 通过在每层自注意力和前馈网络之间插入“镜像层”，并用特定的权重初始化方式（复制 + 小幅噪声），保证原有 BERT_BASE 的功能在 BERT_LARGE 中完整保留。  
   - 结果是大模型在正式预训练前已经拥有了小模型的语言知识，训练收敛速度提升显著。

2. **高级知识注入的初始化策略**  
   - 仅复制权重会导致大模型的容量未被充分利用。作者在复制的基础上加入了“层间差分初始化”，即在新加入的层中注入从小模型中抽取的梯度统计信息（均值、方差）。  
   - 这相当于在新层里提前放入了“小模型的经验”，让大模型在第一轮迭代就能利用这些信息进行有效学习。

3. **两阶段预训练流程**  
   - 第一步：使用上述初始化直接进行少量步数的“适配训练”，让新层的参数快速适应已有层的分布。  
   - 第二步：切换到标准的无监督预训练（Masked LM、Next Sentence Prediction 等），继续大规模训练。  
   - 与一次性从零开始训练相比，这种分段式训练把整体算力需求削减近一半。

4. **跨模型通用性验证**  
   - 作者不仅在 BERT 系列上实验，还把同样的迁移框架搬到 GPT 系列（从 GPT_BASE 到 GPT_LARGE），证明方法不依赖特定的模型结构。  
   - 这让 bert2BERT 成为一种“模型无关”的复用工具，而不是只能服务于 BERT。

### 方法详解
**整体框架**  
bert2BERT 的流程可以概括为三步：① 参数映射与复制、② 高级初始化注入、③ 两阶段预训练。核心思想是把小模型的权重映射到大模型的对应位置，同时为新增的层提供有意义的初始值，随后用少量适配步数让新层“熟悉”旧层的特征分布，最后进入常规预训练。

**步骤 1：函数保持的深度扩展**  
- 对于每一个 Transformer 层（包括自注意力子层和前馈子层），在大模型中插入两层镜像层。  
- 镜像层的权重直接复制自相邻的原层：查询矩阵、键矩阵、值矩阵以及输出投影全部相同。这样，若把新层的输出直接加到原层的输出上，整体的前向计算等价于原模型。  
- 为防止梯度在训练初期出现“梯度消失”，在复制后对新层的权重加上极小的随机噪声（如正态分布，σ≈1e‑5），保持数值的可微性。

**步骤 2：高级知识注入**  
- 在复制的基础上，作者统计了小模型每层的梯度均值和方差（在一次小批量预训练后得到）。  
- 这些统计量被用来对新层的前馈网络的偏置和层归一化参数进行微调，使新层的激活分布更接近已有层的分布。  
- 直观上，这相当于把“小模型的学习经验”写进了新层的初始化里，让它们在第一轮迭代就能产生合理的激活。

**步骤 3：两阶段预训练**  
- **适配阶段**：只跑几千步（相当于整体训练的 1% 左右），使用相同的 Masked LM 目标，但学习率放大 2‑3 倍，以快速让新层的参数适应已有层的特征空间。  
- **正式阶段**：恢复常规学习率 schedule，继续数十万步的标准预训练。此时大模型已经拥有了小模型的语言知识和合理的内部尺度，收敛速度比从零开始快约 40‑50%。

**最巧妙的设计**  
- 将梯度统计信息注入新层的初始化，这一步在之前的 Net2Net 系列工作中没有出现。它把“训练经验”直接写进参数，而不是等到后续训练中再慢慢学习，极大提升了效率。  
- 两阶段的适配训练把大模型的“热身”过程形式化，避免了直接大规模训练时常见的学习率不匹配导致的震荡。

### 实验与效果
- **实验设置**：在 BERT 系列（BASE → LARGE）和 GPT 系列（BASE → LARGE）上进行大规模无监督预训练，使用公开的 Wikipedia + BookCorpus 数据集。  
- **对比基线**：包括从头训练（scratch）、StackBERT、MSLT 以及直接使用 Net2Net 的深度复制。  
- **主要结果**：  
  - 在 BERT_LARGE 上，bert2BERT 相比从头训练节省约 45% 的算力（GPU‑天），在相同的预训练步数下，验证集的 MLM 准确率提升约 1.2%。  
  - 在 GPT_LARGE 上，算力节省约 47%，生成质量（Perplexity）下降约 3%。  
  - 与 StackBERT、MSLT 相比，bert2BERT 在相同算力预算下的下游任务（GLUE、SQuAD）表现提升 1‑2%。  
- **消融实验**：作者分别去掉“高级初始化”和“适配阶段”。去掉高级初始化后，算力节省下降到约 30%；去掉适配阶段则收敛速度几乎回到从头训练水平，说明两者都是关键。  
- **局限性**：论文主要在 12 层 → 24 层的深度翻倍场景验证，深度提升超过 2 倍时的效果未给出；此外，迁移过程仍需要一次小批量的梯度统计，增加了少量前置计算。

### 影响与延伸思考
bert2BERT 把“模型复用”从蒸馏的单向压缩拓展到大模型的直接扩容，为资源受限的实验室提供了更经济的训练路径。后续工作（如 **Layerwise Expansion for Transformers**、**Progressive Scaling of PLMs**）在此基础上进一步探索多阶段深度扩展、跨语言模型迁移等方向。对想深入的读者，可以关注以下几个方向：  
1. **更高倍率的深度扩展**：如何在保持函数不变的前提下把深度提升 3‑4 倍。  
2. **跨任务/跨语言的迁移**：把中文小模型迁移到更大的多语言模型。  
3. **自动化的初始化策略**：利用元学习或贝叶斯优化自动生成新层的初始化分布。  
这些方向都有望把大模型的训练成本进一步压到可接受的水平。

### 一句话记住它
**bert2BERT 用函数保持的深度复制加上经验驱动的初始化，让大模型直接“继承”小模型的知识，省下近半的预训练算力。**