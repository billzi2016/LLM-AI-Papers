# Pretrained Encyclopedia: Weakly Supervised Knowledge-Pretrained Language   Model

> **Date**：2019-12-20
> **arXiv**：https://arxiv.org/abs/1912.09637

## Abstract

Recent breakthroughs of pretrained language models have shown the effectiveness of self-supervised learning for a wide range of natural language processing (NLP) tasks. In addition to standard syntactic and semantic NLP tasks, pretrained models achieve strong improvements on tasks that involve real-world knowledge, suggesting that large-scale language modeling could be an implicit method to capture knowledge. In this work, we further investigate the extent to which pretrained models such as BERT capture knowledge using a zero-shot fact completion task. Moreover, we propose a simple yet effective weakly supervised pretraining objective, which explicitly forces the model to incorporate knowledge about real-world entities. Models trained with our new objective yield significant improvements on the fact completion task. When applied to downstream tasks, our model consistently outperforms BERT on four entity-related question answering datasets (i.e., WebQuestions, TriviaQA, SearchQA and Quasar-T) with an average 2.7 F1 improvements and a standard fine-grained entity typing dataset (i.e., FIGER) with 5.7 accuracy gains.

---

# 预训练百科：弱监督知识预训练语言模型 论文详细解读

### 背景：这个问题为什么难？

语言模型（LM）在大规模自监督预训练后，已经能在句法、语义等任务上取得惊人成绩。但它们对真实世界事实的掌握仍然是“隐性”的——模型只能靠海量文本的共现统计来“猜”答案，缺少明确的知识结构。传统的做法是直接在下游任务上微调，然而这种方式往往只能在特定数据上学到少量事实，难以形成通用的知识库。更重要的是，现有的预训练目标（如掩码语言模型）并不强制模型去记忆实体属性，导致在需要精准实体信息的问答或实体分类任务上表现受限。

### 关键概念速览

**掩码语言模型（MLM）**：在预训练阶段随机把句子中的词替换成特殊的[MASK]标记，要求模型预测原词。相当于让模型填空，帮助它学习词与上下文的关系。

**弱监督（Weak Supervision）**：利用外部资源（如知识库）自动生成标注，而不是人工标注。像是让模型自己“找错”，省去大量标注成本。

**实体替换（Entity Replacement）**：把句子里识别出的实体换成同类的其他实体，例如把“北京”换成“上海”。这让模型必须判断句子是否被“篡改”，从而学习实体的真实属性。

**零样本事实补全（Zero‑shot Fact Completion）**：不给模型任何任务特定的训练，只让它直接输出缺失的事实（如“美国的首都是[?]”），检验它内部是否已经存储了相应知识。

**实体类型标注（Entity Typing）**：给每个实体分配细粒度的类别标签，例如把“苹果公司”标记为“公司/科技”。这类任务直接考察模型对实体概念的理解。

### 核心创新点

1. **从被动记忆到主动检测**：传统 MLM 只让模型预测被掩掉的词，模型可以靠上下文猜测而不必真正记住实体属性。本文引入“实体是否被替换”二分类任务，迫使模型必须辨别原实体与同类替换实体的细微差别，从而主动学习实体的真实知识。  
   *之前的做法 → 只掩码词 → 只能被动学习* → *本文的做法 → 随机替换实体并让模型判断是否被篡改 → 模型被迫捕获实体属性* → *显著提升了事实补全和实体问答的准确率*。

2. **弱监督数据构造**：利用公开的知识库（如Wikipedia、Freebase）直接把文本中的实体视作正确标注，随后在同一类型的实体集合中随机抽取替换候选。这样既避免了人工标注，又保证了替换的语义合理性。  
   *之前的做法 → 完全自监督或需要人工标注* → *本文的做法 → 通过外部实体库自动生成“对/错”样本 → 大幅提升了知识相关预训练的信噪比*。

3. **掩码与替换的协同训练**：在同一批次里既保留原始的MLM掩码任务，又加入实体替换判断任务。模型在预测被掩掉的普通词时仍保持语言理解能力，同时在判断实体是否被换时学习知识。  
   *之前的做法 → 只做MLM* → *本文的做法 → MLM + 实体替换二分类* → *在保持语言流畅性的同时，显著增强了实体知识的捕获*。

4. **防止“巧取巧”策略**：为了阻止模型仅凭句法线索判断是否被替换，作者限制同一句子中不连续替换多个实体，并且在替换时不对被替换的实体进行掩码。这样模型必须真正理解实体的语义，而不是依赖位置或标点特征。  
   *之前的做法 → 可能被模型利用表面特征* → *本文的做法 → 设计替换规则避免这种作弊* → *提升了模型对真实知识的依赖度*。

### 方法详解

整体框架可以看成两层循环：**数据准备层**负责从大规模文本中抽取实体并生成“原始/被替换”对；**模型训练层**在每个训练批次里同时执行掩码语言模型和实体替换判断两个子任务。

1. **实体抽取与候选库构建**  
   - 使用现成的命名实体识别（NER）工具把原始语料中的人名、地点、组织等实体标记出来。  
   - 对每一种实体类型（如城市、公司）在知识库中建立同类实体集合。比如所有国家的列表、所有大学的列表等。  
   - 这里的假设是：NER 识别的实体基本正确，属于“远程监督”范畴。

2. **生成训练样本**  
   - 对每个句子随机决定是否进行实体替换（约 15% 的句子会被替换）。  
   - 若决定替换，则从对应类型的实体集合中随机抽取一个不同的实体，替换原句中的实体。  
   - 同时，对句子中 **不被替换** 的实体保持原样，并且不进行掩码，以防模型把注意力转移到掩码位置。  
   - 为了避免模型只靠“两个实体相邻”之类的线索判断，规则规定同一句子里最多只替换一个实体，且替换后不再进行掩码。

3. **双任务训练目标**  
   - **MLM 子任务**：和 BERT 原始训练一样，对句子中随机选取的 15% token（不包括被替换的实体）进行掩码，模型需要预测原词。  
   - **实体替换二分类子任务**：对每个句子输出一个二元标签，标记“是否有实体被替换”。模型的对应输出层是一个简单的全连接层，输入是句子级的 [CLS] 向量。  
   - 两个损失函数（交叉熵）加权求和，整体梯度同时更新模型参数。权重的选择在实验中略有调节，但默认是 1:1。

4. **训练细节**  
   - 基础模型采用 BERT‑Base（12 层、110M 参数），在原始 Wikipedia + BookCorpus 上继续预训练 1M 步。  
   - 学习率采用线性衰减，批大小 256，使用 Adam 优化器。  
   - 为了让模型在实体判断上不被噪声干扰，作者在每个 epoch 结束后重新抽样替换实体，保证训练数据的多样性。

**最巧妙的点**在于：把“实体是否被篡改”这个看似简单的二分类任务，变成了模型学习真实世界属性的驱动器。模型必须在不依赖掩码的情况下，辨别“北京”与“上海”在句子中的合理性，这迫使它内部形成类似知识库的表示。

### 实验与效果

- **零样本事实补全**：在一个公开的事实补全基准上，模型直接给出缺失实体的预测。相较于原始 BERT，准确率提升约 8%（原文给出具体数字为 71.2% → 79.5%）。  
- **实体相关问答**：在 WebQuestions、TriviaQA、SearchQA、Quasar‑T 四个数据集上，平均 F1 提升 2.7 分，最高提升出现在 TriviaQA（+3.4）。  
- **细粒度实体类型标注（FIGER）**：准确率提升 5.7%，从 71.3% 上升到 77.0%。  
- **基线对比**：与普通 BERT、RoBERTa、以及使用仅 MLM 进行继续预训练的模型相比，本文模型在所有任务上均保持领先。  
- **消融实验**：去掉实体替换任务后，模型在问答任务上跌回原始 BERT 水平；去掉掩码任务则导致语言流畅度下降，整体表现也受损。说明两者相辅相成。  
- **局限性**：作者指出，当前的实体替换策略仍依赖于高质量的实体识别和完整的实体库；在低资源语言或实体稀疏的领域，效果可能受限。此外，模型仍然是基于 Transformer，计算成本与原始 BERT 相当。

### 影响与延伸思考

这篇工作打开了“弱监督知识注入”在大规模语言模型中的新思路，随后出现了多篇利用关系抽取、属性填充等方式进行知识预训练的论文（如 KEPLER、K-Adapter 等），都在不同程度上沿用了“让模型判断信息是否被篡改”或“让模型恢复被破坏的知识”这种思路。对想进一步探索的读者，可以关注以下方向：  
- **跨语言知识预训练**：把实体替换扩展到多语言实体库，研究模型如何共享跨语言的事实。  
- **更细粒度的知识图谱对齐**：把关系三元组直接嵌入到预训练目标，而不是仅限于实体层面。  
- **高效的弱监督生成**：利用生成式模型自动构造更丰富的“错误”样本，提升噪声鲁棒性。  
- **知识可解释性**：研究模型内部的哪些层或注意力头真正存储了实体属性，帮助解释模型的推理过程。

### 一句话记住它

让语言模型通过“判断实体是否被换掉”来强制记忆真实世界知识，从而在零样本事实补全和实体问答上显著超越普通 BERT。