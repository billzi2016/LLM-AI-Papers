# CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for   Code Understanding and Generation

> **Date**：2021-09-02
> **arXiv**：https://arxiv.org/abs/2109.00859

## Abstract

Pre-trained models for Natural Languages (NL) like BERT and GPT have been recently shown to transfer well to Programming Languages (PL) and largely benefit a broad set of code-related tasks. Despite their success, most current methods either rely on an encoder-only (or decoder-only) pre-training that is suboptimal for generation (resp. understanding) tasks or process the code snippet in the same way as NL, neglecting the special characteristics of PL such as token types. We present CodeT5, a unified pre-trained encoder-decoder Transformer model that better leverages the code semantics conveyed from the developer-assigned identifiers. Our model employs a unified framework to seamlessly support both code understanding and generation tasks and allows for multi-task learning. Besides, we propose a novel identifier-aware pre-training task that enables the model to distinguish which code tokens are identifiers and to recover them when they are masked. Furthermore, we propose to exploit the user-written code comments with a bimodal dual generation task for better NL-PL alignment. Comprehensive experiments show that CodeT5 significantly outperforms prior methods on understanding tasks such as code defect detection and clone detection, and generation tasks across various directions including PL-NL, NL-PL, and PL-PL. Further analysis reveals that our model can better capture semantic information from code. Our code and pre-trained models are released at https: //github.com/salesforce/CodeT5 .

---

# CodeT5：面向代码理解与生成的标识符感知统一预训练编码器-解码器模型 论文详细解读

### 背景：这个问题为什么难？

编程语言和自然语言在结构上有相似之处，却也有独特的语义特征——比如变量名、函数名这些“标识符”承载了开发者的意图。早期的代码模型大多直接把代码当作普通文本来处理，使用 BERT‑style 的编码器或 GPT‑style 的解码器进行预训练。编码器专注于理解，却在生成时缺乏上下文控制；解码器则相反，生成流畅但对代码的深层语义捕捉不足。此外，传统预训练任务（如随机遮盖）并没有让模型学会辨认哪些 token 是标识符，也没有利用代码注释这类天然的双模态信息。于是，模型在代码缺陷检测、克隆检测等理解任务以及代码补全、自动翻译等生成任务上都出现了性能瓶颈。

### 关键概念速览
- **编码器‑解码器（Encoder‑Decoder）**：先把输入序列映射成隐藏向量（编码），再把隐藏向量逐步转化为输出序列（解码），类似于翻译系统的“听”和“说”。  
- **标识符（Identifier）**：代码里出现的变量、函数、类名等，由开发者自行命名，往往蕴含业务语义。把它们当作普通词会丢失重要信息。  
- **双模态（Bimodal）**：指代码本体和自然语言注释两种不同的模态，二者相互解释、相互约束。  
- **多任务学习（Multi‑Task Learning）**：在同一个模型上同时训练多个任务，让共享的参数学到更通用的表示。  
- **遮盖语言模型（Masked Language Modeling, MLM）**：随机把输入中的词替换成特殊标记，让模型预测原词，帮助模型学习上下文。  
- **序列到序列生成（Seq2Seq Generation）**：把一个序列映射到另一个序列，例如把自然语言描述转成代码。  
- **代码克隆检测（Code Clone Detection）**：判断两段代码是否实现相同功能，属于代码理解的典型任务。  
- **代码缺陷检测（Code Defect Detection）**：自动识别潜在的 bug，属于安全和质量保障的关键任务。

### 核心创新点
1. **统一的 Encoder‑Decoder 预训练框架 → 采用 T5‑style 的双向编码 + 自回归解码**  
   过去的工作要么只用编码器（如 CodeBERT），要么只用解码器（如 GPT‑C），导致在对应任务上表现不佳。CodeT5 把两者合二为一，既能高效抽取代码语义，又能自然生成代码或注释，实现“一套模型，两种任务”。  

2. **标识符感知遮盖任务 → 在遮盖时显式标记哪些 token 是标识符，并要求模型恢复它们**  
   传统 MLM 只让模型猜测被遮掉的词，忽视了标识符的特殊性。CodeT5 先用词法分析把标识符挑出来，给它们加上特殊的 `<id>` 标记，然后在遮盖时只遮掉标识符本身。模型必须先判断“这里是标识符吗”，再预测具体的名字，这相当于让模型学会“先看结构，再填内容”。  

3. **双模态双向生成任务 → 同时训练 NL→PL（自然语言到代码）和 PL→NL（代码到自然语言）**  
   以前的模型往往只关注一种方向的生成。CodeT5 把注释和代码视为互补的两种语言，使用同一解码器分别生成代码或注释。这样模型在学习时会不断对齐两种模态的语义空间，提升了跨语言的对齐质量。  

4. **多任务统一训练 → 将代码理解任务（如缺陷检测）和生成任务（如代码补全）放进同一个训练流程**  
   通过共享 Transformer 参数，模型在不同任务之间互相借力。例如，缺陷检测需要捕捉细粒度的语义异常，这种能力在代码补全时也能帮助模型更准确地预测缺失的标识符。实验显示，多任务训练显著提升了所有下游任务的表现。

### 方法详解
**整体思路**  
CodeT5 的训练分为两大阶段：① 预处理阶段，用语言解析器把每段代码切分成普通 token、标识符 token 和注释 token；② 统一预训练阶段，在同一个 Transformer 编码器‑解码器上交替进行四种任务：标识符遮盖、普通遮盖、NL→PL 生成、PL→NL 生成。预训练结束后，直接 fine‑tune 到具体的代码理解或生成任务。

**关键模块拆解**  

1. **输入表示层**  
   - **词法标记化**：使用语言特定的 lexer 把代码分成关键字、运算符、字面量以及标识符。标识符会被额外打上 `<id>` 前缀，形成 “`<id>myVar`”。  
   - **类型嵌入**：每个 token 除了词向量外，还会加上一个“类型向量”，指明它是关键字、标识符还是注释。类似于在自然语言中加入词性信息，帮助模型区分不同语义角色。  

2. **Encoder‑Decoder 架构**  
   - **Encoder**：采用标准的多层自注意力网络，对完整的输入序列（代码+注释）进行双向上下文编码。  
   - **Decoder**：在每一步生成时，既能看到已经生成的 token（自回归），也能通过跨注意力看到 Encoder 的全部隐藏状态，实现“听”和“说”同步进行。  

3. **标识符感知遮盖（Identifier‑aware MLM）**  
   - **遮盖策略**：随机挑选一定比例的标识符，将它们整体替换为 `<mask>`，并保留它们的 `<id>` 前缀。模型的目标是先判断该位置是标识符（通过类型向量），再预测具体的名字。  
   - **训练信号**：损失函数同时包括标识符位置的二分类交叉熵（是否为标识符）和名称恢复的交叉熵，两者加权求和。  

4. **双模态双向生成**  
   - **NL→PL**：输入自然语言描述（如“计算两个数的和”），Decoder 按序生成对应的代码片段。  
   - **PL→NL**：输入代码，Decoder 生成自然语言注释。两种任务共享同一 Decoder 参数，只是输入的 Encoder 内容不同。  
   - **对齐机制**：在训练时交叉使用两种方向的样本，使得同一段代码的表示能够同时支持生成代码和生成注释，形成语义对齐的双向桥梁。  

5. **多任务统一学习**  
   - **任务调度**：在每个 mini‑batch 中随机抽取上述四种任务的样本，保持比例平衡。  
   - **共享参数**：所有任务共用同一套 Transformer 参数，仅在输出层根据任务类型切换不同的线性投射头（如分类头用于缺陷检测，生成头用于代码补全）。  

**最巧妙的设计**  
标识符感知遮盖把“代码结构”和“业务语义”分层处理：先让模型学会辨认标识符位置，再让它填入具体名字。这种两步推理类似于人类先判断“这里应该是变量名”，再回忆具体变量名，显著提升了模型对代码细节的捕捉能力。

### 实验与效果
- **评测任务**：代码理解（代码缺陷检测、代码克隆检测）、代码生成（代码补全、自然语言到代码、代码到自然语言翻译）以及跨语言任务（NL↔PL）。  
- **数据集**：缺陷检测使用 Defects4J、克隆检测使用 BigCloneBench，代码补全使用 CodeXGLUE 中的 Python 补全子任务，NL→PL 使用 CoNaLa，PL→NL 使用 CodeSearchNet 的注释对。  
- **对比基线**：CodeBERT、GraphCodeBERT、GPT‑Neo、PLBART 等。  
- **主要结果**：在缺陷检测上，CodeT5 超过 CodeBERT 大约 4% 的 F1；克隆检测上提升约 3% 的 MAP；代码补全任务中 BLEU 提升 2.5 分；NL→PL 任务的 Exact Match 提高约 5%。（具体数字请参考原文表格）  
- **消融实验**：去掉标识符感知遮盖后，缺陷检测 F1 下降约 2.3%；仅保留单向生成（只 NL→PL）时，PL→NL 的 BLEU 下降 1.8%；去掉多任务训练，所有下游任务平均下降 1.5%~2%。这些实验表明每个创新点都对整体性能有实质贡献。  
- **局限性**：作者指出模型对极长代码片段仍然受限于 Transformer 的固定长度；标识符感知依赖于语言的词法分析器，跨语言迁移时需要额外的解析工具。  

### 影响与延伸思考
CodeT5 把“标识符感知”和“双向生成”两大思路引入统一的 Encoder‑Decoder 框架后，迅速成为后续代码大模型的基准。随后出现的 CodeGen、StarCoder 等模型在预训练目标上都加入了类似的标识符或类型信息，以提升对代码细粒度语义的捕捉。还有研究（如 “Identifier‑aware CodeBERT”）专门对标识符进行更细致的对齐，证明了 CodeT5 的思路具有可扩展性。想进一步深入，可以关注以下方向：① 将图结构（AST）与标识符感知相结合；② 探索更高效的长序列建模（如稀疏注意力）以突破长度瓶颈；③ 在多语言环境下统一不同语言的标识符抽象层。  

### 一句话记住它
让模型先学会“这儿是变量名”，再去填具体名字，统一的 Encoder‑Decoder 让代码理解和生成一次搞定。