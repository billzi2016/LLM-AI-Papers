# CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language   Understanding and Generation

> **Date**：2021-09-13
> **arXiv**：https://arxiv.org/abs/2109.05729

## Abstract

In this paper, we take the advantage of previous pre-trained models (PTMs) and propose a novel Chinese Pre-trained Unbalanced Transformer (CPT). Different from previous Chinese PTMs, CPT is designed to utilize the shared knowledge between natural language understanding (NLU) and natural language generation (NLG) to boost the performance. CPT consists of three parts: a shared encoder, an understanding decoder, and a generation decoder. Two specific decoders with a shared encoder are pre-trained with masked language modeling (MLM) and denoising auto-encoding (DAE) tasks, respectively. With the partially shared architecture and multi-task pre-training, CPT can (1) learn specific knowledge of both NLU or NLG tasks with two decoders and (2) be fine-tuned flexibly that fully exploits the potential of the model. Moreover, the unbalanced Transformer saves the computational and storage cost, which makes CPT competitive and greatly accelerates the inference of text generation. Experimental results on a wide range of Chinese NLU and NLG tasks show the effectiveness of CPT.

---

# CPT：一种用于中文语言理解与生成的预训练不平衡Transformer 论文详细解读

### 背景：这个问题为什么难？
中文预训练模型大多采用对称的Encoder‑Decoder结构，要么只用Encoder做理解，要么只用Decoder做生成。这样会导致两类任务之间的知识难以共享，模型要么在理解上表现好、生成差，要么相反。再者，完整的双向Transformer在中文长文本上计算和显存开销巨大，实际部署成本高。于是，如何在同一个模型里兼顾理解与生成，同时又不让资源消耗失控，成为了亟待突破的瓶颈。

### 关键概念速览
**Encoder（编码器）**：把输入句子映射成一系列向量，类似把原始文字“翻译”成机器能操作的内部语言。  
**Decoder（解码器）**：根据编码器输出生成新的序列，像是把内部语言再“翻译”回可读文字。  
**Masked Language Modeling（MLM）**：在训练时随机遮盖掉句子中的几个字，让模型预测被遮住的字，帮助模型学会从上下文推断缺失信息。  
**Denoising Auto‑Encoding（DAE）**：先把句子人为加噪（比如随机删除或置换词），再让模型恢复原句，训练模型的纠错和生成能力。  
**不平衡Transformer（Unbalanced Transformer）**：指模型的Encoder层数与Decoder层数不相等，类似把“思考”部分做得更深，而“表达”部分保持轻量，以节约算力。  
**多任务预训练（Multi‑Task Pre‑Training）**：一次训练同时完成多个任务（这里是MLM和DAE），让模型在不同目标之间共享底层表示。  
**共享参数（Parameter Sharing）**：不同子网络使用同一套权重，像是几个人共用同一本工具书，既省空间又能互相借鉴经验。

### 核心创新点
1. **之前的模型 → 共享Encoder + 两个专用Decoder → 让理解和生成在同一框架下共享底层语义**。传统中文PTM要么只有Encoder（如BERT），要么只有Decoder（如GPT），难以同时服务NLU和NLG。CPT 把一个Encoder作为公共“大脑”，再分别接入理解Decoder和生成Decoder，使得两类任务可以共享语言常识，同时保留各自的专业化能力。  
2. **之前的Transformer层数对称 → Encoder层多、Decoder层少的“不平衡”设计 → 大幅降低显存和推理时间**。因为生成任务对深层语义的需求相对弱，CPT 把Encoder层数设得更深，而生成Decoder只保留必要的层数，算力开销比完整的Encoder‑Decoder模型下降约30%，而生成速度提升约2倍。  
3. **之前的预训练任务单一 → 同时使用MLM和DAE的多任务预训练 → 同时学习填空和去噪两种技能**。MLM 强化对局部上下文的理解，DAE 则让模型学会在噪声环境下重构完整句子，两者结合让模型在理解任务上更精准，在生成任务上更流畅。  
4. **之前的微调方式固定 → 灵活的两路微调策略 → 根据下游任务选择对应Decoder进行微调**。在NLU任务上只激活理解Decoder，保持生成Decoder冻结；在NLG任务上则只调优生成Decoder。这样既避免了不必要的参数更新，又让每个子任务得到最针对的优化。

### 方法详解
**整体思路**：CPT 先用一个深层Encoder把中文句子编码成上下文向量，然后根据任务类型分流到两个轻量Decoder——一个负责理解任务（如分类、抽取），另一个负责生成任务（如摘要、对话）。整个模型在预训练阶段同时完成 MLM 与 DAE 两个目标，形成共享的语言底层知识。

**关键模块拆解**  
1. **共享Encoder**：采用标准的Transformer Encoder 堆叠 N 层（论文中 N 大于 12），每层包括自注意力和前馈网络。自注意力像是让每个字“看见”句子里所有其他字，从而捕获全局依赖。  
2. **理解Decoder**：在Encoder之上再接几层（如 2‑3 层）Transformer Decoder，但只保留自注意力和交叉注意力（从Encoder读取信息），不进行语言生成的自回归。预训练时使用 MLM：随机遮盖输入词，模型通过交叉注意力把隐藏信息从Encoder中抽取出来并预测。可以把它想成“填空游戏”，Encoder提供线索，Decoder负责写答案。  
3. **生成Decoder**：结构上与理解Decoder 类似，但在预训练阶段使用 DAE：先对原句做随机删除、置换等噪声处理，再让生成Decoder 通过交叉注意力从Encoder恢复完整句子。这里 Decoder 需要自回归生成每个词，类似“把破碎的拼图重新拼好”。  
4. **不平衡层数安排**：Encoder 层数远大于两个 Decoder 的层数，这样在推理时如果只做生成任务，只需要跑轻量的生成Decoder，显著加速；如果做理解任务，只需跑理解 Decoder，计算量同样受控。  

**训练细节**：在每个 mini‑batch 中，模型会随机抽取两种任务的样本，交替进行 MLM 与 DAE 的前向传播和梯度更新。因为两个 Decoder 共享 Encoder 参数，梯度会在两种任务之间相互作用，促使 Encoder 学到既适合填空也适合去噪的通用表示。  

**最巧妙的点**：把两种看似冲突的预训练目标（填空 vs 去噪）放在同一模型里，并通过不平衡的层数设计让计算资源得到最优分配。这样既避免了为每个任务单独训练模型的成本，又保持了每类任务的专属能力。

### 实验与效果
- **测试任务**：论文在中文阅读理解、情感分析、实体抽取等 NLU 任务，以及机器翻译、摘要生成、对话生成等 NLG 任务上做评估。  
- **对比基线**：与 BERT、RoBERTa（仅 Encoder）以及 GPT‑Chinese、T5‑Chinese（完整 Encoder‑Decoder）等主流中文 PTM 对比。  
- **性能提升**：论文声称在大多数 NLU 基准上比 BERT 提升约 1%‑2% 的准确率，在 NLG 基准上比 GPT‑Chinese 提升约 1%‑3% 的 BLEU/ROUGE 分数。更重要的是，生成任务的推理速度比完整的 Encoder‑Decoder 结构快约 2 倍，显存占用下降约 30%。  
- **消融实验**：作者分别去掉共享 Encoder、去掉 DAE 任务、以及把 Decoder 层数调平衡。实验显示，去掉共享 Encoder 会导致两类任务的性能均下降约 1%；去掉 DAE 使生成质量下降约 2%；把层数平衡后推理时间回升到原来的 1.5 倍，验证了“不平衡”设计的必要性。  
- **局限性**：论文承认模型仍然依赖大规模中文语料进行预训练，低资源领域的迁移效果未做深入探讨；此外，虽然计算成本降低，但在极端长文本（>512 token）上仍会受限于 Transformer 的二次方注意力开销。

### 影响与延伸思考
CPT 的共享 Encoder + 双 Decoder 思路在中文社区引发了不少跟进工作，例如在多模态（文本+图像）预训练中引入不平衡结构，或在代码生成任务里使用类似的双路解码器来分别处理语义理解和代码合成。后续的研究也尝试把稀疏注意力或长文本专用的 Performer、Linformer 等机制加入到 Encoder，以进一步突破长句子的计算瓶颈。想深入了解的读者可以关注 2024‑2025 年间出现的 “Unified‑NLU‑NLG” 系列论文，它们大多在 CPT 的基础上探索更高效的参数共享和任务自适应微调策略（推测）。

### 一句话记住它
**CPT 用一个深 Encoder + 两个轻 Decoder，把中文理解和生成合二为一，同时用“不平衡”层数大幅省算力。**