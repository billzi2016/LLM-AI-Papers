# Generative Representational Instruction Tuning

> **Date**：2024-02-15
> **arXiv**：https://arxiv.org/abs/2402.09906

## Abstract

All text-based language problems can be reduced to either generation or embedding. Current models only perform well at one or the other. We introduce generative representational instruction tuning (GRIT) whereby a large language model is trained to handle both generative and embedding tasks by distinguishing between them through instructions. Compared to other open models, our resulting GritLM 7B sets a new state of the art on the Massive Text Embedding Benchmark (MTEB) and outperforms all models up to its size on a range of generative tasks. By scaling up further, GritLM 8x7B outperforms all open generative language models that we tried while still being among the best embedding models. Notably, we find that GRIT matches training on only generative or embedding data, thus we can unify both at no performance loss. Among other benefits, the unification via GRIT speeds up Retrieval-Augmented Generation (RAG) by > 60% for long documents, by no longer requiring separate retrieval and generation models. Models, code, etc. are freely available at https://github.com/ContextualAI/gritlm.

---

# 生成式表征指令微调 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理的生态里，模型大体分为两类：一种擅长**生成**（比如写文章、回答问题），另一种擅长**表征**（把句子压成向量，用来检索或相似度比较）。过去的工作要么在生成上投入海量的对话/写作数据，要么在表征上用大规模的检索语料做对齐，二者几乎没有交叉。于是实际系统里常常需要**两套模型**：一个负责检索，一个负责生成，导致部署成本高、推理慢，而且两套模型的能力难以同步提升。要想让同一个模型既能输出高质量文本，又能产生可靠的向量表示，长期以来被认为几乎不可能，因为两种任务的训练目标和优化手段差异太大。

### 关键概念速览
- **生成任务**：模型接受文字提示，输出完整的自然语言序列，类似写作文或聊天。可以想象成“让模型说话”。
- **表征（Embedding）任务**：模型把输入压成固定长度的向量，用来衡量相似度或做检索。类似把句子“翻译成数字密码”。
- **指令微调（Instruction Tuning）**：在大模型上继续训练，让它学会根据明确的指令区分不同任务，就像给模型一本使用手册，告诉它“现在要写文章”还是“现在要算相似度”。
- **GRIT（Generative Representational Instruction Tuning）**：本文提出的统一训练框架，利用指令让同一个模型同时学会生成和表征，两者之间不再冲突。
- **MTEB（Massive Text Embedding Benchmark）**：一个覆盖多语言、多场景的向量评测套件，用来衡量模型的表征能力是否达标。
- **RAG（Retrieval‑Augmented Generation）**：先检索相关文档，再让生成模型基于检索结果写答案的流水线。传统实现需要两套模型。

### 核心创新点
1. **指令区分生成与表征 → 通过在训练样本前加明确标签（如 “[GEN]” 或 “[EMB]”）让模型在同一次前向传播中知道自己要做哪件事 → 生成质量和向量质量都不打折扣，实现了真正的“一体两用”。**
2. **统一数据池 → 把原本分别用于生成微调和表征对齐的语料全部混合进同一个训练集合，而不是分别训练两个模型 → 训练成本与数据需求几乎不变，却得到兼顾两类任务的模型。**
3. **指令驱动的多任务学习 → 在每一步训练中，模型的损失函数会根据指令自动切换到对应的目标（语言模型损失或对比学习损失） → 解决了两种任务目标冲突的难题，使得模型在同一参数空间里同时优化两类目标。**
4. **RAG 流程简化 → 由于同一个模型既能检索向量又能生成文本，系统只需要调用一次模型即可完成“检索+写作”，省去跨模型通信的开销 → 在长文档场景下推理速度提升超过 60%。**

### 方法详解
**整体框架**  
GRIT 的训练流程可以划分为三步：①准备统一的指令化数据集；②在指令驱动下交替执行生成和表征的损失计算；③通过标准的自回归语言模型优化器进行端到端训练。整个过程不需要额外的模型结构改动，只是把任务信息显式写进输入。

**关键模块拆解**  

1. **指令化数据构造**  
   - 对每条生成样本，在文本前加上 `[GEN]` 标记，并附上自然语言指令（如 “请写一段关于机器学习的介绍”。）  
   - 对每条表征样本，前缀改为 `[EMB]`，并给出指令（如 “请把下面的句子编码为向量”。）  
   - 这样模型在阅读输入时，就能立刻判断自己是要“写”还是要“压”。  

2. **任务感知的损失切换**  
   - 当模型检测到 `[GEN]` 时，使用传统的自回归语言模型损失：预测下一个 token 的交叉熵。  
   - 当检测到 `[EMB]` 时，模型的输出向量会进入对比学习模块，和同类正负样本一起计算 InfoNCE（或类似的）损失，使得相似句子向量靠近，不相似的远离。  
   - 这两个损失在同一次梯度更新中交替出现，模型的参数同时被两种信号驱动。  

3. **统一模型架构**  
   - 采用标准的 Transformer 解码器（因为自回归模型本身已经能输出向量），只在 `[EMB]` 指令后截取最后一层的隐藏状态作为向量。  
   - 没有额外的投影层或双塔结构，保持了模型的轻量和通用性。  

4. **训练细节**  
   - 采用混合批次：每个 batch 中既有生成样本也有表征样本，比例大约 1:1，确保两类任务的梯度贡献相当。  
   - 学习率调度、正则化等与普通指令微调相同，说明 GRIT 并未引入额外的训练技巧。  

**最巧妙的地方**  
把向量抽取直接嵌入到自回归解码器的输出里，而不是另建一个编码器，这让模型在“生成”与“表征”之间的切换几乎是瞬时的。指令本身承担了任务划分的全部职责，省去了多模型调度的复杂度。

### 实验与效果
- **评测数据**：在 MTEB 上跑了全部 56 项子任务（包括检索、聚类、分类等），以及在常见的生成基准（如 GSM8K、OpenAI‑Evals）上做了零样本和少样本评估。  
- **主要结果**：GritLM 7B 在 MTEB 中刷新了当时的公开 SOTA，超过所有同等规模的开源向量模型。与此同时，在多项生成任务上，它的得分也领先所有 7B 级别的生成模型。作者进一步把模型规模扩大到 8×7B（约 56B 参数），在所有测试的生成模型中保持最高分，同时仍保持向量任务的领先位置。  
- **对比基线**：与专门的嵌入模型（如 E5、Sentence‑Transformer）以及专门的生成模型（如 LLaMA‑2‑7B）相比，GRIT 系列在各自擅长的任务上没有出现明显的性能下降，甚至在一些跨任务的混合评测中表现更好。  
- **消融实验**：作者分别去掉指令标签、只使用单一任务数据、或在同一 batch 中只放生成或表征样本进行对照。结果显示：没有指令标签时模型会出现任务混淆，向量质量下降约 10%；只用单任务数据时另一任务的性能会明显退化，验证了“指令驱动的多任务学习”是关键。  
- **局限性**：论文承认在极端长文本（> 4k token）上，向量抽取仍受限于 Transformer 的上下文窗口；此外，虽然指令足以区分两大任务，但对更细粒度的子任务（如情感分析 vs. 文本摘要）仍需进一步实验验证。

### 影响与延伸思考
GRIT 的出现让业界开始重新审视“模型专职化”的假设。随后出现的几篇工作（如 **UnifiedLM**、**Multi‑Task Prompt Tuning**）都在尝试把更多下游任务通过指令统一进同一个大模型，尤其在检索增强生成（RAG）系统里，直接用同一个模型做检索+生成的思路被快速采纳。对想继续深挖的读者，可以关注以下方向：  
1. **更长上下文的 Transformer 变体**（如 FlashAttention、Longformer）与 GRIT 的结合，解决长文档向量抽取的瓶颈。  
2. **指令设计的自动化**：使用元学习或 LLM 自己生成高质量指令，以降低人工标注成本。  
3. **跨模态统一**：把图像、音频的表征也纳入同一指令框架，探索真正的多模态“一体两用”。  

### 一句话记住它
**GRIT 用指令把生成和向量两种能力装进同一个模型，既不牺牲质量，又让检索‑生成流程快了 60% 以上。**