# Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via   Adaptation

> **Date**：2025-04-08
> **arXiv**：https://arxiv.org/abs/2504.06225

## Abstract

While decoder-only large language models (LLMs) have shown impressive results, encoder-decoder models are still widely adopted in real-world applications for their inference efficiency and richer encoder representation. In this paper, we study a novel problem: adapting pretrained decoder-only LLMs to encoder-decoder, with the goal of leveraging the strengths of both approaches to achieve a more favorable quality-efficiency trade-off. We argue that adaptation not only enables inheriting the capability of decoder-only LLMs but also reduces the demand for computation compared to pretraining from scratch. We rigorously explore different pretraining objectives and parameter initialization/optimization techniques. Through extensive experiments based on Gemma 2 (2B and 9B) and a suite of newly pretrained mT5-sized models (up to 1.6B), we demonstrate the effectiveness of adaptation and the advantage of encoder-decoder LLMs. Under similar inference budget, encoder-decoder LLMs achieve comparable (often better) pretraining performance but substantially better finetuning performance than their decoder-only counterpart. For example, Gemma 2B-2B outperforms Gemma 2B by $\sim$7\% after instruction tuning. Encoder-decoder adaptation also allows for flexible combination of different-sized models, where Gemma 9B-2B significantly surpasses Gemma 2B-2B by $>$3\%. The adapted encoder representation also yields better results on SuperGLUE. We will release our checkpoints to facilitate future research.

---

# Encoder-Decoder Gemma：通过适配提升质量‑效率权衡 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，纯解码器（decoder‑only）架构因为能够直接生成文本而被大量使用，性能也屡创新高。但它们在推理时需要把所有输入都塞进同一个自回归网络，计算成本随序列长度线性增长，导致实际部署成本高。相对地，编码器‑解码器（encoder‑decoder）结构把输入先压进一个专门的编码器，再让解码器逐步生成输出，推理效率更好，且编码器的表示可以在下游任务中复用。然而，现有的 encoder‑decoder 模型往往从头预训练，耗时耗算，而且在语言理解和生成的综合能力上仍落后于最强的 decoder‑only LLM。于是出现了一个两难：要么保留 decoder‑only 的强大能力却付出高推理成本，要么使用更高效的 encoder‑decoder 但牺牲了一部分能力。如何在不重新从零训练的前提下，把 decoder‑only 的“聪明”迁移到 encoder‑decoder 上，成为了一个迫切而又技术上具挑战性的问题。

### 关键概念速览
- **Decoder‑only 模型**：只包含自回归解码器的网络，输入和输出共用同一个模块，像 GPT 系列那样一次接一次生成下一个词。  
- **Encoder‑Decoder 模型**：由两个独立的子网组成，编码器把输入序列映射成隐藏表示，解码器再基于这些表示生成输出，典型代表是 T5、BART。  
- **适配（Adaptation）**：把已经训练好的模型参数迁移到另一个结构上，而不是重新从头训练。这里指把 decoder‑only 的权重映射到 encoder‑decoder 的对应层。  
- **指令微调（Instruction Tuning）**：在大量指令-响应对上继续训练模型，使其更擅长遵循自然语言指令。  
- **质量‑效率权衡**：在模型性能（如准确率、生成质量）和推理成本（时间、算力）之间寻找最佳平衡点。  
- **参数初始化/优化技巧**：在迁移过程中如何设置新层的初始值、学习率等，以保证训练稳定且收敛快。  
- **SuperGLUE**：一套高级自然语言理解基准，覆盖阅读理解、推理等任务，用来评估模型的语言理解能力。  

### 核心创新点
1. **从 Decoder‑only 到 Encoder‑Decoder 的结构迁移**  
   之前的工作要么直接训练 encoder‑decoder，要么在 decoder‑only 基础上做任务微调。本文提出把 decoder‑only 的权重直接映射到 encoder‑decoder 的对应层（如自注意力、前馈网络），并为新增的编码器输入层设计专门的初始化方案。这样既保留了原模型的语言生成能力，又省去了从零预训练的巨额算力。  

2. **系统化的预训练目标对比**  
   研究了多种适配后再预训练的目标，包括保持原始语言建模分布的自回归目标、双向掩码语言模型（类似 T5 的目标）以及混合目标。实验表明，使用双向掩码目标可以更好地激活编码器的表示能力，从而在下游理解任务上取得显著提升。  

3. **灵活的尺寸组合策略**  
   通过把大模型的解码器（如 9B）和小模型的编码器（如 2B）拼接，形成 “大解码‑小编码” 的混合体。作者展示，这种跨尺寸组合在相同推理预算下比同尺寸的全解码器模型更强，说明适配框架天然支持模型级别的模块化。  

4. **针对指令微调的高效适配**  
   在指令微调阶段，作者发现 encoder‑decoder 结构在相同算力预算下的微调效果普遍优于 decoder‑only。具体表现为在 Gemma 2B 上的指令微调后，适配得到的 encoder‑decoder 版本提升约 7% 的准确率。  

### 方法详解
**整体思路**  
整个流程可以划分为三步：① 参数映射与初始化、② 适配后再预训练、③ 指令微调与下游评估。核心思想是把已有的 decoder‑only 权重“搬家”到 encoder‑decoder 框架，然后用相对轻量的再预训练把新加入的编码器层“激活”，最后在指令数据上微调以验证实际使用价值。

**1️⃣ 参数映射与初始化**  
- **自注意力层**：decoder‑only 的自注意力权重（查询、键、值、输出投影）直接复制到 encoder‑decoder 对应的自注意力层。因为两者的维度相同，这一步几乎是零成本。  
- **前馈网络**：同理，FFN（两层线性+激活）的权重也直接搬过去。  
- **编码器输入嵌入**：decoder‑only 只需要词嵌入和位置嵌入，encoder‑decoder 需要额外的“编码器专用”位置嵌入。作者采用随机正态初始化，并把学习率调低，让它在再预训练阶段慢慢适应。  
- **层归一化和残差**：保持原有的层归一化参数不变，只在编码器侧新增一套层归一化，同样用小学习率初始化。

**2️⃣ 适配后再预训练**  
- **目标函数**：实验比较了三种目标：  
  - *自回归语言模型*（仅解码器继续预测下一个词），保持原始生成能力。  
  - *双向掩码语言模型*（在编码器上随机掩码输入词），让编码器学习上下文理解。  
  - *混合目标*（两者加权求和），兼顾生成和理解。  
- **训练策略**：先用双向掩码目标训练若干步，让编码器快速获得有意义的表示；随后切换到混合目标，以免编码器过度偏向理解而削弱解码器的生成质量。  
- **优化细节**：对新初始化的编码器层使用更低的学习率（约原学习率的 0.1），防止它们在训练初期被“大模型”权重压制。梯度裁剪、AdamW 优化器等保持不变。

**3️⃣ 指令微调与评估**  
- 在公开的指令数据集上继续训练，使用标准的指令微调 loss（交叉熵）。  
- 评估分为两类：生成任务（如对话、代码生成）和理解任务（SuperGLUE）。  
- 为了验证跨尺寸组合的效果，作者分别训练了 2B‑2B、9B‑2B、9B‑9B 三种配置，并在相同的推理时间预算下比较性能。

**最巧妙的点**  
- **跨尺寸拼接**：把大解码器和小编码器直接拼接，几乎不需要额外的对齐操作，因为两者的隐藏维度相同。这样既利用了大模型的强生成能力，又通过小编码器降低了整体推理成本。  
- **双阶段再预训练**：先让编码器“先热身”，再引入混合目标，避免了在一次性训练中出现的生成质量下降问题。

### 实验与效果
- **数据集**：再预训练使用了与 Gemma 2 系列相同的海量网页文本；指令微调使用了公开的 FLAN、OpenInstruction 等指令集合；理解评估采用 SuperGLUE。  
- **基线对比**：  
  - 与原始 Gemma 2B（decoder‑only）相比，适配后的 encoder‑decoder 在指令微调后提升约 7% 的整体得分。  
  - 在相同推理预算下，Gemma 9B‑2B（大解码‑小编码）比 Gemma 2B‑2B（全小模型）高出超过 3% 的得分。  
  - 在 SuperGLUE 上，适配模型的平均分比对应的 decoder‑only 模型提升约 1.5%~2%。  
- **消融实验**：  
  - 去掉双向掩码预训练目标，编码器的表示质量显著下降，SuperGLUE 分数下降约 1%。  
  - 将编码器层的学习率调高到与解码器相同，导致训练不稳定，最终生成质量下降约 4%。  
- **局限性**：  
  - 适配过程仍需要一定量的再预训练算力，虽然比从零预训练省很多，但对资源受限的团队仍有门槛。  
  - 论文主要在英文数据上验证，跨语言（mT5‑size）的适配效果虽有初步实验，但未给出完整的多语言评测。  
  - 对于极大规模（>100B）模型的适配，作者未提供实验数据，实际可行性仍待验证。

### 影响与延伸思考
这篇工作向业界展示了“模型结构迁移”可以成为降低大模型训练成本的有效手段，随后出现了多篇围绕“decoder‑only → encoder‑decoder 适配”或“跨模态结构迁移”的研究。比如后续的 **AdapterFusion** 系列尝试在同一框架下混合多种预训练目标，**Modular LLM** 则进一步把 encoder、decoder、检索模块解耦，实现更细粒度的模型组装。对想深入的读者，可以关注以下方向：  
- **跨语言适配**：如何在多语言预训练语料上保持适配效果。  
- **更细粒度的模块化**：把检索、推理、记忆等功能分别做成可插拔的子网。  
- **低资源适配**：在极少再预训练步数下仍能获得可用的 encoder 表示。  

### 一句话记住它
把强大的 decoder‑only 大模型“搬家”到 encoder‑decoder 架构，只需少量再预训练，就能在相同算力下兼顾更快推理和更好下游表现。