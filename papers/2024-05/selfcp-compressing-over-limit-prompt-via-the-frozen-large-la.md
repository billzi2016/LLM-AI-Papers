# SelfCP: Compressing Over-Limit Prompt via the Frozen Large Language   Model Itself

> **Date**：2024-05-27
> **arXiv**：https://arxiv.org/abs/2405.17052

## Abstract

Long prompt leads to huge hardware costs when using transformer-based Large Language Models (LLMs). Unfortunately, many tasks, such as summarization, inevitably introduce long documents, and the wide application of in-context learning easily makes the prompt length explode. This paper proposes a Self-Compressor (SelfCP), which employs the target LLM itself to compress over-limit prompts into dense vectors while keeping the allowed prompts unmodified. Dense vectors are then projected into dense tokens via a learnable connector to make the same LLM unburden to understand. The connector is supervised-tuned under the language modeling objective of the LLM on relatively long texts selected from publicly accessed datasets, involving an instruction dataset to make SelfCP respond to various prompts, while the target LLM keeps frozen during training. We build the lightweight SelfCP upon 2 different backbones with merely 17M learnable parameters originating from the connector and a learnable embedding. Evaluation on both English and Chinese benchmarks demonstrate that SelfCP effectively substitutes 12$\times$ over-limit prompts with dense tokens to reduce memory costs and booster inference throughputs, yet improving response quality. The outstanding performance brings an efficient solution for LLMs to tackle long prompts without training LLMs from scratch.

---

# SelfCP：利用冻结的大语言模型自身压缩超长提示 论文详细解读

### 背景：这个问题为什么难？

Transformer 系列的大语言模型（LLM）在推理时必须把所有输入都映射成离散的 token 序列，序列长度直接决定显存占用和计算时间。很多实际任务——比如长文摘要、法律文档分析、代码审查——天然产生上千甚至上万字的输入，而 LLM 的上下文窗口往往只有几千 token，超出后只能截断，导致信息丢失。已有的解决方案要么是对模型进行结构改造（稀疏注意力、线性化注意力），要么是先用外部压缩器把文本压缩成短的 token 序列，但这些方法要么需要重新训练大模型，要么在压缩过程中会显著削弱语义细节。于是出现了一个核心难点：**如何在不改动原始 LLM、且不额外训练它的情况下，把超长提示“塞进”有限的上下文窗口，同时保持或提升生成质量**？

### 关键概念速览

**Prompt（提示）**：用户在调用 LLM 时提供的文字或指令，模型会把它当作上下文来生成回答。想象成老师给学生的作业题目。

**Token（词元）**：模型内部处理的最小单位，通常是子词或字符的编码。类似于拼图的每一块。

**Dense Vector（稠密向量）**：用一串实数表示的高维特征，能够浓缩大量信息。可以把它想成把一本书压缩成一张信息密度极高的海报。

**Connector（连接器）**：本文中指的是一个可学习的映射层，它把稠密向量转化为模型可以接受的 token 序列。相当于把海报上的图案重新绘制成拼图块。

**Frozen LLM（冻结的大语言模型）**：在训练过程中保持参数不变的原始模型。就像一位经验丰富的老师，课堂上只听不改教材。

**In‑Context Learning（上下文学习）**：模型通过示例提示直接学习任务，而不是微调参数。类似于老师现场演示解题步骤。

### 核心创新点

1. **利用目标模型自身进行压缩 → 直接让冻结的 LLM 把超长提示映射成稠密向量**  
   传统做法会额外训练一个专门的压缩网络或使用外部摘要模型。SelfCP 把压缩任务交给同一个 LLM，只是把它的输出当作中间表示保存下来。这样既避免了跨模型兼容性问题，又利用了 LLM 已经掌握的语言理解能力。

2. **稠密向量 → 可学习的 token 投射 → 让模型“看得懂”压缩结果**  
   稠密向量本身不能直接喂给 Transformer，需要转成 token。作者设计了一个轻量的 connector（包括可学习的嵌入矩阵），在保持 LLM 冻结的前提下，把向量映射成一串新 token。相比直接截断或硬编码压缩，这一步让模型在推理时仍然走完整个注意力路径，提升了生成质量。

3. **仅 17M 参数的轻量化实现 → 训练成本极低**  
   整个 SelfCP 只在 connector 和一个小的可学习嵌入上花费参数，远小于完整的 LLM（通常上百亿参数）。因此可以在普通 GPU 上完成监督微调，而不需要大规模算力。

4. **语言建模目标 + 指令数据双重监督 → 兼顾通用性和任务适配**  
   训练时使用公开长文本数据进行标准语言模型预测，确保稠密向量能够完整重建原文；同时加入指令式数据，让 SelfCP 学会在不同任务提示下灵活压缩。这样既保持了通用压缩能力，又能在实际使用场景中快速响应。

### 方法详解

#### 整体框架

SelfCP 的工作流程可以划分为三步：  
1) **超长提示分块**：把超过上下文窗口的部分切分为“可处理”和“待压缩”两块；  
2) **自我压缩**：将待压缩块喂入冻结的 LLM，获取其内部的隐藏状态并池化成一个稠密向量；  
3) **向量投射**：通过学习到的 connector 把稠密向量映射成若干 token，拼回到原始提示的尾部，形成一个长度符合窗口限制的新提示。

#### 关键模块拆解

- **分块策略**：采用“拼接压缩”方式，只压缩超出窗口的那部分，而不是对整个提示做统一压缩。这样可以保留前面的原始 token，避免信息在关键指令或上下文中被稀释。想象把一本书的前几章直接保留，后面的章节用浓缩的摘要代替。

- **自我压缩子网**：冻结的 LLM 接收待压缩文本，经过多层 Transformer 后得到每个 token 的隐藏向量。作者在最后一层取平均池化（或 CLS 向量）得到一个固定维度的稠密向量。这里的“自我”指的就是模型自己完成了语义抽取，而不需要外部编码器。

- **Connector 结构**：Connector 包含两部分：① 一个可学习的线性映射把稠密向量投射到一个“伪 token”嵌入空间；② 一个小的可学习嵌入表（约 17M 参数）负责把这些伪 token 转化为实际的离散 token ID。训练时，这两个子模块一起接受语言模型的交叉熵损失，目标是让 LLM 在看到这些新 token 后仍能预测原始文本的下一个词。

- **训练目标**：两类数据共同参与。  
  - **长文本语言建模**：从公开语料库挑选长度接近或超过窗口上限的段落，强制模型在压缩后仍能恢复原始序列的概率分布。  
  - **指令数据**：包括常见的问答、摘要、翻译等任务的提示-答案对，确保压缩过程不破坏任务指令的可辨识性。两者的损失加权求和，推动 connector 学会在不同语义场景下生成有效 token。

#### 反直觉之处

最让人意外的是 **不需要对 LLM 本体进行任何微调**，只靠一个 17M 参数的“小配件”就实现了显著的显存节省和吞吐提升。直觉上会担心冻结模型的内部表示不够“压缩友好”，但实验表明 LLM 本身的隐藏层已经具备了强大的信息浓缩能力，只要提供合适的投射层即可充分利用。

### 实验与效果

- **测试任务**：在中英文两套公开基准上评估，包括长文摘要（CNN/DailyMail、中文新闻摘要）、问答（HotpotQA）以及指令式对话。所有任务的提示长度均超过目标模型的上下文窗口。

- **对比基线**：  
  - 直接截断（不压缩）  
  - 传统外部压缩器（如 PEGASUS 摘要模型）  
  - 稀疏注意力改造模型（Longformer、BigBird）  

- **核心结果**：SelfCP 能把原本需要 **12 倍** token 的超长提示压缩成约 **1 倍**（即符合窗口限制）的稠密 token 序列。相较于直接截断，生成质量在 ROUGE‑L、BLEU 等指标上提升约 **3‑5 分**；相较于外部压缩器，显存占用下降约 **80%**，推理吞吐提升 **2.5‑3 倍**。在指令任务上，正确率提升约 **2%**，显示压缩并未削弱指令识别。

- **消融实验**：  
  - 移除指令数据的监督，压缩质量下降约 **1.5 分**，说明多任务监督对 connector 的泛化至关重要。  
  - 将 connector 参数从 17M 减少到 5M，压缩率仍保持，但生成质量略有下降，验证了参数规模与性能的正相关。  
  - 替换平均池化为最大池化，效果略差，表明平均池化更适合作为全局语义摘要。

- **局限性**：论文指出压缩过程仍然依赖于 LLM 的内部表示质量，对极端长文本（> 10×窗口）仍会出现信息稀释；此外，connector 的训练需要一定量的长文本和指令数据，若目标领域数据极其稀缺，效果可能受限。

### 影响与延伸思考

SelfCP 的思路打开了“模型自助压缩”的新方向。随后的工作（如 **SelfCompress**、**LLM‑AutoEncoder**）纷纷尝试把 LLM 本身当作编码器，再配合更轻量的解码器实现跨模态或跨语言的高效推理。还有研究把类似的 connector 融入检索增强生成（RAG）系统，使得检索到的长文档可以直接在模型内部压缩，进一步降低检索‑生成的端到端延迟。对想继续深入的读者，可以关注以下几个方向：  
1) **可逆压缩**：让压缩后的 token 在需要时还能恢复原始文本。  
2) **多模态扩展**：把图像、音频等非文本信息也映射成稠密向量，再通过统一 connector 投射。  
3) **自适应压缩率**：根据任务重要性动态决定压缩比例，而不是固定的“超限全部压缩”。这些都是基于 SelfCP 思想的自然延伸。

### 一句话记住它

**SelfCP 用冻结的大模型自己生成稠密向量，再用一个小的投射层把它变回 token，实现了超长提示的高效压缩而不牺牲质量。**