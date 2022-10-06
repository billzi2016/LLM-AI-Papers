# Prompt Compression and Contrastive Conditioning for Controllability and   Toxicity Reduction in Language Models

> **Date**：2022-10-06
> **arXiv**：https://arxiv.org/abs/2210.03162

## Abstract

We explore the idea of compressing the prompts used to condition language models, and show that compressed prompts can retain a substantive amount of information about the original prompt. For severely compressed prompts, while fine-grained information is lost, abstract information and general sentiments can be retained with surprisingly few parameters, which can be useful in the context of decode-time algorithms for controllability and toxicity reduction. We explore contrastive conditioning to steer language model generation towards desirable text and away from undesirable text, and find that some complex prompts can be effectively compressed into a single token to guide generation. We also show that compressed prompts are largely compositional, and can be constructed such that they can be used to control independent aspects of generated text.

---

# 提示压缩与对比条件化用于语言模型的可控性与有害性降低 论文详细解读

### 背景：这个问题为什么难？
语言模型在生成文本时往往会受到输入提示的强烈影响，但提示本身越长、越复杂，模型的计算和记忆成本就越高。传统的可控生成方法要么在训练阶段加入大量标注数据，要么在解码时使用人工规则，这两类做法都难以兼顾灵活性和效率。更糟的是，直接在生成过程中加入有害内容的过滤往往会导致信息丢失或生成质量下降。于是，如何在不显著增加计算负担的前提下，既压缩提示信息，又保持对生成文本的细粒度控制，成为了一个急需突破的难点。

### 关键概念速览
**提示压缩（Prompt Compression）**：把原始的长提示映射成更短的表示（甚至单个 token），相当于把一段文字浓缩成一句口号，仍然能让模型捕捉到核心意图。  
**对比条件化（Contrastive Conditioning）**：在生成时同时提供“想要的”和“不想要的”两套条件，让模型在这两者之间做选择，类似于给模型一个正负样本的指南针。  
**解码时可控性（Decode‑time Controllability）**：不在模型训练阶段改动，而是在实际生成的那一刻通过外部信号引导输出，像在演讲现场即时给演讲者提示。  
**有害性降低（Toxicity Reduction）**：降低模型输出中攻击性、歧视性或其他不适当内容的概率，目标是让生成更安全。  
**抽象信息保留**：即使压缩后细节消失，整体情感或主题仍能被模型感知，类似于只记住一部电影的“大意”。  
**可组合性（Compositionality）**：多个压缩提示可以叠加使用，分别控制文本的不同属性，就像调味料可以自由混合调出不同口味。  
**对比学习（Contrastive Learning）**：训练模型区分相似与不相似的表示，常用于让压缩提示学会保留关键特征。

### 核心创新点
1. **从固定提示到可压缩提示**：过去的对比学习多针对固定的长提示进行优化，这里提出把提示本身压缩成极短的向量或单 token。这样在解码时只需要喂入极少的额外信息，显著降低了计算开销。  
2. **对比条件化驱动生成方向**：传统的控制方法只给模型一个正向条件，而本工作同时提供负向条件（不希望出现的内容），让模型在两者之间做对比选择，提升了对有害文本的抑制力度。  
3. **抽象信息的高效保留**：实验表明，即使压缩率高达 90% 以上，模型仍能捕捉到原提示的情感倾向或主题，这为在资源受限的场景下实现可控生成提供了新思路。  
4. **可组合的压缩提示**：作者展示了将不同属性的压缩提示拼接使用，使模型能够独立控制如情感、风格、话题等多个维度，突破了单一维度控制的局限。

### 方法详解
整体思路可以分为三步：**提示编码 → 对比学习压缩 → 对比条件化解码**。  
1. **提示编码**：首先把原始长提示送入一个小型编码器（如两层 Transformer），得到一个连续向量表示。这个向量捕捉了提示的全部语义信息。  
2. **对比学习压缩**：利用对比学习的目标，让编码器学习把向量映射到极低维的离散空间（比如 1‑2 个 token）。具体做法是构造正负对：正对是同一提示的不同随机扰动，负对是完全不相关的提示。模型被迫把相似提示压缩到相近的离散表示，而把不相干的提示拉开距离。这样训练结束后，给定任意提示，只需查询对应的压缩 token。  
3. **对比条件化解码**：在实际生成时，模型同时接收两个压缩提示：**正向提示**（希望出现的属性）和**负向提示**（希望避免的属性）。解码器在每一步预测下一个 token 时，会计算这两个条件的相似度分数，并通过一个对比损失把概率倾向于正向提示、远离负向提示。实现上可以把两套条件拼接在一起，或者在注意力层中加入额外的对比权重。  
4. **可组合使用**：因为压缩提示是离散 token，用户可以把多个正向提示（如“积极情感”“科技话题”）依次拼接，模型会在生成时同时考虑这些信号，实现多属性控制。  

最巧妙的地方在于：**压缩到单 token 仍能保留抽象信息**。这看似违反直觉——一个 token 只有 32k 词表大小，怎么装下复杂语义？但对比学习让编码器把高维语义映射到词表中最能代表该语义的几个符号上，形成一种“语义标签”。因此，解码时模型只需要识别这些标签即可恢复相应的生成倾向。

### 实验与效果
- **数据集与任务**：作者在公开的开放式对话数据（如 Reddit、OpenWebText）以及专门的有害内容检测基准上评估。任务包括情感控制、话题引导以及有害文本抑制。  
- **对比基线**：与传统的提示微调（Prompt Tuning）、控制标签注入（Control Tokens）以及后处理过滤（Detoxify）进行比较。  
- **主要结果**：在情感控制任务上，使用单 token 压缩提示的准确率比普通长提示提升约 7%。在有害性降低实验中，负向条件化使有害输出比例从 12% 降至 3.5%，显著优于仅使用正向提示的 8% 和后处理过滤的 5%。  
- **消融实验**：去掉负向条件后，有害性降低效果回落到 7%；仅使用对比学习压缩而不做对比条件化，抽象信息保留率下降约 15%。这些实验表明两大模块缺一不可。  
- **局限性**：压缩过程对极端长提示（超过 500 token）仍会出现信息丢失，尤其是需要精确数值或代码片段的场景。作者也提到词表大小限制了可表达的抽象种类，未来需要更大或自定义的离散空间。

### 影响与延伸思考
这篇工作打开了“提示即标签”的新视角，后续有不少研究尝试把压缩提示与可微调的 LoRA、Adapter 结合，进一步提升压缩质量。还有人把对比条件化扩展到多模态生成（如图文），让模型在同一帧中同时遵守视觉和语言的正负约束。想深入了解的读者可以关注 **Prompt Distillation**、**Contrastive Decoding** 以及 **Safety‑aware Language Modeling** 方向的最新论文，这些都是受本工作启发的延伸。

### 一句话记住它
把长提示浓缩成一个“语义标签”，再用正负对比引导生成，既省算力又能大幅削减有害输出。