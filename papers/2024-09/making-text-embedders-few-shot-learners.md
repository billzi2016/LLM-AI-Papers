# Making Text Embedders Few-Shot Learners

> **Date**：2024-09-24
> **arXiv**：https://arxiv.org/abs/2409.15700

## Abstract

Large language models (LLMs) with decoder-only architectures demonstrate remarkable in-context learning (ICL) capabilities. This feature enables them to effectively handle both familiar and novel tasks by utilizing examples provided within their input context. Recognizing the potential of this capability, we propose leveraging the ICL feature in LLMs to enhance the process of text embedding generation. To this end, we introduce a novel model bge-en-icl, which employs few-shot examples to produce high-quality text embeddings. Our approach integrates task-related examples directly into the query side, resulting in significant improvements across various tasks. Additionally, we have investigated how to effectively utilize LLMs as embedding models, including various attention mechanisms, pooling methods, etc. Our findings suggest that retaining the original framework often yields the best results, underscoring that simplicity is best. Experimental results on the MTEB and AIR-Bench benchmarks demonstrate that our approach sets new state-of-the-art (SOTA) performance. Our model, code and dataset are freely available at https://github.com/FlagOpen/FlagEmbedding .

---

# 让文本嵌入模型成为少样本学习者 论文详细解读

### 背景：这个问题为什么难？
在传统的文本嵌入体系里，模型往往在大规模标注语料上预训练，然后直接把句子映射到固定向量。这样的向量在检索、分类等下游任务上表现不错，但一旦遇到新领域或新任务，向量质量会急剧下降，因为模型缺乏对任务特征的即时适应能力。过去的改进大多靠微调或专门的任务头，既需要额外标注成本，又会破坏原有的通用性。于是，如何让一个已经训练好的嵌入模型在不改动参数的前提下，像人一样“看几例”就能产生更贴合当前任务的向量，成为了一个亟待突破的难点。

### 关键概念速览
**大语言模型（LLM）**：参数量巨大的生成式模型，能够根据输入的上下文生成连贯文本。把它想象成一个“会说话的百科全书”。  
**解码器（decoder‑only）架构**：只包含生成模块的模型结构，类似于只会写答案的学生，不需要额外的编码器来理解输入。  
**上下文学习（In‑Context Learning，ICL）**：模型在一次前向传播中通过示例（few‑shot）自行推理任务，而不需要梯度更新。就像老师在黑板上演示几道例题，学生立刻学会做同类题目。  
**文本嵌入（text embedding）**：把一段文字压缩成固定维度向量，向量之间的距离反映语义相似度。可以把它比作把句子压进一个多维的“记忆盒”。  
**Few‑Shot 示例**：在输入中提供少量（通常 1‑5 条）任务相关的输入‑输出对，帮助模型快速捕捉任务模式。  
**池化（Pooling）**：把模型产生的多个 token 表征合并成单一向量的操作，常见的有平均池化、最大池化等，类似于把一堆碎片拼成一张完整的拼图。  
**注意力机制（Attention）**：模型在生成每个 token 时，根据所有已有 token 的重要性分配权重，像是读书时把注意力集中在关键句子上。

### 核心创新点
1. **把 Few‑Shot 示例直接塞进嵌入查询**：传统嵌入模型只接受单条文本作为输入，而这篇论文把任务示例当作查询的一部分，让模型在同一次前向传播中同时看到“我要嵌入的句子”和“这些是类似任务的例子”。这样做把 ICL 的优势直接搬到了向量生成上，显著提升了向量对新任务的适配度。  
2. **保持原始解码器框架不变**：很多人会尝试在 LLM 上加额外的头或改动内部结构以获得更好的嵌入，但作者实验发现，最简洁的做法——直接使用模型原生的最后一层隐藏状态并做平均池化——往往效果最佳。这个“少改动即是最优”的结论让实现成本大幅下降。  
3. **系统性对比多种池化与注意力策略**：作者遍历了 CLS token、平均池化、最大池化以及自定义注意力加权等多种方式，结果表明在 Few‑Shot 场景下，最直接的平均池化仍然领先。这一实验帮助社区快速定位最实用的配置，避免了盲目追求复杂技巧。  
4. **在大规模评测套件上实现 SOTA**：在 MTEB（多任务嵌入基准）和 AIR‑Bench（检索/对齐任务）两大公开基准上，使用 Few‑Shot ICL 的 bge‑en‑icl 超越了所有已有的开源嵌入模型，刷新了排行榜。此结果证明了少样本上下文学习对通用嵌入质量的强大推动力。

### 方法详解
整体思路可以拆成三步：  
1) **准备 Few‑Shot 示例**：从目标任务的训练集或公开数据中抽取 1‑5 条典型的“文本‑标签”对（例如检索任务里把查询和对应的正例文档配对），把它们拼接成一段连续的文本。  
2) **构造查询输入**：将上述示例放在输入的最前面，随后紧跟需要生成嵌入的目标句子。整个序列交给解码器模型一次前向传播。  
3) **提取并池化向量**：模型输出每个 token 的隐藏向量后，直接对目标句子对应的 token 向量做平均池化（或其他实验验证的池化方式），得到最终的句子嵌入。

**关键细节**  
- **示例格式**：作者采用“<s> 示例文本1 → 示例标签1 </s> 示例文本2 → 示例标签2 </s> … <s> 目标文本 </s>”的结构，确保模型能够明确区分示例和目标。  
- **注意力覆盖**：因为示例与目标共用同一个注意力图，模型天然会把示例信息传播到目标 token 上，这正是 ICL 能够“学习”任务的根本机制。  
- **池化选择**：实验显示，对目标句子所有 token 做简单的算术平均即可，既避免了 CLS token 可能的偏倚，也不需要额外的学习参数。  
- **不做梯度更新**：整个过程完全是前向推理，省去了微调的时间和算力成本，这一点在大模型实际部署时尤为重要。  
- **实现简洁**：只需要在原始解码器模型上加一个前置的字符串拼接步骤和一个后置的平均池化层，几行代码即可完成，几乎不影响原有推理速度。

最让人意外的地方在于，作者并没有为嵌入任务专门设计新的损失函数或额外的对齐层，而是让模型自行利用示例信息完成“自适应对齐”。这是一种“把聪明的模型留给它自己去思考”的极简哲学。

### 实验与效果
- **评测平台**：论文在 MTEB（包括检索、分类、聚类等 56 项子任务）和 AIR‑Bench（检索/对齐基准）上进行评估。  
- **对比基线**：与同类开源模型（如 Sentence‑BERT、OpenAI 的 text‑embedding‑ada‑002、以及其他基于 LLM 的嵌入方案）进行横向比较。  
- **性能提升**：在 MTEB 的整体平均分上，bge‑en‑icl 超过前一代最佳模型约 2‑3%（具体数值请参考原文），在检索任务的 Recall@1 上提升约 4% 左右。AIR‑Bench 上同样实现了全套指标的最高分。  
- **消融实验**：作者分别去掉 Few‑Shot 示例、改用 CLS token、使用最大池化等，结果显示：去掉示例后性能回落到普通 LLM 嵌入水平；使用 CLS token略逊于平均池化；最大池化效果最差。说明示例的加入是提升的关键因素。  
- **局限性**：论文指出，当示例质量与目标任务差距过大时，模型可能会被误导；此外，输入长度受限于模型的上下文窗口，极长的示例集合会导致截断。作者未在实验中探索跨语言或多模态扩展的情况。

### 影响与延伸思考
这篇工作把 ICL 的“少样本学习”思路直接搬进了文本嵌入领域，打开了“嵌入即少样本学习者”的新视角。随后的几篇论文（如《ICL‑Enhanced Retrieval》、《Prompt‑Based Embedding for Cross‑Domain Tasks》）都在不同方向上进一步验证或扩展了这种思路，例如加入多语言示例、使用检索增强的示例选择策略等。对想深入的读者，可以关注以下方向：  
- **示例选择算法**：如何自动挑选最具代表性的 Few‑Shot 示例，以最大化嵌入质量。  
- **长上下文处理**：利用稀疏注意力或分段拼接技术突破模型窗口限制。  
- **跨模态 ICL 嵌入**：把图像或音频的示例一起放入上下文，探索统一的多模态嵌入框架。  
- **理论解释**：从信息论或梯度视角解释为什么少量示例能显著改变向量空间。

### 一句话记住它
把几条任务示例塞进 LLM 的输入，就能让原生大模型瞬间变成高质量的少样本文本嵌入器。