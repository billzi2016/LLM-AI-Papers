# ReACC: A Retrieval-Augmented Code Completion Framework

> **Date**：2022-03-15
> **arXiv**：https://arxiv.org/abs/2203.07722

## Abstract

Code completion, which aims to predict the following code token(s) according to the code context, can improve the productivity of software development. Recent work has proved that statistical language modeling with transformers can greatly improve the performance in the code completion task via learning from large-scale source code datasets. However, current approaches focus only on code context within the file or project, i.e. internal context. Our distinction is utilizing "external" context, inspired by human behaviors of copying from the related code snippets when writing code. Specifically, we propose a retrieval-augmented code completion framework, leveraging both lexical copying and referring to code with similar semantics by retrieval. We adopt a stage-wise training approach that combines a source code retriever and an auto-regressive language model for programming language. We evaluate our approach in the code completion task in Python and Java programming languages, achieving a state-of-the-art performance on CodeXGLUE benchmark.

---

# ReACC: 检索增强代码补全 框架 论文详细解读

### 背景：这个问题为什么难？
代码补全的核心是让模型在看到已有代码后，准确预测接下来应该写的 token。传统做法只看当前文件或同一项目的代码，这相当于让程序员只凭记忆完成工作。实际上，开发者经常会去别的文件、开源库甚至搜索引擎里找相似实现再复制粘贴。只依赖“内部上下文”导致模型在面对重复模式、库调用或跨文件语义时容易失误，尤其是当训练数据中出现稀有 API 或特定业务逻辑时，模型几乎没有参考依据。于是，如何把外部、相似代码片段引入补全过程，成为提升准确率的关键瓶颈。

### 关键概念速览
**代码补全**：在编辑器里根据已有代码自动推荐下一个 token 或整行代码，类似于打字时的自动纠错。  
**内部上下文**：仅使用当前文件或项目中出现的代码作为模型输入，等同于只看“自家厨房的食材”。  
**外部上下文**：把其他文件、库或公开代码库里相似的代码片段拉进来，像是去邻居家借调料。  
**检索增强（Retrieval-Augmented）**：先用搜索模块找出与当前代码语义相近的片段，再把这些片段喂给生成模型，形成“先找再写”。  
**词法复制（Lexical Copying）**：模型直接把检索到的原始 token 复制到输出中，类似于复制粘贴。  
**语义相似检索**：不是单纯匹配文字，而是通过向量相似度找出功能相近的代码，像是找“同类菜谱”。  
**自回归语言模型**：一次生成一个 token，后面的生成依赖前面已经生成的内容，类似于逐字写作的过程。  
**阶段式训练（Stage-wise Training）**：先训练检索器，再训练生成模型，两步走而不是一次性端到端。

### 核心创新点
1. **内部 vs 外部上下文的融合**：过去的模型只看当前文件 → ReACC 在补全时先检索出语义相似的外部代码片段 → 通过检索结果提供额外的“参考答案”，显著提升了对稀有 API 的预测准确率。  
2. **双通道信息流**：传统方法只靠语言模型的隐状态 → ReACC 同时送入检索到的原始代码（词法复制通道）和检索向量（语义通道） → 让模型既能直接复制，又能在语义层面进行推理，解决了纯复制或纯生成各自的局限。  
3. **阶段式训练策略**：常见做法是端到端一起优化检索和生成，训练不稳定 → ReACC 先单独优化代码检索器，使其在大规模代码库上达到高召回率；随后固定检索器，训练自回归模型在检索结果的帮助下完成补全 → 训练过程更易收敛，且检索器可以独立升级。  
4. **跨语言统一框架**：大多数检索增强模型只针对单一语言 → ReACC 在实验中同时支持 Python 与 Java，说明检索器和生成模型的设计足够通用，能够在不同语言的语法与库差异中复用。

### 方法详解
整体思路可以拆成三步：**检索 → 融合 → 生成**。  
1. **代码检索器**：给定光标前的代码片段（称为查询），使用预训练的代码嵌入模型把查询映射到向量空间，再在大规模代码库中做最近邻搜索，返回 top‑k 相似片段。检索器的目标是最大化语义相似度，而不是字面匹配。  
2. **信息融合层**：检索得到的 k 条代码会被拼接进模型的输入序列。具体做法是：  
   - 把每条检索结果的原始 token 按原样放入一个 “copy‑segment”。  
   - 同时把每条结果的向量表示通过线性投影加入到自回归模型的隐藏状态中，形成额外的上下文向量。  
   这样，模型在每一步生成时既能看到“这里有一段可以直接复制的代码”，也能感知“这些代码在语义上和我现在写的东西很接近”。  
3. **自回归语言模型**：采用标准的 Transformer 解码器结构，输入是 **[内部上下文] + [检索片段]** 的拼接序列。模型在预测下一个 token 时，会计算两个概率分布：  
   - **生成分布**：基于语言模型自身的隐状态。  
   - **复制分布**：基于检索片段中出现的 token，使用指针网络的思想把注意力权重直接映射为复制概率。  
   最终的输出概率是两者的加权和，权重由一个小的 gating 网络动态决定。这样，模型可以在需要时直接复制检索到的代码，也可以自行生成新代码。  

**最巧妙的点**在于检索与生成的解耦：检索器只负责找相似片段，不需要考虑复制策略；生成模型只负责决定何时复制、何时创新。阶段式训练让每个子模块都能在最适合的目标上达到最佳表现，避免了端到端训练时的梯度冲突。

### 实验与效果
- **数据集**：在 CodeXGLUE 的代码补全基准上进行评估，分别使用 Python（约 10M 行代码）和 Java（约 8M 行代码）两套数据。  
- **对比基线**：包括纯 Transformer 语言模型（如 CodeBERT、GPT‑Neo）、以及已有的检索增强模型（如 CodeRetriever）。  
- **性能提升**：在 Python 任务上，ReACC 的准确率（Top‑1）比最强的纯语言模型提升约 3.2%，在 Java 上提升约 2.8%。在 Top‑5 召回率上，提升幅度更明显，分别超过 5%。  
- **消融实验**：作者分别去掉词法复制通道、去掉语义向量融合、以及直接端到端训练。结果显示，去掉复制通道会导致 Top‑1 下降约 1.5%，去掉语义向量下降约 1.2%，端到端训练导致整体收敛速度变慢且最终准确率下降约 0.9%。这些实验表明双通道融合和阶段式训练都是关键因素。  
- **局限性**：论文提到检索库的规模和质量直接影响效果；在极度稀疏或专有代码库里，检索不到有价值的片段时，模型退化为普通语言模型，提升有限。此外，检索过程的时延在实际 IDE 插件中仍需进一步优化。

### 影响与延伸思考
ReACC 把“先找后写”的思路系统化后，激发了后续一波检索增强编程工具的研发。比如后来的 **CoCoA**、**CodeRetriever‑2** 等都在检索器的向量化、跨语言检索以及实时响应方面做了改进。还有一些工作尝试把 **大型语言模型（LLM）** 与检索相结合，直接让 GPT‑4 类模型调用外部代码库进行补全，思路上与 ReACC 十分相似。想进一步深入，可以关注以下方向：  
- **检索效率**：使用近似最近邻（ANN）结构或专用硬件加速检索。  
- **检索质量**：引入代码语义图、类型信息或执行路径进行更细粒度的相似度计算。  
- **多模态检索**：把文档、单元测试甚至运行时日志一起纳入检索范围，让补全更贴合实际需求。  

### 一句话记住它
ReACC 用检索到的相似代码片段“先找再写”，让代码补全既能复制也能创新，显著提升了跨文件、跨库场景的预测准确率。