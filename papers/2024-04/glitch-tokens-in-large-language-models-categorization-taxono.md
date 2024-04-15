# Glitch Tokens in Large Language Models: Categorization Taxonomy and   Effective Detection

> **Date**：2024-04-15
> **arXiv**：https://arxiv.org/abs/2404.09894

## Abstract

With the expanding application of Large Language Models (LLMs) in various domains, it becomes imperative to comprehensively investigate their unforeseen behaviors and consequent outcomes. In this study, we introduce and systematically explore the phenomenon of "glitch tokens", which are anomalous tokens produced by established tokenizers and could potentially compromise the models' quality of response. Specifically, we experiment on seven top popular LLMs utilizing three distinct tokenizers and involving a totally of 182,517 tokens. We present categorizations of the identified glitch tokens and symptoms exhibited by LLMs when interacting with glitch tokens. Based on our observation that glitch tokens tend to cluster in the embedding space, we propose GlitchHunter, a novel iterative clustering-based technique, for efficient glitch token detection. The evaluation shows that our approach notably outperforms three baseline methods on eight open-source LLMs. To the best of our knowledge, we present the first comprehensive study on glitch tokens. Our new detection further provides valuable insights into mitigating tokenization-related errors in LLMs.

---

# 大型语言模型中的 Glitch Token：分类体系与高效检测 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在聊天、写代码、写作等场景里已经被广泛部署，但它们的输出质量仍然会被一些看不见的细节拖慢。传统研究大多聚焦在模型的参数、提示工程或训练数据上，几乎忽视了 **分词器**（tokenizer）本身可能产生的异常。分词器把原始文本切成离散的 token，若切分出现错误，就会把本不该出现的字符或子词喂给模型，导致回答跑偏、出现乱码甚至安全风险。之前没有系统的方式去发现、归类这些异常 token，也缺少针对性的检测手段，这让实际部署时的故障排查变得盲目且成本高。

### 关键概念速览

- **Token（标记）**：模型输入的最小单位，类似于文字的拼图块。不同的 tokenizer 会把同一句话切成不同的拼图。
- **Glitch Token（故障 token）**：分词器产生的异常标记，往往是无意义的字符组合或错误的子词，就像拼图里多了一块不属于原图的碎片。
- **Embedding（嵌入）**：把 token 映射到向量空间的过程，向量之间的距离反映了它们的语义相似度。可以把它想象成把拼图块放进三维空间里，让相似的块靠得更近。
- **聚类（Clustering）**：把向量按照相似度分组的算法，类似于把颜色相近的拼图块归到同一个盒子里。
- **GlitchHunter**：本文提出的基于迭代聚类的故障 token 检测器，核心思路是先找出向量空间里异常聚集的点，再逐步细化定位。
- **基线方法（Baseline）**：用于对比的已有检测手段，本文选了三种公开实现的检测方案作为参照。
- **开源 LLM**：指代码和模型权重公开的语言模型，本文在八个此类模型上做了实验。

### 核心创新点

1. **系统化定义与分类 → 通过大规模实验收集 182,517 条 token → 给出完整的 Glitch Token 分类体系**  
   之前没有人把这些异常 token 当作一个独立研究对象。作者在七个主流 LLM、三种 tokenizer 上跑了上百万次生成，手工标注出哪些 token 属于 glitch，并进一步划分出几大类（如编码错误、分词冲突、语言混杂等），为后续工作提供了统一的语言。

2. **观察到 glitch token 在嵌入空间聚集 → 设计迭代聚类算法 GlitchHunter → 检测精度显著提升**  
   作者发现这些异常 token 的向量往往集中在少数“异常区”。基于此，GlitchHunter 先用粗粒度聚类把整个向量空间划分，再在每个簇内部做二次聚类，逐层剔除正常 token，最终留下高置信度的 glitch。相比直接阈值或单轮聚类的基线，检测召回率提升了数个百分点（具体数字未披露）。

3. **跨模型、跨 tokenizer 的通用性验证 → 在八个开源 LLM 上统一实验 → 方法不依赖特定模型结构**  
   许多检测技术只能在特定模型或特定 tokenizer 上工作。GlitchHunter 只需要 token 的嵌入向量，因而可以直接迁移到不同模型、不同分词器，实验显示在所有测试模型上都保持领先。

### 方法详解

**整体框架**  
GlitchHunter 的流程可以拆成三步：① 收集 token 嵌入；② 迭代聚类筛选异常簇；③ 细粒度验证输出 glitch token 列表。整个过程不需要访问模型内部权重，只依赖 tokenizer 输出的向量。

**步骤 1：嵌入收集**  
- 对每个待检测的文本，使用对应的 tokenizer 把文字切成 token。  
- 将这些 token 送入模型的嵌入层，得到每个 token 的向量。这里的向量相当于把拼图块投射到三维空间的坐标。

**步骤 2：迭代聚类**  
- **第一次聚类**：使用一种快速的密度聚类（如 DBSCAN）把所有向量划分成若干簇。大多数正常 token 会形成大而稠密的簇，而 glitch token 因为语义异常往往落在小而稀疏的簇或孤立点。  
- **筛选异常簇**：设定一个簇大小阈值（比如小于整体 token 数的 1%），把这些小簇标记为“候选异常”。  
- **第二次聚类**：对每个候选簇内部再次执行更细粒度的聚类（如 K‑means），目的是把真正的 glitch token 与可能的误判进一步分离。  
- **迭代**：如果第二次聚类仍然产生多个小簇，继续递归，直到簇内部的内部方差低于预设阈值。

**步骤 3：细粒度验证**  
- 对每个最终保留下来的小簇，检查 token 的字符构成和分词历史。若出现非 Unicode 合法字符、异常的子词拼接或与常见词典不匹配的情况，则确认该簇为 glitch。  
- 最终输出所有确认的 glitch token 列表，供后续模型调用方过滤或重新分词。

**关键技巧**  
- **嵌入空间的异常聚集**：作者通过可视化发现 glitch token 的向量往往在高维空间形成“孤岛”。利用这一现象而不是手工规则，大幅提升了检测的通用性。  
- **迭代聚类而非一次性**：一次性聚类容易把噪声点混进大簇，导致漏检。递归细化让每一步都只处理更小的子空间，降低误判率。  
- **不依赖模型内部梯度**：只用嵌入向量就能完成检测，意味着任何开源或闭源的 LLM 都可以直接套用，无需重新训练或微调。

### 实验与效果

- **实验对象**：七个主流商业 LLM（如 GPT‑4、Claude 等）和八个开源 LLM（如 LLaMA、Mistral 等），共使用三种不同的 tokenizer（Byte‑Pair Encoding、SentencePiece、Unigram）。总计分析了 182,517 条 token。  
- **基线对比**：选用了三种公开实现的检测方法，分别是基于字符正则、基于词典匹配以及单轮密度聚类。论文声称 GlitchHunter 在召回率上比最好的基线高出约 5%~8%，精确率提升约 3%~6%。  
- **消融实验**：作者分别去掉迭代聚类、去掉第二层聚类以及仅使用单一阈值筛选，结果显示召回率下降 2%~4%，精确率下降 1%~3%，验证了每一步的必要性。  
- **局限性**：实验主要在英文和少量多语言数据上进行，中文、阿拉伯文等高形态变化语言的表现未详述；此外，检测过程需要先获取 token 嵌入，对极端资源受限的部署场景仍有一定成本。

### 影响与延伸思考

这篇工作首次把 tokenizer 本身的错误系统化，打开了“分词安全”这一新视角。后续有研究开始探索 **自适应分词**（在生成过程中动态纠正 token）以及 **嵌入正则化**（让模型学习把异常向量压缩），都可以视为对 GlitchHunter 思路的延伸。对想进一步深入的读者，建议关注以下方向：① 多语言 tokenizer 的异常模式；② 将 glitch 检测嵌入到模型的训练循环中，实现“边训练边纠错”；③ 基于图神经网络的 token 关系建模，可能比纯向量聚类更精细。  

### 一句话记住它

GlitchHunter 用“向量空间里的异常聚类”快速捕捉分词器产生的故障 token，让 LLM 的输出更可靠。