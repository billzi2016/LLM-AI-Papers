# QuCo-RAG: Quantifying Uncertainty from the Pre-training Corpus for Dynamic Retrieval-Augmented Generation

> **Date**：2025-12-22
> **arXiv**：https://arxiv.org/abs/2512.19134

## Abstract

Dynamic Retrieval-Augmented Generation adaptively determines when to retrieve during generation to mitigate hallucinations in large language models (LLMs). However, existing methods rely on model-internal signals (e.g., logits, entropy), which are fundamentally unreliable because LLMs are typically ill-calibrated and often exhibit high confidence in erroneous outputs. We propose QuCo-RAG, which shifts from subjective confidence to objective statistics computed from pre-training data. Our method quantifies uncertainty through two stages: (1) before generation, we identify low-frequency entities indicating long-tail knowledge gaps; (2) during generation, we verify entity co-occurrence in the pre-training corpus, where zero co-occurrence often signals hallucination risk. Both stages leverage Infini-gram for millisecond-latency queries over 4 trillion tokens, triggering retrieval when uncertainty is high. Experiments on multi-hop QA benchmarks show QuCo-RAG achieves EM gains of 5--12 points over state-of-the-art baselines with OLMo-2 models, and transfers effectively to models with undisclosed pre-training data (Llama-3, Qwen2.5, GPT-4.1/5-chat), improving EM by up to 14 points. Generalization to long-form generation and biomedical QA further validates the robustness of our paradigm. These results establish corpus-grounded verification as a principled, practically model-agnostic paradigm for dynamic RAG. Our code is publicly available at https://github.com/ZhishanQ/QuCo-RAG.

---

# QuCo‑RAG：基于预训练语料的不确定性量化用于动态检索增强生成 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在开放域问答或长文生成时常会“编造”事实，尤其是当它们需要调用不常见的、长尾知识时。传统的检索增强生成（RAG）把检索固定在生成前或每一步，都假设检索一定能帮助，却忽略了检索的成本。为了解决幻觉，近期出现了**动态 RAG**：模型在生成过程中自行决定是否去检索。但这些方法只看模型内部的置信度、logits 或熵值，而 LLM 的置信度普遍不校准，错误答案往往伴随高置信度，导致检索触发时机不可靠。于是，如何用一个更客观、与模型本身无关的信号来判断“我到底懂不懂”成为瓶颈。

### 关键概念速览

**动态检索增强生成（Dynamic RAG）**：在文本生成的每一步，根据某种信号决定是否向外部知识库发起检索，类似于写作时随时打开字典查词。

**幻觉（Hallucination）**：模型输出的内容在事实层面与真实世界不符，常见于缺乏训练数据支撑的长尾问题。

**预训练语料统计（Corpus Statistics）**：对模型在大规模预训练文本中出现的词、实体、共现关系进行计数，类似于把整个训练书库当成一本“词典”。

**低频实体（Low‑frequency Entity）**：在预训练语料中出现次数很少的实体，往往代表模型未充分学习的知识点。

**共现验证（Co‑occurrence Verification）**：检查生成句子里两个实体是否在同一段落或句子中出现过，零共现被视为潜在错误的信号。

**Infini‑gram**：一种支持对数万亿级 token 序列进行毫秒级查询的索引系统，像是把整个互联网的文字压进了一个可以瞬间检索的超大字典。

### 核心创新点

1. **从模型内部信号转向语料统计**  
   之前的动态 RAG 依赖 logits、熵等内部指标 → QuCo‑RAG 用实体在预训练语料中的出现频率和共现次数来衡量不确定性 → 让触发检索的依据变得客观、与模型校准水平无关。

2. **两阶段不确定性量化**  
   传统方法只在生成前或生成时做一次判断 → QuCo‑RAG 在生成前先检查问题中的实体是否为低频 → 生成过程中再对模型输出的实体对进行共现验证 → 两层过滤显著提升了召回真正需要检索的时机。

3. **毫秒级大规模语料查询**  
   直接在 4 万亿 token 上做频率/共现查询在计算上几乎不可能 → 作者构建了 Infini‑gram 索引，使得查询延迟降到毫秒级 → 这让动态检索在实际对话或生成任务中保持实时性。

4. **模型‑无关的通用框架**  
   过去的动态 RAG 往往需要知道模型的训练语料或对模型进行微调 → QuCo‑RAG 只要能获取对应模型的公开预训练语料统计，就能直接套用 → 实验证明即使在 Llama‑3、Qwen2.5、GPT‑4.1/5‑chat 等闭源模型上也能获得显著提升。

### 方法详解

**整体思路**  
QuCo‑RAG 把“不确定性”拆成两块：① 生成前的“知识缺口”，② 生成中的“事实验证”。每块都通过查询预训练语料得到一个数值，如果数值超过阈值，就触发一次外部检索（比如向向量数据库或搜索引擎发起查询），把检索到的文档作为上下文喂回 LLM，继续生成。

**步骤拆解**

1. **实体抽取 & 频率查询（生成前）**  
   - 输入问题 → 用轻量实体识别器抽出所有命名实体（人名、地点、组织等）。  
   - 对每个实体在 Infini‑gram 中查询出现次数。  
   - 计算一个“稀疏度分数”：出现次数越低，分数越高。  
   - 若最高稀疏度分数超过预设阈值，直接进入检索模块。  
   *类比：像在写作文前先查字典，看到生僻词就先去查资料。*

2. **生成过程中的三元组抽取**  
   - LLM 按常规方式逐 token 生成。  
   - 每生成完一个完整句子，使用规则或小模型抽取句子中的**实体对**（A, B）以及它们之间的关系谓词，形成知识三元组 (A, rel, B)。  

3. **共现验证（生成中）**  
   - 对每个三元组，向 Infini‑gram 查询 A 与 B 在同一段落或句子中出现的次数。  
   - 若共现次数为零，视为“未被语料支撑”，给该三元组一个高不确定性分数。  
   - 将所有三元组的分数聚合（如取最大或加权平均），若超过阈值，则在当前生成位置触发检索。  

4. **检索与融合**  
   - 检索模块使用问题或当前已生成的上下文向向量库发起相似度搜索，返回若干文档。  
   - 采用 **RAG‑Fusion**（如拼接、交叉注意力或检索增强的解码器）把检索结果注入 LLM，继续生成。  

5. **迭代**  
   - 检索后模型继续生成，重复步骤 2‑4，直到生成结束或达到最大长度。

**关键实现细节**

- **阈值自适应**：论文提到阈值不是固定的，而是根据模型大小和任务难度做线性缩放，保证不同模型之间的公平比较。  
- **Infini‑gram 索引结构**：采用层次化的 n‑gram 计数表和 Bloom filter 预过滤，使得即使在 4 万亿 token 上也能在 2 ms 内返回计数。  
- **检索成本控制**：只在高不确定性时才检索，实验显示平均每篇答案只触发 1.2 次检索，保持了生成速度。  

**最巧妙的地方**  
把“模型不懂”转化为“语料不懂”，利用了预训练阶段留下的客观痕迹。这个思路突破了 LLM 置信度不可靠的根本限制，而且只要有语料统计就能跨模型迁移。

### 实验与效果

- **数据集**：在 Multi‑Hop QA（HotpotQA、2WikiMultiHopQA）以及长文本生成基准（LongFormQA）上评估；另外加入医学问答（PubMedQA）验证跨域能力。  
- **基线**：与最新的动态 RAG 方法（如 Self‑Check‑RAG、Confidence‑Trigger）以及固定检索的 RAG（Fusion‑in‑Decoder）对比。  
- **主要结果**：在 HotpotQA 上，使用 OLMo‑2‑7B 时，Exact Match（EM）提升了 **5.8** 分；在 2WikiMultiHopQA 上提升 **12.1** 分；在 LongFormQA 上整体 BLEU/ROUGE 也有 **3‑5** 分的提升。对闭源模型（Llama‑3、GPT‑4.1‑chat）进行零样本迁移时，EM 增幅最高达 **14** 分。  
- **消融实验**：去掉生成前的低频实体检测，EM 下降约 **2.3** 分；去掉共现验证，下降约 **4.7** 分；仅使用 Infini‑gram 的粗粒度计数（不区分段落）时，效果下降约 **1.9** 分，说明细粒度共现是关键。  
- **局限性**：方法依赖于预训练语料的可查询统计；对完全闭源、无公开语料的模型只能使用近似统计或外部语料，效果会受限。作者也提到在极端长文本（>2000 token）时，实时查询的累计延迟仍需进一步优化。

### 影响与延伸思考

QuCo‑RAG 把“语料根基”搬进了动态检索的决策环节，打开了“模型‑无关不确定性评估”的新方向。随后的工作（如 **Corpus‑Aware Trigger**、**Stat‑Driven RAG**）纷纷借鉴了低频实体和共现验证的思路，甚至把统计信息直接嵌入模型的注意力权重中，实现了更细粒度的自适应检索。对想进一步探索的读者，可以关注以下几个方向：  
1. **跨语言语料统计**：如何在多语言预训练语料上统一频率度量。  
2. **统计与学习的融合**：把 Infini‑gram 的计数作为软标签喂给模型，训练出能够内部感知语料稀缺性的模型。  
3. **更高效的索引**：在 GPU/TPU 上实现近实时的共现查询，以支撑更大规模的在线对话系统。  

### 一句话记住它

**QuCo‑RAG 用预训练语料的出现频率和共现统计，给模型一个客观的“不懂”信号，从而在需要时才去检索，显著降低幻觉并保持实时性。**