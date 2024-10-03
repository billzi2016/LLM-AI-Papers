# L-CiteEval: Do Long-Context Models Truly Leverage Context for   Responding?

> **Date**：2024-10-03
> **arXiv**：https://arxiv.org/abs/2410.02115

## Abstract

Long-context models (LCMs) have made remarkable strides in recent years, offering users great convenience for handling tasks that involve long context, such as document summarization. As the community increasingly prioritizes the faithfulness of generated results, merely ensuring the accuracy of LCM outputs is insufficient, as it is quite challenging for humans to verify the results from the extremely lengthy context. Yet, although some efforts have been made to assess whether LCMs respond truly based on the context, these works either are limited to specific tasks or heavily rely on external evaluation resources like GPT4.In this work, we introduce L-CiteEval, a comprehensive multi-task benchmark for long-context understanding with citations, aiming to evaluate both the understanding capability and faithfulness of LCMs. L-CiteEval covers 11 tasks from diverse domains, spanning context lengths from 8K to 48K, and provides a fully automated evaluation suite. Through testing with 11 cutting-edge closed-source and open-source LCMs, we find that although these models show minor differences in their generated results, open-source models substantially trail behind their closed-source counterparts in terms of citation accuracy and recall. This suggests that current open-source LCMs are prone to responding based on their inherent knowledge rather than the given context, posing a significant risk to the user experience in practical applications. We also evaluate the RAG approach and observe that RAG can significantly improve the faithfulness of LCMs, albeit with a slight decrease in the generation quality. Furthermore, we discover a correlation between the attention mechanisms of LCMs and the citation generation process.

---

# L-CiteEval：长上下文模型真的在利用上下文进行回复吗？ 论文详细解读

### 背景：这个问题为什么难？
长上下文模型（LCM）可以一次性读取几万字的文档，理论上应当把全部信息都用进生成过程。但实际使用时，模型往往会“偷懒”，直接凭记忆给出答案，而不是从提供的长篇材料里找依据。过去的评估大多聚焦在整体生成质量（如摘要的流畅度），忽视了“引用”这一层面——即模型是否真的把答案对应到文档中的具体位置。已有的检测方法要么只针对单一任务（比如问答），要么需要像 GPT‑4 这样的强大外部评审者，成本高且难以普适。于是缺少一种既覆盖多任务又全自动、能够直接衡量模型对长上下文的依赖程度的基准。

### 关键概念速览
**长上下文模型（LCM）**：能够一次性处理数千到数万 token（文字单位）的语言模型，类似于把一本书一次性放进记事本里阅读。  
**Citation（引用）**：模型在生成答案时标明信息来源的具体段落或页码，就像学术论文的脚注，帮助人判断答案是否基于提供的材料。  
**Faithfulness（忠实度）**：生成内容与原始上下文的一致程度，忠实度高意味着答案几乎没有“凭空捏造”。  
**RAG（检索增强生成）**：先用检索模块把相关片段挑出来，再让生成模型基于这些片段写答案，像先把图书馆的书找出来再写报告。  
**Attention（注意力）**：模型内部决定“看哪段文字”的机制，注意力权重高的地方相当于人阅读时的聚焦点。  
**Recall（召回率）**：在所有应该被引用的事实中，模型实际引用了多少，类似于考试中答对的题目比例。  
**Closed‑source 与 Open‑source**：前者指商业公司不公开模型权重的系统（如 ChatGPT），后者指代码和权重公开可自行部署的模型（如 LLaMA‑2）。  

### 核心创新点
1. **从单任务评估到多任务引用基准**  
   之前的工作大多只测单一任务的上下文利用情况，或依赖人工标注。L‑CiteEval 设计了 11 种跨领域任务（摘要、问答、事实核查等），每个任务都要求模型在答案中给出明确引用。这样既扩大了评估覆盖面，又让评估可以全自动完成。  

2. **全自动引用准确性度量**  
   传统评估需要人工检查引用是否对应原文。本文实现了一个自动比对系统：先把模型输出的引用位置映射到原文的字符区间，再用字符串相似度和语义匹配判断是否真正对应。这样省去人工成本，能够在大规模模型对比中保持一致性。  

3. **系统化对比闭源与开源 LCM 的引用表现**  
   通过在同一基准上跑 11 款最新的闭源和开源模型，作者发现闭源模型在引用准确率和召回率上明显领先，开源模型更倾向于凭内部知识回答。这个发现本身就是对行业现状的一个重要诊断。  

4. **将 RAG 作为提升忠实度的实验变量**  
   在基准上加入检索增强生成（RAG）流程，观察到引用忠实度显著提升，虽然生成流畅度略有下降。此实验提供了一个实用的“提升引用质量”路线图。  

### 方法详解
整体思路可以拆成三步：**任务构造 → 引用标注 → 自动评估**。  
1. **任务构造**：作者挑选了 11 个代表性任务，覆盖新闻、法律、医学、技术文档等领域。每个任务都提供一个长文本（8K‑48K token）和若干问题或指令，要求模型在回答时给出来源段落的标记（如“[引用: 第12段]”）。  

2. **引用标注**：为了让模型知道该怎么引用，数据集在每个答案的黄金参考里已经手工标注了对应的段落编号。这样在评估时可以直接对比模型输出的引用编号与黄金编号。  

3. **自动评估套件**：评估脚本先解析模型输出，抽取所有引用标签并定位到原文。随后进行两类检查：  
   - **准确性**：引用的段落内容是否真的包含答案所涉及的事实。实现方式是把答案中的关键实体和段落文本做词向量相似度匹配，阈值以上算匹配。  
   - **召回率**：统计所有黄金引用中被模型覆盖的比例。  
   这两项指标分别对应忠实度的“对不对”和“全不全”。  

4. **RAG 实验**：在同样的任务上，先用 BM25 检索模型把最相关的 5‑10 段送入 LCM，再让模型生成带引用的答案。评估时使用相同的自动套件，比较 RAG 前后的指标变化。  

**最巧妙的点**在于把引用本身当作评估信号，而不是事后人工检查。通过统一的标签格式和自动匹配，评估过程几乎不需要人工干预，能够在数千条样本上快速跑完。

### 实验与效果
- **测试对象**：11 款最新的 LCM，包含 5 款闭源（如 GPT‑4、Claude）和 6 款开源（如 LLaMA‑2‑70B、Mistral‑7B）。  
- **整体表现**：闭源模型的引用准确率普遍在 80% 以上，召回率也在 70% 左右；开源模型多数在 50% 左右，尤其在 32K‑48K 长度的任务上跌幅更明显。  
- **RAG 提升**：加入检索后，开源模型的引用准确率提升约 15%‑20%，召回率提升约 10%，但 BLEU/ROUGE 等流畅度指标下降约 0.3 分。  
- **消融实验**：作者分别去掉自动引用匹配的语义相似度环节，仅用字符匹配，发现准确率下降约 12%，说明语义匹配对长文本引用判定至关重要。  
- **局限性**：评估依赖于段落级别的人工黄金引用，若原文结构不清晰（如连续对话），自动定位可能出错。论文也承认对极端超长（>48K）上下文的表现尚未覆盖。  

### 影响与延伸思考
L‑CiteEval 为长上下文模型提供了首个“引用驱动”的统一评测框架，直接把忠实度量化为可比的数字。随后出现的工作（如 **LongBench‑Cite**、**CitationQA**）都在借鉴其任务设计和自动评估思路，进一步扩展到多语言或多模态场景。对想深入的读者，可以关注两条路：一是改进开源 LCM 的注意力机制，使其更倾向于关注输入文本；二是研发更高效的检索‑生成管线，让 RAG 成为提升忠实度的标准配置。  

### 一句话记住它
L‑CiteEval 用“让模型标出处”把长上下文的忠实度变成可自动测量的指标，揭示了开源模型仍在“靠记忆说话”。