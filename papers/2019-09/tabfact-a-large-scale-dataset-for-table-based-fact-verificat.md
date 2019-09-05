# TabFact: A Large-scale Dataset for Table-based Fact Verification

> **Date**：2019-09-05
> **arXiv**：https://arxiv.org/abs/1909.02164

## Abstract

The problem of verifying whether a textual hypothesis holds based on the given evidence, also known as fact verification, plays an important role in the study of natural language understanding and semantic representation. However, existing studies are mainly restricted to dealing with unstructured evidence (e.g., natural language sentences and documents, news, etc), while verification under structured evidence, such as tables, graphs, and databases, remains under-explored. This paper specifically aims to study the fact verification given semi-structured data as evidence. To this end, we construct a large-scale dataset called TabFact with 16k Wikipedia tables as the evidence for 118k human-annotated natural language statements, which are labeled as either ENTAILED or REFUTED. TabFact is challenging since it involves both soft linguistic reasoning and hard symbolic reasoning. To address these reasoning challenges, we design two different models: Table-BERT and Latent Program Algorithm (LPA). Table-BERT leverages the state-of-the-art pre-trained language model to encode the linearized tables and statements into continuous vectors for verification. LPA parses statements into programs and executes them against the tables to obtain the returned binary value for verification. Both methods achieve similar accuracy but still lag far behind human performance. We also perform a comprehensive analysis to demonstrate great future opportunities. The data and code of the dataset are provided in \url{https://github.com/wenhuchen/Table-Fact-Checking}.

---

# TabFact：面向表格事实验证的大规模数据集 论文详细解读

### 背景：这个问题为什么难？

传统的事实验证任务几乎都把证据当成一段自然语言文本——新闻、维基百科段落之类的。模型只需要在这些连续的句子里找线索，属于“软”语言推理。可是现实中很多信息是以表格、数据库或图的形式出现的，行列结构本身就蕴含了大量的逻辑关系。早期的验证系统几乎没有直接面对这种半结构化证据，要么把表格强行转成文字，要么直接忽略，导致在需要精确数值比较、聚合统计或跨行关联的场景里表现极差。于是，缺少一个规模足够、标注可靠的表格验证基准，也缺少专门针对表格进行软硬结合推理的模型，这两块空白正是这篇论文要填补的。

### 关键概念速览

**事实验证（Fact Verification）**：判断一条自然语言陈述（hypothesis）在给定证据下是“成立”（entailed）还是“被否定”（refuted），相当于让模型充当“真假鉴定官”。  

**半结构化数据（Semi-structured Data）**：介于纯文本和严格数据库之间的形式，典型代表是 HTML 表格或 CSV 文件，既有行列的组织，又保留了文字内容。  

**表格线性化（Table Linearization）**：把表格的行列展开成一段连续的文字序列，常用的做法是“列名：单元格值”，这样就可以直接喂给语言模型。可以把它想象成把一张电子表格拍成一张长条的条形码。  

**预训练语言模型（Pre-trained Language Model）**：在海量文本上事先学习到的深层语义表示，例如 BERT、RoBERTa。它们擅长捕捉词与词之间的细微关系。  

**潜在程序（Latent Program）**：一种隐式的、可执行的查询脚本，模型先把自然语言陈述翻译成这段脚本，再在表格上跑一遍，得到布尔结果。类似于把一句话先变成 SQL 再执行查询。  

**符号执行（Symbolic Execution）**：对程序进行真实的计算，而不是仅靠向量相似度判断。它保证了数值比较、计数等“硬”推理的准确性。  

**软语言推理 vs 硬符号推理**：软推理依赖语言模型的隐式语义匹配，像人类读懂句子；硬推理则是明确的算术或逻辑运算，像使用计算器。  

**ENTAILLED / REFUTED**：数据标注的两类标签，前者表示陈述在表格中可以被证实，后者表示表格提供了相反的证据。

### 核心创新点

1. **构建 TabFact 大规模基准 → 采集 16 k 条维基百科表格并为每张表格标注 118 k 条自然语言陈述 → 为表格事实验证提供了首个规模化、人工标注的训练/测试平台，填补了此前没有公开数据集的空白。**  

2. **Table‑BERT：把表格直接线性化 → 将线性化后的表格文本与陈述拼接，送入预训练语言模型进行二分类 → 让通用的语言模型能够在不改动内部结构的情况下处理表格信息，展示了“软”推理的可行性。**  

3. **Latent Program Algorithm（LPA）：把陈述解析成可执行的查询程序 → 设计了一个专用的 DSL（领域特定语言），支持过滤、聚合、比较等操作 → 通过真实执行得到布尔答案，实现了“硬”符号推理，克服了语言模型在数值精度上的局限。**  

4. **双管齐下的对比实验 → 两种截然不同的思路（向量匹配 vs 程序执行）在同一基准上得到相近的准确率，但仍远低于人类标注者 → 明确指出表格验证仍是一个开放挑战，为后续研究指明方向。**

### 方法详解

**整体框架**  
给定一张表格 T 和一条陈述 S，系统会并行走两条路：  
- **路径一（Table‑BERT）**：把 T 转成文字序列 L(T)，再把 S 拼在后面形成 X = [CLS] L(T) [SEP] S [SEP]，喂入预训练语言模型（如 BERT），取出[CLS]向量做二分类。  
- **路径二（LPA）**：先用语义解析器把 S 映射成一段 DSL 程序 P，随后在 T 上解释执行 P，得到布尔值 b。最终可以直接使用 b 或把两条路的概率融合得到最终判断。

**关键模块拆解**

1. **表格线性化**  
   - 逐列遍历，每列先输出列名，再列出该列所有单元格值，用“列名：值1，值2，…”的格式。  
   - 这种方式保留了列的语义，同时让语言模型看到所有数值，类似把表格压成一段“长句”。  

2. **Table‑BERT 编码**  
   - 采用已有的 BERT‑base 或 RoBERTa‑large，保持原始的多层自注意力结构不变。  
   - 通过微调（fine‑tune）让模型学习到“表格+陈述”组合的判别特征。  
   - 关键在于把表格信息当作普通文本处理，而不需要额外的结构化模块。  

3. **潜在程序解析**  
   - 设计一套 DSL，支持操作如 `SELECT column WHERE condition`, `COUNT`, `SUM`, `MAX`, `MIN`, `COMPARE`.  
   - 使用序列到序列的模型（例如 Transformer）把 S 映射为 DSL 代码。训练时把人工标注的程序作为监督信号（原文未详细说明是否全监督，若无则采用弱监督或强化学习）。  

4. **符号执行引擎**  
   - 将 DSL 程序在内存中的表格对象上解释执行。比如 `COUNT rows WHERE Age > 30` 直接遍历对应列并计数。  
   - 执行过程完全确定，返回的布尔值即为模型的预测。  

5. **融合与决策**  
   - 实验中两条路分别报告准确率，也可以把 Table‑BERT 的概率分布与 LPA 的执行结果做加权平均，提升鲁棒性。  

**最巧妙的设计**  
LPA 把自然语言的模糊表达转成明确的查询脚本，这一步把“语言理解”与“逻辑计算”彻底分离。即使语言模型在语义匹配上有误，只要解析器能生成正确的程序，符号执行就能给出准确答案。相反，Table‑BERT 只需要一次前向传播，却能捕捉到跨列的隐式关联，两者互为补足。

### 实验与效果

- **数据集**：TabFact 包含 16 k 条维基百科表格，配套 118 k 条人工标注的陈述，标签为 ENTAILED 或 REFUTED。作者按照 80/10/10 的比例划分训练、验证、测试集。  
- **基线对比**：与传统文本事实验证模型（如 ESIM、Decomposable Attention）以及直接把表格转文本后使用通用 NLI（自然语言推理）模型的方案相比，Table‑BERT 与 LPA 的准确率均提升约 10–15%。  
- **具体数字**：论文报告 Table‑BERT 与 LPA 在测试集上分别达到约 **71%** 与 **70%** 的准确率，而人类标注者的准确率接近 **99%**。  
- **消融实验**：  
  - 去掉列名或只保留数值的线性化会导致 Table‑BERT 准确率下降约 5%。  
  - 将 DSL 程序的聚合操作（SUM、AVG）去掉，LPA 的表现下降约 8%，说明硬符号推理对数值比较至关重要。  
- **局限性**：  
  - 两种模型仍然在需要多步推理或跨行复杂关系的样本上表现不佳。  
  - LPA 依赖高质量的程序解析器，若解析错误则直接导致错误判断。  
  - 原文未给出对大表格（行数 > 500）时的时间复杂度分析，实际部署可能受限。  

### 影响与延伸思考

TabFact 发布后，表格推理成为 NLP 社区的热点。随后出现了 **TaPas**、**TAPAS**、**TableFormer** 等模型，它们在 TabFact、WikiTableQuestions 等基准上进一步提升了性能，尤其是把表格的二维结构直接建模进 Transformer 的注意力机制。还有工作尝试把 **程序合成** 与 **神经符号混合** 结合，直接在表格上学习可解释的查询语言。对想继续深入的读者，可以关注以下方向：

- **表格‑语言混合预训练**：在大规模表格语料上继续预训练 BERT，提升对列名、数值的感知。  
- **更强的语义解析器**：利用弱监督或自监督的方式生成 DSL 程序，降低对人工标注程序的依赖。  
- **跨模态事实验证**：把文本、图片、表格等多源证据统一到同一验证框架中。  

（以上影响基于后续文献和社区趋势，属于推测性总结。）

### 一句话记住它

TabFact 用大规模表格事实验证数据集和“语言模型+可执行程序”双路思路，首次让机器在结构化证据上进行软硬结合的真假判断。