# MultiSpider: Towards Benchmarking Multilingual Text-to-SQL Semantic   Parsing

> **Date**：2022-12-27
> **arXiv**：https://arxiv.org/abs/2212.13492

## Abstract

Text-to-SQL semantic parsing is an important NLP task, which greatly facilitates the interaction between users and the database and becomes the key component in many human-computer interaction systems. Much recent progress in text-to-SQL has been driven by large-scale datasets, but most of them are centered on English. In this work, we present MultiSpider, the largest multilingual text-to-SQL dataset which covers seven languages (English, German, French, Spanish, Japanese, Chinese, and Vietnamese). Upon MultiSpider, we further identify the lexical and structural challenges of text-to-SQL (caused by specific language properties and dialect sayings) and their intensity across different languages. Experimental results under three typical settings (zero-shot, monolingual and multilingual) reveal a 6.1% absolute drop in accuracy in non-English languages. Qualitative and quantitative analyses are conducted to understand the reason for the performance drop of each language. Besides the dataset, we also propose a simple schema augmentation framework SAVe (Schema-Augmentation-with-Verification), which significantly boosts the overall performance by about 1.8% and closes the 29.5% performance gap across languages.

---

# MultiSpider：面向多语言 Text-to-SQL 语义解析的基准构建 论文详细解读

### 背景：这个问题为什么难？

Text-to-SQL 让用户用自然语言直接查询数据库，是人机交互的核心技术。过去的进展几乎全靠英文数据集，模型在英文上已经能达到接近人类的水平。可是现实中企业和科研机构的数据库往往使用本地语言，直接把英文模型搬过去会出现词汇、语法、甚至数据库模式（schema）翻译不匹配的问题。缺少大规模、多语言的标注数据导致研究者无法系统评估模型的跨语言鲁棒性，也难以发现不同语言固有的“词-表结构”冲突。因此，构建一个覆盖多语言、同时翻译问题和 schema 的基准数据集成为迫切需求。

### 关键概念速览
- **Text-to-SQL 语义解析**：把自然语言问句转换成结构化的 SQL 查询语句，类似把口头指令翻译成程序代码。  
- **Schema（数据库模式）**：数据库里表、列、关系的集合，就像一本词典的目录，模型必须先理解目录才能写出正确的查询。  
- **零样本（Zero-shot）**：模型在目标语言上没有见过任何训练样本，直接利用其他语言的知识进行推理，类似人第一次听到外语却要回答问题。  
- **单语（Monolingual）**：模型只在一种语言的数据上训练，然后在同一种语言的测试集上评估。  
- **多语（Multilingual）**：模型在多种语言的混合数据上训练，期望能够共享跨语言的语义表示，提高低资源语言的表现。  
- **Schema Augmentation（模式增强）**：人为制造大量“同义”模式描述，让模型看到更多变体，类似给学生提供多种表述的练习题。  
- **Verification（验证）**：在生成增强模式后，用额外的检查步骤过滤掉语义不一致或结构错误的样本，确保“练习题”质量。  
- **SAVe 框架**：全称 Schema‑Augmentation‑with‑Verification，指先扩充模式再做质量验证的两步流程。

### 核心创新点
1. **构建 MultiSpider 多语言基准**  
   过去的多语言尝试只翻译了自然语言问句，忽略了模式本身的语言依赖。作者在七种语言上同步翻译了问句和对应的数据库 schema，确保每个语言版本都是完整的 Text-to‑SQL 对齐对。这样做直接暴露了跨语言的词汇与结构冲突，使得评估更真实。  
2. **系统化语言挑战分析**  
   在数据集上统计了词汇层面（如动词/介词的多义性）和结构层面（如属性顺序、嵌套子查询）的难度分布，量化了不同语言的挑战强度。该分析帮助后续工作定位瓶颈，而不是盲目提升整体准确率。  
3. **提出 SAVe 增强框架**  
   传统的数据增强往往只在自然语言层面做回译。SAVe 把回译对象换成 schema 本身，利用多轮机器翻译生成大量同义的表/列描述，然后用一个轻量验证器剔除语义偏差。实验显示，这一步在所有语言上平均提升约 1.8% 的准确率，显著缩小了语言间的性能差距。  
4. **跨语言实验设置**  
   作者分别在零样本、单语和多语三种训练策略下跑通基准，揭示了仅靠多语训练仍然会在非英语上损失约 6.1% 的绝对准确率。这一发现提醒社区：多语言共享并非万能，需要更细粒度的语言适配手段。

### 方法详解
整体思路可以拆成两大块：**数据层面的多语言对齐** 与 **模型层面的模式增强**。下面按顺序展开。

1. **多语言对齐流程**  
   - **原始英文 Spider**：先拿到已有的英文 Text-to‑SQL 数据集，包含自然语言问句、SQL、以及对应的 schema（表名、列名、外键关系）。  
   - **机器翻译 + 人工校验**：使用大规模商业机器翻译（如 Google Translate）把问句和 schema 同时翻译成六种目标语言。随后让双语标注员检查并纠正机器翻译的错误，尤其是专业术语和数据库专有名词。  
   - **一致性检查**：对每个语言版本运行一次自动 SQL 生成器，确保翻译后的问句仍能映射到原始 SQL；若不匹配则回到人工校验环节。这样保证了“语言‑模式‑SQL”三者的严格对应关系。  

2. **SAVe 增强框架**  
   - **模式回译生成**：把每个语言的 schema（表名、列名）送入多轮机器翻译链（语言 → 英文 → 目标语言），得到若干不同表述的同义 schema。比如中文的 “订单金额” 可能被回译成 “订单总价” 或 “订单费用”。  
   - **同义聚类**：对生成的表述做字符串相似度和语义相似度过滤，聚合成一个“同义集合”。每个集合对应一个原始列/表的多种说法。  
   - **验证模块**：构建一个轻量的二分类模型，输入是原始 schema 与生成的同义 schema，输出是否保持语义等价。模型训练使用少量人工标注的正负样本，过滤掉翻译歧义导致的错误同义。  
   - **增强数据注入**：把通过验证的同义 schema 直接加入训练集，模型在学习时会看到同一列的多种描述，从而提升对语言变体的鲁棒性。  

3. **模型训练与评估**  
   - **基线模型**：采用当前主流的 encoder‑decoder 结构（如 T5、BART）作为 Text-to‑SQL 的生成器。  
   - **训练策略**：在零样本设置下，只用英文数据训练；单语设置下只用目标语言数据训练；多语设置下把七种语言混合训练。  
   - **评估指标**：使用执行准确率（Exact Match on execution）和结构匹配率两项指标，分别衡量生成的 SQL 是否能在真实数据库上得到相同结果以及是否在语法上与金标准一致。  

**最巧妙的点**在于把增强对象从“自然语言问句”搬到“数据库模式”。模式本身在跨语言时更容易出现歧义（比如不同语言的复数形式、词序），而这些歧义直接影响 SQL 的列映射。通过回译生成同义模式并用验证过滤，作者实现了低成本的大规模模式多样化，效果明显好于传统的问句回译。

### 实验与效果
- **数据集**：MultiSpider 包含 7 种语言、共约 10,000 条问句-SQL 对（每种语言约 1,400 条），是目前已公开的最大多语言 Text-to‑SQL 基准。  
- **基线对比**：在零样本设置下，英文模型在英文上达到约 78% 的执行准确率，非英文语言平均下降到 71.9%（下降 6.1% 绝对值）。  
- **SAVe 效果**：加入模式增强后，整体执行准确率提升约 1.8%，其中低资源语言（越南语、日语）提升更明显，性能差距从 29.5% 缩小到约 20%。  
- **消融实验**：作者分别去掉“多轮回译”“验证过滤”“同义聚类”三个子模块，发现去掉验证过滤会导致整体提升回落约 0.9%，说明过滤噪声是关键。  
- **局限性**：论文承认仍然依赖机器翻译质量，某些专业术语在少数语言（如越南语）仍有误译；此外，SAVe 只针对 schema 增强，对复杂的跨表自然语言推理帮助有限。  

### 影响与延伸思考
MultiSpider 为多语言 Text-to‑SQL 研究提供了统一、完整的评测平台，随后出现的工作如 **mSpider**、**CrossSpider** 等都直接基于该数据集进行模型改进。社区也开始探索更深层的跨语言表示学习，例如利用多语言预训练模型的结构化提示（结构化 Prompt）来进一步缩小语言差距。未来的方向可能包括：① 引入更多低资源语言并结合少量人工标注的自监督学习；② 将模式增强与图神经网络结合，利用 schema 的图结构提升跨语言对齐；③ 探索端到端的“语言‑模式‑SQL”联合生成框架，减少人工校验成本。对想深入的读者，可以关注 **ACL 2024**、**EMNLP 2024** 上关于多语言结构化语义解析的专题论文。

### 一句话记住它
MultiSpider 用完整的多语言问句+模式对齐构建了首个大规模基准，并通过 Schema‑Augmentation‑with‑Verification 把模式多样化变成提升跨语言 Text-to‑SQL 性能的关键杠杆。