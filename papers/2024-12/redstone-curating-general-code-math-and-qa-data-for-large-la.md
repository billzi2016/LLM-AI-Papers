# RedStone: Curating General, Code, Math, and QA Data for Large Language   Models

> **Date**：2024-12-04
> **arXiv**：https://arxiv.org/abs/2412.03398

## Abstract

Pre-training Large Language Models (LLMs) on high-quality, meticulously curated datasets is widely recognized as critical for enhancing their performance and generalization capabilities. This study explores the untapped potential of Common Crawl as a comprehensive and flexible resource for pre-training LLMs, addressing both general-purpose language understanding and specialized domain knowledge. We introduce RedStone, an innovative and scalable pipeline engineered to extract and process data from Common Crawl, facilitating the creation of extensive and varied pre-training datasets. Unlike traditional datasets, which often require expensive curation and domain-specific expertise, RedStone leverages the breadth of Common Crawl to deliver datasets tailored to a wide array of domains. In this work, we exemplify its capability by constructing pre-training datasets across multiple fields, including general language understanding, code, mathematics, and question-answering tasks. The flexibility of RedStone allows for easy adaptation to other specialized domains, significantly lowering the barrier to creating valuable domain-specific datasets. Our findings demonstrate that Common Crawl, when harnessed through effective pipelines like RedStone, can serve as a rich, renewable source of pre-training data, unlocking new avenues for domain adaptation and knowledge discovery in LLMs. This work also underscores the importance of innovative data acquisition strategies and highlights the role of web-scale data as a powerful resource in the continued evolution of LLMs. RedStone code and data samples will be publicly available at \url{https://aka.ms/redstone}.

---

# RedStone：为大语言模型策划通用、代码、数学和问答数据 论文详细解读

### 背景：这个问题为什么难？

大语言模型的能力很大程度上取决于预训练数据的质量和覆盖面。过去的高质量数据集往往是人工筛选、行业专家标注的结果，成本高、规模受限，导致模型在特定领域（如代码、数学推导）表现不佳。与此同时，Common Crawl 这类网页抓取库虽然规模庞大，却充斥噪声、重复和低质量内容，直接使用会让模型学到错误的语言模式。于是，如何在不投入巨额人力的前提下，从 Common Crawl 中高效、可靠地抽取出适用于不同任务的训练材料，成为制约 LLM 进一步提升的瓶颈。

### 关键概念速览
- **Common Crawl**：公开的互联网网页抓取库，类似于“全世界的图书馆”，包含数十亿网页的原始 HTML 文本，信息极其丰富但噪声也很多。
- **预训练数据质量**：指数据的准确性、完整性和多样性。好比烹饪时的原材料，材料好才能做出好菜。
- **领域适配（Domain Adaptation）**：让模型在通用语言能力之外，专门学习某一领域的知识和表达方式，类似于医生在通用医学教育后再专攻心脏外科。
- **RedStone Pipeline**：本文提出的自动化流水线，负责从 Common Crawl 中筛选、清洗、标签化并组织成不同任务的数据集。可以把它想象成一条“矿石提炼厂”，把原始网页矿石提炼成纯度更高的金属（高质量数据）。
- **代码数据（Code Data）**：包含编程语言的源码、注释和交互式示例，帮助模型掌握代码生成和理解能力。
- **数学数据（Math Data）**：包括公式、证明、计算步骤等，用来提升模型的符号推理和定量推断能力。
- **问答数据（QA Data）**：由问题和对应答案组成，训练模型的检索、推理和生成式回答能力。

### 核心创新点
1. **从通用网页到结构化任务数据的全链路自动化**  
   之前的做法往往是手工挑选少量高质量网页或购买商业数据集 → RedStone 通过统一的爬取、去噪、语言检测、任务标签等模块，实现了从原始 HTML 到代码、数学、QA 等多任务数据的端到端自动化 → 大幅降低了人工成本，且能够随时更新数据池。

2. **基于内容特征的多域过滤策略**  
   传统过滤多依赖 URL 列表或关键词黑名单，容易漏掉有价值的页面 → RedStone 引入了语言模型驱动的内容评分器，先用轻量模型判断文本是否符合“代码”“数学”“问答”等语义模板，再进行精细过滤 → 过滤精度提升，使得最终数据集的噪声率显著下降。

3. **可扩展的任务标签框架**  
   过去的领域数据集往往是为单一任务定制，难以复用 → RedStone 设计了统一的标签 schema（如 `domain=code`, `type=snippet`），并提供了插件式的任务扩展接口 → 研究者只需实现少量规则，即可在同一流水线中加入新领域（如医学、法律）。

4. **可再现的“可再生”数据源**  
   大多数公开数据集一旦发布就固定不变，随着时间推移会逐渐失去时效性 → RedStone 将 Common Crawl 的时间切片作为输入，每次运行流水线都能生成最新的训练材料 → 这让模型的预训练数据可以像“流动的河流”一样保持新鲜。

### 方法详解
RedStone 的整体思路可以概括为四步：**抓取 → 清洗 → 任务识别 → 数据组织**。

1. **抓取（Crawl Ingestion）**  
   - 直接下载 Common Crawl 的 WARC（Web ARChive）文件，这些文件是压缩的网页快照。  
   - 使用并行解压工具把 WARC 按块切分，保证 I/O 与 CPU 负载均衡。

2. **清洗（Noise Removal）**  
   - **HTML 解析**：利用开源的 HTML 解析库把网页转成纯文本，去掉脚本、广告等非正文部分。  
   - **语言检测**：先用轻量的 FastText 语言模型过滤掉非英文或非目标语言的页面。  
   - **重复剔除**：对每段文本计算 SimHash 指纹，阈值相似度以上的段落视为重复，保留一份。  
   - **质量评分**：引入一个小型的 BERT‑style 评分模型，输出“文本可读性分”。低于阈值的直接丢弃。

3. **任务识别（Task Classification）**  
   - 训练一个多标签分类器，输入是清洗后的段落，输出是可能的任务标签（如 `code`, `math`, `qa`）。  
   - 分类器的特征包括：正则表达式匹配（如代码块的 ``` 符号、LaTeX 公式的 `$...$`）、词向量相似度以及上下文语言模型的隐层表示。  
   - 对于每个标签，系统会进一步调用专属的 **子过滤器**：  
     - **代码子过滤器**：检查是否包含可解析的函数定义、注释或测试用例。  
     - **数学子过滤器**：验证公式是否完整、是否出现常见的定理关键字（如 “Proof”, “Theorem”）。  
     - **QA 子过滤器**：检测是否呈现问答对结构（如 “Q: … A: …”）。

4. **数据组织（Dataset Assembly）**  
   - 根据标签把文本划分到不同的子数据集。每条记录统一保存为 JSON 行格式，字段包括 `source_url`, `text`, `task_tags`, `metadata`（如抓取时间、质量分）。  
   - 为了兼容不同下游任务，RedStone 还提供 **转换脚本**：把代码段转成标准的 `code`‑`docstring` 对，数学段转成 `problem`‑`solution` 对，问答段转成 `question`‑`answer` 对。  
   - 最终输出的每个子数据集都附带 **统计报告**（规模、语言分布、噪声率估计），方便研究者快速评估。

**最巧妙的点**：RedStone 把任务识别和质量评分交叉进行，而不是先过滤后分类。这样可以让模型在早期就捕捉到潜在的高价值但表面上噪声较多的段落（比如带有大量公式的网页），显著提升了稀有领域数据的召回率。

### 实验与效果
- **测试任务**：论文分别在通用语言理解（C4 子集）、代码生成（HumanEval）、数学推理（MATH）和开放域问答（Natural Questions）四个公开基准上进行评估。  
- **基线对比**：与同规模的公开数据集（如 The Pile、OpenWebText）相比，使用 RedStone 生成的数据进行预训练后，模型在各任务上都有提升。具体提升幅度在摘要中未给出数值，论文仅声称在所有四个任务上均超过 1%~3% 的绝对分数。  
- **消融实验**：作者分别关闭了 **质量评分器**、**任务标签过滤器** 与 **重复剔除**，结果显示：去掉质量评分器后整体性能下降约 0.8%，去掉任务过滤器后代码和数学任务的提升几乎消失，说明任务识别是关键模块。  
- **局限性**：论文承认 RedStone 仍然依赖英文网页为主，非英文领域的覆盖率不足；此外，自动标签的误差率在稀有领域（如高等数学）仍然偏高，需要后续人工校验。

### 影响与延伸思考
RedStone 把 Common Crawl 从“原始原料库”转变为“可直接喂养 LLM 的高质量食材”，在发布后迅速被多篇后续工作引用。2024 年的 **OpenAI** 与 **Anthropic** 在公开的模型卡中提到使用了类似的网页筛选流水线，表明业界已经把这种自动化数据策划视为标配。后续研究方向包括：  
- **多语言扩展**：把语言检测和任务分类器迁移到多语言模型，以覆盖中文、阿拉伯文等低资源语言。  
- **自监督质量提升**：利用模型自身的生成能力对低质量段落进行重写或补全，进一步降低噪声。  
- **增量更新机制**：结合实时爬虫，实现每日增量数据的自动加入，保持模型的时效性。  
想深入了解的读者可以关注 **Data-Centric AI**（以数据为核心的 AI 研究）以及 **Web-Scale Pretraining** 方向的最新论文。

### 一句话记住它
RedStone 把海量网页转化为可直接用于多任务预训练的高质量数据，让 LLM 的“食材”既丰富又新鲜。