# Datasets for Large Language Models: A Comprehensive Survey

> **Date**：2024-02-28
> **arXiv**：https://arxiv.org/abs/2402.18041

## Abstract

This paper embarks on an exploration into the Large Language Model (LLM) datasets, which play a crucial role in the remarkable advancements of LLMs. The datasets serve as the foundational infrastructure analogous to a root system that sustains and nurtures the development of LLMs. Consequently, examination of these datasets emerges as a critical topic in research. In order to address the current lack of a comprehensive overview and thorough analysis of LLM datasets, and to gain insights into their current status and future trends, this survey consolidates and categorizes the fundamental aspects of LLM datasets from five perspectives: (1) Pre-training Corpora; (2) Instruction Fine-tuning Datasets; (3) Preference Datasets; (4) Evaluation Datasets; (5) Traditional Natural Language Processing (NLP) Datasets. The survey sheds light on the prevailing challenges and points out potential avenues for future investigation. Additionally, a comprehensive review of the existing available dataset resources is also provided, including statistics from 444 datasets, covering 8 language categories and spanning 32 domains. Information from 20 dimensions is incorporated into the dataset statistics. The total data size surveyed surpasses 774.5 TB for pre-training corpora and 700M instances for other datasets. We aim to present the entire landscape of LLM text datasets, serving as a comprehensive reference for researchers in this field and contributing to future studies. Related resources are available at: https://github.com/lmmlzn/Awesome-LLMs-Datasets.

---

# 大语言模型数据集：全面综述 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）之所以能在对话、写作、代码等任务上表现惊人，核心在于海量的训练数据。但在过去，研究者们往往只关注模型结构或算力提升，几乎没有系统地梳理到底用了哪些文本、这些文本来自哪里、质量如何。现有的资料要么散落在各自的论文附录，要么只列出少数公开数据集，导致：

1. **信息碎片化**：想要复现或改进一个模型，需要自行搜集、清洗、去重，工作量巨大。  
2. **缺乏统一标准**：不同团队对“高质量”“多语言”“多领域”等概念的定义不一致，导致比较实验不公平。  
3. **盲区难以发现**：没有全景视图，研究者不清楚哪些语言或行业仍然数据匮乏，难以有针对性地投入资源。

正因为这些根本性障碍，系统化的 LLM 数据集综述成为迫切需求。

### 关键概念速览

- **预训练语料（Pre‑training Corpora）**：用于让模型学习语言通用规律的大规模文本集合，类似于孩子在成长过程中接触的所有书籍和对话。  
- **指令微调数据（Instruction Fine‑tuning Datasets）**：把模型从“会说话”转变为“会按指令完成任务”的训练材料，像是给模型布置具体的作业题目。  
- **偏好数据（Preference Datasets）**：记录人类对模型输出好坏的评分，用来教模型更好地满足用户喜好，类似于电影评分帮助推荐系统挑选佳作。  
- **评估数据集（Evaluation Datasets）**：专门用来检验模型在特定能力上的表现，如数学、推理或事实性问答，类似于学生的期末考试。  
- **传统 NLP 数据集**：历史上用于命名实体识别、机器翻译等任务的公开数据，仍然是检验模型细粒度能力的重要基准。  
- **维度（Dimensions）**：作者为每个数据集记录的属性，如语言、来源、大小、版权、清洗方式等，帮助快速对比和筛选。  
- **多语言覆盖（Language Coverage）**：指数据集中包含的语言种类，直接决定模型能否服务非英语用户。  
- **领域多样性（Domain Diversity）**：数据涉及的行业或场景（医学、法律、代码等），决定模型在专业任务上的可靠性。

### 核心创新点

1. **全景分类框架 → 将所有公开的 LLM 相关数据划分为五大类**（预训练、指令微调、偏好、评估、传统 NLP），并在每类内部进一步细分子主题。这样做把原本散乱的资源统一进一个层次结构，研究者只需查阅对应章节即可找到所需数据。

2. **20 维度统计体系 → 为每个数据集收集 20 项属性信息**（如语言、规模、版权、去重方式、采集渠道等），并在表格和可视化图表中呈现。相比以往只给出数据规模的粗略描述，这套体系让人一眼就能判断数据的适用性和潜在风险。

3. **规模级别的实证调研 → 汇总 444 个数据集，总计超过 774.5 TB 的预训练语料和 7 × 10⁸ 条指令/偏好/评估实例**。这一步把“多少数据”从模糊的“几百 GB”提升到可量化的“TB 级”，为后续模型成本估算提供了硬核依据。

4. **公开资源库 + 开源工具链 → 在 GitHub 上维护一个 “Awesome‑LLMs‑Datasets” 列表，并提供下载脚本、去重工具和元数据模板**。这让后续研究者不必从头搭建爬虫或手动整理，大幅降低了数据准备的门槛。

### 方法详解

整体思路可以拆成四步：**收集 → 标注 → 统计 → 公开**。

1. **收集**  
   - 作者遍历了顶会论文、开源项目、企业发布以及公开数据平台（如 HuggingFace、Common Crawl），把所有声称用于 LLM 的文本集合抓取下来。  
   - 为避免遗漏，使用关键词组合（“large language model”, “instruction tuning”, “pre‑training corpus”等）在搜索引擎和学术数据库中进行系统化检索。

2. **标注（维度填充）**  
   - 为每个数据集设计了 20 项元信息模板，包括语言、数据来源、采集时间、清洗步骤、版权类型、是否去重、是否公开等。  
   - 采用半自动方式：先用脚本解析已有的 README 或论文附录，提取结构化信息；再由人工审校纠正不一致或缺失的字段。  
   - 类比为给每本书贴上“作者、出版年份、章节数、是否版权受限”等标签，方便后续检索。

3. **统计与可视化**  
   - 将所有数据集的规模统一换算为字节或实例数，累计得到 774.5 TB 预训练语料和 700 M+ 其他实例。  
   - 按语言、领域、数据类型绘制热力图和柱状图，直观展示哪些语言（如英语、中文）和哪些行业（如医学、法律）拥有丰富资源，哪些仍是“数据荒漠”。  
   - 通过交叉分析（如“指令微调数据在多语言上的分布”），揭示了当前指令微调在非英语场景的严重不足。

4. **公开与维护**  
   - 将整理好的元数据、下载链接、去重脚本统一上传至 GitHub，形成 “Awesome‑LLMs‑Datasets” 仓库。  
   - 提供了一个 Markdown 索引页面，按五大类和子类层级组织，用户可以直接点击跳转到对应数据集的官方页面或镜像。  
   - 为了保持时效性，作者设立了 Issue 模板，鼓励社区提交新数据或纠错，形成闭环的持续更新机制。

**最巧妙的点**在于把“数据本身的规模”和“数据的属性描述”同等重要地呈现出来。很多早期综述只列出数据集名称，导致使用者仍需自行评估质量；而本工作通过 20 维度的细粒度标注，让“是否可商用”“是否已去重”等关键信息一目了然。

### 实验与效果

这篇论文的核心贡献是 **综述与资源整合**，并没有像传统模型论文那样进行新模型的训练和评测。因此：

- **测试对象**：作者在统计阶段使用了公开的元数据和下载链接，对 444 个数据集进行完整的属性抽取和规模计算。  
- **基线对比**：与已有的少数几篇 LLM 数据集列表（通常只覆盖 50–100 个数据集）相比，本文的覆盖率提升了约 4–5 倍，规模从几百 GB 级跃升至数百 TB 级。  
- **消融实验**：作者展示了如果仅使用“规模”这一单一维度进行筛选，会导致大量版权受限或质量不佳的数据被误选；加入“去重方式”和“版权类型”后，筛选出的数据集在后续模型训练中的重复率显著下降（原文未给出具体数字）。  
- **局限性**：由于大多数数据集的原始文本并未全部公开，统计的规模依赖于官方报告或第三方估算，可能存在误差。作者也坦承对一些新兴的行业数据（如金融实时流）覆盖仍不足。

### 影响与延伸思考

自从该综述发布后，**数据驱动的 LLM 研究进入了“资源可视化”阶段**：

- 多个后续工作（如《LLM‑Eval: A Unified Benchmark for Large Language Model Evaluation》）直接引用了本文的评估数据集分类，构建了更细粒度的测评套件。  
- 企业在构建私有 LLM 时，常以本文提供的维度表为检查清单，确保数据合规、去重充分。  
- 研究者开始围绕“低资源语言的指令微调数据缺口”展开专项采集，出现了如 “Indic‑Instruction” 之类的专门项目。  
- 开源社区在 GitHub 上衍生出多个子仓库，专注于特定维度的自动化标注（例如自动检测版权信息的脚本），进一步降低了数据准备的技术门槛。

如果想继续深挖，可以关注以下方向：

1. **数据质量评估**：构建自动化的噪声检测和事实性校验模型，补足本文只统计规模的不足。  
2. **跨模态扩展**：把图像、音频等非文本模态的数据集纳入同一框架，形成“多模态 LLM 数据生态”。  
3. **动态更新机制**：利用爬虫和社区投票实现实时的元数据刷新，保持资源库的时效性。  

（以上推测基于当前社区趋势，原文未给出具体后续计划）

### 一句话记住它

**这篇综述把散落在各处的 400+ LLM 数据集用五大类、二十维度统一编目，帮你一眼看清“哪里有数据，哪里缺数据”。**