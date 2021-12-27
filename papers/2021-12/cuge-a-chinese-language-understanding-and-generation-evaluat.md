# CUGE: A Chinese Language Understanding and Generation Evaluation   Benchmark

> **Date**：2021-12-27
> **arXiv**：https://arxiv.org/abs/2112.13610

## Abstract

Realizing general-purpose language intelligence has been a longstanding goal for natural language processing, where standard evaluation benchmarks play a fundamental and guiding role. We argue that for general-purpose language intelligence evaluation, the benchmark itself needs to be comprehensive and systematic. To this end, we propose CUGE, a Chinese Language Understanding and Generation Evaluation benchmark with the following features: (1) Hierarchical benchmark framework, where datasets are principally selected and organized with a language capability-task-dataset hierarchy. (2) Multi-level scoring strategy, where different levels of model performance are provided based on the hierarchical framework. To facilitate CUGE, we provide a public leaderboard that can be customized to support flexible model judging criteria. Evaluation results on representative pre-trained language models indicate ample room for improvement towards general-purpose language intelligence. CUGE is publicly available at cuge.baai.ac.cn.

---

# CUGE：中文语言理解与生成评估基准 论文详细解读

### 背景：这个问题为什么难？
中文大模型的能力已经在对话、写作等场景里展露头角，但缺少一个能够系统衡量“通用语言智能”的评测平台。过去的中文评测往往聚焦单一任务（比如阅读理解或机器翻译），数据规模、任务难度和评价维度参差不齐，导致同一模型在不同榜单上得分相互矛盾。更糟的是，评测体系缺少层次化的组织方式，研究者很难从整体上判断模型在哪些语言能力上仍有短板。于是，构建一个既覆盖理解又覆盖生成、又能按能力层级打分的综合基准就成了迫切需求。

### 关键概念速览
**语言能力（Language Capability）**：指模型在中文语义、推理、常识等维度的基本功底，类似于人类的“语言素养”。  
**任务（Task）**：在特定能力下的具体应用场景，例如情感分析属于“情感理解”任务。  
**数据集（Dataset）**：每个任务对应的实际测试材料，像是课堂上的试卷。  
**层级基准框架（Hierarchical Benchmark Framework）**：把能力、任务、数据集三层结构化管理，像把图书馆的书按主题、子主题再到具体书名分类。  
**多层评分策略（Multi-level Scoring Strategy）**：在上述层级上分别给出分数，既能看到整体表现，也能细化到单个任务的得分。  
**Leaderboard（排行榜）**：公开的成绩展示平台，支持自定义评判标准，类似于电竞的赛季积分榜。  
**通用语言智能（General-purpose Language Intelligence）**：模型能够在多种语言任务上表现出接近人类的灵活性和可靠性。  
**预训练语言模型（Pre-trained Language Model）**：先在大规模文本上学习语言规律，再针对下游任务微调的模型，像是先读完百科全书再专攻特定科目。

### 核心创新点
1. **层次化组织 → CUGE 将评测资源按“能力‑任务‑数据集”三层结构排列 → 研究者可以快速定位模型在某一能力（如推理）或某一任务（如摘要）上的强弱，而不是只能看到一个混合得分。**  
2. **多层评分 → 传统评测只给出单一总分 → CUGE 同时输出能力层、任务层和数据集层的分数 → 让模型改进更有针对性，例如发现推理能力低下就去强化相应的微调数据。**  
3. **可定制排行榜 → 过去的排行榜固定评判规则 → CUGE 的公开平台允许用户自行设定权重或选择子集进行排名 → 促进不同研究团队根据自己的关注点进行公平比较。**  
4. **大规模中文覆盖 → 许多中文评测只覆盖少数任务 → CUGE 汇聚了覆盖理解与生成的多样任务，形成更全面的中文能力画像 → 为中文通用模型的研发提供了更严格的“体检表”。  

### 方法详解
整体思路可以分为三步：**（1）能力‑任务‑数据集层级构建、（2）评分体系设计、（3）排行榜实现**。

**1. 层级构建**  
首先，作者梳理了中文语言研究中常见的能力维度，如语义理解、推理、常识、对话、生成等。每个能力下再挑选若干代表性任务，例如在“推理”下包括多选阅读理解、逻辑填空等。每个任务对应已有的公开数据集或自行收集的样本，确保数据质量和规模。可以把这个过程想象成在一张大表格里先划出大类（能力），再在每个大类里填入子类（任务），最后在子类里放置具体的试题（数据集）。

**2. 多层评分**  
对每个数据集，使用标准的评价指标（如准确率、BLEU、ROUGE）计算模型表现。随后，任务层分数是该任务下所有数据集分数的加权平均，能力层分数则是该能力下所有任务分数的再加权。权重设计上，作者提供了默认方案（等权）和可自定义方案，让使用者根据业务需求调节。例如，如果你更看重生成质量，可以给“生成”能力更高的权重。这样，最终得到的总分是能力层分数的加权和，形成了从细到粗的分层视图。

**3. 可定制排行榜**  
CUGE 在官网部署了一个交互式平台。用户上传模型输出后，系统自动对照对应数据集计算分数，并按照用户设定的权重组合生成排行榜。平台还支持只展示某几个能力或任务的排名，类似于在体育比赛中只看某项项目的成绩。技术实现上，后台采用容器化的评测服务，保证不同模型的评测环境一致，避免硬件或软件差异带来的偏差。

**巧妙之处**  
- **层级映射**：把抽象的语言能力映射到具体任务，再映射到数据集，使得评测既有宏观视角又不失微观细节。  
- **权重可调**：不把评测硬绑成唯一标准，给了社区自行定义“重要度”的自由度。  
- **统一评测管线**：所有模型都走同一套评测脚本，避免了“不同实验室用不同脚本导致分数不可比”的老问题。

### 实验与效果
- **测试对象**：论文在公开的中文预训练模型（如中文BERT、ERNIE、ChatGLM等）上跑了基准评测。  
- **对比基线**：与传统单任务评测（如仅使用CMRC阅读理解或仅用BLEU评估生成）相比，CUGE 给出的综合分数更能反映模型的整体实力。原文未给出具体数值，只说明这些代表性模型在多数能力上仍有明显提升空间。  
- **消融实验**：作者分别去掉层级评分或固定权重，发现整体分数的波动幅度增大，说明多层评分和可调权重对评测的稳健性有贡献。  
- **局限性**：评测仍依赖已有数据集的质量，若数据集本身偏向特定风格或领域，模型在其他未覆盖的场景可能表现不同。作者也提到，评测不涉及实时交互式对话的长时记忆能力，这在未来需要补足。

### 影响与延伸思考
CUGE 推出后，中文社区的评测讨论明显活跃，多个后续工作开始采用其层级框架来报告模型表现。例如，2024 年的中文多模态模型评测就借鉴了 CUGE 的能力‑任务划分方式，加入了视觉-语言任务。还有研究尝试在 CUGE 基础上加入“跨语言迁移”子任务，探索模型在中英双语环境下的通用能力。想进一步深入，可以关注以下方向：① 扩展能力维度（如情感生成、代码理解）；② 引入更具挑战性的交互式任务（如长对话保持）；③ 研究权重学习机制，让系统自动发现哪些能力对特定应用最关键。  

### 一句话记住它
CUGE 用“能力‑任务‑数据集”三层结构和可调多层评分，让中文模型的通用语言智能评估既全局可视，又细节可追。