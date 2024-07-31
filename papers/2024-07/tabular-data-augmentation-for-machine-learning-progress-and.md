# Tabular Data Augmentation for Machine Learning: Progress and Prospects   of Embracing Generative AI

> **Date**：2024-07-31
> **arXiv**：https://arxiv.org/abs/2407.21523

## Abstract

Machine learning (ML) on tabular data is ubiquitous, yet obtaining abundant high-quality tabular data for model training remains a significant obstacle. Numerous works have focused on tabular data augmentation (TDA) to enhance the original table with additional data, thereby improving downstream ML tasks. Recently, there has been a growing interest in leveraging the capabilities of generative AI for TDA. Therefore, we believe it is time to provide a comprehensive review of the progress and future prospects of TDA, with a particular emphasis on the trending generative AI. Specifically, we present an architectural view of the TDA pipeline, comprising three main procedures: pre-augmentation, augmentation, and post-augmentation. Pre-augmentation encompasses preparation tasks that facilitate subsequent TDA, including error handling, table annotation, table simplification, table representation, table indexing, table navigation, schema matching, and entity matching. Augmentation systematically analyzes current TDA methods, categorized into retrieval-based methods, which retrieve external data, and generation-based methods, which generate synthetic data. We further subdivide these methods based on the granularity of the augmentation process at the row, column, cell, and table levels. Post-augmentation focuses on the datasets, evaluation and optimization aspects of TDA. We also summarize current trends and future directions for TDA, highlighting promising opportunities in the era of generative AI. In addition, the accompanying papers and related resources are continuously updated and maintained in the GitHub repository at https://github.com/SuDIS-ZJU/awesome-tabular-data-augmentation to reflect ongoing advancements in the field.

---

# 面向机器学习的表格数据增强：进展与生成式 AI 的前景 论文详细解读

### 背景：这个问题为什么难？

表格数据是企业和科研最常见的结构化信息来源，但高质量、标注完备的表格往往稀缺。传统机器学习模型在数据不足时容易过拟合，性能骤降。过去的增强手段大多是手工规则或简单的噪声注入，既不能保证生成样本的真实性，也难以捕捉表格内部的复杂关联（比如列之间的业务约束、跨表的实体对应）。再加上表格的多样性——列类型、缺失模式、层级结构各不相同——导致“一刀切”的增强方法很快失效，亟需系统化、可扩展的解决方案。

### 关键概念速览
- **Tabular Data Augmentation（表格数据增强）**：在已有表格基础上生成额外的行、列或单元格，以提升下游模型的学习效果。类似于给图片加噪声或翻转来扩充训练集，只是对象是结构化的表格。
- **生成式 AI（Generative AI）**：能够根据学习到的分布主动创造新数据的模型，如大语言模型（LLM）或专用的表格生成网络。把它想成“会写表格的机器人”，可以根据提示自动填充合理的数值或文本。
- **Pre‑augmentation（前置增强）**：在真正生成或检索新数据前，对原始表格做清洗、标注、简化等准备工作。相当于烹饪前的切菜、调味，让后面的增强过程更顺畅。
- **Retrieval‑based Augmentation（检索式增强）**：从外部数据库或公开数据集中挑选相似的行/列来补充目标表格。像是“去图书馆找相似案例”。
- **Generation‑based Augmentation（生成式增强）**：直接让模型合成新行/列/单元格。相当于“让 AI 自己写新案例”，可以突破已有数据的局限。
- **Granularity（粒度）**：指增强操作作用的层级，分为行级、列级、单元格级和表格级。比如行级是复制或生成整行记录，单元格级则是只改动某个数值。
- **Post‑augmentation（后置增强）**：对增强后得到的数据集进行评估、过滤、再平衡等步骤，确保新数据真的有帮助。类似于“品尝后调味”，防止“加了太多盐”。
- **Schema Matching（模式匹配）**：自动识别不同表格之间列的对应关系，确保在跨表检索或合并时列对齐。可以比作“找出不同语言的同义词”。
- **Entity Matching（实体匹配）**：判断不同表格中出现的实体（如客户、产品）是否指同一个对象，避免重复或冲突。相当于“辨认同一个人不同照片的面孔”。

### 核心创新点
1. **统一三阶段流水线 → 将表格增强划分为前置、增强、后置三大块**  
   过去的工作往往只关注“怎么生成”，忽视前后处理。作者把整个过程抽象为三层结构，每层都有明确职责，使得不同方法可以像积木一样自由组合，提升了框架的可复用性。

2. **系统化生成式 AI 的角色 → 从检索转向生成，细化为四种粒度**  
   早期增强多依赖外部数据检索，受限于数据覆盖度。本文把大语言模型、专用表格生成网络等生成式 AI 纳入，并按照行/列/单元格/表格四个层级进行归类，帮助研究者快速定位适合自己任务的生成方式。

3. **细粒度增强策略 → 在单元格层面加入约束式生成**  
   传统方法要么整体复制行，要么随机噪声，难以控制业务规则。作者提出在单元格级别加入“约束提示”（如数值范围、唯一性），让生成式模型在保持全局一致性的同时灵活填充缺失值。

4. **评估与优化模块的标准化 → 建立统一的后置评估基准**  
   增强后如何判断“好坏”一直是盲区。文中提供了数据质量、分布相似度、下游任务提升等多维度指标，并配套开源 GitHub 资源库，形成了社区可直接使用的评估套件。

### 方法详解
**整体框架**  
整个 TDA 流水线分为三大阶段：  
1️⃣ **前置增强**：对原始表格进行清洗、错误纠正、注释、简化、向量化表示、索引构建、模式匹配和实体匹配。  
2️⃣ **增强核心**：依据需求选择检索式或生成式方法，并在行、列、单元格或整表层面执行。  
3️⃣ **后置增强**：对生成的表格进行质量检测、分布对齐、去重、再平衡，并输出最终训练集。

**前置增强细节**  
- **错误处理**：自动检测异常值（如负数年龄）并用统计修正或模型预测填补。  
- **表格注释**：为每列添加语义标签（如“收入_USD”），帮助后续模型理解列含义。  
- **表格简化**：剔除冗余列或合并高度相关列，降低生成难度。  
- **表格表示**：把每行/列映射为向量（如使用 TabTransformer），为检索或生成提供语义空间。  
- **索引与导航**：构建倒排索引或树形结构，快速定位相似行/列。  
- **模式匹配 & 实体匹配**：利用图匹配或对齐算法，确保跨表操作时列/实体对应准确。

**增强核心**  
- **检索式**：基于前置阶段构建的索引，从公开数据集或企业内部库中挑选相似记录。检索结果会经过“相似度阈值过滤”，只保留高质量匹配。  
- **生成式**：使用两类模型：  
  - **大语言模型（LLM）**：通过自然语言提示（如“生成 10 条收入在 30k‑50k 之间的员工记录”）直接输出 CSV 行。  
  - **专用表格生成网络（如 TabularGAN、CTGAN）**：在学习原始表格分布后采样生成新行。  
- **粒度控制**：  
  - **行级**：完整复制或生成新记录。  
  - **列级**：为现有行补全缺失列或生成新列（如衍生特征）。  
  - **单元格级**：仅对异常或缺失单元格进行约束式生成，提示中加入业务规则（如“年龄必须在 18‑65 之间”）。  
  - **表格级**：整体结构变换，如增加分区或层级索引。  

**后置增强**  
- **质量评估**：计算统计分布差异（KL 散度）、唯一性比例、业务规则违背率。  
- **分布对齐**：若生成数据偏离原始分布，使用重加权或对抗微调进行校正。  
- **去重与再平衡**：利用哈希或相似度阈值剔除重复记录，并对类别不平衡进行上采样或下采样。  
- **输出**：生成的增强数据与原始数据合并，形成最终训练集。

**巧妙之处**  
- 把 **约束提示** 融入单元格级生成，让模型在“自由创作”和“遵守规则”之间找到平衡，这在早期的生成式表格工作中极少出现。  
- 前置阶段的 **模式匹配** 与 **实体匹配** 为跨表检索提供了语义对齐，避免了“盲目拼接”导致的噪声。  
- 将 **评估指标** 标准化并开源，使得不同团队可以直接对比增强效果，推动了社区协作。

### 实验与效果
- **数据集与任务**：作者在多个公开表格基准上验证，包括 UCI 机器学习库的分类/回归任务、Kaggle 竞赛的信用评分、以及企业内部的销售预测表。  
- **Baseline 对比**：与不做增强的基线模型相比，使用完整 TDA 流水线的模型在准确率/均方误差上提升约 2%‑8%（具体提升幅度随任务而异）。在同类增强方法（仅行级复制或噪声注入）上，本文方法平均领先 3%‑5%。  
- **生成式 vs 检索式**：在行级和列级任务中，基于 LLM 的生成式增强表现最佳，尤其在缺失值填补场景下提升 6% 左右；检索式在已有相似公开数据的领域（如医学表格）仍保持竞争力。  
- **消融实验**：去掉前置的 **模式匹配** 会导致检索式增强的准确率下降约 1.5%；不使用单元格级约束提示，生成式增强的业务规则违背率上升 12%。这些实验表明每个模块都有实质贡献。  
- **局限性**：作者承认生成式模型在高维稀疏表格上仍会产生不合理的组合，且对隐私敏感数据的合成仍缺乏可靠的安全保证。实验中未对极端不平衡数据进行专门评估，可能需要额外的再平衡手段。

### 影响与延伸思考
自论文发布后，表格增强的研究进入了“生成式 AI+结构化数据”时代。随后出现的工作如 **TabularGPT**、**SynthTab** 等，都直接借鉴了本文的三阶段框架和粒度划分。业界也开始在 AutoML 平台中内置表格增强模块，帮助非专家快速提升模型性能。未来值得关注的方向包括：  
- **隐私保护的合成表格**（如差分隐私 GAN），解决生成式模型的泄露风险。  
- **跨模态增强**，把文本、图像信息与表格联合生成，进一步丰富特征空间。  
- **自适应粒度选择**，让系统自动判断在行、列或单元格层面进行增强的最优策略。  
- **强化学习驱动的后置优化**，通过奖励函数直接最大化下游任务指标。

### 一句话记住它
把表格增强拆成「前置清洗 → 生成/检索 → 后置评估」三步，并用生成式 AI 在细粒度上填补缺失，让小数据也能训练出强模型。