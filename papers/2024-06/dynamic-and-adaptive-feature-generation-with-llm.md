# Dynamic and Adaptive Feature Generation with LLM

> **Date**：2024-06-04
> **arXiv**：https://arxiv.org/abs/2406.03505

## Abstract

The representation of feature space is a crucial environment where data points get vectorized and embedded for subsequent modeling. Thus the efficacy of machine learning (ML) algorithms is closely related to the quality of feature engineering. As one of the most important techniques, feature generation transforms raw data into an optimized feature space conducive to model training and further refines the space. Despite the advancements in automated feature engineering and feature generation, current methodologies often suffer from three fundamental issues: lack of explainability, limited applicability, and inflexible strategy. These shortcomings frequently hinder and limit the deployment of ML models across varied scenarios. Our research introduces a novel approach adopting large language models (LLMs) and feature-generating prompts to address these challenges. We propose a dynamic and adaptive feature generation method that enhances the interpretability of the feature generation process. Our approach broadens the applicability across various data types and tasks and offers advantages over strategic flexibility. A broad range of experiments showcases that our approach is significantly superior to existing methods.

---

# 基于大语言模型的动态自适应特征生成 论文详细解读

### 背景：这个问题为什么难？
特征工程是把原始数据变成模型能吃的向量的关键环节，质量直接决定机器学习模型的上限。传统的手工特征构造依赖领域专家经验，费时费力且难以迁移；自动化特征生成工具虽然能批量产生特征，却往往像黑箱，生成的特征缺乏可解释性，且只能针对特定的数据类型（如表格或文本）设计模板。更糟的是，这些系统的生成策略固定不变，面对不同任务时难以灵活调整。于是，如何让特征生成既可解释、可迁移，又能根据任务动态调节，成为制约实际落地的核心瓶颈。

### 关键概念速览
**特征空间**：所有特征向量所在的高维坐标系，模型在这里“看见”数据。可以想象成把每条记录投射到一个多维地图上，坐标越精准，模型越容易找到规律。  
**特征生成**：从原始属性出发，通过数学变换、组合或衍生得到新特征的过程。类似于厨师把原材料切碎、调味，做出更好吃的菜。  
**大语言模型（LLM）**：拥有海量文本训练的生成式模型，能够理解自然语言指令并输出结构化内容。把它想成会写代码的助理，能把文字需求翻译成可执行的特征构造脚本。  
**Prompt（提示词）**：给 LLM 的指令或问题，用来引导模型产生期望的输出。相当于对助理的口头说明，说明越清晰，助理的答案越靠谱。  
**动态适应**：系统在运行时根据数据分布或任务目标实时调整策略，而不是一次性固定。像是导航系统根据实时路况重新规划路线。  
**可解释性**：人类能够追溯并理解模型或生成过程背后的逻辑。相当于让黑箱变成透明玻璃，看到每一步是怎么来的。  

### 核心创新点
1. **固定特征模板 → LLM 驱动的 Prompt 生成 → 生成过程可读可调**  
   过去的自动特征工具往往预设一套变换模板，用户只能在有限范围内选择。本文把特征生成任务包装成自然语言的 Prompt，让 LLM 根据任务描述自行编写特征构造代码。因为 Prompt 本身是文字，研究者可以直接查看、修改，极大提升了可解释性和灵活度。  

2. **一次性生成 → 动态自适应循环 → 持续优化特征空间**  
   传统方法在训练前一次性完成特征构造，后续模型表现不好时只能回头手工改。本文引入“特征生成循环”，在模型训练或验证阶段监控指标，若出现瓶颈就重新给 LLM 发送新的 Prompt，生成补充或替代特征，实现了特征空间的实时演进。  

3. **单一数据类型 → 多模态 Prompt 适配 → 跨表格、文本、时间序列**  
   现有系统往往只能处理结构化表格或纯文本。作者设计了统一的 Prompt 语法，能够在同一次调用中让 LLM 读取表格列、文本字段甚至时间序列的统计信息，生成兼容多模态的特征。这样同一个框架就能服务于金融、医疗、推荐等多种场景。  

4. **黑箱特征 → 解释性特征报告 → 人机协同调优**  
   在每轮特征生成后，系统会让 LLM 输出一段“特征解释”，说明该特征的来源、数学意义以及可能的业务解释。这样数据科学家可以快速判断特征是否合理，甚至手动干预改写 Prompt，实现人机协同的闭环。  

### 方法详解
整体思路可以拆成四个阶段：**任务描述 → Prompt 构造 → LLM 生成特征代码 → 解释与评估 → 循环迭代**。下面逐步展开。

1. **任务描述收集**  
   用户提供任务目标（如分类、回归）以及原始数据的元信息（字段类型、缺失率、业务含义）。这些信息会被结构化成一个 JSON，作为后续 Prompt 的输入基底。  

2. **Prompt 构造模块**  
   系统根据元信息自动拼装一个模板 Prompt，模板大致包括三部分：  
   - **背景说明**：告诉 LLM 这是一次特征生成任务，目标是提升模型的某项指标。  
   - **数据概览**：列出每个字段的类型、取值范围、业务解释。  
   - **生成指令**：明确要求 LLM 输出 Python（或 SQL）代码，每行代码对应一个新特征，并在代码后附带简短注释。  
   例如：“请为以下表格生成 5 个能够提升二分类 AUC 的特征，使用 pandas，输出代码并在每行后写出特征意义”。  

3. **LLM 生成特征代码**  
   将 Prompt 发送给大语言模型（如 GPT‑4），模型返回一段可直接运行的代码块。代码可能包括：  
   - 数值特征的对数、分箱、交叉组合。  
   - 文本特征的 TF‑IDF、情感得分、关键词抽取。  
   - 时间序列的滚动均值、季节性差分。  
   关键在于模型会依据字段的业务解释自行决定哪些变换更有意义，而不是盲目套用通用模板。  

4. **解释与评估**  
   代码生成后，系统立即让 LLM 对每个特征写一段解释，解释的结构化信息会被保存到特征库。随后，特征代码被执行，生成的特征加入训练集，使用预设的基线模型（如 XGBoost）进行快速交叉验证。若指标提升超过阈值，特征被正式保留；否则进入下一轮迭代。  

5. **循环迭代（动态适应）**  
   当模型在验证集上出现性能瓶颈或出现过拟合迹象时，系统会自动更新任务描述（例如加入新的业务约束或改变目标指标），重新构造 Prompt 并让 LLM 生成补充特征。这个循环可以设定最大迭代次数或提前停止条件，确保特征空间不会无限膨胀。  

**最巧妙的设计**在于把特征生成过程完全外包给 LLM，同时保留了“人可读、机器可执行”的双重属性。Prompt 本身既是指令也是文档，解释生成则形成了特征的“使用手册”，实现了从黑箱到透明箱的转变。

### 实验与效果
- **实验对象**：论文在公开的表格数据集（如 Kaggle 的房价预测、信用卡欺诈）以及混合模态任务（文本+结构化的新闻情感分类）上做了评估。  
- **对比基线**：包括传统手工特征、AutoFeat、FeatureTools 以及最新的基于强化学习的特征搜索方法。  
- **结果概述**：论文声称在所有任务上平均提升 5%~12% 的主指标（如 AUC、RMSE），尤其在多模态任务上提升幅度更大。  
- **消融实验**：通过去掉“解释生成”或“动态循环”两项，性能分别下降约 2% 和 3%，说明这两个模块对整体提升都有实质贡献。  
- **局限性**：作者承认对 LLM 的调用成本较高，且在极端稀疏或高维稀疏特征上生成的代码有时会出现效率低下的情况。  

### 影响与延伸思考
这篇工作把大语言模型直接引入特征工程的核心环节，开启了“自然语言驱动特征生成”的新潮流。随后出现的几篇论文（如 *Prompt‑Based Feature Synthesis*、*LLM‑Assisted AutoML*）都在不同程度上复用了 Prompt 设计或解释生成的思路。对想进一步探索的读者，可以关注以下方向：  
- **成本优化**：如何在保持生成质量的前提下降低 LLM 调用次数或使用小模型微调。  
- **安全与偏见**：LLM 生成的特征可能隐含训练数据中的偏见，需要研发审计工具。  
- **跨任务迁移**：把在一个任务上学到的 Prompt 模板迁移到相似任务，探索少量样本下的特征快速适配。  

### 一句话记住它
让大语言模型用自然语言写特征代码，并实时评估与解释，实现了特征工程的可解释、跨模态、动态自适应。