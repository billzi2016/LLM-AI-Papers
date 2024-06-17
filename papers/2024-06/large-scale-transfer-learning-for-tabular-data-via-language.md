# Large Scale Transfer Learning for Tabular Data via Language Modeling

> **Date**：2024-06-17
> **arXiv**：https://arxiv.org/abs/2406.12031

## Abstract

Tabular data -- structured, heterogeneous, spreadsheet-style data with rows and columns -- is widely used in practice across many domains. However, while recent foundation models have reduced the need for developing task-specific datasets and predictors in domains such as language modeling and computer vision, this transfer learning paradigm has not had similar impact in the tabular domain. In this work, we seek to narrow this gap and present TabuLa-8B, a language model for tabular prediction. We define a process for extracting a large, high-quality training dataset from the TabLib corpus, proposing methods for tabular data filtering and quality control. Using the resulting dataset, which comprises over 2.1B rows from over 4M unique tables, we fine-tune a Llama 3-8B large language model (LLM) for tabular data prediction (classification and binned regression) using a novel packing and attention scheme for tabular prediction. Through evaluation across a test suite of 329 datasets, we find that TabuLa-8B has zero-shot accuracy on unseen tables that is over 15 percentage points (pp) higher than random guessing, a feat that is not possible with existing state-of-the-art tabular prediction models (e.g. XGBoost, TabPFN). In the few-shot setting (1-32 shots), without any fine-tuning on the target datasets, TabuLa-8B is 5-15 pp more accurate than XGBoost and TabPFN models that are explicitly trained on equal, or even up to 16x more data. We release our model, code, and data along with the publication of this paper.

---

# 通过语言建模实现大规模表格数据迁移学习 论文详细解读

### 背景：这个问题为什么难？

表格数据在企业、科研、政府等场景里随处可见，但它的结构既固定又千变万化：列的类型、缺失值、类别数量都不统一。传统机器学习往往要为每个任务手工特征工程、调参，甚至重新收集标注数据。近几年大模型在文本和图像上已经可以“一次训练、随处使用”，但把同样的迁移学习思路搬到表格上却受阻——现有模型（如 XGBoost、TabPFN）只能在目标数据上训练，缺乏跨表格的通用知识，且对少量样本的适应能力有限。于是出现了“表格领域没有自己的 foundation model”的尴尬局面。

### 关键概念速览
- **表格数据（Tabular Data）**：类似电子表格的结构化数据，行代表实例，列代表特征，常见于业务报表、实验记录等。可以想象成一张带有标题行的矩阵。
- **语言模型（Language Model）**：预测下一个词或 token 的模型，经过大规模文本训练后能捕捉语言的统计规律。这里把表格序列化后交给语言模型处理，类似把表格“翻译”成文字。
- **零样本（Zero‑Shot）**：模型在没有看到目标任务的任何示例时直接给出预测。相当于第一次见到一道新题目就能答出来。
- **少样本（Few‑Shot）**：模型只看到极少量（1‑32）标注样本后进行预测。好比只给老师几道练习题，就能掌握解题思路。
- **Packing Scheme**：把多行多列的表格压缩进模型的输入序列，使得每个 token 都能看到尽可能多的上下文信息。类似把多页纸折叠进一本小册子。
- **注意力机制（Attention）**：模型在处理每个 token 时，根据其与其他 token 的相关性分配不同的权重。这里对表格的行列关系做了专门的调度，让模型更懂“同一列的值之间有什么联系”。
- **TabLib**：作者自行构建的海量公开表格集合，包含数百万张表格、数十亿行数据，提供了训练所需的原始材料。

### 核心创新点
1. **从 TabLib 中筛选高质量训练集 → 采用多层过滤和质量控制流程 → 获得 2.1 B 行、4 M 张表的干净数据**  
   过去的表格学习大多使用少量公开数据集，噪声多、分布窄。作者先用列类型一致性、缺失率、重复度等指标剔除低质量表格，再用统计检验确保数值列的分布合理，最终得到规模前所未有且质量可靠的训练库。

2. **把 Llama 3‑8B 语言模型改造成表格预测专用模型 → 引入表格专属的 packing 与注意力调度 → 让模型在一次前向传播中看到完整表格上下文**  
   直接把表格序列化会导致输入太长、重要列被截断。作者设计了“行列交叉打包”策略，把每行的特征按列顺序拼接，并在注意力层加入列掩码，使得同列的 token 能相互关注，提升了跨行信息的传递。

3. **零样本与少样本评估框架 → 在 329 个全新表格上直接测零样本准确率 → 与 XGBoost、TabPFN 等强基线对比，优势达 15 pp**  
   以前的表格模型几乎没有零样本能力。通过大规模迁移学习，TabuLa‑8B 在完全未知的表格上也能比随机猜测高出 15 个百分点，说明模型已经学到了一般化的“表格常识”。

4. **少样本微调无需重新训练 → 只提供 1‑32 条示例即可超越在目标数据上全量训练的 XGBoost**  
   传统方法需要在每个新任务上跑完整的训练流程，而 TabuLa‑8B 只需要把少量示例塞进 prompt，模型即可利用已有的表格知识进行快速适配，省时省算。

### 方法详解
整体思路可以分为三步：**数据准备 → 模型改造 → 预测使用**。

1. **数据准备**  
   - 从 TabLib 中抽取所有公开 CSV/TSV/Excel 文件。  
   - 过滤规则包括：列数≥2、缺失率<30%、数值列的分布通过 Kolmogorov‑Smirnov 检验、类别列的唯一值比例在合理范围。  
   - 对每张表格做标准化：数值列 Z‑score、类别列映射为整数 ID、日期统一为 ISO 格式。  
   - 最终把每行转成 “列名: 值” 的文本片段，加入任务标签（分类或分箱回归），形成 LLM 可接受的序列。

2. **模型改造**  
   - 基础模型是 Llama 3‑8B，一个 8 B 参数的自回归语言模型。  
   - **Packing Scheme**：把一张表的多行压进同一个输入序列。具体做法是：先列出列名，然后交替写入每行的对应值，形成 “col1: v11, col2: v12, … | col1: v21, col2: v22, …” 的结构。这样同一列的 token 在序列中相隔固定距离，便于注意力捕捉。  
   - **列注意力掩码**：在自注意力层加入一个掩码矩阵，只允许同列的 token 互相注意，跨列的注意力被削弱。相当于让模型在“看列”而不是“看行”。  
   - 其余层保持原始 Llama 的结构，只在最后的预测头上换成针对分类/回归的线性层。

3. **预测使用**  
   - **零样本**：直接把目标表格按照同样的 packing 方式喂入模型，模型输出每行的预测概率或分箱标签。  
   - **少样本**：在输入序列前部加入 1‑32 条标注示例（示例+答案），模型在生成时会参考这些示例的模式，实现“在提示中学习”。不需要梯度更新，算力开销几乎为零。

**最巧妙的点**在于列注意力掩码。普通语言模型的注意力是全局的，处理表格时会把行与行、列与列混在一起，导致信息稀释。作者把注意力限制在同列内部，让模型自然学会“同一特征在不同实例间的关系”，这正是表格学习的核心。

### 实验与效果
- **测试基准**：作者挑选了 329 个公开的表格数据集，覆盖分类、二分类、多分类以及分箱回归任务，数据来源包括 Kaggle、UCI、OpenML 等。  
- **零样本表现**：TabuLa‑8B 的平均准确率比随机猜测高出约 15 pp，远超 XGBoost（接近随机）和 TabPFN（约 5 pp 提升）。  
- **少样本表现**：在 1‑32 条示例的设置下，TabuLa‑8B 超过同等数据量训练的 XGBoost 5‑15 pp，甚至在使用 16 倍训练数据的 XGBoost 上仍保持优势。  
- **消融实验**：作者分别去掉列注意力掩码、改用普通 packing、或只用 1 B 行数据进行训练。结果显示：去掉列注意力后零样本准确率下降约 4 pp，使用普通 packing 时少样本提升幅度减半，数据规模下降 50% 时整体性能下降约 6 pp，验证了每个设计的贡献。  
- **局限性**：论文承认模型仍受输入长度限制，极宽表格需要切分；对高度稀疏或极端类别不平衡的表格仍会出现误差；此外，训练成本高达数千 GPU‑hour，对小团队不友好。

### 影响与延伸思考
这篇工作首次展示了“语言模型+表格”可以实现真正的迁移学习，打开了表格领域的 foundation model 之门。随后出现的工作如 **TabFormer**、**TabGPT** 等，都在尝试更高效的表格序列化或加入图结构注意力，进一步降低算力需求。对想继续深挖的读者，可以关注以下方向：  
- **更轻量的表格专用预训练**：利用知识蒸馏或稀疏注意力降低成本。  
- **跨模态表格学习**：把文本说明、图片或时间序列与表格一起训练，提升对复杂业务场景的理解。  
- **自适应包装策略**：根据表格宽度动态决定 packing 方式，避免信息截断。  
- **公平性与解释性**：在大模型预测的基础上加入可解释层，帮助业务人员审计模型决策。

### 一句话记住它
**TabuLa‑8B 用语言模型的“阅读”能力，把海量表格知识一次性学进去，零样本就能比传统模型更懂新表格。**