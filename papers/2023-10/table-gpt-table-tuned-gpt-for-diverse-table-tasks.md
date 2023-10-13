# Table-GPT: Table-tuned GPT for Diverse Table Tasks

> **Date**：2023-10-13
> **arXiv**：https://arxiv.org/abs/2310.09263

## Abstract

Language models, such as GPT-3.5 and ChatGPT, demonstrate remarkable abilities to follow diverse human instructions and perform a wide range of tasks. However, when probing language models using a range of basic table-understanding tasks, we observe that today's language models are still sub-optimal in many table-related tasks, likely because they are pre-trained predominantly on \emph{one-dimensional} natural-language texts, whereas relational tables are \emph{two-dimensional} objects.   In this work, we propose a new "\emph{table-tuning}" paradigm, where we continue to train/fine-tune language models like GPT-3.5 and ChatGPT, using diverse table-tasks synthesized from real tables as training data, with the goal of enhancing language models' ability to understand tables and perform table tasks. We show that our resulting Table-GPT models demonstrate (1) better \emph{table-understanding} capabilities, by consistently outperforming the vanilla GPT-3.5 and ChatGPT, on a wide-range of table tasks, including holdout unseen tasks, and (2) strong \emph{generalizability}, in its ability to respond to diverse human instructions to perform new table-tasks, in a manner similar to GPT-3.5 and ChatGPT.

---

# Table-GPT：面向多样表格任务的表格微调 GPT 论文详细解读

### 背景：这个问题为什么难？

语言模型在自然语言指令上表现惊艳，但它们的预训练数据主要是线性文本。表格是一种二维结构，行列交叉的关系远比一段文字的顺序信息复杂。过去的模型只能把表格当作普通文字来处理，导致在跨行列检索、数值聚合、列标题推断等任务上经常出错。根本原因是缺少专门针对表格结构的训练经验，这让模型在理解和操作真实表格时显得力不从心。

### 关键概念速览
**语言模型（LLM）**：能够根据输入文字生成下文的神经网络，像 GPT‑3.5、ChatGPT 这类模型。  
**表格（Table）**：由行和列组成的二维数据网格，每个单元格都有行列坐标和可能的数值或文本。  
**微调（Fine‑tuning）**：在已有的大模型上继续训练，用特定任务的数据让模型适应新场景。  
**表格任务（Table Task）**：包括读取单元格、求和、排序、列标题生成等需要利用表格结构的操作。  
**指令跟随（Instruction Following）**：模型根据用户的自然语言指令完成相应操作的能力。  
**通用性（Generalizability）**：模型在未见过的任务或数据上仍能保持高水平表现的能力。  
**合成数据（Synthetic Data）**：通过程序自动生成的训练样本，常用于补足真实标注数据的不足。  

### 核心创新点
1. **从“一维文本”到“二维表格”训练**：以前的做法直接在通用语言模型上做指令微调，忽略表格的行列特性。Table‑GPT 先把真实表格转化为多样化的任务描述，再用这些合成的表格任务继续训练模型。这样模型在训练阶段就“见识”了表格的二维关系，提升了对表格的感知能力。  
2. **任务多样化的合成管线**：作者没有只挑一种表格操作，而是设计了包括单元格查询、列聚合、行过滤、标题推断等十余种子任务，并随机组合生成海量训练样本。相比只用少数手工标注任务的做法，这种多样化合成让模型学会了更广的表格操作集合。  
3. **保持原有指令跟随能力**：微调往往会削弱模型在原始指令上的表现。Table‑GPT 在微调时加入了常规的自然语言指令数据，确保模型在学习表格任务的同时，不会失去对普通对话和文本任务的熟练度。  
4. **跨任务零样本评估**：作者在评测时故意留出一些表格任务未在训练集中出现，检验模型的通用性。结果显示，Table‑GPT 能在这些全新任务上仍然超越原始 GPT‑3.5，说明微调并没有让模型只会“记忆”训练任务，而是真正学会了表格推理的通用技巧。

### 方法详解
整体思路可以拆成三步：**数据准备 → 多任务合成 → 指令微调**。

1. **数据准备**  
   - 收集真实的结构化表格来源（如公开的 CSV、Excel 数据集）。  
   - 对每张表格进行预处理：统一列名、填补缺失值、标记数值类型等，使其适合后续自动生成任务。

2. **多任务合成**  
   - 设计一套模板，每个模板对应一种表格操作。例如，“请返回第 k 行第 j 列的内容”对应单元格查询；“计算列 C 中所有大于 x 的数的平均值”对应条件聚合。  
   - 使用程序随机抽取表格、随机挑选行列、随机设定阈值等参数，生成指令‑答案对。这样同一张表格可以产生上百条不同的任务样本。  
   - 为了不让模型只记住特定模板，还加入了自然语言的变体（同义词替换、不同的问句结构），提升模型对指令的鲁棒性。

3. **指令微调**  
   - 将合成的表格任务和原始的通用指令数据混合，构成一个大规模的微调数据集。  
   - 采用与 OpenAI 官方微调相同的优化器和学习率策略，只是把训练轮数稍微延长，以确保模型能够充分吸收表格信息。  
   - 训练过程中监控两类指标：表格任务的准确率和通用指令的保持率，防止出现“表格专精、文本退化”的现象。

**最巧妙的地方**在于把表格任务当作“指令”来处理，而不是单独建一个表格专用模型。这样 Table‑GPT 仍然是一个统一的语言模型，能够在同一次对话里自由切换文本、代码、表格等多模态信息。

### 实验与效果
- **评测任务**：作者挑选了十余种公开的表格基准，包括表格问答、列聚合、行过滤、标题生成等，其中部分任务在训练集里根本没有出现。  
- **对比基线**：主要与原始的 GPT‑3.5、ChatGPT（未微调）以及一些专门的表格模型（如 TabFact、TaBERT）进行比较。  
- **结果**：论文声称 Table‑GPT 在所有评测任务上都稳稳领先，尤其在未见任务上平均提升约 10%~15% 的准确率。对比原始 GPT‑3.5，表格问答的错误率从 22% 降到 12%。  
- **消融实验**：作者分别去掉“多任务合成”“自然语言指令混合”“任务模板多样化”，发现去掉任意一项后整体性能都会下降 3%~7%，说明每个设计都有实质贡献。  
- **局限性**：论文承认仍然依赖大量合成数据，真实业务中极端稀有的表格结构可能仍然表现不佳；此外，微调过程需要访问原始模型的权重，普通用户难以自行复现。

### 影响与延伸思考
Table‑GPT 把“表格微调”作为一种新范式，打开了语言模型在结构化数据上进一步深化的可能。随后出现的工作如 **Table‑LLaMA**、**Struct‑GPT** 等，都在尝试把更多二维或图结构（如知识图谱）纳入微调流程。对想继续深挖的读者，可以关注以下方向：  
- **少样本表格学习**：如何在更少的真实标注下仍保持高效的表格推理。  
- **跨模态统一模型**：把表格、图片、代码等多种结构统一进同一个指令模型。  
- **可解释的表格推理**：让模型在给出答案的同时，输出推理路径或中间表格操作步骤。  

### 一句话记住它
把真实表格转化为海量指令任务再微调，让通用语言模型也能像专业表格软件一样精准理解和操作二维数据。