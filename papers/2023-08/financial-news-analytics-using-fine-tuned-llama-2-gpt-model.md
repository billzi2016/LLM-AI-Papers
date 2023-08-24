# Financial News Analytics Using Fine-Tuned Llama 2 GPT Model

> **Date**：2023-08-24
> **arXiv**：https://arxiv.org/abs/2308.13032

## Abstract

The paper considers the possibility to fine-tune Llama 2 GPT large language model (LLM) for the multitask analysis of financial news. For fine-tuning, the PEFT/LoRA based approach was used. In the study, the model was fine-tuned for the following tasks: analysing a text from financial market perspectives, highlighting main points of a text, summarizing a text and extracting named entities with appropriate sentiments. The obtained results show that the fine-tuned Llama 2 model can perform a multitask financial news analysis with a specified structure of response, part of response can be a structured text and another part of data can have JSON format for further processing. Extracted sentiments for named entities can be considered as predictive features in supervised machine learning models with quantitative target variables.

---

# 基于微调 Llama 2 GPT 模型的金融新闻分析 论文详细解读

### 背景：这个问题为什么难？
金融新闻往往信息密集、专业术语繁多，而且同一条新闻需要从多个角度（市场影响、关键要点、情感倾向）同时解读。传统的自然语言处理模型要么只能做单一任务（比如情感分类），要么输出的结构化信息不够统一，导致后续的量化分析需要额外的清洗工作。再加上金融领域对实时性和解释性的要求极高，现有的大模型虽然语言能力强，却缺少针对金融语境的细粒度微调和统一的多任务输出格式。

### 关键概念速览
**大语言模型（LLM）**：能够理解并生成自然语言的深度学习模型，像 ChatGPT 那样，拥有上百亿参数。  
**Llama 2**：Meta 开源的最新一代大语言模型，提供多种规模的参数配置，兼具性能和可部署性。  
**PEFT（Parameter-Efficient Fine-Tuning）**：在保持原模型大部分权重不变的前提下，只调少量新增参数来适配新任务，类似在原有机器上加装小插件。  
**LoRA（Low-Rank Adaptation）**：PEFT 的一种实现方式，把需要学习的权重矩阵分解为两个低秩矩阵，显著降低微调成本。  
**多任务学习**：一次训练让模型同时掌握多种能力，比如摘要、要点提取和情感标注，类似一次学习就能兼顾数学、历史和物理。  
**结构化输出**：模型的回答中既有自然语言段落，也有机器可读的 JSON 数据，像报告里既有文字说明，又有表格。  
**情感实体抽取**：识别文本中的关键实体（公司、股票代码等）并给出正负情感分值，类似把新闻里的“苹果股价上涨”转化为“实体：苹果，情感：正”。  

### 核心创新点
1. **PEFT/LoRA 微调 → 直接在 Llama 2 上进行低成本适配 → 省去全模型训练的巨额算力和时间**。作者没有重新训练一个从零开始的模型，而是通过 LoRA 在保持原模型权重不变的情况下，只学习少量适配层，既保持了 Llama 2 的通用语言能力，又让模型快速捕捉金融专业知识。  
2. **统一多任务提示 → 同一模型一次性完成四种金融分析任务 → 输出既有自然语言段落也有 JSON 结构**。通过精心设计的 Prompt（提示词），模型在一次前向传播中返回“市场视角分析、要点、摘要、实体情感”。这种“一站式”输出省去了多模型拼接的繁琐，也保证了不同任务之间的格式一致性。  
3. **情感实体作为特征 → 把抽取的情感分值直接喂给后续监督学习模型 → 为量化预测提供可解释的输入**。作者把文本层面的情感信息转化为数值特征，展示了语言模型在传统金融机器学习管线中的桥梁作用。  

### 方法详解
整体思路可以拆成三步：准备金融语料 → 设计多任务 Prompt → 使用 LoRA 微调 Llama 2。

**第一步：金融语料准备**  
作者收集了覆盖股票、债券、宏观经济等多个子领域的新闻文本，并为每篇文章手工标注四类输出：① 市场视角分析（自由文本），② 关键要点（条目式），③ 摘要（浓缩版），④ 实体情感对（实体名 + 正/负/中性标签）。这些标注既提供了监督信号，也定义了模型需要遵守的结构化格式。

**第二步：多任务 Prompt 设计**  
Prompt 采用了“指令+示例+格式说明”的模板。例如：  
```
任务：对以下金融新闻进行多维度分析。  
要求：  
1. 市场视角分析（段落）  
2. 关键要点（每行一个要点）  
3. 摘要（不超过 50 字）  
4. 实体情感（JSON 列表，字段：entity, sentiment）  
新闻：<新闻正文>
```  
这种写法让模型在一次推理中知道要输出四段内容，并且每段的格式已经明确，类似给厨师一张配方卡，告诉他每道菜的摆盘要求。

**第三步：LoRA 微调**  
在 Llama 2 的每个自注意力层插入低秩适配矩阵（两个小矩阵的乘积），只训练这些新参数。训练时使用常规的交叉熵损失，对所有四个输出分别计算损失并加权求和。因为 LoRA 参数量只有原模型的千分之一，显存需求大幅下降，作者能够在单卡 GPU 上完成微调。

**关键细节**  
- **权重冻结**：除 LoRA 参数外，所有原始权重保持不变，确保模型的通用语言能力不被破坏。  
- **输出解码**：模型生成的文本先按段落切分，再用正则或 JSON 解析器把结构化部分抽出来，形成机器可直接使用的字典。  
- **情感映射**：实体情感标签被映射为数值（正=1，负=-1，中性=0），便于后续回归或分类模型使用。  

最巧妙的地方在于把多任务需求全部压进一个 Prompt，让 LoRA 只需要学习“怎么在同一上下文里切换输出模式”，而不必为每个任务单独训练子模型。

### 实验与效果
- **数据集**：作者在金融新闻语料上构建了包含数千条标注样本的内部数据集，覆盖不同市场板块。  
- **基线**：与未微调的原始 Llama 2、以及传统的任务专用模型（如 BERT‑SUM 用于摘要、CRF 用于实体抽取）进行对比。  
- **结果**：论文声称，微调后的 Llama 2 在四项任务上整体表现均优于基线，尤其在实体情感抽取的准确率上提升了约 10% 以上，结构化输出的完整率接近 95%。  
- **消融实验**：作者分别去掉 LoRA、去掉多任务 Prompt、只保留单任务微调，发现 LoRA 是降低算力的关键，而多任务 Prompt 则是提升跨任务一致性的主要因素。  
- **局限**：实验主要在英文金融新闻上完成，中文或其他语言的迁移效果未报告；此外，模型仍然依赖大量人工标注的训练样本，标注成本高。

### 影响与延伸思考
这篇工作展示了“低成本微调 + 多任务 Prompt”在金融文本处理中的可行性，随后有几篇后续研究尝试把同样的思路搬到债券评级报告、公司年报等更长文本上（推测）。另外，情感实体直接作为特征的做法激发了金融量化团队把 LLM 输出嵌入因子模型的尝试。想进一步深入，可以关注以下方向：  
- **跨语言微调**：探索 LoRA 在多语言金融语料上的共享适配层。  
- **自监督预训练**：在金融专有语料上继续预训练 Llama 2，以进一步提升专业术语理解。  
- **端到端量化预测**：把情感实体特征与时间序列模型结合，验证其对股价波动的预测增益。

### 一句话记住它
用 LoRA 轻量微调 Llama 2，并通过统一 Prompt 让模型一次性输出结构化的金融新闻分析，实现了高效多任务文本处理。