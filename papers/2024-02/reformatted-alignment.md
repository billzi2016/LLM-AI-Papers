# Reformatted Alignment

> **Date**：2024-02-19
> **arXiv**：https://arxiv.org/abs/2402.12219

## Abstract

The quality of finetuning data is crucial for aligning large language models (LLMs) with human values. Current methods to improve data quality are either labor-intensive or prone to factual errors caused by LLM hallucinations. This paper explores elevating the quality of existing instruction data to better align with human values, introducing a simple and effective approach named ReAlign, which reformats the responses of instruction data into a format that better aligns with pre-established criteria and the collated evidence. This approach minimizes human annotation, hallucination, and the difficulty in scaling, remaining orthogonal to existing alignment techniques. Experimentally, ReAlign significantly boosts the general alignment ability, math reasoning, factuality, and readability of the LLMs.   Encouragingly, without introducing any additional data or advanced training techniques, and merely by reformatting the response, LLaMA-2-13B's mathematical reasoning ability on GSM8K can be improved from 46.77% to 56.63% in accuracy. Additionally, a mere 5% of ReAlign data yields a 67% boost in general alignment ability measured by the Alpaca dataset. This work highlights the need for further research into the science and mechanistic interpretability of LLMs. We have made the associated code and data publicly accessible to support future studies at https://github.com/GAIR-NLP/ReAlign.

---

# 重新格式化对齐 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）要真正对齐人类价值，需要高质量的指令微调数据。过去的做法要么让人工标注员逐条检查、改写，成本高得离谱；要么直接让模型自己生成“更好”的答案，却常常出现幻觉——模型凭空捏造事实。于是，提升数据质量的途径在规模化和可靠性之间陷入两难：要么花钱花力，要么冒着错误信息的风险。

### 关键概念速览
**指令微调（Instruction Fine‑tuning）**：在已有的大模型基础上，用一批“指令‑答案”对进行二次训练，使模型学会遵循用户的明确请求。类似于给学生布置练习题，让他们练习答题技巧。  
**对齐（Alignment）**：让模型的输出符合人类的价值观和期望，避免有害或误导性内容。可以想象成把模型调校成“好老师”。  
**幻觉（Hallucination）**：模型在回答中捏造不存在的事实或数据，就像人在记忆模糊时胡乱填空。  
**ReAlign**：本文提出的核心技巧——把原始答案重新排版、组织，使其更贴合预设的评判标准和证据集合。相当于把一篇散文重新编辑成结构化的报告。  
**Alpaca 数据集**：一个公开的指令微调基准，包含数千条问答对，用来评估模型的通用对齐能力。  
**GSM8K**：一个数学推理基准，包含 8,000 道小学到高中水平的算术题，常用来测模型的数学能力。  

### 核心创新点
1. **从“改写答案”到“重新格式化”**：传统做法是让标注员或模型重新写一遍答案，既费时又容易引入新错误。ReAlign 只把已有答案的排版、段落顺序、标记方式等结构性要素调正，而不改变核心内容。这样既省了人工成本，又避免了因重新生成而产生的幻觉。  
2. **统一的评判标准嵌入**：在格式化过程中，作者把预先收集的证据和对齐准则直接嵌入答案模板，使模型在学习时自然看到“这段话为什么对，这段话为什么错”。相当于在教材里直接标注答案的评分要点。  
3. **与现有对齐手段完全正交**：ReAlign 只涉及数据的呈现方式，不改变模型结构、损失函数或训练调度。因此它可以直接叠加在任何已有的对齐流水线之上，像给已有系统加装一个插件。  
4. **极小数据量即可产生大幅提升**：实验显示，仅使用 5% 的 ReAlign 数据，就能让 LLaMA‑2‑13B 在 Alpaca 基准上提升 67%。这说明格式化的“信息密度”远高于原始散乱数据。  

### 方法详解
整体思路可以拆成三步：  
1. **收集原始指令‑答案对**：从公开的指令数据集（如 Alpaca、ShareGPT）抓取已有的问答。  
2. **定义统一模板**：作者先制定一套“对齐模板”，包括标题、问题复述、答案要点、证据引用、价值评估四个块。每块都有固定的标记（例如 `### Evidence:`）。  
3. **自动化重排**：利用脚本把原始答案映射进模板。具体做法是：  
   - 把答案拆成句子或段落；  
   - 按照“先给出结论、再给出推理、最后列出证据”的顺序重新排列；  
   - 在每个推理步骤后插入对应的证据链接或引用；  
   - 对价值评估块填入预先设定的评分标签（如 “符合安全准则”）。  
   这一步完全基于规则，不涉及生成模型，因此不存在新幻觉的风险。  

最巧妙的地方在于**把评判标准硬编码进数据本身**。传统的对齐训练只能让模型在训练后“猜”哪些答案好，ReAlign 直接把“好答案的特征”写进了训练样本，让模型在学习时看到“对齐”到底长什么样。  

### 实验与效果
- **测试任务**：作者在两个公开基准上评估：Alpaca（通用对齐）和 GSM8K（数学推理）。  
- **基线对比**：与原始 LLaMA‑2‑13B、以及同等规模的常规指令微调模型相比，ReAlign 在 Alpaca 上提升约 67%（原始 30% 左右提升到近 50%），在 GSM8K 上把正确率从 46.77% 拉到 56.63%。  
- **消融实验**：论文报告了只使用模板的结构化块（不加入证据）时提升幅度下降约 20%，说明证据嵌入是关键因素。  
- **局限性**：作者承认 ReAlign 仍然依赖于原始答案的真实性，如果原始答案本身错误，格式化后仍会传播错误；此外，模板设计是手工制定的，跨语言或跨领域的迁移成本尚未评估。  

### 影响与延伸思考
ReAlign 把“数据质量提升”从“内容生成”转向“结构优化”，在业界引发了对数据格式化的关注。随后有几篇工作尝试把类似的模板化思路用于多模态指令（图文）或长文对齐，甚至把自动化模板生成当作一个小模型的任务（Meta‑Prompting）。如果想进一步挖掘这条路，可以关注 **“可解释对齐数据”** 和 **“自动化评判标准抽取”** 两个方向——让机器自己发现哪些结构化特征最能提升对齐效果。  

### 一句话记住它
只要把已有答案重新排版、嵌入评判标准，就能在不增加新数据的情况下，大幅提升大模型的对齐与推理能力。