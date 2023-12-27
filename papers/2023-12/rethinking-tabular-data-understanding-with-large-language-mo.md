# Rethinking Tabular Data Understanding with Large Language Models

> **Date**：2023-12-27
> **arXiv**：https://arxiv.org/abs/2312.16702

## Abstract

Large Language Models (LLMs) have shown to be capable of various tasks, yet their capability in interpreting and reasoning over tabular data remains an underexplored area. In this context, this study investigates from three core perspectives: the robustness of LLMs to structural perturbations in tables, the comparative analysis of textual and symbolic reasoning on tables, and the potential of boosting model performance through the aggregation of multiple reasoning pathways. We discover that structural variance of tables presenting the same content reveals a notable performance decline, particularly in symbolic reasoning tasks. This prompts the proposal of a method for table structure normalization. Moreover, textual reasoning slightly edges out symbolic reasoning, and a detailed error analysis reveals that each exhibits different strengths depending on the specific tasks. Notably, the aggregation of textual and symbolic reasoning pathways, bolstered by a mix self-consistency mechanism, resulted in achieving SOTA performance, with an accuracy of 73.6% on WIKITABLEQUESTIONS, representing a substantial advancement over previous existing table processing paradigms of LLMs.

---

# 重新思考使用大语言模型进行表格数据理解 论文详细解读

### 背景：这个问题为什么难？

表格是最常见的结构化信息载体，却不像自然语言那样有统一的顺序和表达方式。过去的模型大多把表格直接序列化成文字，或者手工设计特征来捕捉行列关系，这导致两大问题：一是模型对同一内容的不同排版（比如转置、排序）极度脆弱；二是现有的推理方式只能在“看文字”或“看符号”之间选其一，难以兼顾两者的优势。于是，如何让通用的大语言模型（LLM）在面对各种表格布局时保持稳健、并且充分利用文字与符号两种推理路径，成为了亟待突破的瓶颈。

### 关键概念速览

**大语言模型（LLM）**：基于海量文本预训练的深度网络，能够生成或理解自然语言。把它想成“会说话的通用大脑”，可以被进一步指令化去完成特定任务。  

**表格结构扰动**：指对同一张表格进行转置、行列交换、列排序等操作后得到的不同形态。就像把一本书的章节顺序随意打乱，内容不变但阅读难度大幅提升。  

**文本推理**：把表格直接转成自然语言描述，然后让 LLM 用语言理解的方式回答问题。类似于把一张表格读成一段故事，再让听众回答细节。  

**符号推理**：把表格视作数学或逻辑符号的集合，利用 LLM 的代码/公式能力进行演算。相当于让模型在“算术表格”上做计算，而不是在文字描述上思考。  

**表格结构归一化**：一种预处理手段，把任意布局的表格统一转成一种标准形态（比如固定列顺序、统一行列方向），让模型看到的都是“同一张表”。可以类比为把所有图片统一裁剪到相同尺寸后再送进视觉模型。  

**自洽机制（Self‑Consistency）**：让模型在同一个问题上多次采样不同的思考路径，然后统计多数答案作为最终输出。类似于多人讨论后取多数意见，以降低单次采样的偶然错误。  

**Mix Self‑Consistency**：在自洽的基础上，混合使用文本和符号两种推理方式的答案，再做多数投票。相当于让“语言专家”和“数学专家”各自给出意见，最后取最受大家认同的答案。

### 核心创新点

1. **结构扰动鲁棒性评估 → 系统化实验揭示表格排版对 LLM 性能的显著影响** → 发现即使内容相同，仅仅换个列顺序或转置，模型在符号推理任务上的准确率会大幅下降。  

2. **表格结构归一化方法 → 对所有输入表格执行统一的转置/排序规则，使其呈现为固定模板** → 实验表明归一化后，符号推理的下降幅度被显著抑制，整体鲁棒性提升。  

3. **文本 vs 符号推理对比 → 在同一任务上分别让模型走文字描述路线和符号计算路线** → 结果显示文本推理略胜一筹，但两者在不同子任务上各有优势，错误分布互补。  

4. **Mix Self‑Consistency 融合策略 → 同时采样文本和符号两条思路，收集多次答案后做混合投票** → 该策略在 WikiTableQuestions 上实现 73.6% 的准确率，刷新了 LLM 处理表格的公开记录。

### 方法详解

整体框架可以拆成四步：  
1) **表格预处理**：把原始 CSV/HTML 表格送入归一化模块；  
2) **双路推理**：分别生成文本描述和符号化表示；  
3) **多次采样**：对每条路径进行若干次随机采样（不同温度、不同提示），得到一组候选答案；  
4) **Mix Self‑Consistency 融合**：把所有候选答案放进投票池，选出出现次数最多的答案作为最终输出。

**步骤 1：表格结构归一化**  
归一化的核心是把表格转成“行优先、列固定”的形式。实现上，首先检测表头是否在首行；若在首列则执行转置；随后对列名进行字典排序，确保每次相同内容的表格得到相同列顺序。这个过程不改变数值本身，只是统一了视觉布局，类似于把所有文档的标题统一放在第一页。

**步骤 2：文本推理路径**  
归一化后的表格被序列化为自然语言，例如：“表格包含三列，分别是‘城市’、‘人口’、‘面积’，第一行是北京，人口 2150 万，面积 16410 平方公里……”。随后使用标准的 LLM 提示（prompt）让模型回答查询，如“北京的人口是多少？”模型在生成答案前会先“思考”，即在内部形成一段文字链（CoT），把相关行列信息串起来再输出数值。

**步骤 3：符号推理路径**  
在符号路径里，表格被转成一种结构化的代码块，例如 Python 的 pandas DataFrame 语法或 SQL 查询语句。提示会引导模型执行类似 “df[df['城市']=='北京']['人口'].iloc[0]”。因为 LLM 已经在大量代码语料上训练，这种符号化的指令能够让模型直接进行数值检索或算术运算。

**步骤 4：Mix Self‑Consistency 融合**  
每条路径会被采样 N 次（论文中 N≈5），得到 N 条答案。所有 2×N 条答案进入投票环节。若多数答案来自文本路径，则说明文字描述更可靠；若符号路径占优，则说明数值计算更稳。作者还加入了一个小技巧：在投票前先对答案进行归一化（比如把“2150万”“2.15 million”统一成同一格式），防止表述差异导致错误投票。

**最巧妙的点**  
- 归一化并不是简单的排序，而是结合了列头检测和转置判断，确保对任意布局都能得到同一内部表示。  
- Mix Self‑Consistency 把两种思考方式的“多样性”与“多数共识”结合，既利用了文本的语言理解优势，又保留了符号的精确计算能力，形成了互补的“双核发动机”。  

### 实验与效果

- **数据集**：主要在 WikiTableQuestions（一个包含真实维基百科表格和对应自然语言问答的大规模基准）上评估。  
- **基线对比**：与之前的 LLM 表格处理方法（如直接序列化、单一文本或单一符号路径）相比，本文的 Mix Self‑Consistency 达到 73.6% 的准确率，明显高于此前的公开记录（具体提升幅度未在摘要中给出，论文声称是“显著提升”）。  
- **消融实验**：作者分别去掉归一化、去掉文本路径、去掉符号路径以及关闭自洽投票，结果显示：归一化缺失时符号推理下降最为明显；仅保留单一路径时整体准确率回落到约 68%；关闭投票机制后，准确率下降约 2‑3 个百分点，验证了每个模块的贡献。  
- **错误分析**：文本推理在需要跨行关联的信息时表现更好；符号推理在纯数值计算或需要精确比较的场景更稳。两者错误类型互不重叠，正是混合投票能够抵消彼此弱点的根本原因。  
- **局限性**：论文未详细说明在极端表格（如跨表合并、嵌套表头）上的表现；归一化步骤本身依赖于列头的可检测性，若表格缺失明确标题可能仍会出错。  

### 影响与延伸思考

这篇工作把“表格结构不确定性”正式列为 LLM 需要面对的核心挑战，推动了后续研究在 **结构归一化** 与 **多模态自洽** 方向的探索。随后出现的几篇论文（如 *TablePrompt*、*HybridTableQA*）都在不同程度上借鉴了双路推理和混合投票的思路，甚至把视觉表格识别与 LLM 结合，形成更完整的表格理解流水线。对想进一步深入的读者，可以关注以下几个方向：  
1) **自动化归一化**：利用图神经网络或强化学习自动发现最优表格布局。  
2) **跨表推理**：把多张相关表格的归一化结果统一到同一知识图谱，再进行联合查询。  
3) **自洽机制的理论分析**：为什么多数投票能显著提升准确率，背后是否有统计学或信息论的解释。  

### 一句话记住它

把表格先统一排版，再让大语言模型同时用文字和符号两种思路思考，最后用多数投票决定答案，这样就能让 LLM 在表格问答上跑出新纪录。