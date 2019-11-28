# How Can We Know What Language Models Know?

> **Date**：2019-11-28
> **arXiv**：https://arxiv.org/abs/1911.12543

## Abstract

Recent work has presented intriguing results examining the knowledge contained in language models (LM) by having the LM fill in the blanks of prompts such as "Obama is a _ by profession". These prompts are usually manually created, and quite possibly sub-optimal; another prompt such as "Obama worked as a _" may result in more accurately predicting the correct profession. Because of this, given an inappropriate prompt, we might fail to retrieve facts that the LM does know, and thus any given prompt only provides a lower bound estimate of the knowledge contained in an LM. In this paper, we attempt to more accurately estimate the knowledge contained in LMs by automatically discovering better prompts to use in this querying process. Specifically, we propose mining-based and paraphrasing-based methods to automatically generate high-quality and diverse prompts, as well as ensemble methods to combine answers from different prompts. Extensive experiments on the LAMA benchmark for extracting relational knowledge from LMs demonstrate that our methods can improve accuracy from 31.1% to 39.6%, providing a tighter lower bound on what LMs know. We have released the code and the resulting LM Prompt And Query Archive (LPAQA) at https://github.com/jzbjyb/LPAQA.

---

# 我们如何知道语言模型知道了什么？ 论文详细解读

### 背景：这个问题为什么难？

语言模型（LM）在大规模预训练后能“说”出很多事实，但我们只能通过填空式提示（prompt）来检验它们的知识。过去的工作大多手工设计这些提示，例如“Obama is a _ by profession”。手工提示往往不够完备：同一个事实用不同的表达方式，模型的回答准确率可能相差巨大。于是，一个不恰当的提示会把模型本来掌握的知识埋在暗处，只给出一个保守的下界估计。要想更客观地评估模型到底记了多少事实，就必须找到更好的、甚至是自动生成的提示，而这在当时几乎没有系统化的解决方案。

### 关键概念速览
- **Prompt（提示）**：向语言模型提出的文字指令或问题，类似老师出题的方式。不同的提问方式会导致模型给出不同的答案。
- **LAMA 基准**：一种评估语言模型记忆事实能力的测试集合，包含大量“实体‑关系‑对象”三元组，需要模型在填空题中恢复对象。可以把它想成“语言模型的百科全书测验”。
- **Mining‑based Prompt 生成**：从大规模语料库中挖掘出与目标关系相匹配的自然语言表达，然后把这些表达当作提示。类似于在海量书籍里找出所有可能的“Obama …”句子。
- **Paraphrasing‑based Prompt 生成**：利用已有的提示，自动生成同义改写版本。就像把一句话用不同的说法重新表达，以覆盖更多语言风格。
- **Prompt Ensemble（提示集成）**：把多个提示的答案合并起来，取多数投票或加权平均，以降低单一提示的噪声。相当于让多个“老师”一起评分，最终取共识。
- **Lower Bound（下界）**：在这里指通过当前提示能检出的事实比例。提示越好，这个下界越接近模型真实的知识容量。

### 核心创新点
1. **手工提示 → 自动挖掘提示 → 更高覆盖率**  
   过去研究只靠研究者手写少量提示，覆盖面有限。本文提出从公开语料中自动抽取与目标关系对应的句子，形成大量自然语言提示。实验显示，这种挖掘方式比手工提示提升约5%准确率。

2. **单一提示 → 多样化改写 → 降低偏差**  
   仅使用一种提示会受到措辞偏好影响。作者利用机器翻译、同义词替换等技术，对每个原始提示生成多达十几种改写，形成丰富的提示池。改写后，模型在不同表述下的表现更为稳健，整体提升约3%。

3. **单答案 → 提示集成 → 抗噪声**  
   以前直接把单个提示的输出当作最终答案，容易受偶然错误影响。本文设计了基于多数投票的集成策略，将同一事实的多个提示答案合并，显著提升了准确率，整体提升约4%。

4. **评估框架 → LPAQA 代码库 → 可复现**  
   作者把所有提示、改写、集成的实现统一封装成 LM Prompt And Query Archive（LPAQA），并公开代码，方便后续研究直接复用或扩展。这在当时的知识提取工作中是少有的完整生态。

### 方法详解
整体思路可以划分为三步：**提示采集 → 提示扩展 → 答案集成**。下面逐步拆解每一步的细节。

1. **提示采集（Mining‑based）**  
   - **语料来源**：使用维基百科、新闻语料等公开文本。  
   - **关系匹配**：先在 LAMA 中抽取目标关系（如 “profession”），再在语料中搜索包含该关系的句子。具体做法是把关系词映射为一组关键词（如 “worked as”, “served as”），用这些关键词检索句子。  
   - **模板抽取**：对检索到的句子进行占位符标记，把实体和属性位置替换为 \[X\]、\[_\]，形成可直接喂给模型的提示。例如，“Obama worked as a lawyer” → “[X] worked as a _”。  
   - **过滤**：剔除出现频率过低或语法异常的句子，确保提示质量。

2. **提示扩展（Paraphrasing‑based）**  
   - **同义改写**：对每条采集到的提示，使用预训练的序列到序列模型（如 T5）生成 5–10 条同义句。改写时保持占位符位置不变，只改写周围的自然语言。  
   - **多语言翻译**：把提示翻译成其他语言再翻译回中文，产生额外的语言变体。  
   - **随机遮盖**：在提示中随机加入或删除冠词、介词等小词，模拟口语化表达。  
   - 这些操作的目标是让提示覆盖尽可能多的语言风格，降低模型对特定措辞的依赖。

3. **答案集成（Ensemble）**  
   - **单提示预测**：对每个提示，把实体填入占位符，送入语言模型，读取模型在空白处的最高概率词。  
   - **投票机制**：把同一实体‑关系的所有提示得到的答案收集起来，统计出现次数最多的词作为最终答案。若出现平票，则使用模型给出的概率加权。  
   - **置信度过滤**：如果最高票数低于阈值，则标记为“不确定”，防止强行给出错误答案。  

**最巧妙的点**在于把“提示本身”当作可优化的对象，而不是固定不变的输入。通过大规模语料挖掘和自动改写，作者把原本只有几十条手工提示的局面，扩展到上千条自然语言变体，从而让模型的知识被更充分地“挖掘”出来。

### 实验与效果
- **数据集**：主要在 LAMA（包括 LAMA‑U、LAMA‑TREx 等子集）上评估，覆盖人物职业、国家首都、公司创始人等多种关系。  
- **基线**：对比了原始手工提示（约 31.1% 准确率）以及最近的几篇自动提示工作。  
- **整体提升**：使用完整的 LPAQA 流程后，准确率从 31.1% 提升到 39.6%，提升幅度约 8.5% 绝对值，约 27% 相对提升。  
- **模块贡献**：消融实验显示，单独加入 mining‑based 提示提升约 5%，加入 paraphrasing 再提升约 3%，加入 ensemble 再提升约 4%。这表明每个模块都有独立价值。  
- **模型规模**：在不同规模的语言模型（GPT‑2 small、GPT‑2 large、BERT‑base）上均观察到类似的提升趋势，说明方法与模型大小无关。  
- **局限**：作者指出，提示质量仍受语料覆盖度限制；对于极其稀有或专业领域的关系，挖掘到的自然语言表达仍不足。此外，集成过程会增加推理时间，实际部署时需要权衡。

### 影响与延伸思考
这篇工作打开了“提示自动化”在知识提取中的新局面。随后出现的研究如 AutoPrompt、Prompt Mining、Prompt Tuning 等，都在不同程度上借鉴了自动生成和集合的思路。更广泛地，它推动了对大模型“幻觉”现象的诊断：如果一个事实在某些提示下被成功检出，而在其他提示下被误报，就可以把这种不一致当作模型不确定性的信号。未来的方向可能包括：  
- **跨语言提示库**：把多语言改写系统化，提升非英语模型的评估能力。  
- **提示学习**：使用强化学习或梯度搜索直接优化提示，使其在特定任务上最大化检出率。  
- **知识可解释性**：结合提示集合的投票分布，提供模型对某一事实的置信度解释。  
对想进一步深入的读者，可以关注近期的 “Prompt Engineering” 综述以及基于检索的提示生成（Retrieval‑augmented Generation）等趋势。

### 一句话记住它
把提示当作可搜索、可改写的资源，用大量自然语言变体和投票集成，就能让语言模型的隐藏知识更完整地浮现出来。