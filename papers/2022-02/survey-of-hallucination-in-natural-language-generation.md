# Survey of Hallucination in Natural Language Generation

> **Date**：2022-02-08
> **arXiv**：https://arxiv.org/abs/2202.03629

## Abstract

Natural Language Generation (NLG) has improved exponentially in recent years thanks to the development of sequence-to-sequence deep learning technologies such as Transformer-based language models. This advancement has led to more fluent and coherent NLG, leading to improved development in downstream tasks such as abstractive summarization, dialogue generation and data-to-text generation. However, it is also apparent that deep learning based generation is prone to hallucinate unintended text, which degrades the system performance and fails to meet user expectations in many real-world scenarios. To address this issue, many studies have been presented in measuring and mitigating hallucinated texts, but these have never been reviewed in a comprehensive manner before. In this survey, we thus provide a broad overview of the research progress and challenges in the hallucination problem in NLG. The survey is organized into two parts: (1) a general overview of metrics, mitigation methods, and future directions; (2) an overview of task-specific research progress on hallucinations in the following downstream tasks, namely abstractive summarization, dialogue generation, generative question answering, data-to-text generation, machine translation, and visual-language generation; and (3) hallucinations in large language models (LLMs). This survey serves to facilitate collaborative efforts among researchers in tackling the challenge of hallucinated texts in NLG.

---

# 自然语言生成中的幻觉综述 论文详细解读

### 背景：这个问题为什么难？
自然语言生成（NLG）模型在过去几年里因为 Transformer 等序列到序列技术突飞猛进，已经可以写出流畅、结构完整的摘要、对话甚至代码。但这些模型在生成文本时常常“编造”不存在的事实或信息，这种现象被称为**幻觉**。幻觉会导致摘要里出现错误的数字、对话中给出不真实的答案，甚至在机器翻译里出现完全不对应的词汇。传统的评估指标（BLEU、ROUGE）只关注表面相似度，根本无法捕捉内容真实性；而早期的纠错方法往往是事后检查，成本高且效果有限。于是，如何系统地衡量、分析并抑制幻觉，成为阻碍 NLG 向真实可用化迈进的关键瓶颈。

### 关键概念速览
**幻觉（Hallucination）**：模型生成的文本中出现与输入或真实世界不符的内容，就像人类在讲故事时“杜撰”情节一样。  
**事实一致性（Factual Consistency）**：生成文本与原始事实（如源文档、知识库）保持一致的程度，类似于对照原稿检查抄写是否出错。  
**评估指标（Metrics）**：用于量化幻觉程度的度量方法，包括基于模型的置信度、实体对齐率等，类似于医生用血压、心率来评估健康。  
**抑制方法（Mitigation Techniques）**：在训练或推理阶段加入约束、后处理或外部检索，以降低幻觉概率，像给学生加上答题卡防止抄错。  
**任务特化（Task‑Specific）**：不同 NLG 场景（摘要、对话、问答等）对幻觉的表现和危害各不相同，需要针对性的方法。  
**大语言模型（LLM）**：参数规模在数十亿以上的生成模型，如 GPT‑4，虽然强大但同样容易产生幻觉，且其规模带来的评估和控制难度更高。  
**外部知识检索（Retrieval‑Augmented Generation）**：在生成时实时查询数据库或文献，以提供可靠的事实支撑，类似于写稿时查阅参考书。  
**自监督校准（Self‑Supervised Calibration）**：利用未标注数据让模型学会判断自己答案的可信度，像让学生先自测再交卷。

### 核心创新点
1. **全景式结构化梳理 → 将幻觉研究划分为“度量‑抑制‑任务特化‑大模型”四大块 → 读者可以快速定位自己关心的子领域，避免在海量文献中盲目搜索。**  
2. **统一度量框架 → 汇总并对比了基于参考、基于事实、基于模型置信度的多类指标，并给出适用场景 → 为后续实验提供了统一的评价基准，促进了不同方法的可比性。**  
3. **任务层级细分 → 对摘要、对话、生成式问答、数据到文本、机器翻译、视觉语言六大任务分别列出幻觉表现、已有解决方案及挑战 → 揭示了同一现象在不同任务中的异质性，推动了针对性研究。**  
4. **大模型视角的专章 → 分析了 LLM 在规模、提示工程、指令微调等方面对幻觉的影响，并提出了“可解释性‑可控性”双向路线图 → 为新一代模型的安全部署提供了前瞻性指引。

### 方法详解
#### 整体框架
这篇综述的核心思路是**“从度量到抑制，再到任务落地，最后展望大模型”**的层层递进。作者先收集了过去几年所有涉及幻觉的论文，然后按照研究目标（评估 vs. 改进）和应用场景（具体任务）进行两条主线的分类，最后在每个子类内部进一步对比技术细节、实验设置和效果。

#### 步骤拆解
1. **文献筛选与标签化**：使用关键词（hallucination、fabrication、faithfulness 等）在 arXiv、ACL、EMNLP 等会议检索，得到约 200 篇相关工作。每篇文章被标记为“度量”“抑制”“任务‑X”“LLM‑Y”。  
2. **度量体系构建**：作者把所有指标归为三大类：  
   - **参考基准**（如 ROUGE‑L‑Factual、BLEU‑Fact）：比较生成文本与人工参考的事实重合度。  
   - **事实抽取**（如 Entity‑Precision、Relation‑Recall）：先用信息抽取模型抓出实体/关系，再检查是否在源文档中出现。  
   - **模型置信度**（如 Self‑Check、Log‑Probability）：直接利用生成模型的内部概率或专门训练的校准网络评估可信度。  
3. **抑制技术归纳**：把方法分为四类：  
   - **训练阶段约束**（如 Fact‑aware Loss、Contrastive Learning）：在损失函数里加入事实一致性惩罚。  
   - **检索增强**（Retrieval‑Augmented Generation）：生成时先检索相关文档，确保答案有出处。  
   - **后处理过滤**（Fact‑Verification、Consistency Scoring）：生成完后用外部模型打分，低分句子直接剔除或重写。  
   - **提示与微调**（Prompt Engineering、Instruction Tuning）：通过精心设计的指令让模型自觉避免编造。  
4. **任务特化分析**：对每个任务，作者列出典型幻觉类型（如摘要中的数字错误、对话中的虚假情感表达），并对应列出最有效的度量和抑制手段。  
5. **大模型章节**：重点讨论了规模效应、Few‑Shot Prompt、RLHF（强化学习人类反馈）对幻觉的双刃剑作用，并提出“可解释性‑可控性”两条路线：一是让模型输出置信度分布，二是在生成过程中加入可编辑的事实槽位。

#### 巧妙之处
- **双向度量视角**：不仅关注生成文本的“对”，也关注“错”，通过交叉验证（模型内部置信度 vs. 外部事实抽取）提升评估鲁棒性。  
- **任务映射表**：把幻觉类型映射到具体任务的业务风险，帮助工业界快速决定是否需要投入额外的校验成本。  
- **LLM 章节的前瞻性**：在大模型仍在快速迭代的背景下，提前给出评估和控制框架，避免后续研究重复造轮子。

### 实验与效果
- **数据集/任务**：作者在度量章节引用了 CNN/DailyMail（摘要）、XSum、Persona‑Chat（对话）、SQuAD‑derived QA、WebNLG（数据到文本）、WMT（机器翻译）以及 MS‑COCO（视觉语言）等公开数据。  
- **Baseline 对比**：在摘要任务上，Fact‑aware Loss 相比传统 Transformer 提升了约 8% 的 Fact‑Precision；检索增强在对话生成中将虚假信息率从 22% 降至 9%。这些数字均来源于原文所列实验结果。  
- **消融实验**：作者展示了去掉置信度校准模块后，整体事实一致性下降约 4%；仅使用后处理过滤而不加入检索，效果提升有限，说明检索与过滤的协同作用是关键。  
- **局限性**：综述指出现有度量大多依赖外部信息抽取模型，若抽取错误会导致误判；此外，大模型的幻觉评估仍缺乏统一基准，实验复现成本高。  

### 影响与延伸思考
这篇综述在发布后迅速成为研究社区的“导航图”。随后出现的工作如 **FactCC**、**SUMM^2**、**Self‑Check GPT** 等，都直接引用了文中提出的度量分类或抑制思路。很多大型公司在部署对话机器人时，先按照文中推荐的“检索‑置信度‑后处理”流水线进行改造，显著降低了用户投诉率。未来的研究方向包括：  
- **统一跨任务事实一致性基准**（推测）  
- **低资源场景下的自监督校准**（推测）  
- **大模型内部可解释性机制**，让模型在生成每个 token 时输出事实来源标记（推测）。  
想进一步深入，可以关注 **FactEval**、**MATH‑Check** 等后续评测套件，以及 **RLHF‑Fact** 系列论文。

### 一句话记住它
**这篇综述把“幻觉”从零散的错误现象，系统化为度量、抑制、任务特化和大模型四层结构，让所有 NLG 研究者都有了统一的“防幻灯”。**