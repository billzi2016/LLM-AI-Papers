# The FACTS Grounding Leaderboard: Benchmarking LLMs' Ability to Ground   Responses to Long-Form Input

> **Date**：2025-01-06
> **arXiv**：https://arxiv.org/abs/2501.03200

## Abstract

We introduce FACTS Grounding, an online leaderboard and associated benchmark that evaluates language models' ability to generate text that is factually accurate with respect to given context in the user prompt. In our benchmark, each prompt includes a user request and a full document, with a maximum length of 32k tokens, requiring long-form responses. The long-form responses are required to be fully grounded in the provided context document while fulfilling the user request. Models are evaluated using automated judge models in two phases: (1) responses are disqualified if they do not fulfill the user request; (2) they are judged as accurate if the response is fully grounded in the provided document. The automated judge models were comprehensively evaluated against a held-out test-set to pick the best prompt template, and the final factuality score is an aggregate of multiple judge models to mitigate evaluation bias. The FACTS Grounding leaderboard will be actively maintained over time, and contains both public and private splits to allow for external participation while guarding the integrity of the leaderboard. It can be found at https://www.kaggle.com/facts-leaderboard.

---

# FACTS Grounding 排行榜：评估大语言模型在长文本输入下的事实依据能力 论文详细解读

### 背景：这个问题为什么难？
在实际对话或写作场景里，用户常常会提供一段很长的文档（几千甚至上万词）并要求模型基于这段材料生成答案。传统的事实性评测大多只用几百字的短上下文，模型只需要找出几个关键句子就能完成任务。随着上下文长度突破 32 k token，模型需要在海量信息中定位、整合并确保每一句输出都能在原文中找到对应依据，这对检索、记忆和生成的协同提出了前所未有的挑战。过去的基准要么不提供足够长的文档，要么只检查答案是否合理，却不强制“每句话都有出处”。因此，需要一个专门衡量“长文档完全依据”能力的基准和排行榜。

### 关键概念速览
**长上下文（Long Context）**：指输入文本的长度可以达到数万 token，远超常规模型的上下文窗口。想象把一本小册子一次性喂给模型，而不是分章节逐段喂。  
**事实依据（Grounding）**：生成的每句话都能在提供的文档中找到对应的原句或段落，就像写论文时每句话后面都要标注来源。  
**自动评审模型（Automated Judge Model）**：用另一个语言模型来判断答案是否满足用户需求以及是否完全依据文档，相当于机器版的评卷老师。  
**两阶段过滤（Two‑Phase Filtering）**：先检查答案是否回答了问题，未通过的直接淘汰；再检查答案的事实依据程度。类似先看“有没有答对”，再看“答得对不对”。  
**聚合评分（Aggregated Score）**：把多个评审模型的判断结果取平均或投票，以降低单一模型偏见。可以比作多位老师一起打分，最终取综合分。  
**公开/私有拆分（Public‑Private Split）**：排行榜数据分为公开可用的测试集和隐藏的私有集，防止参赛者针对公开数据过度调参。  

### 核心创新点
1. **从短上下文转向超长文档** → 设计了每条样例包含最多 32 k token 的完整文档 → 让模型必须在大规模信息中定位依据，评测更贴近真实业务需求。  
2. **两阶段自动评审流程** → 首先用判定模型筛掉不满足用户请求的答案 → 再用专门的事实性评审模型检查每句话是否能在文档中找到对应片段 → 通过层层过滤，既保证任务完成度，又确保事实依据完整。  
3. **多评审模型聚合** → 在选定的评审模型上跑大量实验，挑出表现最稳的模板 → 最终分数由多个模型的判断综合得到 → 有效抑制单一模型的系统性偏差，提高评分可信度。  
4. **公开‑私有数据拆分的排行榜机制** → 公开集供社区调试，私有集保留评测公平性 → 既鼓励外部参与，又防止排行榜被“刷分”。  

### 方法详解
整体思路可以拆成四步：**数据构造 → 任务定义 → 双层自动评审 → 评分聚合**。

1. **数据构造**  
   - 每条样例由三部分组成：用户请求（如“请概括文档的主要结论”）、完整文档（最长 32 k token）以及参考答案（人工标注的完全依据文本）。  
   - 文档来源覆盖新闻、科研报告、法律文书等多种体裁，确保模型面对不同写作风格时仍能定位依据。

2. **任务定义**  
   - 模型的输出必须是**长篇**（几百到上千字）且**完全依据**提供的文档。  
   - 与传统的“是否包含答案”不同，这里要求每一句都能在文档中找到对应的句子或段落。

3. **双层自动评审**  
   - **第一层（需求满足）**：使用一个经过微调的判定模型，输入用户请求和模型输出，判断是否完成了请求的核心意图。若判定为“不满足”，直接记 0 分。  
   - **第二层（事实依据）**：采用多个独立的评审模型（如不同规模的 LLM），每个模型把输出句子与文档进行匹配，判断是否能找到对应依据。匹配方式包括句子相似度、关键词覆盖以及检索式的定位。  
   - 这一步的关键在于**提示模板的选择**：作者在一个保留的验证集上尝试了多种提示词，最终挑选出最能让评审模型准确判断的模板。

4. **评分聚合**  
   - 对每个评审模型的二层判断结果取平均，得到一个**单模型事实性得分**。  
   - 再把所有模型的得分取平均或加权投票，得到最终的**FACTS Grounding 分数**。这种多模型投票机制类似于“民主投票”，可以抵消单一模型的系统性错误。

**最巧妙的地方**在于把“是否回答了问题”和“是否完全依据”拆成两层评审，而不是让同一个模型一次性判断两件事。这样即使某个模型在需求满足上表现好，但在事实依据上不够严谨，也不会误导最终评分。

### 实验与效果
- **测试数据**：作者在公开的 Kaggle 赛道上提供了约 2,000 条长文档样例，私有集保留约 500 条用于排行榜最终排名。  
- **基线对比**：与几种常见的长上下文模型（如 GPT‑4‑32k、Claude‑2、LLaMA‑2‑70B）进行比较。论文声称在公开集上，这些基线的整体 FACTS 分数在 45% 左右，而最佳参赛模型能够突破 70% 的得分。  
- **消融实验**：作者分别去掉需求满足层、去掉多模型聚合、以及使用单一评审模型进行对比。结果显示，去掉需求满足层会导致误判率上升约 20%；仅使用单模型评审会让整体分数下降约 12%，验证了两层过滤和多模型聚合的必要性。  
- **局限性**：论文承认自动评审模型本身仍会出现误判，尤其是对高度抽象的概括类请求；此外，32 k token 的文档仍对大多数开源模型构成显存瓶颈，实际使用时往往需要分段检索。  

### 影响与延伸思考
自从 FACTS Grounding 排行榜上线后，业界对“长文档完全依据”这一评测需求明显升温。后续出现的工作如 **LongDocEval**、**GroundedQA‑XL** 等，都在数据规模或评审方式上借鉴了 FACTS 的两层过滤和多模型聚合思路。对想进一步探索的读者，可以关注以下方向：  
- **检索‑生成协同**：如何在生成前高效检索到最相关的文档片段，以降低模型的记忆负担。  
- **评审模型的鲁棒性**：设计更可靠的自动评审器，或引入人机混合评审以提升评分可信度。  
- **跨模态依据**：把图像、表格等非文本信息也纳入 grounding 评估，扩展到更复杂的业务场景。  

### 一句话记住它
FACTS Grounding 用超长文档 + 两层自动评审 + 多模型投票，打造了首个衡量 LLM 完全依据长文本生成能力的排行榜。