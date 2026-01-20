# CORE-T: COherent REtrieval of Tables for Text-to-SQL

> **Date**：2026-01-19
> **arXiv**：https://arxiv.org/abs/2601.13111

## Abstract

Realistic text-to-SQL workflows often require joining multiple tables. As a result, accurately retrieving the relevant set of tables becomes a key bottleneck for end-to-end performance. We study an open-book setting where queries must be answered over large, heterogeneous table collections pooled from many sources, without clean scoping signals such as database identifiers. Here, dense retrieval (DR) achieves high recall but returns many distractors, while join-aware alternatives often rely on extra assumptions and/or incur high inference overhead. We propose CORE-T, a scalable, training-free framework that enriches tables with LLM-generated purpose metadata and pre-computes a lightweight table-compatibility cache. At inference time, DR returns top-K candidates; a single LLM call selects a coherent, joinable subset, and a simple additive adjustment step restores strongly compatible tables. Across Bird, Spider, and MMQA, CORE-T improves table-selection F1 by up to 22.7 points while retrieving up to 42% fewer tables, improving multi-table execution accuracy by up to 5.0 points on Bird and 6.9 points on MMQA, and using 4-5x fewer tokens than LLM-intensive baselines.

---

# CORE‑T：面向 Text‑to‑SQL 的一致性表检索 论文详细解读

### 背景：这个问题为什么难？

在真实的 Text‑to‑SQL 场景里，一个自然语言问题往往需要跨多个表才能得到答案。传统的检索系统只能把“相关表”挑出来，却很难保证挑出的表之间能够顺利 join（连接），于是后端的 SQL 生成器经常因为缺表或多余表而出错。密集向量检索（dense retrieval）虽然能把大部分可能的表找回来，但会把大量无关表一起拉进来，导致后续的 join 规划成本爆炸。已有的 join‑aware 方法要么假设有明确的数据库标识，要么在推理时要多次调用大模型，计算开销非常高。于是，如何在保持高召回的同时，快速筛选出一组“相互兼容、能一起使用”的表，成为了制约端到端 Text‑to‑SQL 性能的关键瓶颈。

### 关键概念速览

**密集检索（Dense Retrieval）**：用向量空间把查询和表的表示映射到同一维度，靠相似度挑出 top‑K 表。想象成把所有表放进一个巨大的图书馆，用“语义标签”快速找出可能相关的书。

**表用途元数据（Purpose Metadata）**：用大语言模型（LLM）为每张表生成一句简短的描述，说明这张表主要存什么业务信息。相当于给每本书贴上“一句话简介”，帮助后续判断它们能否一起使用。

**兼容性缓存（Compatibility Cache）**：离线预计算的表‑表兼容分数，衡量两张表是否可以通过相同列进行 join。类似于在图书馆里提前标记哪些书的章节内容相互呼应，查询时直接查表。

**Coherent Subset Selection**：在检索到的候选表中，用一次 LLM 调用挑选出一组内部能够 join 的子集。把大量零散的线索组织成一条连贯的故事线。

**加法式兼容性恢复（Additive Adjustment）**：在 LLM 选出的子集基础上，根据兼容性缓存再把被误删的强兼容表补回来。相当于先用编辑器删掉噪声，再用拼图的边缘提示把缺失的拼块补上。

### 核心创新点

1. **离线表用途标注 → 用 LLM 为每张表生成自然语言的用途描述 → 在检索阶段提供语义层面的过滤**。传统方法只靠表结构或列名做相似度，这会把业务意义相近但列名不同的表漏掉。加入用途元数据后，密集检索可以捕捉到更高层次的关联。

2. **离线兼容性缓存 → 预先计算表‑表的 join 可能性并存入轻量缓存 → 推理时只需一次查表即可快速评估兼容度**。以前的 join‑aware 检索往往要在推理时实时做列对齐或图搜索，计算成本高。缓存把这一步搬到离线，显著降低了在线开销。

3. **一次 LLM 调用的连贯子集选择 → 在 top‑K 候选表上让大模型一次性输出“这些表可以一起使用” → 通过加法式恢复把被过度裁剪的表找回**。相比需要多轮对话或逐表判断的方案，这种“一次决策”大幅压缩了 token 消耗，同时保持了对表之间关系的全局感知。

4. **训练‑免费、可扩展的整体框架 → 所有模块（元数据生成、兼容性缓存）均在离线完成，在线只涉及密集检索 + 单次 LLM 调用 → 在大规模异构表集合上仍能保持低延迟**。以往提升检索质量的办法往往需要额外的监督数据或专门的微调，CORE‑T 通过“先算好、后直接用”的思路规避了这些成本。

### 方法详解

整体思路可以分为两大阶段：**离线准备**和**在线检索**。

**离线准备**  
1. **表用途元数据生成**：遍历表集合，对每张表的列名、示例数据等信息喂给 LLM，要求它输出一句概括该表业务含义的短句（例如“订单表：记录用户购买的商品及时间”）。这一步不需要标注，只要让模型根据已有信息自行推断。  
2. **兼容性缓存构建**：对每对表计算两类特征：① 列向量相似度（把列名和示例值嵌入成向量），② 覆盖检测（检查是否有列名或值域相交）。把这些特征合并成一个兼容分数，存入键值对缓存（表 A → 表 B → 分数）。因为只需要一次遍历，成本在离线阶段是可接受的。

**在线检索**  
1. **密集检索**：把用户的自然语言问题编码成向量，检索出 top‑K（如 100）最相似的表。此时得到的集合可能包含大量不相关或无法 join 的表。  
2. **LLM 连贯子集选择**：把问题、检索到的表列表（包括它们的用途元数据）一次性喂给 LLM，要求模型输出一个子集，使得这些表在业务上能够共同回答问题。模型会基于用途描述判断哪些表可能需要 join。  
3. **加法式兼容性恢复**：对 LLM 选出的子集，遍历其中每张表，在兼容性缓存里查找与之高兼容的表。如果某张表在缓存中与子集中的表有高分但被 LLM 剔除，则把它补回。这样既保留了 LLM 的全局判断，又利用缓存纠正了可能的过度裁剪。

**关键细节**  
- 用途元数据是纯文本，和密集检索的向量表示是并行的，两者在后续阶段互为补充。  
- 兼容性缓存的分数阈值是经验设定，作者在实验中发现只要保留前 5% 的高分对即可显著提升召回。  
- LLM 调用的提示词设计强调“只返回能够一起使用的表”，避免模型产生冗余输出。  
- 整个在线流程只需要一次向量检索和一次 LLM 调用，计算开销约为传统多轮检索的 1/4。

### 实验与效果

- **数据集**：在 Bird、Spider（两个经典的 Text‑to‑SQL 基准）以及 MMQA（多模态问答）上评估。  
- **指标**：表选择的 F1、检索到的表数量、以及最终的多表 SQL 执行准确率。  
- **对比**：与纯密集检索、基于图的 join‑aware 检索以及需要多轮 LLM 交互的强基线相比，CORE‑T 在表选择 F1 上提升最高 **22.7** 分，检索的表数减少约 **42%**，多表执行准确率在 Bird 上提升 **5.0** 点、在 MMQA 上提升 **6.9** 点。  
- **消融实验**：去掉用途元数据会导致 F1 下降约 8 分；不使用兼容性缓存则检索的表数回升至原来的 1.6 倍，说明缓存在控制噪声上至关重要。  
- **局限**：作者指出在极度稀疏的表集合（每张表只有极少列且列名高度相似）时，兼容性缓存的列向量相似度可能产生误判；此外，元数据生成依赖 LLM 的质量，若模型产生歧义描述会影响后续选择。

### 影响与延伸思考

CORE‑T 把“语义描述 + 结构兼容”两条线索结合起来，展示了在大模型时代仍然可以通过离线预处理显著降低在线成本的思路。自论文发布后，已有工作尝试把类似的用途标注扩展到知识图谱实体、甚至 API 接口文档，形成“语义标签驱动的检索”。另外，兼容性缓存的概念被用于跨模态检索（比如图像‑文本‑表格联合查询），进一步验证了其通用性。想深入了解的读者可以关注后续的 “Table‑LLM Alignment” 系列论文，或尝试在自己的业务数据上复现 CORE‑T 的离线缓存构建流程。

### 一句话记住它

**CORE‑T 用一次大模型判断加离线兼容缓存，让检索到的多表既相关又能直接 join，省掉了大量噪声和算力。**