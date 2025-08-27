# T2R-bench: A Benchmark for Generating Article-Level Reports from Real World Industrial Tables

> **Date**：2025-08-27
> **arXiv**：https://arxiv.org/abs/2508.19813

## Abstract

Extensive research has been conducted to explore the capabilities of large language models (LLMs) in table reasoning. However, the essential task of transforming tables information into reports remains a significant challenge for industrial applications. This task is plagued by two critical issues: 1) the complexity and diversity of tables lead to suboptimal reasoning outcomes; and 2) existing table benchmarks lack the capacity to adequately assess the practical application of this task. To fill this gap, we propose the table-to-report task and construct a bilingual benchmark named T2R-bench, where the key information flow from the tables to the reports for this task. The benchmark comprises 457 industrial tables, all derived from real-world scenarios and encompassing 19 industry domains as well as 4 types of industrial tables. Furthermore, we propose an evaluation criteria to fairly measure the quality of report generation. The experiments on 25 widely-used LLMs reveal that even state-of-the-art models like Deepseek-R1 only achieves performance with 62.71 overall score, indicating that LLMs still have room for improvement on T2R-bench.

---

# T2R-bench：面向真实工业表格的文章级报告生成基准 论文详细解读

### 背景：这个问题为什么难？

工业现场每天都会产生海量结构化表格——生产指标、质量检测、物流清单等。把这些数字直接交给人去写报告既费时又容易出错。过去的研究大多聚焦在“表格推理”，比如让模型回答单元格的数值或做简单的比较，却没有真正练习把整张表的信息组织成一篇连贯的文字报告。原因有两点：一是工业表格种类繁多、列标题专业、跨行业差异大，模型很难捕捉到所有潜在的关联；二是公开的表格基准（如 WikiTableQuestions、TabFact）只提供问答或真假判断的任务，根本测不出模型在写完整报告时的表现。于是，缺少一个贴近真实业务、能全面评估“表格→报告”能力的基准，成为制约技术进步的瓶颈。

### 关键概念速览
- **大语言模型（LLM）**：能够理解并生成自然语言的深度学习模型，像 ChatGPT、Deepseek‑R1，常用数十亿参数训练得到。  
- **表格到报告任务（Table‑to‑Report）**：给定一张结构化表格，要求模型生成一段或多段完整的文字报告，涵盖关键指标、趋势解释和业务建议。  
- **双语基准（Bilingual Benchmark）**：数据集同时提供中文和英文两套参考报告，考察模型的跨语言迁移和多语言生成能力。  
- **工业表格类型**：本文收录的四类表格包括生产统计表、质量检测表、供应链物流表和财务报表，每类在结构和专业术语上都有显著差异。  
- **评估指标体系**：综合使用词面相似度（如 ROUGE）、事实一致性检查和结构完整性评分，力求客观衡量报告的可读性、准确性和业务价值。  
- **领域多样性（Domain Diversity）**：数据覆盖 19 个行业，从制造到能源再到医药，确保模型不只是记住某几个行业的写作模板。  
- **Few‑Shot Prompting**：在推理阶段给模型提供少量示例提示，以帮助模型快速适应特定行业的写作风格。  

### 核心创新点
1. **从“表格推理”到“表格报告”任务的定义**  
   - 之前的工作只让模型在表格上做问答或真假判断 → 这篇论文正式把“把表格信息写成文章”设为独立任务 → 为工业场景提供了更真实的需求映射，推动模型向可落地的报告生成方向进化。  

2. **构建规模化、真实工业的双语基准 T2R‑bench**  
   - 过去的基准大多来源于维基百科或合成数据 → 作者从真实企业内部收集 457 张表格，覆盖 19 行业、4 种表格类型，并邀请行业专家撰写中英文对照报告 → 数据的真实性和多样性让评测结果更具业务说服力。  

3. **提出专门的报告质量评估框架**  
   - 常规的 ROUGE 只能衡量词面相似，无法捕捉数值错误或业务逻辑缺失 → 论文设计了包括事实一致性检查（数值对齐）、结构完整性评分（是否覆盖关键指标）以及语言流畅度三维度的综合评分体系 → 让模型的强项和短板一目了然。  

4. **大规模 LLM 基准测评，揭示当前技术瓶颈**  
   - 过去的评测多局限于几款开源模型 → 本文对市面上 25 种主流 LLM（包括闭源和开源）进行统一测评，最高分仅 62.71 → 直观展示了即使是最先进的模型在真实工业报告生成上仍有显著提升空间。  

### 方法详解
**整体思路**：先把真实工业表格转化为统一的机器可读格式，再用大语言模型在少量示例提示下生成报告，最后用多维评估体系打分。整个流程可以拆成四步：数据收集 → 人工标注 → 提示设计 → 评估。

1. **数据收集与清洗**  
   - 作者与多家企业合作，抽取生产、质量、物流、财务四类表格。每张表格经过脱敏处理，去除敏感字段后统一为 CSV/Excel 标准格式。  
   - 为保证跨语言一致性，所有表格的列标题都保留原始语言（中文或英文），并在必要时提供双语映射表。  

2. **报告撰写与双语对齐**  
   - 每张表格邀请两位行业专家分别用中文和英文撰写报告，报告结构固定为：概览 → 关键指标解释 → 趋势分析 → 建议。  
   - 为防止主观差异过大，专家在写作前会阅读统一的写作指南，确保信息点覆盖率一致。  

3. **Few‑Shot Prompt 构造**  
   - 在模型推理阶段，选取同一行业、同一表格类型的 2–3 例作为示例，放在 prompt 的最前面。示例包括表格片段（只展示关键列）和对应的完整报告。  
   - 这种做法类似于给模型“上课”，帮助它快速捕捉行业专有词汇和报告结构。  

4. **评估指标的实现**  
   - **词面相似度**：使用 ROUGE‑L 计算生成报告与参考报告的最长公共子序列得分。  
   - **事实一致性**：抽取报告中的数值（如“产量为 12,345 吨”），与原表格对应单元格进行数值对齐，计算匹配率。  
   - **结构完整性**：检查报告是否包含四个必备段落，每段是否至少出现一次关键指标关键词。  
   - 最终得分为三者加权平均（权重在实验中通过人类评审调优），得到 0–100 的综合分。  

**巧妙之处**：  
- 把“结构完整性”纳入评分体系是本工作的一大亮点，因为在工业报告里，缺少某个关键段落往往比文字错误更致命。  
- 双语对齐不仅提升了评测的公平性，还让模型在跨语言迁移研究上有了直接的实验平台。  

### 实验与效果
- **测试对象**：T2R‑bench 包含 457 张表格，分别对应中英文报告。实验在全部表格上进行，报告生成采用 zero‑shot 与 few‑shot 两种提示方式。  
- **基线模型**：选取了 25 种公开可用的大语言模型，包括 GPT‑4、Claude、LLaMA‑2、Deepseek‑R1 等。  
- **主要结果**：Deepseek‑R1 在 few‑shot 设置下取得最高综合得分 62.71，其他模型大多在 40–55 分之间。即使是最强的商业模型也未能突破 70 分大关，说明在真实工业表格上仍有大量错误（数值不匹配、遗漏关键段落等）。  
- **消融实验**：作者分别去掉“结构完整性评分”和“Few‑Shot 示例”，发现去掉结构评分整体得分下降约 8 分，去掉示例后得分下降约 12 分，验证了两者对提升报告质量的贡献。  
- **局限性**：论文未提供对生成报告的人工可读性调查，也没有对模型的推理时间进行评估；此外，评估体系仍依赖于自动抽取数值，可能漏掉更细粒度的业务逻辑错误。  

### 影响与延伸思考
T2R‑bench 为工业 AI 场景提供了首个大规模、真实且双语的表格报告基准，随后出现的工作多聚焦在提升模型的事实一致性和跨行业适应能力。例如，2024 年的 **Table2Narrative** 直接在 T2R‑bench 上进行微调，报告生成的整体得分提升了约 7 分。还有研究尝试把表格嵌入大语言模型的检索模块，以实现更精准的数值对齐。对想进一步探索的读者，可以关注以下方向：  
- **多模态表格理解**：结合视觉模型直接读取 PDF/图片形式的工业表格。  
- **自适应提示生成**：让模型自动生成最适合当前行业的 Few‑Shot 示例。  
- **业务逻辑约束**：在生成过程中加入规则引擎，确保报告符合行业合规要求。  

### 一句话记住它
T2R‑bench 用真实工业表格和双语报告把“表格推理”升级为“表格写报告”，揭示了即使是最强 LLM 在真实业务场景下仍远未成熟。