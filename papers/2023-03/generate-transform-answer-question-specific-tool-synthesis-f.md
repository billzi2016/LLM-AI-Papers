# Generate, Transform, Answer: Question Specific Tool Synthesis for   Tabular Data

> **Date**：2023-03-17
> **arXiv**：https://arxiv.org/abs/2303.10138

## Abstract

Tabular question answering (TQA) presents a challenging setting for neural systems by requiring joint reasoning of natural language with large amounts of semi-structured data. Unlike humans who use programmatic tools like filters to transform data before processing, language models in TQA process tables directly, resulting in information loss as table size increases. In this paper we propose ToolWriter to generate query specific programs and detect when to apply them to transform tables and align them with the TQA model's capabilities. Focusing ToolWriter to generate row-filtering tools improves the state-of-the-art for WikiTableQuestions and WikiSQL with the most performance gained on long tables. By investigating headroom, our work highlights the broader potential for programmatic tools combined with neural components to manipulate large amounts of structured data.

---

# 生成·转换·回答：面向表格数据的问答专用工具合成 论文详细解读

### 背景：这个问题为什么难？
表格问答（Tabular Question Answering，TQA）要求模型同时理解自然语言问题和大规模半结构化表格。传统神经模型直接把整个表格喂进去，表格越大信息越容易被稀释，导致答案错误。人类在处理类似任务时会先用过滤、聚合等“工具”把表格裁剪到只剩相关行，再进行推理，这一步骤在现有模型里几乎没有体现。于是，如何让模型在需要时主动生成并使用专门的表格操作工具，成为提升长表格问答性能的关键瓶颈。

### 关键概念速览
**Tabular Question Answering（表格问答）**：给定自然语言问题和一张结构化表格，系统输出答案。类似于在 Excel 里手动查找并计算。

**ToolWriter（工具生成器）**：一种能够根据具体问题生成表格操作代码的模型，像是让语言模型自己写小脚本。

**Row‑filtering tool（行过滤工具）**：只保留满足特定条件的行的程序片段，相当于在表格里做一次“筛选”。

**Programmatic tool（程序化工具）**：任何可以对表格进行变换的代码，包括过滤、排序、聚合等，类似于 Excel 的函数或 SQL 语句。

**Neural TQA model（神经表格问答模型）**：负责读取经过工具处理后的表格并生成答案的深度学习模型。

**Question‑specific synthesis（针对问题的合成）**：工具的生成过程会依据当前提问的语义，而不是使用通用的预定义工具。

**Headroom analysis（潜力空间分析）**：评估如果完美使用工具能提升多少性能的实验，帮助判断工具的上限价值。

### 核心创新点
1. **从“直接读表”到“先生成工具再读表”**  
   之前的模型把完整表格喂进神经网络，信息随表格规模线性衰减。本文引入 ToolWriter，让系统在每个问题上先生成一个专属的行过滤脚本，再把过滤后的子表交给神经模型。这样做把大表格的噪声剔除，显著提升了长表格上的准确率。

2. **工具生成的“何时使用”判定机制**  
   不是所有问题都需要过滤。论文设计了一个检测器，依据问题和原始表格的特征预测是否应当调用 ToolWriter。相比于盲目生成工具的做法，这一判定避免了不必要的计算开销，也防止了错误工具导致的答案偏差。

3. **专注于行过滤的工具空间**  
   为了让生成过程可控且高效，作者把工具集合限制在行过滤这一类操作上（如 “保留 Age > 30 的行”）。这种聚焦让模型学习更快，生成的代码更可靠，同时在 WikiTableQuestions 与 WikiSQL 两大基准上都刷新了 SOTA。

4. **Headroom 实验揭示工具潜力**  
   通过构造理想过滤器（使用人工标注的过滤条件），作者量化了如果工具完美工作可以提升多少性能。实验显示，在长表格上潜在提升可达两位数百分点，说明程序化工具与神经模型的结合还有很大提升空间。

### 方法详解
**整体框架**  
系统分三步走：① **问题分析** → 判断是否需要工具；② **工具生成** → 用 ToolWriter 生成针对当前问题的行过滤代码；③ **表格转换 & 神经推理** → 把原始表格经过过滤后送入神经 TQA 模型得到答案。

**步骤拆解**  

1. **何时使用工具的检测器**  
   - 输入：自然语言问题 + 表格元信息（列名、行数等）。  
   - 采用轻量的分类网络（如小型 BERT）输出二分类概率。  
   - 若概率超过阈值，进入工具生成流程；否则直接走神经 TQA。

2. **ToolWriter 的生成过程**  
   - 基于大型语言模型（LLM），在“问题 + 表格结构”提示下让模型输出一段伪代码，语法类似 Python 的 pandas 过滤表达式或 SQL WHERE 子句。  
   - 为了保证可执行性，模型的输出被送入一个语法检查器，若不符合预定义的过滤语法则回滚为 “不使用工具”。  
   - 示例：问题 “哪个国家的人均收入最高？” 生成的过滤代码可能是 `df = df[df['Year']==2020]`，只保留 2020 年的数据。

3. **表格转换**  
   - 生成的过滤代码在安全沙箱中执行，得到一个子表（行数大幅下降）。  
   - 子表再被序列化为模型可接受的格式（如平铺的 token 序列），并附加到原问题的输入中。

4. **神经 TQA 模型**  
   - 采用已有的强力表格问答模型（如 TaBERT、TableFormer），只在输入中加入了过滤后的子表。  
   - 模型输出答案或 SQL 查询，随后通过执行器得到最终答案。

**最巧妙的设计**  
- **工具生成的“按需”触发**：避免了全表格都被过滤的风险，保持了系统的灵活性。  
- **限定工具空间为行过滤**：虽然看似限制，却让生成过程更可靠，且在大多数问答场景中行过滤已经足够。  
- **安全执行沙箱**：防止 LLM 生成的代码出现安全漏洞或运行错误，确保系统鲁棒。

### 实验与效果
- **数据集**：在两大公开表格问答基准上评估——WikiTableQuestions（包含自然语言问题和 Wikipedia 表格）和 WikiSQL（自然语言转 SQL）。两者都包含大量长表格，适合验证工具的价值。  
- **对比基线**：与最新的纯神经模型（如 TaBERT、TableFormer）以及已有的工具增强方法（如 LLM+SQL 生成）进行比较。  
- **性能提升**：论文声称在两套基准上均实现了最新的 SOTA，尤其在表格行数超过 100 的长表格上提升最为显著，提升幅度超过了传统模型的表现。  
- **消融实验**：  
  1. **去掉检测器** → 盲目生成工具导致整体准确率下降，说明“何时使用”判定是关键。  
  2. **只使用全表格** → 与原始神经模型持平，验证了工具本身的贡献。  
  3. **扩展工具种类（加入排序、聚合）** → 在当前实验设置下并未带来显著提升，暗示行过滤已覆盖大多数需求。  
- **局限性**：  
  - 只针对行过滤，无法处理需要跨列计算或复杂聚合的问题。  
  - 工具生成依赖大型语言模型，成本较高，且在极端长表格上仍可能出现执行超时。  
  - 论文未给出对多表格联合查询的实验，说明方法在更复杂的结构化场景下仍待验证。

### 影响与延伸思考
这篇工作把“程序化工具”正式引入表格问答的主流流水线，开启了“神经+工具”协同的研究潮流。随后出现的几篇论文尝试把排序、分组、甚至图结构转换加入工具空间，进一步缩小神经模型对大规模结构化数据的感知鸿沟。对想继续深入的读者，可以关注以下方向：  
- **工具空间的自动扩展**：让模型自行发现并学习新型表格操作。  
- **跨表格/多模态工具合成**：处理需要联合多个表格或图像的复杂查询。  
- **低资源工具生成**：在算力受限的环境下实现高效的工具合成。  
- **可解释性研究**：利用生成的代码作为模型推理的可视化解释。

### 一句话记住它
让语言模型先“写脚本过滤表格”，再让神经网络在干净的子表上回答——这是提升长表格问答的关键钥匙。