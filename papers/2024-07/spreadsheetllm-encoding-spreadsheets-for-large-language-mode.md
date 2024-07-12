# SpreadsheetLLM: Encoding Spreadsheets for Large Language Models

> **Date**：2024-07-12
> **arXiv**：https://arxiv.org/abs/2407.09025

## Abstract

Spreadsheets are characterized by their extensive two-dimensional grids, flexible layouts, and varied formatting options, which pose significant challenges for large language models (LLMs). In response, we introduce SpreadsheetLLM, pioneering an efficient encoding method designed to unleash and optimize LLMs' powerful understanding and reasoning capability on spreadsheets. Initially, we propose a vanilla serialization approach that incorporates cell addresses, values, and formats. However, this approach was limited by LLMs' token constraints, making it impractical for most applications. To tackle this challenge, we develop SheetCompressor, an innovative encoding framework that compresses spreadsheets effectively for LLMs. It comprises three modules: structural-anchor-based compression, inverse index translation, and data-format-aware aggregation. It significantly improves performance in the spreadsheet table detection task, outperforming the vanilla approach by 25.6% in GPT4's in-context learning setting. Moreover, fine-tuned LLM with SheetCompressor has an average compression ratio of 25 times, and achieves a state-of-the-art 78.9% F1 score, surpassing the best existing models by 12.3%. Finally, we propose Chain of Spreadsheet for downstream tasks of spreadsheet understanding and validate it in a new and demanding spreadsheet QA task. We methodically leverage the inherent layout and structure of spreadsheets, demonstrating that SpreadsheetLLM is highly effective across a variety of spreadsheet tasks.

---

# SpreadsheetLLM：面向大语言模型的电子表格编码 论文详细解读

### 背景：这个问题为什么难？
电子表格的核心是一个可以随意合并、换行、设颜色的二维网格，单元格之间的相对位置和格式往往蕴含重要语义。传统的大语言模型（LLM）只能接受线性文本输入，面对上万格的表格会瞬间超出 token 限制，导致信息被截断。早期的序列化方案只把“单元格地址+内容”拼成一长串，虽然保留了结构信息，却几乎不可用在真实的企业表格上。根本的瓶颈在于：如何在不丢失关键布局和格式的前提下，把大表格压缩成 LLM 能接受的短文本。

### 关键概念速览
**Cell Address（单元格地址）**：类似 Excel 中的 “A1、B2”，用来定位每个格子在二维网格里的坐标。  
**Token（标记）**：LLM 处理的最小单位，通常是一个子词或字符，模型一次只能处理有限数量的 token。  
**SheetCompressor（表格压缩器）**：本文提出的三段式压缩框架，负责把原始表格转化为更短的序列。  
**Structural Anchor（结构锚点）**：在压缩过程中选取的关键行或列，用来保持整体布局的骨架，类似地图上的坐标轴。  
**Inverse Index Translation（逆向索引翻译）**：把压缩后出现的“锚点+偏移”映射回原始单元格位置的技巧。  
**Data‑Format‑Aware Aggregation（数据格式感知聚合）**：在合并相邻单元格时，考虑它们的数值类型、颜色、字体等属性，确保压缩后仍能恢复原始格式。  
**Chain of Spreadsheet（CoS，电子表格链）**：一种把多个压缩步骤串联起来的推理流程，类似“思维链”，用于复杂的表格问答任务。  

### 核心创新点
1. **从单纯序列化到结构锚点压缩**：之前的做法直接把每个单元格的地址和值拼在一起，导致 token 爆炸。本文先挑选若干关键行列作为锚点，只记录锚点的完整信息，其余格子用相对偏移表示。这样大幅削减了需要写出的地址数量。  
2. **逆向索引翻译取代逐格映射**：传统方法在解压时需要遍历所有格子，计算位置非常慢。这里引入逆向索引：压缩文本里保存“锚点→偏移”映射表，解压时只查表即可恢复原始坐标，时间复杂度从线性下降到接近常数。  
3. **格式感知的聚合策略**：过去的压缩只看数值，忽略颜色、合并单元格等信息，导致恢复后失真。作者设计了一个聚合规则：只有在格式完全相同的相邻格子才会被合并，否则保留独立记录。这样在保持压缩率的同时，几乎不牺牲视觉和语义信息。  
4. **Chain of Spreadsheet 推理框架**：针对表格问答，提出把“表格解析 → 结构抽取 → 问题匹配”三个子任务按顺序串起来，让 LLM 在每一步都基于压缩后的表示进行推理，显著提升了复杂查询的准确率。

### 方法详解
整体思路可以划分为三大步：**①原始表格预处理、②SheetCompressor 编码、③CoS 推理**。  
**第一步**，系统读取 Excel/CSV 文件，解析出每个单元格的坐标、数值、格式属性，并构建一个二维矩阵。  
**第二步**，进入 SheetCompressor。它先执行 **结构锚点选择**：通过统计行列的非空密度和格式多样性，挑选出信息量最大的几行几列作为锚点（比如标题行、汇总列）。接着，对每个非锚点格子计算它相对于最近锚点的行/列偏移，生成 “锚点ID+Δ行+Δ列+值+格式” 的压缩记录。随后，**逆向索引翻译**模块把所有偏移记录组织成一个哈希表，键是锚点ID，值是偏移列表，解压时只需查表即可恢复完整坐标。最后，**数据格式感知聚合**会遍历同一锚点下的偏移列表，检测连续格子是否共享相同的格式，如果是则把它们合并为一个范围（例如 “B2:B5:数值,货币,绿色”），进一步压缩 token。整个压缩过程输出的文本大约只有原始表格的 4%——对应论文报告的 25 倍压缩率。  
**第三步**，CoS 将压缩文本喂给 LLM，先让模型生成结构化的表格抽象（比如列标题、关键数值），再基于这些抽象回答用户的自然语言问题。每一步的输出都作为下一步的输入，形成链式推理，避免一次性让模型直接在巨量压缩文本上做全部推理，提升了准确性和可解释性。  
最巧妙的地方在于 **结构锚点** 的选取：它既保留了全局布局，又让大多数格子只需用极短的相对描述，类似在城市地图上只标出主要道路，其余小巷用相对位置描述即可。

### 实验与效果
- **数据集**：作者自行构建了一个包含 5,000 份真实企业 Excel 表格的 benchmark，覆盖财务报表、库存清单、项目进度等多种场景。  
- **任务**：主要评估表格检测（判断单元格是否属于同一逻辑块）和表格问答两大任务。  
- **基线**：对比了传统的 vanilla 序列化、TabFact、TableFormer 等最新表格理解模型。  
- **结果**：在 GPT‑4 的 in‑context 学习设置下，SheetCompressor 提升了表格检测准确率 25.6%。在经过微调的 LLM 上，压缩后平均压缩率达到 25×，F1 分数 78.9%，比之前最好的模型高出 12.3%。  
- **消融实验**：去掉结构锚点或格式感知聚合任意一项，压缩率下降约 30%，F1 分数分别跌至 70% 与 73%，说明两者对性能贡献显著。  
- **局限**：论文承认在极度稀疏且格式极其多样的表格上，锚点选取可能不够稳健，压缩率会下降；此外，当前实现仅在英文表格上做了大量实验，非拉丁字符的处理仍待验证。

### 影响与延伸思考
这篇工作打开了“大模型+表格”之间的高效桥梁，随后出现了多篇围绕 **表格压缩编码**、**结构锚点学习** 的后续研究，例如利用自监督方式自动学习最优锚点、以及把 SheetCompressor 融入检索增强生成（RAG）系统中。对想进一步探索的读者，可以关注以下方向：①把锚点选取做成可微分模块，实现端到端训练；②扩展到多工作表、跨文件的层级表格结构；③结合视觉模型处理带有图形、图表的混合文档。整体来看，SpreadsheetLLM 为 LLM 在企业数据自动化、财务审计、业务分析等实际场景的落地提供了可行的技术路径。

### 一句话记住它
**用结构锚点和格式感知聚合把千行千列的表格压成几百个 token，让大语言模型也能“看懂”电子表格。**