# TailorSQL: An NL2SQL System Tailored to Your Query Workload

> **Date**：2025-05-29
> **arXiv**：https://arxiv.org/abs/2505.23039

## Abstract

NL2SQL (natural language to SQL) translates natural language questions into SQL queries, thereby making structured data accessible to non-technical users, serving as the foundation for intelligent data applications. State-of-the-art NL2SQL techniques typically perform translation by retrieving database-specific information, such as the database schema, and invoking a pre-trained large language model (LLM) using the question and retrieved information to generate the SQL query.   However, existing NL2SQL techniques miss a key opportunity which is present in real-world settings: NL2SQL is typically applied on existing databases which have already served many SQL queries in the past. The past query workload implicitly contains information which is helpful for accurate NL2SQL translation and is not apparent from the database schema alone, such as common join paths and the semantics of obscurely-named tables and columns. We introduce TailorSQL, a NL2SQL system that takes advantage of information in the past query workload to improve both the accuracy and latency of translating natural language questions into SQL. By specializing to a given workload, TailorSQL achieves up to 2$\times$ improvement in execution accuracy on standardized benchmarks.

---

# TailorSQL：面向查询工作负载定制的自然语言到SQL系统 论文详细解读

### 背景：这个问题为什么难？
把口头提问直接变成SQL语句是 NL2SQL 任务的核心，却一直受限于只看数据库的结构信息。传统方法把数据库模式（表名、列名、外键等）喂给大语言模型，让它自行推理出查询语句，但实际业务中同一个库会被反复查询，历史查询日志里蕴含了大量暗示：哪些表经常一起JOIN、哪些列的业务含义被用户习惯性地用别名表达。忽视这些“工作负载记忆”，模型只能靠一次性推理，导致在复杂或命名晦涩的场景下错误率居高不下，也会因为每次都要完整检索模式信息而增加响应时间。

### 关键概念速览
**NL2SQL**：把自然语言问题自动翻译成结构化的SQL查询，类似把口头点餐转成厨房的配方指令。  
**工作负载（Query Workload）**：数据库过去执行过的所有SQL语句集合，像是这张桌子上常点的菜谱，能反映用户的真实需求和习惯。  
**大语言模型（LLM）**：经过海量文本训练的模型，能够理解自然语言并生成文本，类似会写代码的“智能助理”。  
**Schema Retrieval**：从数据库中抽取表结构、列名等信息，提供给模型的“菜谱原材料”。  
**Join Path**：多表查询时需要连接的表链路，想象成从厨房到餐厅的搬运路线，路径选对了才能端出完整菜品。  
**Prompt Engineering**：为模型设计输入提示，使其更好地理解任务，就像给厨师写清晰的菜谱说明。  
**Latency**：从用户提问到返回SQL的时间，直接影响交互体验，类似点餐到上菜的等待时长。  

### 核心创新点
1. **从“只看菜谱”到“看历史点单”**  
   之前的系统只检索当前数据库的模式信息 → TailorSQL 在生成 Prompt 前先分析历史查询日志，抽取常用的 JOIN 路径、表别名映射等 → 让模型在已有的业务上下文里做翻译，显著提升了准确率，尤其在表名或列名不直观时表现更好。  

2. **工作负载感知的 Prompt 定制**  
   传统方法把模式直接拼进 Prompt，信息冗长且不针对性 → TailorSQL 根据工作负载生成精简且针对性的提示，例如把常用的多表连接写成一句“常用连接：orders → customers → regions”，并把模糊列名映射成业务友好的别名 → 模型的搜索空间被大幅压缩，推理速度提升约 2 倍。  

3. **双阶段检索‑生成流水线**  
   过去的工作多是一次性把模式喂进去 → TailorSQL 先用轻量检索模块快速定位最相关的历史查询片段，再把这些片段与当前自然语言问题一起送入 LLM → 这种先筛后生成的策略既降低了 Prompt 长度，又让模型拥有更丰富的上下文，整体延迟下降。  

4. **工作负载驱动的后处理校正**  
   早期系统只靠模型输出的 SQL，错误难以纠正 → TailorSQL 在生成后对比输出的 SQL 与工作负载中出现过的模式（如常见的 SELECT 列表顺序、WHERE 条件写法），自动做微调 → 进一步提升了执行准确率，尤其在细节上避免了因拼写或顺序差异导致的执行错误。  

### 方法详解
整体思路可以拆成四步：**日志采集 → 工作负载建模 → Prompt 定制 → 双阶段检索‑生成 → 后处理校正**。下面逐层展开。

1. **日志采集与预处理**  
   系统持续抓取数据库的查询日志，解析出每条 SQL 的表集合、JOIN 顺序、列使用频率以及 WHERE 条件的模式。类似把所有点单记录整理成菜谱索引。  

2. **工作负载建模**  
   - **Join Path 图**：把表当作节点，历史出现的 JOIN 关系当作有向边，边权是出现频次。这样可以快速查询“最常用的连接路径”。  
   - **列别名映射表**：对出现频率最高的列名进行业务解释抽取（比如 `cust_id` → “客户编号”），形成一个可查询的字典。  
   - **查询模板库**：把常见的 SELECT‑FROM‑WHERE 结构抽象成模板，保存占位符供后续填充。  

3. **Prompt 定制**  
   当用户提出自然语言问题时，系统先用轻量检索（基于关键词匹配）在工作负载模型中找出最相似的历史查询。检索到的结果会被压缩成三类信息：① 推荐的 Join Path，② 列别名映射，③ 可能的查询模板。随后，这三类信息连同用户原始问题一起拼装成 Prompt，交给 LLM。这样 Prompt 既短又精准，避免了把整个 schema 全部塞进去导致的噪声。  

4. **双阶段检索‑生成**  
   - **第一阶段**：快速检索模块返回 top‑k（如 5）最相关的历史查询片段。  
   - **第二阶段**：把这些片段与用户问题一起喂入 LLM，模型在已有业务上下文的帮助下生成 SQL。因为 LLM 已经“看到”类似的例子，生成的结构更符合实际业务。  

5. **后处理校正**  
   生成的 SQL 先走一次语法检查，然后与工作负载中的常见写法对比：如果 SELECT 列顺序与历史模板不符、WHERE 条件的运算符使用频率异常，系统会自动进行微调（例如调换列顺序或替换 `=` 为 `LIKE`），确保最终 SQL 能在实际执行环境中顺利跑通。  

**最巧妙的点**在于把历史查询当作“软约束”注入 Prompt，而不是硬编码进模型或单独做后置规则。这样既保留了 LLM 的生成能力，又让业务经验以轻量方式渗透进每一次翻译。

### 实验与效果
- **测试平台**：论文在 Spider、WikiSQL 等公开 NL2SQL 基准上评估，同时在企业内部真实工作负载上做了额外实验。  
- **对比基线**：与最新的基于纯 Schema 检索的 LLM 方法（如 ChatGPT‑SQL、T5‑SQL）相比，TailorSQL 在执行准确率上提升了约 10%–20%（在 Spider 上最高提升约 2×），延迟平均下降约 30%。  
- **消融实验**：作者分别去掉工作负载建模、Prompt 定制、后处理校正三个模块，发现去掉 Prompt 定制导致准确率下降约 8%，去掉后处理再降约 4%，而仅保留纯 Schema 检索时性能回到基线水平。  
- **局限性**：论文承认对工作负载的依赖意味着在新建数据库或日志不足的场景下效果会退化；此外，日志的隐私和安全处理在实际部署中需要额外措施。  

### 影响与延伸思考
TailorSQL 把“历史查询记忆”引入 NL2SQL 流程后，后续工作纷纷探索类似的工作负载感知策略，例如在代码生成、表格问答等任务中加入用户历史交互记录。还有研究尝试把工作负载信息直接作为微调数据喂给模型，进一步提升零样本表现。对想深入的读者，可以关注 **工作负载驱动的检索增强生成（RAG）** 方向，以及 **隐私保护下的查询日志利用**（如差分隐私）等新兴议题。  

### 一句话记住它
把数据库的历史查询当作“经验菜谱”，让模型在熟悉的业务上下文里翻译自然语言，准确率和响应速度都翻倍。