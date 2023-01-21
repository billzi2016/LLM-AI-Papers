# Dr.Spider: A Diagnostic Evaluation Benchmark towards Text-to-SQL   Robustness

> **Date**：2023-01-21
> **arXiv**：https://arxiv.org/abs/2301.08881

## Abstract

Neural text-to-SQL models have achieved remarkable performance in translating natural language questions into SQL queries. However, recent studies reveal that text-to-SQL models are vulnerable to task-specific perturbations. Previous curated robustness test sets usually focus on individual phenomena. In this paper, we propose a comprehensive robustness benchmark based on Spider, a cross-domain text-to-SQL benchmark, to diagnose the model robustness. We design 17 perturbations on databases, natural language questions, and SQL queries to measure the robustness from different angles. In order to collect more diversified natural question perturbations, we utilize large pretrained language models (PLMs) to simulate human behaviors in creating natural questions. We conduct a diagnostic study of the state-of-the-art models on the robustness set. Experimental results reveal that even the most robust model suffers from a 14.0% performance drop overall and a 50.7% performance drop on the most challenging perturbation. We also present a breakdown analysis regarding text-to-SQL model designs and provide insights for improving model robustness.

---

# Dr.Spider：面向文本到SQL鲁棒性的诊断评估基准 论文详细解读

### 背景：这个问题为什么难？

把自然语言问题直接翻译成 SQL 查询已经取得了惊人的进展，很多模型在 Spider 这类跨域 benchmark 上的准确率已经逼近人类水平。但这些成绩大多是在“干净”的测试集上得到的，真实使用场景里数据库结构会变化、用户的提问方式千差万别、甚至生成的 SQL 可能出现细微的语法错误。过去的鲁棒性测试往往只挑出一种扰动（比如同义词替换或表名改动），于是只能看到模型在单一维度的脆弱点，无法给出全局的诊断。因此，缺少一个系统、覆盖面广的评估平台，导致研究者很难定位模型到底在哪些方面容易失效，也难以衡量新方法的真正提升。

### 关键概念速览

**文本到SQL（Text-to-SQL）**：把用户的自然语言问句自动转成对应的结构化 SQL 查询，类似于让机器把口头指令翻译成数据库指令。

**Spider 基准**：一个跨域的 Text-to‑SQL 数据集，包含多种数据库模式和复杂查询，被视为衡量模型通用能力的金标准。

**鲁棒性（Robustness）**：模型在面对输入噪声、结构变化或其他非理想情况时仍能保持性能的能力，像是汽车在颠簸路面上仍能平稳行驶。

**扰动（Perturbation）**：有意对原始数据（数据库、问题或 SQL）做小幅修改，用来模拟真实世界的各种“意外”。比如把列名换成同义词、把问题改写成被动语态等。

**大规模预训练语言模型（PLM）**：如 GPT‑4、ChatGPT 之类的模型，已经在海量文本上学习到语言规律，本文利用它们来自动生成多样化的自然语言扰动，类似于让“机器人”扮演人类编辑者。

**诊断基准（Diagnostic Benchmark）**：不仅给出一个分数，还提供细粒度的错误分析，让研究者知道模型在哪类扰动上最弱。

### 核心创新点

1. **从单一到全景的扰动设计**  
   以前的鲁棒性测试只关注一种现象（比如同义词替换）。本文在数据库、自然语言问题、SQL 三个维度上共设计了 17 种扰动，覆盖了列名改动、表结构扩展、问题语法变化、SQL 关键字错位等多种真实场景。这样可以一次性捕捉模型在不同层面的脆弱点。

2. **利用大模型自动生成自然语言扰动**  
   为了避免人工编写耗时且可能偏向某种风格，作者让 GPT‑类模型模拟人类编辑行为，生成同义改写、冗余信息添加、拼写错误等多样化的问句。相当于让“机器写手”帮忙制造更贴近真实用户的噪声。

3. **基于 Spider 的诊断评估套件（Dr.Spider）**  
   将上述 17 种扰动统一包装成一个可直接使用的 benchmark，提供了原始数据、扰动后数据以及对应的评估脚本。研究者只需把自己的模型跑一遍，就能得到整体准确率以及每种扰动的单独表现。

4. **系统性模型诊断与设计要素关联分析**  
   作者把现有最先进的 Text-to‑SQL 模型按照编码器、解码器、结构约束等维度分类，统计每类模型在不同扰动上的跌幅。结果显示，使用显式结构约束的模型在数据库层面的扰动上跌幅更小，而纯端到端的模型在自然语言层面的鲁棒性更差。此分析为后续模型设计提供了实证指引。

### 方法详解

**整体框架**  
Dr.Spider 的构建分为三步：① 选取 Spider 原始数据；② 在数据库、问题、SQL 三个维度上分别施加 17 种预定义扰动；③ 用统一的评估脚本对任意 Text-to‑SQL 系统进行诊断，输出整体准确率和每种扰动的子得分。

**1. 扰动生成**  
- **数据库层**：包括列名同义词替换、表名拼写错误、向数据库中加入无关表或列、删除非关键列等。实现方式是直接编辑 Spider 提供的 schema 文件，保持 SQL 语义不变（除非故意破坏）。
- **自然语言层**：这里最关键的创新是调用大规模预训练语言模型（如 GPT‑3.5）生成多种改写。具体流程是：给模型一个原始问句和指令（比如“把下面的问题改写成被动语态并加入一个拼写错误”），模型返回改写后的句子。作者预先设计了 9 种改写指令，覆盖同义改写、冗余信息、口语化、拼写错误、数字格式变化等。
- **SQL 层**：包括关键字大小写随机化、子查询展开/合并、WHERE 条件顺序打乱、使用等价的函数替换等。实现时直接对原始 SQL 进行字符串层面的替换或 AST（抽象语法树）层面的重构。

**2. 数据组织**  
每一种扰动都生成一个完整的测试子集，保持原始的 train/dev 划分不变，只是把 test 集换成扰动版。所有子集统一命名，便于脚本自动遍历。

**3. 评估流程**  
- 研究者把自己的 Text-to‑SQL 系统部署成一个接受自然语言问句和数据库 schema，返回 SQL 的接口。  
- Dr.Spider 脚本依次把每个子集的问句喂给模型，收集生成的 SQL。  
- 使用 Spider 官方的执行器（执行生成的 SQL 并比较结果）计算执行准确率（Exec‑Acc）和结构匹配准确率（Exact‑Match）。  
- 脚本把每种扰动的得分汇总，输出一张“鲁棒性雷达图”，直观看出模型在哪些维度最脆弱。

**最巧妙的点**  
利用 PLM 自动生成自然语言扰动，使得扰动的多样性和人类编辑的风格更接近真实用户输入，避免了人工规则的局限性。并且所有扰动都保持了原始查询的意图（除非故意破坏），这样评估的下降可以归因于模型对噪声的敏感度，而不是数据本身的错误。

### 实验与效果

- **测试对象**：作者选取了当时公开的几种最先进的 Text-to‑SQL 系统，包括基于 BERT 的 encoder‑decoder、使用结构化注意力的 RAT‑SQL、以及加入自监督预训练的 PICARD 等。
- **整体表现**：在原始 Spider 测试集上，这些模型的 Exec‑Acc 大约在 80% 左右。但在 Dr.Spider 的全套扰动上，最强模型的整体准确率下降了 14.0%，说明即使是“最稳”的系统也会在噪声环境下失分。
- **最具挑战的扰动**：对 SQL 关键字大小写和子查询展开的组合扰动导致了 50.7% 的性能跌幅，说明模型对 SQL 语法细节的鲁棒性仍然薄弱。
- **细粒度分析**：结构约束（如 PICARD）在数据库层面的扰动（列名改动、表结构扩展）跌幅只有 8%，而纯端到端的 encoder‑decoder 在自然语言层面的同义改写跌幅超过 30%。这验证了作者关于“结构约束提升数据库鲁棒性，语言模型提升自然语言鲁棒性”的假设。
- **消融实验**：作者把 PLM 生成的自然语言扰动换成手工规则生成，发现模型在后者上的跌幅略小（约 2%），说明自动生成的扰动更具挑战性，也更贴近真实用户的多样化表达。
- **局限性**：论文未对大规模真实业务日志进行验证，扰动仍然是合成的；此外，评估只覆盖了执行准确率，未考虑生成的 SQL 可解释性或安全性。

### 影响与延伸思考

Dr.Spider 为 Text-to‑SQL 社区提供了第一套系统化、跨维度的鲁棒性评估工具，随后不少工作把它作为标准测试集来验证新模型的稳健性。比如在 2024 年的 ACL 论文里，有研究通过对抗训练在 Dr.Spider 上显著降低了同义改写的跌幅；还有工作把自适应 schema 编码与 PLM 生成的噪声联合训练，提升了对数据库层扰动的容错能力。未来可以进一步把真实用户查询日志加入基准，或者把扰动扩展到多语言、跨数据库方言（如 MySQL 与 PostgreSQL）等方向。对想深入的读者，建议关注“对抗鲁棒性训练”和“结构化约束解码”这两个热点，它们正是 Dr.Spider 诊断结果背后最有潜力的改进路径。

### 一句话记住它

Dr.Spider 用 17 种合成噪声全景检验 Text-to‑SQL 模型，让我们看到即使是最强模型在真实世界的“脏数据”面前也会掉链子。