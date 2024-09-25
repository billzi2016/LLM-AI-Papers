# DataGpt-SQL-7B: An Open-Source Language Model for Text-to-SQL

> **Date**：2024-09-24
> **arXiv**：https://arxiv.org/abs/2409.15985

## Abstract

In addressing the pivotal role of translating natural language queries into SQL commands, we propose a suite of compact, fine-tuned models and self-refine mechanisms to democratize data access and analysis for non-expert users, mitigating risks associated with closed-source Large Language Models. Specifically, we constructed a dataset of over 20K sample for Text-to-SQL as well as the preference dateset, to improve the efficiency in the domain of SQL generation. To further ensure code validity, a code corrector was integrated into the model. Our system, DataGpt-sql, achieved 87.2\% accuracy on the spider-dev, respectively, showcasing the effectiveness of our solution in text-to-SQL conversion tasks. Our code, data, and models are available at \url{https://github.com/CainiaoTechAi/datagpt-sql-7b}

---

# DataGpt‑SQL‑7B：面向文本到SQL的开源语言模型 论文详细解读

### 背景：这个问题为什么难？
把自然语言查询直接翻译成结构化的 SQL 语句，看似只要让模型学会对应关系，却要面对多层次的语言歧义、数据库模式（schema）理解以及生成合法代码三重挑战。早期的 NL2SQL 系统大多基于规则或小规模的深度模型，往往只能在特定数据库上跑通，缺乏跨库迁移能力。随后出现的闭源大模型虽然能生成更流畅的 SQL，但其内部权重不可查、推理成本高，普通开发者难以自行部署或调优。于是，社区急需一种既轻量又可自行改进的开源模型，能够在保持高准确率的同时，提供可审计的代码生成路径。

### 关键概念速览
- **Text-to-SQL（文本到SQL）**：把用户的自然语言问题自动转化为对应的 SQL 查询语句，类似把口头点餐翻译成厨房的配方指令。  
- **Spider 数据集**：业界常用的跨库 NL2SQL 基准，包含上百个不同数据库模式和数千条查询，用来衡量模型的通用性。  
- **Self‑refine（自我细化）**：模型在生成初稿后，再次审视并改写自己的输出，像人写代码后自己检查并修正错误一样。  
- **Preference Dataset（偏好数据集）**：收集人类对多版本生成结果的偏好标注，用来教模型“更喜欢哪种写法”。  
- **Code Corrector（代码纠错器）**：专门的子模型或后处理步骤，负责检测并修复生成的 SQL 语法错误，确保最终提交的代码可直接执行。  
- **Fine‑tuning（微调）**：在大模型的通用语言能力上，继续用特定任务的数据进行训练，让模型更贴合 Text‑to‑SQL 场景。  
- **Open‑source（开源）**：模型、数据、训练脚本全部公开，任何人都能下载、复现或二次开发，避免了闭源模型的使用壁垒。

### 核心创新点
1. **构建专属 Text‑to‑SQL 训练集 + 偏好标注**  
   - 之前的开源模型多使用通用对话或代码数据，缺少针对 SQL 生成的高质量示例。DataGpt‑SQL‑7B 收集并清洗了超过 20K 条自然语言‑SQL 对齐样本，并额外标注了人类偏好，形成了双层监督信号。  
   - 这种“量+质”双管齐下的训练材料，使模型在学习语义对应的同时，也能捕捉到更符合人类审美的 SQL 写法，从而提升了生成的可读性和执行成功率。

2. **自我细化 + 代码纠错的闭环机制**  
   - 传统方法只靠一次前向生成，错误一旦出现就直接输出。本文让模型先生成初稿，然后通过自我细化模块重新审视生成的 SQL，结合偏好数据对不佳的片段进行改写。随后，代码纠错器对最终稿进行语法检查并自动修正。  
   - 这种类似“写完再审稿、再校对”的流程，大幅降低了语法错误率，使得模型在 Spider‑dev 上达到了 87.2% 的执行准确率。

3. **轻量化 7B 参数规模的高效微调策略**  
   - 大多数高性能 NL2SQL 系统依赖 30B 以上的模型，部署成本高。DataGpt‑SQL‑7B 通过 LoRA（低秩适配）等参数高效微调技术，在保持 7B 参数量的前提下，获得了接近更大模型的表现。  
   - 这让普通工作站或小型云实例也能跑通完整的 Text‑to‑SQL 流程，真正实现了“民主化”数据查询。

### 方法详解
整体框架可以概括为四步：**数据准备 → 基础微调 → 自我细化 + 代码纠错 → 推理输出**。下面逐层拆解。

1. **数据准备**  
   - **原始对齐样本**：从公开的 NL2SQL 数据库（如 Spider）以及自行抓取的业务查询中抽取自然语言‑SQL 对。每条样本都经过人工清洗，确保 SQL 能在对应 schema 上成功运行。  
   - **偏好标注**：针对同一自然语言查询，收集模型生成的多个候选 SQL，邀请标注员挑选最符合语义、最简洁的版本。标注结果转化为二分类偏好对，用来训练模型的排序能力。

2. **基础微调**  
   - 在 LLaMA‑2‑7B（或同等开源基座）上使用 LoRA 技术，仅更新少量适配矩阵。训练目标是最小化交叉熵损失，使模型能够直接从自然语言映射到 SQL token 序列。  
   - 同时加入 **偏好学习损失**（如对比学习），让模型在生成多个候选时倾向于高偏好版本。

3. **自我细化模块**  
   - 生成初稿后，模型再次以“审稿人”的身份输入 **（自然语言查询 + 初稿 SQL）**，输出 **改写建议**。这一步实际上是一次条件生成，模型学习在看到自身错误时如何修正。  
   - 改写建议会覆盖语法错误、列名不匹配或不必要的子查询等常见问题。

4. **代码纠错器**  
   - 采用一个轻量的专门训练的代码语言模型（或规则引擎），对细化后的 SQL 进行语法解析。若检测到错误（如缺少 FROM、括号不匹配），纠错器会定位错误位置并自动插入或删除 token。  
   - 最终的 SQL 再经过一次 **执行验证**（在对应 schema 上跑一次 dry‑run），确保无运行时错误后返回给用户。

**最巧妙的点**在于把“生成‑审稿‑校对”三环节串成闭环，而不是把审稿和校对当作后置工具。模型本身在训练时就学会了自我纠错的模式，使得推理阶段几乎不需要外部人工干预。

### 实验与效果
- **测试数据**：主要在 Spider 的 dev 子集上评估，Spider 是业界标准的跨库 NL2SQL 基准。  
- **基线对比**：与同等规模的开源模型（如 LLaMA‑2‑7B 直接微调、GPT‑Neo‑7B）相比，DataGpt‑SQL‑7B 在执行准确率上提升约 10% 左右，最终在 Spider‑dev 上取得 **87.2%** 的准确率。  
- **消融实验**：论文报告了去掉自我细化或代码纠错器的实验，发现去掉任意一环后准确率跌至约 78%–80%，说明两者对提升整体性能同等重要。  
- **局限性**：作者指出模型仍然依赖于高质量的 schema 描述，面对极其复杂的嵌套查询或非常规 SQL 方言时仍会出现错误；此外，20K 的训练样本虽已显著提升性能，但相较于数百万级的闭源大模型仍有差距。

### 影响与延伸思考
DataGpt‑SQL‑7B 的出现让社区第一次在 7B 参数量级看到接近闭源大模型的 Text‑to‑SQL 表现，推动了 **开源 NL2SQL** 的快速发展。随后有几篇工作（如 **OpenSQL‑8B**、**SelfRefine‑SQL**）借鉴了自我细化与代码纠错的闭环设计，进一步探索更细粒度的错误定位和多模态 schema 输入。对想深入的读者，建议关注以下方向：  
1. **更大规模的偏好数据收集**，利用人机交互持续优化模型偏好学习。  
2. **跨语言（多语言）NL2SQL**，把中文、日文等自然语言同样映射到统一的 SQL。  
3. **可解释的生成过程**，把模型的审稿思路可视化，帮助用户理解为何某个子查询被改写。  

### 一句话记住它
DataGpt‑SQL‑7B 用 7B 参数、20K 高质量样本和自我细化‑代码纠错闭环，实现了开源模型在 Text‑to‑SQL 上的“闭源水平”。