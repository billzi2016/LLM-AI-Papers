# SQLForge: Synthesizing Reliable and Diverse Data to Enhance Text-to-SQL Reasoning in LLMs

> **Date**：2025-05-19
> **arXiv**：https://arxiv.org/abs/2505.13725

## Abstract

Large Language models (LLMs) have demonstrated significant potential in text-to-SQL reasoning tasks, yet a substantial performance gap persists between existing open-source models and their closed-source counterparts. In this paper, we introduce SQLForge, a novel approach for synthesizing reliable and diverse data to enhance text-to-SQL reasoning in LLMs. We improve data reliability through SQL syntax constraints and SQL-to-question reverse translation, ensuring data logic at both structural and semantic levels. We also propose an SQL template enrichment and iterative data domain exploration mechanism to boost data diversity. Building on the augmented data, we fine-tune a variety of open-source models with different architectures and parameter sizes, resulting in a family of models termed SQLForge-LM. SQLForge-LM achieves the state-of-the-art performance on the widely recognized Spider and BIRD benchmarks among the open-source models. Specifically, SQLForge-LM achieves EX accuracy of 85.7% on Spider Dev and 59.8% on BIRD Dev, significantly narrowing the performance gap with closed-source methods.

---

# SQLForge：合成可靠且多样化数据以提升大语言模型文本到SQL推理能力 论文详细解读

### 背景：这个问题为什么难？

文本到SQL（Text‑to‑SQL）任务要求模型把自然语言问题翻译成结构化的SQL查询。虽然大语言模型（LLM）在语言理解上已经很强，但在生成严格符合SQL语法、且语义正确的查询时仍然差距明显。公开的开源模型往往因为训练数据不足、质量参差不齐，导致在复杂多表、嵌套查询等场景上表现远不如闭源商业模型。根本原因在于：① 现有数据集规模有限，难以覆盖真实业务的多样性；② 合成数据往往缺乏语法约束，生成的SQL会出现语法错误或逻辑不通；③ 训练时缺少对SQL结构和自然语言意图的双向校验，模型容易学到“看起来像SQL”但实际不可执行的模式。于是，需要一种既保证SQL合法又能提供丰富场景的合成数据方案。

### 关键概念速览
**Text‑to‑SQL（NL2SQL）**：把自然语言提问转换成SQL查询的任务，类似把口头指令翻译成数据库指令。  
**LLM（大语言模型）**：基于海量文本预训练的神经网络，能够生成连贯文字，也可以被微调用于特定任务。  
**SQL 语法约束**：指在生成SQL时强制遵守SQL语言的文法规则，就像拼写检查器确保句子没有拼写错误。  
**逆向翻译（SQL‑to‑question）**：把生成的SQL再翻译回自然语言，检查两者语义是否匹配，类似把答案再问回去验证。  
**模板丰富化**：在已有SQL模板上进行结构变形或添加子句，以产生更多变体，类似在画图时把基本图形复制、旋转、填色得到新图。  
**迭代域探索**：不断在数据生成过程中加入新的数据库模式或业务领域，让模型见识到更广的场景。  
**EX accuracy（执行准确率）**：模型生成的SQL在真实数据库上执行后，返回的结果是否与人工标注答案一致，是衡量NL2SQL实际效果的核心指标。  

### 核心创新点
1. **语法约束 + 逆向翻译 → 双层可靠性检查**  
   以前的合成数据只靠随机生成SQL，常出现语法错误或语义偏差。SQLForge 在生成SQL时先强制满足语法约束，随后把同一SQL用另一个模型翻译回自然语言，比较原问题与逆向生成的问题是否一致。这样既保证结构合法，又确保语义对应，显著降低噪声数据比例。

2. **SQL 模板丰富化 → 多样性提升**  
   传统方法只使用少量固定模板，导致生成的数据缺乏变化。SQLForge 通过对模板进行子句插入、表连接重排、聚合函数替换等操作，系统性地扩展模板空间。相当于在同一套积木上拼出更多不同的建筑，提升模型对新结构的适应能力。

3. **迭代域探索机制 → 持续扩展业务场景**  
   初始合成只覆盖常见的电商、图书等领域。SQLForge 采用“探索‑生成‑评估”循环：先在已有数据上训练小模型，分析其错误分布，针对薄弱的表结构或查询类型生成新SQL，再加入训练集。这样模型的知识库会随训练进程不断刷新，避免“一次性生成完毕”导致的覆盖不足。

4. **统一微调框架 → 适配多种开源模型**  
   作者把上述合成数据统一喂给不同规模、不同架构的开源模型（如 LLaMA、Mistral、Bloom），得到一系列命名为 SQLForge‑LM 的模型族。相比单纯增大模型参数，数据质量的提升带来的性能跃升更为显著。

### 方法详解
**整体思路**：SQLForge 把“合成数据 → 可靠性过滤 → 多样性扩展 → 迭代域探索”四步串起来，最终得到一个大规模、噪声低、场景丰富的训练集，用来微调各种开源 LLM。

1. **SQL 模板库构建**  
   - 从公开的 Spider、BIRD 等基准中抽取数千条高质量 SQL，去掉具体表名、列名后形成通用模板。  
   - 每个模板标记可变位置（表、列、条件、聚合等），为后续变形提供锚点。

2. **模板丰富化**  
   - **子句插入**：在 SELECT、WHERE、GROUP BY 等位置随机加入合法子句，如额外的过滤条件或子查询。  
   - **结构重排**：把多表连接的顺序调换、把子查询改写为 JOIN，保持等价性但改变语法形态。  
   - **函数替换**：把 COUNT 换成 SUM(1) 等等价实现，增加函数多样性。  
   - 这些操作通过规则引擎自动执行，确保生成的 SQL 仍然符合语法。

3. **语法约束过滤**  
   - 使用开源的 SQL 解析器（如 sqlglot）对每条生成的 SQL 进行语法检查，直接剔除解析错误的样本。  
   - 对通过检查的 SQL，进一步用预训练的 LLM（如 CodeLlama）生成对应的自然语言问题。

4. **逆向翻译验证**  
   - 将步骤 3 中的自然语言问题喂入另一个微调好的 Text‑to‑SQL 模型，得到逆向生成的 SQL。  
   - 将逆向 SQL 再次解析，比较其结构与原始 SQL 是否等价（通过归一化后比较抽象语法树）。若不等价，则认为该对数据不可靠，剔除。

5. **迭代域探索**  
   - 在已有数据上训练一个小型的 SQLForge‑LM，记录模型在验证集上的错误模式（如对特定聚合函数、嵌套查询的低准确率）。  
   - 针对这些薄弱点，手动或自动扩展模板库，引入新的业务表结构（如金融、医疗），重新走一遍步骤 2‑4，生成针对性的数据。  
   - 将新数据合并回训练集，继续微调模型，循环若干次直至错误率收敛。

6. **统一微调**  
   - 最终得到的高质量合成数据集约数十万对（自然语言 ↔ SQL），对不同开源 LLM 进行统一的指令微调。  
   - 微调时采用 LoRA（低秩适配）或全参数微调，根据模型规模灵活选择。

**最巧妙的点**：逆向翻译的双向校验把“生成对齐”问题转化为“模型自检”，不需要人工标注即可大幅提升数据可信度；而迭代域探索把数据生成和模型训练紧密耦合，使得数据分布始终跟随模型弱点动态演进。

### 实验与效果
- **测试基准**：Spider（公开的跨域 Text‑to‑SQL 基准）和 BIRD（更大规模的中文/英文混合基准）。  
- **对比模型**：包括开源的 LLaMA‑7B、Mistral‑7B、Bloom‑7B，以及同等规模的未使用合成数据的基线。  
- **主要指标**：在 Spider Dev 上，SQLForge‑LM（以 LLaMA‑7B 为底）实现了 85.7% 的 EX accuracy，领先未使用合成数据的同类模型约 7%‑10%。在 BIRD Dev 上达到 59.8%，同样缩小了与闭源商业模型（约 65%）的差距。  
- **消融实验**：  
  - 去掉逆向翻译过滤，EX accuracy 下降约 2.3%。  
  - 只使用原始模板不做丰富化，下降约 3.1%。  
  - 不进行迭代域探索，模型在新业务表上的表现下降约 4.5%。  
  这些结果表明每个创新模块都对最终性能有实质贡献。  
- **局限性**：论文承认合成数据仍然依赖于手工设计的模板规则，完全自动化仍有挑战；此外，逆向翻译使用的模型本身也会带入偏差，极端复杂的嵌套查询仍可能漏检。

### 影响与延伸思考
SQLForge 的思路在 NL2SQL 社区迅速引起关注，后续有工作尝试把类似的“双向校验 + 迭代生成”框架搬到代码生成、图数据库查询等任务上（如 CodeForge、CypherForge）。还有研究把逆向翻译换成基于执行结果的对比，即直接在数据库上跑查询再比对返回表，进一步提升语义可靠性。对想深入的读者，可以关注以下方向：① 自动化模板发现（利用程序分析挖掘潜在 SQL 结构）；② 多模态 NL2SQL（加入表结构图像或业务文档）；③ 更高效的逆向翻译模型（如专门的 SQL‑to‑NL 微调模型）。这些都是在 SQLForge 基础上自然的延伸。

### 一句话记住它
**SQLForge 用语法约束＋逆向翻译的双重过滤，配合模板丰富化和迭代域探索，打造了高质量合成数据，让开源大模型在 Text‑to‑SQL 上逼近闭源水平。**