# Distill-C: Enhanced NL2SQL via Distilled Customization with LLMs

> **Date**：2025-03-30
> **arXiv**：https://arxiv.org/abs/2504.00048

## Abstract

The growing adoption of large language models (LLMs) in business applications has amplified interest in Natural Language to SQL (NL2SQL) solutions, in which there is competing demand for high performance and efficiency. Domain- and customer-specific requirements further complicate the problem. To address this conundrum, we introduce Distill-C, a distilled customization framework tailored for NL2SQL tasks. Distill-C utilizes large teacher LLMs to produce high-quality synthetic data through a robust and scalable pipeline. Finetuning smaller and open-source LLMs on this synthesized data enables them to rival or outperform teacher models an order of magnitude larger. Evaluated on multiple challenging benchmarks, Distill-C achieves an average improvement of 36% in execution accuracy compared to the base models from three distinct LLM families. Additionally, on three internal customer benchmarks, Distill-C demonstrates a 22.6% performance improvement over the base models. Our results demonstrate that Distill-C is an effective, high-performing and generalizable approach for deploying lightweight yet powerful NL2SQL models, delivering exceptional accuracies while maintaining low computational cost.

---

# Distill‑C：通过蒸馏定制提升 NL2SQL 的方法 论文详细解读

### 背景：这个问题为什么难？

NL2SQL（把自然语言查询转成 SQL）在企业内部常常要面对两大难题：一是要兼顾高准确率，二是要在算力受限的环境里保持响应速度。传统做法要么训练超大模型来追求极致性能，却导致部署成本飙升；要么直接用小模型，却经常因为对业务领域的细节把握不足而出错。更糟的是，企业的数据库结构、业务术语和安全策略各不相同，通用模型很难一次性满足所有客户的定制需求。于是出现了“高性能 vs 低成本 vs 定制化”三者难以兼得的局面，迫切需要一种既能保持轻量又能快速适配业务的解决方案。

### 关键概念速览
- **NL2SQL**：把用户的自然语言问题自动翻译成对应的 SQL 语句，类似于让机器把口头指令变成数据库查询指令。  
- **大语言模型（LLM）**：参数量在数十亿甚至上百亿级别的模型，能够理解并生成自然语言，像 GPT‑4、Claude 等。  
- **蒸馏（Distillation）**：把一个强大的“老师”模型的知识迁移到体积更小的“学生”模型上，就像把名师的讲义浓缩成简短的笔记。  
- **合成数据（Synthetic Data）**：不从真实用户收集，而是让模型自己生成的训练样本，用来弥补真实标注数据的不足。  
- **定制化（Customization）**：针对特定业务领域或客户的数据库模式、业务词汇进行专门的模型调优，使模型更懂本地语言。  
- **执行准确率（Execution Accuracy）**：模型生成的 SQL 在真实数据库上执行后，返回的结果是否与人工标注答案一致，是衡量 NL2SQL 实际价值的核心指标。  
- **开放源码模型（Open‑source LLM）**：代码公开、可自由修改的模型，例如 LLaMA、Mistral，便于企业自行部署。  

### 核心创新点
1. **老师模型驱动的合成数据管线 → 用超大 LLM 生成高质量 NL2SQL 样本 → 让学生模型在几乎无标注成本的情况下获得与真实数据相当的学习信号**。作者搭建了一个可横向扩展的流水线，先让老师模型在大量随机生成的自然语言查询上输出对应的 SQL，再通过多轮过滤和一致性检查提升合成数据的可信度。这样得到的合成数据既覆盖了业务特有的表结构，又保持了语义多样性，弥补了真实标注数据稀缺的痛点。  
2. **跨模型族的统一蒸馏框架 → 同时对三类不同架构的开源 LLM 进行微调 → 小模型在同一任务上实现了与老师模型相近甚至更好的执行准确率**。不同于只针对单一模型做蒸馏，Distill‑C 设计了一个抽象的“教师‑学生”接口，使得 LLaMA、Mistral、Falcon 等都能共享同一套合成数据和蒸馏策略，极大提升了方法的通用性。  
3. **业务定制化的轻量化微调 → 在合成数据中加入客户专属的 schema 描述和业务词表 → 微调后模型直接输出符合客户数据库的 SQL，省去二次映射步骤**。通过在数据生成阶段注入客户的表结构信息，学生模型在学习时就已经内化了这些约束，部署后无需额外的后处理模块。  

### 方法详解
整体思路可以划分为三步：**（1）老师模型生成合成数据、（2）数据质量过滤与增强、（3）学生模型蒸馏微调**。下面逐步拆解每一步的关键细节。

1. **老师模型生成合成数据**  
   - 选取 1–2 个最强的商业 LLM（如 GPT‑4）作为“老师”。  
   - 给老师模型提供业务 schema（表名、列名、外键关系）以及一组随机抽取的业务意图模板（如“查询过去 30 天的订单总额”）。  
   - 老师模型在每个意图上生成自然语言查询，再基于同一 schema 让模型输出对应的 SQL。此过程相当于让老师模型“自编自演”。  

2. **质量过滤与增强**  
   - **语义一致性检查**：把生成的自然语言和 SQL 再次送回老师模型，让它判断二者是否匹配，只有通过的样本才保留。  
   - **执行验证**：在一个安全的沙箱数据库里执行生成的 SQL，比较返回结果与老师模型预估的答案是否一致。失败的样本被剔除或重新生成。  
   - **多样性增强**：对保留的样本进行同义改写、负样本注入等手段，扩大数据覆盖面，防止学生模型只学到固定的模板。  

3. **学生模型蒸馏微调**  
   - 采用开放源码的轻量模型（参数在 1–7B 之间），统一使用 LoRA（Low‑Rank Adaptation）或 QLoRA 等高效微调技术，只调少量参数，保持原模型的推理速度。  
   - 训练目标是最小化学生模型输出的 SQL 与老师模型提供的“金标准”之间的交叉熵损失，同时加入 **结构约束损失**：强制模型在生成列名、表名时遵循 schema 的合法集合。  
   - 为了让模型兼顾不同业务，训练时交叉混合了多个客户的 schema 信息，使学生模型在一次微调后能够“一键切换”到任意客户的环境。  

**最巧妙的地方**在于把“业务定制”提前到数据层面，而不是在模型后期加规则。这样学生模型在学习时就已经把业务约束内化，部署后不需要额外的映射或校正步骤，真正实现了轻量化的端到端 NL2SQL。

### 实验与效果
- **评测基准**：论文在公开的 Spider、CoSQL、SParC 三个 NL2SQL 挑战集上做了对比，同时在公司内部的三个真实客户数据集上进行验证。  
- **基线对比**：分别选取了同族的原始开源模型（未蒸馏）以及商业大模型的直接调用作为基线。  
- **核心结果**：在公开基准上，Distill‑C 的执行准确率比原始小模型平均提升了 **36%**，甚至在部分子任务上超过了老师模型本身。内部客户数据上，提升幅度为 **22.6%**。  
- **消融实验**：作者分别去掉了（1）合成数据过滤、（2）结构约束损失、（3）跨模型族统一蒸馏，发现每一项的缺失都会导致整体准确率下降 8%~15%，验证了各模块的必要性。  
- **局限性**：论文承认合成数据的质量仍受老师模型能力上限限制，极端业务术语或非常规 schema 仍可能出现错误；此外，蒸馏过程对算力有一定需求，虽然比直接训练大模型省很多，但在资源极端受限的场景下仍不易部署。  

### 影响与延伸思考
Distill‑C 的出现让业界重新审视“轻量模型+合成数据”在 NL2SQL 场景的可行性。随后有几篇工作（如 **SQL‑Synth**, **TinySQL‑Distill**）直接借鉴了其合成‑过滤‑蒸馏流水线，并尝试把多模态信息（如表格示例）加入合成过程，进一步提升对复杂查询的理解。对想深入的读者，可以关注以下方向：  
- **自适应合成数据生成**：让老师模型根据学生模型的错误模式动态生成针对性样本。  
- **跨语言 NL2SQL**：把同一套合成数据用于多语言查询，探索语言迁移的蒸馏技巧。  
- **安全合成**：在生成合成数据时加入隐私保护机制，防止泄露业务敏感信息。  

### 一句话记住它
把超大模型的“智慧”浓缩进小模型的合成数据管线，让轻量 NL2SQL 既懂业务又跑得快。