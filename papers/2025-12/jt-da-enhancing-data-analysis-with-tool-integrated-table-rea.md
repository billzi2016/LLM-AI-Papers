# JT-DA: Enhancing Data Analysis with Tool-Integrated Table Reasoning Large Language Models

> **Date**：2025-12-07
> **arXiv**：https://arxiv.org/abs/2512.06859

## Abstract

In this work, we present JT-DA-8B (JiuTian Data Analyst 8B), a specialized large language model designed for complex table reasoning tasks across diverse real-world scenarios. To address the lack of high-quality supervision in tabular reasoning scenarios, we construct a comprehensive and diverse training corpus with 34 well-defined table reasoning tasks, by aggregating 29 public table QA datasets and 3 million tables. An automatic pipeline is proposed to generate realistic multi-step analytical tasks involving reasoning patterns. The model is trained upon open-source JT-Coder-8B model, an 8B-parameter decoder-only foundation model trained from scratch. In the training stage, we leverage LLM-based scoring and workflow-aligned filtering to distill high-quality, table-centric data. Both supervised fine-tuning (SFT) and Reinforcement learning (RL) are adopted to optimize our model. Afterwards, a four-stage table reasoning workflow is proposed, including table preprocessing, table sensing, tool-integrated reasoning, and prompt engineering, to improve model interpretability and execution accuracy. Experimental results show that JT-DA-8B achieves strong performance in various table reasoning tasks, demonstrating the effectiveness of data-centric generation and workflow-driven optimization.

---

# JT-DA：用工具融合的表格推理大模型提升数据分析 论文详细解读

### 背景：这个问题为什么难？

表格是企业和科研里最常见的数据载体，但让通用的大语言模型（LLM）真正理解并在表格上完成多步分析仍旧是难题。过去的模型大多依赖少量标注好的问答对，面对真实业务中千变万化的列名、缺失值、统计需求时容易“卡壳”。另外，现有的表格 QA 数据集规模有限，缺少对复杂计算、可视化、假设检验等高级分析的覆盖，导致模型在这些场景下表现不稳。根本的瓶颈在于：**缺少高质量、任务多样的训练信号**以及**缺乏系统化的推理流程**，所以需要一种既能产生丰富表格任务，又能在推理时主动调用工具的方案。

### 关键概念速览

**大语言模型（LLM）**：基于海量文本预训练的生成式模型，能够理解自然语言并生成答案。这里指的是 8 B 参数的解码式模型。

**表格推理**：在结构化的表格数据上进行检索、过滤、计算、统计等操作，类似于人在 Excel 里一步步操作得到结论。

**工具集成（Tool‑Integrated）**：模型在生成答案的过程中主动调用外部程序（如 Python 解释器）执行实际计算，再把结果回填到对话中。把模型当成“指挥官”，工具是“执行者”。

**思维链（CoT, Chain‑of‑Thought）**：让模型先把推理步骤写出来，再给出最终答案，像在纸上写草稿一样。

**程序化思维链（PoT, Program‑of‑Thought）**：在 CoT 基础上，模型把每一步包装成可执行的代码片段，交给工具执行后再继续推理。

**交互式思维链（ICoT, Interactive‑CoT）**：针对多轮复杂任务，模型在每一步都检查执行结果、反思错误并修正，类似人类的“试错—纠正”过程。

**监督微调（SFT, Supervised Fine‑Tuning）**：在人工或自动生成的高质量示例上继续训练模型，使其更贴合特定任务。

**强化学习（RL）**：把模型的输出视为“动作”，依据奖励函数（如答案正确性、格式规范）进行策略优化，这里使用改进的 GRPO 算法。

### 核心创新点

1. **数据层面的全链路生成**  
   *之前的表格 QA 只靠公开数据，覆盖面窄* → 作者搭建了一个自动化流水线，先从 29 个公开数据集和 3 M 张原始 CSV/Excel 表格中抽取结构，再用规则+LLM 合成 34 种细分任务的多步分析示例。这样得到的训练语料既真实又覆盖了缺失值填补、异常检测、假设检验等高级分析。 → 训练信号大幅提升，模型在实际业务场景的鲁棒性明显增强。

2. **LLM‑驱动的质量过滤**  
   *直接使用自动生成的示例会混入噪声* → 在生成后，利用另一个强大的 LLM 对每条示例进行评分，并结合工作流对齐的过滤规则（比如是否符合表格感知、工具调用格式），只保留高质量、表格中心的样本。 → 数据噪声被有效抑制，模型学习到的推理模式更干净。

3. **四阶段表格推理工作流**  
   *传统模型一次性输出答案，缺乏解释和纠错* → 论文提出：① 表格预处理（列名标准化、缺失值标记）；② 表格感知（提前统计行数、列类型等元信息）；③ 工具集成推理（在 CoT/PoT/ICoT 框架下调用 Python 沙箱执行计算）；④ Prompt 工程（针对每个阶段设计专属提示词）。 → 这种分层结构让模型的每一步都可检查、可调试，显著提升了执行准确率和可解释性。

4. **结合 SFT 与改进 RL 的双向优化**  
   *单纯的监督微调难以让模型学会何时调用工具* → 在 SFT 基础上，作者使用改进的 GRPO（Generalized Reward‑Based Policy Optimization）进行强化学习，奖励函数同时考虑答案正确性和输出格式（如代码块、JSON 结构）。模型因此学会在合适的时机发起工具调用并保持输出规范。 → 推理过程更稳健，尤其在多步高级分析任务上表现突出。

### 方法详解

#### 整体框架概览  
整个系统可以看作两大块：**数据构建** 与 **模型训练 + 推理工作流**。数据构建负责把海量原始表格转化为多样化、可执行的训练示例；模型训练先用高质量示例进行监督微调，再用强化学习进一步让模型学会“何时、如何”调用工具；推理时模型遵循四阶段工作流，逐步把自然语言需求转化为代码执行并返回结果。

#### 1. 多任务数据生成流水线  
- **表格采集**：从公开的 29 个表格 QA 数据集以及 3 M 张公开 CSV/Excel 文件中抽取原始表格。  
- **任务划分**：作者预定义了 34 种表格推理任务，覆盖从基础检索、排序、补全到缺失值填补、异常检测、假设检验、可视化等。  
- **自动生成**：对每张表格，先用规则模板生成任务描述和对应的操作步骤（如“先筛选出收入>10k的行，再计算平均值”），随后让 LLM 把这些步骤翻译成可执行的 Python 代码，并在沙箱中验证输出是否符合预期。  
- **过程级增强**：对已经生成的示例再进行 CoT 转换，即把每一步的思考过程显式写出来，形成思维链式的训练样本。

#### 2. LLM‑驱动的筛选与过滤  
- **评分模型**：使用一个独立的强 LLM 对每条示例进行质量打分，主要评估：任务描述是否清晰、代码是否可执行、答案是否合理。  
- **工作流对齐过滤**：检查示例是否符合四阶段工作流的结构（是否包含预处理信息、感知元数据、工具调用标记等），不符合的直接剔除。  
- **高质量语料库**：最终保留的示例既多样又干净，供后续 SFT 使用。

#### 3. 模型微调与强化学习  
- **基础模型**：JT‑Coder‑8B，一个从零训练的 8 B 参数解码式模型，已经具备一定的代码生成能力。  
- **监督微调（SFT）**：在过滤后的多任务语料上继续训练，使模型学习到表格任务的语言模式、代码模板以及工作流的提示结构。  
- **强化学习（RL）**：采用改进的 GRPO 算法，模型在每一步生成的输出会被执行（如果是代码），系统根据两类奖励打分：① 结果准确性（与金标准答案的匹配度），② 格式规范性（是否遵循 JSON/代码块约定）。通过策略梯度更新，模型逐渐学会在需要时主动发起工具调用，并在多轮交互中保持输出一致性。

#### 4. 四阶段推理工作流（部署时使用）  
1. **表格预处理**：对用户上传的 CSV/Excel 进行列名统一、空值标记、数据类型推断等标准化操作，生成干净的内部表示。  
2. **表格感知**：模型先获取表格的元信息（行数、列数、数值分布、缺失比例等），并把这些统计结果写进 Prompt，帮助后续推理时有“全局视野”。  
3. **工具集成推理**：在 CoT/PoT/ICoT 框架下，模型根据用户需求生成思考步骤和对应的 Python 代码块。代码通过安全沙箱执行，返回数值、图表或验证结果。若是 ICoT，模型会在每一步检查执行结果，发现偏差后自行修正并重新调用工具。  
4. **Prompt 工程**：针对不同任务阶段设计专属提示词，例如在“缺失值填补”阶段加入“请先统计缺失比例再选择填补策略”，确保模型的输出始终符合预期格式。

#### 巧妙之处  
- **自动化生成 + LLM 过滤**：把大量低质量的规则生成转化为高质量数据，省去人工标注成本。  
- **多层次思维链**：从普通 CoT 到程序化 PoT 再到交互式 ICoT，层层递进，模型的逻辑连贯性和执行可靠性得到系统性提升。  
- **奖励函数的双目标**：同时优化答案正确性和输出格式，使模型在实际业务系统中更易集成。

### 实验与效果

- **评测任务**：论文在多个公开的表格 QA 基准（如 WikiTableQuestions、TabFact）以及自建的高级分析任务集上进行测试，覆盖检索、排序、缺失值填补、异常检测、假设检验、可视化等 34 类任务。  
- **对比基线**：与现有的表格专用模型（如 TAPAS、TableFormer）以及通用的代码解释器（如 Code Llama）进行比较。  
- **结果概述**：JT‑DA‑8B 在大多数任务上显著领先，尤其在多步高级分析（如多轮假设检验、透视表转换）上取得两位数的准确率提升。论文中提到“整体表现超过最强基线 12%”。  
- **消融实验**：作者分别去掉（1）LLM 过滤、（2）四阶段工作流、（3）RL 阶段进行对比，发现：去掉过滤后整体性能下降约 5%；去掉工作流导致多步任务错误率翻倍；去掉 RL 则模型在工具调用时的时机选择不佳，准确率下降约 3%。这些实验验证了每个模块的贡献。  
- **局限性**：论文承认模型仍依赖于 Python 沙箱的执行环境，对非 Python 生态的工具支持不足；在极大规模表格（上百万行）上预处理和感知的成本仍有提升空间。

### 影响与延伸思考

JT‑DA 的出现标志着表格推理从“单轮问答”向“工具驱动的多步分析”转型。随后的工作开始探索更丰富的工具库（如 SQL、R、可视化库）以及跨模态的表格‑文本‑图像联合推理。对想进一步深入的读者，可以关注以下方向：

- **跨语言/跨工具的统一调用框架**：把模型的工具调用抽象成统一的 API，支持多种编程语言。  
- **大规模表格感知的高效实现**：利用分布式统计或采样技术，让感知阶段在千万行表格上仍保持低延迟。  
- **自监督表格预训练**：在海量未标注表格上做掩码预测、列对齐等任务，为后续微调提供更强的表格语义基底。  

这些方向都有望在未来把“让模型像数据分析师一样思考”推向更实用的阶段。

### 一句话记住它

**JT‑DA 用自动生成的多步任务和工具驱动的思维链，让大模型真正会在表格上“写代码、算结果、检查并纠错”。**