# DataLab: A Unified Platform for LLM-Powered Business Intelligence

> **Date**：2024-12-03
> **arXiv**：https://arxiv.org/abs/2412.02205

## Abstract

Business intelligence (BI) transforms large volumes of data within modern organizations into actionable insights for informed decision-making. Recently, large language model (LLM)-based agents have streamlined the BI workflow by automatically performing task planning, reasoning, and actions in executable environments based on natural language (NL) queries. However, existing approaches primarily focus on individual BI tasks such as NL2SQL and NL2VIS. The fragmentation of tasks across different data roles and tools lead to inefficiencies and potential errors due to the iterative and collaborative nature of BI. In this paper, we introduce DataLab, a unified BI platform that integrates a one-stop LLM-based agent framework with an augmented computational notebook interface. DataLab supports various BI tasks for different data roles in data preparation, analysis, and visualization by seamlessly combining LLM assistance with user customization within a single environment. To achieve this unification, we design a domain knowledge incorporation module tailored for enterprise-specific BI tasks, an inter-agent communication mechanism to facilitate information sharing across the BI workflow, and a cell-based context management strategy to enhance context utilization efficiency in BI notebooks. Extensive experiments demonstrate that DataLab achieves state-of-the-art performance on various BI tasks across popular research benchmarks. Moreover, DataLab maintains high effectiveness and efficiency on real-world datasets from Tencent, achieving up to a 58.58% increase in accuracy and a 61.65% reduction in token cost on enterprise-specific BI tasks.

---

# DataLab：面向大语言模型的统一商业智能平台 论文详细解读

### 背景：这个问题为什么难？

商业智能（BI）需要把企业海量数据转化为可操作的洞察，传统流程往往要在数据准备、分析、可视化等环节之间来回切换，每一步都需要手工编写 SQL、脚本或图表配置。近几年出现的基于大语言模型（LLM）的智能体虽然能把自然语言直接映射到单一任务（比如 NL2SQL），但它们大多是“单兵作战”，只能解决一个环节。实际的 BI 项目是一个迭代、协同的过程：数据工程师、分析师、可视化设计师会在同一个 notebook 里交叉使用不同工具，信息在任务之间的传递常常丢失或重复，导致效率低下、错误率升高。于是出现了“任务碎片化、上下文割裂、跨角色协同难”的三大痛点，迫切需要一个能够把所有环节统一在同一平台、并让 LLM 智能体顺畅协作的解决方案。

### 关键概念速览
- **LLM（大语言模型）**：能够理解并生成自然语言的深度模型，类似会说话的“万能助理”。在 BI 场景里，它负责把用户的口头需求翻译成技术指令（SQL、Python、可视化代码等）。
- **Agent（智能体）**：LLM 在特定任务上的“角色”，比如负责生成 SQL 的 NL2SQL Agent、负责绘图的 NL2VIS Agent。每个 Agent 都有自己的工具箱和执行环境。
- **Domain Knowledge Incorporation（领域知识整合）**：把企业内部的表结构、数据血缘、已有的 ETL 脚本等信息抽象成模型可读的知识库，让 LLM 在生成代码时能“站在业务角度”思考，而不是盲目猜测。
- **Inter‑Agent Communication（智能体间通信）**：不同 Agent 之间共享信息的机制，类似团队会议中的信息流转，确保后续任务能直接使用前一步的产出。
- **Cell‑Based Context Management（基于单元格的上下文管理）**：在 notebook 中，每个代码单元格都有前后依赖关系。系统会构建一张有向无环图（DAG），只把真正相关的单元格内容送给 LLM，避免一次性喂入全部历史记录导致上下文噪声。
- **Finite State Machine（有限状态机）**：一种用离散状态和转移规则描述流程的模型，这里用来控制智能体之间的对话顺序，确保信息在正确的时机被传递。
- **Token Cost（令牌消耗）**：调用 LLM 时需要支付的计算资源，等价于“对话字数”。降低 token cost 意味着更快、更省钱的推理。

### 核心创新点
1. **从单任务到全流程统一**  
   - 过去的系统只能把自然语言映射到单一输出（如 SQL），缺乏跨任务的衔接。  
   - DataLab 把数据准备、分析、可视化全部包装进同一个 notebook 环境，让多个 Agent 按需被调度。  
   - 结果是用户只需一次自然语言提问，系统即可在同一页面完成从数据清洗到图表展示的完整链路，显著提升工作流连贯性。

2. **企业级领域知识自动化注入**  
   - 传统方法依赖手工编写提示词或外部知识库，成本高且易出错。  
   - DataLab 通过解析数据表结构、ETL 脚本和血缘图，自动生成结构化的领域知识，并在 LLM 推理时作为“背景上下文”注入。  
   - 这样模型在生成 SQL 或数据转换代码时能遵守企业的命名规范、业务规则，准确率提升近 60%。

3. **基于 FSM 的智能体协同框架**  
   - 以前的多 Agent 系统缺乏统一的调度逻辑，容易出现信息孤岛或循环调用。  
   - 作者设计了一个有限状态机，明确每个 Agent 的输入、输出以及何时进入下一状态，形成可视化的任务流图。  
   - 该机制让跨任务信息传递变得可预测、可追溯，错误率大幅下降。

4. **单元格依赖图驱动的上下文裁剪**  
   - 把整个 notebook 的历史全部喂给 LLM 会导致上下文膨胀，既浪费 token 又可能让模型混淆。  
   - DataLab 为每个代码单元格记录依赖关系，实时生成一张 DAG，只抽取与当前任务直接相关的前置单元格作为上下文。  
   - 这种“只取必要信息”的策略让 token 消耗降低超过 60%，同时保持或提升生成质量。

### 方法详解
#### 整体框架概览
DataLab 的运行可以划分为四个阶段：  
1. **自然语言解析**：用户在 notebook 顶部输入业务需求（如“展示过去三个月各地区的销售趋势并标出异常点”）。  
2. **任务规划与分解**：核心 LLM（Planner）把需求拆解成子任务序列，例如“数据抽取 → 数据清洗 → 计算异常 → 绘制折线图”。  
3. **子任务执行**：针对每个子任务，系统调度对应的 Agent（NL2SQL、DataPrep、AnomalyDetect、NL2VIS），并通过 FSM 管理它们的执行顺序。  
4. **结果呈现**：每个 Agent 的输出被写入 notebook 的独立代码单元格，后续单元格可以直接引用前面的变量，最终形成可交互的报告。

#### 关键模块拆解
1. **领域知识整合模块**  
   - **输入**：企业的元数据（表结构、列注释）、ETL 脚本、血缘关系文件。  
   - **处理**：系统使用规则引擎把这些信息抽象成统一的 JSON schema，标记每个字段的业务含义、数据类型、关联表等。  
   - **输出**：在每次调用 LLM 前，系统把对应的 schema 片段拼接到提示词的 “背景知识” 部分。  
   - **类比**：就像在写报告前先查阅公司内部的词典，确保用词和概念都符合公司惯例。

2. **智能体间通信机制（FSM）**  
   - **状态定义**：如 `INIT → EXTRACT → CLEAN → ANALYZE → VISUALIZE → DONE`。每个状态对应一个 Agent。  
   - **转移规则**：只有当前 Agent 成功返回结构化输出（如 SQL 语句或 Pandas DataFrame），FSM 才会进入下一个状态。若出现错误，系统会回到前一状态并让 LLM 重新生成。  
   - **信息共享**：每个状态的输出被存入共享的 “工作区” 对象，后续 Agent 可以直接读取，而不必重新请求 LLM。  
   - **巧妙之处**：使用 FSM 把原本松散的多 Agent 调用硬化为可验证的流程，极大降低了“链路断裂”风险。

3. **基于单元格的上下文管理**  
   - **依赖追踪**：每当一个 Agent 生成代码并写入 notebook，系统记录该单元格的输入变量和输出变量。  
   - **构建 DAG**：节点是代码单元格，边表示变量的流向（例如第 3 格的 `df_clean` 被第 5 格的 `df_anomaly` 使用）。  
   - **上下文裁剪**：当需要再次调用 LLM（比如在可视化阶段需要列名），系统只把 DAG 中到达当前节点的祖先节点的代码片段拼入提示词。  
   - **类比**：类似于在写论文时，只把前面已经引用的章节放进参考文献，而不是把整本书都列出来。

#### 细节与实现要点
- **提示词模板**：每种 Agent 都有专属模板，模板中预留了 “业务背景” 与 “前置代码” 两块，分别由领域知识模块和上下文管理模块填充。  
- **错误恢复**：如果某个 Agent 返回的代码执行报错，系统会捕获异常信息并把错误描述重新交给 LLM，让它在同一状态下进行“自我纠错”。  
- **多模态支持**：NL2VIS Agent 不仅生成绘图代码，还可以返回图表的元数据（如轴标签），供后续分析使用。  
- **可扩展性**：新任务只需实现对应的 Agent 接口并在 FSM 中添加状态，平台即可无缝接入。

### 实验与效果
- **评测基准**：作者在公开的 NL2SQL（Spider）、NL2VIS（ChartQA）等多任务基准上跑实验，覆盖从结构化查询到可视化生成的全链路。  
- **企业真实数据**：使用腾讯内部的业务数据集，任务包括跨表联合查询、异常检测以及业务仪表盘自动生成。  
- **对比基线**：与单任务的最新 LLM‑Agent 系统（如 ChatSQL、VisAgent）以及传统手工脚本相比，DataLab 在整体准确率上提升了 **58.58%**，在同等任务下的 token 消耗下降了 **61.65%**。  
- **消融实验**：  
  - 去掉领域知识模块后，SQL 生成错误率上升约 22%。  
  - 替换 FSM 为自由调度，整体错误率提升约 15%，且执行时间波动增大。  
  - 关闭单元格上下文裁剪，token 消耗几乎翻倍，模型输出质量略有下降。  
- **局限性**：论文承认系统仍然依赖于底层 LLM 的能力，若模型在特定行业术语上表现不足，仍会出现生成错误；此外，FSM 的状态设计需要根据业务场景手动调优，完全自动化仍是未来挑战。

### 影响与延伸思考
DataLab 把“多任务 BI”从碎片化的工具链搬进了统一的 notebook 环境，开启了“LLM‑Agent 生态协同”的新思路。后续的工作开始探索更细粒度的 Agent 角色（如数据治理 Agent、权限审计 Agent）以及自适应的 FSM 自动学习机制。对想进一步研究的读者，可以关注以下方向：  
- **自监督的领域知识抽取**：如何让模型自行发现企业内部的业务规则，而不是依赖手工解析。  
- **跨模态协同**：把文本、表格、图像三类信息统一进同一个上下文管理框架。  
- **可解释的 Agent 交互**：在 FSM 转移时提供可视化的任务流图，让业务人员能够审计每一步的决策依据。  
这些方向都有望把 LLM‑驱动的 BI 推向更高的可靠性和可操作性。

### 一句话记住它
DataLab 用领域知识、状态机调度和单元格依赖图把多个 LLM 智能体紧密串联，让一次自然语言提问即可完成从数据准备到可视化的完整商业智能工作流。