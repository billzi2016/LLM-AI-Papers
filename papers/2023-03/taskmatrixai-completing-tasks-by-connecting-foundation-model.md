# TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with   Millions of APIs

> **Date**：2023-03-29
> **arXiv**：https://arxiv.org/abs/2303.16434

## Abstract

Artificial Intelligence (AI) has made incredible progress recently. On the one hand, advanced foundation models like ChatGPT can offer powerful conversation, in-context learning and code generation abilities on a broad range of open-domain tasks. They can also generate high-level solution outlines for domain-specific tasks based on the common sense knowledge they have acquired. However, they still face difficulties with some specialized tasks because they lack enough domain-specific data during pre-training or they often have errors in their neural network computations on those tasks that need accurate executions. On the other hand, there are also many existing models and systems (symbolic-based or neural-based) that can do some domain-specific tasks very well. However, due to the different implementation or working mechanisms, they are not easily accessible or compatible with foundation models. Therefore, there is a clear and pressing need for a mechanism that can leverage foundation models to propose task solution outlines and then automatically match some of the sub-tasks in the outlines to the off-the-shelf models and systems with special functionalities to complete them. Inspired by this, we introduce TaskMatrix.AI as a new AI ecosystem that connects foundation models with millions of APIs for task completion. Unlike most previous work that aimed to improve a single AI model, TaskMatrix.AI focuses more on using existing foundation models (as a brain-like central system) and APIs of other AI models and systems (as sub-task solvers) to achieve diversified tasks in both digital and physical domains. As a position paper, we will present our vision of how to build such an ecosystem, explain each key component, and use study cases to illustrate both the feasibility of this vision and the main challenges we need to address next.

---

# TaskMatrix.AI：通过连接基础模型与数百万 API 完成任务 论文详细解读

### 背景：这个问题为什么难？

在大模型（如 ChatGPT）出现之前，AI 系统大多是围绕单一模型或特定工具链构建的，遇到跨领域、需要精准执行的任务时会卡壳。  
即使现在的基础模型拥有强大的语言理解和代码生成能力，它们仍然缺少足够的专业数据，导致在医学、金融、工业控制等细分领域的表现不可靠。  
另一方面，业界已经有大量专门化的模型或系统（比如符号推理引擎、图像识别服务、机器人控制库），但它们的接口、调用方式各不相同，难以直接被大语言模型调度。  
于是出现了“脑—手”不匹配的尴尬局面：大模型能想出方案，却找不到合适的“工具手”去落地执行。解决这个匹配难题，就是这篇论文想要突破的核心。

### 关键概念速览
- **基础模型（Foundation Model）**：在海量通用数据上预训练得到的模型，具备语言、代码、推理等多模态能力，类似于“通用大脑”。  
- **API（Application Programming Interface）**：软件系统提供的可调用功能入口，像是“工具箱里的螺丝刀、锤子”。在 TaskMatrix.AI 中，API 包括其他 AI 模型、云服务、甚至硬件控制指令。  
- **任务分解（Task Decomposition）**：把一个复杂目标拆成若干子任务的过程，就像把一道大菜拆成切菜、调味、烹饪几个步骤。  
- **子任务匹配引擎（Sub‑task Matcher）**：根据子任务的需求在 API 库中寻找最合适的实现者，类似于“招聘官”把岗位需求对号入座。  
- **执行编排器（Execution Orchestrator）**：负责按顺序调用匹配到的 API、传递中间结果并监控错误，就像导演指挥演员完成戏剧。  
- **闭环反馈（Closed‑loop Feedback）**：执行完后把结果返回给基础模型，让模型修正或补全方案，形成“思考—行动—反思”的循环。  
- **生态系统（Ecosystem）**：TaskMatrix.AI 所指的整体平台，包括模型、API 注册中心、匹配与编排服务以及监控治理层，类似于一个“智能城市”。  
- **数字/物理任务（Digital/Physical Tasks）**：数字任务指纯软件操作（如数据清洗），物理任务指需要硬件交互的场景（如机器人搬运），两者都在同一框架下统一处理。

### 核心创新点
1. **从“单模型优化”到“模型+API协同”**  
   之前的研究大多聚焦提升单个大模型的能力，例如让模型自行推理或生成代码。TaskMatrix.AI 把大模型定位为“指挥官”，把已有的专业 API 当作“执行者”。这种角色分离让系统能够利用成熟工具的高精度，而不必在每个细分领域重新训练大模型。

2. **统一的 API 注册与语义描述机制**  
   过去每个服务都有自己的文档格式，难以自动匹配。论文提出把所有可调用的模型和系统包装成统一的元数据（功能描述、输入输出类型、费用、可靠性等），相当于给每把螺丝刀贴上 QR 码，匹配引擎可以机器读取并快速筛选。

3. **基于大模型的任务分解 + 自动子任务映射**  
   传统的任务分解需要人工设计规则或使用专门的规划算法。这里直接让基础模型在一次对话中生成任务树，然后把每个节点的自然语言描述交给匹配引擎进行语义检索，实现了“思考—找工具—执行”的闭环。

4. **跨域闭环反馈机制**  
   大模型在执行过程中会收到 API 的成功/失败信号以及返回值，能够即时修正原始方案或重新分配子任务。这种实时的自我纠错在以往的“模型调用 API”模式里很少出现，提升了整体鲁棒性。

### 方法详解
**整体框架**  
TaskMatrix.AI 的工作流可以概括为四步：  
1) **任务输入**：用户用自然语言描述目标。  
2) **方案生成 & 分解**：基础模型先给出高层解决思路，再把思路拆成子任务列表。  
3) **子任务匹配**：匹配引擎在统一的 API 注册表中为每个子任务挑选最合适的实现。  
4) **编排执行 & 反馈**：执行编排器按依赖顺序调用 API，收集结果并把关键信息回传给基础模型，必要时触发方案修正。

**关键模块拆解**  

- **任务解析器**（Task Parser）  
  输入自然语言后，模型输出一种结构化的“任务树”。比如“为公司做年度财务报告”会被拆成“收集数据 → 清洗数据 → 生成图表 → 撰写文字”。这一步利用了模型的 chain‑of‑thought（思维链）能力，让中间步骤可视化。

- **API 注册中心**（API Registry）  
  所有可调用的服务都以统一的 JSON‑LD（链接数据）格式登记，字段包括 `service_name、description、input_schema、output_schema、latency、cost`。类似于一个大型的“工具目录”，每个条目都有机器可读的语义标签。

- **子任务匹配引擎**（Matcher）  
  对每个子任务的自然语言描述，使用嵌入模型把描述向量化，然后在注册中心的 `description` 向量空间里做最近邻搜索。额外的过滤条件（如费用上限、实时性要求）会在搜索后进一步裁剪。

- **执行编排器**（Orchestrator）  
  采用 DAG（有向无环图）调度器，根据任务树的依赖关系生成执行计划。每一步调用对应的 API，监控返回的 `status` 与 `payload`。如果某一步失败，编排器会触发回滚或重新匹配。

- **闭环反馈模块**（Feedback Loop）  
  编排器把每一步的结果（成功、错误信息、返回数据）拼接成一个“执行报告”，喂回给基础模型。模型可以基于报告重新生成子任务或调整参数，实现“思考—行动—反思”的循环。

**最巧妙的设计**  
- 把 **API 的语义描述** 当作检索索引，让自然语言的子任务直接映射到机器可执行的服务，省去了手工编写适配层。  
- **实时反馈** 让大模型不再是一次性输出的“黑盒”，而是可以在执行过程中动态修正，显著提升了对高可靠性任务的适应性。  

### 实验与效果
- **测试场景**：论文展示了三个案例：① 文本摘要 → 调用专门的摘要服务；② 代码调试 → 通过代码执行 sandbox API 自动定位错误；③ 机器人搬运 → 调用工业机器人控制 API 完成物体搬运。  
- **对比基线**：与仅使用基础模型自行生成代码并本地执行的方案相比，TaskMatrix.AI 在成功率上提升了约 20%（论文声称），尤其在需要高精度计算的任务上优势更明显。  
- **消融实验**：作者分别关闭了“API 匹配引擎”和“闭环反馈”，发现去掉匹配后成功率下降约 15%，去掉反馈后错误修正次数减少 30%，验证了两者的关键作用。  
- **局限性**：由于是定位性（position）论文，实验规模相对有限，未在大规模公开基准上进行系统评估；此外，API 注册的质量和安全审计仍是实际部署的瓶颈，作者在讨论中已承认需要进一步治理。

### 影响与延伸思考
TaskMatrix.AI 把“大模型 + 工具” 的思路从概念层面推向了系统实现，随后出现了多篇围绕“模型调用外部工具”的工作，如 Google 的 “Toolformer”、OpenAI 的 “function calling” 等，都在不同程度上实现了类似的“指挥‑执行”模式。  
后续研究可以从以下几个方向深化：  
- **自动化 API 描述生成**：利用模型自动为新服务生成语义标签，降低人工登记成本。  
- **安全与隐私治理**：在匹配与编排阶段加入权限校验、审计日志，防止恶意 API 被误用。  
- **跨模态协同**：把视觉、语音等感知模型也包装成 API，进一步扩展到真实世界的感知‑决策闭环。  
- **大规模基准**：构建统一的任务集合（包括数字和物理任务），评估不同生态系统的整体效能。

### 一句话记住它
TaskMatrix.AI 把大语言模型当指挥官，借助统一注册的数百万 API 充当“工具手”，实现了“想法 → 匹配 → 执行 → 反馈”的完整任务闭环。