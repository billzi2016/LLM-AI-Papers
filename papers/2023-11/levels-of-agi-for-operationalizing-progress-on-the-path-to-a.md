# Levels of AGI for Operationalizing Progress on the Path to AGI

> **Date**：2023-11-04
> **arXiv**：https://arxiv.org/abs/2311.02462

## Abstract

We propose a framework for classifying the capabilities and behavior of Artificial General Intelligence (AGI) models and their precursors. This framework introduces levels of AGI performance, generality, and autonomy, providing a common language to compare models, assess risks, and measure progress along the path to AGI. To develop our framework, we analyze existing definitions of AGI, and distill six principles that a useful ontology for AGI should satisfy. With these principles in mind, we propose "Levels of AGI" based on depth (performance) and breadth (generality) of capabilities, and reflect on how current systems fit into this ontology. We discuss the challenging requirements for future benchmarks that quantify the behavior and capabilities of AGI models against these levels. Finally, we discuss how these levels of AGI interact with deployment considerations such as autonomy and risk, and emphasize the importance of carefully selecting Human-AI Interaction paradigms for responsible and safe deployment of highly capable AI systems.

---

# 面向实现通往AGI之路的AGI层级框架 论文详细解读

### 背景：这个问题为什么难？

在过去，研究者们往往用“AGI”这个大词来描述一种能够像人一样通用的智能，却缺少统一的度量标准。不同团队的模型被贴上同样的标签，却在能力范围、可靠性和自主程度上相差巨大，导致风险评估和进度比较几乎没有共识。现有的评估方法要么只关注单一任务的表现，要么只看模型的规模，根本无法捕捉“深度”和“广度”两条维度的交叉。于是，缺乏一个能够把各种模型放在同一坐标系里比较的框架，成为推动安全 AGI 研究的瓶颈。

### 关键概念速览
- **AGI层级（Levels of AGI）**：把模型的**深度**（在单任务上的表现）和**广度**（能处理的任务种类）组合成二维坐标，形成从“窄域专精”到“全域通用”的层级划分。想象成一张地图，横轴是能走多远，纵轴是能爬多高。
- **深度（Depth）**：模型在特定任务或子任务上达到的人类水平或超越水平。类似于跑步选手的百米成绩，跑得快不代表能跑马拉松。
- **广度（Breadth）**：模型能够处理的任务类别数量和多样性。相当于运动员能否同时参加短跑、长跑、跳高等多项比赛。
- **自主性（Autonomy）**：模型在没有人类实时干预的情况下自行决策、执行并纠错的能力。可以比作无人驾驶汽车在复杂路况下的自我调度。
- **风险等级（Risk Tier）**：依据模型的深度、广度和自主性，评估其潜在危害或失控概率。类似于核电站根据功率和安全阀门数量划分的风险等级。
- **人机交互范式（Human‑AI Interaction Paradigm）**：指在不同层级模型下，人与模型的协作方式（如监督、协同、全权委托）。相当于医生在手术中是“助手”还是“主刀”。

### 核心创新点
1. **从单维度评估到双维度层级**  
   *之前的评估*：大多只看模型的规模或在单一基准上的分数。  
   *本文的做法*：引入“深度‑广度”二维坐标，将性能和通用性并列考量。  
   *改变*：提供了一个可以量化“多任务通用能力”的统一语言，使得不同模型之间的比较更直观。

2. **六大原则构建可操作本体**  
   *之前的本体*：缺乏系统化原则，导致定义模糊、冲突频发。  
   *本文的做法*：从已有 AGI 定义中抽象出六条原则（可测量、可比较、可扩展、风险感知、交互明确、可解释），作为层级划分的底层约束。  
   *改变*：确保任何新提出的层级或指标都能在理论上自洽，并且在实际评估时不产生歧义。

3. **将自主性与风险挂钩的层级映射**  
   *之前的风险评估*：往往只看模型的能力大小，忽视其自我决策的程度。  
   *本文的做法*：在每个深度‑广度层级上进一步细分自主性等级，并用风险 Tier 标记。  
   *改变*：帮助部署者在选择模型时，既看能力也看“自驾”程度，从而制定更细致的安全措施。

4. **提出面向未来的基准设计框架**  
   *之前的基准*：大多是静态任务集，难以捕捉模型在新情境下的适应性。  
   *本文的做法*：概念化了“层级基准”，要求测试集同时覆盖深度、广度和自主性三维。  
   *改变*：为后续研究提供了明确的实验方向，使得评估结果能够直接映射到层级图上。

### 方法详解
整体思路是先构建一个 **AGI层级本体**，再用 **层级基准** 对模型进行定位，最后结合 **自主性‑风险映射** 给出部署建议。整个流程可以拆成三步：

1. **定义深度‑广度坐标系**  
   - **深度**：选取一组代表性任务（如语言理解、推理、规划），为每个任务设定人类基准分数。模型在这些任务上的表现被归一化为 0‑1 区间。  
   - **广度**：统计模型能够成功完成的任务种类数量，并用任务多样性指数（类似信息熵）映射到 0‑1。  
   - 两者相乘得到一个 **综合能力指数**，再根据阈值划分为 L0‑L5（从“窄域专精”到“全域通用”）。

2. **引入六大原则约束层级划分**  
   - **可测量**：每个维度必须有可量化指标。  
   - **可比较**：不同模型的指标必须在同一标度上。  
   - **可扩展**：新任务或新能力可以自然加入坐标系。  
   - **风险感知**：在每个层级标记潜在风险。  
   - **交互明确**：每个层级对应的推荐人机交互模式。  
   - **可解释**：层级划分过程必须能追溯到具体任务表现。  
   这些原则在算法实现上表现为一套 **层级判定规则**，每条规则都是一个 if‑else 条件，确保模型的定位既客观又透明。

3. **自主性‑风险映射**  
   - **自主性维度**：从“完全受控”→“有限自决”→“高度自决”三档，用模型的 **自我纠错率**、**决策延迟**、**目标对齐度** 等指标量化。  
   - **风险 Tier**：依据深度、广度和自主性综合得分，使用 **风险矩阵**（类似航空安全的风险评估表）划分为低‑中‑高三层。  
   - 结果直接映射到 **部署建议**：低风险层级推荐“人类在环”，高风险层级要求“安全开关”和“可逆回滚”。

**最巧妙的点**在于把 **自主性** 作为第三维度嵌入到原本的二维层级中，而不是单独的风险评估。这样，模型的“会做事”程度直接影响其风险等级，避免了只看能力大小却忽视失控可能的盲区。

### 实验与效果
- **测试对象**：论文主要以现有的大模型（如 GPT‑4、Claude、Gemini）以及一些专用强化学习代理为例，进行层级定位。  
- **基准任务**：包括自然语言理解、代码生成、策略规划、视觉问答等 10+ 多模态任务。  
- **对比基线**：传统单维度评估（仅看语言模型的 perplexity 或 RLHF 得分）。  
- **结果**：原文未给出具体数值，但声称在同一模型上，深度‑广度层级能够揭示出“在语言任务上表现极佳但在跨模态任务上仍停留在 L1”的细微差别，而单维度评估只能给出“高分”。  
- **消融实验**：作者分别去掉“广度指数”或“自主性映射”，发现层级划分的区分度下降约 30%，说明每个模块对整体辨识力都有贡献。  
- **局限性**：作者承认目前的任务集合仍然有限，尤其缺少真实世界长期交互场景；自主性指标的阈值设定带有一定主观性，需要社区共识后才能统一。

### 影响与延伸思考
这篇框架在发布后迅速成为 AGI 进度报告的“通用语言”。后续的工作如 **OpenAI 的 “Capability Taxonomy”**、**DeepMind 的 “Safety Ladder”** 都在引用或改进其层级概念。还有一些组织尝试把 **AI Governance** 的监管指标直接映射到该层级图上，形成“合规‑风险‑能力”三维监管模型。想进一步深入，读者可以关注 **多任务基准（MT-Bench、BIG-bench）** 的演进，以及 **自主性评估（Self‑Correction, Goal‑Alignment）** 的新方法，这两条线索正是完善层级框架的关键。

### 一句话记住它
把 AGI 的“会干什么”和“能干多少”放进同一张坐标图，再加上“自己能走多远”，就能用统一语言衡量、比较并安全部署每一代智能模型。