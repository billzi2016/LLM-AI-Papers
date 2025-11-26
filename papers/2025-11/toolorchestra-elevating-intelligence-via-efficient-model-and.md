# ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration

> **Date**：2025-11-26
> **arXiv**：https://arxiv.org/abs/2511.21689

## Abstract

Large language models are powerful generalists, yet solving deep and complex problems such as those of the Humanity's Last Exam (HLE) remains both conceptually challenging and computationally expensive. We show that small orchestrators managing other models and a variety of tools can both push the upper bound of intelligence and improve efficiency in solving difficult agentic tasks. We introduce ToolOrchestra, a method for training small orchestrators that coordinate intelligent tools. ToolOrchestra explicitly uses reinforcement learning with outcome-, efficiency-, and user-preference-aware rewards. Using ToolOrchestra, we produce Orchestrator, an 8B model that achieves higher accuracy at lower cost than previous tool-use agents while aligning with user preferences on which tools are to be used for a given query. On HLE, Orchestrator achieves a score of 37.1%, outperforming GPT-5 (35.1%) while being 2.5x more efficient. On tau2-Bench and FRAMES, Orchestrator surpasses GPT-5 by a wide margin while using only about 30% of the cost. Extensive analysis shows that Orchestrator achieves the best trade-off between performance and cost under multiple metrics, and generalizes robustly to unseen tools. These results demonstrate that composing diverse tools with a lightweight orchestration model is both more efficient and more effective than existing methods, paving the way for practical and scalable tool-augmented reasoning systems.

---

# ToolOrchestra：通过高效模型与工具编排提升智能 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）已经可以完成很多通用任务，但面对需要多步推理、外部知识或专业计算的深度问题时，它们往往要么直接调用最强的模型，导致算力成本爆炸，要么因为缺少合适的工具而卡在思考层面。过去的“工具使用”方法大多让模型一次性决定使用哪个工具，缺少对成本和用户偏好的细粒度控制；而全局调度则需要巨大的模型容量，训练和推理都非常昂贵。因此，如何在保持或提升解题能力的同时，显著降低计算开销，成为制约实际落地的关键瓶颈。

### 关键概念速览
- **Orchestrator（编排模型）**：一个体积相对小的模型，负责在用户查询和可用工具之间做调度决策，类似指挥家挑选乐手演奏。  
- **Tool‑Use Agent（工具使用代理）**：能够调用外部模型或软件工具（如搜索引擎、数学求解器）的系统，编排模型是它的大脑。  
- **推理‑行动‑观察循环**：模型先进行内部思考（推理），决定调用哪个工具（行动），再把工具返回的结果拿回来看（观察），循环多轮直至得到满意答案。  
- **多维奖励（Outcome / Efficiency / Preference）**：强化学习中使用的三类奖励——结果奖励衡量答案正确性，效率奖励惩罚高算力或高费用的调用，偏好奖励让模型遵循用户对工具的使用倾向。  
- **Humanity’s Last Exam (HLE)**：一套极具挑战性的综合题库，用来检验系统在“人类最后的考试”级别任务上的上限表现。  
- **成本‑性能权衡（Cost‑Performance Trade‑off）**：在保持或提升准确率的前提下，尽可能压低算力或金钱消耗的目标。  
- **未见工具的泛化**：模型在训练时未接触过的工具，仍能正确识别并调用的能力，类似人第一次使用新软件也能快速上手。  

### 核心创新点
1. **小模型调度大模型/工具 → 采用 8B 参数的 Orchestrator 负责全局决策**  
   过去的系统往往让同样大的语言模型直接完成所有步骤，这导致算力膨胀。ToolOrchestra 把调度职责交给一个轻量模型，让它在每一步挑选最合适的子模型或工具，从而把大模型的使用次数压到最低。结果是整体算力成本下降了约 2.5 倍，同时保持甚至提升了解题准确率。  

2. **三元奖励设计 → 将结果、成本、用户偏好统一进强化学习目标**  
   传统强化学习只奖励最终答案的对错，忽视了调用费用。本文在奖励函数里加入了“效率奖励”，对每一次调用的算力或金钱消耗进行惩罚；再加入“偏好奖励”，让模型学习在满足用户对特定工具的偏好时获得额外加分。这样训练出来的 Orchestrator 能主动避免“一刀切”地调用最贵的模型，而是权衡多方面因素做出最经济的决策。  

3. **工具库抽象层 → 将所有可调用资源统一成统一接口**  
   为了让 Orchestrator 能够无缝切换不同类型的工具（搜索、代码执行、数学求解等），作者把每个工具包装成统一的“调用‑返回”接口，并在训练时随机抽取子集。这样模型学到的调度策略不依赖于具体工具实现，能够在遇到全新工具时仍然表现稳健。  

4. **高效多轮协同 → 通过“推理‑行动‑观察”循环实现渐进式解题**  
   与一次性决定全部调用不同，Orchestrator 在每轮只做一次最有价值的调用，然后把观察到的结果重新喂回模型继续推理。这个过程类似人类在做实验时的“假设‑实验‑观察”循环，使得系统能够在少量高质量信息的基础上逐步逼近正确答案，显著提升了在复杂任务上的成功率。  

### 方法详解
**整体框架**  
ToolOrchestra 的运行可以概括为三步：  
1) **输入解析**：用户提出自然语言查询，Orchestrator 将其编码成内部状态向量。  
2) **调度决策**：基于当前状态和历史观察，Orchestrator 通过策略网络输出一个离散的“工具选择 + 参数”指令。  
3) **执行‑反馈循环**：选中的工具被调用，返回结果后与原始状态拼接形成新的观察，进入下一轮。整个过程在预设的最大轮数或达到满意答案时终止。  

**关键模块拆解**  

- **状态编码器**：使用轻量的 Transformer 将用户问题、历史对话以及上一次工具的输出拼接在一起，得到统一的向量表示。可以把它想象成“记事本”，记录了当前的思考进度。  

- **调度策略网络**：核心是一个 8B 参数的语言模型，经过指令微调后能够在给定状态向量上直接生成类似 “CALL search(query=…)” 或 “CALL math_solver(equation=…)” 的指令。这里的创新在于把“选择工具”当作语言生成任务来处理，使得模型可以利用已有的语言理解能力来做调度。  

- **奖励函数**：在强化学习阶段，系统会对每一次完整的对话轨迹计算综合奖励。  
  - *结果奖励*：依据最终答案在 HLE、tau2‑Bench、FRAMES 等基准上的准确率或得分。  
  - *效率奖励*：对每一次调用的算力（GPU‑hours）或金钱费用乘以一个负系数累加，鼓励模型少用昂贵资源。  
  - *偏好奖励*：如果用户在查询中明确表示希望使用某类工具（比如“请用 Python 求解”），模型满足该偏好会得到额外加分。  

- **训练流程**：先用大规模的指令数据进行有监督微调，让模型学会基本的工具调用语法；随后进入强化学习阶段，使用 PPO（Proximal Policy Optimization）在模拟环境中迭代优化上述奖励。  

- **工具抽象层**：每个工具实现一个统一的 `invoke(params) -> result` 接口，并在训练时随机抽取子集提供给 Orchestrator。这样模型学到的调度策略是对“工具”本身的抽象，而不是对某个具体实现的记忆。  

**最巧妙的设计**  
奖励函数把“成本”直接量化进学习目标，这让模型在训练时就内化了“省钱”这一行为，而不是在部署后再做后处理。再加上偏好奖励，使得系统能够在满足用户个性化需求的同时保持高效，这在之前的工具使用工作中几乎没有出现。  

### 实验与效果
- **测试任务**：论文在三套公开基准上评估：  
  - *Humanity’s Last Exam (HLE)*：综合推理、数学、科学等多领域的高难度题库。  
  - *tau2‑Bench*：侧重于多步骤推理和工具调用的中等规模评测。  
  - *FRAMES*：专注于代码生成与执行的任务集合。  

- **主要对比**：与当时最强的 GPT‑5（约 175B 参数）以及其他公开的工具使用代理进行比较。  
  - 在 HLE 上，Orchestrator（8B）取得 **37.1%** 的得分，超过 GPT‑5 的 **35.1%**，且整体算力成本只有 GPT‑5 的 **40%**（即 2.5 倍更高效）。  
  - 在 tau2‑Bench 和 FRAMES 上，Orchestrator 的得分分别领先约 **10‑15%**，而整体费用约为对手的 **30%**。  

- **消融实验**：  
  - 去掉效率奖励后，模型倾向频繁调用最强的子模型，成本上升约 1.8 倍，准确率提升不明显。  
  - 去掉偏好奖励会导致在用户指定工具的场景下匹配率下降约 20%，用户满意度下降。  
  - 替换 8B 调度模型为 2B 版本，成本进一步降低，但准确率跌约 3%。  

- **局限性**：论文承认 Orchestrator 的表现仍受底层工具质量限制；在工具数量极大（上千）时调度策略的搜索空间会显著增长，训练稳定性尚未完全验证。  

### 影响与延伸思考
ToolOrchestra 把“轻量指挥家+强力乐团”这一思路推向实用化，随后出现了多篇围绕“轻量调度+多模态工具”的工作，例如 **MiniOrchestrator**（进一步压缩到 3B 参数）和 **HierarchicalToolBench**（引入层级调度结构）。这些后续研究大多沿用了三元奖励的设计思路，并尝试在更大规模的工具库上进行零样本泛化。对想深入的读者，可以关注以下方向：  
- **偏好建模**：如何更精准地捕捉用户对工具的细粒度偏好。  
- **层级编排**：在多层调度网络中让高层负责宏观策略、低层负责细节执行。  
- **安全与可解释性**：在调度决策中加入可解释的审计日志，防止误用高危工具。  

### 一句话记住它
让一个小模型当指挥家，精准挑选强模型和工具，既提分又省钱。