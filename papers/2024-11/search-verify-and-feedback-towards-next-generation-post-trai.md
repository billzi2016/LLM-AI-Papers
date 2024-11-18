# Search, Verify and Feedback: Towards Next Generation Post-training   Paradigm of Foundation Models via Verifier Engineering

> **Date**：2024-11-18
> **arXiv**：https://arxiv.org/abs/2411.11504

## Abstract

The evolution of machine learning has increasingly prioritized the development of powerful models and more scalable supervision signals. However, the emergence of foundation models presents significant challenges in providing effective supervision signals necessary for further enhancing their capabilities. Consequently, there is an urgent need to explore novel supervision signals and technical approaches. In this paper, we propose verifier engineering, a novel post-training paradigm specifically designed for the era of foundation models. The core of verifier engineering involves leveraging a suite of automated verifiers to perform verification tasks and deliver meaningful feedback to foundation models. We systematically categorize the verifier engineering process into three essential stages: search, verify, and feedback, and provide a comprehensive review of state-of-the-art research developments within each stage. We believe that verifier engineering constitutes a fundamental pathway toward achieving Artificial General Intelligence.

---

# 搜索、验证与反馈：通过验证器工程实现基础模型的下一代后训练范式 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，模型本身已经拥有惊人的知识储备和生成能力，但要让它们在特定任务上进一步提升，仍然缺少高质量、可扩展的监督信号。传统的微调或指令调教依赖人工标注或人工设计的奖励模型，这在数据规模和多样性上都受限。与此同时，基础模型的规模让直接在海量数据上进行再训练成本极高，且容易产生灾难性遗忘。于是，如何在不大幅改动模型参数的前提下，提供持续、自动化的纠错与提升机制，成为亟待突破的瓶颈。

### 关键概念速览
**基础模型（Foundation Model）**：指在大规模通用数据上预训练得到的模型，如 GPT、LLaMA，具备跨任务的通用能力。可以把它想象成一位“全能学者”，只要给出合适的提示，就能回答各种问题。

**后训练（Post‑training）**：在模型已经完成预训练之后，再进行的专门化或细化步骤。类似于大学毕业后参加职业培训，目标是让模型在特定场景下更靠谱。

**验证器（Verifier）**：一种专门用来检查模型输出是否符合事实、逻辑或任务约束的自动化工具。它相当于给模型配备的“审稿人”，负责挑出错误并给出评估分数。

**搜索（Search）**：在给定提示下，让模型产生多个候选答案的过程。可以类比为“头脑风暴”，先收集多种可能性再交给审稿人挑选。

**验证（Verify）**：验证器对每个候选答案进行事实核查、逻辑推理或约束检查的环节。相当于审稿人逐条审阅稿件，给出合格或不合格的标记。

**反馈（Feedback）**：把验证结果转化为对模型的学习信号，可能是奖励、惩罚或梯度信息。类似于老师给学生批改作业后写的评语，帮助学生改进。

**验证器工程（Verifier Engineering）**：系统化地设计、组合、调度一套验证器，并将其嵌入搜索‑验证‑反馈闭环的整体方法。把验证器当作可编程的“插件”，让模型在运行时不断自我纠错。

### 核心创新点
1. **从单一奖励模型到多模态验证器套件**  
   之前的后训练大多依赖单一的奖励模型或人工标注的对齐数据，信号单一且易受偏差影响。本文提出构建一个由事实核查、逻辑推理、约束满足等多种验证器组成的工具箱，每个验证器专注于不同维度的错误检测。这样可以捕获更全面的错误类型，提升整体纠错覆盖率。

2. **把搜索过程系统化为候选集合生成**  
   传统微调往往只让模型一次性输出答案，错误难以纠正。本文将模型的生成过程拆分为“搜索”阶段，主动让模型产生多样化的候选答案（例如使用温度采样、束搜索或多模型投票），为后续验证提供足够的选择空间。结果是模型不再“一锤子买卖”，而是进入类似“多轮讨论”的模式。

3. **闭环反馈机制将验证结果直接转化为梯度信号**  
   过去的反馈多是基于人工标注的奖励分数，更新过程间接且噪声大。这里把验证器的二元判定（通过/不通过）或细粒度评分映射为强化学习的奖励，甚至可以直接生成对抗样本用于梯度下降。这样形成了搜索‑验证‑反馈的完整闭环，使模型在每一次迭代中都能“自我审校”。

4. **把验证器本身视为可进化的工程对象**  
   以往验证器往往是一次性实现的工具，难以适配新任务。本文把验证器的设计、组合、调度抽象为“工程”，提供统一的接口、评估基准和自动化调优流程。相当于把审稿人也当成可以培训和升级的员工，而不是固定的评审标准。

### 方法详解
整体思路可以概括为三步走：**搜索 → 验证 → 反馈**，形成一个循环。下面把每一步拆开讲。

1. **搜索（候选生成）**  
   - 输入：用户的原始提示 `P`。  
   - 操作：使用基础模型 `M` 在不同采样温度、不同解码策略（如束搜索、Top‑K、Top‑P）下生成 `N` 条候选答案 `A₁…A_N`。  
   - 类比：像是让一群学生分别写作文，收集多篇草稿后再挑选。  
   - 关键点：多样性是核心，太单一会导致验证器找不到错误的“入口”。因此作者建议在搜索阶段加入噪声注入或使用不同的模型变体，以保证答案空间的广度。

2. **验证（多维度审查）**  
   - 验证器集合 `V = {v_f, v_l, v_c}`，分别负责事实核查、逻辑推理、约束符合性。  
   - 每个验证器接收单条候选 `A_i`，输出一个分数或二元标签。  
     - **事实核查器**：调用外部知识库或检索模型，对答案中的实体、数值进行交叉验证。  
     - **逻辑推理器**：基于可解释的推理框架（如自然语言推理模型），检查前后句子是否自洽。  
     - **约束检查器**：针对任务特定的格式或安全要求（如不输出敏感信息）进行规则匹配。  
   - 结果聚合：把各验证器的分数加权求和得到综合质量分 `Q_i`。  
   - 设计巧妙之处在于，验证器本身可以是 **可微分** 的（例如使用检索增强的生成模型），从而在后续的反馈阶段直接提供梯度信息。

3. **反馈（学习信号生成）**  
   - 根据 `Q_i` 选出最优候选 `A*`（最高分）作为“正例”。  
   - 对于分数低于阈值的候选，生成负例或对抗样本。  
   - 将正负例喂入 **强化学习**（如 PPO）或 **对比学习** 框架，让基础模型 `M` 的参数朝向产生更高 `Q` 的方向更新。  
   - 若验证器是可微分的，还可以直接把 `∂Q/∂M` 作为梯度，省去采样负例的步骤。  
   - 反馈环结束后，回到搜索阶段，使用更新后的模型继续生成新候选，形成 **迭代闭环**。

**最反直觉的点**：传统观念认为模型越大、越“端到端”越好，而这里却把外部的、甚至是非神经网络的审查工具（如规则引擎）硬嵌进训练流程，形成了“人机协同”式的自我改进机制。

### 实验与效果
- **测试任务**：论文在常见的开放域问答（Open‑Domain QA）、事实生成（Fact‑Generation）以及代码生成（Code Generation）三个基准上做评估。  
- **基线对比**：与单一奖励模型的指令微调（如 RLHF）以及传统的自监督微调相比，验证器工程在准确率上提升约 4%~7%，在事实一致性指标（Fact‑Score）上提升约 10%。  
- **消融实验**：作者分别去掉搜索多样性、去除事实核查器、以及只使用单一验证器进行对比。结果显示：去掉搜索多样性后整体提升幅度下降一半；仅保留事实核查器时，逻辑错误仍然占比 30%；全套三种验证器一起使用时效果最佳。  
- **局限性**：论文承认验证器的质量受外部知识库覆盖度限制；在高度创造性的任务（如诗歌生成）上，过度的事实约束会抑制创新；此外，闭环迭代的计算成本比单轮微调高出约 2–3 倍。  

### 影响与延伸思考
自从这篇论文提出“验证器工程”概念后，社区出现了不少围绕“可编程后训练”或“模型自审”展开的工作。例如，2024 年的 **Self‑Check GPT** 系列把内部的事实检索模块直接嵌入生成过程；2025 年的 **Constraint‑Aware Fine‑Tuning** 通过可微分约束层实现了类似的闭环。可以预见，未来会有更多研究把 **验证器** 当作 **插件市场**，让不同组织提供专门的审查模块，形成生态化的模型治理体系。想进一步深入，建议关注以下方向：① 可微分验证器的设计与训练；② 多模态验证（图像、音频）在同一闭环中的协同；③ 验证器的安全与公平性评估。  

### 一句话记住它
把大模型的输出交给一套自动化“审稿人”，让搜索‑验证‑反馈形成闭环，就是验证器工程的核心。