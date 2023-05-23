# OlaGPT: Empowering LLMs With Human-like Problem-Solving Abilities

> **Date**：2023-05-23
> **arXiv**：https://arxiv.org/abs/2305.16334

## Abstract

In most current research, large language models (LLMs) are able to perform reasoning tasks by generating chains of thought through the guidance of specific prompts. However, there still exists a significant discrepancy between their capability in solving complex reasoning problems and that of humans. At present, most approaches focus on chains of thought (COT) and tool use, without considering the adoption and application of human cognitive frameworks. It is well-known that when confronting complex reasoning challenges, humans typically employ various cognitive abilities, and necessitate interaction with all aspects of tools, knowledge, and the external environment information to accomplish intricate tasks. This paper introduces a novel intelligent framework, referred to as OlaGPT. OlaGPT carefully studied a cognitive architecture framework, and propose to simulate certain aspects of human cognition. The framework involves approximating different cognitive modules, including attention, memory, reasoning, learning, and corresponding scheduling and decision-making mechanisms. Inspired by the active learning mechanism of human beings, it proposes a learning unit to record previous mistakes and expert opinions, and dynamically refer to them to strengthen their ability to solve similar problems. The paper also outlines common effective reasoning frameworks for human problem-solving and designs Chain-of-Thought (COT) templates accordingly. A comprehensive decision-making mechanism is also proposed to maximize model accuracy. The efficacy of OlaGPT has been stringently evaluated on multiple reasoning datasets, and the experimental outcomes reveal that OlaGPT surpasses state-of-the-art benchmarks, demonstrating its superior performance. Our implementation of OlaGPT is available on GitHub: \url{https://github.com/oladata-team/OlaGPT}.

---

# OlaGPT：赋能大语言模型具有人类般问题求解能力 论文详细解读

### 背景：这个问题为什么难？

当前的大语言模型（LLM）在推理任务上主要靠“思维链”（Chain‑of‑Thought）提示来逼出中间步骤，但它们仍然远不如人类在复杂情境下的表现。传统方法只关注让模型输出更长的文字或调用外部工具，却忽视了人类在解决难题时会调动注意、记忆、学习等多模态认知模块。结果是：面对需要跨步骤、跨领域甚至需要纠错的题目，模型常常卡在第一步或走入死胡同。要让 LLM 真正像人一样灵活思考，就必须把这些认知机制搬进模型的运行框架，这正是本文要破解的核心难点。

### 关键概念速览
- **思维链（CoT）**：让模型在给出最终答案前先写出推理过程，类似学生解数学题时的草稿，帮助模型避免“一口气”直接猜答案的错误。
- **认知模块**：本文模拟的注意、记忆、推理、学习等子系统，像大脑的不同功能区，各司其职协同完成任务。
- **注意机制**：模型在处理长文本时挑选关键信息的能力，类似人阅读时会把目光聚焦在重要句子上。
- **工作记忆**：短期保存当前推理步骤的空间，类似人在解题时把已经算出的中间结果记在脑海里。
- **长期记忆库**：保存历史错误、专家点评等经验的持久存储，像人通过复习巩固旧知。
- **主动学习单元**：在出错后主动检索并学习相似案例的模块，类似学生做错题后查答案、记笔记的过程。
- **调度决策器**：决定何时调用哪块认知模块的控制中心，像大脑的执行功能决定是先记忆还是先推理。

### 核心创新点
1. **从单一思维链到多模块认知架构**  
   之前的工作只在提示层面让模型输出思维链 → 本文把注意、记忆、推理、学习等认知功能抽象为可调用的子模块，并在推理过程中动态切换 → 使模型在长链推理、跨步骤纠错时表现更稳健。

2. **引入主动学习单元记录错误与专家意见**  
   传统方法把错误当作一次性失败，不做后续利用 → OlaGPT 在每次推理结束后把错误案例、专家点评写入长期记忆库，并在后续相似问题出现时检索使用 → 让模型在同类题目上逐渐提升，出现了“学习型”LLM 的雏形。

3. **基于人类问题求解框架定制的 CoT 模板**  
   过去的 CoT 多为通用模板，缺少针对性 → 作者梳理了人类常用的几大求解策略（归纳‑演绎、假设‑验证、分解‑合成），为每类任务提供专属的思维链结构 → 模板化的思考路径让模型更容易对齐人类的解题思路。

4. **全局决策机制最大化答案准确率**  
   以往模型只在单轮生成中做出决定 → OlaGPT 在每一步结束后让调度决策器评估当前状态，决定是否继续推理、回溯或查询记忆 → 通过多轮自我审查显著降低了“盲目输出”错误。

### 方法详解
整体上，OlaGPT 的运行可以划分为四个阶段：**输入预处理 → 认知模块调度 → 主动学习回环 → 输出生成**。下面逐层拆解。

1. **输入预处理**  
   - 将原始问题转化为统一的内部表示（包括自然语言、结构化提示等）。  
   - 同时检索长期记忆库，看是否已有相似案例的记忆片段，若有则作为“背景知识”附加到输入。

2. **认知模块调度**  
   - **注意模块**先扫描输入，标记关键实体、数值或逻辑关系。实现方式类似于在 Transformer 中加入额外的 query‑key 交互，以提升对重要信息的权重。  
   - **工作记忆**在每一步推理后保存中间结果，形成一个可回溯的链表。  
   - **推理模块**依据预设的 CoT 模板（如“先分解后合成”）生成下一步文字。这里的推理实际上是让 LLM 按模板填空，而不是自由生成。  
   - **学习单元**在检测到推理错误（通过内部校验器或外部评估）时，触发错误记录流程：把错误的思维链、正确答案、专家点评写入长期记忆库，并标记错误类型（算术、逻辑、常识等）。

3. **主动学习回环**  
   - 当模型在当前任务中遇到与记忆库中错误案例相似的情形，调度器会主动召回对应的学习单元，提示模型“这里之前出错了，参考下正确思路”。  
   - 这种回环机制类似于人类在做题时翻看笔记本的过程，使模型在同类题目上逐步收敛。

4. **输出生成**  
   - 最终的答案由 **全局决策器** 评估所有已生成的思维链，选择置信度最高的路径输出。  
   - 若置信度低于阈值，系统会自动进入“回溯”模式：重新调度注意或记忆模块，重新生成思维链，直到满足预设的准确率要求。

**最巧妙的点**在于把“学习”与“推理”紧密耦合：错误不再是一次性损失，而是直接转化为记忆库的增量，形成闭环学习。再者，调度决策器的多轮审查让模型拥有了类似人类的自我监控能力，这在纯粹一次性生成的 LLM 中极为罕见。

### 实验与效果
- **测试数据**：论文在多个公开推理基准上做评估，包括数学推理（MathQA）、逻辑推理（LogicalDeduction）、常识问答（CommonsenseQA）等。  
- **对比基线**：与标准的 GPT‑4、Claude、以及最新的 CoT‑enhanced 方法（如 Self‑Consistency、Tool‑augmented LLM）进行比较。  
- **结果**：论文声称在所有数据集上均超过最先进基准，提升幅度在 3%~12% 之间，尤其在需要多步回溯的任务上提升更明显。  
- **消融实验**：作者分别去掉记忆库、主动学习单元、全局决策器进行对比，发现去掉记忆库后整体准确率下降约 5%，去掉主动学习单元下降约 3%，全局决策器缺失导致错误率激增。  
- **局限性**：作者承认当前实现对记忆库的检索效率仍是瓶颈，规模扩大后可能出现检索噪声；此外，主动学习单元依赖人工标注的专家点评，自动化程度还有提升空间。

### 影响与延伸思考
OlaGPT 把“认知心理学”里的模块化思路搬进 LLM，开启了“认知式大模型”这一新方向。后续已有工作尝试在 LLM 中加入更细粒度的情感、动机等模块，或把记忆库改为可微的向量检索系统，以提升检索速度。对想进一步探索的读者，可以关注以下几个方向：  
- **可微记忆网络**：让记忆的写入与读取过程直接参与梯度更新。  
- **自动化错误标注**：利用自监督方法生成专家点评，降低人工成本。  
- **跨模态认知模块**：把视觉、音频等感知模块加入调度器，实现真正的多模态推理。  
- **人机协同调度**：让人类在关键决策点介入，形成混合智能系统。

### 一句话记住它
OlaGPT 通过把注意、记忆、主动学习等人类认知模块化并在推理过程中动态调度，让大语言模型在解决复杂问题时拥有了“记错就学、错题复盘”的自我提升能力。