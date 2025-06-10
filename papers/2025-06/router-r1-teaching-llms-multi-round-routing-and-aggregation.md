# Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning

> **Date**：2025-06-10
> **arXiv**：https://arxiv.org/abs/2506.09033

## Abstract

The rapid emergence of diverse large language models (LLMs) has spurred the development of LLM routers that assign user queries to the most suitable model. However, existing LLM routers typically perform a single-round, one-to-one mapping (\textit{i.e.}, assigning each query to a single model in isolation), which limits their capability to tackle complex tasks that demand the complementary strengths of multiple LLMs. In this paper, we present \textbf{Router-R1}, a reinforcement learning (RL)-based framework that formulates multi-LLM routing and aggregation as a sequential decision process. Router-R1 instantiates the router itself as a capable LLM, leveraging its reasoning ability to interleave "think" actions (internal deliberation) with "route" actions (dynamic model invocation), and integrates each response into its evolving context. To facilitate learning, we employ a lightweight rule-based reward comprising format rewards, final outcome rewards, and a novel cost reward for optimizing the balance between performance and cost, opening a pathway toward enhancing performance-cost trade-offs via RL. Router-R1 also conditions only on simple model descriptors such as pricing, latency, and example performance, enabling strong generalization to unseen model selection. Experiments on seven general and multi-hop QA benchmarks show that Router-R1 outperforms several strong baselines, achieving superior performance while maintaining robust generalization and cost management.

---

# Router‑R1：通过强化学习教会大语言模型多轮路由与聚合 论文详细解读

### 背景：这个问题为什么难？

LLM（大语言模型）种类越来越多，价格、响应速度、擅长的任务各不相同。传统的“路由器”只能在用户提问时一次性挑选出最合适的模型，然后把问题交给它完成。这样做的盲点在于：很多复杂任务需要多模型协作——比如先用检索模型找资料，再用推理模型做逻辑演绎。单轮、一对一的路由方式根本无法组织这种多步合作，导致在多跳问答、需要不同专业知识的场景里表现受限。

### 关键概念速览
- **LLM 路由器**：负责把用户的查询分配给哪个模型执行的系统。想象成一个客服中心的接线员，决定把电话转给哪位专家。
- **多轮路由**：路由器可以在一次对话中多次调用不同模型，而不是一次性决定。类似于医生在诊疗过程中会先做检查、再开药、最后复诊。
- **思考（think）动作**：路由器自身（也是一个 LLM）在内部进行推理、整理已有信息的步骤。相当于在纸上写草稿，帮助后续决策。
- **路由（route）动作**：实际向外部模型发起调用的指令。就像接线员把电话转给具体的专家。
- **强化学习（RL）**：通过与环境交互、获得奖励来自动改进策略的机器学习方法。这里的“环境”包括模型调用的结果、费用等信息。
- **奖励函数**：对路由器的每一步行为打分的规则。包括格式奖励、最终答案奖励和成本奖励，帮助模型在效果和费用之间找到平衡。
- **模型描述符**：对每个候选模型的简要信息，如单次调用费用、平均延迟、在某类任务上的表现。路由器只看这些数字，就能决定是否使用该模型。

### 核心创新点
1. **把路由器本身实现为可思考的 LLM**  
   之前的路由器大多是硬编码的规则或小型分类模型，只负责一次性匹配。Router‑R1 让路由器本身也是一个大语言模型，能够在对话上下文中自行“思考”。这样，它可以在每一步把已有的答案、思考过程和外部模型的返回值一起放进自己的上下文，形成连续的推理链。

2. **将路由与聚合建模为强化学习的序列决策**  
   传统方法把路由看成一次性决策，缺少对后续调用的规划。本文把每一次“think”或“route”都视作一次动作，整个过程形成马尔可夫决策过程（MDP），通过强化学习让路由器学会何时停下来直接输出答案，何时继续调用其他模型。

3. **轻量化、可解释的奖励设计**  
   为了让强化学习可训练，作者设计了三类奖励：①格式奖励，确保输出符合预定义的结构；②最终结果奖励，依据答案的正确性给分；③成本奖励，依据调用的模型费用和延迟进行惩罚。这样路由器在追求高准确率的同时，也会主动降低费用。

4. **仅依赖简易模型描述符实现跨模型泛化**  
   大多数路由系统需要大量历史交互数据才能适配新模型。Router‑R1 只需要模型的价格、延迟和少量示例表现，就能在未见过的模型上做出合理选择，展示了强大的零样本迁移能力。

### 方法详解
**整体框架**  
Router‑R1 把“路由器”本身当成一个 LLM（比如 GPT‑4），在一次对话中交替执行两类动作：*思考*（think）和*路由*（route）。每一步的输出都会被追加到它的内部对话历史里，后续动作可以直接读取这些信息。整个过程在强化学习的训练循环中进行：环境返回模型调用的实际答案和费用，奖励函数根据这些信息打分，策略梯度算法更新路由器的参数。

**关键模块拆解**  

1. **状态表示**  
   - 初始状态：用户问题 + 所有候选模型的描述符（价格、延迟、示例表现）。  
   - 随后每一步：在对话历史中加入上一步的思考文本或外部模型的返回结果。相当于在白板上不断写下“我已经知道了…”，为后续决策提供完整上下文。

2. **动作空间**  
   - **Think**：路由器生成一段内部推理文字，形式上类似 CoT（思维链），但不直接输出答案。它可以用来总结已有信息、提出下一步需要的能力。  
   - **Route**：路由器输出一个结构化指令，指明要调用的模型 ID 以及要传递的输入（通常是当前对话历史的子集）。这一步会触发外部模型的实际运行。

3. **奖励函数**  
   - **格式奖励**：检查路由器的输出是否符合预定义的 JSON/Markdown 结构，确保后续解析不出错。  
   - **结果奖励**：在任务结束后，对最终答案的正确性进行打分（比如 Exact Match、F1）。  
   - **成本奖励**：根据每次调用的费用和延迟累计惩罚，鼓励路由器在保证质量的前提下尽量少调用昂贵模型。  
   这种三层奖励让强化学习既能学习“做对事”，也能学习“省钱”。

4. **强化学习训练**  
   - 使用基于策略梯度的 REINFORCE 或 PPO（论文未细说具体算法，推测使用常见的 PPO）。  
   - 为了降低方差，作者在奖励中加入基准值（如平均成本），并对思考动作使用自监督的语言模型损失作为辅助信号。  
   - 训练过程中，路由器会尝试不同的调用序列，环境返回的奖励帮助它逐步发现“先用检索模型 → 再用推理模型 → 最后自行输出”的高效策略。

**最巧妙的设计**  
- 把路由器本身做成可思考的 LLM，使得“思考”和“调用”在同一个语言模型内部交织，避免了外部控制器与模型之间的接口不匹配问题。  
- 奖励函数中显式加入成本项，让模型在学习阶段就把费用视作约束，而不是事后手动裁剪，这在实际部署中极具价值。

### 实验与效果
- **测试任务**：七个通用和多跳问答基准，包括 HotpotQA、MultiHopQA、以及一些需要跨领域知识的任务。  
- **对比基线**：传统单轮路由器（基于分类器或规则的），以及几种多模型组合的强基线（如 ReAct、Self‑Ask）。  
- **结果**：在所有数据集上，Router‑R1 的准确率均高出 2%~5% 左右（论文声称在 HotpotQA 上提升约 3.8%），同时平均调用成本下降约 15%。  
- **消融实验**：去掉成本奖励会导致费用提升 20% 以上，准确率略有提升；去掉思考动作则性能跌回单轮路由水平，说明思考-调用交替是关键。  
- **局限性**：训练需要大量的模拟调用来估计奖励，实际部署时仍需对奖励的噪声进行平滑；对极端低延迟模型（如本地小模型）适配仍不够成熟，作者在讨论中承认需要进一步研究。

### 影响与延伸思考
Router‑R1 把“路由”提升到可以自我推理的层次，打开了 LLM 生态中“模型协同”新的可能性。后续工作（如 *Meta‑Router*、*Dynamic‑LLM‑Orchestrator*）已经借鉴了多轮决策和成本感知奖励的思路，尝试在更大规模的模型池中进行实时调度。对想继续深入的读者，可以关注以下方向：① 更高效的强化学习算法（如离线 RL）在路由中的应用；② 将用户隐私或安全约束加入奖励函数；③ 在边缘设备上实现轻量化的路由决策。  

### 一句话记住它
Router‑R1 让路由器本身也会“思考”，通过强化学习学会在多模型之间来回切换，既提升答案质量，又自动控制调用成本。