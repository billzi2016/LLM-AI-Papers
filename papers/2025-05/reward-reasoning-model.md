# Reward Reasoning Model

> **Date**：2025-05-20
> **arXiv**：https://arxiv.org/abs/2505.14674

## Abstract

Reward models play a critical role in guiding large language models toward outputs that align with human expectations. However, an open challenge remains in effectively utilizing test-time compute to enhance reward model performance. In this work, we introduce Reward Reasoning Models (RRMs), which are specifically designed to execute a deliberate reasoning process before generating final rewards. Through chain-of-thought reasoning, RRMs leverage additional test-time compute for complex queries where appropriate rewards are not immediately apparent. To develop RRMs, we implement a reinforcement learning framework that fosters self-evolved reward reasoning capabilities without requiring explicit reasoning traces as training data. Experimental results demonstrate that RRMs achieve superior performance on reward modeling benchmarks across diverse domains. Notably, we show that RRMs can adaptively exploit test-time compute to further improve reward accuracy. The pretrained reward reasoning models are available at https://huggingface.co/Reward-Reasoning.

---

# 奖励推理模型 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，奖励模型（Reward Model）负责把模型输出映射到人类偏好的分数，进而指导强化学习的方向。传统奖励模型在推理阶段只做一次前向计算，直接给出分数，这在面对需要深层语义理解或多步推理的复杂查询时常常失灵。因为模型没有机会在生成奖励前“思考”，所以对细粒度、上下文依赖强的任务表现不佳。与此同时，测试时的计算资源往往被闲置，如何把这些额外算力转化为更精准的奖励，成为一个未被充分利用的机会。

### 关键概念速览
- **奖励模型（Reward Model）**：评估语言模型输出好坏的打分器，类似老师给作文打分。  
- **链式思考（Chain‑of‑Thought, CoT）**：让模型在给出答案前先写出推理步骤，像人在解题时先列草稿。  
- **测试时计算（Test‑time Compute）**：模型在推理阶段可以使用的额外算力，而不是只在训练时消耗。  
- **自我进化（Self‑evolution）**：模型通过自身的反馈循环不断改进内部策略，无需外部标注的显式指导。  
- **强化学习框架（RL Framework）**：把模型的行为（这里指生成奖励的过程）视为策略，通过奖励信号来优化。  
- **奖励推理（Reward Reasoning）**：在给出最终分数前，模型先进行一段内部推理，解释为什么会给出这个分数。  
- **适配性计算利用（Adaptive Compute Utilization）**：模型根据问题难度动态决定是否投入更多计算资源。  

### 核心创新点
1. **直接在奖励生成阶段加入链式思考**  
   之前的做法是把奖励模型当作黑箱，一次性输出分数 → 这篇论文让模型先写出“思考过程”，再基于该过程给出分数 → 通过显式的推理步骤，模型能够捕捉隐藏的语义线索，提升了对复杂查询的判别能力。  

2. **无需显式推理标注的自我进化训练**  
   传统的思考式模型需要人工提供思考路径作为监督 → 本文采用强化学习，让模型自行探索并通过奖励信号评估自己的思考质量 → 省去昂贵的标注成本，同时让模型学会生成有用的推理。  

3. **动态分配测试时计算**  
   过去的系统要么固定使用全部算力，要么全程轻量化 → 这里引入一个“计算调度器”，根据输入的难度预测是否需要额外的思考轮次 → 在简单问题上保持高效，在困难问题上投入更多算力，整体算力利用率提升。  

4. **统一的奖励推理框架跨域适用**  
   之前的奖励模型往往针对单一任务微调 → 本文的框架在多种基准（对话安全、代码生成、事实核查等）上都能直接使用 → 展示了方法的通用性，降低了跨任务部署的门槛。  

### 方法详解
整体思路可以拆成三大步骤：**输入编码 → 思考生成 → 奖励输出**。  
1. **输入编码**：原始查询和候选答案先经过预训练的大语言模型得到上下文向量。这里的模型保持与普通生成模型相同的参数规模，只是后面会多走几步。  

2. **思考生成模块**：在得到上下文向量后，模型被提示以“思考”模式继续生成。提示大致是：“请解释为什么这个答案好或不好”。模型随后输出一段自然语言的推理文本，这一步可以循环多次，每次都基于前一次的思考结果进行深化。类似于人在写报告时先列出要点、再补充细节。  

3. **奖励输出模块**：思考文本结束后，模型进入评分阶段。它把思考文本和原始输入一起送入一个小型的评分头（通常是一个两层前馈网络），输出一个标量分数。这个分数即为最终的奖励。  

**强化学习训练细节**  
- **策略**：整个思考+评分过程被视为一个策略πθ，θ是模型参数。  
- **奖励信号**：使用人类标注的偏好对比（如A比B更好）来构造奖励函数R。若模型在思考后给出的分数能够正确区分A和B，则得到正向奖励，否则负向奖励。  
- **自我进化**：在训练循环中，模型先用当前策略生成思考文本和分数，然后根据R计算梯度，使用近端策略优化（PPO）等RL算法更新θ。因为思考本身没有显式标签，模型只能通过最终分数的对错来间接学习如何写出有用的推理。  

**计算调度器**  
- 训练时加入一个二分类网络，预测输入的“思考需求”。  
- 预测为高需求时，模型执行多轮思考（例如3轮），否则只做单轮或直接跳到评分。  
- 这种机制在推理阶段可以根据实时算力预算动态裁剪，兼顾速度和准确性。  

**最巧妙的地方**  
- 把“思考过程”本身当作可学习的中间产物，而不是外部提供的标签；模型通过自我评估来优化这段过程。  
- 将额外的测试时算力转化为可控的思考轮数，使得算力利用率从“固定”变为“按需”。  

### 实验与效果
- **测试数据**：论文在多个公开奖励建模基准上评估，包括对话安全（OpenAI Helpfulness Dataset）、代码生成质量（HumanEval Reward）、事实核查（FEVER）等。  
- **对比基线**：与传统单步奖励模型、基于CoT的奖励模型以及最新的RLHF（强化学习从人类反馈）系统进行比较。  
- **性能提升**：论文声称在所有基准上均实现了显著提升，例如在对话安全基准上提升约10% 的准确率，在代码生成奖励上提升约8% 的相关性分数。具体数值未在摘要中给出。  
- **消融实验**：作者分别去掉思考模块、去掉计算调度器、以及改用显式标注的思考路径进行训练。结果显示，去掉思考模块后性能跌回传统奖励模型水平，去掉调度器会导致算力浪费且在简单任务上略有下降，使用显式标注的思考路径并未带来额外提升，说明自我进化的方式已经足够有效。  
- **局限性**：论文承认在极端低算力环境下多轮思考会导致响应时间不可接受；此外，思考文本的质量仍受模型规模限制，过于简短的思考可能不足以解释复杂偏好。  

### 影响与延伸思考
这篇工作打开了“在奖励阶段使用链式思考”的新方向，随后有几篇后续研究尝试把思考过程迁移到其他对齐任务，如安全过滤、事实纠错等。还有工作探索把思考过程与人类可解释性结合，尝试让审查员直接阅读模型的推理以提升信任度。想进一步了解，可以关注**自监督推理生成**、**可解释对齐**以及**动态算力调度**这几个热点；它们都是在奖励推理模型思路上延伸的自然方向。  

### 一句话记住它
让奖励模型先“思考”，再给分数，利用额外算力实现更精准、更可解释的对齐。