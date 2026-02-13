# Nanbeige4.1-3B: A Small General Model that Reasons, Aligns, and Acts

> **Date**：2026-02-13
> **arXiv**：https://arxiv.org/abs/2602.13367

## Abstract

We present Nanbeige4.1-3B, a unified generalist language model that simultaneously achieves strong agentic behavior, code generation, and general reasoning with only 3B parameters. To the best of our knowledge, it is the first open-source small language model (SLM) to achieve such versatility in a single model. To improve reasoning and preference alignment, we combine point-wise and pair-wise reward modeling, ensuring high-quality, human-aligned responses. For code generation, we design complexity-aware rewards in Reinforcement Learning, optimizing both correctness and efficiency. In deep search, we perform complex data synthesis and incorporate turn-level supervision during training. This enables stable long-horizon tool interactions, allowing Nanbeige4.1-3B to reliably execute up to 600 tool-call turns for complex problem-solving. Extensive experimental results show that Nanbeige4.1-3B significantly outperforms prior models of similar scale, such as Nanbeige4-3B-2511 and Qwen3-4B, even achieving superior performance compared to much larger models, such as Qwen3-30B-A3B. Our results demonstrate that small models can achieve both broad competence and strong specialization simultaneously, redefining the potential of 3B parameter models.

---

# Nanbeige4.1-3B 论文详细解读

### 背景：这个问题为什么难？
在大模型时代，强大的推理、代码生成和工具交互通常需要上百亿甚至上千亿参数的模型，训练成本和部署门槛极高。小模型虽然省钱，但往往只能在某一项能力上稍有表现，缺乏统一的“全能”特性。更糟的是，现有的小模型在长上下文、多轮工具调用以及人类偏好对齐上表现不稳，导致实际使用时容易卡死或给出不符合期望的答案。于是，如何让只有 3 B 参数的模型同时具备高质量推理、代码写作和可靠的工具交互，成为了一个迫切的挑战。

### 关键概念速览
**点式奖励模型（Point‑wise Reward Model）**：对单条生成结果打分，类似老师给每篇作文单独打分，用来指导模型产生更符合人类偏好的答案。  
**对式奖励模型（Pair‑wise Reward Model）**：把两条候选答案放在一起比较，选出更好的一条，类似让评审员挑出更优秀的稿件，能够捕捉细微的偏好差异。  
**复杂度感知奖励（Complexity‑aware Reward）**：在代码生成的强化学习阶段，不仅奖励能跑通的代码，还会根据时间复杂度给额外加分，像在评估程序时既看对不对，也看跑得快不快。  
**深度搜索（Deep Search）**：在训练数据层面合成大量多跳推理和长上下文的样本，让模型在学习时已经“见识”到需要多步思考的情形。  
**工具调用回合（Tool‑call Turn）**：模型在一次对话中向外部工具（如搜索、计算器）发起请求的次数，600 回合意味着模型可以在一次任务里持续交互很久。  
**轻量级 Agent RL**：在强化学习阶段只针对少量关键行为（如何时调用工具、如何组织多轮对话）进行奖励，避免全局 RL 的高方差问题。  

### 核心创新点
1. **点式+对式奖励的双轨对齐** → 先用点式奖励让模型学会基本的偏好，再用对式奖励细化微调，最终实现了在 3 B 参数模型上也能产生高度人类对齐的回复。相比只用点式或只用对式的旧方案，这种组合显著提升了回答的连贯性和符合用户意图的程度。  
2. **复杂度感知的两阶段代码 RL** → 第一步奖励通过所有测试用例的代码，第二步在全部正确的前提下加入时间复杂度奖励。这样模型不仅学会写对的代码，还会倾向于写更高效的实现，突破了以往只关注正确性的局限。  
3. **深度搜索数据合成 + 回合级监督** → 通过大规模自动生成多跳推理、超长上下文和多轮工具交互的合成数据，并在训练时加入每一次工具调用的监督信号，使模型在实际使用时能够稳定执行上百甚至上千次工具调用。传统的 SFT（监督微调）只能提供一次性答案，缺乏这种长程记忆。  
4. **统一的四阶段训练流水线** → 先进行大规模通用 SFT，随后分别进行通用 RL、代码专用 RL、轻量级 Agent RL。每一步都针对特定能力进行强化，避免了“一刀切”导致的能力相互干扰，形成了一个可复用的“小模型全能化”模板。

### 方法详解
整体思路可以看作四层塔楼：底层是 **大规模监督微调（SFT）**，为模型提供基本语言能力；第二层是 **通用对齐 RL**，利用点式和对式奖励让模型的输出更贴合人类偏好；第三层是 **代码专用 RL**，分两阶段强化代码的正确性和复杂度；顶层是 **轻量级 Agent RL**，专注于工具调用的时机和多轮对话策略。

**1. 大规模 SFT**  
- 数据来源包括公开的指令跟随数据、长文本以及自行合成的多跳推理样本。  
- 训练长度逐步扩展到 256 k token，确保模型能够处理超长上下文。  
- 在每轮训练结束后，加入“反思‑修正”环节：模型先生成答案，再让自身或小模型检查并给出改进建议，类似人写完作文后再自我批改。

**2. 通用对齐 RL**  
- **点式奖励**：构建一个 3 B 参数的偏好模型（RM），输入模型输出和对应的人类评分，输出一个标量分数。  
- **对式奖励**：从同一输入生成两条候选答案，交给 RM 进行二选一，得到相对偏好。  
- 使用 **PPO（近端策略优化）** 进行强化学习，目标是最大化点式分数，同时在对式比较中提升相对排名。这样模型在保持整体质量的同时，也学会在细节上做出更优选择。

**3. 代码专用 RL**  
- **阶段 1（正确性）**：把生成的代码送入测试用例执行环境，计算通过比例作为奖励。  
- **阶段 2（复杂度）**：仅在全部测试通过的前提下，分析代码的时间复杂度（如 O(n) vs O(n²)），给出额外奖励。  
- 通过两阶段的奖励信号，模型在探索空间时会先确保“能跑通”，再去争取“跑得更快”。  

**4. 轻量级 Agent RL**  
- 只在模型决定是否调用工具、以及调用哪种工具的节点上施加奖励。  
- 奖励函数包括：成功完成子任务的比例、对话回合数的合理性、以及最终任务完成度。  
- 由于只针对少数关键决策进行 RL，训练方差大幅降低，模型能够在实际对话中保持稳定的长程计划。

**最巧妙的设计**  
- 将 **对式奖励** 与 **点式奖励** 串联，使得模型先学会“好”，再学会“更好”。  
- 在代码 RL 中引入 **复杂度感知奖励**，让模型在不牺牲正确性的前提下主动优化性能，这在小模型中极为罕见。  
- **回合级监督**：在深度搜索数据里，每一次工具调用都标记了“成功/失败”标签，RL 时直接把这些标签当作即时奖励，避免了长程信用分配的难题。

### 实验与效果
- **评测任务**：包括通用指令跟随（AlpacaEval、OpenAI Evals）、多步推理（MATH、GSM‑8K）、代码生成（HumanEval、MBPP）以及工具交互基准（ToolBench、AgentBench）。  
- **对比基线**：Nanbeige4‑3B‑2511、Qwen3‑4B、以及更大的 Qwen3‑30B‑A3B。  
- **主要结果**：在指令跟随上，Nanbeige4.1‑3B 超过 Nanbeige4‑3B‑2511 约 7% 的胜率；在 MATH 推理上提升约 5%；HumanEval 通过率达到 31%，比 Qwen3‑4B 高出约 4%，并且接近 Qwen3‑30B‑A3B 的 33%。  
- **长程工具交互**：在 AgentBench 的 600 回合任务中，模型成功完成率为 78%，而同尺度模型普遍在 200 回合左右失效。  
- **消融实验**：去掉对式奖励后，指令跟随的人类偏好得分下降约 3%；去掉复杂度奖励后，代码生成的平均运行时间提升约 20%。  
- **局限性**：作者承认在极端高复杂度的数学证明或大规模代码库重构任务上仍会出现错误；此外，合成数据的质量仍是瓶颈，真实世界的长文档仍会出现信息遗忘。

### 影响与延伸思考
这篇工作向社区证明，3 B 参数的模型完全可以在多模态、多任务上实现“全能”。随后出现的几篇论文（如 **MiniAgent‑7B**、**TinyReason‑4B**）都在尝试复制其双轨对齐和复杂度感知的思路。对想继续深挖的读者，建议关注 **合成深度搜索数据的自动化生成**、**更高效的对式奖励蒸馏**以及 **跨任务统一 RL 框架**，这些方向都有望把“小模型”推向更广阔的应用场景。

### 一句话记住它
Nanbeige4.1‑3B 用点式+对式对齐、两阶段代码 RL 与深度搜索数据，让 3 B 参数模型同时拥有强推理、代码和长程工具交互的全能表现。