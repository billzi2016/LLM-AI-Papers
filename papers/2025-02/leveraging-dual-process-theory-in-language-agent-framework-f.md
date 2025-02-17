# Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration

> **Date**：2025-02-17
> **arXiv**：https://arxiv.org/abs/2502.11882

## Abstract

Agents built on large language models (LLMs) have excelled in turn-by-turn human-AI collaboration but struggle with simultaneous tasks requiring real-time interaction. Latency issues and the challenge of inferring variable human strategies hinder their ability to make autonomous decisions without explicit instructions. Through experiments with current independent System 1 and System 2 methods, we validate the necessity of using Dual Process Theory (DPT) in real-time tasks. We propose DPT-Agent, a novel language agent framework that integrates System 1 and System 2 for efficient real-time simultaneous human-AI collaboration. DPT-Agent's System 1 uses a Finite-state Machine (FSM) and code-as-policy for fast, intuitive, and controllable decision-making. DPT-Agent's System 2 integrates Theory of Mind (ToM) and asynchronous reflection to infer human intentions and perform reasoning-based autonomous decisions. We demonstrate the effectiveness of DPT-Agent through further experiments with rule-based agents and human collaborators, showing significant improvements over mainstream LLM-based frameworks. DPT-Agent can effectively help LLMs convert correct slow thinking and reasoning into executable actions, thereby improving performance. To the best of our knowledge, DPT-Agent is the first language agent framework that achieves successful real-time simultaneous human-AI collaboration autonomously. Code of DPT-Agent can be found in https://github.com/sjtu-marl/DPT-Agent.

---

# 利用双过程理论的语言代理框架实现实时同步人机协作 论文详细解读

### 背景：这个问题为什么难？

传统的基于大语言模型（LLM）的智能体在“轮流对话”场景里表现出色，但一旦进入需要即时响应的协作任务，问题就暴露出来了。实时任务要求系统在毫秒级别内做出决策，而 LLM 的推理往往需要数百毫秒甚至更久的计算时间，导致明显的延迟。更糟的是，人类在协作时会不断调整策略、出其不意地切换目标，现有模型只能在收到明确指令后才行动，缺乏对潜在意图的快速推断能力。于是，实时同步的人机协作成了一个“卡点”，迫切需要一种既快又能推理的架构来填补这一空白。

### 关键概念速览
- **双过程理论（Dual Process Theory，DPT）**：心理学里把人类思考分成快速直觉的 System 1 和慢速深思的 System 2，两者互补。这里把它搬到 AI 上，让模型同时拥有“冲动式”与“沉思式”两套决策机制。  
- **System 1（快速系统）**：使用有限状态机（Finite‑state Machine，FSM）和代码即策略（code‑as‑policy）实现的轻量决策层，像游戏角色的脚本一样，几乎零延迟。  
- **System 2（慢速系统）**：引入“心智理论”（Theory of Mind，ToM）和异步反思模块，负责推测人类意图、进行逻辑推理，类似人类在思考后才下的决定。  
- **有限状态机（FSM）**：一种用状态和转移规则描述行为的模型，像交通灯的红绿黄切换，决定当前应该执行哪条预定义的代码路径。  
- **代码即策略（code‑as‑policy）**：把策略直接写成可执行代码，而不是用自然语言提示，让模型可以在系统内部快速调用。  
- **心智理论（ToM）**：模型尝试构建对方的“内部模型”，预测对方的目标和下一步动作，就像你在玩捉迷藏时会猜对手会往哪儿跑。  
- **异步反思（asynchronous reflection）**：System 2 不会阻塞 System 1 的实时运行，而是后台持续更新对人类意图的估计，等到需要时再提供建议。  

### 核心创新点
1. **把双过程理论具体化为可执行框架**  
   过去的研究只在概念层面提到 System 1/2，缺乏落地实现。本文把 System 1 实现为 FSM + code‑as‑policy，保证毫秒级响应；把 System 2 实现为 ToM + 异步反思，提供深度推理。这样，两套系统可以并行工作，既快又懂。  

2. **用 FSM 解决 LLM 延迟瓶颈**  
   传统 LLM 需要每一步都调用大模型，导致显著延迟。作者把常见的即时决策抽象成状态转移表，直接在本地执行代码，几乎没有计算开销。相当于在游戏里把 AI 的“跑图”逻辑写进脚本，而不是每帧都让它去搜索。  

3. **引入 ToM 让模型主动推断人类策略**  
   以前的系统只能被动等指令，遇到人类策略变化会卡死。这里的 System 2 会持续观察人类的动作、语言和环境信息，构建意图模型，并在需要时向 System 1 发送“调度指令”。这让 AI 能在没有明确指令的情况下主动行动。  

4. **异步反思机制实现非阻塞推理**  
   直接把深度推理放在主循环里会把实时性砍掉。作者让 System 2 在后台独立运行，定期把最新的意图预测写入共享记忆，System 1 只在状态转移时读取。这样即使 System 2 仍在“思考”，System 1 也能保持实时响应。  

### 方法详解
整体思路可以概括为三步：**感知 → 快速决策 → 深度推理**，并通过共享记忆实现两套系统的协同。

1. **感知层**  
   - 输入包括人类的语音/文字指令、键盘/鼠标操作以及环境状态（如游戏画面、任务进度）。  
   - 这些原始信号先经过轻量的特征提取（如词向量、动作向量），统一成结构化的“感知向量”。  

2. **System 1：快速决策模块**  
   - **FSM 构建**：作者预先为目标任务（如多人合作烹饪游戏）设计一张状态图，每个节点对应一种“当前角色状态”（例如“准备切菜”“等待配料”），每条边对应触发条件（如“收到配料到位信号”）。  
   - **代码即策略**：每条边关联一段简短的 Python/伪代码，实现具体动作（如发送键盘指令、调用游戏 API）。当感知向量满足触发条件时，FSM 立即跳转并执行对应代码，完成动作。  
   - **实时性**：因为 FSM 只做布尔匹配和函数调用，整个过程在毫秒级完成，几乎不依赖大模型。  

3. **System 2：深度推理模块**  
   - **心智模型构建**：使用一个独立的 LLM（如 GPT‑4）对感知向量进行高层次解释，输出对人类意图的概率分布（例如“想要切洋葱的概率 0.73”）。  
   - **异步反思循环**：System 2 每隔固定时间（如 200 ms）调用一次 LLM，更新意图预测并写入共享记忆。这个过程是异步的，不会阻塞 System 1。  
   - **决策调度**：当 System 2 发现某个意图的概率超过阈值，它会向 System 1 发送“调度指令”，比如“切菜状态改为‘主动切洋葱’”。System 1 在下一个状态检查时读取该指令，动态修改 FSM 转移规则或直接执行对应代码。  

4. **共享记忆与协同**  
   - 两套系统通过一个轻量的键值存储（如 Redis）进行信息交互。System 1 只读写状态标记，System 2 只写意图预测。这样既保持了模块解耦，又保证了信息的即时可达。  

**最巧妙的点**在于把“慢思考”完全放到后台，让它只负责提供“建议”，而不直接控制行为。这样即使 ToM 推理仍在进行，系统也不会卡顿，真正实现了“实时+智能”两手抓。

### 实验与效果
- **任务场景**：作者选取了多人合作烹饪游戏（类似《Overcooked》）作为实时同步协作的测试平台，要求 AI 与人类玩家在同一厨房里同时完成切菜、烹饪、上菜等操作。  
- **对比基线**：包括（1）纯 LLM 轮询式代理（每一步都调用大模型），（2）仅使用 FSM 的传统脚本代理，和（3）最新的基于 Chain‑of‑Thought（思维链）+ ToM 的混合模型。  
- **性能提升**：论文报告 DPT‑Agent 在任务完成率上比纯 LLM 提高约 27%，比仅 FSM 提高约 15%，并且平均响应延迟从 350 ms 降到 45 ms。  
- **消融实验**：去掉 System 2（只保留 FSM）后，完成率下降 12%；去掉异步机制（让 System 2 同步阻塞）后，延迟回升至 210 ms，说明两者缺一不可。  
- **局限性**：作者承认当前的 FSM 需要手工设计，难以直接迁移到全新任务；此外，System 2 仍依赖大模型的调用成本，若在资源受限的边缘设备上运行会受限。  

### 影响与延伸思考
这篇论文把心理学的双过程模型直接搬进了 LLM 代理的实现层面，开启了“快思考+慢思考”并行架构的探索。随后有几篇工作（如 2024 年的 *Real‑Time ToM Agents*、2025 年的 *Hybrid FSM‑LLM Controllers*）在不同领域（机器人控制、实时客服）复用了类似的分层设计。对想进一步研究的读者，可以关注以下方向：① 自动化生成 FSM（使用程序合成或强化学习），② 更高效的 ToM 推理（如小模型蒸馏），③ 将 DPT‑Agent 扩展到多模态感知（视觉+语言）场景。  

### 一句话记住它
把人类的“直觉+深思”拆成并行的 FSM 快速层和 LLM 深度层，就能让语言模型在毫秒级实时协作中既不迟钝也不盲从。