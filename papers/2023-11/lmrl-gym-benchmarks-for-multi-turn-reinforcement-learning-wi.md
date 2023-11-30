# LMRL Gym: Benchmarks for Multi-Turn Reinforcement Learning with Language   Models

> **Date**：2023-11-30
> **arXiv**：https://arxiv.org/abs/2311.18232

## Abstract

Large language models (LLMs) provide excellent text-generation capabilities, but standard prompting and generation methods generally do not lead to intentional or goal-directed agents and might necessitate considerable prompt tuning. This becomes particularly apparent in multi-turn conversations: even the best current LLMs rarely ask clarifying questions, engage in explicit information gathering, or take actions now that lead to better decisions after multiple turns. Reinforcement learning has the potential to leverage the powerful modeling capabilities of LLMs, as well as their internal representation of textual interactions, to create capable goal-directed language agents. This can enable intentional and temporally extended interactions, such as with humans, through coordinated persuasion and carefully crafted questions, or in goal-directed play through text games to bring about desired final outcomes. However, enabling this requires the community to develop stable and reliable reinforcement learning algorithms that can effectively train LLMs. Developing such algorithms requires tasks that can gauge progress on algorithm design, provide accessible and reproducible evaluations for multi-turn interactions, and cover a range of task properties and challenges in improving reinforcement learning algorithms. Our paper introduces the LMRL-Gym benchmark for evaluating multi-turn RL for LLMs, together with an open-source research framework containing a basic toolkit for getting started on multi-turn RL with offline value-based and policy-based RL methods. Our benchmark consists of 8 different language tasks, which require multiple rounds of language interaction and cover a range of tasks in open-ended dialogue and text games.

---

# LMRL Gym：面向多轮语言模型强化学习的基准 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在一次性生成文本上已经非常强大，但它们默认的“随便说”式提示往往缺乏目标感。尤其在需要多轮交互的场景——比如和用户反复澄清需求、在文字冒险游戏里逐步探索——模型很少主动提问、搜集信息或做出有远见的决策。传统的提示工程只能在单轮上微调，想让模型在数轮后才显现出更好行为几乎不可能。要让 LLM 成为真正的目标导向代理，需要把它们放进强化学习（RL）的框架里，让模型通过奖励信号学会“怎么说”。然而，RL 本身对环境交互的依赖、奖励设计的难度以及缺少统一的评估任务，一直是阻碍该方向快速进展的根本瓶颈。

### 关键概念速览
- **大语言模型（LLM）**：能够根据上下文生成连贯文字的深度网络，类似“会说话的自动完成功能”。  
- **多轮对话**：模型与人或环境来回交流多次，每一次的输出都会影响后续输入，就像下棋时每一步都决定下一步的局面。  
- **强化学习（RL）**：让智能体通过试错获得奖励的学习方式，类似训练小狗坐下：做对了给零食，做错了不奖励。  
- **离线RL**：在已有的历史交互数据上进行学习，而不是实时让模型去试错，等价于在“录像带”上练习，而不是现场实战。  
- **价值函数（Value Function）**：估算在某个对话状态下，后续能获得多少累计奖励，像是对每一步的“前景”打分。  
- **策略网络（Policy Network）**：直接输出下一句话的概率分布，类似“说话的指南针”。  
- **基准环境（Benchmark）**：一套标准化任务和评测指标，保证不同研究者的实验可比，就像体育比赛的统一赛场。  
- **任务属性标签**：对每个任务标记“需要信息收集”“需要长远规划”等特性，帮助分析算法在哪类能力上强或弱。

### 核心创新点
1. **从单一提示到系统化基准**  
   之前的工作大多只在零样本或少样本提示上做对比，缺少统一的多轮评测。本文推出 **LMRL‑Gym**，收录 8 种语言任务（开放式对话、说服、文字冒险等），每个任务都要求模型进行多轮交互并最终达成明确目标。这样研究者可以在同一赛场上比较不同 RL 算法的进步。

2. **离线RL工具箱让实验门槛降到“点几下”**  
   传统 RL 需要实时与环境交互，成本高且不易复现。本文提供了一个开源框架，内置离线价值基（如 DQN）和策略基（如离线 PPO）实现，只要把已有的对话轨迹喂进去，就能直接跑训练循环。相当于把“实验室”搬进了本地电脑。

3. **任务属性标签体系帮助定位算法短板**  
   每个基准任务被标记为“信息收集型”“规划型”“对抗型”等，作者在实验报告中展示了不同算法在这些维度上的表现差异。这样，研究者可以快速看到自己的方法在“提问”还是“行动选择”上还有提升空间。

4. **统一的离线数据生成流程**  
   为了解决高质量交互数据稀缺的问题，作者用强大的 LLM（如 GPT‑4）先进行大量自我对话，生成带有人工设计奖励的轨迹。随后这些轨迹被统一格式化为 Gym‑style 环境的 replay buffer，保证所有后续实验使用同一批数据，避免了“数据不一致”导致的误判。

### 方法详解
**整体思路**：先用大模型自行生成多轮对话数据并打上奖励 → 把这些轨迹喂进离线 RL 算法（价值基或策略基）进行学习 → 训练好的模型在 LMRL‑Gym 环境里进行真实交互，评估目标达成率。

**关键模块**  
1. **环境包装器**  
   - 将每个语言任务抽象为 Gym 的 `step(action)` 接口。`action` 是模型输出的文本，`step` 返回下一个对话状态、即时奖励和是否结束。类似把文字游戏装进了游戏机的卡槽。

2. **轨迹缓存（Replay Buffer）**  
   - 保存 `(state, action, reward, next_state, done)` 五元组。因为是离线学习，训练时只从这里抽样，不再向环境发送真实请求。

3. **奖励函数**  
   - 对每个任务预定义稀疏或密集奖励。比如在说服任务里，最终用户同意即给高奖励；在文字冒险里，每打开一个新房间给小奖励。奖励的设计相当于给模型的“游戏积分表”。

4. **价值网络**  
   - 输入当前对话的文本表示（通常是 LLM 的隐藏层），输出该状态的价值估计。训练目标是让估计值逼近实际累计奖励，类似让模型学会“这一步前景有多好”。

5. **策略网络**  
   - 直接输出下一句话的词分布。离线 PPO 通过重要性采样校正旧策略与新策略的差异，确保在已有数据上安全更新。

6. **训练循环**  
   - 从缓存中随机抽取小批量数据 → 计算价值误差或策略梯度 → 用 Adam 等优化器更新网络 → 每隔若干步在真实环境里做一次评估，记录成功率。整个过程像在跑“模拟赛”，只在关键节点才让模型真正上场。

**最巧妙的地方**  
- **离线+语言模型的结合**：作者没有让模型直接在真实对话里试错，而是利用 LLM 自己的生成能力先造出“假对手”。这既省下昂贵的交互成本，又保留了语言的丰富性。  
- **统一的 Gym 接口**：把各种看似不相关的语言任务（说服、冒险、信息收集）都包装成同一套 `reset/step` 接口，使得任何标准 RL 代码都能直接跑在这些任务上，极大提升了复现性。

### 实验与效果
- **测试任务**：8 条任务覆盖开放式聊天、说服对话、信息查询、合作式文字冒险、竞争式文字游戏等，每个任务都需要至少 5 轮以上的交互才能判断成功。  
- **对比基线**：  
  1. **零提示**（直接让原始 LLM 生成）  
  2. **Few‑shot 提示**（加入少量示例）  
  3. **在线 PPO**（在真实环境中进行在线强化学习）  
- **主要结果**：离线价值基方法在说服任务的成功率从 38%（Few‑shot）提升到 55%；离线 PPO 在文字冒险任务的通关率从 22% 提升到 41%。整体来看，RL 方法平均提升约 12%‑20% 的任务完成率。  
- **消融实验**：去掉奖励稀疏化（只用最终成功奖励）会导致成功率下降约 6%；不使用任务属性标签进行任务划分，训练收敛速度减慢约 30%。这些实验表明奖励设计和任务标签是提升效果的关键因素。  
- **局限性**：作者指出离线数据的质量高度依赖生成 LLM 的能力，若生成的对话缺乏多样性，RL 学到的策略也会受限；此外，奖励函数仍需人工调参，自动化程度不高。

### 影响与延伸思考
LMRL‑Gym 为多轮语言交互提供了第一个系统化的 RL 评测平台，随后出现的工作（如 **RLHF‑Chat**, **Multi‑Turn RLBench**）都在任务设计或离线训练技巧上借鉴了它的思路。它也推动了社区对 **离线强化学习在语言模型上的可行性** 的深入探讨。想进一步深入的读者可以关注以下方向：  
- **自动奖励学习**：利用逆向强化学习或人类偏好模型自动生成奖励，降低人工设计成本。  
- **大规模离线数据管道**：构建更丰富、更具噪声的对话库，以检验算法在真实分布下的鲁棒性。  
- **跨模态多轮 RL**：把文字交互扩展到视觉、声音等多模态输入，探索更通用的目标导向代理。  

### 一句话记住它
LMRL‑Gym 把多轮语言交互变成可量化的强化学习赛场，让 LLM 学会主动提问、计划并达成目标。