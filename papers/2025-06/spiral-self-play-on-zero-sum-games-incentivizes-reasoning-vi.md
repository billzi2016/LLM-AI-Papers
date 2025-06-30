# SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning

> **Date**：2025-06-30
> **arXiv**：https://arxiv.org/abs/2506.24119

## Abstract

Recent advances in reinforcement learning have shown that language models can develop sophisticated reasoning through training on tasks with verifiable rewards, but these approaches depend on human-curated problem-answer pairs and domain-specific reward engineering. We introduce SPIRAL, a self-play framework where models learn by playing multi-turn, zero-sum games against continuously improving versions of themselves, generating an automatic curriculum of stronger opponents, and eliminating the need for human supervision. To enable this self-play training at scale, we implement a fully online, multi-turn, multi-agent reinforcement learning system for LLMs and propose role-conditioned advantage estimation (RAE) to stabilize multi-agent training. SPIRAL produces reasoning capabilities that transfer broadly, improving performance by up to 10% across a suite of 8 reasoning benchmarks on 4 different models spanning Qwen and Llama model families, outperforming supervised fine-tuning on 25,000 expert game trajectories. Multi-game training (TicTacToe, Kuhn Poker, Simple Negotiation) yields the strongest results, with improvements observed across both base and instruction-tuned models. Analysis of chain-of-thought traces reveals that games develop distinct cognitive patterns that transfer to improve reasoning performance, with different games developing complementary strengths. Even models which have already been trained on reasoning tasks using RLVR, like DeepSeek-R1-Distill-Qwen-7B, still benefit from our approach. These results demonstrate that zero-sum games naturally develop transferable reasoning capabilities across diverse model architectures and training stages, highlighting a promising direction for autonomous reasoning development. Our code can be found in https://github.com/spiral-rl/spiral.

---

# SPIRAL：通过零和多智能体多回合强化学习的自博弈激励推理 论文详细解读

### 背景：这个问题为什么难？

在语言模型里让模型学会像人一样推理，一直是个硬核挑战。过去的成功大多依赖“人类标注的题目‑答案对”或是针对特定领域精心设计的奖励函数，这意味着训练过程需要大量人工投入，而且模型的推理能力往往只能在训练时涉及的任务上发挥。换句话说，模型缺少一种能够自行生成、评估并提升推理难度的机制。于是，如何让大模型在没有人工监督的情况下，自动产生有意义的学习目标并逐步提升自己的推理水平，成为了亟待突破的瓶颈。

### 关键概念速览
- **自博弈（Self‑Play）**：模型与自身的不同版本进行对抗，就像棋手和自己的过去版本对弈，能够不断制造更强的对手，推动自身进步。  
- **零和游戏（Zero‑Sum Game）**：一种两方对抗的博弈，双方的收益总和为零，一方赢即另一方输，保证了竞争的纯粹性。  
- **多回合（Multi‑Turn）**：游戏不是一次性决策，而是需要在多轮交互中逐步做出行动，类似于对话或谈判，需要记忆和前后文推理。  
- **多智能体强化学习（Multi‑Agent RL）**：同时训练多个智能体，每个智能体的策略会影响其他智能体的奖励，训练过程比单智能体更复杂。  
- **角色条件优势估计（Role‑Conditioned Advantage Estimation, RAE）**：在多智能体环境中，为每个角色单独估计优势（即策略改进的方向），帮助梯度更稳。可以把它想成给每位玩家配备专属的“成绩单”。  
- **链式思考（Chain‑of‑Thought, CoT）**：模型在给出最终答案前，先把思考步骤写出来，类似于人做数学题时的草稿。  
- **自动课程（Automatic Curriculum）**：训练过程中，难度会随对手实力自然提升，模型不需要外部手工安排学习顺序。  

### 核心创新点
1. **从监督数据到自博弈**：传统方法需要成千上万的人类标注题目来提供奖励信号。SPIRAL 把奖励完全交给零和游戏本身的输赢结果，让模型在自博弈中自行产生学习信号。这样做把“人类提供答案”替换成“游戏本身告诉谁赢了”，彻底摆脱了人工标注的依赖。  
2. **全在线多回合多智能体系统**：以往的多智能体强化学习大多在离线环境里先生成轨迹再训练，效率低且难以适配大语言模型。SPIRAL 实现了一个实时的训练循环：模型在每一步生成对话/行动，立即计算奖励并更新参数，保证了训练规模可以匹配数十亿参数的 LLM。  
3. **角色条件优势估计（RAE）**：在多智能体对抗中，普通的优势估计会因为不同角色的奖励分布差异而产生高方差，导致训练不稳。RAE 为每个角色单独估计优势，显著降低了梯度噪声，使得大模型在自博弈中能够保持收敛。  
4. **多游戏联合训练**：作者没有局限于单一游戏，而是让模型同时在井字棋、Kuhn 扑克和简易协商三种零和游戏中训练。不同游戏强调的推理模式互补，最终得到的模型在通用推理基准上提升最大，证明了“游戏多样性 = 推理多样性”。  

### 方法详解
**整体框架**  
SPIRAL 的训练可以划分为四个阶段：① 初始化模型（可以是任意预训练的 LLM），② 生成自博弈对局，③ 计算基于输赢的奖励并进行角色条件优势估计，④ 用多智能体强化学习的梯度更新模型。整个过程是全在线的：每轮对局结束后立即进行梯度更新，随后生成下一轮更强的对手。

**关键模块拆解**  

1. **自博弈对局生成**  
   - **角色分配**：在每局游戏中，模型被克隆成两个角色（如“X”和“O”），每个角色拥有独立的隐藏状态（相当于记忆）。  
   - **轮流行动**：模型在每个回合接收当前局面描述（包括对手上一步的文字动作），输出自己的下一步文字指令。因为是语言模型，指令可以是自然语言描述的棋子位置、出牌或谈判提议。  
   - **对手进化**：每隔若干轮，系统会把当前最强的策略保存为“历史版本”，新一轮的自博弈会在旧版本和最新版本之间进行，形成自动课程。  

2. **奖励与优势估计**  
   - **零和奖励**：游戏结束时，胜者得到 +1，败者 -1，平局 0。这样奖励天然符合零和特性。  
   - **角色条件优势估计（RAE）**：对每个角色，系统会收集该角色在对局中的价值估计（比如使用价值网络），再减去该角色的基准价值，得到优势。因为每个角色的基准是独立的，梯度不会因为对手的策略波动而被放大。  

3. **多智能体强化学习更新**  
   - **策略梯度**：使用类似 PPO（近端策略优化）的算法，但在优势计算上换成 RAE。  
   - **共享语言模型**：虽然两个角色的策略是同一个模型的不同实例，但梯度会在两个方向上累加，等价于在同一网络上进行对抗学习。  
   - **在线更新**：每完成一局游戏，就立即执行一次梯度步，保证模型的参数始终在追赶最新的对手。  

4. **多游戏并行**  
   - **任务调度**：训练循环会随机抽取三种游戏之一进行对局，保持每种游戏的比例大致相同。  
   - **共享参数**：所有游戏共用同一语言模型参数，只是输入/输出的格式不同。这样模型在不同游戏中学到的推理技巧会相互迁移。  

**最巧妙的设计**  
RAE 是整个系统的“稳压器”。如果直接使用普通优势估计，模型在对抗中会出现梯度爆炸或消失，尤其是当对手策略快速升级时。RAE 把每个角色的价值基准固定在该角色自身的历史表现上，等价于给每位玩家配了一个“个人教练”，让学习过程更平滑。  

### 实验与效果
- **测试任务**：作者在 8 项推理基准上评估了 SPIRAL 的迁移能力，包括数学推理、逻辑推理、常识问答等。  
- **模型族**：实验覆盖了 Qwen 系列和 Llama 系列的 4 种模型，从 7B 到 13B 参数不等。  
- **对比基线**：与传统的监督微调（使用 25,000 条专家游戏轨迹）以及 RLVR（基于人类奖励的强化学习）相比，SPIRAL 在所有基准上都有提升，最高提升约 10%。例如，在 GSM8K（数学推理）上，7B 模型的准确率从 38% 提升到 44%。  
- **消融实验**：去掉 RAE 后，训练不稳定，最终性能下降约 3%~5%；只使用单一游戏（如仅玩井字棋）时，提升幅度约为 4%，而多游戏联合训练则达到 10%，说明游戏多样性是关键。  
- **局限性**：论文指出，当前框架仍然依赖于能够用自然语言描述的游戏规则，难以直接迁移到更复杂的、规则不易语言化的任务上。此外，训练成本仍然高于单纯的监督微调，需要多卡 GPU 长时间运行。  

### 影响与延伸思考
SPIRAL 把“自博弈”从棋类游戏扩展到语言模型的推理训练，打开了大模型自主生成学习课程的新思路。后续工作已经开始探索更丰富的零和场景（如多方谈判、合作-竞争混合游戏）以及把 RAE 融入更通用的多智能体框架。还有研究尝试把自博弈产生的对话数据直接用于少量监督微调，形成“自监督+自博弈”双管齐下的训练方案。想进一步了解，可以关注以下方向：① 更高效的在线多智能体 RL 算法（如分布式 PPO 改进），② 将自博弈与人类反馈相结合的混合奖励体系，③ 将游戏规则学习（即让模型自己发现规则）与 SPIRAL 结合，真正实现“无规则自博弈”。  

### 一句话记住它
让大语言模型在零和多回合游戏中自我对弈，自动生成更强对手，从而在没有任何人工标注的情况下学会通用推理。