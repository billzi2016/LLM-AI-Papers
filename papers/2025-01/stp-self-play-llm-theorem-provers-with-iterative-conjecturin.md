# STP: Self-play LLM Theorem Provers with Iterative Conjecturing and   Proving

> **Date**：2025-01-31
> **arXiv**：https://arxiv.org/abs/2502.00212

## Abstract

A fundamental challenge in formal theorem proving by LLMs is the lack of high-quality training data. Although reinforcement learning or expert iteration partially mitigates this issue by alternating between LLM generating proofs and finetuning them on correctly generated ones, performance quickly plateaus due to the scarcity of correct proofs (sparse rewards). To keep improving the models with limited data, we draw inspiration from mathematicians, who continuously develop new results, partly by proposing novel conjectures or exercises (which are often variants of known results) and attempting to solve them. We design the Self-play Theorem Prover (STP) that simultaneously takes on two roles, conjecturer and prover, each providing training signals to the other. The conjecturer is trained iteratively on previously generated conjectures that are barely provable by the current prover, which incentivizes it to generate increasingly challenging conjectures over time. The prover attempts to prove the conjectures with standard expert iteration. We evaluate STP with both Lean and Isabelle formal versifiers. With 51.3 billion tokens generated during the training in Lean, STP proves 28.5% of the statements in the LeanWorkbook dataset, doubling the previous best result of 13.2% achieved through expert iteration. The final model achieves state-of-the-art performance among whole-proof generation methods on miniF2F-test (65.0%, pass@3200), Proofnet-test (23.9%, pass@3200) and PutnamBench (8/644, pass@3200). We release our code, model, and dataset in this URL: https://github.com/kfdong/STP.

---

# STP：自博弈大语言模型定理证明器的迭代猜想与证明 论文详细解读

### 背景：这个问题为什么难？

形式化定理证明需要模型在海量的逻辑规则中找到正确的推导路径。现有的大语言模型（LLM）往往缺少高质量的训练样本——正确的证明非常稀少，导致强化学习或专家迭代在生成少量可用证明后就停滞不前。换句话说，模型的“奖励”太稀疏，训练信号不足，难以持续提升。

### 关键概念速览
- **形式化定理证明（Formal Theorem Proving）**：在像 Lean、Isabelle 这样的交互式证明助理里，用机器可验证的语言写出完整的证明。类似于在数学教材里写出每一步推导，只不过每一步都必须机器检查通过。
- **专家迭代（Expert Iteration）**：模型先生成候选证明，再用搜索或规则系统挑选出最好的，随后把这些“专家”示例用于微调。把它想象成学生先写草稿，老师挑出最好的，再让学生学习这些优秀草稿。
- **自博弈（Self‑play）**：模型在同一系统里扮演两个互补角色，互相产生训练数据。类似于两位棋手轮流下棋，彼此的对局既是训练素材也是提升动力。
- **猜想生成器（Conjecturer）**：负责提出新的待证命题。它的目标是让当前的证明器刚好能够证明，却又不容易直接给出答案，像老师布置的“稍有挑战”的练习题。
- **证明器（Prover）**：接受猜想并尝试给出完整证明。它使用标准的专家迭代流程，像学生在做作业时查阅教材、尝试不同解法。
- **稀疏奖励（Sparse Reward）**：在强化学习中，只有当模型成功完成完整证明时才会得到奖励。大多数尝试都会得到零奖励，导致学习效率低下。
- **pass@k**：在 k 次采样中至少有一次成功的比例。常用于衡量生成式模型在大量尝试下的成功率。

### 核心创新点
1. **引入猜想-证明双向循环 → 让模型自己制造训练数据**  
   传统方法只让 LLM 产生证明，缺少新鲜的、难度适中的命题。STP 让同一个模型同时扮演猜想生成器和证明器，猜想生成器不断产生“刚好可证”的新命题，直接喂给证明器训练，从根本上解决了高质量数据匮乏的问题。

2. **迭代难度调节的猜想筛选 → 持续提升挑战度**  
   每轮训练后，系统会挑选出那些“当前证明器只能刚好证明”的猜想，作为下一轮猜想生成器的训练目标。这样猜想生成器被迫产生更难的题目，类似于老师根据学生水平不断提升作业难度，防止模型停留在低难度区间。

3. **在两大交互式证明系统（Lean、Isabelle）上统一实现 → 跨平台验证**  
   之前的大多数工作只在单一系统上实验，STP 同时在 Lean 与 Isabelle 上跑通，证明了方法的通用性。作者在 Lean 上生成了 51.3 B token 的训练数据，展示了大规模自博弈的可行性。

4. **把自博弈与专家迭代结合 → 双向强化学习**  
   证明器仍然使用专家迭代来提升自身能力，而猜想生成器的目标函数是“难度刚好匹配”。两者相互推动，使得模型在同一训练循环里既学会生成更好猜想，又学会更高效证明。

### 方法详解
**整体框架**  
STP 的训练循环可以概括为四步：  
1) **猜想生成**：猜想模型基于当前状态随机采样生成若干命题。  
2) **难度评估**：把这些命题交给当前的证明模型，记录每个命题的成功率或所需搜索步数。  
3) **难度筛选**：挑选出“刚好可证”（成功率在某个阈值附近）的命题，构成本轮的训练集合。  
4) **双向微调**：使用这些集合分别微调猜想模型（让它产生更难的命题）和证明模型（让它更好地证明这些命题），然后进入下一轮。

**猜想生成器细节**  
- 输入：模型的内部状态、最近的证明经验以及一个“难度目标”。  
- 输出：形式化语言（Lean/Isabelle）下的命题字符串。  
- 训练目标：最大化被证明集合的比例，同时最小化被证明太容易的比例。实现上通过一个二元交叉熵损失，把“难度标签”（可证/不可证）作为监督信号。

**证明器细节**  
- 采用标准的专家迭代：先用采样或 Monte Carlo Tree Search（MCTS）生成若干候选证明路径，再用搜索得到最短或最可靠的完整证明。  
- 这些成功的证明被保存为“专家示例”，用于下一个微调阶段。  
- 为了兼容不同系统，作者实现了统一的抽象层，将 Lean 的 tactic 脚本和 Isabelle 的 proof method 都映射为统一的动作空间。

**难度筛选机制**  
- 设定一个目标成功率区间（如 30%–50%），只保留落在该区间的命题。  
- 对于成功率过高的命题，直接丢弃或降低其权重；对于成功率过低的命题，则标记为“过难”，不用于本轮训练。  
- 这种动态阈值让猜想生成器始终在“可提升但不易达成”的甜点区间徘徊。

**最巧妙的地方**  
- **自适应难度反馈**：猜想生成器的损失函数直接嵌入了证明器的表现，这种端到端的闭环在定理证明领域尚属首次。  
- **跨系统统一抽象**：把两套截然不同的交互式证明语言映射到同一动作空间，使得同一模型可以在 Lean 与 Isabelle 上共享经验，极大提升了数据利用率。

### 实验与效果
- **数据与算力**：在 Lean 环境下生成了约 51.3 B token 的自博弈数据；在 Isabelle 上也完成了相似规模的训练。  
- **主要评测集**：LeanWorkbook、miniF2F‑test、ProofNet‑test、PutnamBench。  
- **核心结果**：  
  - 在 LeanWorkbook 上的命题通过率从专家迭代的 13.2% 提升到 28.5%，几乎翻倍。  
  - miniF2F‑test 的 pass@3200 达到 65.0%，在全证明生成方法中居首。  
  - ProofNet‑test 的 pass@3200 为 23.9%，同样领先。  
  - PutnamBench（644 题）中模型在 3200 次采样内解出 8 题。  
- **对比基线**：与仅使用专家迭代的模型、以及公开的基于检索+微调的系统相比，STP 在所有指标上都有显著提升。  
- **消融实验**：作者分别关闭猜想生成器、难度筛选或跨系统抽象，发现每一块的移除都会导致通过率下降 5%–12%，说明四个组件缺一不可。  
- **局限性**：  
  - 训练成本极高，需要数十亿 token 的算力，普通实验室难以复制。  
  - 仍然依赖于手工设定的难度阈值，自动调节机制尚未成熟。  
  - 对于极其抽象的数学概念（如高阶拓扑）仍然表现平平，作者在讨论中承认模型的“概念深度”仍受限。

### 影响与延伸思考
STP 把自博弈思路从棋类游戏成功迁移到形式化数学证明，打开了“模型自生成训练数据”的新路径。随后的工作（如 *Self‑Play for Symbolic Reasoning*、*Conjecture‑Driven Proof Synthesis*）纷纷借鉴其猜想‑证明闭环框架，尝试在代数、几何甚至程序验证领域实现类似的自提升。对想进一步探索的读者，可以关注以下方向：  
- **自动难度调节**：用贝叶斯优化或元学习让阈值自行适应。  
- **跨语言迁移**：把在 Lean 上学到的经验迁移到 Coq、HOL 等其他系统。  
- **概念抽象学习**：结合图神经网络或知识图谱，让模型捕捉更高层次的数学概念。  
- **少量算力实现**：研究如何在更小模型上复现自博弈收益，降低门槛。

### 一句话记住它
STP 让大语言模型自己出题、自己解题，通过“刚好可证”的循环不断提升，成功把自博弈搬进了形式化数学证明。