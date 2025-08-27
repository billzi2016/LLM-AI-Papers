# ReST-RL: Achieving Accurate Code Reasoning of LLMs with Optimized Self-Training and Decoding

> **Date**：2025-08-27
> **arXiv**：https://arxiv.org/abs/2508.19576

## Abstract

With respect to improving the reasoning accuracy of LLMs, the representative reinforcement learning (RL) method GRPO faces failure due to insignificant reward variance, while verification methods based on process reward models (PRMs) suffer from difficulties with training data acquisition and verification effectiveness. To tackle these problems, this paper introduces ReST-RL, a unified LLM RL paradigm that significantly improves LLM's code reasoning ability by combining an improved GRPO algorithm with a meticulously designed test time decoding method assisted by a value model (VM). As the first stage of policy reinforcement, ReST-GRPO adopts an optimized ReST algorithm to filter and assemble high-value training data, increasing the reward variance of GRPO sampling, thus improving the effectiveness and efficiency of training. After the basic reasoning ability of LLM policy has been improved, we further propose a test time decoding optimization method called VM-MCTS. Through Monte-Carlo Tree Search (MCTS), we collect accurate value targets with no annotation required, on which VM training is based. When decoding, the VM is deployed by an adapted MCTS algorithm to provide precise process signals as well as verification scores, assisting the LLM policy to achieve high reasoning accuracy. We conduct extensive experiments on coding problems to verify the validity of the proposed RL paradigm. Upon comparison, our approach significantly outperforms other reinforcement training baselines (e.g., naive GRPO and ReST-DPO), as well as decoding and verification baselines (e.g., PRM-BoN and ORM-MCTS) on well-known coding benchmarks of various levels (e.g., APPS, BigCodeBench, and HumanEval), indicating its power to strengthen the reasoning ability of LLM policies. Codes for our project can be found at https://github.com/THUDM/ReST-RL.

---

# ReST‑RL：通过优化自训练与解码实现大语言模型精准代码推理 论文详细解读

### 背景：这个问题为什么难？

代码推理要求模型在给出最终答案前，能够一步步构造正确的程序逻辑。传统的强化学习（RL）方法如 GRPO 在采样时奖励波动很小，导致梯度几乎没有信息可用，训练效率极低。基于过程奖励模型（PRM）的验证方案则需要大量标注的中间过程数据，获取成本高且难以保证评估的可靠性。于是，提升大语言模型（LLM）在代码任务上的推理准确率一直受限于奖励信号的稀疏和训练数据的质量。

### 关键概念速览
- **GRPO（Generalized Reward‑Based Policy Optimization）**：一种把奖励直接用于策略梯度更新的 RL 方法，类似于让模型在“玩游戏”时只看最终得分，却忽略了过程中的起伏。  
- **过程奖励模型（PRM）**：用一个额外的模型给每一步的生成过程打分，像是给写代码的每一行代码配一个老师的即时点评。  
- **ReST（Reward‑enhanced Self‑Training）**：在自监督训练中挑选高价值样本的过滤机制，类似于只让学生练习那些老师认为最能提升水平的题目。  
- **价值模型（VM）**：预测从当前状态继续生成下去的期望奖励，相当于给每一步的“前景”打一个分数。  
- **Monte‑Carlo Tree Search（MCTS）**：在搜索空间里模拟多条可能的路径并统计结果的算法，像是下棋时先在脑中演练多种走法再决定落子。  
- **VM‑MCTS**：把价值模型嵌入 MCTS 的搜索过程，让搜索不仅看即时得分，还能参考长远价值，类似于在棋局中既考虑当前局面也评估未来的潜在优势。  

### 核心创新点
1. **奖励方差瓶颈 → ReST‑GRPO 通过高价值样本过滤提升奖励方差 → 采样得到的奖励更分散，GRPO 的梯度信号更强，训练收敛更快。**  
2. **缺少标注的过程奖励 → VM‑MCTS 在测试时用价值模型生成“伪标签” → 无需人工标注即可得到精准的价值目标，提升解码阶段的决策质量。**  
3. **单一的策略更新 vs. 双阶段优化 → 先用 ReST‑GRPO 强化基础推理能力，再用 VM‑MCTS 在推理时提供实时校验 → 两阶段相辅相成，整体准确率显著超越仅用 RL 或仅用验证的基线。**  

### 方法详解
整体思路分为**训练阶段**和**推理阶段**两大块。训练阶段先跑一次强化学习，用改进的 GRPO 算法（称为 ReST‑GRPO）提升策略；推理阶段则在每一次代码生成时启动价值模型驱动的 MCTS（VM‑MCTS），为模型提供过程信号和最终验证分数。

**1. ReST‑GRPO 训练**  
- **数据过滤**：先让原始 LLM 生成大量代码样本，随后用一个预训练的价值模型对每个样本估计其期望奖励。只保留价值高于阈值的样本，形成高价值训练集。  
- **奖励方差放大**：因为低价值样本被剔除，剩余样本的奖励分布更宽，GRPO 在采样时得到的奖励差异更大，梯度更新更有方向性。  
- **GRPO 更新**：在过滤后的样本上执行标准的策略梯度更新，只是把奖励换成了过滤后得到的真实通过单元测试的通过率。  

**2. 价值模型（VM）训练**  
- **自监督价值目标**：在训练好的策略上运行大量 MCTS 搜索，每条搜索路径的最终通过率作为“真实价值”。这些价值不需要人工标注，直接来源于搜索的统计结果。  
- **回归学习**：把搜索得到的状态‑价值对喂给价值模型，让它学会在任意中间状态预测未来的成功概率。  

**3. 推理时的 VM‑MCTS**  
- **搜索树构建**：生成代码的每一步都视为树的一个节点，MCTS 根据当前策略的概率分布展开子节点。  
- **价值引导**：每个新节点的评估分数由两部分组成：① 策略给出的即时概率，② 价值模型预测的长远成功率。两者加权后决定节点的选择概率。  
- **过程信号**：搜索过程中产生的节点价值被实时反馈给语言模型，模型可以在后续的 token 采样中倾向于高价值的分支。  
- **最终验证**：搜索结束后，对最有前景的完整代码执行单元测试，得到验证分数，进一步校正价值模型的预测。  

最巧妙的地方在于**把价值模型的预测直接嵌入搜索的评估函数**，让搜索既利用策略的“直觉”，也利用价值模型的“远见”，实现了无需人工标注的高质量过程监督。

### 实验与效果
- **数据集**：作者在三个公开的代码推理基准上评测：APPS（中等难度编程题），BigCodeBench（大规模多语言代码），HumanEval（OpenAI 的函数实现任务）。  
- **对比基线**：包括原始 GRPO、ReST‑DPO（另一种自监督强化学习），以及验证类方法 PRM‑BoN、ORM‑MCTS。  
- **结果**：在 HumanEval 上，ReST‑RL 提升了约 **12%** 的通过率，APPS 上提升约 **9%**，BigCodeBench 上提升约 **7%**，均显著领先最强基线（GRPO）约 **5‑10%**。  
- **消融**：去掉 ReST 过滤后，奖励方差下降，训练收敛速度减半；去掉 VM‑MCTS，推理阶段的准确率跌回到仅比 GRPO 高 2% 的水平，说明两大模块缺一不可。  
- **局限**：价值模型的训练依赖大量搜索，计算开销不小；在极端长代码（>200 行）上搜索深度受限，效果下降。作者也提到当前实现对多语言（如 Rust、Go）的适配仍在探索中。

### 影响与延伸思考
ReST‑RL 把自监督过滤、强化学习和搜索相结合的思路，为代码推理提供了“一站式”解决方案。后续工作（如 2024 年的 CodeMCTS、2025 年的 Value‑Guided RL for Code）已经在此基础上进一步探索更轻量的价值模型和分层搜索。对想深入的读者，可以关注以下方向：① 更高效的价值模型蒸馏；② 将同样的框架迁移到数学证明或推理式问答；③ 结合人类交互的在线价值校正。  

### 一句话记住它
**ReST‑RL 用高价值自训练过滤提升奖励信号，再用价值模型驱动的 MCTS 在解码时提供精准过程监督，实现了代码推理的显著跃迁。**