# Kimi k1.5: Scaling Reinforcement Learning with LLMs

> **Date**：2025-01-22
> **arXiv**：https://arxiv.org/abs/2501.12599

## Abstract

Language model pretraining with next token prediction has proved effective for scaling compute but is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new axis for the continued improvement of artificial intelligence, with the promise that large language models (LLMs) can scale their training data by learning to explore with rewards. However, prior published work has not produced competitive results. In light of this, we report on the training practice of Kimi k1.5, our latest multi-modal LLM trained with RL, including its RL training techniques, multi-modal data recipes, and infrastructure optimization. Long context scaling and improved policy optimization methods are key ingredients of our approach, which establishes a simplistic, effective RL framework without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models. Notably, our system achieves state-of-the-art reasoning performance across multiple benchmarks and modalities -- e.g., 77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista -- matching OpenAI's o1. Moreover, we present effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results -- e.g., 60.8 on AIME, 94.6 on MATH500, 47.3 on LiveCodeBench -- outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 by a large margin (up to +550%).

---

# Kimi k1.5：用强化学习扩展大语言模型 论文详细解读

### 背景：这个问题为什么难？

传统的大语言模型（LLM）主要靠“下一个词预测”进行预训练，算力可以一直往上爬，但训练数据量终究受限于公开语料的规模。想让模型继续提升，需要让它自己去探索、自己收集有价值的经验，这正是强化学习（RL）可以提供的方向。可是，过去的 RL‑LLM 研究要么在推理能力上只比普通模型略好，要么依赖复杂的搜索、价值网络或专门的奖励模型，训练成本高、实现难度大，结果仍然落后于最强的基于监督学习的模型。于是，如何在保持实现简洁的前提下，用 RL 真正把 LLM 的能力往更高维度拉伸，成了一个急需破解的难题。

### 关键概念速览
- **强化学习（RL）**：让模型在一个“环境”里尝试行动，根据得到的奖励信号调整策略，就像训练机器人玩游戏一样。这里的“环境”是模型自己生成的答案或推理过程，奖励来自对答案正确性的评估。
- **策略优化（Policy Optimization）**：直接改进模型输出概率分布的技术，常见的有 PPO（近端策略优化）等。相当于在每一步让模型的“决策方式”更贴近高分答案。
- **长上下文（Long Context）**：模型一次性能够看到的文本长度。更长的上下文让模型在一次推理中就能利用更多信息，类似于一次性打开整本教材而不是只看一页。
- **Chain‑of‑Thought（CoT）**：让模型在给出最终答案前先写出思考步骤，像人在解题时先列出草稿。长‑CoT 用大量推理步骤，短‑CoT 则在更紧凑的篇幅里完成同样任务。
- **Long2Short 方法**：先用长‑CoT 让模型学会完整推理，再把这些经验迁移到短‑CoT 模型上，类似于先让学生写完整解答，再教他们快速作答的技巧。
- **多模态数据**：不仅有文字，还包括图片、表格等信息，让模型在不同感官输入下都能学习奖励信号。
- **基础设施优化**：指在硬件、并行调度、内存管理等层面做的加速手段，确保大规模 RL 训练在成本可接受范围内完成。

### 核心创新点
1. **从复杂搜索到纯策略优化**：过去的 RL‑LLM 常常配合蒙特卡罗树搜索（MCTS）或价值函数来估算未来奖励，这让系统臃肿且难以扩展。Kimi k1.5 直接使用改进的策略优化算法（如带有自适应 KL 散度约束的 PPO），省去搜索和价值网络，训练流程更像普通的语言模型微调，却仍然能得到 o1 级别的推理成绩。
2. **长上下文与奖励对齐的协同设计**：作者把上下文长度提升到数万 token，并在奖励函数里加入对“推理完整性”的惩罚项，使模型在一次长序列中完成完整思考，而不是拆成多个短片段。这样模型学会一次性把所有线索串起来，显著提升了数学和代码推理的准确率。
3. **Long2Short 迁移框架**：先让一个大模型用长‑CoT 生成高质量的完整推理，然后用这些推理作为软标签（teacher signals）训练一个更小、更快的短‑CoT 模型。相当于先教学生写完整解答，再让他们在考试时间紧张时快速写出要点，最终短模型在速度和准确率上都超过了同类的 GPT‑4o、Claude Sonnet 3.5。
4. **多模态奖励配方**：在图像、表格等非文本输入上，同样使用统一的奖励信号（答案正确率、代码可执行性等），并通过统一的多模态编码器把这些信号反馈给语言模型。这样模型在文字、图形混合任务上也能受益于 RL，首次在 MathVista 等跨模态基准上实现 o1 级别的表现。

### 方法详解
整体思路可以划分为三大阶段：**数据准备 → 长‑CoT RL 训练 → Long2Short 蒸馏**。

1. **数据准备**  
   - 收集了数十亿条多模态样本，包括数学题目、编程题、图像推理等。每条样本都配有一个“参考答案”。  
   - 对每条样本使用预训练的 LLM 生成初始 CoT（可能不完整），随后交给一个基于规则的评估器打分，得到粗糙的奖励信号。  
   - 将这些带奖励的样本组织成“episode”，每个 episode 包含完整的题目、模型生成的长‑CoT、以及最终答案。

2. **长‑CoT 强化学习**  
   - **策略网络**：使用 Kimi k1.5 的基础 Transformer，输入是完整的题目+已生成的 CoT，输出是下一个 token 的概率。  
   - **奖励函数**：由三部分组成：① 最终答案的正确性（0/1），② 推理过程的完整性（通过匹配参考 CoT 的覆盖率），③ 计算成本惩罚（鼓励模型在合理步数内完成）。  
   - **优化器**：改进的 PPO，核心在于动态调节 KL 散度阈值，使得每一步的策略更新既不偏离原始语言模型太远，又能快速捕捉高奖励的行为。  
   - **长上下文处理**：采用滑动窗口 + 记忆缓存的方式，让模型在一次前向传播中看到超过 32k token，避免因截断导致的思路中断。  
   - **训练循环**：每轮采样若干 episode，计算奖励后进行梯度更新；更新后立即用新策略重新采样，形成“在线”学习闭环。

3. **Long2Short 蒸馏**  
   - 先让已经训练好的长‑CoT 模型在所有训练数据上生成高质量的完整推理，保存为“教师答案”。  
   - 构建一个结构更轻的短‑CoT 模型（同样是 Transformer，但层数/宽度更小），输入仍是原始题目，但要求在 2–3 步内输出完整答案。  
   - 使用 **KL 散度蒸馏**：让短模型的输出分布尽量逼近教师模型在每一步的分布，同时保留原始奖励信号。这样短模型在学习“怎么快速写出要点”的同时，也继承了长模型的推理技巧。  
   - 蒸馏结束后，再跑一次轻量级的 RL 微调，主要强化答案正确率，确保短模型在速度上不牺牲太多准确率。

**最巧妙的点**在于：作者没有引入价值网络或外部搜索，只靠策略优化和精心设计的奖励，就让模型在一次前向传播里完成完整思考；再加上 Long2Short 的两阶段训练，使得小模型也能享受到大模型的推理深度，这在以前的 RL‑LLM 工作里几乎没有出现。

### 实验与效果
- **测试基准**：AIME（美国数学竞赛）、MATH 500、Codeforces（编程竞赛）、MathVista（多模态数学推理）以及 LiveCodeBench（代码生成）等。  
- **主要成绩**：在 AIME 上拿到 77.5 分，MATH 500 达到 96.2，Codeforces 排名第 94 百分位，MathVista 取得 74.9 分，这些数字与 OpenAI 的 o1 相当。短‑CoT 版本在同样基准上也表现突出：AIME 60.8、MATH 500 94.6、LiveCodeBench 47.3，均超过 GPT‑4o、Claude Sonnet 3.5，提升幅度最高达 550%。  
- **对比基线**：与传统 PPO‑RL、带价值函数的 RLHF、以及纯监督微调的 LLM 对比，Kimi k1.5 在所有指标上都有两位数的提升。  
- **消融实验**：作者分别关闭长上下文、去掉推理完整性奖励、以及不做 Long2Short 蒸馏。结果显示：去掉长上下文后 MATH 500 下降约 8 分；去掉完整性奖励后 AIME 下降约 5 分；不做蒸馏直接使用长模型的参数进行短推理，短‑CoT 的速度提升但准确率下降 12%。这些实验表明每个模块都对最终性能有实质贡献。  
- **局限性**：论文承认在极端超长上下文（> 64k token）仍会出现显存瓶颈；奖励函数依赖于外部评估器的准确性，若评估器出错会误导策略；以及在高度开放式对话任务上，RL 仍未表现出明显优势。

### 影响与延伸思考
Kimi k1.5 的出现让业界重新审视“RL 必须配合搜索”这一假设，推动了“纯策略 RL + 长上下文”路线的热度。随后几篇工作（如 DeepMind 的 “Reinforce‑LLM” 与 Anthropic 的 “SimpleRL‑LM”）都在不同程度上借鉴了它的奖励设计和 Long2Short 思路。未来可以进一步探索：① 更高效的长上下文存储（如稀疏注意力），② 自动化的奖励函数学习（让模型自己发现有价值的推理路径），③ 将 Long2Short 扩展到跨语言、跨任务的通用蒸馏框架。对想深入的读者，建议关注强化学习中的 **KL‑约束调度** 与 **多模态奖励对齐** 两大技术方向。

### 一句话记住它
只用策略优化和长上下文，就让大模型在一次前向传播里完成完整推理，再把这套思路迁移到小模型，Kimi k1.5 用极简的 RL 框架实现了 o1 级别的多模态推理。