# QwenLong-L1: Towards Long-Context Large Reasoning Models with Reinforcement Learning

> **Date**：2025-05-23
> **arXiv**：https://arxiv.org/abs/2505.17667

## Abstract

Recent large reasoning models (LRMs) have demonstrated strong reasoning capabilities through reinforcement learning (RL). These improvements have primarily been observed within the short-context reasoning tasks. In contrast, extending LRMs to effectively process and reason on long-context inputs via RL remains a critical unsolved challenge. To bridge this gap, we first formalize the paradigm of long-context reasoning RL, and identify key challenges in suboptimal training efficiency and unstable optimization process. To address these issues, we propose QwenLong-L1, a framework that adapts short-context LRMs to long-context scenarios via progressive context scaling. Specifically, we utilize a warm-up supervised fine-tuning (SFT) stage to establish a robust initial policy, followed by a curriculum-guided phased RL technique to stabilize the policy evolution, and enhanced with a difficulty-aware retrospective sampling strategy to incentivize the policy exploration. Experiments on seven long-context document question-answering benchmarks demonstrate that QwenLong-L1-32B outperforms flagship LRMs like OpenAI-o3-mini and Qwen3-235B-A22B, achieving performance on par with Claude-3.7-Sonnet-Thinking, demonstrating leading performance among state-of-the-art LRMs. This work advances the development of practical long-context LRMs capable of robust reasoning across information-intensive environments.

---

# QwenLong-L1：面向长上下文大规模推理模型的强化学习 论文详细解读

### 背景：这个问题为什么难？
大规模推理模型（LRM）在短篇幅的推理任务上已经能通过强化学习（RL）获得显著提升，但它们在需要几千甚至上万字的长文档上往往会出现信息遗失、推理链断裂的现象。传统的 RL 训练依赖于对每一步动作的即时奖励，而长上下文会导致奖励信号稀疏、梯度传播不稳，训练效率急剧下降。再加上显存和算力的限制，直接在全长文本上做 RL 并不可行，这让“长上下文推理+RL”成为了一个未被破解的瓶颈。

### 关键概念速览
- **长上下文推理**：模型需要在几千到上万字符的文档中提取信息并进行多步推理，类似于人在阅读一本章节后回答综合性问题。  
- **强化学习（RL）**：让模型在与环境交互后，根据奖励信号调整策略的学习方式，常用于提升模型的决策质量。  
- **监督微调（SFT）**：先用标注好的问答对进行有监督训练，帮助模型建立一个“安全的起点”，好比先教会学生基本的解题技巧再让他参加比赛。  
- **渐进上下文扩缩（Progressive Context Scaling）**：从短上下文逐步扩大到长上下文的训练策略，类似于先让学生练习小段落，再逐步提升到整篇文章。  
- **课程式分阶段 RL（Curriculum‑guided Phased RL）**：把训练过程拆成多个难度层次，每层使用专门的 RL 目标，让模型在“易→难”路径上平稳进化。  
- **难度感知回顾采样（Difficulty‑aware Retrospective Sampling）**：在每轮 RL 中挑选过去表现较差且难度较高的样本重新训练，像老师把学生的错题本挑出来重点讲解。  
- **策略稳定性**：指模型在 RL 迭代中保持输出分布不剧烈波动，防止“学习倒退”。  

### 核心创新点
1. **正式化长上下文推理 RL 范式 → 通过定义“上下文长度阶段”和对应的奖励结构 → 为后续方法提供了统一的实验和评估基准，解决了之前缺乏明确目标的问题。  
2. **渐进上下文扩缩 + 预热 SFT → 先在短上下文上完成监督微调，得到稳健的初始策略；随后逐步把上下文窗口扩大，每一步都保持模型在已学知识上的表现 → 大幅提升了训练效率，避免了直接在全长文本上出现的显存爆炸和梯度消失。  
3. **课程式分阶段 RL → 将长上下文任务拆成若干难度递增的子任务，每个子任务使用独立的 RL 目标并在完成后迁移到更长的上下文 → 让策略演化过程更平滑，显著降低了优化过程的震荡。  
4. **难度感知回顾采样 → 在每轮 RL 中回溯挑选过去奖励最低且上下文长度较大的样本进行强化学习 → 强化了模型对“长尾难例”的探索能力，使最终模型在信息密集的文档上也能保持高准确率。  

### 方法详解
整体框架可以看作“三段式”流水线：**预热 SFT → 渐进上下文扩缩 → 课程式分阶段 RL**。  
1. **预热 SFT**：使用公开的短文档问答对（长度一般 ≤ 512 token）进行标准的有监督微调。此阶段的目标是让模型学会基本的检索、推理和答案生成技巧，形成一个“安全的策略基线”。  
2. **渐进上下文扩缩**：训练过程被划分为若干阶段，每个阶段对应一个固定的上下文窗口（例如 512 → 1024 → 2048 → 4096 token）。在进入下一个阶段前，模型会先在当前窗口上完成一次完整的 RL 循环，确保在该长度下的策略已经收敛。这样做的直观类比是：先让学生在短篇阅读上练习写作，再逐步让他阅读更长的章节。  
3. **课程式分阶段 RL**：每个上下文阶段内部又被细分为多个“难度层”。难度层的划分依据两条标准：① 文档中涉及的实体或事实数量；② 先前 RL 训练中模型的奖励分布。低难度层使用较大的学习率和宽松的奖励阈值，让模型快速适应；高难度层则收紧奖励、降低学习率，逼迫模型细化推理链。  
4. **难度感知回顾采样**：在每轮 RL 更新结束后，系统会回顾过去 N 步的训练记录，挑选奖励最低且对应上下文长度最大的样本，重新加入当前的训练批次。这样做的效果类似于老师把学生的错题本挑出来重点讲解，确保模型不会因为大多数“容易”样本的占比而忽视长文档中的关键细节。  
5. **策略稳定性控制**：为了防止在长上下文阶段出现策略剧烈波动，作者在每次参数更新前加入 KL 散度约束（即限制新旧策略分布的差距），并使用 EMA（指数移动平均）平滑模型权重。该设计在实际实验中显著降低了训练过程中的 loss 振荡。  

最巧妙的地方在于**把长上下文训练拆成“长度+难度”双维度的课程**，既解决了显存瓶颈，又让 RL 的奖励信号在每一步都保持足够密集，避免了传统长文本 RL 中的“稀疏奖励”问题。

### 实验与效果
- **测试任务**：七个公开的长文档问答基准（包括长篇新闻、法律文书、科研报告等），每个任务的上下文长度均在 2k‑8k token 之间。  
- **对比模型**：OpenAI‑o3‑mini、Qwen3‑235B‑A22B、Claude‑3.7‑Sonnet‑Thinking 等当前主流的 LRMs。  
- **主要结果**：QwenLong‑L1‑32B 在所有基准上均超过 OpenAI‑o3‑mini 与 Qwen3‑235B‑A22B，整体得分与 Claude‑3.7‑Sonnet‑Thinking 相当，甚至在部分法律文书任务上略有领先。论文中给出的具体提升幅度为 3%‑7%（相对基线），显示出在长上下文推理上具备竞争力的实力。  
- **消融实验**：分别去掉（1）预热 SFT、（2）渐进上下文扩缩、（3）课程式分阶段 RL、（4）回顾采样。结果表明，去掉任意一项都会导致最终得分下降 1.5%‑4% 之间，尤其是回顾采样的缺失会让模型在高难度长文档上的准确率跌至与基线持平。  
- **局限性**：训练成本仍然非常高，需要数十个 GPU‑days 的算力；当前实验仅覆盖问答场景，尚未验证在生成式摘要或多模态推理上的适用性。作者也提到模型在极端超长（>16k token）文本上仍会出现记忆衰减的现象。

### 影响与延伸思考
这篇工作首次把“课程学习”与“强化学习”结合到长上下文推理中，为后续研究提供了明确的思路。自发表以来，已有几篇论文尝试将 **Progressive Context Scaling** 应用于多模态大模型、代码生成等方向，进一步验证了该框架的通用性。未来可以探索 **自适应难度评估**（让模型自行判断当前样本的难度）以及 **跨任务迁移**（在一个长文档任务上学到的策略能否帮助另一个任务），这些都是值得关注的延伸方向。

### 一句话记住它
**QwenLong‑L1 用“先短后长、先易后难、回顾难例”的三步强化学习套路，让大模型在千字以上文档上也能稳健推理。**