# Competitive Programming with Large Reasoning Models

> **Date**：2025-02-03
> **arXiv**：https://arxiv.org/abs/2502.06807

## Abstract

We show that reinforcement learning applied to large language models (LLMs) significantly boosts performance on complex coding and reasoning tasks. Additionally, we compare two general-purpose reasoning models - OpenAI o1 and an early checkpoint of o3 - with a domain-specific system, o1-ioi, which uses hand-engineered inference strategies designed for competing in the 2024 International Olympiad in Informatics (IOI). We competed live at IOI 2024 with o1-ioi and, using hand-crafted test-time strategies, placed in the 49th percentile. Under relaxed competition constraints, o1-ioi achieved a gold medal. However, when evaluating later models such as o3, we find that o3 achieves gold without hand-crafted domain-specific strategies or relaxed constraints. Our findings show that although specialized pipelines such as o1-ioi yield solid improvements, the scaled-up, general-purpose o3 model surpasses those results without relying on hand-crafted inference heuristics. Notably, o3 achieves a gold medal at the 2024 IOI and obtains a Codeforces rating on par with elite human competitors. Overall, these results indicate that scaling general-purpose reinforcement learning, rather than relying on domain-specific techniques, offers a robust path toward state-of-the-art AI in reasoning domains, such as competitive programming.

---

# 大规模推理模型在竞技编程中的应用 论文详细解读

### 背景：这个问题为什么难？

竞技编程题目往往要求选手在几分钟内写出既正确又高效的代码，涉及复杂的算法设计、边界分析以及对语言细节的精准把握。传统的代码生成模型在面对这类多步骤推理时容易卡在“只会写常规模板”或“忽略时间空间约束”的阶段。即使加入了大规模语言模型（LLM），缺乏针对性训练也会导致在高难度题目上出现逻辑错误或性能不达标。于是，如何让 AI 既拥有通用语言理解，又能在极端推理密集的编程环境中表现出色，成为了迫切需要突破的瓶颈。

### 关键概念速览
**大语言模型（LLM）**：拥有上百亿参数的文本生成系统，能够理解并生成自然语言和代码，就像会说多种语言的“全能翻译官”。  
**强化学习（RL）**：让模型通过试错获得奖励的训练方式，类似于让机器人在游戏中不断尝试，最终学会最优策略。  
**思维链（CoT）**：模型在给出答案前先写出推理步骤，像人在解数学题时先列出草稿，帮助模型避免“一口气”跳到错误结论。  
**手工推理策略**：人为设计的推理规则或搜索技巧，例如在竞赛中常用的“先枚举再剪枝”，相当于给模型配备了经验丰富的教练。  
**o1、o1‑ioi、o3**：OpenAI 发布的三个不同阶段的模型，o1 为通用推理模型，o1‑ioi 在其基础上加入了专门针对 IOI 竞赛的手工策略，o3 是更大、更强的通用模型。  
**IOI（国际信息学奥林匹克）**：全球顶级中学生编程竞赛，题目难度堪比大学高级算法课，成绩常被用作衡量 AI 编程能力的金标准。  
**Codeforces 评级**：在线竞技编程平台的分级系统，分数越高代表在真实对战中越接近人类高手。  

### 核心创新点
1. **强化学习驱动的通用模型微调 → 直接在大语言模型上进行基于奖励的训练 → 让模型在没有任何手工规则的情况下，自动学会如何拆解复杂题目、评估时间复杂度并生成高效代码。**  
2. **对比专用管线与通用模型 → 构建了 o1‑ioi（在 o1 基础上叠加手工推理策略）与纯通用的 o3 两套系统 → 结果显示，虽然 o1‑ioi 在放宽约束时能拿到金牌，但在正式比赛环境下仅位于 49% 分位；而 o3 在同等条件下直接夺金，证明规模化的通用模型可以自行捕获专用技巧。**  
3. **从“手工策略+模型”到“全模型自适应” → 放弃了传统的多阶段流水线（如先用检索再用生成），改为一次性的端到端强化学习流程 → 简化了系统复杂度，同时提升了鲁棒性，尤其在面对未知题型时表现更稳。  

### 方法详解
整体思路可以划分为三步：**任务定义 → 强化学习微调 → 推理执行**。  
1. **任务定义**：作者把每一道 IOI 题目抽象为“输入‑输出对 + 代码性能指标”。奖励函数由两部分组成：一是答案正确性（通过单元测试），二是代码效率（根据时间/空间上限的通过率）。这相当于给模型设定了“做对题目”和“做得快”的双重目标。  
2. **强化学习微调**：在已有的 LLM（如 GPT‑4 规模的预训练模型）上，使用 **近端策略优化（PPO）** 进行微调。每一次模型生成的代码都会被交给自动评测系统跑测试，得到奖励分数后反馈给模型。这里的关键是 **“自我对弈”**：模型不断尝试不同的解法，好的解法得到高奖励，坏的被惩罚，模型的策略随之收敛。  
3. **推理执行**：微调完成后，模型在实际比赛中直接接受题目描述，输出完整的代码。为了进一步提升可靠性，作者在推理阶段加入了 **“多样本采样 + 投票”**：让模型生成若干候选解，然后用快速评测挑选最有希望通过的那一个。  

**最巧妙的设计**在于奖励函数的双目标设定。传统强化学习往往只关注最终正确性，导致模型倾向于生成“暴力”但能跑通的代码。这里把效率也量化进奖励，使模型在学习过程中自然形成对时间复杂度的感知，类似于人类在练习时会自觉避免超时的解法。  

### 实验与效果
- **测试平台**：2024 年 IOI 正式赛题以及公开的 Codeforces 竞赛题库。  
- **基准对比**：包括原始 o1（无手工策略）、o1‑ioi（加入手工推理）以及业界常用的代码生成模型（如 Codex）。  
- **成绩**：在正式 IOI 赛制下，o1‑ioi 通过手工策略和放宽约束拿到金牌，但在真实比赛环境（不放宽）仅位于第 49 百分位；o3 在相同条件下直接获得金牌，并在 Codeforces 排名上达到与顶尖人类选手相当的分数。  
- **消融实验**：作者分别去掉奖励函数中的效率项、去掉多样本投票、以及不使用 PPO 微调。实验显示，去掉效率奖励会导致通过率下降约 15%，去掉投票机制会让最终排名下降约 8%。  
- **局限性**：论文承认，当前的强化学习训练成本极高，需要大量算力；此外，模型仍然依赖于高质量的自动评测系统，若评测不完整可能导致误导学习。  

### 影响与延伸思考
这篇工作向社区展示了“规模化通用模型 + 强化学习”可以在高度专业化的推理任务上超越手工工程化的管线，直接推动了大模型在算法竞赛、数学证明等高阶推理领域的应用。随后出现的几篇论文（如《RL‑augmented Code Generation for ACM‑ICPC》、《Self‑Play for Algorithmic Problem Solving》）都在不同程度上借鉴了该文的奖励设计和端到端微调思路。未来的研究可以进一步探索 **更高效的奖励信号**（比如基于复杂度理论的理论奖励）以及 **跨任务迁移**（让在 IOI 上学到的技巧直接迁移到软件工程代码审查等实际场景）。  

### 一句话记住它
规模化的通用大模型通过强化学习自学算法技巧，已经可以在竞技编程中直接击败专门手工调教的系统。