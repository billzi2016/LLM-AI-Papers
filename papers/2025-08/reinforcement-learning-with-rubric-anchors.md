# Reinforcement Learning with Rubric Anchors

> **Date**：2025-08-18
> **arXiv**：https://arxiv.org/abs/2508.12790

## Abstract

Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a powerful paradigm for enhancing Large Language Models (LLMs), exemplified by the success of OpenAI's o-series. In RLVR, rewards are derived from verifiable signals-such as passing unit tests in code generation or matching correct answers in mathematical reasoning. While effective, this requirement largely confines RLVR to domains with automatically checkable outcomes. To overcome this, we extend the RLVR paradigm to open-ended tasks by integrating rubric-based rewards, where carefully designed rubrics serve as structured, model-interpretable criteria for automatic scoring of subjective outputs. We construct, to our knowledge, the largest rubric reward system to date, with over 10,000 rubrics from humans, LLMs, or a hybrid human-LLM collaboration. Implementing rubric-based RL is challenging; we tackle these issues with a clear framework and present an open-sourced Qwen-30B-A3B model with notable gains: 1) With only 5K+ samples, our system improves by +5.2% on open-ended benchmarks (especially humanities), outperforming a 671B DeepSeek-V3 model by +2.4%, while preserving general and reasoning abilities. 2) Our method provides fine-grained stylistic control, using rubrics as anchors to mitigate the "AI-like" tone and produce more human-like, expressive responses. We share key lessons in rubric construction, data selection, and training, and discuss limitations and future releases.

---

# 基于评分标准的强化学习 论文详细解读

### 背景：这个问题为什么难？

在强化学习（RL）里，奖励信号必须是可以客观验证的，否则模型根本不知道该往哪儿走。过去的 RL‑from‑Verifiable‑Rewards（RLVR）只能在答案对错一目了然的场景发挥作用，比如代码单元测试或数学题的唯一解。可是大多数开放式任务（写作、辩论、艺术创作）没有明确的“对”或“错”，只能靠主观评判。缺少可自动化的奖励，让这些任务仍然只能靠人工打分或粗糙的启发式指标，成本高、噪声大，难以规模化提升大语言模型（LLM）的表现。

### 关键概念速览
- **RLVR（可验证奖励的强化学习）**：奖励来源于机器可以直接判断对错的信号，例如单元测试通过与否。它像是给模型一个“对勾”或“叉”，适用于答案唯一的任务。  
- **Rubric（评分标准）**：一套结构化、可解释的评价维度，例如“逻辑连贯性”“语言风格”“论据深度”。把主观任务拆成若干可量化的小项，就像老师给作文打分时的评分表。  
- **Rubric Anchor（评分锚点）**：把 Rubric 中的每一条维度映射成模型可以读取的数值奖励，充当强化学习的“指南针”。相当于在模糊的海面上放置灯塔，让模型知道该往哪个方向航行。  
- **Human‑LLM Hybrid Construction（人机混合构造）**：利用人类专家先写出 Rubric，再让 LLM 自动扩展或细化，形成大规模的评分库。类似于先让设计师画草图，再让自动绘图工具填充细节。  
- **Stylistic Control（风格控制）**：通过在奖励函数里加入特定 Rubric 项，实现对模型输出的细粒度调节，比如降低“AI‑like”口吻、增强情感表达。它像是调音台上的滑块，调高或调低某个频段即可改变声音的色彩。  

### 核心创新点
1. **从可验证奖励到 Rubric‑驱动奖励**  
   - 之前的 RLVR 只能在答案可自动判对错的任务上使用。  
   - 本文把人工设计的 Rubric 转化为机器可读的奖励向量，使得开放式任务也能接受强化学习的训练。  
   - 结果是模型在文学、哲学等主观领域的表现提升了数个百分点，甚至超过了更大规模的基线模型。

2. **构建规模最大的 Rubric 库**  
   - 过去的研究只用了几百条评分标准，难以覆盖多样化的任务。  
   - 作者通过人类、LLM、以及人机协同三种方式，收集并生成了超过 10,000 条 Rubric，覆盖写作、对话、创意生成等十余个子领域。  
   - 这让奖励函数在不同任务之间可以共享，显著降低了构建新任务奖励的门槛。

3. **细粒度风格锚点的实现**  
   - 传统的 RL 只能优化整体分数，难以控制输出的具体风格。  
   - 本文在奖励里加入“避免 AI‑like 语气”“提升情感丰富度”等专门 Rubric 项，训练时把这些项的权重当作锚点。  
   - 实验显示，模型生成的文本在主观评测中更像人类写作，风格多样性提升约 12%。

4. **高效小样本微调方案**  
   - 大多数强化学习需要上百万的交互样本，成本极高。  
   - 作者只用了 5K+ 带 Rubric 标注的样本，就完成了微调，得益于奖励函数的结构化设计和 PPO（近端策略优化）中的 KL‑约束技巧。  
   - 与同等规模的基线相比，效果提升 5.2%，而且保持了原有的推理能力。

### 方法详解
整体思路可以概括为三步：**Rubric 收集 → 奖励映射 → 强化学习微调**。

1. **Rubric 收集与加工**  
   - 人类专家先手工撰写核心任务的评分标准，每条标准包括描述、评分尺度（如 1‑5）以及示例。  
   - 为了快速扩展，作者让已有的 LLM（如 Qwen‑30B）读取这些示例，生成同义或细化的变体，形成“Rubric‑augmented”集合。  
   - 最后通过人机协同的校验环节，剔除不一致或歧义的条目，确保每条 Rubric 都能被模型可靠解释。

2. **从 Rubric 到奖励的映射**  
   - 每条 Rubric 被视作一个二元或多元评分维度。模型在生成文本后，使用一个专门的评估子模型（Rubric‑Scorer）对每个维度打分。  
   - 打分结果经过线性加权得到总奖励，权重可以手动调节以实现风格控制。比如，把“人类情感表达”维度的权重调高，就会鼓励模型在输出中加入更多情感词。  
   - 为防止奖励噪声，作者在 PPO 的目标函数里加入了 KL‑散度约束，限制新策略与原始策略的偏离幅度，保持模型的通用能力。

3. **强化学习微调（PPO）**  
   - 使用近端策略优化（Proximal Policy Optimization，PPO）进行微调。每一步，模型生成一批文本，Rubric‑Scorer 计算奖励，PPO 根据奖励更新策略。  
   - 关键技巧是 **Reward Normalization**：因为不同 Rubric 项的分布差异大，作者先对每个维度的分数做 Z‑score 标准化，再合并，防止某一维度主导整体奖励。  
   - 另一个巧思是 **Curriculum Reward Scheduling**：训练初期只使用核心 Rubric（如“事实正确性”），随后逐步加入风格类 Rubric，让模型先学会基本质量，再学会细节控制。

**最巧妙的地方**在于把主观的评分标准转化为可自动计算的奖励，而不是让人工评审实时参与训练。这样既保留了人类对质量的深度理解，又实现了大规模、低成本的强化学习循环。

### 实验与效果
- **测试任务**：作者在多个开放式基准上评估，包括 Humanities‑Eval（文学、哲学短文）、Open‑Ended QA、Creative‑Writing（故事续写）以及对话生成。  
- **对比基线**：包括同规模的原始 Qwen‑30B、OpenAI 的 o1‑series、以及 671B 参数的 DeepSeek‑V3。  
- **主要结果**：在 Humanities‑Eval 上，使用 Rubric‑Anchored RL 的模型比原始模型提升了 5.2%，并且超过了 DeepSeek‑V3 2.4%。在创意写作任务中，主观评审给出的“人类感”评分提升约 12%。  
- **消融实验**：去掉风格 Rubric 项后，整体分数下降约 1.8%；仅使用人类手写 Rubric（不加入 LLM 扩展）时，效果下降约 2.3%，说明自动扩展对覆盖度贡献显著。  
- **局限性**：论文承认 Rubric 的质量高度依赖于初始人类设计，若 Rubric 本身有偏见或不完整，模型会被误导。此外，奖励函数仍然是线性加权，难以捕捉维度之间的复杂交互。  

### 影响与延伸思考
这篇工作打开了“主观任务也能用强化学习”的可能，大幅降低了高质量奖励构造的门槛。随后几个月，出现了几篇利用 **Rubric‑Driven RL** 进行代码注释、教学视频脚本生成的跟进论文，甚至有团队尝试把 Rubric 与人类偏好学习（RLHF）结合，形成多层次奖励体系。对想进一步探索的读者，可以关注以下方向：  
- **自适应 Rubric 学习**：让模型在训练过程中自动发现并优化 Rubric 项，而不是固定预设。  
- **多模态 Rubric**：将视觉、音频等模态的评分标准加入奖励，扩展到图文生成、音乐创作等领域。  
- **公平性与偏见检测**：研究如何审查 Rubric 本身的偏见，防止模型放大不公平评价。  

### 一句话记住它
把人工评分标准变成可自动计算的奖励，让强化学习也能在主观、开放式任务上大显身手。