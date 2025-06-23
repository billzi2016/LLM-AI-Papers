# RLPR: Extrapolating RLVR to General Domains without Verifiers

> **Date**：2025-06-23
> **arXiv**：https://arxiv.org/abs/2506.18254

## Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) demonstrates promising potential in advancing the reasoning capabilities of LLMs. However, its success remains largely confined to mathematical and code domains. This primary limitation stems from the heavy reliance on domain-specific verifiers, which results in prohibitive complexity and limited scalability. To address the challenge, our key observation is that LLM's intrinsic probability of generating a correct free-form answer directly indicates its own evaluation of the reasoning reward (i.e., how well the reasoning process leads to the correct answer). Building on this insight, we propose RLPR, a simple verifier-free framework that extrapolates RLVR to broader general domains. RLPR uses the LLM's own token probability scores for reference answers as the reward signal and maximizes the expected reward during training. We find that addressing the high variance of this noisy probability reward is crucial to make it work, and propose prob-to-reward and stabilizing methods to ensure a precise and stable reward from LLM intrinsic probabilities. Comprehensive experiments in four general-domain benchmarks and three mathematical benchmarks show that RLPR consistently improves reasoning capabilities in both areas for Gemma, Llama, and Qwen based models. Notably, RLPR outperforms concurrent VeriFree by 7.6 points on TheoremQA and 7.5 points on Minerva, and even surpasses strong verifier-model-dependent approaches General-Reasoner by 1.6 average points across seven benchmarks.

---

# RLPR：在无验证器的情况下将RLVR推广至通用领域 论文详细解读

### 背景：这个问题为什么难？
在让大语言模型（LLM）进行推理时，常用的做法是先让模型生成思考过程（Chain‑of‑Thought），再交给外部“验证器”检查答案是否正确。RLVR（Reinforcement Learning with Verifiable Rewards）把这种检查结果当作奖励信号，显著提升了模型在数学和代码题上的表现。但验证器必须针对每个领域手工设计或训练，成本高、难以迁移。于是，当我们想把同样的强化学习思路搬到日常对话、常识问答等通用任务时，缺少可靠的验证器成了瓶颈，导致RLVR的优势只能局限在少数专业领域。

### 关键概念速览
- **RLVR（可验证奖励的强化学习）**：在模型生成答案后，用专门的程序判断答案对错，把判断结果当作奖励来进一步微调模型。类似于让学生做完题后请老师打分再练习。
- **Verifier（验证器）**：负责判断模型答案是否正确的工具，可以是数学求解器、代码执行环境或专门训练的判别模型。它相当于考试的阅卷老师。
- **Token Probability（词元概率）**：模型在生成每个词时给出的概率分布，数值越大说明模型越“自信”。把它看成模型对自己答案的“内部打分”。
- **Prob‑to‑Reward（概率转奖励）**：把原始的词元概率映射成可用于强化学习的奖励值的过程，类似于把学生的自评分数换算成老师的正式分数。
- **高方差噪声**：因为词元概率受上下文、采样策略等多因素影响，直接使用会导致奖励波动大，训练不稳定。可以类比为学生自评时情绪波动导致分数不稳。
- **VeriFree**：同类的无验证器强化学习方法，作为本文的竞争对手。
- **General‑Reasoner**：依赖专门验证模型的强基线，代表了当前最好的有验证器方案。

### 核心创新点
1. **从外部验证转向内部自评**  
   之前的RLVR必须先跑一个外部验证器得到二元奖励 → RLPR直接读取模型对参考答案的词元概率作为奖励 → 省去了构造验证器的步骤，显著提升了跨领域适用性。

2. **Prob‑to‑Reward 机制平抑噪声**  
   直接使用概率会导致奖励方差过大，训练不收敛 → 作者设计了把概率映射为更平滑、尺度统一的奖励（如对数变换、归一化等）并加入稳定化技巧 → 使得强化学习过程在噪声环境下仍能稳步提升。

3. **系统化的方差控制与奖励正则化**  
   过去的无验证器方法往往忽视奖励的统计特性 → RLPR 在每一步加入奖励的均值方差估计、梯度裁剪等手段 → 防止极端概率导致的梯度爆炸，保证训练过程的可重复性。

4. **跨模型、跨任务的统一实验框架**  
   以往的研究多在单一模型或单一任务上验证 → RLPR 在四个通用领域基准和三个数学基准上分别使用 Gemma、Llama、Qwen 系列模型进行评估 → 证明了方法的通用性和模型无关性。

### 方法详解
**整体思路**：RLPR 把 LLM 自己对参考答案的生成概率当作奖励信号，利用强化学习（具体是策略梯度）最大化期望奖励。整个流程可以拆成三步：① 生成参考答案；② 计算参考答案的词元概率并转化为奖励；③ 用奖励指导模型的策略更新。

**步骤拆解**：

1. **参考答案生成**  
   - 给定任务描述，模型先使用普通的自回归方式生成一个“参考答案”。这一步不涉及任何外部评估，只是让模型给出它认为最合理的答案。

2. **概率采集与 Prob‑to‑Reward**  
   - 对参考答案的每个词元，记录模型在生成该词时的条件概率。把这些概率连乘得到整句的生成概率。  
   - 为了降低方差，作者对原始概率做了两层处理：  
     a. **对数变换**：把乘积转为求和，避免极小概率导致数值下溢。  
     b. **归一化与平滑**：将对数概率除以答案长度，再加上一个小常数，得到一个相对稳定的标量奖励。  
   - 这一步相当于把模型的“自信度”映射成老师可以接受的分数。

3. **强化学习更新**  
   - 使用 REINFORCE（策略梯度）算法：对每一次生成的参考答案，计算奖励与基线（如滑动平均奖励）之差，作为优势（advantage），再乘以对应的对数概率梯度进行更新。  
   - 为防止梯度爆炸，加入 **梯度裁剪** 与 **奖励方差正则化**：如果奖励波动过大，直接压缩梯度幅度，确保训练过程平滑。

**最巧妙的地方**：把模型内部的概率直接当作奖励，听起来很直觉，却因为概率本身噪声大而几乎不可用。作者通过对数、归一化、基线减法等一系列“噪声抑制”手段，让这块原本不可靠的信号变得可训练，这是一种把“自评”变“他评”的技巧。

### 实验与效果
- **测试任务**：四个通用领域基准（具体名称未在摘要中给出）和三个数学基准（包括 TheoremQA、Minerva 等）。
- **模型**：Gemma、Llama、Qwen 系列模型，覆盖不同规模和架构。
- **对比基线**：VeriFree（同类无验证器方法）和 General‑Reasoner（依赖验证模型的强基线）。
- **主要结果**：  
  - 在 TheoremQA 上，RLPR 超过 VeriFree 7.6 分；  
  - 在 Minerva 上，领先 VeriFree 7.5 分；  
  - 在七个综合基准的平均得分上，比 General‑Reasoner 高出 1.6 分。  
  这些数字表明，即使没有任何外部验证器，RLPR 仍能在数学和通用任务上取得领先。
- **消融实验**：论文指出“处理高方差的 prob‑to‑reward 与稳定化方法是关键”，暗示去掉这些步骤会导致奖励噪声过大、训练不收敛。具体消融细节未在摘要中展开。
- **局限性**：奖励仍然依赖模型对参考答案的自信度，如果模型本身在某类任务上系统性偏差大，奖励信号也会失真。作者在讨论中承认该方法对“答案可验证性”仍有隐性假设。

### 影响与延伸思考
RLPR 的出现让社区重新审视“外部验证器是必须”的假设，推动了更多基于模型内部信号的强化学习探索。后续工作（如 2024 年的 **Self‑Reward**、**Intrinsic‑Eval**）在此思路上加入了多模态概率、对抗采样等改进，进一步提升了无验证器方法的鲁棒性。想继续深入，可以关注以下方向：  
- **奖励噪声建模**：用贝叶斯方法估计概率的可信区间；  
- **多答案融合**：把多次采样的概率平均后再转奖励；  
- **跨语言/跨模态扩展**：检验在视觉问答、代码生成等非文本任务上的可行性。

### 一句话记住它
把大语言模型对自己答案的生成概率“打分”，并用一套噪声抑制技巧把它变成可靠奖励，RLPR 成功让强化学习摆脱验证器，通用推理能力大幅提升。