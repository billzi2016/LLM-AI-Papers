# Direct Nash Optimization: Teaching Language Models to Self-Improve with   General Preferences

> **Date**：2024-04-04
> **arXiv**：https://arxiv.org/abs/2404.03715

## Abstract

This paper studies post-training large language models (LLMs) using preference feedback from a powerful oracle to help a model iteratively improve over itself. The typical approach for post-training LLMs involves Reinforcement Learning from Human Feedback (RLHF), which traditionally separates reward learning and subsequent policy optimization. However, such a reward maximization approach is limited by the nature of "point-wise" rewards (such as Bradley-Terry model), which fails to express complex intransitive or cyclic preference relations. While advances on RLHF show reward learning and policy optimization can be merged into a single contrastive objective for stability, they yet still remain tethered to the reward maximization framework. Recently, a new wave of research sidesteps the reward maximization presumptions in favor of directly optimizing over "pair-wise" or general preferences. In this paper, we introduce Direct Nash Optimization (DNO), a provable and scalable algorithm that marries the simplicity and stability of contrastive learning with theoretical generality from optimizing general preferences. Because DNO is a batched on-policy algorithm using a regression-based objective, its implementation is straightforward and efficient. Moreover, DNO enjoys monotonic improvement across iterations that help it improve even over a strong teacher (such as GPT-4). In our experiments, a resulting 7B parameter Orca-2.5 model aligned by DNO achieves the state-of-the-art win-rate against GPT-4-Turbo of 33% on AlpacaEval 2.0 (even after controlling for response length), an absolute gain of 26% (7% to 33%) over the initializing model. It outperforms models with far more parameters, including Mistral Large, Self-Rewarding LM (70B parameters), and older versions of GPT-4.

---

# 直接纳什优化：利用通用偏好教语言模型自我提升 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）训练完毕后，想让它们更符合人类价值观和使用需求，业界普遍采用“人类反馈强化学习”（RLHF）。传统的 RLHF 把两件事拆开：先学一个奖励模型（把人类偏好映射成数值），再用强化学习最大化这个奖励。这个流程有两个根本缺陷：一是奖励只能给出“点对点”的分数，难以捕捉偏好之间的循环或不传递性（比如 A > B、B > C、C > A）；二是奖励最大化把优化目标硬性限定在一个标量上，导致训练不稳定、容易出现模式崩塌。虽然最近的工作把奖励学习和策略优化合并成对比式目标，提升了稳定性，但仍然受限于“最大化奖励”这一假设。于是出现了直接在“偏好对”上做优化的思路，但缺少既稳健又有理论保障的实现方案。

### 关键概念速览

**RLHF（人类反馈强化学习）**：先用人类标注的比较数据训练一个奖励模型，再用强化学习让语言模型的输出分数更高。像先给学生打分，再让学生针对高分答案练习。

**点对点奖励（point‑wise reward）**：每条生成文本对应一个标量分数，类似给每道题打分。无法表达“这两条答案互相比较更好”之外的关系。

**对比学习（contrastive learning）**：把好答案和坏答案放在一起，让模型学会把好答案的概率拉高、坏答案的概率压低。像让模型在两张卡片中挑出更大的那张。

**纳什均衡（Nash equilibrium）**：在两人零和游戏里，双方都没有动机单方面改变策略。把模型自身和“老师”模型视作两位玩家，找到一种互不改进的平衡点。

**直接纳什优化（Direct Nash Optimization，DNO）**：一种把 RLHF 重新表述为两玩家博弈的算法，直接在偏好对上做回归，既保持对比学习的简洁，又拥有纳什均衡的理论保证。

**批量 on‑policy**：模型在每轮迭代中使用当前策略生成的样本来更新自己，像在跑步时每一步都根据刚跑完的步伐调整姿势。

**单调改进（monotonic improvement）**：每一次参数更新都保证模型的整体表现不下降，类似每次练习都确保成绩不低于上一次。

### 核心创新点

1. **从奖励最大化到直接偏好优化**  
   *之前的做法*：先训练奖励模型，再最大化该奖励，受点对点奖励限制。  
   *本文的做法*：把偏好对直接当作回归目标，用一个对比式损失让模型在每对答案上学习“更好”。  
   *改变*：摆脱了奖励函数的单调假设，能够自然处理循环或不传递的偏好关系。

2. **把 RLHF 视作两玩家纳什博弈**  
   *之前的视角*：RLHF 被看作单一的优化过程（先学奖励、后学策略）。  
   *本文的做法*：将模型自身和强老师（如 GPT‑4）抽象为博弈双方，目标是找到一个纳什均衡，使得双方在当前策略下都不想单方面改变。  
   *改变*：提供了一个明确的游戏论解释，进而得到“单调改进”理论保证。

3. **批量 on‑policy 回归目标的实现**  
   *传统的对比学习*：往往使用离线采样的负例，可能与当前策略不匹配。  
   *本文的做法*：每轮用当前模型生成的答案对（包括与老师模型的比较）直接构造回归目标，保持“on‑policy”。  
   *改变*：实现了训练过程的高效闭环，避免了离线数据的分布漂移。

4. **简洁可扩展的实现**  
   *以往的 RLHF*：需要复杂的 PPO 或者 KL‑约束等技巧，代码实现繁琐。  
   *本文的做法*：只需一个普通的最小二乘回归层，加上对比式采样，几行代码即可跑通。  
   *改变*：大幅降低了工程门槛，使得小模型（7B）也能在有限算力下实现显著提升。

### 方法详解

#### 整体框架

DNO 的训练循环可以概括为四步：  
1) **采样**：用当前模型（学生）生成一批候选答案；同时让强老师模型（如 GPT‑4）对同一输入生成参考答案。  
2) **构造偏好对**：根据外部偏好 oracle（可以是人工标注、模型评估或自动评分）得到“学生答案 vs 老师答案”的偏好标签（谁更好）。  
3) **回归目标**：把每个偏好对映射为一个标量差值目标，目标值为 +1（学生更好）或 -1（老师更好），并用最小二乘回归让模型的对数概率差逼近这个目标。  
4) **参数更新**：对整个批次做一次梯度下降，得到新的学生模型。循环往复，直至偏好胜率不再提升。

#### 关键模块拆解

- **偏好对生成**  
  类比于两位评委对同一道题的答案进行比较，DNO 把学生模型的答案和老师模型的答案放在一起，让 oracle 给出“更好”的判定。这里的 oracle 可以是 GPT‑4 本身的评分函数，也可以是人工标注的二选一数据。

- **回归式对比损失**  
  传统对比学习使用交叉熵把好答案的概率推向 1、坏答案推向 0。DNO 把好坏关系映射成一个实数差值（+1 / -1），然后最小化模型输出的对数概率差与该差值的平方误差。直白来说，就是让模型说“我更倾向于答案 A 而不是 B”，并且这个倾向的强度要和 oracle 给出的偏好强度匹配。

- **批量 on‑policy**  
  每一次梯度更新都使用当前模型自己生成的答案对，确保训练数据的分布始终和模型的最新策略保持一致。这样避免了离线数据导致的“分布漂移”，类似于跑步时每一步都根据最新的姿势微调。

- **单调改进的理论保障**  
  作者把学生-老师两方的交互建模为零和博弈，证明在每次回归更新后，学生模型的期望对手优势（即在偏好对上胜率）不会下降。换句话说，模型每一步都在“向更好”迈进，类似于爬山时每次都保证不往下走。

#### 最巧妙的设计

把偏好对直接映射为回归目标，而不是先训练一个奖励模型再最大化，这一步看似简单，却彻底绕开了奖励函数的“点对点”限制。再加上把整个过程包装成两玩家纳什博弈的数学框架，使得单调改进不再是经验性的，而是有理论支撑的必然结果。

### 实验与效果

- **测试平台**：作者在 AlpacaEval 2.0 上评估模型，使用 7B 参数的 Orca‑2.5 作为基线模型，教师模型为 GPT‑4‑Turbo。  
- **主要结果**：经过 DNO 微调后，Orca‑2.5 在与 GPT‑4‑Turbo 的对话对抗中赢得 33% 的胜率（控制了回答长度后仍保持），相较于原始模型的 7% 提升了 26 个绝对百分点，刷新了同等规模模型的记录。  
- **对比基线**：在同样的评测中，Mistral Large（更大参数量）以及 Self‑Rewarding LM（70B）均未能超过 DNO 微调后的 7B 模型。  
- **消融实验**：论文报告了去掉 on‑policy 采样、改用离线奖励模型或去掉回归目标的对比实验，发现每一项都显著削弱胜率，尤其是去掉 on‑policy 会导致约 10% 的性能回落。  
- **局限性**：作者承认 DNO 仍依赖高质量的偏好 oracle；如果 oracle 本身有偏差，模型会放大这些偏差。此外，单轮对比只考虑两条答案，未探索多答案集合的更复杂偏好结构。

### 影响与延伸思考

DNO 把 RLHF 重新包装成纳什博弈的视角，为“直接优化偏好”提供了理论与实现的双重支撑。自论文发布后，出现了多篇工作尝试将其他类型的偏好（如层次化、分段式）嵌入同样的对比回归框架，甚至把多模态（文本+图像）偏好也纳入纳什优化。对想进一步深入的读者，可以关注以下方向：  
- **偏好 oracle 的自我提升**：如何让模型在没有外部强教师的情况下自行生成可靠的偏好信号。  
- **多方博弈扩展**：把两玩家扩展到多模型协同或竞争的情形，探索更丰富的均衡概念。  
- **理论分析深化**：在更一般的非零和游戏或带噪声偏好的情况下，证明单调改进的条件。

### 一句话记住它

**DNO 用对比回归直接在偏好对上做纳什博弈，让小模型在每一步都稳步超越强老师。**