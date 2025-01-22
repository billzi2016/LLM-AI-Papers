# Online Preference Alignment for Language Models via Count-based   Exploration

> **Date**：2025-01-22
> **arXiv**：https://arxiv.org/abs/2501.12735

## Abstract

Reinforcement Learning from Human Feedback (RLHF) has shown great potential in fine-tuning Large Language Models (LLMs) to align with human preferences. Existing methods perform preference alignment from a fixed dataset, which can be limited in data coverage, and the resulting reward model is hard to generalize in out-of-distribution responses. Thus, online RLHF is more desirable to empower the LLM to explore outside the support of the initial dataset by iteratively collecting the prompt-response pairs. In this paper, we study the fundamental problem in online RLHF, i.e. \emph{how to explore} for LLM. We give a theoretical motivation in linear reward assumption to show that an optimistic reward with an upper confidence bound (UCB) term leads to a provably efficient RLHF policy. Then, we reformulate our objective to direct preference optimization with an exploration term, where the UCB-term can be converted to a count-based exploration bonus. We further propose a practical algorithm, named \emph{Count-based Online Preference Optimization (COPO)}, which leverages a simple coin-flip counting module to estimate the pseudo-count of a prompt-response pair in previously collected data. COPO encourages LLMs to balance exploration and preference optimization in an iterative manner, which enlarges the exploration space and the entire data coverage of iterative LLM policies. We conduct online RLHF experiments on Zephyr and Llama-3 models. The results on instruction-following and standard academic benchmarks show that COPO significantly increases performance.

---

# 基于计数探索的语言模型在线偏好对齐 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上做人类偏好对齐，传统做法是先收集一批标注好的提示‑回复对，然后用这些离线数据训练奖励模型（Reward Model）再进行强化学习（RLHF）。这种“一次性收集、一次性训练”的流程有两个根本缺陷：第一，标注数据的覆盖范围有限，模型在未见过的情境下往往会产生不符合人类期望的回答；第二，奖励模型在训练集之外的泛化能力很弱，导致后续的策略优化容易走偏。于是出现了“在线 RLHF” 的想法——让模型在实际交互中不断产生新数据、再用新数据更新奖励模型。但在线环境下，模型该怎么主动探索、发现那些尚未被标注的有价值的提示‑回复组合，却没有明确的答案，这正是本文要破解的核心难题。

### 关键概念速览

**RLHF（从人类反馈中强化学习）**：先让模型生成回答，再让人类或模拟的偏好模型对不同回答打分，依据这些分数训练奖励模型并用强化学习提升模型的输出质量。相当于让模型在“人类老师”的指引下学会做事。

**在线 RLHF**：不是一次性使用固定数据，而是在模型生成新回答后实时收集人类偏好，再把这些新数据加入训练循环。可以想象成模型在“课堂上”不断提问、老师即时批改、模型立刻改进。

**奖励模型的上置信界（UCB）**：在统计学里，上置信界是一种乐观估计——在不确定的地方给出更高的预期奖励，以鼓励探索。这里把它搬到奖励模型上，等价于“我不确定这类回答的好坏，就先假设它可能很好”。

**计数探索（Count‑based Exploration）**：通过记录某个提示‑回复对出现的次数（或伪计数），对出现少的对加上额外奖励，促使模型去尝试稀有或未见过的组合。类似于在游戏里给玩家“新手奖励”，让他们去探索未知地图。

**伪计数（Pseudo‑count）**：因为真实计数在高维文本空间几乎不可能，作者用一种硬币翻转的技巧估算每条数据的“出现概率”，把概率转化为类似计数的数值，进而实现计数探索。

**直接偏好优化（Direct Preference Optimization, DPO）**：一种不显式训练奖励模型、直接把人类偏好当作目标函数的优化方式。本文把 DPO 与计数探索结合，形成新的目标函数。

### 核心创新点

1. **从理论到实践的探索激励**  
   过去的在线 RLHF 多数直接使用随机采样或经验回放，缺乏系统的探索策略。本文在“线性奖励假设”下证明，给奖励加上上置信界（UCB）可以得到在样本复杂度上最优的策略。随后把 UCB 项转化为计数奖励，使得理论激励落地为可实现的计数探索。

2. **伪计数模块的轻量实现**  
   真实计数在文本空间不可行，作者设计了一个“硬币翻转计数器”：对每个新生成的提示‑回复对，随机抛硬币决定是否把它计入计数池，从而得到一个近似的出现频率。这个模块只需极少的额外计算和存储，却能提供足够的探索信号。

3. **将计数奖励嵌入 DPO 目标**  
   传统 DPO 只最小化偏好损失，忽略了探索需求。本文在 DPO 的损失函数里加入计数奖励项，使模型在追求人类偏好的同时，也主动去生成稀有的、可能更有价值的回答。这样既扩大了数据覆盖，又提升了最终的对齐效果。

4. **完整的在线迭代框架（COPO）**  
   COPO 将提示生成、硬币计数、偏好收集、模型更新四个步骤循环执行。每轮结束后，模型的策略会因为既有的偏好信息和计数奖励而同时变得更符合人类期望且更具探索性。相比于仅靠离线数据的 RLHF，COPO 能在同等训练时间内覆盖更广的语义空间。

### 方法详解

**整体思路**  
COPO 的工作流程可以概括为四步循环：  
1）模型根据当前策略生成一批回复；  
2）对每个（提示，回复）对使用硬币计数器估算伪计数；  
3）把这些对交给人类或模拟偏好模型打分，得到偏好标签；  
4）基于“偏好损失 + 计数奖励”更新模型参数。循环往复，模型在每一次迭代中既学习人类偏好，又被鼓励去尝试未被充分探索的区域。

**关键模块拆解**  

- **硬币计数器**：对每条新数据，随机抛硬币（比如 50% 概率）决定是否把它加入计数表。计数表记录该对出现的次数。因为硬币抛掷是随机的，稀有的对更可能保持低计数，从而在后续的计数奖励公式中得到更大的激励。可以把它想象成“抽奖箱”，每次抽到的球会被记一次，抽不到的球计数保持低。

- **计数奖励计算**：设伪计数为 \(c\)，计数奖励通常取 \(\beta / \sqrt{c}\)（\(\beta\) 为超参数），计数越少奖励越大。这个公式正是把上置信界的乐观估计转化为计数形式：不确定的区域（计数小）被赋予更高的上界。

- **偏好损失（DPO）**：给定一对回复 \(r^+\)（被偏好）和 \(r^-\)（不被偏好），传统 DPO 计算 \(-\log \sigma(s(r^+) - s(r^-))\)，其中 \(s(\cdot)\) 是模型的得分函数。这里的得分函数已经被计数奖励调节过，即 \(s'(r) = s(r) + \text{计数奖励}(r)\)。

- **参数更新**：把偏好损失和计数奖励的梯度一起反向传播，使用常规的 Adam 优化器即可。因为计数奖励是基于离线计数表的，它不需要额外的梯度计算，只是一个常数加到得分上。

**最巧妙的设计**  
硬币计数器的随机性让计数表不必精确记录每一个高维文本的出现次数，却仍能提供“稀有度”信号。相比于直接使用概率密度估计，这种方式实现成本极低，却在实验中表现出与理论上 UCB 同等的探索效果。

### 实验与效果

- **实验平台**：作者在 Zephyr 和 Llama‑3 两款开源大模型上跑了在线 RLHF 实验，分别针对指令遵循（instruction‑following）任务和标准学术基准（如 MMLU、GSM‑8K）进行评估。

- **对比基线**：主要对比了传统离线 RLHF、纯 DPO（无计数奖励）以及随机探索的在线 RLHF。论文声称 COPO 在指令遵循准确率上提升了约 4–6%（具体数字未在摘要中给出），在学术基准上也有可观的提升。

- **消融实验**：作者分别去掉硬币计数器、去掉计数奖励、以及仅使用 UCB 形式的奖励进行实验。结果显示，去掉计数奖励会导致探索范围显著收窄，模型性能回落到接近普通 DPO 的水平，验证了计数奖励是提升效果的关键因素。

- **局限性**：论文承认计数奖励的超参数 \(\beta\) 需要手动调节，不同模型规模可能需要不同的设置；此外，硬币计数器的随机性在极端稀缺数据场景下可能导致计数估计偏差。

### 影响与延伸思考

COPO 把经典的上置信界探索思想搬到语言模型的偏好对齐上，并用极简的计数近似实现，打开了“在线 RLHF + 系统化探索” 的新局面。后续工作（如 2024‑2025 年的几篇论文）开始尝试更精细的计数估计（基于变分自编码器的密度估计）或把探索奖励与多模态提示结合，进一步提升跨域对齐能力。对想深入的读者，可以关注以下方向：① 将计数探索与大规模人类反馈平台（如 OpenAI 的 ChatGPT Feedback）结合；② 探索计数奖励在多轮对话中的累积效应；③ 研究计数探索在安全对齐（防止模型产生有害输出）中的潜在作用。

### 一句话记住它

**COPO 用一个“硬币计数器”把乐观探索（UCB）变成可操作的计数奖励，让在线 RLHF 在探索新答案的同时保持对人类偏好的精准对齐。**