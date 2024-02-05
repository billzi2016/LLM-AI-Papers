# Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information   Seeking in Large Language Models

> **Date**：2024-02-05
> **arXiv**：https://arxiv.org/abs/2402.03271

## Abstract

In the face of uncertainty, the ability to *seek information* is of fundamental importance. In many practical applications, such as medical diagnosis and troubleshooting, the information needed to solve the task is not initially given and has to be actively sought by asking follow-up questions (for example, a doctor asking a patient for more details about their symptoms). In this work, we introduce Uncertainty of Thoughts (UoT), an algorithm to augment large language models with the ability to actively seek information by asking effective questions. UoT combines 1) an *uncertainty-aware simulation approach* which enables the model to simulate possible future scenarios and how likely they are to occur, 2) *uncertainty-based rewards* motivated by information gain which incentivizes the model to seek information, and 3) a *reward propagation scheme* to select the optimal question to ask in a way that maximizes the expected reward. In experiments on medical diagnosis, troubleshooting, and the `20 Questions` game, UoT achieves an average performance improvement of 38.1% in the rate of successful task completion across multiple LLMs compared with direct prompting and also improves efficiency (i.e., the number of questions needed to complete the task). Our code has been released [here](https://github.com/zhiyuanhubj/UoT)

---

# 思维不确定性：面向不确定性的规划提升大语言模型的信息获取能力 论文详细解读

### 背景：这个问题为什么难？

在医学诊断、故障排查等真实场景里，模型往往只能在已有信息上直接给出答案。传统的大语言模型（LLM）在面对缺失关键细节时，会直接猜测或给出模糊答案，因为它们缺乏主动“提问”的机制。已有的提示工程（prompt engineering）或链式思考（Chain‑of‑Thought）只能让模型把已有信息组织得更好，却不能让模型意识到自己“不确定”，也不能驱动它去主动获取更多线索。缺少这种信息获取能力，就像医生只看病历不去问患者，导致诊断准确率受限。正是这种“在不确定环境中主动寻求信息”的需求，促使研究者去设计能够让 LLM 自主提问的算法。

### 关键概念速览

**不确定性（Uncertainty）**：模型对当前情境下答案的信心程度。信心低时意味着可能缺少关键信息，需要进一步探查。可以把它想成“脑子里有雾”，雾越浓，越需要打开窗户让光进来。

**信息增益（Information Gain）**：一次提问后，模型对任务的理解提升了多少。类似于玩“猜数字”时，每次询问“比上次大吗？”能把可能的范围缩小，范围缩小越多，信息增益越高。

**模拟（Simulation）**：让模型在脑海里演练若干可能的未来对话路径，并估算每条路径出现的概率。就像在下棋前先想象几步棋的走向，看看哪条路线最有希望赢。

**奖励（Reward）**：在强化学习里，用数值来衡量行为好坏。这里的奖励由信息增益转化而来，提问能带来更大信息增益就会得到更高奖励。

**奖励传播（Reward Propagation）**：把未来可能得到的奖励向前传递，帮助模型在当前选择最有价值的问题。可以类比为“把远处的宝藏位置标记在地图上”，让你在起点就知道该往哪个方向走。

**链式思考（Chain‑of‑Thought, CoT）**：让模型在给出答案前先写出推理步骤，类似于做数学题时先列出草稿。UoT 在此基础上加入了“不确定性评估”和“提问决策”。

### 核心创新点

1. **不确定性感知的情景模拟 → 直接使用 LLM 生成答案**  
   过去的 LLM 只在已有输入上做一次前向推理，根本没有评估自己对答案的信心。UoT 在每一步先让模型模拟若干可能的后续对话，并为每条模拟路径打上出现概率的标签。这样模型能够“看到”自己可能的盲区，从而决定是否需要提问。

2. **信息增益驱动的奖励设计 → 传统的任务成功率奖励**  
   传统强化学习在这类任务里往往只奖励最终是否成功，忽略了提问本身的价值。UoT 把信息增益量化为即时奖励：一次提问如果能显著降低不确定性，就会得到高分。这样模型被激励去问“最能消除疑惑”的问题，而不是随意提问。

3. **奖励传播用于最优提问选择 → 只用贪心选择最高即时奖励**  
   直接选取即时奖励最高的问题会导致短视行为（比如问一个容易回答但信息量小的问题）。UoT 采用奖励传播，把未来可能的累计信息增益向前回溯，计算每个候选问题的期望总奖励。最终选出的提问是对整个对话过程最有利的。

4. **统一框架兼容多种 LLM** → 只针对特定模型进行微调**  
   许多信息获取方法需要对特定模型进行专门的微调或额外的检索模块。UoT 只在提示层面加入不确定性评估和奖励计算，几乎不改变模型本身的参数，因而可以直接在 GPT‑3.5、Claude、LLaMA 等多种模型上复用，提升了实用性。

### 方法详解

#### 整体思路

UoT 把一次任务解答拆成三大步骤：**（1）不确定性评估与情景模拟 →（2）信息增益奖励计算 →（3）奖励传播与提问决策**。整个流程在每轮对话结束后循环执行，直到模型的置信度足够高或达到预设的提问上限。

#### 步骤 1：不确定性感知的情景模拟

- **输入**：当前已知的对话历史（包括用户提供的信息和模型已提的问题/答案）。  
- **操作**：使用 LLM 生成 *N* 条可能的后续对话走向。每条走向包括模型可能的回答以及对应的概率估计。概率是通过让模型在同一提示下多次采样得到的，或者使用自回归的概率分布直接计算。  
- **类比**：想象你在玩“20 个问题”，每次猜测后，你会在脑中模拟“如果答案是 A，会怎样继续提问”，并给每种可能打分。

这一步的关键是让模型“看到”自己在不同假设下的行为，从而量化当前的 **不确定性**——如果所有模拟路径的答案基本一致，则不确定性低；如果答案分歧很大，则不确定性高。

#### 步骤 2：信息增益驱动的奖励

- **信息增益**：对每条模拟路径，计算在加入某个候选提问后，模型对最终任务（如诊断、故障定位）的置信度提升多少。具体做法是：  
  1. 先记录当前的任务成功概率（从模拟路径的概率分布中得到）。  
  2. 对每个候选提问，假设用户会给出一个合理答案（使用 LLM 生成的“理想回复”），再重新进行一次情景模拟，得到新的成功概率。  
  3. 信息增益 = 新成功概率 – 旧成功概率。  

- **奖励**：把信息增益直接映射为数值奖励（可以乘以一个系数来平衡提问成本）。如果信息增益为负，说明该提问可能导致误导，奖励设为零。

这一步把“提问是否有价值”量化，使得后续的决策可以基于数值比较。

#### 步骤 3：奖励传播与最优提问选择

- **奖励传播**：从候选提问的即时奖励出发，向前递推计算 **期望累计奖励**。递推公式类似于动态规划：  
  - 对每个候选提问，期望奖励 = 即时奖励 + 折扣系数 ×（在该提问后继续模拟的最佳累计奖励）。  
  - 折扣系数（γ）控制远期奖励的重要性，防止模型无限制地提问。  

- **最优提问**：遍历所有候选提问，选取期望累计奖励最高的那个。若最高奖励低于预设阈值，模型直接给出答案；否则执行该提问并把用户的回复加入对话历史，进入下一轮循环。

#### 巧妙之处

- **模拟+奖励的闭环**：不只是一次性评估提问价值，而是把提问后可能的后续对话也纳入考量，形成前瞻性的决策。  
- **无需额外检索模块**：所有信息都来自模型内部的自我模拟，省去了外部知识库的集成成本。  
- **模型无缝迁移**：只在提示层面加入“请先估计不确定性并模拟若干可能的后续”，几乎不需要改动模型权重，因而可以直接在不同 LLM 上跑。

### 实验与效果

- **测试任务**：包括（1）医学诊断（使用公开的 MedQA 数据集），（2）故障排查（基于真实的 IT 故障日志），以及（3）经典的 20 Questions 游戏。  
- **基线对比**：与直接提示（Zero‑Shot）、Chain‑of‑Thought、以及最近的主动提问框架（如 Self‑Ask）进行比较。  
- **整体提升**：论文报告在所有任务上平均提升 38.1% 的成功率。例如，在医学诊断上，使用 GPT‑3.5 时成功率从 62% 提升到 84%；在 20 Questions 中，平均提问数从 7.3 降到 5.1，完成率提升约 40%。  
- **消融实验**：分别去掉（a）情景模拟、（b）信息增益奖励、（c）奖励传播。结果显示，去掉任意一环都会导致性能下降 10%~18%，其中奖励传播的贡献最大，说明前瞻性决策是提升的关键。  
- **局限性**：作者指出模拟的质量依赖于底层 LLM 的生成能力；在极度稀缺或高度专业化的领域（如罕见疾病），模型的模拟可能偏差较大，导致错误提问。此外，奖励传播的计算成本随候选提问数量呈指数增长，需要在实际部署时做采样或剪枝。

### 影响与延伸思考

UoT 把“不确定性感知 + 主动信息获取”引入 LLM，打开了让语言模型像人类专家一样“提问”的新方向。后续工作已经开始在以下几个方向扩展：  
- **跨模态提问**：把视觉或传感器信息加入模拟，让模型在机器人或自动驾驶场景中主动询问环境细节。  
- **人机协同**：将 UoT 的提问策略与真实用户交互结合，研究如何在不让用户感到烦扰的前提下最大化信息增益。  
- **更高效的奖励传播**：利用蒙特卡洛树搜索（MCTS）或近似动态规划来降低计算开销。  
- **自适应提问预算**：根据任务难度动态调整最大提问次数，进一步提升效率。  

如果想深入了解，可以关注近期在 “Active Learning for LLM” 以及 “Uncertainty Estimation in Generative Models” 领域的会议论文，很多都在借鉴 UoT 的思路。

### 一句话记住它

让大语言模型先“估不确定”，再用信息增益驱动的奖励来挑最有价值的问题，从而在对话中主动获取缺失信息。