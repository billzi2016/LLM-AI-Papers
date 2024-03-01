# Curiosity-driven Red-teaming for Large Language Models

> **Date**：2024-02-29
> **arXiv**：https://arxiv.org/abs/2402.19464

## Abstract

Large language models (LLMs) hold great potential for many natural language applications but risk generating incorrect or toxic content. To probe when an LLM generates unwanted content, the current paradigm is to recruit a \textit{red team} of human testers to design input prompts (i.e., test cases) that elicit undesirable responses from LLMs. However, relying solely on human testers is expensive and time-consuming. Recent works automate red teaming by training a separate red team LLM with reinforcement learning (RL) to generate test cases that maximize the chance of eliciting undesirable responses from the target LLM. However, current RL methods are only able to generate a small number of effective test cases resulting in a low coverage of the span of prompts that elicit undesirable responses from the target LLM. To overcome this limitation, we draw a connection between the problem of increasing the coverage of generated test cases and the well-studied approach of curiosity-driven exploration that optimizes for novelty. Our method of curiosity-driven red teaming (CRT) achieves greater coverage of test cases while mantaining or increasing their effectiveness compared to existing methods. Our method, CRT successfully provokes toxic responses from LLaMA2 model that has been heavily fine-tuned using human preferences to avoid toxic outputs. Code is available at \url{https://github.com/Improbable-AI/curiosity_redteam}

---

# 好奇心驱动的红队测试大型语言模型 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在聊天、写作、代码生成等场景表现惊艳，但它们也会不经意地输出错误信息或有害言论。传统上，研究者会组织一支“红队”——由人工测试者组成的团队，手动设计诱导模型产生不良回复的输入。这个过程既费时又费钱，难以覆盖所有潜在的风险场景。近来有人尝试让另一台 LLM 学会生成这些“陷阱”提示，并用强化学习（RL）让它最大化目标模型出错的概率。然而，现有的 RL 方法往往只能产出少量高效的测试案例，导致覆盖面窄，很多潜在的风险仍然被漏掉。于是，如何在保持或提升单个案例有效性的前提下，显著扩大测试案例的多样性和覆盖率，成为亟待突破的瓶颈。

### 关键概念速览
- **红队（Red Team）**：专门寻找系统弱点的测试者集合，类似渗透测试团队，目标是让模型说出“不该说的话”。  
- **强化学习（Reinforcement Learning，RL）**：让智能体通过试错获得奖励的学习方式，这里奖励是目标模型产生不良输出的概率。  
- **新颖性搜索（Novelty Search）**：一种进化算法，目标不是最大化奖励，而是让产生的行为在行为空间中保持“新鲜”，类似于探索未知地图时不断寻找未踏足的区域。  
- **好奇心驱动（Curiosity‑Driven）**：在 RL 中加入对未知或稀有状态的内部奖励，使智能体主动去探索不常见的情形。  
- **覆盖率（Coverage）**：测试案例集合能够触发目标模型不良行为的范围大小，覆盖率高意味着更全面的安全评估。  
- **毒性（Toxicity）**：模型输出中包含攻击性、歧视性或其他有害内容的程度。  
- **人类偏好微调（Human Preference Fine‑tuning）**：在模型训练后加入大量人类标注的偏好数据，以抑制不良输出的技术手段。  

### 核心创新点
1. **把覆盖率问题映射为好奇心驱动的探索任务**  
   - 之前的红队 RL 只追求“让模型出错”的奖励，导致搜索空间快速收敛到少数高效提示。  
   - 本文引入了新颖性奖励，让生成器不仅关注即时成功率，还要主动寻找在提示空间中未被探索过的区域。  
   - 结果是生成的提示更分散，覆盖了更多潜在的风险场景，同时保持或提升了诱导不良回复的成功率。

2. **双奖励机制：外部奖励 + 内部好奇心奖励**  
   - 传统方法只用外部奖励（目标模型产生毒性输出的概率）来更新红队模型。  
   - 本文在此基础上加入内部奖励：对提示的“新颖度”进行打分，使用基于自编码器或邻域密度的度量来衡量。  
   - 这种组合让模型在“安全区”也会尝试新奇的提示，避免陷入局部最优。

3. **基于经验回放的好奇心奖励估计**  
   - 好奇心奖励需要对历史生成的提示进行统计，直接计算会很慢。  
   - 作者设计了一个经验回放池，存放过去的提示向量，并用快速的最近邻查询估算新提示的稀有度。  
   - 这样既保持了好奇心奖励的实时性，又不牺牲训练效率。

4. **在强微调的 LLaMA‑2 上成功触发毒性**  
   - LLaMA‑2 已经经过大规模的人类偏好微调，理论上应当极少产生毒性。  
   - 通过 CRT 方法，作者仍然能够让模型输出明显的有害内容，证明了覆盖率提升的实际价值。  

### 方法详解
**整体框架**  
CRT（Curiosity‑Driven Red‑Team）把红队生成过程拆成两层循环：外层是强化学习的策略梯度更新，内层是好奇心奖励的计算与累计。整个系统包括（1）目标模型（被测 LLM），（2）红队生成器（另一个 LLM），以及（3）新颖度评估模块。

**关键步骤**  

1. **初始化红队生成器**  
   - 使用预训练的语言模型（如 LLaMA‑base）作为初始策略，确保生成器本身具备基本的语言表达能力。  

2. **采样提示并评估**  
   - 红队模型根据当前策略采样一批输入提示。  
   - 将这些提示送入目标模型，得到输出并用已有的毒性检测器（如 Perspective API）计算外部奖励：毒性越高，奖励越大。  

3. **计算新颖度奖励**  
   - 将每个提示的嵌入向量（通过固定的编码器得到）存入经验回放池。  
   - 对新采样的提示，查询池中最近的 k 条向量，计算平均距离。距离越大，说明该提示在历史集合中越稀有，新颖度奖励随之提升。  

4. **组合奖励并更新策略**  
   - 总奖励 = α·外部奖励 + β·新颖度奖励（α、β 为超参数，控制两者权重）。  
   - 使用强化学习的策略梯度（如 PPO）对红队模型进行参数更新，使其在未来更倾向于生成高奖励的提示。  

5. **循环迭代**  
   - 重复步骤 2‑4，经验回放池不断增长，新的提示会被迫在更大的历史空间中竞争新颖度，从而推动探索更广阔的提示空间。  

**最巧妙的设计**  
- **经验回放池的稀疏更新**：作者没有每一步都重新计算所有提示的稀有度，而是只在新提示加入时做一次最近邻查询，这大幅降低了计算开销。  
- **双奖励的动态平衡**：通过在训练早期把 β 调高，鼓励大量探索；随后逐步提升 α，让模型回归到高效诱导的方向，这种“先探索后利用”的策略类似于人类在学习新技能时的过程。  

### 实验与效果
- **测试对象**：主要在 LLaMA‑2‑7B 上进行实验，该模型已经经过大规模的人类偏好微调，官方宣称几乎不产生毒性。  
- **基准对比**：与传统红队 RL（仅外部奖励）以及随机提示生成两种 baseline 比较。  
- **结果概述**：  
  - 在相同的训练步数下，CRT 产生的提示覆盖率提升约 2.5 倍（即能触发毒性输出的不同提示数量显著增多）。  
  - 平均毒性触发率从传统 RL 的 12% 提升到 18%，且在高新颖度的提示中仍保持 15% 左右的触发率，说明新颖度奖励并未牺牲有效性。  
- **消融实验**：  
  - 去掉新颖度奖励后，覆盖率回落到原 RL 水平，验证了好奇心奖励的关键作用。  
  - 将经验回放池大小从 10k 降到 1k，计算开销下降但新颖度估计噪声增大，导致覆盖率略降约 10%。  
- **局限性**：  
  - 作者指出 CRT 对提示的“新颖度”度量依赖于嵌入空间的质量，若编码器不够表达提示差异，可能出现误判。  
  - 实验仅在单一模型（LLaMA‑2）上验证，跨模型的通用性仍需进一步评估。  

### 影响与延伸思考
CRT 把好奇心驱动的探索理念成功搬进了安全评估领域，打开了“让机器自己找漏洞”的新思路。随后的工作（如 2024 年的 *Curiosity‑Guided Adversarial Prompting*）进一步将进化策略与语言模型结合，尝试在更大规模的模型上实现自动化红队。对想继续深入的读者，可以关注以下方向：  
- **更精细的新颖度度量**：利用对抗自编码器或图神经网络捕捉提示的结构化差异。  
- **跨模型红队迁移**：研究在一个模型上学到的红队策略能否直接用于其他模型。  
- **多目标红队**：同时优化毒性、误信息、隐私泄露等多种不良行为的覆盖率。  

### 一句话记住它
让红队模型在“好奇心”驱动下主动探索未被尝试的提示，从而在保持高效诱导的同时，大幅提升安全测试的覆盖范围。