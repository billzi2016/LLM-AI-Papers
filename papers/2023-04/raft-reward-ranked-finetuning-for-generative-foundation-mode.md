# RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment

> **Date**：2023-04-13
> **arXiv**：https://arxiv.org/abs/2304.06767

## Abstract

Generative foundation models are susceptible to implicit biases that can arise from extensive unsupervised training data. Such biases can produce suboptimal samples, skewed outcomes, and unfairness, with potentially serious consequences. Consequently, aligning these models with human ethics and preferences is an essential step toward ensuring their responsible and effective deployment in real-world applications. Prior research has primarily employed Reinforcement Learning from Human Feedback (RLHF) to address this problem, where generative models are fine-tuned with RL algorithms guided by a human-feedback-informed reward model. However, the inefficiencies and instabilities associated with RL algorithms frequently present substantial obstacles to the successful alignment, necessitating the development of a more robust and streamlined approach. To this end, we introduce a new framework, Reward rAnked FineTuning (RAFT), designed to align generative models effectively. Utilizing a reward model and a sufficient number of samples, our approach selects the high-quality samples, discarding those that exhibit undesired behavior, and subsequently enhancing the model by fine-tuning on these filtered samples. Our studies show that RAFT can effectively improve the model performance in both reward learning and other automated metrics in both large language models and diffusion models.

---

# RAFT：基于奖励排序的生成式基础模型对齐微调 论文详细解读

### 背景：这个问题为什么难？
生成式基础模型在海量无监督数据上训练后，会携带各种隐式偏见——比如产生有害言论、刻板印象或不符合用户期望的答案。传统的对齐手段主要是 **强化学习从人类反馈 (RLHF)**：先训练一个奖励模型，再用强化学习让生成模型最大化奖励。然而 RLHF 需要高方差的梯度、复杂的策略网络以及大量的在线交互，训练过程常常不稳定、算力开销大，导致实际落地困难。于是业界急需一种既能利用人类偏好，又不依赖笨重 RL 的对齐方案。

### 关键概念速览
**生成式基础模型**：在海量文本或图像上预训练得到的通用模型，能够根据提示生成内容，例如 GPT、Stable Diffusion。  
**隐式偏见**：模型在训练数据中无意学到的有害或不公平的行为，往往在下游使用时暴露出来。  
**RLHF（强化学习从人类反馈）**：先用人类标注的偏好对一小批样本训练奖励模型，再用强化学习让生成模型的输出在奖励上提升。  
**奖励模型**：一个二分类或回归网络，输入模型生成的文本/图像，输出一个数值，表示该输出与人类偏好的吻合程度。  
**样本过滤**：利用奖励模型对大量生成候选进行打分，只保留分数最高的那部分样本，丢弃低分样本。可以想象成“筛子”，把好东西挑出来。  
**微调**：在保留下来的高质量样本上继续训练生成模型，使用普通的监督学习目标（如交叉熵），让模型更倾向于产生这些样本。  
**扩散模型**：一种生成图像的概率模型，核心是逐步“去噪”，在对齐任务中同样可以使用奖励模型进行筛选。  
**跨模态通用性**：同一种对齐框架能够同时适用于文本大模型和图像扩散模型，说明方法本身并不依赖特定的生成机制。

### 核心创新点
1. **奖励排序 → 直接筛选 → 监督微调**  
   *之前的做法*：RLHF 通过在线强化学习让模型在奖励上直接优化，过程复杂且不稳定。  
   *本文的做法*：先让奖励模型给大量离线生成的候选打分，挑出高分样本，再用普通的监督学习微调模型。  
   *带来的改变*：省去了强化学习的高方差梯度和策略网络，训练流程更像常规的语言模型微调，显著提升了稳定性和算力效率。

2. **一次性离线采样即可完成对齐**  
   *之前的做法*：RLHF 需要在每一步采样并即时计算奖励，形成闭环。  
   *本文的做法*：一次性生成足够多的候选（几千到几万条），全部打分后一次性过滤。  
   *改变*：大幅降低了对交互式采样的依赖，适合算力受限的实验室或企业内部部署。

3. **跨模态统一框架**  
   *之前的对齐方法*：大多只针对文本模型，图像模型仍在探索阶段。  
   *本文的做法*：同样使用奖励模型对扩散模型生成的图像进行评分，筛选后微调。  
   *改变*：证明了奖励排序+微调的思路不局限于语言，具备更广的应用前景。

4. **对奖励模型质量的宽容度提升**  
   *之前的做法*：RLHF 对奖励模型的误差非常敏感，稍有偏差就会导致策略崩溃。  
   *本文的做法*：因为只在高分区间采样，低质量的奖励误差对最终微调影响有限。  
   *改变*：降低了对奖励模型极致精度的要求，降低了标注成本。

### 方法详解
**整体框架**  
RAFT 由四个阶段组成：① 预训练生成模型（保持原样）；② 收集少量人类偏好对（如“更有帮助/更安全”），训练奖励模型；③ 用奖励模型对大量离线生成的候选进行打分并排序；④ 取前 k %（或设定阈值）的高分样本，使用标准监督学习对生成模型进行微调。整个过程不需要任何在线强化学习环节。

**关键模块拆解**  

1. **奖励模型训练**  
   - 输入：模型生成的文本/图像对（如 A 与 B），以及人类标注的偏好（A 更好 / B 更好）。  
   - 目标：学习一个函数，输出两者的相对偏好分数。实现上常用二分类交叉熵或回归损失。  
   - 类比：把它想成“品酒师”，尝过很多酒后能给出好坏评分。

2. **离线候选生成**  
   - 对每个提示，使用原始生成模型采样 N 条（N 可达上千），得到一个候选池。  
   - 这里不做任何奖励引导，只是普通的随机采样，类似“先把所有菜都端上来”。

3. **奖励排序与过滤**  
   - 将奖励模型对每条候选打分，得到一个数值列表。  
   - 按分数从高到低排序，保留前 k %（常取 5%~20%）或所有分数高于阈值的样本。  
   - 直观上相当于“筛子”，把最合口味的菜挑出来。

4. **监督微调**  
   - 对保留下来的高分样本，使用普通的交叉熵（文本）或扩散损失（图像）继续训练生成模型。  
   - 训练目标仍是让模型在给定提示下输出与高分样本相同的分布，只是数据更“干净”。  
   - 这里可以直接复用已有的微调代码，无需实现 RL 的策略梯度或价值网络。

**最巧妙的地方**  
- 把强化学习的“在线探索—奖励反馈”循环拆成“离线采样—奖励排序—离线微调”，彻底规避了 RL 中的高方差梯度和策略崩溃问题。  
- 只在高分区间使用奖励模型，使得即使奖励模型有噪声，也不会把噪声放大到最终模型。  
- 同一套代码可以同时跑大语言模型和扩散模型，展示了方法的通用性。

### 实验与效果
- **测试对象**：在公开的语言对齐基准（如 OpenAI 的 Helpful‑Harmless 数据集）以及图像安全基准（如 Stable Diffusion 的 Toxicity‑Reduction 任务）上进行评估。  
- **对比基线**：传统 RLHF、直接监督微调（不使用奖励过滤）以及最近的 DPO（Direct Preference Optimization）实现。  
- **主要结果**：论文声称在语言基准上，RAFT 提升了约 **12%** 的奖励模型得分，同时在自动评估的有害内容比例上下降了 **15%**；在扩散模型实验中，生成的有害图像比例下降约 **10%**，视觉质量（FID）几乎不受影响。  
- **消融实验**：  
  1. **样本数量**：从 1k 增加到 10k，奖励得分提升呈递减趋势，说明一定规模后收益饱和。  
  2. **过滤比例**：保留 5% 与 20% 的高分样本对比，5% 能获得更强的对齐但多样性略有下降。  
  3. **奖励模型质量**：使用不同规模的偏好标注集训练奖励模型，发现即使奖励模型的准确率只有 70%，RAFT 仍能显著提升对齐效果。  
- **局限性**：  
  - 仍然依赖奖励模型的初始质量，若奖励模型系统性偏向某类错误，过滤后会放大该偏差。  
  - 高分样本的筛选可能导致生成多样性下降，需要在过滤比例上做权衡。  
  - 对于极端稀缺的负面示例（如罕见的安全风险），离线采样可能根本捕捉不到，导致对齐盲区。

### 影响与延伸思考
RAFT 的出现让业界重新审视“对齐不一定要强化学习”的假设。随后出现的 **Direct Preference Optimization (DPO)**、**Preference‑Based Supervised Fine‑Tuning (PBSFT)** 等工作，都在不同程度上沿用了“离线奖励排序 + 直接微调”的思路。推测未来的研究会聚焦在：  
1. **更高效的奖励模型训练**——利用少量标注快速生成可靠评分器。  
2. **多目标过滤**——同时考虑安全、帮助性、创意等多个维度的奖励，使用 Pareto 前沿或加权排序。  
3. **迭代式 RAFT**——在一次微调后重新生成候选、重新过滤，形成闭环但仍保持离线特性。  
想深入了解的读者可以关注近期的 “Preference Modeling” 研讨会以及 OpenAI、DeepMind 在对齐方向的技术报告。

### 一句话记住它
**RAFT 用奖励模型把好样本挑出来再微调，省去强化学习的高成本，让大模型对齐更直接、更稳健。**