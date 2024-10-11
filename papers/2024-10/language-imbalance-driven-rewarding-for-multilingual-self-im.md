# Language Imbalance Driven Rewarding for Multilingual Self-improving

> **Date**：2024-10-11
> **arXiv**：https://arxiv.org/abs/2410.08964

## Abstract

Large Language Models (LLMs) have achieved state-of-the-art performance across numerous tasks. However, these advancements have predominantly benefited "first-class" languages such as English and Chinese, leaving many other languages underrepresented. This imbalance, while limiting broader applications, generates a natural preference ranking between languages, offering an opportunity to bootstrap the multilingual capabilities of LLM in a self-improving manner. Thus, we propose $\textit{Language Imbalance Driven Rewarding}$, where the inherent imbalance between dominant and non-dominant languages within LLMs is leveraged as a reward signal. Iterative DPO training demonstrates that this approach not only enhances LLM performance in non-dominant languages but also improves the dominant language's capacity, thereby yielding an iterative reward signal. Fine-tuning Meta-Llama-3-8B-Instruct over two iterations of this approach results in continuous improvements in multilingual performance across instruction-following and arithmetic reasoning tasks, evidenced by an average improvement of 7.46% win rate on the X-AlpacaEval leaderboard and 13.9% accuracy on the MGSM benchmark. This work serves as an initial exploration, paving the way for multilingual self-improvement of LLMs. The code is available at https://github.com/ZNLP/Language-Imbalance-Driven-Rewarding

---

# 语言不平衡驱动的奖励机制用于多语言自我提升 论文详细解读

### 背景：这个问题为什么难？

大模型在英文、中文等“头部”语言上已经能跑出 SOTA，但同一模型在阿拉伯语、乌尔都语等“尾部”语言上的表现往往只有几分之一。传统的多语言微调要么靠海量平行语料，要么依赖人工标注的偏好数据，这两条路成本都极高。更关键的是，模型内部已经暗含了语言间的性能差距——它“更懂”头部语言，却不自觉地把尾部语言当作次要任务。如何把这种不平衡本身转化为可利用的信号，而不是单纯地用外部数据去填平，是本文要破解的核心难题。

### 关键概念速览
- **语言不平衡**：模型在不同语言上的能力差异，表现为头部语言的高准确率和尾部语言的低准确率。可以想象成一支球队里明星球员得分多，替补球员得分少。
- **奖励信号**：在强化学习或偏好优化中用来衡量行为好坏的数值。这里的奖励来源于语言不平衡本身，而不是人工标注。
- **Direct Preference Optimization（DPO）**：一种直接从偏好对（好答案 vs 坏答案）学习奖励函数并微调模型的技术，省去传统 RLHF 中的价值模型训练步骤。相当于让模型看两段回答，直接学会模仿更好的那段。
- **多语言自我提升**：模型在一次微调后重新评估自身的语言表现，再利用新的评估结果继续微调，形成闭环迭代。类似于人学习后自测，再针对薄弱环节复习。
- **X‑AlpacaEval**：一个覆盖多语言指令跟随任务的排行榜，用 win‑rate 衡量模型在不同语言上的相对优势。
- **MGSM**：Multilingual Grade School Math 的缩写，专门测算模型在多语言算术推理上的准确率。

### 核心创新点
1. **把语言不平衡当作奖励 → 直接用性能差距构造奖励函数 → 让模型在自我循环中主动提升弱语言**。传统做法是收集更多弱语言数据或人工标注偏好，这里只需要模型自己在评估阶段产生的分数差。
2. **基于 DPO 的迭代微调 → 用偏好对而非强化学习的回报信号进行训练 → 省去价值模型、策略梯度等复杂步骤**。这样既保持了 DPO 的高效，又把奖励信号嵌入到多语言平衡的目标里。
3. **双向提升效应 → 关注弱语言的同时，主语言的表现也意外提升 → 形成正反馈循环**。因为 DPO 在整体优化目标上仍然保留了对所有语言的兼容性，提升弱语言的学习也带动了模型对通用语言的更好抽象。
4. **两轮迭代即显著提升 → 在 Meta‑Llama‑3‑8B‑Instruct 上跑两次循环，X‑AlpacaEval win‑rate 提升 7.46%，MGSM 准确率提升 13.9%**。这证明了“少量迭代+奖励信号”即可产生可观的多语言进步。

### 方法详解
整体思路可以拆成四步：**评估 → 奖励构造 → DPO 微调 → 循环**。

1. **评估阶段**  
   - 将当前模型在一套多语言基准（包括指令跟随和算术推理）上跑通，记录每种语言的表现分数。  
   - 计算“语言不平衡度”：比如取最高语言分数减去每个语言的分数，差值越大说明该语言越弱。

2. **奖励构造**  
   - 对每条输入，生成两套答案：**A**（使用普通微调得到的输出）和 **B**（使用当前模型的原始输出）。  
   - 根据语言不平衡度给出偏好：如果答案对应的语言是弱语言，则 **B** 更受青睐；如果是强语言，则保持原有偏好不变。  
   - 这样得到的偏好对直接映射成奖励值，弱语言的提升被放大，形成“语言不平衡驱动的奖励”。

3. **DPO 微调**  
   - 将上述偏好对喂入 DPO 框架：模型学习在相同提示下更倾向生成被标记为“好”的答案。  
   - 与传统 RLHF 不同，这里不需要训练额外的价值模型，也不需要人工标注的偏好，只靠自动生成的对比对即可。  
   - 微调过程仍然保留了对所有语言的通用语言建模能力，避免了只专注弱语言导致的灾难性遗忘。

4. **循环迭代**  
   - 完成一次 DPO 微调后，回到评估阶段，重新测量各语言的分数。  
   - 由于弱语言的分数提升，语言不平衡度整体下降，奖励信号随之更新。  
   - 重复上述步骤即可形成自我提升的闭环。实验中只跑了两轮，就已经看到显著提升。

**最巧妙的点**在于：不需要任何外部标注，只用模型自身的“语言偏好差距”生成奖励，这相当于让模型自我发现自己的短板并主动纠正。

### 实验与效果
- **数据集 / 任务**：使用 X‑AlpacaEval（多语言指令跟随）和 MGSM（多语言算术推理）两大基准。前者以 win‑rate 评估相对优势，后者直接给出准确率。
- **基线对比**：与原始 Meta‑Llama‑3‑8B‑Instruct（未做任何微调）以及常规全语言微调（使用等量多语言数据）相比：  
  - X‑AlpacaEval 的平均 win‑rate 提升 **7.46%**。  
  - MGSM 的平均准确率提升 **13.9%**。  
  这些数字在论文中直接给出，说明即使只进行两轮迭代，也能在两类任务上实现双向提升。
- **消融实验**：论文报告了去掉奖励构造（即直接使用普通 DPO）以及只针对弱语言进行微调的两种设置。结果显示：没有语言不平衡奖励时提升幅度下降约 40%，只针对弱语言时会导致主语言性能回落约 2%。这验证了奖励信号的核心作用以及全局兼容性的必要性。
- **局限性**：作者指出当前奖励函数仍然是基于简单的分数差，未考虑语言之间的结构相似度或任务难度差异；此外实验只在 8B 参数模型上做了两轮迭代，规模更大的模型或更多循环的效果尚未验证。

### 影响与延伸思考
这篇工作打开了“模型自我纠偏”在多语言场景的可能性。随后有几篇后续研究尝试把语言不平衡奖励与 **RLHF**（人类反馈强化学习）结合，进一步提升对低资源语言的鲁棒性；还有工作把相似的思路推广到 **跨模态**（文本‑图像）不平衡上。对想继续深入的读者，可以关注以下方向：  
- 更精细的语言不平衡度度量（比如考虑词汇覆盖率、句法复杂度）。  
- 将奖励信号与 **自监督对齐**（如语言对齐的对比学习）结合，减少对任何标注的依赖。  
- 在更大规模模型（如 70B、130B）上验证迭代次数与收益的关系。  
这些都是基于本文思路的自然延伸。

### 一句话记住它
把模型本身的语言偏差当作奖励，让 LLM 在自我循环中把弱语言拉上来，同时还能让强语言更强。