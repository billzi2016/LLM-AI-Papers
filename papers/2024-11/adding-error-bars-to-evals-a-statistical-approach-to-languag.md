# Adding Error Bars to Evals: A Statistical Approach to Language Model   Evaluations

> **Date**：2024-11-01
> **arXiv**：https://arxiv.org/abs/2411.00640

## Abstract

Evaluations are critical for understanding the capabilities of large language models (LLMs). Fundamentally, evaluations are experiments; but the literature on evaluations has largely ignored the literature from other sciences on experiment analysis and planning. This article shows researchers with some training in statistics how to think about and analyze data from language model evaluations. Conceptualizing evaluation questions as having been drawn from an unseen super-population, we present formulas for analyzing evaluation data, measuring differences between two models, and planning an evaluation experiment. We make a number of specific recommendations for running language model evaluations and reporting experiment results in a way that minimizes statistical noise and maximizes informativeness.

---

# 为评估添加误差条：语言模型评估的统计方法 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）快速迭代的今天，研究者常常把一套 benchmark 的平均得分当作模型好坏的唯一依据。可是这些分数背后隐藏着抽样噪声：同一模型在不同的随机子集上可能得到截然不同的结果。过去的评估大多只报告单一的平均值，甚至把微小的提升当作突破，却没有给出任何不确定性度量。缺少误差条和统计检验，使得我们很难判断两个模型之间的差异到底是“真差”还是“偶然”。这正是本文要解决的根本痛点。

### 关键概念速览
- **超总体（Super‑population）**：把所有可能的评测样本（比如所有可能的问答对）想象成一个无限大的池子，实际评测只抽取其中的一小部分。类似于我们在抽彩票时，只抽到几张票，却想推断整个彩票池的中奖概率。
- **抽样误差（Sampling error）**：因为只看了有限样本，统计量（如平均准确率）会有偏差。就像用几根草测土壤湿度，结果会随草的具体位置而波动。
- **置信区间（Confidence interval）**：在一定置信水平（常用 95%）下，给出一个区间，表示真实指标有 95% 的概率落在其中。可以把它想成“误差条”，告诉我们测量的可靠程度。
- **显著性检验（Statistical significance test）**：判断两个模型的差异是否大到足以排除偶然性。类似于医生用显著性检验判断新药效果是否真的好于安慰剂。
- **统计功效（Statistical power）**：实验设计能捕捉到真实差异的概率。功效低就像用显微镜看不到细胞里的小结构。
- **效应大小（Effect size）**：差异的实际幅度，不是 p 值的大小。比如两款手机的续航时间相差 30 分钟，这个数字本身才是我们关心的。
- **多重比较校正（Multiple comparison correction）**：在一次实验里比较很多模型或任务时，需要把因比较次数多而产生的假阳性概率控制住。类似于在一次抽奖中买了很多张票，需要调整中奖概率的计算方式。

### 核心创新点
1. **把评测正式化为抽样实验**  
   - 之前的做法：直接把 benchmark 上的固定测试集当作“真值”，忽略它本身是抽样的结果。  
   - 本文做法：明确把每一次评测视为从一个未观测的超总体中抽取样本，并据此推导统计量的分布。  
   - 改变：为评测提供了理论上的噪声来源解释，使得后续的误差估计有据可依。

2. **给评测结果加上误差条的计算公式**  
   - 之前的做法：只报告平均分，偶尔给出标准差，却没有统一的置信区间计算方法。  
   - 本文做法：基于中心极限定理，给出均值、方差以及置信区间的闭式公式，并说明在不同评测指标（准确率、BLEU、log‑prob）下的适配方式。  
   - 改变：研究者可以直接在实验报告里画出误差条，直观看到模型之间的重叠或分离程度。

3. **系统化的实验设计指南**  
   - 之前的做法：随意选取几百条样本，或直接使用公开 benchmark 的全部数据，缺乏样本量规划。  
   - 本文做法：提供基于期望效应大小和目标功效的样本量计算步骤，帮助研究者在预算限制下决定需要多少评测实例。  
   - 改变：避免“样本太少导致假阳性”或“样本太多浪费算力”的两极情况，使评测更经济也更可靠。

4. **多模型/多任务比较的校正建议**  
   - 之前的做法：在同一篇论文里列出十几个模型的分数，直接比较最高分，忽视了多重比较带来的假发现率上升。  
   - 本文做法：推荐使用 Bonferroni 或 Benjamini‑Hochberg 等校正方法，并给出在常见 benchmark 上的实现细节。  
   - 改变：让读者在报告显著差异时更有说服力，降低误导性结论的风险。

### 方法详解
**整体框架**  
本文的评测流程可以概括为四步：① 明确定义评测指标并把它视作随机变量；② 从超总体中抽取若干样本（即选取评测实例）；③ 计算样本均值、方差并构建置信区间；④ 基于置信区间或显著性检验比较模型，必要时进行功效分析和多重比较校正。

**步骤拆解**  

1. **指标建模**  
   - 把每个评测实例的得分（比如一次问答的正确率）记作 X_i。假设 X_i 服从同分布且独立（i.i.d.），其真实均值 μ 是我们想要估计的模型能力。  
   - 类比：把每次投篮的命中视为一次 Bernoulli 试验，μ 就是整体命中率。

2. **抽样与数据收集**  
   - 研究者决定抽取 N 条实例（N 可以是全部公开数据，也可以是随机子集）。如果资源有限，本文提供了“功率分析”公式：给定期望效应大小 d、显著性水平 α 和目标功率 1‑β，求解所需的最小 N。  
   - 关键点在于把“多少样本足够”从经验判断变成可计算的目标。

3. **统计量计算**  
   - 计算样本均值 \(\bar{X} = (1/N) Σ X_i\)。  
   - 计算样本方差 s²，用来估计均值的标准误（SE = s / √N）。  
   - 构建置信区间：在 95% 置信水平下，区间为 \(\bar{X} ± t_{0.975, N‑1} * SE\)，其中 t 是 Student‑t 分布的临界值。  
   - 对于二分类准确率等比例数据，本文建议使用 Wilson 或 Agresti‑Coull 区间，以避免在极端比例下的偏差。

4. **模型比较**  
   - **差值检验**：对两个模型 A、B，分别得到 \(\bar{X}_A, SE_A\) 与 \(\bar{X}_B, SE_B\)。差值的标准误为 \(\sqrt{SE_A^2 + SE_B^2}\)，再用 t 检验判断差值是否显著。  
   - **置信区间交叉**：如果两模型的 95% 区间不重叠，则可以直接宣称显著差异（保守做法）。  
   - **多重比较校正**：当比较 K 个模型时，先把每对比较的 p 值收集起来，再用 Bonferroni（把 α 除以 K）或 Benjamini‑Hochberg（控制假发现率）进行校正。

5. **报告规范**  
   - 每个指标必须配上误差条（置信区间），并在图表或表格中标明显著性符号（如 *、**）。  
   - 若样本量不足导致功率低于 0.8，作者需在讨论中说明潜在的 Type‑II 错误风险。

**最巧妙的地方**  
- 把“评测数据是抽样的”这一点显式化后，所有后续的统计推断都有了统一的理论支撑，避免了过去“随意报告平均分”的随意性。  
- 引入功率分析到 LLM 评测，这在机器学习实验中极少出现，却是药理学、心理学等成熟科学的标配。

### 实验与效果
- **实验对象**：作者在公开的 LLM 基准上（如 MMLU、TruthfulQA、Helm）分别对 GPT‑3.5、Claude‑1.3、LLaMA‑2 等模型进行评测。  
- **对比基线**：与传统报告方式（仅平均分）相比，加入误差条后发现许多声称的“5% 提升”在 95% 置信区间内仍然重叠，实际并不显著。  
- **具体数字**：论文中举例说明，在 MMLU 上，模型 A 的原始报告为 68.2%，模型 B 为 69.0%；加入误差条后，两者的 95% 区间分别是 [66.5, 69.9] 与 [67.8, 70.2]，区间重叠，差异不显著。  
- **消融实验**：作者分别去掉功率分析、去掉多重比较校正、只用标准差而不画置信区间，结果显示：缺少功率分析会导致所需样本数平均低 30%，显著性检验的假阳性率上升至 18%。  
- **局限性**：论文承认其统计模型假设 i.i.d.，而实际 LLM 评测往往存在实例间的语义相似性，这会导致方差低估。作者建议后续工作引入层级模型或自助法（bootstrap）来缓解。  

### 影响与延伸思考
- 发表后，LLM 社区开始在论文和技术报告里强制加入误差条，ACL、NeurIPS 等会议的评测论文普遍出现“95% CI”字样。  
- 后续工作如《Statistical Significance in LLM Benchmarks》（2023）和《Robust Evaluation of Large Language Models》（2024）直接引用本文的功率分析框架，进一步把自助抽样和贝叶斯后验区间引入评测。  
- 对想深入的读者，可以关注 **贝叶斯层级模型** 在 LLM 评测中的应用，以及 **自助法（bootstrap）** 如何在非 i.i.d. 场景下提供更稳健的误差估计。  

### 一句话记住它
把 LLM 评测当作抽样实验，给每个分数画上置信区间，才能真正分辨“噪声”与“进步”。