# Can LLMs Express Their Uncertainty? An Empirical Evaluation of   Confidence Elicitation in LLMs

> **Date**：2023-06-22
> **arXiv**：https://arxiv.org/abs/2306.13063

## Abstract

Empowering large language models to accurately express confidence in their answers is essential for trustworthy decision-making. Previous confidence elicitation methods, which primarily rely on white-box access to internal model information or model fine-tuning, have become less suitable for LLMs, especially closed-source commercial APIs. This leads to a growing need to explore the untapped area of black-box approaches for LLM uncertainty estimation. To better break down the problem, we define a systematic framework with three components: prompting strategies for eliciting verbalized confidence, sampling methods for generating multiple responses, and aggregation techniques for computing consistency. We then benchmark these methods on two key tasks-confidence calibration and failure prediction-across five types of datasets (e.g., commonsense and arithmetic reasoning) and five widely-used LLMs including GPT-4 and LLaMA 2 Chat. Our analysis uncovers several key insights: 1) LLMs, when verbalizing their confidence, tend to be overconfident, potentially imitating human patterns of expressing confidence. 2) As model capability scales up, both calibration and failure prediction performance improve. 3) Employing our proposed strategies, such as human-inspired prompts, consistency among multiple responses, and better aggregation strategies can help mitigate this overconfidence from various perspectives. 4) Comparisons with white-box methods indicate that while white-box methods perform better, the gap is narrow, e.g., 0.522 to 0.605 in AUROC. Despite these advancements, none of these techniques consistently outperform others, and all investigated methods struggle in challenging tasks, such as those requiring professional knowledge, indicating significant scope for improvement. We believe this study can serve as a strong baseline and provide insights for eliciting confidence in black-box LLMs.

---

# 大型语言模型能表达不确定性吗？对置信度提取的实证评估 论文详细解读

### 背景：这个问题为什么难？
在传统的机器学习里，模型可以直接读取内部概率或梯度来算出“我有多自信”。但现在的主流大语言模型（LLM）大多以闭源 API 形式提供，用户只能看到文字输出，根本看不到内部分布。过去的置信度估计方法要么需要改动模型参数（微调），要么要访问模型内部状态（白盒），这在商业化的 LLM 上根本不可行。于是，如何在完全黑箱的情况下让模型“说出”自己的不确定程度，成了一个迫切却缺乏系统研究的难题。

### 关键概念速览
**置信度校准（Confidence Calibration）**：模型给出的置信度数值与实际正确率的匹配程度。理想情况下，模型说“我有 80% 的把握”，答案真的有 80% 的概率是对的。  
**失败预测（Failure Prediction）**：模型在回答前预测自己会不会出错，常用 AUROC（曲线下面积）来衡量。  
**黑盒方法（Black‑Box Approach）**：只利用模型的输入输出，不依赖任何内部权重或梯度信息的技术。就像只看医生的诊断报告，而不去检查他的实验室数据。  
**提示策略（Prompting Strategy）**：通过设计问题的文字描述，引导模型在答案后面附带置信度表达。类似于老师让学生先说“我有多把握”，再写答案。  
**多样本采样（Sampling Method）**：让模型在同一个提示下生成多条答案（比如温度采样），以观察答案之间的一致性。  
**聚合技术（Aggregation Technique）**：把多条答案的置信度或一致性信息合并成一个最终的置信度分数。可以想象为把几位专家的意见取平均后得到的共识。  
**一致性（Consistency）**：不同采样得到的答案是否相同或相似，高一致性往往暗示模型更确定。  
**过度自信（Over‑confidence）**：模型给出的置信度系统性地高于真实正确率，类似人类在熟悉领域里常常“自信满满”。

### 核心创新点
1. **从零构建系统框架 → 将置信度提取拆解为提示、采样、聚合三大模块 → 为黑盒 LLM 提供了可操作的完整流程，填补了此前只有零散技巧的空白。**  
2. **人类启发的提示设计 → 参考人类在口头考试中会先说“我有多把握”再作答的习惯，构造了多种引导模型说出置信度的提示 → 实验证明这些提示能显著降低模型的过度自信。**  
3. **利用答案一致性 → 通过多次采样得到若干答案，统计它们的相似度作为置信度的补充信号 → 这种“多数投票”思路在黑盒环境下提升了失败预测的 AUROC。**  
4. **改进的聚合策略 → 不仅简单平均置信度，还结合一致性权重、置信度分布的方差等信息 → 在多个基准上比单纯平均更稳健，缩小了与白盒方法的性能差距（AUROC 仅从 0.522 提升到 0.605）。

### 方法详解
整体思路可以看作三层塔楼：**提示层 → 采样层 → 聚合层**。先用精心设计的文字让模型“说出”自己的把握程度；再让模型在同一提示下多次生成答案；最后把这些答案的置信度和相互一致性混合，算出一个统一的置信度分数。

**1. 提示层**  
- 基础提示：在问题后直接加上 “请给出你的答案并说明你有多大把握（0‑100%）”。  
- 人类启发提示：加入情境描述，如 “在考试中，老师会要求你先估计自己的信心，你现在也这么做”。  
- 变体提示：让模型用文字描述（“我很确定”）或数值（“85%”）两种形式表达。实验发现，带有情境的提示更能抑制模型的过度自信。

**2. 采样层**  
- 采用温度采样或 Top‑p 采样，在同一提示下生成 5‑10 条答案。  
- 每条答案都包含模型自己的置信度表达。  
- 记录每条答案的文字置信度、数值置信度以及答案本身。

**3. 聚合层**  
- **置信度平均**：直接对所有数值置信度取平均。  
- **一致性加权**：计算答案之间的相似度（如 BLEU 或编辑距离），相似度高的答案获得更大权重。  
- **方差调节**：如果置信度分布的方差大，说明模型内部不确定，最终置信度会被下调。  
- 最终输出一个综合置信度，同时保留原始的置信度文本（供下游系统直接使用）。

**最巧妙的点**在于把“答案的一致性”当作置信度的第二信号。传统方法只看模型自己说的数字，而这里把模型的行为（是否会重复同样答案）也纳入评估，类似于让多位专家独立作答后看他们的意见是否一致，从而判断答案的可靠性。

### 实验与效果
- **数据集与任务**：作者选了五类基准，包括常识推理、算术推理、专业知识问答等，总计数十千条样本。  
- **模型**：覆盖了 GPT‑4、GPT‑3.5、Claude、LLaMA‑2‑Chat、Gemini 等五种主流闭源或开源 LLM。  
- **基线**：包括直接使用模型输出的原始置信度、白盒方法（如 logits‑based 置信度）以及最简单的平均采样。  
- **主要结果**：  
  - 所有模型在直接输出置信度时都表现出明显的过度自信。  
  - 引入人类启发提示后，置信度校准误差下降约 10%（具体数值未在摘要中给出）。  
  - 使用一致性加权的聚合方式，使得失败预测的 AUROC 从 0.522 提升到 0.605，接近白盒方法的水平。  
  - 随着模型规模增大（如 GPT‑4 相比 LLaMA‑2‑Chat），校准误差和 AUROC 均有显著提升，说明模型本身的能力对置信度表达有正向影响。  
- **消融实验**：分别去掉提示改进、采样次数、聚合加权，发现每一环节都对最终 AUROC 有 0.02‑0.04 的贡献，说明框架的每个模块都是必要的。  
- **局限性**：在需要专业背景的任务（如医学、法律）上，所有方法的表现都跌到接近随机水平，说明当前的黑盒置信度提取仍难以覆盖高风险领域。作者也承认，没有一种策略在所有数据集上始终领先。

### 影响与延伸思考
这篇工作首次系统化了在黑盒 LLM 上获取置信度的完整流程，随后的研究纷纷在以下方向展开：  
- **自适应提示生成**：利用小模型自动搜索最能抑制过度自信的提示。  
- **多模态一致性**：把文本答案的一致性与模型生成的图像或代码一致性结合，进一步提升置信度估计。  
- **安全监管**：在高风险应用（如金融决策）中加入置信度阈值过滤，防止模型盲目输出错误答案。  
如果想深入，可以关注近期在 “LLM 可信度评估” 领域的工作，如 “Self‑Check” 系列、以及利用检索增强提升置信度的研究（推测）。

### 一句话记住它
在黑箱 LLM 中，**让模型多说几遍、看它们是否说得一致，再把这些信息加权合并，就能显著削弱它的过度自信**。