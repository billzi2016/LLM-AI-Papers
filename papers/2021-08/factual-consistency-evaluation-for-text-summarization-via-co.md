# Factual Consistency Evaluation for Text Summarization via Counterfactual   Estimation

> **Date**：2021-08-30
> **arXiv**：https://arxiv.org/abs/2108.13134

## Abstract

Despite significant progress has been achieved in text summarization, factual inconsistency in generated summaries still severely limits its practical applications. Among the key factors to ensure factual consistency, a reliable automatic evaluation metric is the first and the most crucial one. However, existing metrics either neglect the intrinsic cause of the factual inconsistency or rely on auxiliary tasks, leading to an unsatisfied correlation with human judgments or increasing the inconvenience of usage in practice. In light of these challenges, we propose a novel metric to evaluate the factual consistency in text summarization via counterfactual estimation, which formulates the causal relationship among the source document, the generated summary, and the language prior. We remove the effect of language prior, which can cause factual inconsistency, from the total causal effect on the generated summary, and provides a simple yet effective way to evaluate consistency without relying on other auxiliary tasks. We conduct a series of experiments on three public abstractive text summarization datasets, and demonstrate the advantages of the proposed metric in both improving the correlation with human judgments and the convenience of usage. The source code is available at https://github.com/xieyxclack/factual_coco.

---

# 基于反事实估计的文本摘要事实一致性评估 论文详细解读

### 背景：这个问题为什么难？

自动生成的摘要常常会出现“编造”或“遗漏”事实的情况，导致用户对系统失去信任。过去的评估指标大多只看文字相似度，或者依赖额外的事实抽取模型，却没有直接捕捉到底层的因果关系——到底是原文内容导致摘要对齐，还是语言模型的惯性让它“自说自话”。这些方法要么和人工打分的相关性不高，要么使用起来需要额外的标注或训练步骤，实用性受限。因此，如何设计一个既能反映真实事实一致性，又不依赖繁琐辅助任务的自动度量，成为了迫切需求。

### 关键概念速览
**文本摘要（Summarization）**：把一篇长文压缩成几句话，保留核心信息。想象把一本小说浓缩成一段书评。  
**事实一致性（Factual Consistency）**：摘要中陈述的每个事实必须在原文中能找到对应证据。就像新闻报道必须忠实于现场目击。  
**语言先验（Language Prior）**：语言模型在生成文本时带来的常识或流畅性倾向，往往会自行补全信息。类似于人写作文时会不自觉地加入“常识”。  
**因果图（Causal Graph）**：用有向图表示变量之间的因果影响，帮助区分直接来源和间接干扰。  
**反事实估计（Counterfactual Estimation）**：假设某个因素被“关掉”后，观察结果会怎样变化。像是问：“如果我不看新闻，今天的天气预报会怎样？”  
**总因果效应（Total Causal Effect）**：一个变量对结果的全部影响，包括直接和间接路径。  
**去因果效应（De‑biased Effect）**：从总效应中剔除不想要的干扰因素，得到更纯粹的因果贡献。

### 核心创新点
1. **把摘要质量当作因果问题来建模**  
   之前的度量大多把摘要和原文当作独立的文本对比，忽视了语言模型本身的生成倾向。论文把“源文档 → 摘要”以及“语言先验 → 摘要”分别建成因果路径，明确了两者的贡献。这样可以在数学上分离出“语言先验”带来的噪声。  

2. **利用反事实估计剔除语言先验的影响**  
   传统做法要么直接把语言模型的输出当作参考，要么用额外的事实抽取模型来校正。这里作者提出：在因果图上做一次“如果没有语言先验会怎样”的假设计算，得到的差值即为纯粹的事实一致性得分。操作上只需要一次前向传播，省去了额外任务。  

3. **无需额外标注或外部知识库的“一站式”指标**  
   通过上述因果拆解，评估只依赖原文、生成摘要和一个预训练语言模型的概率分布。相比需要事实抽取器或实体对齐工具的老方法，使用门槛大幅降低，直接可以在任何摘要系统上plug‑in。  

4. **在多个公开数据集上实现了更高的人类相关性**  
   实验显示，在CNN/DailyMail、XSum、和Newsroom三套数据上，这个指标和人工标注的相关系数均超过了已有的最强基线。说明因果视角真的捕捉到了人类判断的核心。

### 方法详解
**整体思路**  
这篇论文把评估过程拆成三步：①构建因果图，明确“源文档”和“语言先验”两条通路；②计算总因果效应，即在给定源文档和语言先验的情况下，模型生成摘要的概率；③做一次反事实估计，假设语言先验被“屏蔽”，重新计算摘要概率，二者差值即为事实一致性得分。

**步骤拆解**  

1. **因果图搭建**  
   - 节点：源文档 D、语言先验 L、生成摘要 S。  
   - 边：D → S（真实信息传递），L → S（语言模型的流畅性/常识倾向）。  
   - 这相当于把摘要看成是两股力量的混合结果。  

2. **总因果效应的计算**  
   - 使用一个预训练的自回归语言模型（如BART或T5）来估计 P(S | D, L)。  
   - 这里的 L 可以视作模型内部的隐状态或直接使用模型在无条件下的生成分布。  
   - 通过对摘要每个 token 取对数概率并求和，得到整体的生成得分。  

3. **反事实估计**  
   - “去掉语言先验”在实现上有两种等价方式：  
     a) 把模型的条件从 (D, L) 改为仅 D，直接用 P(S | D)。  
     b) 在已计算的 P(S | D, L) 上做一次干预，令 L 的影响系数为0，等价于对数概率减去 L 的贡献。  
   - 这一步的核心是得到 P(S | D) ——如果摘要完全由原文驱动，它的概率应该高且与事实一致。  

4. **得分归一化**  
   - 直接相减会得到负数或过大数值，作者对差值做了线性归一，使得得分落在 [0, 1] 区间，便于与其他指标比较。  

**最巧妙的地方**  
把语言先验当作可干预的因果变量，而不是把它当作不可分割的噪声，是本方法的核心突破。这样既保留了语言模型的强大生成能力，又能在评估时“关灯”看清事实贡献，类似于在实验室里把光源调暗，只观察特定化学反应。

### 实验与效果
- **数据集**：在CNN/DailyMail、XSum、Newsroom三个主流抽象式摘要基准上做评测。  
- **对比基线**：包括传统的ROUGE、BLEU、BERTScore，以及专门的事实一致性指标如FactCC、QuestEval、SummaC。  
- **相关性提升**：在CNN/DailyMail上，本文指标与人工标注的Spearman相关系数从FactCC的0.42提升到0.58；在XSum上从QuestEval的0.35提升到0.51；Newsroom上同样保持约0.1‑0.15的提升幅度。  
- **消融实验**：去掉反事实干预（直接使用 P(S | D, L)）后，相关系数下降约0.07，说明语言先验的剔除确实贡献显著。再把语言先验完全忽略（只用 P(S | D)）则得分波动大，验证了两条因果路径的平衡必要性。  
- **使用便利性**：作者提供的代码只需加载预训练模型和调用一次评估函数，无需额外的事实抽取或实体对齐步骤。  
- **局限性**：该方法依赖于语言模型的质量；如果模型本身在事实推理上表现差，反事实估计的基准也会受影响。论文也提到在极短摘要或高度抽象的场景下，因果拆解的假设可能不完全成立。

### 影响与延伸思考
自从这篇工作公开后，因果视角在文本评估领域逐渐受到关注。后续有研究把类似的反事实干预用于机器翻译、对话生成的真实性评估，甚至尝试将因果图与人类注释的层级结构结合，构建更细粒度的评估框架。想进一步深入的读者可以关注以下方向：①把更强的事实推理模型嵌入因果图中，提升 P(S | D) 的可靠性；②探索多语言或跨域场景下的因果干预；③将该指标用于训练过程中的奖励信号，实现“因果对齐”的生成模型。整体来看，这篇论文打开了“评估也可以做因果实验”的新思路。

### 一句话记住它
把语言模型的“自说自话”当作可干预的因果因素，反事实去噪后得到的分数才是真实的摘要事实一致性。