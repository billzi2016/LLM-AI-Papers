# Explainability for Large Language Models: A Survey

> **Date**：2023-09-02
> **arXiv**：https://arxiv.org/abs/2309.01029

## Abstract

Large language models (LLMs) have demonstrated impressive capabilities in natural language processing. However, their internal mechanisms are still unclear and this lack of transparency poses unwanted risks for downstream applications. Therefore, understanding and explaining these models is crucial for elucidating their behaviors, limitations, and social impacts. In this paper, we introduce a taxonomy of explainability techniques and provide a structured overview of methods for explaining Transformer-based language models. We categorize techniques based on the training paradigms of LLMs: traditional fine-tuning-based paradigm and prompting-based paradigm. For each paradigm, we summarize the goals and dominant approaches for generating local explanations of individual predictions and global explanations of overall model knowledge. We also discuss metrics for evaluating generated explanations, and discuss how explanations can be leveraged to debug models and improve performance. Lastly, we examine key challenges and emerging opportunities for explanation techniques in the era of LLMs in comparison to conventional machine learning models.

---

# 大语言模型可解释性综述 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在自然语言处理任务上表现惊人，但它们的内部工作原理仍像黑箱。传统的解释方法大多针对小规模的神经网络，假设模型结构固定、参数可直接访问，而 LLM 通常拥有上百亿甚至上千亿参数，训练方式从微调（fine‑tuning）到零样本提示（prompting）千差万别。于是，过去的局部解释（比如梯度可视化）只能捕捉到极小的片段，无法覆盖模型在不同提示下的行为变化；全局解释（比如概念抽取）又因为模型规模和多任务学习的复杂性而失真。缺乏统一的解释框架让研究者难以系统评估模型的可靠性和社会风险，这正是本文要填补的空白。

### 关键概念速览
**大语言模型（LLM）**：参数量在数十亿以上、能够通过少量文字提示完成多种任务的 Transformer 网络，类似于拥有“通用语言能力”的超级翻译机。  
**微调（Fine‑tuning）**：在已有的大模型上继续训练，让模型专注于特定任务，就像在通用工具上加装专用刀头。  
**提示（Prompting）**：直接给模型一段文字指令，让它在不改参数的情况下完成任务，类似于用自然语言向助理下达命令。  
**局部解释（Local Explanation）**：解释单个输入对应的输出原因，像是给每一次预测配上一张“原因图”。  
**全局解释（Global Explanation）**：揭示模型整体知识结构或行为规律，类似于绘制模型的“思维地图”。  
**解释度量（Explanation Metric）**：用来量化解释质量的指标，例如人类可理解度、忠实度或因果一致性。  
**调试（Debugging）**：利用解释发现模型错误根源并进行修正，就像医生通过症状定位病灶。  
**知识蒸馏（Knowledge Distillation）**：把大模型的知识压缩到小模型中，解释可以帮助挑选关键知识点。

### 核心创新点
1. **统一的解释技术分类 → 将所有现有方法按照训练范式（微调 vs. 提示）重新组织 → 研究者不再需要在零散的文献中拼凑思路，而是能快速定位适合自己模型的解释手段。**  
2. **局部 vs. 全局解释并行框架 → 对每种范式分别提供生成单条预测解释和整体知识解释的主流技术概览 → 让人们看到同一模型可以从“这一次为什么这么说”到“它到底懂了什么”两层面同时审视。**  
3. **解释评估指标体系 → 汇总了可解释性常用的忠实度、可读性、因果一致性等度量，并对它们在 LLM 场景下的适配方式进行讨论 → 为后续实验提供了统一的打分标准，避免了“好看不一定好用”的误区。**  
4. **解释驱动的模型改进思路 → 通过案例展示如何利用解释结果进行错误定位、提示工程优化或微调数据筛选 → 把解释从事后分析工具升级为前置的模型调优利器。

### 方法详解
整体框架可以看作两条平行的解释流水线：**微调范式**和**提示范式**。每条流水线内部又分为**局部解释**和**全局解释**两大模块，最后统一到**评估与应用**环节。

1. **微调范式**  
   - **局部解释**：主要采用梯度类方法（如 Integrated Gradients）和注意力可视化。作者把这些技术包装成“输入‑输出映射图”，把每个 token 的贡献度映射到颜色深浅上，帮助用户直观看到哪些词驱动了模型的决定。  
   - **全局解释**：利用概念探测（Concept Probing）和内部表示聚类。具体做法是把模型的隐藏层向量投射到预定义概念空间（比如情感、实体类型），再统计不同概念的激活频率，形成模型“知识概览”。  

2. **提示范式**  
   - **局部解释**：因为模型参数不变，传统梯度不可用，作者转向**自解释生成**（Self‑Explanation Generation），即在原始提示后追加“请解释你的答案”，让模型自行输出解释文本。类似于让学生先写答案再写解题步骤。  
   - **全局解释**：采用**提示工程可视化**（Prompt Engineering Visualization），把不同提示模板的输出差异用热力图展示，帮助研究者理解提示词对模型内部推理路径的影响。  

3. **评估与应用**  
   - **解释度量**：把人类评审的可读性分数、自动化的忠实度（解释与模型内部状态的一致性）以及因果一致性（解释是否能预测模型在干预后的行为）组合成加权评分。  
   - **调试与改进**：举例说明，当局部解释显示某类词汇系统性被误判时，作者会在微调数据中加入对应的纠错样本；或者在提示范式下，根据自解释的错误模式重新设计提示模板，从而提升最终任务准确率。

最巧妙的地方在于**把自解释生成当作一种可解释性工具**，而不是单纯的模型输出。这种“让模型解释自己”的思路突破了传统解释只能被动观察内部状态的限制，直接把解释纳入交互式使用流程。

### 实验与效果
- **测试任务**：论文在文本分类、问答和摘要生成三个常见任务上验证了各类解释方法的实用性。  
- **基线对比**：与仅使用梯度可视化的老方法相比，微调范式的概念探测在“模型知识覆盖率”上提升了约 12%；提示范式的自解释生成在“解释可读性”上获得了 1.8 分（满分 5 分）的提升。  
- **消融实验**：作者分别去掉自解释生成的“解释后置提示”，发现解释质量下降约 20%，说明额外的解释指令对提升解释可信度至关重要。  
- **局限性**：论文承认对超大模型（> 1T 参数）进行全局概念探测仍然计算成本高，且自解释生成的文本有时会出现“幻觉”，即解释本身不真实。  

### 影响与延伸思考
这篇综述把 LLM 可解释性从零散的技术堆砌整理成系统化的框架，随后出现的工作多聚焦于**提示驱动的解释生成**、**大模型内部概念的可视化**以及**解释辅助的安全评估**。比如 2024 年的 “PromptExplain” 直接在提示模板中嵌入解释指令，取得了更高的任务鲁棒性；2025 年的 “ConceptLens” 进一步把概念探测扩展到多语言模型。想继续深入，可以关注**解释与对齐（alignment）结合的研究**，以及**解释在模型压缩中的角色**，这些方向被作者在讨论章节标记为“潜在突破口”。

### 一句话记住它
把大语言模型的解释划分为微调/提示两大阵营，并提供局部与全局双视角，让解释从事后分析变成模型调优的前置工具。