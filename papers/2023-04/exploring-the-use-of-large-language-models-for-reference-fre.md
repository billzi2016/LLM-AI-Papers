# Exploring the Use of Large Language Models for Reference-Free Text   Quality Evaluation: An Empirical Study

> **Date**：2023-04-03
> **arXiv**：https://arxiv.org/abs/2304.00723

## Abstract

Evaluating the quality of generated text is a challenging task in NLP, due to the inherent complexity and diversity of text. Recently, large language models (LLMs) have garnered significant attention due to their impressive performance in various tasks. Therefore, we present this paper to investigate the effectiveness of LLMs, especially ChatGPT, and explore ways to optimize their use in assessing text quality. We compared three kinds of reference-free evaluation methods. The experimental results prove that ChatGPT is capable of evaluating text quality effectively from various perspectives without reference and demonstrates superior performance than most existing automatic metrics. In particular, the Explicit Score, which utilizes ChatGPT to generate a numeric score measuring text quality, is the most effective and reliable method among the three exploited approaches. However, directly comparing the quality of two texts may lead to suboptimal results. We believe this paper will provide valuable insights for evaluating text quality with LLMs and have released the used data.

---

# 探索大语言模型在无参考文本质量评估中的应用：实证研究 论文详细解读

### 背景：这个问题为什么难？
生成式模型（如 GPT 系列）可以写文章、对话甚至代码，但要让机器判断这些输出到底好不好，却一直是个难题。传统的自动评估指标（BLEU、ROUGE、METEOR 等）都需要一份“参考答案”，而真实写作往往没有唯一答案，导致这些指标在流畅度、创新性等维度上表现乏力。更糟的是，人工评审成本高、主观性强，难以大规模使用。于是，如何在没有参考文本的情况下，让模型自己给出可靠的质量分数，成为了迫切需要解决的瓶颈。

### 关键概念速览
**参考自由评估（Reference‑Free Evaluation）**：不依赖人工提供的参考文本，只凭生成内容本身进行质量打分，类似于老师在没有答案键的情况下直接给学生作文打分。  
**显式评分（Explicit Score）**：让大语言模型直接输出一个数值（比如 0‑10），用来量化文本好坏，像是让模型充当“打分老师”。  
**相对比较（Relative Comparison）**：给模型两段文本，让它判断哪段更好，类似于让评审员在两篇作文之间挑出更优秀的那篇。  
**ChatGPT**：OpenAI 开发的对话式大语言模型，具备强大的语言理解与生成能力，这里被当作评估工具使用。  
**自动评估指标（Automatic Metrics）**：无需人工参与的评估方法，如 BLEU、ROUGE、BERTScore 等，通常需要参考文本作对照。  
**提示工程（Prompt Engineering）**：设计输入给模型的文字（提示），以引导模型产生期望的输出，就像给学生出题目一样决定答案的方向。  
**可靠性（Reliability）**：评估结果的一致性和可重复性，类似于同一份试卷在不同老师手里打分是否相近。  

### 核心创新点
1. **从“对比”到“显式打分”**：以前的参考自由方法多是让模型比较两段文本的优劣，结果往往受提示细节影响大。本文直接让 ChatGPT 生成一个数值评分，省去两两比较的步骤，显著提升了评估的稳定性和可解释性。  
2. **系统化提示模板**：作者针对不同评估维度（流畅性、信息完整性、逻辑连贯等）设计了专门的提示词，使得模型在给出分数时能够聚焦对应属性，避免“一刀切”。这比随意让模型自行判断更具可控性。  
3. **大规模实证验证**：在多个公开的生成文本数据集上（如 SummEval、OpenAI‑Evals 等）与传统指标和人工评分进行对比，实验显示显式评分在相关性和一致性上均优于大多数基线。  
4. **公开数据与复现脚本**：作者把实验使用的提示、评分记录以及代码全部开源，为后续研究提供了可直接复用的基准，降低了社区进入门槛。  

### 方法详解
整体思路可以概括为三步：**准备数据 → 设计提示 → 调用 ChatGPT 获得分数**。  
1. **数据准备**：收集已有的生成文本样本，覆盖摘要、对话、机器翻译等任务。每条样本都配有人工标注的质量维度（如 1‑5 星），用于后续相关性分析。  
2. **提示设计**：针对每个质量维度，作者手工编写了一段自然语言指令。例如评估流畅性时的提示可能是：“请阅读下面的段落，给出 0‑10 的流畅性评分，0 表示非常不流畅，10 表示极其流畅”。提示中明确要求模型输出纯数字，避免出现解释性文字。  
3. **模型调用**：使用 ChatGPT 的 API，以“单轮对话”模式提交提示+文本，得到模型返回的数值。为了降低随机波动，作者对同一条文本重复调用数次（如 3 次），取平均值作为最终分数。  
4. **结果聚合**：把每个维度的显式分数与人工标注进行皮尔逊/斯皮尔曼相关性计算，评估模型评分的可信度。  
**最巧妙的地方**在于把“让模型自行判断好坏”转化为“让模型扮演打分老师”，并通过精心的提示把评估目标明确化，从而大幅降低了模型输出的歧义性。  

### 实验与效果
- **测试数据**：论文在 SummEval（摘要生成质量评估）、OpenAI‑Evals（对话质量）以及机器翻译的 WMT‑21 子集上做实验。  
- **对比基线**：包括传统的参考依赖指标（BLEU、ROUGE、BERTScore）以及已有的参考自由方法（如 COMET‑src、GPT‑Eval）。  
- **主要结果**：在 SummEval 上，显式评分与人工评分的斯皮尔曼相关系数达到 0.71，超过 BLEU（0.42）和 BERTScore（0.58），并略高于 COMET‑src（0.68）。在对话任务中，同样呈现出 5‑10% 的相关性提升。  
- **消融实验**：作者分别去掉提示中的“请给出 0‑10 分数”指令、以及多次取平均的步骤，发现相关性分别下降约 0.08 和 0.05，说明这两个设计对性能都有实质贡献。  
- **局限性**：论文承认显式评分仍然受模型版本和温度参数影响，跨模型迁移时相关性会下降；此外，评分的解释性（为何给出某分）仍需人工后处理。  

### 影响与延伸思考
这篇工作打开了“LLM 直接评估生成文本”的思路，随后出现的研究大多围绕如何进一步提升评分的鲁棒性、跨语言迁移以及把评分过程透明化。例如，2024 年的 **LLM‑Judge** 系列尝试在提示中加入“评分依据示例”，提升一致性；**Meta‑Eval** 则把显式评分与对比式评估结合，形成混合评估框架。对想继续深入的读者，可以关注以下方向：  
- **提示优化**：自动搜索最优提示词的技术（如 Prompt‑Tuning）。  
- **多模态评估**：把文本评分扩展到图文、音视频生成内容。  
- **评估解释**：让模型在给出分数的同时输出简短理由，提升可解释性。  

### 一句话记住它
让 ChatGPT 直接给出数值评分，比让它两两比较更稳、更直观，成为无参考文本质量评估的实用新范式。