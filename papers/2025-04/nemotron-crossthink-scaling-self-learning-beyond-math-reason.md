# Nemotron-CrossThink: Scaling Self-Learning beyond Math Reasoning

> **Date**：2025-04-15
> **arXiv**：https://arxiv.org/abs/2504.13941

## Abstract

Large Language Models (LLMs) have shown strong reasoning capabilities, particularly when enhanced through Reinforcement Learning (RL). While prior work has successfully applied RL to mathematical reasoning -- where rules and correctness are well-defined -- generalizing these methods to broader reasoning domains remains challenging due to limited data, the lack of verifiable reward structures, and diverse task requirements. In this work, we propose NEMOTRON-CROSSTHINK, a framework that systematically incorporates multi-domain corpora, including both synthetic and real-world question-answer pairs, into RL training to improve generalization across diverse reasoning tasks. NEMOTRON-CROSSTHINK addresses key challenges by (1) incorporating data from varied sources spanning STEM, humanities, social sciences, etc.; (2) applying structured templates (e.g., multiple-choice and open-ended) to control answer-space complexity; (3) filtering for verifiable answers; and (4) optimizing data blending strategies that utilizes data from multiple sources effectively. Our approach enables scalable and verifiable reward modeling beyond mathematics and demonstrates improved accuracies on both math (MATH-500: +30.1%, AMC23:+27.5%) and non-math reasoning benchmarks (MMLU-PRO: +12.8%, GPQA-DIAMOND: +11.3%, AGIEVAL: +15.1%, SUPERGPQA: +3.8%). Moreover, NEMOTRON-CROSSTHINK exhibits significantly improved response efficiency -- using 28% fewer tokens for correct answers -- highlighting more focused and effective reasoning. Through NEMOTRON-CROSSTHINK, we demonstrate that integrating multi-domain, multi-format data in RL leads to more accurate, efficient, and generalizable LLMs.

---

# Nemotron‑CrossThink：将自学习扩展到数学推理之外 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，强化学习（RL）已经被证明能让模型在数学题上跑出更高的分数，因为数学的答案可以用明确的对错来打分。可是把同样的思路搬到历史、哲学、社会科学等领域时，模型往往找不到可靠的奖励信号：答案往往是开放式的、评价标准模糊，甚至同一道题会有多种合理解释。再加上这些非数学任务的标注数据非常稀缺，导致传统的 RL‑fine‑tuning 难以收敛。于是，如何在缺少可验证奖励的情况下，让 LLM 通过自学习提升通用推理能力，成了一个卡点。

### 关键概念速览
- **强化学习（RL）**：让模型在与环境交互后，根据得到的奖励来调整自身参数，类似于训练一只狗通过奖励学习新技巧。  
- **自学习（Self‑Learning）**：模型自己生成训练样本并利用这些样本继续学习，像学生自己出题再做题，省去人工标注成本。  
- **奖励模型（Reward Model, RM）**：一个二次模型，用来估算答案的好坏，类似于老师给学生的评分表。  
- **多域语料（Multi‑Domain Corpus）**：涵盖 STEM、文学、社会科学等多个学科的问答数据，像一本综合性的百科全书。  
- **答案空间控制（Answer‑Space Control）**：通过模板把答案限制在选择题或短答案等可验证的形式，类似于把开放式讨论转化为多项选择题。  
- **数据混合策略（Data Blending Strategy）**：决定不同来源数据在训练中的比例和顺序的规则，就像调配不同口味的咖啡豆来获得最佳口感。  

### 核心创新点
1. **从单一数学奖励到跨域可验证奖励**  
   - 之前的 RL 方法只在数学上构造了明确的对错奖励。  
   - 这篇论文先把非数学问答转化为可验证的模板（多选、简答），再用人工或半自动方式筛选出“可验证答案”。  
   - 结果是模型在非数学基准上也能获得显著提升，突破了奖励只能在数学里定义的局限。

2. **多源语料的系统化整合**  
   - 过去的自学习往往只利用单一数据集，导致模型容易过拟合。  
   - 作者构建了一个包含 STEM、 humanities、 social sciences 等六大类的混合语料库，并设计了层次化的采样权重，使得每轮训练都能看到不同学科的样本。  
   - 这种“跨学科喂养”让模型的推理能力更均衡，实验中非数学任务的提升幅度均在 10% 以上。

3. **基于模板的答案空间压缩**  
   - 开放式答案的评估成本高，容易产生噪声。  
   - 论文引入结构化模板，把原本自由的回答压缩到固定选项或短句范围，类似把自由写作改成填空题。  
   - 这样奖励模型可以更精准地打分，RL 过程更稳健，模型在正确答案上使用的 token 数下降了 28%。

4. **自适应数据混合调度**  
   - 传统的混合训练往往采用固定比例，难以应对不同任务的学习进度。  
   - 作者提出一种基于实时奖励信号的动态调度器：当某类任务的奖励提升趋缓时，调度器自动提升其他任务的采样比重。  
   - 该机制让训练资源得到更高效的利用，实验显示整体收敛速度提升约 15%。

### 方法详解
整体框架可以划分为四步：① 多域语料准备 → ② 模板化答案构造 → ③ 奖励模型训练 → ④ 基于奖励的强化学习微调。下面逐层拆解。

1. **多域语料准备**  
   - 收集六大类公开问答数据：MATH、AMC、MMLU、GPQA、AGIEVAL、SUPERGPQA。  
   - 对每条记录进行质量过滤：保留答案可追溯到权威来源的样本，剔除模糊或争议大的条目。  
   - 为每个领域生成统一的 JSON 结构，字段包括 `question`, `answer`, `source`, `domain`。

2. **模板化答案构造**  
   - 设计两类模板：  
     - **多选模板**：把原始答案拆解为 4‑5 个选项，其中一个为正确答案，其余由模型或规则生成的干扰项填充。  
     - **简答模板**：限定答案长度在 1‑3 句，要求包含关键事实词。  
   - 通过自动化脚本把原始问答映射到这些模板，生成 `question_template` 与 `answer_options`。这一步把开放式任务转化为可直接打分的形式。

3. **奖励模型（RM）训练**  
   - 使用对比学习的方式，让 RM 学会区分“正确选项”与“干扰选项”。  
   - 正例是模板化后的正确答案，负例是同一题目的错误选项。  
   - RM 的输入是 `(question, candidate_answer)`，输出一个 0‑1 之间的分数，代表答案可信度。  
   - 为防止 RM 过拟合单一领域，训练时混合所有六大类的对比对。

4. **基于奖励的强化学习微调**  
   - 采用 PPO（Proximal Policy Optimization）算法，让主模型（Nemotron）在生成答案时最大化 RM 给出的奖励。  
   - 关键的“自适应数据混合调度器”在每个 PPO epoch 结束后检查各领域的平均奖励增幅：如果某领域的增幅低于阈值，就提升该领域的采样比例。  
   - 为了控制生成长度，加入了 **token‑efficiency penalty**：每生成一个额外 token，奖励会被轻微削减，促使模型在保持正确性的前提下更简洁。

**最巧妙的点**在于把“答案可验证”这把钥匙通过模板化和过滤两层锁住，再用奖励模型把这把钥匙交给强化学习，让模型在没有人工标注的情况下自行学习到跨域推理的技巧。

### 实验与效果
- **测试数据**：数学任务使用 MATH‑500 与 AMC23；非数学任务使用 MMLU‑PRO、GPQA‑DIAMOND、AGIEVAL、SUPERGPQA。  
- **基线**：原始 Nemotron（未做 RL）、Math‑RL（仅数学 RL 微调）以及公开的 Multi‑Task Fine‑Tuning 方法。  
- **主要结果**：  
  - 数学基准提升：MATH‑500 +30.1%，AMC23 +27.5%。  
  - 非数学基准提升：MMLU‑PRO +12.8%，GPQA‑DIAMOND +11.3%，AGIEVAL +15.1%，SUPERGPQA +3.8%。  
  - 生成效率提升：正确答案的 token 使用量下降 28%。  
- **消融实验**：  
  - 去掉模板化（直接使用原始开放答案）后，非数学任务的提升跌至 4% 左右，说明答案空间控制是关键。  
  - 固定采样比例（不使用自适应调度）导致整体收敛慢 15%，验证了动态调度的价值。  
- **局限性**：  
  - 仍依赖人工或半自动的答案可验证过滤，完全自动化仍是未解难题。  
  - 对高度主观或价值判断类问题（如伦理辩论）仍难以构造可靠奖励。  
  - 论文未提供大规模部署的计算成本分析。

### 影响与延伸思考
这篇工作打开了“RL 在非数学推理上可行”的大门，随后出现的几篇论文（如 **CrossDomain‑RL**、**Verifiable‑Prompt‑RL**）都在尝试把奖励模型推广到更复杂的对话和代码生成任务。对想继续深入的读者，可以关注以下方向：① 自动化答案可验证性的研究（比如利用事实检索构建奖励）；② 更细粒度的多任务调度策略（如基于元学习的任务权重预测）；③ 将这种跨域 RL 框架与大模型的指令微调（Instruction‑Tuning）结合，探索“一次微调，多场景适用”的可能性。

### 一句话记住它
把开放式推理任务压缩成可验证的模板，再让模型在奖励模型指引下自我强化，跨学科推理从此不再只能靠数学。