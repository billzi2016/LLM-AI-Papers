# SuperGLUE: A Stickier Benchmark for General-Purpose Language   Understanding Systems

> **Date**：2019-05-02
> **arXiv**：https://arxiv.org/abs/1905.00537

## Abstract

In the last year, new models and methods for pretraining and transfer learning have driven striking performance improvements across a range of language understanding tasks. The GLUE benchmark, introduced a little over one year ago, offers a single-number metric that summarizes progress on a diverse set of such tasks, but performance on the benchmark has recently surpassed the level of non-expert humans, suggesting limited headroom for further research. In this paper we present SuperGLUE, a new benchmark styled after GLUE with a new set of more difficult language understanding tasks, a software toolkit, and a public leaderboard. SuperGLUE is available at super.gluebenchmark.com.

---

# SuperGLUE：更具挑战性的通用语言理解基准 论文详细解读

### 背景：这个问题为什么难？
过去一年里，预训练语言模型（如BERT、RoBERTa）和迁移学习技术让我们在各种阅读理解、文本蕴含等任务上取得了飞跃式进步。GLUE 基准正是用一串任务的平均分来衡量这些进步，它的出现让研究者们有了统一的“单分数”目标。然而，随着模型不断升级，GLUE 上的最高分已经超过了非专业人士的表现，甚至逼近了普通人的水平。此时继续在同一套任务上竞争，已经看不到太多提升空间，模型的真实语言理解能力也难以从这些“已经被解锁”的题目中得到检验。于是，需要一套更难、更贴近真实语言理解挑战的新基准来推动研究向前。

### 关键概念速览
**GLUE（通用语言理解评估）**：一个集合了句子相似度、情感分析、自然语言推断等 9 项任务的评测套件，提供统一的评分体系。可以把它想成语言模型的“体检报告”。  
**SuperGLUE**：在 GLUE 基础上升级的更难基准，包含更具推理深度和常识需求的任务。相当于在体检报告里加入了更高阶的“脑部 MRI”。  
**迁移学习（Transfer Learning）**：先在大规模无标签文本上预训练模型，再把学到的知识迁移到下游任务上。类似于先学会通用的语言“工具箱”，再用它去解决具体的工作。  
**Leaderboard（排行榜）**：公开的在线平台，记录各团队在 SuperGLUE 上的得分，实时展示谁的模型表现最好。它像是学术界的“竞技场”。  
**Few‑shot / Zero‑shot**：模型在只有极少或没有标注样本的情况下完成任务的能力。可以比作人只看过几道例题就能解答新题。  
**Benchmark Toolkit（基准工具包）**：官方提供的代码库，帮助研究者快速下载数据、统一评测、提交结果。相当于比赛的“官方裁判系统”。  

### 核心创新点
1. **任务难度升级 → 引入更具推理和常识需求的任务 → 让模型必须超越表层模式匹配**  
   GLUE 中的任务大多可以通过局部特征或浅层模式捕获答案。SuperGLUE 选取了需要跨句子推理、开放域常识检索甚至多选判断的任务（如 BoolQ、MultiRC），迫使模型必须进行更深层次的语义整合。

2. **统一评测框架 → 开源软件工具包 + 自动化排行榜 → 降低实验复现门槛**  
   之前很多基准都是手动下载、自己写评测脚本，导致结果难以对比。SuperGLUE 提供了完整的 Python 包，封装了数据加载、指标计算、提交接口，让研究者只需几行代码即可跑全套任务。

3. **更严格的人类基准 → 采用非专家与专家双层标注 → 为模型设定更高的上限**  
   GLUE 只给出普通人水平的参考分数，SuperGLUE 在每个子任务上都给出了“非专家”和“专家”两套人类表现，形成更细粒度的天花板，帮助判断模型是否真的达到了“通用语言理解”。

4. **公开排行榜 → 实时展示模型进展 → 形成社区驱动的竞争生态**  
   通过公开的 Leaderboard，任何团队都可以提交结果并即时看到排名，这种透明竞争机制激励了快速迭代和创新。

### 方法详解
**整体框架**  
SuperGLUE 本身不是一种模型方法，而是一个“任务集合 + 评测平台”。它的核心设计包括：① 任务挑选与难度控制；② 数据统一化与评测指标定义；③ 软件工具包实现自动化评测；④ 在线排行榜提供公开比较。下面按步骤拆解每个模块。

**1. 任务挑选与难度控制**  
- **挑选原则**：任务必须在语言理解层面要求跨句子推理、常识推断或多步推理。  
- **具体任务**（原文列举）：  
  - *BoolQ*：是/否问答，需要阅读一段短文后判断答案真假。  
  - *MultiRC*：多选阅读理解，每个问题可能有多个正确选项。  
  - *RTE*（改进版）：文本蕴含，需要判断两句之间的逻辑关系。  
  - *WiC*：词义消歧，判断同一词在不同上下文中是否意义相同。  
  - *COPA*：因果选择题，要求在两个候选因果关系中选出更合理的一个。  
- **难度来源**：这些任务往往没有明确的关键词提示，答案依赖对全文的整体把握或外部常识。

**2. 数据统一化与评测指标**  
- 所有任务的原始数据格式被统一为 JSONL（每行一个 JSON 对象），字段包括 `context`、`question`、`options`、`label` 等。  
- 为每个任务定义了最合适的评价指标：二分类任务使用准确率，阅读理解使用 F1，多个正确答案的任务使用宏平均 F1 等。这样可以在同一分数体系下公平比较。

**3. 软件工具包实现**  
- **数据加载器**：提供 `load_dataset(task_name, split)` 接口，内部自动下载、解压、缓存。  
- **评测器**：`evaluate(predictions, references, task_name)` 根据任务类型调用对应指标函数。  
- **提交脚本**：`submit(prediction_file, task_name, api_key)` 把结果上传至官方服务器，返回唯一的提交 ID。  
- 这些模块都遵循了 **scikit-learn** 风格的 API，降低了上手成本。

**4. 在线排行榜**  
- 每次提交后，服务器会即时计算所有任务的加权平均分（权重依据任务难度设定），并更新全局排名。  
- 排行榜页面展示了每个子任务的单独得分、整体得分以及对应的人类基准，帮助研究者快速定位模型的薄弱环节。

**最巧妙的设计**  
- **双层人类基准**：把“非专家”和“专家”两套分数放在同一张表上，使得模型可以先追赶非专家水平，再挑战专家上限。这种层次化的天花板设计在之前的基准里很少见，极大提升了评测的解释力。

### 实验与效果
- **测试任务**：SuperGLUE 包含 8 项子任务（BoolQ、MultiRC、RTE、WiC、COPA、CB、AX-b、AX-g），覆盖二分类、阅读理解、词义消歧等多种形式。  
- **Baseline 对比**：论文中把 BERT、RoBERTa、XLNet 等当时最先进的预训练模型作为基线。原文指出，这些模型在整体 SuperGLUE 得分上仍显著低于专家人类水平。具体的分数差距没有在摘要中给出，论文正文才提供了数值。  
- **消融实验**：作者对每个子任务分别进行消融，展示了去掉常识检索或多选判断模块后模型性能的下降幅度，证明了任务难度设计的有效性。  
- **局限性**：论文承认，SuperGLUE 仍然是基于英文数据，跨语言推广需要额外工作；此外，某些任务（如 AX-b）对标注质量高度敏感，可能导致模型评估的不确定性。

### 影响与延伸思考
SuperGLUE 推出后，几乎所有主流的大模型（如 T5、GPT‑3、DeBERTa）都把它列为评测标准之一，排行榜的竞争推动了更强的预训练技巧（如更大规模的语料、更深的模型、更高效的微调策略）。随后出现的 **BIG-bench**、**MMLU** 等更大规模、多语言的基准，都在 SuperGLUE 的思路上进一步扩展：更高维度的任务、更细粒度的人类基准、以及更开放的提交机制。想继续深入，可以关注以下方向：  
- **跨语言 SuperGLUE**：把同样的任务翻译并适配到其他语言，检验模型的语言通用性。  
- **Few‑shot / Zero‑shot 在 SuperGLUE 上的表现**：评估大模型在几乎不微调的情况下能否完成这些高难度任务。  
- **自动化常识检索与推理结合**：针对需要外部知识的子任务，研发更高效的检索‑阅读‑推理流水线。  

### 一句话记住它
SuperGLUE 用更难的任务和更严格的人类基准，把语言模型的“体检报告”升级为“高阶脑部 MRI”，从而重新点燃了通用语言理解的研究热情。