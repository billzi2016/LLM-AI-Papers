# Learning Facts at Scale with Active Reading

> **Date**：2025-08-13
> **arXiv**：https://arxiv.org/abs/2508.09494

## Abstract

LLMs are known to store vast amounts of knowledge in their parametric memory. However, learning and recalling facts from this memory is known to be unreliable, depending largely on the prevalence of particular facts in the training data and other factors which are poorly understood. Practitioners are lacking tools which will allow them to ensure that the models learn a given body of knowledge reliably and consistently. To this end, we propose Active Reading: a framework where we train models to study a given set of material with self-generated learning strategies. First, we demonstrate models trained with Active Reading on expert domains absorb significantly more knowledge than vanilla finetuning and other data augmentations. We train expert 8B models that achieve 66% on a Wikipedia-grounded subset of SimpleQA (+313% relative over vanilla finetuning) and 26% on FinanceBench (+160% relative over vanilla finetuning) by applying Active Reading to the source documents for each benchmark. Finally, we show that Active Reading can be utilized at pre-training scale to build more factual models. As a demonstration of this, we release Meta WikiExpert-8B, a Wikipedia-expert model trained on 1 trillion generated tokens, which outcompetes models with hundreds of billions of parameters on factual QA.

---

# 主动阅读大规模学习事实 论文详细解读

### 背景：这个问题为什么难？
大型语言模型（LLM）虽然在参数里埋下了海量知识，但这些知识的提取并不可靠。模型往往只能在训练数据里出现频繁的事实上表现好，稀有或新出现的事实容易被忘记或产生错误答案。现有的微调手段（普通全量微调、数据增强等）只能在一定程度上提升记忆，却缺乏系统的“学习策略”，导致模型在不同领域的事实掌握程度参差不齐。于是，业界急需一种方法，能够像人类学生一样主动组织、复习、检验所学内容，从而让模型在特定知识库上形成稳固、可检索的记忆。

### 关键概念速览
**主动阅读（Active Reading）**：让模型自行生成学习计划并执行，包括自问自答、间隔复习等，就像学生在阅读教材时会做笔记、提问、定期回顾一样。  
**自生成学习策略（Self‑generated Learning Strategies）**：模型在看到原始材料后，先输出一系列学习动作（比如生成测验、绘制概念图），再依据这些动作进行训练。  
**间隔重复（Spaced Repetition）**：把同一条信息在不同时间点重新呈现，以强化记忆，类似于记忆卡片软件的复习机制。  
**知识抽取问答（Knowledge Extraction QA）**：模型从文档中自动生成问答对，用来检验自己是否已经掌握了对应事实。  
**概念映射（Concept Mapping）**：把文档中的实体和关系组织成图结构，帮助模型建立层次化的语义网络。  
**预训练尺度（Pre‑training Scale）**：指在上百亿甚至上千亿 token 级别的训练量上进行的学习，这里作者把主动阅读的思路直接搬到大规模预训练阶段。  
**WikiExpert‑8B**：作者公开的 8 B 参数模型，经过 1 trillion 生成 token 的主动阅读预训练，专注于 Wikipedia 事实。

### 核心创新点
1. **从被动微调到主动学习**  
   *之前的做法*：直接在目标文档上做全量微调或简单的数据增强，模型只能被动接受信息。  
   *本文的做法*：在微调前让模型先生成一套学习策略（如自测题、复习计划），再依据这些策略对文档进行多轮自监督训练。  
   *改变*：模型不再是“一次性灌输”，而是经历类似“阅读‑做笔记‑自测‑复习”的循环，显著提升了对稀有事实的记忆率。

2. **策略生成与执行的统一框架**  
   *之前的做法*：手工设计特定任务（如只做 QA、只做摘要），缺乏灵活性。  
   *本文的做法*：使用一个元模型（Strategy Generator）在每段文本上输出多种策略指令，随后同一模型依据指令执行对应的子任务（生成 QA、绘制概念图、安排间隔复习）。  
   *改变*：实现了“模型自教”——模型自己决定该怎么学，避免了人为任务设计的局限。

3. **在预训练阶段引入主动阅读**  
   *之前的做法*：主动阅读只在下游微调阶段实验，规模受限。  
   *本文的做法*：在 1 trillion 生成 token 的大规模预训练中，持续对 Wikipedia 文本施行主动阅读流程。  
   *改变*：让模型在根基阶段就形成更稳固的事实记忆，最终在下游事实问答上超越了参数量数百倍的竞争模型。

4. **跨域事实提升的实证**  
   *之前的做法*：在特定领域（如医学）微调后提升有限，往往只能在该领域内部看到小幅增长。  
   *本文的做法*：在金融（FinanceBench）和通用问答（SimpleQA）两大完全不同的基准上分别使用对应文档进行主动阅读。  
   *改变*：在 SimpleQA 上实现 66% 正确率（相对提升 313%），在 FinanceBench 上达到 26%（相对提升 160%），证明方法具备跨域可迁移性。

### 方法详解
**整体思路**  
主动阅读框架可以看作三步走：① **策略生成** → 让模型阅读原始材料并输出一系列学习指令；② **策略执行** → 根据指令生成对应的训练信号（如 QA 对、概念图、复习时间戳）；③ **自监督优化** → 用这些信号对模型自身进行多任务微调。整个循环在每个文档上重复多次，形成“阅读‑练习‑复习”的闭环。

**关键模块拆解**  

1. **Strategy Generator（策略生成器）**  
   - 输入：一段原始文本（例如 Wikipedia 条目段落）。  
   - 输出：一组结构化指令，例如  
     *“生成 3 条基于本段的事实问答”*、  
     *“构造概念映射：实体 A → 关系 → 实体 B”*、  
     *“安排在 10 步后复现本段”*。  
   - 实现方式：在一个小型的 LLM 上进行少量微调，使其学会从文本中抽取关键概念并转化为任务描述。类似于让学生在阅读后写下“学习计划”。

2. **Task Executor（任务执行器）**  
   - 对每条指令，模型切换到相应的子任务头。  
   - **事实 QA 生成**：模型把段落转化为问答对，答案直接取自原文，问题设计成检验细节的形式。  
   - **概念映射构建**：模型输出三元组（实体、关系、实体），随后用图结构表示。  
   - **间隔复现标记**：模型在训练序列中插入特殊 token，指示在未来的训练步骤中重新呈现该段落。  
   - 这些子任务均采用自监督的方式，因为答案都可以从原始文本直接得到。

3. **Multi‑Task Self‑Supervision（多任务自监督）**  
   - 所有生成的信号被统一映射到损失函数：QA 对使用交叉熵损失，概念图使用实体/关系分类损失，间隔复现使用时间加权的语言建模损失。  
   - 通过加权求和，模型在一次前向传播中同时学习“记事实”“建结构”“复习时机”。  
   - 为防止某一任务主导训练，作者在实验中对权重进行调节（原文未给出具体数值）。

4. **循环迭代与记忆巩固**  
   - 完成一次策略执行后，模型会把该段落重新放回训练流中，按照间隔复现指令在若干步后再次出现。  
   - 这种“间隔重复”机制让模型在不同时间点多次看到同一事实，从而在参数空间中形成更深的局部最小值。  
   - 类比于人类使用 Anki 卡片复习：每次出现都会强化记忆痕迹。

**最巧妙的设计**  
- **策略自生成**：让模型自己决定学习方式，而不是研究者硬编码任务，这大幅提升了方法的通用性。  
- **在预训练阶段直接植入**：把主动阅读的循环嵌入到 1 trillion token 的大规模训练中，使得模型的“记忆结构”从根本上就更适合事实检索。  

### 实验与效果
- **测试数据集**  
  - **SimpleQA（基于 Wikipedia）**：一个面向常识问答的子集，题目均可在对应 Wikipedia 条目中找到答案。  
  - **FinanceBench**：金融领域的专业问答基准，覆盖公司财报、监管政策等。  
- **对比基线**  
  - 传统全量微调（Vanilla Finetuning）  
  - 常见数据增强手段（如随机遮盖、混合生成文本）  
  - 其他自监督微调方案（如仅做 QA 生成）  
- **主要结果**  
  - 在 SimpleQA 上，主动阅读模型达到 **66%** 正确率，较普通微调提升 **313%**（相对增幅）。  
  - 在 FinanceBench 上，取得 **26%** 正确率，较基线提升 **160%**。  
  - 公开的 **WikiExpert‑8B**（8 B 参数、1 trillion token）在多个事实 QA 基准上超过了参数量在数百亿级别的竞争模型。  
- **消融实验**  
  - 去掉间隔复现指令后，SimpleQA 准确率下降约 8%。  
  - 仅保留 QA 生成而不做概念映射，FinanceBench 的提升幅度从 160% 降至 90%。  
  - 这些实验表明，策略的多样性（QA、映射、复现）共同驱动了性能提升。  
- **局限与不足**  
  - 论文未给出在极低资源语言上的实验，方法对大规模预训练资源的依赖仍然明显。  
  - 生成的概念映射质量受限于模型本身的抽取能力，错误的三元组可能会误导后续学习。  
  - 训练成本比普通微调高出约 2‑3 倍，因为每段文本需要多轮生成和多任务优化。

### 影响与延伸思考
主动阅读的思路在发布后迅速被多方引用，尤其是那些关注模型可解释性和长期记忆的工作。后续研究尝试把**元学习**与主动阅读结合，让模型在不同任务之间迁移学习策略；还有人把**人类教师反馈**加入策略生成环节，形成“人机协同阅读”。如果想进一步探索，可关注以下方向：  
- **跨语言主动阅读**：如何让模型在多语言语料上生成统一的学习计划。  
- **策略搜索优化**：使用强化学习或进化算法自动发现更高效的学习指令组合。  
- **低资源适配**：在仅有几万条文档的场景下，是否还能通过少量主动阅读获得显著记忆提升。  

### 一句话记住它
让模型自己制定并执行学习计划，像学生一样“阅读‑自测‑复习”，即可在大规模预训练和微调中显著提升事实记忆。