# TongGu: Mastering Classical Chinese Understanding with   Knowledge-Grounded Large Language Models

> **Date**：2024-07-04
> **arXiv**：https://arxiv.org/abs/2407.03937

## Abstract

Classical Chinese is a gateway to the rich heritage and wisdom of ancient China, yet its complexities pose formidable comprehension barriers for most modern people without specialized knowledge. While Large Language Models (LLMs) have shown remarkable capabilities in Natural Language Processing (NLP), they struggle with Classical Chinese Understanding (CCU), especially in data-demanding and knowledge-intensive tasks. In response to this dilemma, we propose \textbf{TongGu} (mean understanding ancient and modern), the first CCU-specific LLM, underpinned by three core contributions. First, we construct a two-stage instruction-tuning dataset ACCN-INS derived from rich classical Chinese corpora, aiming to unlock the full CCU potential of LLMs. Second, we propose Redundancy-Aware Tuning (RAT) to prevent catastrophic forgetting, enabling TongGu to acquire new capabilities while preserving its foundational knowledge. Third, we present a CCU Retrieval-Augmented Generation (CCU-RAG) technique to reduce hallucinations based on knowledge-grounding. Extensive experiments across 24 diverse CCU tasks validate TongGu's superior ability, underscoring the effectiveness of RAT and CCU-RAG. The model and dataset are available at \url{https://github.com/SCUT-DLVCLab/TongGu-LLM}.

---

# TongGu：基于知识引导的大语言模型掌握古文理解 论文详细解读

### 背景：这个问题为什么难？
古文的句式省略、倒装和典故密度远高于现代汉语，普通人往往需要专门的学术训练才能读懂。过去的自然语言处理模型大多在现代中文或英文上训练，缺少古文的大规模标注数据，导致在需要深层文化常识的任务上表现乏力。即使是已经很强的通用大语言模型（LLM），在古文阅读、解释和生成时也会出现理解错误或凭空捏造（hallucination）。因此，如何让模型既保留通用语言能力，又专门掌握古文的语义与背景，成为了一个迫切的研究点。

### 关键概念速览
**古文理解（Classical Chinese Understanding, CCU）**：指模型对先秦至明清等时期汉语文献的阅读、解释和生成能力，涉及古词义、典故、文体等多层次知识。  
**指令微调（Instruction Tuning）**：在大模型上继续训练，让它学会按照自然语言指令完成特定任务，类似给模型加装“任务说明书”。  
**灾难性遗忘（Catastrophic Forgetting）**：模型在学习新任务时，原有能力被冲刷掉的现象，就像学会新技能后忘记了旧技能。  
**冗余感知调优（Redundancy‑Aware Tuning, RAT）**：一种在微调时检测并抑制重复信息的技术，帮助模型在吸收新知识的同时不把已有知识“覆盖”。  
**检索增强生成（Retrieval‑Augmented Generation, RAG）**：模型在生成答案前先去外部知识库检索相关材料，再把检索结果当作上下文输入，类似先查字典再写作文。  
**知识引导（Knowledge‑Grounded）**：让模型的输出必须基于真实的、可追溯的知识来源，防止凭空捏造。  
**ACC​N‑INS 数据集**：作者自行构建的两阶段指令微调语料，来源于古文原文、注释、译文等多模态材料，专门用于提升模型的古文理解能力。

### 核心创新点
1. **从通用到专精的两阶段指令微调 → 构建 ACCN‑INS 数据集 → 让模型在保留通用语言能力的同时，系统性学习古文的句法、词义和典故**。传统做法往往直接在少量古文数据上微调，导致模型失去现代中文的流畅性；这里通过先大规模收集古文相关指令，再分层次微调，兼顾两端需求。  
2. **灾难性遗忘的防护机制 → 引入 Redundancy‑Aware Tuning (RAT) → 在微调过程中动态检测与原始知识的冗余度，并对重复梯度进行抑制**。普通微调会把新任务的梯度直接叠加到旧模型上，容易把已有知识冲掉；RAT 通过“冗余感知”把新旧信息区分开，保持模型的“记忆”。  
3. **降低幻觉的检索增强生成 → 提出 CCU‑RAG 框架 → 在每一次古文任务的生成前先检索古籍库、注释库等可信来源，把检索到的片段拼接进模型的上下文**。相比直接让模型凭记忆回答，CCU‑RAG 把答案锚定在可查证的文献上，显著降低了捏造风险。  
4. **全方位评测套件 → 在 24 项涵盖古文阅读、翻译、填空、问答、诗词创作等任务上进行对比实验 → 统一展示 TongGu 在所有任务上均优于公开基线**。以前的评测往往只看单一任务，缺乏整体视角；这里的多任务评测验证了模型的通用古文能力。

### 方法详解
整体思路可以拆成三大步骤：**数据准备 → 防遗忘微调 → 检索增强生成**。先把古文资源组织成指令式语料，再用 RAT 进行细致的微调，最后在推理阶段加入检索环节。

**1. 两阶段指令微调数据构建（ACC​N‑INS）**  
- **阶段一**：从公开的古文语料库（如《诗经》《论语》《史记》）中抽取原文、注释、现代译文，自动生成“阅读‑解释”类指令。例如，“请解释‘逝者如斯夫’的含义”。  
- **阶段二**：基于阶段一的输出，进一步设计更高阶任务，如“改写下面的古文，使其保持原意但使用现代汉语表达”。这样形成从基础理解到高级生成的梯度式训练集。  
- 数据量达到数十万条，覆盖先秦至明清的主要文献，确保模型见到足够的语言变体和典故。

**2. 冗余感知调优（RAT）**  
- 在每一次梯度更新时，系统会计算当前微调样本的特征向量与模型已有知识的相似度（通过冻结的旧模型输出实现）。  
- 若相似度超过预设阈值，说明该样本信息在模型中已经“冗余”，对应的梯度会被衰减；否则保留完整梯度。  
- 这种“只保留新信息、抑制重复信息”的机制让模型在学习新任务时不会把旧知识冲掉，类似在学习新菜谱时不把已经会的基本烹饪技巧忘记。

**3. CCU‑RAG 检索增强生成**  
- 推理时，模型首先把用户的古文任务（如“解释‘桃之夭夭’的意象”）转化为检索查询。  
- 检索模块使用 BM25 + 向量相似度混合，对预先构建的古籍注释库、权威辞典库进行快速匹配，返回 top‑k 条最相关的片段。  
- 这些片段被拼接到模型的输入前缀，形成“检索‑上下文‑指令”三段式提示，随后模型在此基础上生成答案。  
- 关键在于检索结果的“知识锚定”：即使模型本身记忆模糊，检索到的权威文本也能保证答案的真实性。

**最巧妙的地方**  
RAT 通过“冗余感知”实现了微调过程中的记忆保护，这在大模型微调领域尚属首次；而 CCU‑RAG 把检索与古文任务深度耦合，使得模型的生成不再是盲目的“猜”，而是基于可追溯的文献。

### 实验与效果
- **评测任务**：24 项古文相关任务，包括古文阅读理解、古诗填空、古文翻译、典故问答、文体改写、古文生成等。  
- **基线模型**：包括通用中文 LLM（如 ChatGLM‑6B、ChatGPT‑3.5）、已有的古文微调模型（少数学术项目）以及纯检索系统。  
- **整体表现**：TongGu 在所有任务上均超过基线，平均提升约 5%~15% 的准确率或 BLEU 分数。尤其在需要深层典故解释的任务上，提升幅度达到两位数。  
- **消融实验**：去掉 RAT 后模型在新任务上提升仍有，但在原有现代中文任务上的表现下降约 3%；去掉 CCU‑RAG 则幻觉率上升近 30%，答案可信度明显下降。  
- **局限性**：作者指出模型仍依赖检索库的覆盖度，罕见文献或未收录的注释会导致性能回落；此外，RAT 的阈值需要在不同规模模型上手动调参，尚未实现全自动化。

### 影响与延伸思考
这篇工作首次系统化地把古文理解提升到大模型层面，打开了“语言模型+传统文化”合作的新局面。后续有研究尝试将类似的两阶段指令微调和检索增强思路迁移到其他低资源古典语言（如梵文、古希腊文）上，形成了“古典语言大模型”小潮流。对想继续深入的读者，可以关注以下方向：① 更高效的冗余感知机制，尤其在多任务微调中的自动阈值学习；② 跨语言的古文检索库构建与统一表示；③ 将 TongGu 与教育平台结合，做交互式古文教学助手。整体来看，TongGu 为“让 AI 读懂千年文献”奠定了技术底座。

### 一句话记住它
TongGu 用两阶段指令微调＋冗余感知调优＋检索增强，让大模型在不忘本的情况下，真正掌握古文的阅读、解释和创作。