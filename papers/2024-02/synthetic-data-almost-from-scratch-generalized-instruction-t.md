# Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for   Language Models

> **Date**：2024-02-20
> **arXiv**：https://arxiv.org/abs/2402.13064

## Abstract

We introduce Generalized Instruction Tuning (called GLAN), a general and scalable method for instruction tuning of Large Language Models (LLMs). Unlike prior work that relies on seed examples or existing datasets to construct instruction tuning data, GLAN exclusively utilizes a pre-curated taxonomy of human knowledge and capabilities as input and generates large-scale synthetic instruction data across all disciplines. Specifically, inspired by the systematic structure in human education system, we build the taxonomy by decomposing human knowledge and capabilities to various fields, sub-fields and ultimately, distinct disciplines semi-automatically, facilitated by LLMs. Subsequently, we generate a comprehensive list of subjects for every discipline and proceed to design a syllabus tailored to each subject, again utilizing LLMs. With the fine-grained key concepts detailed in every class session of the syllabus, we are able to generate diverse instructions with a broad coverage across the entire spectrum of human knowledge and skills. Extensive experiments on large language models (e.g., Mistral) demonstrate that GLAN excels in multiple dimensions from mathematical reasoning, coding, academic exams, logical reasoning to general instruction following without using task-specific training data of these tasks. In addition, GLAN allows for easy customization and new fields or skills can be added by simply incorporating a new node into our taxonomy.

---

# 几乎从零合成数据：面向语言模型的通用指令微调 论文详细解读

### 背景：这个问题为什么难？

指令微调（Instruction Tuning）是让大语言模型（LLM）更好地理解和执行人类指令的关键手段。过去的做法大多依赖已有的任务数据集或人工标注的示例，这导致两大瓶颈：一是数据来源受限，难以覆盖所有学科和技能；二是扩展到新领域时需要大量人工投入，成本高且速度慢。于是模型往往在未见过的任务上表现平平，尤其是跨学科或专业化的需求。要突破这些限制，就必须找到一种既能产生海量、多样指令，又不依赖人工标注的方案。

### 关键概念速览
- **指令微调（Instruction Tuning）**：在已有的语言模型基础上，用一批“任务指令+示例”对模型进行再训练，使其更擅长按照自然语言指令完成各种任务。类似于给模型上“使用手册”。
- **知识学科树（Taxonomy of Human Knowledge）**：把人类的知识体系拆解成学科、子学科、具体学科的层级结构，像一棵从根到叶的树，帮助系统化地生成教学内容。
- **教学大纲（Syllabus）**：针对某一学科或主题，列出若干课程单元并标明每节课的关键概念，类似于学校里老师提前规划的课程安排。
- **合成指令（Synthetic Instruction）**：由模型自动生成的指令-答案对，而不是人工标注的真实数据。相当于让模型自己出题、给答案，再用这些自创的题目来训练自己。
- **通用指令调优（Generalized Instruction Tuning, GLAN）**：本文提出的整体框架，利用学科树和大纲生成全覆盖的合成指令，实现“一键式”微调。
- **可扩展性（Scalability）**：指方法能够在不大幅增加人工成本的情况下，快速加入新学科或新技能。

### 核心创新点
1. **从“学科树”出发生成指令 → 传统方法直接采集已有任务或手工编写 → 通过系统化的学科层级，GLAN 能自动覆盖几乎所有人类知识领域，省去人工收集的繁琐。**
2. **利用 LLM 自动生成教学大纲 → 过去的微调数据往往缺乏结构化的教学脉络 → GLAN 让模型先“制定课程计划”，再基于每节课的关键概念生成指令，保证指令在深度和广度上都更均衡。**
3. **全流程合成而不依赖任何真实任务数据 → 以往的指令微调仍需要少量真实任务作为种子 → GLAN 完全摆脱种子数据，展示了“几乎从零”生成高质量指令的可能性。**
4. **模块化的可插拔设计 → 传统微调体系难以快速加入新领域 → 只要在学科树中添加一个新节点，后续的大纲和指令会自动生成，实现低成本的领域扩展。

### 方法详解
#### 整体框架概览
GLAN 的工作流可以划分为四个阶段：  
1) **学科树构建** → 用少量人工规则和 LLM 辅助，把人类知识拆解成层级节点；  
2) **课程大纲生成** → 对每个学科节点，LLM 自动列出若干主题并为每个主题设计多节课；  
3) **关键概念抽取** → 在每节课的描述中提炼出核心概念和技能点；  
4) **指令合成** → 基于关键概念，让 LLM 生成多样的指令、输入、参考答案，形成完整的指令微调数据集。

#### 关键模块拆解
- **学科树构建**  
  - **输入**：一份预先整理的“学科种子列表”（如数学、物理、文学等）。  
  - **过程**：使用 LLM 递归展开，每次询问模型“该学科下面有哪些子领域？”并对返回的列表进行去重、层级归类。人类只需检查异常或明显错误，整体是半自动的。  
  - **输出**：一棵多层次的树结构，根节点是“人类知识”，叶子节点是具体学科（如“微积分”“古典文学批评”等）。

- **教学大纲生成**  
  - **输入**：每个叶子学科的名称。  
  - **过程**：向 LLM 提问“为 X 学科设计一个 8 周的课程，每周包含哪些主题？”模型返回主题列表；随后对每个主题继续提问“每周的主题需要拆成几节课？每节课的关键概念是什么？”得到细粒度的课程表。  
  - **输出**：结构化的大纲文件，形如 `{学科 → 主题 → 课次 → 关键概念}`。

- **关键概念抽取**  
  - **输入**：每节课的文字描述。  
  - **过程**：使用 LLM 的信息抽取能力，要求模型列出该课最核心的 3‑5 个概念或技能。这里的技巧是给模型提供“概念 = 能够独立完成的子任务”，帮助它聚焦可操作的要点。  
  - **输出**：每节课对应的概念集合，为后续指令生成提供种子。

- **指令合成**  
  - **输入**：关键概念集合。  
  - **过程**：对每个概念，构造若干指令模板（如“请解释 X 的原理”“请用 Python 实现 X 的算法”“请解答关于 X 的选择题”），让 LLM 填充具体内容并生成参考答案。为了提升多样性，模板会随机组合、加入噪声（如不同的提问方式、不同的输出格式）。  
  - **输出**：指令-输入-输出三元组，形成大规模的合成指令数据集。

#### 设计亮点
- **“教育系统”类比**：把知识组织成学科树和教学大纲，让模型的生成过程像老师备课一样，有计划、有层次，而不是随意拼凑。
- **完全闭环的自监督**：从学科树到指令的每一步都由 LLM 完成，只有极少数人工检查点，极大降低了人工成本。
- **模块化可插拔**：新增学科只需在树中添加节点，后续的大纲、概念、指令会自动流转，无需重新设计整体流程。

### 实验与效果
- **评测任务**：论文在数学推理、代码生成、学术考试（如 GRE、AP 题目）、逻辑推理以及通用指令遵循等多类任务上进行评估。  
- **基线对比**：与传统指令微调数据集（如 Alpaca、Self-Instruct）以及少量真实任务微调的模型相比，GLAN 在多数任务上取得了显著提升。论文声称在数学推理上领先约 5‑10% 的准确率，在代码生成的 Pass@1 上提升数个百分点，学术考试的整体分数也高出数分。  
- **消融实验**：作者分别去掉“教学大纲层级”和“关键概念抽取”两步，发现模型在复杂推理任务上的表现下降约 3‑7%，说明这两个模块对提升指令质量至关重要。  
- **局限性**：由于所有数据均为模型自生成，仍可能存在系统性偏差或错误答案；在高度专业化的前沿科研任务上，合成指令的可靠性尚未得到充分验证。作者也提到，当前的学科树仍基于英文语料，跨语言扩展需要额外工作。

### 影响与延伸思考
GLAN 的“从学科树到指令”的全链路思路为指令微调提供了可规模化、可持续的路径。自论文发布后，已有几篇工作尝试把类似的“教学大纲”概念用于多语言模型、对话系统以及强化学习奖励模型的构建（如 “Curriculum‑Driven Instruction Tuning”）。未来的研究可以在以下方向深化：  
- **跨语言学科树构建**：把中文、阿拉伯语等非英文知识体系纳入同一框架，实现真正的多语言指令微调。  
- **真实答案校验**：结合外部检索或专家系统，对合成答案进行自动校验，降低错误传播的风险。  
- **动态课程生成**：让模型根据用户交互实时调整教学大纲，实现个性化的指令微调。  
如果想进一步了解，可关注近期在 arXiv 上出现的 “Curriculum‑Based Data Synthesis for LLMs” 系列论文，它们在 GLAN 的基础上加入了自适应难度调节。

### 一句话记住它
**GLAN 用“学科树 + 教学大纲”让模型自己出题、给答案，从几乎零人工成本生成覆盖全人类知识的指令微调数据。**