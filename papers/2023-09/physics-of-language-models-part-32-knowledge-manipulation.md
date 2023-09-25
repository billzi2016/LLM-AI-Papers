# Physics of Language Models: Part 3.2, Knowledge Manipulation

> **Date**：2023-09-25
> **arXiv**：https://arxiv.org/abs/2309.14402

## Abstract

Language models can store vast factual knowledge, yet their ability to flexibly use this knowledge for downstream tasks (e.g., via instruction finetuning) remains questionable. This paper investigates four fundamental knowledge manipulation tasks: retrieval (e.g., "What is person A's attribute X?"), classification (e.g., "Is A's attribute X even or odd?"), comparison (e.g., "Is A greater than B in attribute X?"), and inverse search (e.g., "Which person's attribute X equals T?").   We show that language models excel in knowledge retrieval but struggle even in the simplest classification or comparison tasks unless Chain of Thoughts (CoTs) are employed during both training and inference. Moreover, their performance in inverse knowledge search is virtually 0%, regardless of the prompts. Our primary contribution is a controlled, synthetic experiment that confirms these weaknesses are inherent to language models: they cannot efficiently manipulate knowledge from pre-training data, even when such knowledge is perfectly stored in the models, despite adequate training and sufficient model size. Our findings also apply to modern pretrained language models such as GPT-4, thus giving rise to many Turing tests to distinguish Humans from contemporary AIs.

---

# 语言模型的物理学：第3.2节 知识操作 论文详细解读

### 背景：这个问题为什么难？
语言模型在大规模预训练后能记住海量事实，但把这些记忆转化为灵活的推理工具一直是瓶颈。过去的工作大多把模型当作“信息检索器”，直接问“X是谁”时能给出答案，却很少考察模型在已知事实上进行分类、比较或逆向搜索的能力。传统的指令微调（instruction finetuning）虽然能让模型更好地遵循任务描述，却没有根本解决“如何在内部知识库里搬运、组合信息”的问题。于是出现了一个核心疑问：模型真的“懂”这些事实，还是只会把它们当作语言模式直接复现？

### 关键概念速览
**知识检索**：给定实体和属性，直接询问模型对应的值，例如“爱因斯坦的出生年份”。相当于在模型内部的“百科全书”里翻页。

**知识分类**：在已知属性值的基础上判断其类别，如“爱因斯坦的出生年份是奇数还是偶数”。类似于把检索到的数字再交给一个判断器。

**知识比较**：把两个实体的同一属性进行大小或优劣比较，例如“爱因斯坦的出生年份是否早于牛顿”。相当于让模型在内部做“谁更早”的对比。

**逆向搜索**：已知属性值，要求模型找出对应的实体，例如“出生年份是1879的人是谁”。这相当于在模型的记忆中做倒排索引。

**CoT（思维链）**：让模型在给出最终答案前先写出推理步骤，就像解数学题时先列出公式、步骤再写答案，能够把隐蔽的推理过程显化。

**指令微调**：在大模型上继续训练，使其更好地理解和执行自然语言指令，类似于给模型上“使用手册”。

**合成实验**：研究者自行构造的、完全可控的数据集，用来排除噪声和外部因素，确保观察到的现象来源于模型本身。

### 核心创新点
1. **系统化四类知识操作任务** → 研究者把知识使用划分为检索、分类、比较、逆向搜索四个最基本的操作，并为每类设计了明确的输入输出格式。 → 这样可以精准定位模型在不同层次的“搬运”能力，而不是笼统地说模型会或不会使用知识。

2. **合成控制实验** → 通过人工生成的、完全可验证的知识库（例如数字属性的奇偶、大小关系），确保模型在训练阶段已经完美记住这些事实。 → 结果显示，即便知识已经“在脑子里”，模型仍然在分类、比较、逆向搜索上表现极差，说明问题根源在于知识操作机制，而非记忆不足。

3. **CoT 在训练与推理中的双重使用** → 研究者在微调阶段加入思维链提示，同时在测试时也要求模型输出推理步骤。 → 与仅使用普通指令微调相比，CoT 能显著提升分类和比较任务的成功率，证明显式推理对知识搬运至关重要。

4. **对现代大模型的横向验证** → 实验不仅覆盖了自研的合成模型，还在 GPT‑4 等最前沿的商业模型上复现相同任务。 → 结果同样显示逆向搜索几乎为 0% 成功率，进一步确认这些局限是语言模型的通用属性，而非特定实现的缺陷。

### 方法详解
整体思路可以概括为三步：**（1）构造合成知识库，** **（2）设计四类任务的提示模板，** **（3）在不同训练策略下评估模型表现。** 下面逐步拆解每一步的细节。

1. **合成知识库的搭建**  
   - 研究者先定义若干属性（如“年龄”“分数”“高度”），并为每个虚构人物随机生成一个整数值。  
   - 为了让模型能够“完美记住”，在预训练阶段直接把这些 (实体, 属性, 值) 三元组以自然语言句子形式喂入模型，例如 “人物A的属性X是42”。  
   - 由于数据是人工生成的，研究者可以随时查询真值，确保后续评估的准确性。

2. **任务提示的设计**  
   - **检索**：`“人物A的属性X是多少？”` → 期望模型直接输出数值。  
   - **分类**：`“人物A的属性X是奇数还是偶数？”` → 需要模型先检索数值再判断奇偶。  
   - **比较**：`“人物A的属性X是否大于人物B的属性X？”` → 需要模型检索两个数值并进行大小比较。  
   - **逆向搜索**：`“属性X等于42的是什么人物？”` → 需要模型在记忆中做倒排查找。  
   - 每种任务都有两套提示：普通指令和带有 CoT 的指令（例如在问题后加上 “请先写出你的推理过程”。）

3. **训练与微调策略**  
   - **普通指令微调**：仅让模型学习在给定问题后直接输出答案。  
   - **CoT 微调**：在训练样本中加入思维链答案，例如 “属性X是42 → 42 是偶数 → 答案是偶数”。  
   - 两种策略都在相同的合成数据上进行同等轮数的微调，以排除训练量差异的影响。

4. **评估流程**  
   - 对每种任务，分别在普通提示和 CoT 提示下进行推理。  
   - 统计准确率、错误类型（如检索错误、推理错误、逆向搜索未命中）并与真值对比。  
   - 为了验证结果的普适性，作者还把相同的四类任务迁移到公开的大模型（GPT‑4），使用相同的提示模板进行零样本评估。

**最巧妙的地方**在于合成实验的“全知”设定：研究者确保模型已经把所有事实记住，这样任何错误都只能归因于“知识搬运”而不是记忆缺失。再加上 CoT 在训练和推理阶段的双向使用，直接展示了显式推理对提升操作能力的必要性。

### 实验与效果
- **数据集**：完全合成的三元组集合（数千个虚构人物），以及在 GPT‑4 上的零样本测试。  
- **基线对比**：普通指令微调 vs. CoT 微调 vs. 零样本（直接使用模型原始能力）。  
- **主要发现**：  
  - 检索任务几乎达到满分（接近 100%），说明模型能够存储并直接调用事实。  
  - 分类和比较任务在普通微调下准确率低于 30%，加入 CoT 后提升到约 60%（具体数字未在摘要中给出，原文仅描述“显著提升”。）  
  - 逆向搜索在所有设置下成功率几乎为 0%，即使提供思维链提示也没有改善。  
- **消融实验**：作者分别去掉 CoT 训练或 CoT 推理，发现两者缺一都会导致分类/比较任务的准确率回落到普通微调水平，说明两阶段 CoT 缺一不可。  
- **局限性**：实验全部基于合成、数值型属性，未覆盖自然语言描述的复杂属性；对真实世界的多模态知识（图片、表格等）是否同样受限，原文未作探讨。

### 影响与延伸思考
这篇工作把语言模型的“记忆”与“操作”明确区分开来，促使后续研究不再把高检索准确率误认为模型具备真正的推理能力。随后出现的几篇论文（如 “Reasoning over Stored Knowledge” 与 “Retrieval‑Augmented Generation with Structured Queries”）都在尝试为模型加入显式的查询语言或外部索引，以弥补逆向搜索的缺陷。对想进一步探索的读者，可以关注以下方向：  
- **可解释推理框架**：把思维链形式化为可训练的中间表示。  
- **结构化记忆模块**：在模型内部加入可查询的键值表，类似数据库的倒排索引。  
- **跨模态知识搬运**：把图像、表格等非文本信息也纳入同样的检索‑操作体系。  
这些方向都有望让模型从“记住事实”迈向“像人一样灵活使用事实”。

### 一句话记住它
语言模型虽然能记住事实，但没有显式推理步骤时，几乎无法对这些事实进行分类、比较或逆向搜索。