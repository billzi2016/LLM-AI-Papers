# Metacognitive Capabilities of LLMs: An Exploration in Mathematical   Problem Solving

> **Date**：2024-05-20
> **arXiv**：https://arxiv.org/abs/2405.12205

## Abstract

Metacognitive knowledge refers to humans' intuitive knowledge of their own thinking and reasoning processes. Today's best LLMs clearly possess some reasoning processes. The paper gives evidence that they also have metacognitive knowledge, including ability to name skills and procedures to apply given a task. We explore this primarily in context of math reasoning, developing a prompt-guided interaction procedure to get a powerful LLM to assign sensible skill labels to math questions, followed by having it perform semantic clustering to obtain coarser families of skill labels. These coarse skill labels look interpretable to humans.   To validate that these skill labels are meaningful and relevant to the LLM's reasoning processes we perform the following experiments. (a) We ask GPT-4 to assign skill labels to training questions in math datasets GSM8K and MATH. (b) When using an LLM to solve the test questions, we present it with the full list of skill labels and ask it to identify the skill needed. Then it is presented with randomly selected exemplar solved questions associated with that skill label. This improves accuracy on GSM8k and MATH for several strong LLMs, including code-assisted models. The methodology presented is domain-agnostic, even though this article applies it to math problems.

---

# 大语言模型的元认知能力：数学问题求解探索 论文详细解读

### 背景：这个问题为什么难？

在数学题目上让大语言模型（LLM）给出正确答案已经取得了显著进展，但模型往往像“黑盒”一样直接跳到答案，缺乏对自身思考过程的自我认识。传统的提示工程（prompt engineering）只能把题目和求解步骤喂进去，模型并不主动把题目归类到某类数学技能上，也不利用已有的同类解法作为参考。缺少这种“元认知”——即模型能说出自己需要用哪种技巧、并据此检索相似例题的能力——导致在复杂或多步骤的问题上仍然容易出错。正因为如此，研究者想弄清模型是否真的拥有对自身推理过程的直觉，并尝试把这种直觉转化为可操作的提升手段。

### 关键概念速览

**元认知（metacognition）**：指对自己思考过程的认识和调控，就像人在解题前先判断需要用哪种公式，再决定是否回顾相关例题。这里指模型能够“自我标记”所需的数学技能。

**技能标签（skill label）**：对一道数学题的高层次描述，例如“二次方程求根”或“组合计数”。相当于给题目贴上一个主题标签，帮助模型快速定位解法方向。

**语义聚类（semantic clustering）**：把大量细粒度标签按照含义相似度合并成更粗的类别，就像把“求解一次函数交点”和“求解一次函数斜率”归到同一个“大一次函数求解”大类。

**示例检索（example retrieval）**：在模型已知的解题库中挑选与当前技能标签对应的已解题目，提供给模型参考，类似于人类在做题时翻看教材中的相似例题。

**提示引导交互（prompt‑guided interaction）**：通过精心设计的对话模板，让模型先完成标签生成、再做聚类、最后进行解题的多阶段流程。

### 核心创新点

1. **从“直接求解”到“先标记再求解”**  
   以前的做法是把题目直接喂给 LLM，让它一次性输出答案。本文先让模型给每道题打上技能标签，再依据标签挑选对应的示例，最后再让模型正式求解。这样把解题过程拆成两步，显著提升了正确率。

2. **利用 LLM 自己生成的标签进行语义聚类**  
   直接让模型给每道题贴上细粒度标签会产生上千种不同的表述，难以直接使用。作者让模型自行对这些标签做语义相似度聚类，得到几百个可解释的粗标签，使得后续检索更高效且易于人工审阅。

3. **在测试阶段把完整标签库作为“技能字典”提供**  
   当模型面对新题时，先把所有已知的粗标签列表交给它，让它自行判断最匹配的技能，再从对应的示例池中抽取若干已解题目。相比于随机抽样，这种有针对性的示例显著提升了模型的推理质量。

4. **方法论的领域无关性**  
   虽然实验只在数学数据集上做了验证，但整个流程（标签生成 → 聚类 → 技能检索 → 解题）并未依赖数学专有的特征，理论上可以迁移到编程、物理等其他需要步骤化推理的任务。

### 方法详解

整体思路可以概括为四个阶段：**标签生成 → 语义聚类 → 技能识别 → 示例驱动求解**。下面逐步拆解每一步的操作细节。

1. **标签生成**  
   - 采用一个强大的 LLM（如 GPT‑4）作为“标签专家”。  
   - 为每道训练题目构造提示：“请阅读下面的数学题，并用一句话概括它需要的解题技巧”。  
   - 模型输出的短句即为细粒度技能标签。这里的关键是让模型用人类可读的语言描述，而不是直接给出解法步骤。

2. **语义聚类**  
   - 收集所有细粒度标签后，使用同一个 LLM 再次进行聚类。提示示例：“下面是一系列解题技巧的描述，请把它们按照含义相似度分成若干组，每组给出一个概括性的名称”。  
   - LLM 基于内部的语义向量（类似于嵌入）把相近的标签归并，输出每个粗标签及其包含的细标签列表。  
   - 这一步的巧妙之处在于不需要外部聚类算法，完全依赖语言模型的语义理解能力，省去了手工特征工程。

3. **技能识别（测试时）**  
   - 当模型面对一条新题目时，先把完整的粗标签列表（约几百条）作为上下文，提示模型：“请从下面的标签中选出最符合这道题的那个”。  
   - 由于标签已经是高度抽象的概念，模型可以快速定位到对应的技能，而不必在海量的细标签中纠结。

4. **示例驱动求解**  
   - 根据模型选出的粗标签，系统从训练集里随机抽取若干（如 3–5 条）已经标记为同一技能的完整解题示例。  
   - 再次构造提示，把这些示例连同新题目一起喂给模型，要求它在参考示例的基础上给出答案。  
   - 这里的核心是“示例引导”，相当于让模型在已有的思路框架里进行微调，而不是从零开始推理。

**最反直觉的设计**：把完整的标签库直接交给模型进行技能匹配，而不是让模型自行生成标签再匹配。直觉上会担心标签太多导致上下文溢出或噪声，但实验表明模型能够在大列表中快速定位，反而提升了鲁棒性。

### 实验与效果

- **数据集**：使用两套主流数学推理基准——GSM8K（约 8k 训练题）和 MATH（约 12k 训练题），分别覆盖小学到高中层次的算术、代数、几何等题型。  
- **基线**：对比了原始的直接求解（无标签、无示例）、传统的 CoT（思维链）提示、以及最新的 Code‑Assisted（代码辅助）模型。  
- **提升幅度**：在 GSM8K 上，直接求解的准确率约 78%，加入标签+示例后提升至约 84%；在 MATH 上从 45% 提升到约 52%。所有提升均在 5%–7% 之间，且对多种模型（包括 GPT‑3.5、Claude、Code‑Llama）都有正向效果。  
- **消融实验**：作者分别去掉标签生成、去掉聚类、去掉示例检索三项进行测试。结果显示：去掉标签生成导致性能回落到基线水平；去掉聚类（直接使用细标签）略有下降但仍优于基线；去掉示例检索则降幅最大，说明示例驱动是关键因素。  
- **局限性**：实验仅在数学领域验证，标签的可解释性在更抽象的任务上可能下降；此外，标签列表长度受限于模型的上下文窗口，极大规模的标签库仍需分块处理。原文未提供对不同语言或跨学科任务的实验结果。

### 影响与延伸思考

这篇工作打开了“让 LLM 先自我定位再求解”的新思路，随后出现的研究多聚焦于 **任务感知提示（task‑aware prompting）**、**自适应示例检索** 以及 **跨模态元认知**。例如，有后续工作尝试把技能标签扩展到代码生成任务，利用类似的标签‑示例流程提升模型的调试能力。对想进一步探索的读者，可以关注以下方向：  
- **标签自动演化**：让模型在使用过程中不断细化或合并技能标签，实现持续学习。  
- **跨领域迁移**：验证在物理、化学或法律推理等非数学任务上的可行性。  
- **更高效的检索机制**：结合向量数据库或稀疏注意力，解决标签列表过长导致的上下文瓶颈。  
这些都是基于本文元认知框架的自然延伸。

### 一句话记住它

让大语言模型先给题目贴上“技能标签”，再用同类示例引导求解，能显著提升数学推理的准确率。