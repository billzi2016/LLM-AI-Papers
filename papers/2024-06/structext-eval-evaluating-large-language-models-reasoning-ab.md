# StrucText-Eval: Evaluating Large Language Model's Reasoning Ability in   Structure-Rich Text

> **Date**：2024-06-15
> **arXiv**：https://arxiv.org/abs/2406.10621

## Abstract

The effective utilization of structured data, integral to corporate data strategies, has been challenged by the rise of large language models (LLMs) capable of processing unstructured information. This shift prompts the question: can LLMs interpret structured data directly in its unstructured form? We propose an automatic evaluation data generation method for assessing LLMs' reasoning capabilities on structure-rich text to explore this. Our approach supports 8 structured languages and 29 tasks, generating data with adjustable complexity through controllable nesting and structural width. We introduce StrucText-Eval, a benchmark containing 5,800 pre-generated and annotated samples designed to evaluate how well LLMs understand and reason through structured text. StrucText-Eval is divided into two suites: a regular Test suite (3,712 samples) and a Test-Hard suite (2,088 samples), the latter emphasizing the gap between human and model performance on more complex tasks. Experimental results show that while open-source LLMs achieve a maximum accuracy of 74.9\% on the standard dataset, their performance drops significantly to 45.8\% on the harder dataset. In contrast, human participants reach an accuracy of 92.6\% on StrucText-Eval-Hard, highlighting LLMs' current limitations in handling intricate structural information. The benchmark and generation codes are open sourced in \url{https://github.com/MikeGu721/StrucText-Eval}

---

# StrucText-Eval：结构化文本中大语言模型推理能力评估 论文详细解读

### 背景：这个问题为什么难？

企业里大量信息以表格、树形、JSON 等结构化形式存储，而传统的大语言模型（LLM）擅长处理自然语言的连续文字。把结构化数据直接喂给 LLM，模型往往把层级、键值关系当成普通文字，导致推理错误。之前的评测大多使用纯文本或简单的表格，缺少对嵌套深度、宽度等结构特征的系统考察。于是我们缺乏一种能够量化 LLM 在“结构丰富”文本上理解与推理水平的基准，这正是本文要填补的空白。

### 关键概念速览
- **结构化语言**：指带有明确层级或键值关系的文本形式，如 JSON、XML、YAML 等。可以想象成“带标签的句子”，每个标签决定了信息的组织方式。  
- **嵌套深度（Nesting Depth）**：结构中子结构套在父结构里的层数。类似于俄罗斯套娃，层数越多，模型需要记住的上下文层级越多。  
- **结构宽度（Structural Width）**：同一层级上并列的子节点数量。把它想成“一行里有多少列”，宽度大时信息密度提升，推理难度上升。  
- **可控复杂度生成**：通过调节嵌套深度和宽度来人为设置样本难度的技术。就像调节游戏关卡的怪物数量和强度，让评测更有梯度。  
- **StrucText-Eval**：本文推出的评测套件，包含 5,800 条人工标注的结构化文本样本，分为普通 Test 与更难的 Test‑Hard 两个子集。  
- **Test‑Hard 套件**：专门挑选高嵌套、高宽度的样本，用来暴露模型与人类之间的性能差距。  
- **基准（Benchmark）**：统一的评测平台，提供数据、任务定义和评分脚本，保证不同模型在同一标准下比较。  

### 核心创新点
1. **从语言层面直接生成结构化评测数据**  
   - 之前的评测往往先把结构化信息转成自然语言描述，导致结构信息被稀释。  
   - 本文设计了一套自动化生成器，直接在 8 种结构化语言（如 JSON、XML 等）上采样，并保留原始层级信息。  
   - 结果是评测数据天然保留结构特征，能够真实检验模型对键值、嵌套关系的理解能力。

2. **可调节的复杂度控制机制**  
   - 传统数据集难度不可控，导致模型好坏难以细分。  
   - 作者引入“嵌套深度”和“结构宽度”两个可调参数，生成从浅层到深层、从窄到宽的样本。  
   - 这种梯度式难度让研究者可以系统地观察模型性能随结构复杂度的衰减趋势。

3. **双套件设计：普通 Test 与 Test‑Hard**  
   - 只提供一个统一难度的评测会掩盖模型在极端场景下的缺陷。  
   - 通过筛选高难度样本构建 Test‑Hard，明确展示人类（92.6%）与开源模型（最高 45.8%）的差距。  
   - 这为后续改进提供了明确的目标区间。

4. **全流程开源：数据、生成代码、评测脚本**  
   - 过去很多评测只公开数据，生成过程不透明，复现成本高。  
   - 本文把从语法定义到样本标注的全部代码公开，任何人都能自行扩展到新结构语言或新任务。  
   - 促进了社区围绕结构化推理的协同研发。

### 方法详解
整体思路可以拆成三步：**语法定义 → 参数化采样 → 自动标注**，形成一个闭环的评测数据流水线。

1. **语法定义层**  
   - 为每种结构化语言编写形式化的上下文无关文法（CFG），明确每个键、列表、对象的合法组合方式。  
   - 类比于“拼装玩具说明书”，文法告诉生成器哪些部件可以拼在一起，哪些组合是非法的。

2. **参数化采样层**  
   - 在文法的基础上加入两个控制变量：`depth`（最大嵌套层数）和 `width`（每层最大子节点数）。  
   - 采样时先随机决定当前层的宽度，然后递归生成子结构，直到达到设定的深度上限。  
   - 通过调节这两个变量，生成器可以快速产出从“单层键值对”到“十层深、每层十个子节点”的样本。

3. **任务与答案生成层**  
   - 论文覆盖 29 种推理任务，包括键值检索、路径查询、计数、比较、逻辑组合等。  
   - 对每个生成的结构体，系统根据任务模板自动生成对应的查询问题和正确答案。  
   - 例如，对 JSON `{ "dept": { "sales": { "employees": 12 } } }`，任务可能是“返回 sales 部门的员工数”，答案是 12。

4. **难度标签与划分**  
   - 采样完成后，根据 `depth` 与 `width` 的综合得分给每条样本打上难度标签。  
   - 难度最高的 2,088 条被挑入 Test‑Hard，剩余的 3,712 条构成普通 Test。  
   - 这种划分方式确保 Test‑Hard 真正聚焦在结构复杂度上，而不是随机抽样的偶然性。

5. **评测接口**  
   - 提供统一的 JSON 输入/输出格式，模型只需读取结构化文本并返回答案字符串。  
   - 评分脚本自动比对模型输出与金标准答案，计算准确率。  
   - 由于所有步骤都是脚本化的，研究者可以一键跑全套实验，甚至自行扩展任务或语言。

**最巧妙的点**在于把“结构复杂度”抽象为两个可调参数，并把它们直接嵌入生成过程。这样既避免了手工挑选难例的主观性，又让难度梯度可量化、可复现。

### 实验与效果
- **数据集**：StrucText-Eval 包含 5,800 条样本，覆盖 8 种结构化语言（JSON、XML、YAML、CSV、INI、TOML、HTML、Markdown）和 29 种推理任务。  
- **评测模型**：作者选取了多款开源 LLM（如 LLaMA‑2‑13B、Mistral‑7B、Gemma‑2B 等），并使用了官方提供的 zero‑shot 与 few‑shot 两种提示方式。  
- **结果**：在普通 Test 上，最好的开源模型达到了 **74.9%** 的整体准确率；在 Test‑Hard 上，同一模型的准确率骤降至 **45.8%**。相比之下，人类参与者在 Test‑Hard 上取得 **92.6%** 的准确率，显示出模型在高结构复杂度下仍有显著差距。  
- **基线对比**：论文把模型表现与传统文本推理基准（如 GSM8K、MATH）做了横向比较，发现即使在相似的任务规模下，结构化文本的推理难度更高，模型的相对下降更明显。  
- **消融实验**：作者分别关闭“嵌套深度控制”和“结构宽度控制”，发现去掉任一控制后，生成的样本难度分布趋于单一，模型在 Test‑Hard 上的下降幅度减小约 12%，说明两者共同决定了评测的挑战性。  
- **局限性**：论文承认评测仍然局限于 8 种语言和 29 种任务，未覆盖更复杂的图结构或跨文档关联；此外，评测只使用准确率，未考虑模型的解释性或计算成本。

### 影响与延伸思考
StrucText-Eval 首次提供了系统化、可控的结构化文本推理基准，迅速被社区引用。后续工作如 **StructEval**、**TreeReason** 等在此基础上加入了多模态结构（如图数据库）或更细粒度的计时评测。对想继续深入的读者，可以关注以下方向：  
1. **结构化提示工程**：如何设计 Prompt 让 LLM 更好地捕捉键值层级。  
2. **混合模型**：结合检索式模块或专用解析器，提高对深层嵌套的准确率。  
3. **自监督结构预训练**：在大规模结构化语料上进行预训练，弥补模型对结构感知的不足。  
4. **评测扩展**：加入跨文件、跨表格的推理任务，进一步逼近企业真实数据场景。

### 一句话记住它
StrucText-Eval 用可控的嵌套深度和宽度生成结构化文本，让我们首次能够精准量化大语言模型在“层级信息”上的推理缺口。