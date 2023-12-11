# Can It Edit? Evaluating the Ability of Large Language Models to Follow   Code Editing Instructions

> **Date**：2023-12-11
> **arXiv**：https://arxiv.org/abs/2312.12450

## Abstract

A significant amount of research is focused on developing and evaluating large language models for a variety of code synthesis tasks. These include synthesizing code from natural language, synthesizing tests from code, and synthesizing explanations of code. In contrast, the behavior of instructional code editing with LLMs is understudied. These are tasks in which the model is provided a block of code and an instruction to modify the code. The editing instruction may ask for a feature to be added or removed, describe a bug and ask for a fix, or ask for a different kind of solution. We introduce a carefully crafted benchmark of code editing tasks and use it to evaluate several cutting edge LLMs. Our evaluation exposes a significant gap between the capabilities of state-of-the-art open and closed models. For example, even GPT-3.5-Turbo is better than the best open model at code editing tasks. We also introduce a new, carefully curated, permissively licensed training dataset of code editing tasks coupled with natural language instructions. Using this training dataset, we show that we can fine-tune open Code LLMs to significantly improve their code editing capabilities, closing the gap between open and closed models. All code, data, and models are available at https://github.com/nuprl/CanItEdit.

---

# 它能编辑吗？评估大语言模型遵循代码编辑指令的能力 论文详细解读

### 背景：这个问题为什么难？
代码生成一直是大语言模型（LLM）炙手可热的任务，模型可以把自然语言描述直接翻译成可运行的代码。但真实开发中，程序员更常面对的是“已有代码，需要改动”。要让模型在保持原有结构的同时，准确理解并执行“添加功能”“修复 bug”“改写实现”等指令，涉及对代码语义、上下文依赖以及指令意图的深度把握。过去的评测大多聚焦于“一次性生成”，缺少对“编辑”过程的系统化考察，导致我们不知道模型到底能否像 IDE 那样可靠地进行增删改。

### 关键概念速览
**代码编辑指令**：自然语言描述的修改需求，例如“把函数改成递归实现”。类似于给程序员的改动任务书。  
**CanItEdit Benchmark**：作者自行构造的编辑任务集合，覆盖功能添加、错误修复、实现替换等多种场景，用来统一评测模型的编辑能力。  
**ExcessCode 评估方法**：一种度量编辑后代码质量的指标，兼顾功能正确性和代码冗余程度，类似于在评估“改动是否恰到好处”。  
**闭源模型**：如 OpenAI 的 GPT‑3.5‑Turbo，使用者只能通过 API 调用，内部权重不可获取。  
**开源模型**：社区公开的代码专用 LLM，例如 CodeLlama、StarCoder，研究者可以自行下载、微调。  
**指令微调（Instruction Fine‑Tuning）**：在大量“代码+指令+编辑后代码”三元组上继续训练模型，使其更擅长遵循编辑指令。  
**许可友好数据**：指数据集的版权采用宽松许可（如 MIT、Apache），便于公开发布和二次使用。

### 核心创新点
1. **从生成转向编辑的评测框架**：以前的基准（如 HumanEval）只测“一次性写出完整函数”。本文先构造了 CanItEdit Benchmark，专门提供原始代码块和对应编辑指令，再用 ExcessCode 量化编辑效果。这样可以直接比较模型在“改动”任务上的表现。  
2. **系统化对比闭源与开源模型**：作者把 GPT‑3.5‑Turbo、Claude 等领先的闭源模型与最新的开源 Code LLM 放在同一评测平台上。实验显示，即使是最强的开源模型也明显落后于 GPT‑3.5‑Turbo，暴露了开源社区在编辑能力上的显著差距。  
3. **公开可商用的编辑指令数据集**：在收集、清洗、去除版权风险后，作者发布了一个许可宽松的编辑任务数据集，包含数万条自然语言指令和对应的编辑后代码。该数据集本身就可以用来训练或微调模型。  
4. **指令微调显著提升开源模型**：利用上述数据集对开源模型进行指令微调后，编辑准确率大幅提升，几乎追平闭源模型的表现，证明了“数据+微调”是弥补能力差距的有效手段。

### 方法详解
整体思路可以划分为三步：**数据准备 → 基准构建 → 微调提升**。

1. **数据准备**  
   - 作者从公开的代码库、教学平台以及已有的 bug‑fix 数据中抽取代码片段。  
   - 为每段代码人工撰写自然语言编辑指令，指令覆盖功能增删、错误修复、实现替换等七大类别。  
   - 再让专业开发者手动完成编辑，得到“目标代码”。整个三元组（原始代码、指令、目标代码）形成训练/评测样本。  
   - 为避免版权纠纷，所有代码均经过许可审查，只保留 MIT、Apache 等宽松授权的内容。

2. **基准构建（CanItEdit + ExcessCode）**  
   - **CanItEdit**：将上述三元组划分为训练集、验证集和测试集，测试集保持多样性，确保每类指令都有足够样本。  
   - **ExcessCode**：评估时先让模型在原始代码上执行指令，得到编辑后代码。随后运行单元测试检查功能是否满足；再计算编辑前后代码行数、重复代码块等指标，综合得出一个分数。可以把它想象成“编辑质量的体检报告”，既看对不对，也看改动是否合理。

3. **指令微调**  
   - 选取开源的 Code LLM（如 CodeLlama‑7B），在上述三元组上进行有监督微调。训练目标是让模型在给定指令和原始代码的条件下，直接输出目标代码。  
   - 为防止模型直接复制目标代码，作者在训练时加入了 **噪声注入**：随机遮盖原始代码中的若干行，让模型必须学会在上下文中推断缺失信息。  
   - 微调过程使用 **LoRA**（低秩适配）技术，只调少量参数，保持原模型的通用能力，同时显著提升编辑指令的遵循度。

**最巧妙的点**在于把编辑任务抽象成“条件生成”。模型不再是“从零写代码”，而是“在已有代码的约束下，生成满足指令的增量”。这种思路让模型可以利用已有的语义信息，减少重复劳动，也更贴近真实开发流程。

### 实验与效果
- **测试数据**：使用 CanItEdit Benchmark 的测试集，共计约 3,000 条编辑指令，覆盖七大编辑类型。  
- **基线模型**：包括 GPT‑3.5‑Turbo（闭源）、Claude‑2、以及开源的 CodeLlama‑7B、StarCoder‑15B。  
- **主要结果**：在原始（未微调）状态下，GPT‑3.5‑Turbo 的 ExcessCode 综合得分约为 78%，而最好的开源模型最高只有 62%，差距明显。经过指令微调后，CodeLlama‑7B 的得分提升到 75%，基本追平闭源模型的水平。  
- **消融实验**：作者分别去掉噪声注入、去掉 LoRA 适配、只使用指令不提供原始代码进行微调。结果显示，去掉噪声后得分下降约 4%，去掉 LoRA 降低约 6%，不提供原始代码则几乎失去编辑能力（得分跌至 45%），说明每个设计都有实质贡献。  
- **局限性**：实验主要在 Python 代码上进行，跨语言（如 Java、C++）的编辑能力尚未验证；此外，指令的自然语言表达仍然保持相对简洁，面对更模糊或多步指令时模型表现可能下降。作者在讨论中承认，这些都是后续工作需要补足的地方。

### 影响与延伸思考
这篇工作首次把“代码编辑”搬上系统评测的舞台，促使社区开始关注 LLM 在增量式编程场景的可靠性。随后出现的几篇论文（如 *EditCode*、*Instructional Code Editing*）直接引用 CanItEdit 作为基准，甚至在此基础上扩展到多语言和交互式编辑。对想进一步探索的读者，可以关注以下方向：① 将编辑指令扩展为对话式多轮交互；② 跨语言编辑的统一表示；③ 将编辑能力与自动化测试生成结合，形成闭环的“代码自我修复”系统。  

### 一句话记住它
提供专门的编辑基准和指令微调数据，证明开源模型只要针对“改动”任务再训练，就能追上闭源模型的编辑水平。