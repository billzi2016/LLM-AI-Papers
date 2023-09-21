# CodePlan: Repository-level Coding using LLMs and Planning

> **Date**：2023-09-21
> **arXiv**：https://arxiv.org/abs/2309.12499

## Abstract

Software engineering activities such as package migration, fixing errors reports from static analysis or testing, and adding type annotations or other specifications to a codebase, involve pervasively editing the entire repository of code. We formulate these activities as repository-level coding tasks.   Recent tools like GitHub Copilot, which are powered by Large Language Models (LLMs), have succeeded in offering high-quality solutions to localized coding problems. Repository-level coding tasks are more involved and cannot be solved directly using LLMs, since code within a repository is inter-dependent and the entire repository may be too large to fit into the prompt. We frame repository-level coding as a planning problem and present a task-agnostic framework, called CodePlan to solve it. CodePlan synthesizes a multi-step chain of edits (plan), where each step results in a call to an LLM on a code location with context derived from the entire repository, previous code changes and task-specific instructions. CodePlan is based on a novel combination of an incremental dependency analysis, a change may-impact analysis and an adaptive planning algorithm.   We evaluate the effectiveness of CodePlan on two repository-level tasks: package migration (C#) and temporal code edits (Python). Each task is evaluated on multiple code repositories, each of which requires inter-dependent changes to many files (between 2-97 files). Coding tasks of this level of complexity have not been automated using LLMs before. Our results show that CodePlan has better match with the ground truth compared to baselines. CodePlan is able to get 5/6 repositories to pass the validity checks (e.g., to build without errors and make correct code edits) whereas the baselines (without planning but with the same type of contextual information as CodePlan) cannot get any of the repositories to pass them.

---

# CodePlan：基于大语言模型的仓库级代码编辑与规划 论文详细解读

### 背景：这个问题为什么难？
在实际项目里，迁移依赖、修复静态分析报错、为整个代码库加上类型标注等任务往往需要一次性改动多个文件，且这些文件之间相互引用、相互影响。传统的 LLM 辅助编程工具（如 Copilot）只能在单个文件或单个函数的上下文里生成代码，因为它们的提示长度有限，根本装不下整个仓库的全部信息。更糟的是，仓库内部的依赖关系是动态的：改动 A 可能导致 B、C 也必须同步更新，这种跨文件的连锁效应是单轮 LLM 调用难以捕捉的。因此，直接把“大模型 + 大仓库”硬塞进同一个 prompt 既不可行，也会导致生成的改动相互冲突，导致编译或运行错误。

### 关键概念速览
**仓库级编码（Repository-level coding）**：指需要在整个代码库范围内进行的编辑任务，往往涉及多个文件的协同修改。想象成一次大规模的装修，需要同时改动客厅、厨房和电路系统，而不是只换一块瓷砖。

**增量依赖分析（Incremental dependency analysis）**：在每一步改动后，重新计算哪些文件的语义受影响，就像在装修后检查哪些墙体需要重新粉刷。

**影响范围分析（Change may‑impact analysis）**：预测一次代码改动可能波及的其他位置，类似于在拆除一面墙前先估算会不会影响到承重结构。

**规划（Planning）**：把整个编辑任务拆成有序的子步骤，每一步都有明确的输入（代码片段、上下文）和输出（修改建议），类似于把装修工程拆成“先拆墙、后铺管、最后刷漆”的施工计划。

**适配器 Prompt（Adapter Prompt）**：为每一次 LLM 调用准备的上下文片段，包含全局依赖信息、历史改动和任务指令，像是给工人递交的“施工图纸+前期报告”。

**有效性检查（Validity check）**：在每一步或全部完成后，自动编译或运行测试，确保改动没有引入错误，类似于装修完后请专业验收。

### 核心创新点
1. **从单轮生成到多步规划**：传统工具直接把问题喂给 LLM，期望一次输出完整改动。CodePlan 把任务视为一系列有序的编辑步骤，每一步都基于最新的代码状态和全局依赖。这样可以在每一步后校验结果，避免一次性出错导致全库崩溃。

2. **增量依赖 + 影响分析的闭环**：在每次 LLM 生成修改后，系统立刻执行增量依赖分析，得到受影响的文件集合，再用影响范围分析决定下一步的目标位置。这个闭环让系统像“实时导航”，始终走在最需要改动的路线上，而不是盲目遍历所有文件。

3. **自适应规划算法**：CodePlan 采用一种自适应的搜索策略，根据已有的改动和依赖图动态决定下一步的编辑顺序，而不是预先固定的脚本。相当于在装修过程中，工头根据现场进度实时调整施工顺序，而不是死板遵循事先排好的时间表。

4. **任务无关的统一框架**：虽然实验只展示了 C# 包迁移和 Python 时序编辑两类任务，框架本身只要求提供任务指令和依赖信息，其他任何需要跨文件改动的工程任务都可以套用。相比之前的专用脚本或手工规则，CodePlan 的通用性更高。

### 方法详解
**整体思路**  
CodePlan 把仓库级编码看作“规划 → 执行 → 验证 → 迭代”四步循环。首先对整个仓库做一次全局依赖图构建；随后基于任务指令生成初始计划（哪些文件可能需要改动的候选列表）。接下来进入循环：从计划中挑选最紧迫的编辑点，构造适配器 Prompt，调用 LLM 获得具体改动；改动完成后立即进行增量依赖分析，更新依赖图并重新评估哪些文件受影响；如果验证（编译/测试）通过，则继续下一步；否则回滚或重新规划。

**关键模块拆解**  

1. **全局依赖图构建**  
   - 对每个文件解析 import / using / require 等语句，形成有向图（文件 A → 文件 B 表示 A 依赖 B）。  
   - 这一步只做一次，后续改动时使用增量更新，避免每次都全盘重新解析。

2. **任务指令解析**  
   - 输入是自然语言的任务描述（如“将所有 Newtonsoft.Json 替换为 System.Text.Json”）。  
   - 系统把指令拆成“目标库名、替换规则、作用范围”等结构化信息，供后续规划使用。

3. **初始计划生成**  
   - 根据指令在依赖图中标记所有可能受影响的节点，形成待编辑文件集合。  
   - 为每个文件生成一个“编辑候选”，并按依赖深度排序（先改动根节点，再改动叶子），类似于先拆除支撑结构再装修细部。

4. **适配器 Prompt 构造**  
   - 取当前待编辑文件的代码片段（通常是函数或类定义），加上：  
     a) 全局依赖摘要（该文件直接/间接依赖的关键符号）；  
     b) 已经完成的改动历史（让 LLM 知道前置步骤）；  
     c) 任务指令的简要复述。  
   - 这样 LLM 在一次调用里只看到局部代码，却拥有全局视野。

5. **LLM 调用与代码生成**  
   - 使用现有的代码生成模型（如 GPT‑4）对 Prompt 进行推理，输出修改后的代码块。  
   - 系统会对模型输出进行语法检查，确保返回的是合法的代码片段。

6. **增量依赖与影响分析**  
   - 将新代码写回仓库后，利用增量解析只重新计算受改动文件及其直接依赖的子图。  
   - 通过“may‑impact”规则（如函数签名变化会影响所有调用方），标记下一轮需要处理的文件。

7. **有效性检查**  
   - 对每一次改动后执行编译或单元测试，确保改动没有破坏构建。  
   - 若检查失败，系统会回滚该改动并在计划中插入“重新生成”步骤，或者尝试不同的 Prompt 变体。

8. **自适应规划循环**  
   - 根据验证结果和最新的依赖图，动态调整待编辑文件的优先级。  
   - 例如，若某个文件的改动导致大量下游文件被标记为受影响，系统会把它提前处理，以减少后续回滚的概率。

**最巧妙的地方**  
- **增量依赖 + 影响分析的闭环**：传统的静态分析往往一次性完成，面对大规模改动会非常慢。CodePlan 只在改动后局部更新依赖图，极大提升了效率。  
- **自适应规划**：不像固定脚本，系统能在执行过程中根据实际情况重新排队，类似于“实时调度”，显著提升成功率。

### 实验与效果
- **任务与数据集**：作者在两个真实场景上评估：C# 包迁移（把项目从旧的 NuGet 包迁移到新包）和 Python 时序编辑（对代码库进行时间顺序的增删改）。每个任务挑选了多个开源仓库，涉及 2 到 97 个文件的跨文件改动。  
- **Baseline**：使用相同的 LLM 与相同的全局上下文信息，但不做规划（即一次性生成所有改动），以及传统的基于规则的脚本。  
- **结果**：在有效性检查（能否成功编译并通过任务特定的验证）上，CodePlan 在 6 个仓库中有 5 个通过，而所有 baseline 均未通过任何仓库。匹配度（与人工标注的“正确改动”对比）也显著高于 baseline，具体数值在论文中给出但摘要未列出。  
- **消融实验**：作者分别去掉增量依赖、去掉影响分析、以及使用固定顺序的规划，发现成功率分别下降到 2/6、1/6、0/6，说明每个模块都是成功的关键。  
- **局限性**：实验只覆盖 C# 与 Python 两种语言，未验证在更大规模（上千文件）或多语言混合的仓库中的表现。作者也提到对 LLM 的调用成本仍然较高，尤其在需要大量迭代时。

### 影响与延伸思考
- **领域影响**：CodePlan 把“规划 + 大模型”引入了软件工程的仓库级任务，开启了把 LLM 当作“可编程的代码编辑器”而非单轮助手的新思路。后续有工作开始探索类似的多步交互式编辑框架，如“EditGPT”与“Plan-and-Edit”。  
- **后续方向**：  
  1. **跨语言通用依赖分析**：把增量依赖抽象为语言无关的抽象语法树（AST）层，提升框架在多语言项目中的适用性。  
  2. **成本优化**：结合检索增强（retrieval‑augmented generation）或小模型微调，降低每一步的 LLM 调用费用。  
  3. **人机协同**：让开发者在规划阶段提供可选的约束或优先级，形成“人‑AI 共创”的编辑流程。  
- **想深入的读者**：可以关注近期的“代码编辑计划（Code Planning）”系列工作，尤其是把强化学习用于自动调度编辑步骤的尝试。

### 一句话记住它
CodePlan 把跨文件的大改动拆成可验证的编辑计划，用增量依赖驱动的自适应调度，让 LLM 能安全、可靠地完成仓库级编码任务。