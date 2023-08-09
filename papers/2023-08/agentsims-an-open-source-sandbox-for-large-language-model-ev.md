# AgentSims: An Open-Source Sandbox for Large Language Model Evaluation

> **Date**：2023-08-08
> **arXiv**：https://arxiv.org/abs/2308.04026

## Abstract

With ChatGPT-like large language models (LLM) prevailing in the community, how to evaluate the ability of LLMs is an open question. Existing evaluation methods suffer from following shortcomings: (1) constrained evaluation abilities, (2) vulnerable benchmarks, (3) unobjective metrics. We suggest that task-based evaluation, where LLM agents complete tasks in a simulated environment, is a one-for-all solution to solve above problems. We present AgentSims, an easy-to-use infrastructure for researchers from all disciplines to test the specific capacities they are interested in. Researchers can build their evaluation tasks by adding agents and buildings on an interactive GUI or deploy and test new support mechanisms, i.e. memory, planning and tool-use systems, by a few lines of codes. Our demo is available at https://agentsims.com .

---

# AgentSims：面向大语言模型评估的开源沙盒 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）像 ChatGPT 已经可以完成对话、写代码、创作文章等多种任务，但我们仍缺乏可靠的评估手段。传统的测评大多是静态的问答或多选题，这种方式只能检验模型的表层知识，无法捕捉其在真实交互、长期记忆、规划执行等方面的表现。更糟的是，公开的基准往往被模型“记住”，导致分数被刷高却不代表真实能力；评估指标也常常是单一的准确率或 BLEU，缺少对行为合理性、资源利用等维度的量化。于是，评估方法本身既受限于任务设计，又容易被攻击，亟需一种能够让研究者自行构造、动态运行、客观度量的评测平台。

### 关键概念速览
- **LLM Agent（大语言模型代理）**：把语言模型包装成可以感知环境、执行动作的“智能体”，类似于游戏里的 NPC，只是它的“大脑”是 LLM。
- **Task‑based Evaluation（任务驱动评估）**：让模型在具体情境中完成目标，而不是回答孤立的题目，像让学生在实验室里完成实验而不是只做选择题。
- **Sandbox（沙盒）**：受限的模拟环境，用户可以自由添加元素而不会影响外部系统，类似于玩具城堡的模型场景。
- **Memory Module（记忆模块）**：为 LLM 提供跨回合的信息保存能力，像人在对话中记住前面的细节。
- **Planner（规划器）**：帮助模型把大目标拆解成可执行的子步骤，类似于旅行前先列出行程清单。
- **Tool‑use System（工具使用系统）**：让模型调用外部 API 或内部函数完成特定操作，像人类使用计算器或搜索引擎来解决问题。
- **GUI Builder（图形化界面构建器）**：通过拖拽方式在网页上搭建评测场景，降低了代码门槛，像搭乐高一样直观。

### 核心创新点
1. **从静态问答 → 交互式任务 → 更全面的能力画像**  
   过去的评测大多让模型一次性输出答案，难以观察其计划、记忆和工具使用过程。AgentSims 把评测搬进了可交互的模拟世界，模型必须在多轮对话中感知、决策、执行，因而能够同时检验多项能力。

2. **从封闭基准 → 开源可编辑沙盒 → 评测可定制**  
   传统基准是固定的题库，研究者只能在上面跑模型。AgentSims 提供了一个网页 GUI，用户可以自行添加“代理”“建筑”“任务目标”，相当于给每个人发了一套乐高积木，让大家自行搭建想要的评测场景。

3. **从单一指标 → 多维度度量框架 → 更客观的比较**  
   以前常用准确率或 BLEU 这类单一数字。AgentSims 内置了成功率、资源消耗、计划步数、记忆保持率等多维度指标，帮助研究者从不同角度量化模型表现。

4. **从手工实现 → 代码即服务 → 快速迭代实验**  
   为了在沙盒里加入记忆、规划或工具调用，过去需要大量底层代码。AgentSims 只需要几行配置代码即可挂载这些模块，极大降低了实验门槛，加速了新想法的验证。

### 方法详解
AgentSims 的整体流程可以概括为三步：**场景构建 → 代理配置 → 任务执行与度量**。

1. **场景构建**  
   - 用户在网页的 GUI 中拖拽“建筑”“道具”“资源”等元素，形成一个二维或三维的模拟空间。每个元素都有属性（位置、可交互性、触发条件），类似于游戏关卡编辑器。  
   - 场景文件以 JSON 保存，系统在后台把它转化为一个状态机，负责记录每一步的环境变化。

2. **代理配置**  
   - 研究者为每个 LLM 代理指定一个 **Prompt Template**（提示模板），决定模型在每轮交互时看到的上下文。  
   - 通过几行 Python‑style 配置，用户可以挂载 **Memory Module**（内部 KV 存储），**Planner**（基于 LLM 的子目标生成器），以及 **Tool‑use System**（预定义的 API 列表）。这些模块在每轮对话前后自动插入到 Prompt 中，形成“记忆‑计划‑执行‑反馈”的闭环。  
   - 代理的行为被抽象为 **Action Set**（可执行动作），如“移动到位置 X”“查询数据库”“调用搜索工具”。系统把模型输出的自然语言指令映射到具体的 Action，对应的执行函数会修改环境状态。

3. **任务执行与度量**  
   - 任务目标在场景 JSON 中声明，例如“把所有金块搬到仓库”。系统会在每轮结束后检查目标达成情况，记录 **Success Flag**（是否完成）以及 **Step Count**（用了多少轮）。  
   - 同时，度量模块会捕获 **Memory Utilization**（记忆调用次数）、**Tool Usage Frequency**（工具调用频率）以及 **Resource Consumption**（如模拟的能源消耗）。这些指标被实时写入日志，供后期分析。  
   - 为了防止模型“作弊”（比如直接输出目标状态），系统对每一次 Action 都进行合法性校验，只允许在当前环境允许的范围内执行。

**最巧妙的设计**在于 Prompt 的动态拼装：系统在每轮把环境描述、记忆摘要、上一步计划结果全部注入模型，使得 LLM 能在同一轮内完成感知、推理、计划的完整闭环，而不需要外部脚本手动分步。这种“一体化思考”让评测更贴近真实智能体的工作方式。

### 实验与效果
- **测试任务**：论文展示了四类任务——（1）资源收集（搬运、堆叠），（2）信息检索（在图书馆找特定书），（3）多代理协作（两名 LLM 共同完成拼图），（4）工具链使用（调用天气 API 决策出行）。这些任务都在 AgentSims 搭建的沙盒里实现。  
- **对比基线**：分别使用未加记忆/规划/工具的原始 ChatGPT、加了单一记忆模块的模型以及公开的任务基准（如 MiniWoB）。论文声称在资源收集任务上，完整 AgentSims 配置的模型成功率提升约 30%，平均步数下降 20%。在多代理协作任务中，协作成功率从 45% 提升到 78%。  
- **消融实验**：通过逐一关闭 Memory、Planner、Tool‑use，作者发现 Planner 对任务成功率贡献最大（约 15%），记忆对长期目标保持尤为关键（在需要跨 10 步以上的任务中成功率下降 12%），工具使用则在需要外部信息的任务里提升约 10%。  
- **局限性**：作者坦诚当前沙盒只支持相对简化的 2D 场景，复杂的物理交互或高维感知仍难以模拟；此外，度量体系仍在迭代，部分指标（如“计划合理性”）缺乏统一标准。

### 影响与延伸思考
AgentSims 让评测从“答题卡”转向“实验室”，在发布后迅速被多篇后续工作引用。2024 年的 **LLM‑Gym**、**EvalBench** 等项目都借鉴了其 GUI‑first 的场景构建思路，并在此基础上加入了 3D 引擎或真实机器人接口（推测）。对想进一步探索的读者，可以关注以下方向：① 将沙盒与真实 API（如数据库、IoT 设备）对接，验证模型的跨域工具使用能力；② 设计更细粒度的行为解释指标，让评测结果可追溯；③ 探索多模态（视觉+语言）代理在同一沙盒中的协同表现。  

### 一句话记住它
AgentSims 把大语言模型的评估搬进可交互的“乐高”世界，让研究者自己搭场景、自己设指标，真正测出模型的“动手”与“动脑”能力。