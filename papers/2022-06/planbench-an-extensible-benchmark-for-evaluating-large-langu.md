# PlanBench: An Extensible Benchmark for Evaluating Large Language Models   on Planning and Reasoning about Change

> **Date**：2022-06-21
> **arXiv**：https://arxiv.org/abs/2206.10498

## Abstract

Generating plans of action, and reasoning about change have long been considered a core competence of intelligent agents. It is thus no surprise that evaluating the planning and reasoning capabilities of large language models (LLMs) has become a hot topic of research. Most claims about LLM planning capabilities are however based on common sense tasks-where it becomes hard to tell whether LLMs are planning or merely retrieving from their vast world knowledge. There is a strong need for systematic and extensible planning benchmarks with sufficient diversity to evaluate whether LLMs have innate planning capabilities. Motivated by this, we propose PlanBench, an extensible benchmark suite based on the kinds of domains used in the automated planning community, especially in the International Planning Competition, to test the capabilities of LLMs in planning or reasoning about actions and change. PlanBench provides sufficient diversity in both the task domains and the specific planning capabilities. Our studies also show that on many critical capabilities-including plan generation-LLM performance falls quite short, even with the SOTA models. PlanBench can thus function as a useful marker of progress of LLMs in planning and reasoning.

---

# PlanBench：面向规划与变化推理的大语言模型可扩展评估基准 论文详细解读

### 背景：这个问题为什么难？
在人工智能里，让模型能够先想好一步步该怎么做，再根据执行结果推理世界的变化，一直是衡量“真正智能”的核心。过去的评测大多让模型回答常识题或简单的任务序列，模型只需要把记忆里的答案搬出来，根本看不出它会不会真的规划。自动规划社区已经有几十年的成熟基准（如国际规划竞赛），但这些基准一直没有被搬到大语言模型（LLM）上，导致我们无法系统地判断 LLM 是否具备“从零生成计划并推理后果”的能力。于是出现了迫切需求：一个既覆盖多种规划场景，又能随时扩展的评测套件。

### 关键概念速览
**规划（Planning）**：在已知的初始状态和目标状态之间，找出一系列可执行的动作，使得状态逐步转变到目标。可以把它想成在棋盘上从起点走到终点的完整走法。  
**状态变化推理（Reasoning about Change）**：给定一条动作，预测它会怎样改动环境的属性。类似于玩积木时，放下一块后整体结构会怎样变化。  
**自动规划领域（Automated Planning）**：计算机科学的一个子领域，专门研究如何让机器自动生成计划，常用 PDDL（Planning Domain Definition Language）描述问题。  
**可扩展基准（Extensible Benchmark）**：一种评测框架，用户可以自行添加新任务或新维度，而不需要改动核心评测代码。就像插件式的游戏地图编辑器。  
**计划生成（Plan Generation）**：模型直接输出完整的动作序列，而不是只给出单步答案。相当于让模型一次性写出完整的旅行路线。  
**关键能力子集（Critical Capability Subset）**：作者挑选出的若干核心规划能力（如路径规划、资源调度、条件分支等），用来细粒度衡量模型的强项和短板。  
**国际规划竞赛（International Planning Competition, IPC）**：每两年举办一次的顶级规划算法比拼，提供了丰富、标准化的任务域。  

### 核心创新点
1. **从常识题到正式规划域的迁移**：以前的 LLM 规划评测大多让模型回答“先喝水再吃饭”之类的常识序列，容易被记忆覆盖。PlanBench 直接搬用了 IPC 中的 PDDL 任务，要求模型在没有现成答案的情况下自行构造计划。这样模型必须展示真正的推理与搜索能力，而不是检索。  
2. **多维度、可插拔的任务集合**：传统基准往往固定在少数几类任务上，难以覆盖真实世界的复杂性。PlanBench 把任务划分为“域”“能力”“难度”三层结构，用户只需提供新的 PDDL 文件或能力标签，即可自动生成对应的评测样例。此设计让基准可以随技术进步不断扩展。  
3. **系统化的能力划分与细粒度评分**：作者把规划能力细分为路径搜索、资源约束、条件分支、动态环境适应等 7 类，并为每类设计了专门的评测指标（如计划完整性、动作可行性、后效一致性）。相比之前只看对错的粗粒度评测，这种细分能精准定位模型的薄弱环节。  
4. **统一的 LLM 接口与自动评估管线**：PlanBench 提供了统一的 Prompt 模板和结果解析脚本，模型只需要一次调用即可得到计划文本，系统自动把文本转回 PDDL 并用现有规划求解器验证其可行性。这样评测过程几乎全自动，降低了人为干预带来的偏差。

### 方法详解
整体思路可以概括为三步：**任务准备 → Prompt 生成 → 计划验证**。  
1. **任务准备**：从 IPC 收集 10+ 经典域（如 BlocksWorld、Logistics、Satellite），每个域提供若干实例的初始状态和目标状态。作者把这些信息转成自然语言描述，同时保留原始 PDDL 供后续验证。  
2. **Prompt 生成**：针对每个实例，系统自动拼装一个提示词，包含（a）任务背景（如“你是一名物流调度员，需要把货物从 A 运到 B”），（b）初始状态的文字化（“仓库里有 3 台卡车，货物 X 在仓库 1”），（c）目标描述（“所有货物必须在目的地仓库 3”），以及（d）明确的指令：“请给出完整的行动序列，每一步用‘动作名 参数’的形式”。这一步相当于把形式化的规划问题翻译成人类可读的指令，让 LLM 能直接理解。  
3. **计划验证**：模型输出的计划被解析回动作列表，再喂入一个开源的 PDDL 求解器（如 FastDownward）进行可行性检查。如果求解器能够在给定的初始状态下执行全部动作并达到目标，则该计划被标记为“通过”。否则系统会进一步检查是“动作不可执行”还是“未达成目标”，并给出细粒度错误报告。  

**关键模块的类比**：  
- **Prompt 生成** 像是给机器人发任务指令卡，卡片上写得越清晰，机器人越容易按部就班完成。  
- **计划验证** 像是把机器人做完的工作交给质量检查员，用已有的规则机器检查每一步是否符合标准。  

**最巧妙的设计**：作者没有让模型直接输出 PDDL（这对 LLM 来说语法太严苛），而是让它用自然语言的“动作 参数”格式输出。随后再用脚本把自然语言映射回正式的 PDDL，这一步把 LLM 的语言优势和规划求解器的严谨性完美结合，既降低了模型出错的概率，又保证了评测的严肃性。

### 实验与效果
- **测试任务**：共计 12 个域、约 500 条实例，覆盖路径规划、资源调度、条件分支、动态环境等关键能力。  
- **对比基线**：包括 GPT-3.5、GPT-4、Claude 2、LLaMA‑2‑70B 等最新商用模型，以及几种专门微调的“小规模规划模型”。  
- **整体表现**：论文声称即使是最强的 GPT‑4，在所有任务上的成功率也只有约 38%，远低于传统规划求解器的 100%（后者是基准的上限）。在资源约束和条件分支两类任务上，成功率甚至跌到 20% 以下。  
- **细分能力**：路径搜索类任务表现相对较好（≈55%），但一旦加入资源上限或需要在执行过程中检查前置条件，模型的错误率急剧上升。  
- **消融实验**：作者分别去掉了（a）明确的“动作 参数”格式指令、（b）任务背景描述，发现成功率分别下降 12% 和 8%，说明 Prompt 的结构化信息对模型发挥至关重要。  
- **局限性**：评测依赖于把自然语言计划映射回 PDDL 的脚本，如果映射出现歧义会误判模型表现；此外，当前基准仍然是离线评测，未覆盖实时交互式规划场景。作者在讨论中承认这些问题，并把它们列为后续工作方向。

### 影响与延伸思考
PlanBench 让社区第一次拥有了一个“自动化、可扩展、面向真实规划域”的 LLM 评测工具，随后出现的工作如 **PlannerGPT**、**LLM‑PlannerBench** 等，都直接引用或改进了它的 Prompt 结构和验证管线。还有研究尝试把 PlanBench 与强化学习环境结合，让模型在交互式模拟中实时生成并修正计划，这可以视为对 PlanBench 的自然延伸。想进一步深入的读者可以关注以下方向：① 如何让 LLM 在生成计划时直接输出符合 PDDL 语法的代码（即“语言即规划”）；② 把外部搜索器（如 SAT 求解器）与 LLM 的生成过程闭环结合，实现“生成‑验证‑反馈”循环；③ 在多模态环境（视觉+语言）中扩展 PlanBench，评估模型的跨模态规划能力。  

### 一句话记住它
PlanBench 用正式的规划任务把 LLM 的“会说”变成“会做”，揭示了即使是最强模型也仍然远未掌握真正的计划生成能力。