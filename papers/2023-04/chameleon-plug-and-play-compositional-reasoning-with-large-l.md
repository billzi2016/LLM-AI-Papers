# Chameleon: Plug-and-Play Compositional Reasoning with Large Language   Models

> **Date**：2023-04-19
> **arXiv**：https://arxiv.org/abs/2304.09842

## Abstract

Large language models (LLMs) have achieved remarkable progress in solving various natural language processing tasks due to emergent reasoning abilities. However, LLMs have inherent limitations as they are incapable of accessing up-to-date information (stored on the Web or in task-specific knowledge bases), using external tools, and performing precise mathematical and logical reasoning. In this paper, we present Chameleon, an AI system that mitigates these limitations by augmenting LLMs with plug-and-play modules for compositional reasoning. Chameleon synthesizes programs by composing various tools (e.g., LLMs, off-the-shelf vision models, web search engines, Python functions, and heuristic-based modules) for accomplishing complex reasoning tasks. At the heart of Chameleon is an LLM-based planner that assembles a sequence of tools to execute to generate the final response. We showcase the effectiveness of Chameleon on two multi-modal knowledge-intensive reasoning tasks: ScienceQA and TabMWP. Chameleon, powered by GPT-4, achieves an 86.54% overall accuracy on ScienceQA, improving the best published few-shot result by 11.37%. On TabMWP, GPT-4-powered Chameleon improves the accuracy by 17.0%, lifting the state of the art to 98.78%. Our analysis also shows that the GPT-4-powered planner exhibits more consistent and rational tool selection via inferring potential constraints from instructions, compared to a ChatGPT-powered planner. The project is available at https://chameleon-llm.github.io.

---

# 变色龙：即插即用的组合推理框架 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在自然语言处理上已经能做出惊人的推理，但它们本身只能靠训练时的静态知识，无法实时获取网络信息、调用外部工具或执行精确的数学、逻辑运算。传统的“纯语言”方案要么在知识更新上迟钝，要么在需要精细计算时出错，导致在科学问答、表格推理等知识密集型任务上表现受限。于是出现了“工具使用”思路：让模型调用搜索、计算或视觉模块。但如何让模型在一次对话中灵活、可靠地挑选并组合多个工具，仍是一个未被系统化解决的难题。

### 关键概念速览
- **LLM（大语言模型）**：通过海量文本训练得到的生成式模型，擅长语言理解和生成，但缺乏外部感知和实时计算能力。可以把它想成一个“语言专家”，但不具备“实验室仪器”。
- **Plug‑and‑Play 模块**：可随时接入的独立工具，如搜索引擎、视觉模型或自定义 Python 函数。它们像乐高块，拼在一起即可形成新功能。
- **组合推理（Compositional Reasoning）**：把多个子任务（检索、计算、视觉识别等）按顺序或并行组合完成整体目标。类似于做一道综合题，需要先查资料、再算公式、最后写答案。
- **Planner（规划器）**：由 LLM 驱动的决策器，负责读取用户指令、分析任务需求并生成一系列要调用的工具序列。可以把它比作“导演”，安排演员（各模块）出场顺序。
- **Tool Invocation（工具调用）**：Planner 输出的指令会被系统实际执行，例如发送搜索请求、运行 Python 代码或调用视觉 API。相当于导演喊出“灯光、布景、演员上场”。
- **ScienceQA**：一个多模态科学问答数据集，题目涉及文字、图片和外部知识，需要综合推理。  
- **TabMWP**：表格数学文字题数据集，需要从表格中抽取数值、做算术运算并给出答案，考验模型的数理推理和表格理解能力。

### 核心创新点
1. **从单一模型到工具链的统一框架**  
   - 之前的工作多是把搜索或计算功能硬编码进模型，缺乏统一的调度机制。  
   - 这篇论文把 LLM 当作“指挥官”，让它在运行时动态决定使用哪种外部模块，并以统一的 JSON‑like 语言描述调用序列。  
   - 结果是系统可以在同一次对话里同时使用搜索、视觉、Python 计算等多种能力，显著提升了对复杂任务的适应性。

2. **LLM‑based Planner 的“约束感知”能力**  
   - 早期的 Planner 只根据提示生成工具列表，容易出现不相关或冗余调用。  
   - 作者在 Planner 的提示词中加入了对任务约束的显式推理（例如“只能使用搜索或只能使用表格工具”），让模型在生成计划时自行过滤不符合约束的工具。  
   - 实验表明，这种约束感知显著提升了调用的合理性，尤其在使用 GPT‑4 作为 Planner 时表现更为一致。

3. **模块化的“即插即用”接口**  
   - 传统系统往往把工具深度耦合进模型内部，新增工具需要改代码。  
   - Chameleon 通过统一的 API 把每个工具包装成标准的输入/输出格式，开发者只需实现对应的函数即可加入系统。  
   - 这种设计让系统具备了极高的可扩展性，后续可以轻松接入最新的视觉模型或专用数学求解器。

4. **在两大知识密集型基准上实现显著跃迁**  
   - 在 ScienceQA 上，使用 GPT‑4 作为底层模型的 Chameleon 达到 86.54% 的整体准确率，比之前最好的 few‑shot 方法高出 11.37%。  
   - 在 TabMWP 上，准确率提升到 98.78%，领先前一最佳 17.0%。  
   - 这些数字直接说明了组合推理框架在实际任务中的强大增益。

### 方法详解
#### 整体框架概览
Chameleon 的运行可以划分为三大步骤：**任务解析 → 计划生成 → 工具执行**。首先，系统把用户的自然语言指令送入底层 LLM（如 GPT‑4），得到任务的高层语义表示。接着，Planner（同样是 LLM）基于这段表示和一份“可用工具清单”，生成一段结构化的执行计划。最后，系统按照计划顺序调用对应的外部模块，收集每一步的输出，并把所有中间结果拼回 LLM，得到最终答案。

#### 关键模块拆解
1. **工具库（Tool Registry）**  
   - 每个工具都有唯一的 ID、输入 schema（比如“查询关键词”或“图片路径”）和输出 schema（比如“搜索摘要”或“数值列表”）。  
   - 这些信息被存放在一个 JSON 表中，Planner 在生成计划时会查询该表，确保调用合法。

2. **Planner Prompt（规划提示）**  
   - 作者为 Planner 设计了一个多段式提示：  
     - **任务描述**：原始用户问题。  
     - **可用工具列表**：简要说明每个工具的功能和限制。  
     - **约束说明**：如“只能使用一次搜索”或“必须先进行表格解析”。  
   - 在此基础上，Planner 被要求输出类似下面的结构：  
     ```json
     [
       {"tool":"search", "args":{"query":"photosynthesis process"}},
       {"tool":"vision", "args":{"image_path":"./img1.png"}},
       {"tool":"python", "args":{"code":"..."}}
     ]
     ```
   - 这种结构化输出让后端执行器可以直接解析，无需额外的自然语言理解。

3. **执行引擎（Executor）**  
   - 读取 Planner 的计划后，逐条调用对应的工具。每一次调用的返回值都会被包装成一个“上下文片段”，再喂回底层 LLM，帮助它在后续步骤中利用已有信息。  
   - 例如，搜索得到的摘要会被加进 LLM 的“记忆”，供后面的 Python 计算使用。

4. **结果整合（Answer Synthesis）**  
   - 当所有工具执行完毕，系统把最终的上下文（包括原始问题、所有中间输出）一次性送入 LLM，要求它给出自然语言答案。  
   - 这里的 LLM 负责把碎片化的信息组织成连贯的解释，类似于人类在完成实验后写实验报告。

#### 设计亮点与反直觉之处
- **约束感知的提示工程**：很多人会直接让 LLM 自由生成计划，结果往往出现“无限循环调用”或“无关工具”。作者通过在提示中显式加入约束推理，让模型在生成计划时主动考虑“只能用一次搜索”等规则，这种“让模型先思考约束再行动”的做法在当时并不常见。
- **统一的结构化输出**：而不是让模型输出自然语言描述的步骤，作者强制要求 JSON 格式，这大幅降低了解析错误的概率，也让系统更易于调试和扩展。
- **双层 LLM 结构**：底层 LLM 负责语言理解和答案生成，Planner LLM 负责元决策。把两者分离，使得即使底层模型换成更小的模型，只要 Planner 仍然强大，整体系统仍能保持较好表现。

### 实验与效果
- **测试任务**：论文在两个公开基准上评估：ScienceQA（包含文字、图片和外部知识的科学问答）和 TabMWP（表格数学文字题）。这两个任务都需要跨模态信息检索和精确计算，正好验证了 Chameleon 的组合推理能力。
- **对比基线**：在 ScienceQA 上，作者与之前的 few‑shot GPT‑3.5、CoT（思维链）以及专门的多模态模型（如 LLaMA‑Adapter）对比；在 TabMWP 上，则与最新的表格推理模型（TableFormer、TabFact）以及纯 LLM 方法比较。
- **性能提升**：  
  - ScienceQA：Chameleon（GPT‑4）取得 86.54% 的整体准确率，领先之前最佳的 few‑shot 结果 11.37%。  
  - TabMWP：准确率提升至 98.78%，比前一最佳提升 17.0%。  
  这些数字直接说明了组合工具带来的增益。
- **消融实验**：论文分别去掉 Planner 的约束提示、去掉视觉模块、以及仅使用单一工具（如只用搜索）进行实验。结果显示：约束提示的缺失导致工具选择错误率上升约 12%；去掉视觉模块在含图片的 ScienceQA 子集上准确率下降约 8%；仅使用单一工具时整体性能跌至 70% 以下，验证了每个组件的必要性。
- **局限性**：作者指出，系统对 Planner 的质量高度依赖；如果 Planner 生成的计划出现错误，后续工具调用会全部失效。此外，当前实现仍然需要手工维护工具清单，自动发现新工具的能力尚未探索。

### 影响与延伸思考
Chameleon 把“语言模型 + 工具链”从零散的实验案例提升到系统化、可复用的框架，随后出现了多篇工作尝试在此基础上加入自学习的工具选择、强化学习优化计划、以及更细粒度的模块化接口。比如后续的 **ReAct**、**Toolformer** 系列都在不同程度上借鉴了 Chameleon 的结构化计划输出和约束感知提示。对想继续深入的读者，可以关注以下方向：  
- **自动工具发现**：让模型在运行时自行搜索并注册新工具，而不是预先手工列出。  
- **计划优化的强化学习**：使用奖励信号（如答案正确率）来微调 Planner，使其在特定领域更高效。  
- **跨语言/跨平台的统一工具协议**：制定类似 OpenAI Functions 的标准，让不同模型和工具之间实现即插即用。  
这些都是在 Chameleon 思路上自然延伸的研究热点。

### 一句话记住它
**Chameleon 用 LLM 当导演，动态组装搜索、视觉、计算等工具，让一次对话完成多模态、知识密集的推理。**