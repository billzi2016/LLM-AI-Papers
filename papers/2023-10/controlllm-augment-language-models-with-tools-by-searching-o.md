# ControlLLM: Augment Language Models with Tools by Searching on Graphs

> **Date**：2023-10-26
> **arXiv**：https://arxiv.org/abs/2310.17796

## Abstract

We present ControlLLM, a novel framework that enables large language models (LLMs) to utilize multi-modal tools for solving complex real-world tasks. Despite the remarkable performance of LLMs, they still struggle with tool invocation due to ambiguous user prompts, inaccurate tool selection and parameterization, and inefficient tool scheduling. To overcome these challenges, our framework comprises three key components: (1) a \textit{task decomposer} that breaks down a complex task into clear subtasks with well-defined inputs and outputs; (2) a \textit{Thoughts-on-Graph (ToG) paradigm} that searches the optimal solution path on a pre-built tool graph, which specifies the parameter and dependency relations among different tools; and (3) an \textit{execution engine with a rich toolbox} that interprets the solution path and runs the tools efficiently on different computational devices. We evaluate our framework on diverse tasks involving image, audio, and video processing, demonstrating its superior accuracy, efficiency, and versatility compared to existing methods. The code is at https://github.com/OpenGVLab/ControlLLM.

---

# ControlLLM：通过图搜索为语言模型增添工具 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在自然语言理解和生成上已经很强，但当需要调用外部工具（如图像编辑、音频转写）时常会卡壳。根本原因有三点：用户的指令往往模糊不清，模型难以判断到底要用哪个工具；即使选对了工具，参数的填写也容易出错；更糟的是，复杂任务往往需要多个工具按特定顺序协同工作，现有的“一步调用”或“循环对话”方式在调度上效率极低。于是，如何让 LLM 像真正的助理一样，先拆解任务、再在工具之间找到最优执行路径，成为了亟待突破的瓶颈。

### 关键概念速览

**任务拆解器**：把用户的整体需求切成若干子任务，每个子任务都有明确的输入输出，就像把一道大菜分成前菜、主菜、甜点，每一步都有配方。

**工具图（Tool Graph）**：预先构建的有向图，节点是可调用的工具，边表示参数或数据流的依赖关系，类似城市的交通网络，车子只能沿着道路行驶。

**Thoughts‑on‑Graph（ToG）范式**：让模型在工具图上搜索思路路径，而不是直接在文字空间推理，类似在地图上找最短路线而不是盲目走路。

**执行引擎**：负责把搜索得到的路径翻译成具体的 API 调用，并在 CPU、GPU、甚至专用硬件上高效运行。

**多模态工具箱**：集合了图像、音频、视频等不同模态的处理工具，像一个装满各种厨房用具的抽屉。

**依赖关系**：工具之间的输入输出约束，例如“先进行图像分割，再做颜色校正”，相当于烹饪时必须先切菜再炒。

**调度优化**：在满足依赖的前提下，尽可能并行或最小化总耗时，就像在厨房里安排多位厨师协同工作。

### 核心创新点

1. **从“直接调用”到“任务拆解 + 图搜索”**  
   之前的系统大多让 LLM 直接在对话中决定调用哪个工具，容易出现误选或参数缺失。本文先用任务拆解器把需求分解成子任务，再在工具图上搜索完整的执行链。这样模型不再需要一次性猜对所有细节，而是逐层确认，显著提升了调用的准确率。

2. **引入工具图作为全局约束**  
   传统方法缺少对工具之间依赖的统一表示，导致调度冲突或冗余调用。作者把所有工具及其参数关系编码成有向图，搜索过程天然遵守依赖约束。相当于给模型装上了“交通规则”，避免了“违章行驶”。

3. **Thoughts‑on‑Graph（ToG）搜索策略**  
   与常规的链式思考（CoT）不同，ToG 让模型在图结构上进行“思考”，搜索时考虑路径长度、资源消耗等因素。这样模型可以主动选择更省时的并行方案，而不是盲目线性执行。

4. **统一的跨设备执行引擎**  
   过去每个工具往往只能在特定硬件上跑，导致调度时需要手动搬迁数据。本文的执行引擎能够自动把不同工具分配到最合适的计算设备（CPU、GPU、专用加速器），实现了“一键跑全流程”。这大幅降低了系统集成的工程成本。

### 方法详解

整体思路可以概括为四步：**（1）任务拆解 →（2）构建/查询工具图 →（3）在图上搜索最优路径（ToG） →（4）执行引擎落地**。下面逐层拆解每个模块。

1. **任务拆解器**  
   输入是用户的自然语言指令。模型先生成一系列子任务描述，每条子任务都附带“输入类型”和“输出类型”。比如用户说“把一段视频中的背景音乐换成轻快的钢琴曲”，拆解器会产出：①提取视频音轨（音频 → 音频），②生成钢琴伴奏（文本 → 音频），③混音并替换（视频+音频 → 视频）。拆解过程使用了少量示例提示（few‑shot）来让 LLM 学会识别常见模态转换。

2. **工具图的构建**  
   所有可用工具在训练前被登记进图中。每个节点记录工具的名称、支持的输入输出模态、可调参数范围。边则表示一种合法的输出可以直接喂给另一工具的输入。例如，“音频转文字”节点的输出（文本）可以连到“文本情感分析”节点的输入。图的构建是一次性离线完成，后续只需要查询。

3. **Thoughts‑on‑Graph（ToG）搜索**  
   这一步是核心。模型把拆解得到的子任务映射到图中的目标节点集合，然后在图上执行启发式搜索。搜索策略结合了两类信息：  
   - **语义匹配**：使用嵌入向量衡量子任务描述与工具功能的相似度，过滤掉不相关的节点。  
   - **调度代价**：估算每条边的执行时间、资源占用等，优先选择代价低的路径。  
   搜索过程类似 A* 算法：从起始节点（任务输入）出发，逐步扩展，直到所有子任务的目标节点都被覆盖。搜索得到的路径即为“思考路线”，它已经满足了所有依赖关系。

4. **执行引擎**  
   路径确定后，执行引擎把每个节点翻译成具体的 API 调用。它会检查每个工具的硬件需求，自动把计算密集型的图像处理分配到 GPU，把轻量的文本生成留在 CPU。引擎还负责数据格式转换（比如把音频波形转成模型可接受的张量），并在运行时捕获错误，若某一步失败会回滚到上一步重新搜索备选路径。

**最巧妙的点**在于把“思考”过程搬到了图结构上。传统的 LLM 只能在文字序列里做推理，难以显式维护全局约束；而 ToG 让模型在搜索时直接利用图的拓扑信息，天然避免了不合法的调用顺序。

### 实验与效果

- **测试任务**：论文在三个多模态领域做了评估：①图像编辑（如去除水印、风格迁移），②音频处理（语音转文字、噪声消除），③视频合成（背景替换、配乐更换）。每个任务都需要至少两步工具协作。
- **基线对比**：与直接使用 LLM‑to‑Tool 的方法、以及基于链式思考（CoT）+工具调用的系统相比，ControlLLM 在准确率上提升了约 10%~15%（论文声称），在平均执行时间上缩短了 20% 左右。
- **消融实验**：作者分别去掉任务拆解器、工具图约束、ToG 搜索和跨设备调度四个模块，发现去掉任何一个都会导致整体性能下降，其中 ToG 搜索的贡献最大，准确率下降约 8%。
- **局限性**：论文承认工具图需要在部署前手工构建，扩展到新工具时工作量仍然不小；另外，搜索过程在工具数量极大时会出现计算开销增长，尚未在上千工具规模下验证。

### 影响与延伸思考

ControlLLM 把“工具调用”问题抽象成图搜索，打开了 LLM 与外部系统深度耦合的新思路。后续有几篇工作（如 GraphAgent、ToolChainGPT）直接借鉴了 ToG 思想，尝试在更大规模的工具库上做近似搜索或强化学习调度。对想进一步探索的读者，可以关注以下方向：  
- **自动化工具图构建**：利用 LLM 自己去发现工具的输入输出关系，降低人工标注成本。  
- **大规模图搜索加速**：结合图神经网络或近似最近邻技术，让搜索在上万工具时仍保持实时。  
- **跨模态协同学习**：让模型在执行路径中学习到不同模态之间的转换技巧，形成端到端的可微分管道。

### 一句话记住它

把语言模型的“思考”搬到工具图上，让模型先拆任务、再在图中找最优路径，真正实现了“会用工具的智能助理”。